import time
import json
import numpy as np
import pandas as pd
import threading
import random
from concurrent.futures import ThreadPoolExecutor
from serverless_utils import invoke_worker_lambda

# --- Configuration ---
# 6 types of functions as per benchmark availability
FUNCTIONS = ['linpack', 'gzip', 'image_processing', 'video_processing', 'chameleon', 'matmul']
STRATEGIES = ['mpc_integrated', 'gsight', 'owl', 'baseline']
TRACE_FILE = 'azure_traces/invocations_per_function_sample.csv'
DURATION_MINUTES = 60
RPS_SCALE = 0.2 # 1:5 scale
NOISE_LEVEL = 0.1 # +/- 10%
QOS_TARGET = 180.0

class AcademicExperiment:
    def __init__(self):
        self.results = {s: {f: [] for f in FUNCTIONS} for s in STRATEGIES}
        self.metrics_summary = {}

    def load_trace(self):
        df = pd.read_csv(TRACE_FILE)
        # Select 6 unique functions from trace
        trace_funcs = df['HashFunction'].unique()[:6]
        mapping = {trace_funcs[i]: FUNCTIONS[i] for i in range(len(trace_funcs))}
        
        workload = {}
        for _, row in df[df['HashFunction'].isin(trace_funcs)].iterrows():
            func_name = mapping[row['HashFunction']]
            # Extract minute columns (1, 2, 3...)
            counts = row.iloc[3:].values.astype(float)
            # Scale and add noise
            counts = counts * RPS_SCALE
            noise = np.random.uniform(1-NOISE_LEVEL, 1+NOISE_LEVEL, size=len(counts))
            workload[func_name] = counts * noise
        return workload

    def run_strategy(self, strategy):
        print(f"\n>>> Starting Strategy: {strategy} <<<")
        workload = self.load_trace()
        
        # Initialize strategy in middleware (reset state)
        self._run_single_req(FUNCTIONS[0], strategy, reset=True)
        
        for func in FUNCTIONS:
            print(f"--- Testing Function: {func} ---")
            func_workload = workload.get(func, [5.0]*DURATION_MINUTES)
            
            for m in range(min(len(func_workload), DURATION_MINUTES)):
                rps = max(1.0, func_workload[m])
                print(f"[{strategy}] {func} Minute {m}: RPS={rps:.1f}")
                self._execute_minute(func, strategy, rps)
                
    def _execute_minute(self, func, strategy, rps):
        start_time = time.time()
        num_requests = int(rps * 60)
        interval = 1.0 / rps
        
        with ThreadPoolExecutor(max_workers=30) as executor:
            for i in range(num_requests):
                executor.submit(self._run_single_req, func, strategy)
                time.sleep(interval)
                if time.time() - start_time > 60:
                    break

    def _run_single_req(self, func, strategy, reset=False):
        internal_strategy = strategy
        if strategy == 'baseline':
            internal_strategy = 'hpa_baseline'
            
        task = {
            "task_type": func,
            "id": f"academic-{random.randint(1000,9999)}"
        }
        
        # Integrated MPC mode expects metrics and strategy in payload
        # For simplicity, we use the 'auto' mode which triggers internal MPC logic
        res_wrapped = invoke_worker_lambda(
            decision={}, 
            task=task, 
            mode='auto', 
            strategy=internal_strategy,
            reset_state=reset
        )
        
        if res_wrapped:
            res = res_wrapped.get('response', {})
            e2e = res_wrapped.get('client_duration', 0)
            
            # Extract metadata from response
            meta = res.get('meta', {})
            self.results[strategy][func].append({
                'e2e': e2e,
                'srv': res.get('duration_ms', 0),
                'alloc': res.get('resource_alloc', 1.0),
                'overhead': meta.get('scheduling_overhead_ms', 0),
                'is_cold': res.get('is_cold_start', False),
                'ts': time.time()
            })

    def calculate_metrics(self):
        summary = []
        for s in STRATEGIES:
            all_data = []
            for f in FUNCTIONS:
                all_data.extend(self.results[s][f])
            
            if not all_data: continue
            
            df = pd.DataFrame(all_data)
            qos_viol = (df['e2e'] > QOS_TARGET).mean() * 100
            avg_alloc = df['alloc'].mean()
            density = 1.0 / avg_alloc
            avg_overhead = df['overhead'].mean()
            p90_lat = np.percentile(df['e2e'], 90)
            
            # Utilization (Actual Srv Latency / (Alloc * Benchmark_Base))
            # Rough estimate: 1.0 alloc target is ~135ms
            utilization = (df['srv'].mean() / 135.0) / avg_alloc * 100
            
            summary.append({
                'Strategy': s,
                'QoS Viol %': qos_viol,
                'Density': density,
                'Overhead (ms)': avg_overhead,
                'P90 (ms)': p90_lat,
                'Util %': min(100, utilization)
            })
        return pd.DataFrame(summary)

if __name__ == "__main__":
    exp = AcademicExperiment()
    # Fast validation: 3 minutes per function per strategy
    DURATION_MINUTES = 3 
    
    for s in STRATEGIES:
        exp.run_strategy(s)
        
    print("\n" + "="*70)
    print("FINAL ACADEMIC COMPARISON REPORT (6 Functions, Azure Trace)")
    print("="*70)
    print(exp.calculate_metrics().to_string(index=False))
    print("="*70)
