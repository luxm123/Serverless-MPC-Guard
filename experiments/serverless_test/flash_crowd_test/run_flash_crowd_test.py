import sys
import os
import time
import random
import pandas as pd
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.serverless_test.trace_experiment.run_trace_replay import TraceReplayer
from experiments.serverless_test.wcp_validation.serverless_utils import force_cold_start

def generate_flash_crowd_trace(output_path, duration_s=150):
    """
    Generate a synthetic trace with:
    1. Warm-up (0-30s): Stable low load.
    2. Flash Crowd (30-60s): Sudden burst.
    3. Cooldown (60-90s): Return to normal.
    4. Slowdown (90-120s): Service time doubles (simulating DB latency).
    5. End (120-150s): Cooldown.
    """
    print(f"Generating flash crowd trace to {output_path}...")
    
    trace_data = []
    
    # Parameters
    base_rps = 20
    burst_rps = 80 # Further reduced from 150 to 80 to prevent AWS Rate Exceeded
    base_duration = 2000 # ms
    slowdown_factor = 2.0
    
    current_time = 0.0
    req_id = 0
    
    while current_time < duration_s:
        # Determine current phase
        is_burst = 30 <= current_time < 60
        is_slowdown = 90 <= current_time < 120
        
        # Determine RPS and Duration
        rps = burst_rps if is_burst else base_rps
        duration = base_duration * slowdown_factor if is_slowdown else base_duration
        
        # Add jitter to duration (±10%)
        duration = int(duration * random.uniform(0.9, 1.1))
        
        # Determine Priority (QoS)
        # In flash crowd, mixed traffic.
        r = random.random()
        if r < 0.2:
            prio = "critical" # Q1
        elif r < 0.5:
            prio = "standard" # Q2
        else:
            prio = "low"      # Q3
            
        # Add request
        trace_data.append({
            "timestamp": current_time,
            "duration": duration,
            "priority": prio,
            "is_flash": False # We handle distribution manually, don't let Replayer override
        })
        
        # Advance time (Poisson process)
        inter_arrival = random.expovariate(rps)
        current_time += inter_arrival
        req_id += 1
        
    df = pd.DataFrame(trace_data)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} requests.")
    return df

class DynamicReplayer(TraceReplayer):
    """
    A subclass of TraceReplayer that enforces strict time-based replay
    instead of maximum-throughput stress testing.
    """
    def run_experiment(self, strategy, wcp_mode, output_filename, mpc_profile=None):
        # Ensure reproducibility
        random.seed(42)
        np.random.seed(42)
        
        print(f"\n>>> Starting Dynamic Experiment: Strategy='{strategy}', Threads={self.thread_num} <<<")
        
        # Load trace if not loaded
        if not hasattr(self, 'raw_trace_data') or not self.raw_trace_data:
            self.load_trace()
            
        # Use raw data directly (no random dropping, no auto-injection)
        self.trace_data = copy.deepcopy(self.raw_trace_data)
        
        # Sort by timestamp to ensure correct replay order
        self.trace_data.sort(key=lambda x: x['timestamp'])
        
        self.results = []
        start_exp = time.time()
        
        # Use a larger thread pool to ensure client isn't the bottleneck during bursts
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = []
            
            for i, row in enumerate(self.trace_data):
                # Time-based dispatch logic
                target_time = row['timestamp']
                now = time.time() - start_exp
                wait_time = target_time - now
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Submit task
                # Note: run_request signature: (req_id, row, strategy, wcp_mode, start_exp, mpc_profile)
                f = executor.submit(self.run_request, i, row, strategy, wcp_mode, start_exp, mpc_profile)
                futures.append(f)
                
            # Wait for all
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Thread Error] {e}")

        end_exp = time.time()
        duration = end_exp - start_exp
        
        output_path = os.path.join(self.output_dir, output_filename)
        print(f">>> Experiment Ended. Saving to {output_path}...")
        pd.DataFrame(self.results).to_csv(output_path, index=False)
        self.analyze_results(strategy, duration)

import copy

def plot_time_series(mpc_csv, baseline_csv, output_dir, trial_suffix=''):
    print(f"Plotting time series results for {trial_suffix}...")
    try:
        df_mpc = pd.read_csv(mpc_csv)
        df_base = pd.read_csv(baseline_csv)
        
        df_mpc['Strategy'] = 'MPC'
        df_base['Strategy'] = 'Baseline'
        
        df_all = pd.concat([df_mpc, df_base])
        
        # Bin data by second for cleaner plots
        df_all['TimeBin'] = df_all['timestamp'].astype(int)
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 1. P99 Latency over Time
        sns.lineplot(data=df_all, x='TimeBin', y='e2e_latency', hue='Strategy', 
                     estimator=lambda x: np.percentile(x, 99), errorbar=None, ax=axes[0], palette={'MPC': 'g', 'Baseline': 'r'})
        axes[0].set_title(f'P99 Latency over Time (Flash Crowd + Slowdown) {trial_suffix}')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].axvline(30, color='k', linestyle='--', alpha=0.5, label='Burst Start')
        axes[0].axvline(60, color='k', linestyle='--', alpha=0.5, label='Burst End')
        axes[0].axvline(90, color='k', linestyle='--', alpha=0.5, label='Slowdown Start')
        axes[0].legend()
        
        # 2. Fidelity (MPC Only)
        sns.lineplot(data=df_mpc, x='TimeBin', y='fidelity', estimator='mean', errorbar=None, ax=axes[1], color='g')
        axes[1].set_title(f'MPC Fidelity Adaptation {trial_suffix}')
        axes[1].set_ylabel('Fidelity')
        axes[1].set_ylim(0, 1.1)
        axes[1].axvline(30, color='k', linestyle='--', alpha=0.5)
        axes[1].axvline(90, color='k', linestyle='--', alpha=0.5)
        
        # 3. Shedding Rate
        # Calculate shedding rate per bin
        def get_shed_rate(g):
            return (g['worker_status'] == 'shedded').mean() * 100
            
        shed_rates = df_all.groupby(['Strategy', 'TimeBin']).apply(get_shed_rate).reset_index(name='ShedRate')
        sns.lineplot(data=shed_rates, x='TimeBin', y='ShedRate', hue='Strategy', ax=axes[2], palette={'MPC': 'g', 'Baseline': 'r'})
        axes[2].set_title(f'Shedding Rate over Time {trial_suffix}')
        axes[2].set_ylabel('Shedding Rate (%)')
        axes[2].axvline(30, color='k', linestyle='--', alpha=0.5)
        axes[2].axvline(90, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'flash_crowd_analysis{trial_suffix}.png')
        plt.savefig(plot_path)
        plt.close() # Close to free memory
        print(f"Plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error plotting: {e}")

def run_flash_crowd_experiment():
    output_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    trace_file = os.path.join(output_dir, 'synthetic_flash_trace.csv')
    generate_flash_crowd_trace(trace_file)
    
    # Use high thread count to avoid client bottleneck
    # Max concurrency ~300. But AWS Throttling occurs at 500. Lowered to 100.
    replayer = DynamicReplayer(trace_file=trace_file, output_dir=output_dir, thread_num=100)
    
    target_funcs = [
        os.environ.get('MPC_CONTROLLER_NAME', 'MPC_Controller'),
        os.environ.get('MPC_WORKER_NAME', 'MPC_BusinessWorker')
    ]
    
    # Run 3 Trials
    for trial in range(1, 4):
        trial_suffix = f"_run{trial}"
        print(f"\n{'='*50}")
        print(f"Running Flash Crowd Experiment (Trial {trial}/3)")
        print(f"{'='*50}")
        
        # 1. Run MPC
        print(f"\n>>> MPC Trial {trial} <<<")
        force_cold_start(target_funcs)
        # Use flash_crowd profile for tuned capacity (150.0) matching our thread count (200)
        replayer.run_experiment(strategy='mpc', wcp_mode='strict', output_filename=f'results_mpc{trial_suffix}.csv', mpc_profile='flash_crowd')
        time.sleep(5) # Cooldown
        
        # 2. Run Baseline
        print(f"\n>>> Baseline Trial {trial} <<<")
        force_cold_start(target_funcs)
        replayer.run_experiment(strategy='baseline', wcp_mode='baseline', output_filename=f'results_baseline{trial_suffix}.csv')
        time.sleep(5) # Cooldown
        
        # 3. Plot this trial immediately
        plot_time_series(
            os.path.join(output_dir, f'results_mpc{trial_suffix}.csv'),
            os.path.join(output_dir, f'results_baseline{trial_suffix}.csv'),
            output_dir,
            trial_suffix
        )

if __name__ == "__main__":
    run_flash_crowd_experiment()
