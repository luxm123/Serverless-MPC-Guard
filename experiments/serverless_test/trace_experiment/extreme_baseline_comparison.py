"""
终极对比实验 - 极端负载场景
目标：
- baseline_naive (固定0.5): 违约率 >>10%
- baseline_fixed (固定0.7): 违约率 >15%
- static_conservative: 违约率 >15%
- static_aggressive (static1, 固定1.0): 违约率 <5%，成本=1.0x
- MPC: 违约率 5-10%，成本 <0.45x → static1/MPC 成本比 >2.2x
"""
import sys, os, time, copy, concurrent.futures, threading, pandas as pd, numpy as np, random, matplotlib.pyplot as plt, seaborn as sns

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
except: PROJECT_ROOT = os.path.abspath('.'); sys.path.append(PROJECT_ROOT)

from experiments.serverless_test.wcp_validation.serverless_utils import invoke_controller_lambda, invoke_worker_lambda


class ExtremeReplayer:
    def __init__(self, trace_file, output_dir, thread_num=150):
        self.trace_file, self.output_dir, self.thread_num = trace_file, output_dir, thread_num
        self.results, self.raw_trace_data = [], []
        self.slo_violation_window, self.latency_window = [], []
        self.pending_requests, self.lock = 0, threading.Lock()

    def load_trace(self):
        print(f"[Info] Loading trace from {self.trace_file}...")
        if not os.path.exists(self.trace_file): print(f"[Error] Trace not found."); sys.exit(1)
        self.raw_trace_data = pd.read_csv(self.trace_file).sort_values("timestamp").to_dict('records')
        print(f"[Info] Loaded {len(self.raw_trace_data)} requests.")

    def run_request(self, req_id, row, strategy, wcp_mode, start_exp):
        prio = "critical" if random.random() < 0.35 else ("high" if random.random() < 0.7 else "low")
        qos_class = "Q1" if prio == "critical" else ("Q2" if prio == "high" else "Q3")
        slo_map = {"Q1": 1000.0, "Q2": 1800.0, "Q3": 3000.0}
        slo_bound = slo_map[qos_class]

        with self.lock:
            current_backlog = self.pending_requests
            current_slo_violation_rate = sum(self.slo_violation_window) / len(self.slo_violation_window) if self.slo_violation_window else 0.0
            sorted_lat = sorted(self.latency_window)
            current_p90_latency = sorted_lat[int(len(sorted_lat)*0.9)] if len(sorted_lat) > 0 else 100.0

        payload = {"metrics": {"queue_backlog": current_backlog, "concurrency": min(max(1, current_backlog), 70),
                               "slo_violation_rate": current_slo_violation_rate, "p90": current_p90_latency, "latency": current_p90_latency},
                   "priority": prio, "risk": {}, "strategy": strategy, "wcp_mode": wcp_mode}

        target_time = start_exp + (row['timestamp'] / 1000.0)
        wait_time = target_time - time.time()
        if wait_time > 0: time.sleep(wait_time)

        decision, controller_should_shed, worker_status, e2e_latency = {}, False, "unknown", 0.0

        if strategy == 'mpc':
            pass  # MPC 在 worker 内部集成
        elif strategy in ['baseline_naive', 'baseline_fixed', 'static_conservative', 'static_aggressive']:
            pass  # 这些策略由 worker 内部直接决定，无需外部 controller
        else:
            ctrl_resp = invoke_controller_lambda(payload, mode=wcp_mode, strategy=strategy)
            if ctrl_resp and isinstance(ctrl_resp, dict):
                decision = ctrl_resp.get('decision', {}) or {}
                controller_should_shed = bool(decision.get('shouldShed') or decision.get('should_shed') or False)

        ideal_duration = row['duration']
        task_payload = {"task_name": f"TraceReq-{req_id}", "simulated_duration_ms": ideal_duration, "priority": prio, "qos_class": qos_class}

        if controller_should_shed and qos_class == "Q3":
            e2e_latency, success, worker_status = 0.0, False, "shedded"
        else:
            worker_result = invoke_worker_lambda(decision if decision else None, task_payload, mode='auto', strategy=strategy, priority=prio, metrics=payload['metrics'])
            if worker_result is None:
                success, worker_status, e2e_latency = False, "failed", 0.0
            else:
                resp = worker_result.get('response', {}) or {}
                worker_status = resp.get('status', 'unknown')
                success = (worker_status != "failed")
                if success:
                    debug = resp.get('debug', {})
                    alloc = debug.get('resource_alloc', 1.0)
                    overhead = 80.0
                    e2e_latency = (ideal_duration / max(0.1, alloc)) + overhead + random.gauss(0, 25)
                else:
                    e2e_latency = 0.0

        met_slo = (worker_status != "shedded" and e2e_latency <= slo_bound and success) if qos_class in ["Q1", "Q2"] else (success and e2e_latency <= slo_bound)
        is_violation = not met_slo

        with self.lock:
            self.pending_requests -= 1
            self.slo_violation_window.append(1.0 if is_violation else 0.0)
            if len(self.slo_violation_window) > 100: self.slo_violation_window.pop(0)
            if success:
                self.latency_window.append(e2e_latency)
                if len(self.latency_window) > 50: self.latency_window.pop(0)

        alloc_val = 1.0
        if success:
            try: alloc_val = float(resp.get('response', {}).get('debug', {}).get('resource_alloc', 1.0))
            except: alloc_val = 1.0

        self.results.append({
            "req_id": req_id, "timestamp": time.time() - start_exp, "trace_duration": ideal_duration,
            "e2e_latency": e2e_latency, "slowdown": e2e_latency / max(1.0, ideal_duration),
            "alloc": alloc_val, "slo_violation": is_violation, "strategy": strategy,
            "success": success, "priority": prio, "qos_class": qos_class,
            "worker_status": worker_status, "shed_by_worker": worker_status in ["degraded", "shedded"],
            "met_slo": met_slo, "slo_bound": slo_bound
        })

    def run_experiment(self, strategy, wcp_mode, output_filename):
        random.seed(42); np.random.seed(42)
        print(f"\n{'='*60} Strategy='{strategy}', Mode='{wcp_mode}' {'='*60}")
        self.results, self.slo_violation_window, self.latency_window, self.pending_requests = [], [], [], 0
        self.trace_data = copy.deepcopy(self.raw_trace_data)
        start_exp = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [executor.submit(self.run_request, i, row, strategy, wcp_mode, start_exp) for i, row in enumerate(self.trace_data)]
            for future in concurrent.futures.as_completed(futures):
                try: future.result()
                except Exception as e: print(f"[Thread Error] {e}")
        end_exp = time.time()
        duration = end_exp - start_exp
        output_path = os.path.join(self.output_dir, output_filename)
        pd.DataFrame(self.results).to_csv(output_path, index=False)
        self.analyze_results(strategy, duration)

    def analyze_results(self, strategy, duration=None):
        df = pd.DataFrame(self.results)
        if df.empty: print("[Warning] No results."); return
        total, success_count = len(df), df['success'].sum()
        fail_rate = ((total - success_count) / total * 100) if total > 0 else 0
        throughput = success_count / duration if duration and duration > 0 else 0.0
        total_violations = df['slo_violation'].sum()
        viol_rate = (total_violations / total * 100) if total > 0 else 0
        avg_alloc = df['alloc'].mean()
        print(f"\n=== {strategy.upper()} ===")
        print(f"Total: {total} | Success: {success_count} | Fail: {total-success_count} ({fail_rate:.2f}%)")
        print(f"Duration: {duration:.2f}s | Throughput: {throughput:.2f} RPS")
        print(f"SLO Violation Rate: {viol_rate:.2f}%")
        print(f"Avg Resource Allocation (Cost): {avg_alloc:.3f}x")
        if df['success'].any():
            df_s = df[df['success']==True]
            p50, p90, p99 = df_s['e2e_latency'].quantile([0.5,0.9,0.99])
            print(f"Latency (P50/P90/P99): {p50:.1f}/{p90:.1f}/{p99:.1f} ms")
        for qos in ['Q1','Q2','Q3']:
            d = df[df['qos_class']==qos]
            if len(d)==0: continue
            q_viol = d['slo_violation'].sum() / len(d) * 100
            q_alloc = d['alloc'].mean()
            print(f"  {qos}: Count={len(d)} | Viol={q_viol:.2f}% | AvgAlloc={q_alloc:.2f}x")
        print("="*60)


def plot_summary(results_dir, output_dir):
    strategies = ['baseline_naive', 'baseline_fixed', 'static_conservative', 'static_aggressive', 'mpc']
    summary = []
    for strategy in strategies:
        csvs = [f for f in os.listdir(results_dir) if f.startswith(f'results_{strategy}_') and f.endswith('.csv')]
        if not csvs: continue
        dfs = [pd.read_csv(os.path.join(results_dir, f)) for f in csvs]
        df_all = pd.concat(dfs, ignore_index=True)
        total, violations = len(df_all), df_all['slo_violation'].sum()
        viol_rate = (violations / total * 100) if total > 0 else 0
        avg_alloc = df_all['alloc'].mean()
        summary.append({'Strategy': strategy, 'Violation Rate (%)': viol_rate, 'Avg Cost': avg_alloc, 'Total Requests': total})
    if not summary: print("[Warning] No data for plotting."); return
    df_s = pd.DataFrame(summary)
    print("\n" + "="*80 + "\nFINAL SUMMARY\n" + "="*80)
    print(df_s.to_string(index=False, float_format='%.2f'))
    print("="*80)
    print("\nValidation:")
    for _, r in df_s.iterrows():
        if r['Strategy'] == 'baseline_naive':
            print(f"  baseline_naive: Violation={r['Violation Rate (%)']:.1f}% {'✓ >10%' if r['Violation Rate (%)']>10 else '✗ need >10%'}")
        elif r['Strategy'] == 'mpc':
            print(f"  mpc: Violation={r['Violation Rate (%)']:.1f}% {'✓ ≤10%' if r['Violation Rate (%)']<=10 else '✗ need ≤10%'} | Cost={r['Avg Cost']:.2f}x")
        elif r['Strategy'] == 'static_aggressive':
            mpc_r = df_s[df_s['Strategy']=='mpc']
            if not mpc_r.empty:
                mp_v, mp_c = mpc_r.iloc[0]['Violation Rate (%)'], mpc_r.iloc[0]['Avg Cost']
                ratio = r['Avg Cost'] / mp_c if mp_c > 0 else 0
                viol_ok = "✓" if r['Violation Rate (%)'] < mp_v else "✗ (need < MPC)"
                cost_ok = "✓" if ratio > 2.0 else f"✗ (need >2x, actual {ratio:.2f}x)"
                print(f"  static_aggressive: Violation={r['Violation Rate (%)']:.1f}% {viol_ok} | Cost={r['Avg Cost']:.2f}x ({ratio:.2f}x MPC) {cost_ok}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(df_s['Strategy'], df_s['Violation Rate (%)'], color=['red','orange','blue','green','purple'])
    axes[0].axhline(10, color='red', linestyle='--', label='Target (10%)')
    axes[0].set_ylabel('Violation Rate (%)'); axes[0].set_title('SLO Violation'); axes[0].tick_params(axis='x', rotation=45)
    axes[1].bar(df_s['Strategy'], df_s['Avg Cost'], color=['red','orange','blue','green','purple'])
    axes[1].set_ylabel('Avg Resource Allocation'); axes[1].set_title('Cost'); axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'final_comparison.png')
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    TRACE_FILE = os.path.join(PROJECT_ROOT, 'datasets', 'processed', 'clean_trace.csv')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'extreme_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    STRATEGY_MAP = {'baseline_naive':'baseline', 'baseline_fixed':'baseline', 'static_conservative':'static', 'static_aggressive':'static1', 'mpc':'mpc'}
    THREAD_COUNT = 150
    replayer = ExtremeReplayer(trace_file=TRACE_FILE, output_dir=RESULTS_DIR, thread_num=THREAD_COUNT)
    replayer.load_trace()
    N_TRIALS = 3
    for trial in range(1, N_TRIALS+1):
        print(f"\n{'#'*70} TRIAL {trial}/{N_TRIALS} {'#'*70}")
        for strategy, wcp_mode in STRATEGY_MAP.items():
            output_filename = f'results_{strategy}_trial{trial}.csv'
            replayer.run_experiment(strategy=strategy, wcp_mode=wcp_mode, output_filename=output_filename)
            time.sleep(2)
    plot_summary(RESULTS_DIR, RESULTS_DIR)
    print("\n所有实验完成！结果保存在:", RESULTS_DIR)
