"""
改进的 Baseline 对比实验
基于真实 Azure 轨迹回放，对比不同策略的 SLO 违约率和成本。

策略：
- baseline_naive: 固定分配 0.6（最差基线）
- baseline_fixed: 固定分配 0.8（保守基线）
- static_conservative: 按优先级静态分配（适度成本）
- static_aggressive: 过度分配（高成本，低违约）——对应 static1
- mpc: 模型预测控制（平衡违约率和成本）

评估指标：
- SLO Violation Rate（主指标，目标 ≤10%）
- Cost = avg_alloc（相对成本）
- Throughput
"""
import sys
import os
import time
import copy
import concurrent.futures
import threading
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 动态路径设置
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.path.abspath('.')
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

from experiments.serverless_test.wcp_validation.serverless_utils import invoke_controller_lambda, invoke_worker_lambda


class ImprovedTraceReplayer:
    """改进的轨迹回放器，支持多种baseline策略"""
    def __init__(self, trace_file, output_dir, thread_num=100):
        self.trace_file = trace_file
        self.output_dir = output_dir
        self.thread_num = thread_num
        self.results = []
        self.raw_trace_data = []
        self.slo_violation_window = []  # 滑动窗口用于计算实时 SLO 违约率
        self.latency_window = []  # 滑动窗口用于计算 P90
        self.pending_requests = 0
        self.lock = threading.Lock()

    def load_trace(self):
        """加载轨迹数据"""
        print(f"[Info] Loading trace from {self.trace_file}...")
        if not os.path.exists(self.trace_file):
            print(f"[Error] Trace file not found: {self.trace_file}")
            sys.exit(1)
        self.raw_trace_data = pd.read_csv(self.trace_file).sort_values(by="timestamp").to_dict('records')
        print(f"[Info] Loaded {len(self.raw_trace_data)} requests.")

    def run_request(self, req_id, row, strategy, wcp_mode, start_exp):
        """执行单个请求"""
        # 确定优先级
        prio = "critical" if random.random() < 0.25 else ("high" if random.random() < 0.55 else "low")
        qos_class = "Q1" if prio == "critical" else ("Q2" if prio == "high" else "Q3")

        # 获取实时系统状态
        with self.lock:
            current_backlog = self.pending_requests
            if self.slo_violation_window:
                current_slo_violation_rate = sum(self.slo_violation_window) / len(self.slo_violation_window)
            else:
                current_slo_violation_rate = 0.0
            if self.latency_window:
                sorted_lat = sorted(self.latency_window)
                idx = int(len(sorted_lat) * 0.9)
                current_p90_latency = sorted_lat[idx] if idx < len(sorted_lat) else 100.0
            else:
                current_p90_latency = 100.0

        # SLO 阈值（单位：ms）
        slo_map = {"Q1": 1000.0, "Q2": 1800.0, "Q3": 3000.0}
        slo_bound = slo_map[qos_class]

        # 构建 payload
        payload = {
            "metrics": {
                "queue_backlog": current_backlog,
                "concurrency": min(max(1, current_backlog), 70),
                "slo_violation_rate": current_slo_violation_rate,
                "p90": current_p90_latency,
                "latency": current_p90_latency
            },
            "priority": prio,
            "risk": {},
            "strategy": strategy,
            "wcp_mode": wcp_mode
        }

        # 等待到达时间
        target_time = start_exp + (row['timestamp'] / 1000.0)
        wait_time = target_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)

        # --- 决策阶段 ---
        decision = {}
        controller_should_shed = False

        if strategy == 'mpc':
            # MPC 使用内部控制器（集成模式）
            pass
        elif strategy in ['baseline_naive', 'baseline_fixed', 'static_conservative', 'static_aggressive']:
            # 这些策略不需要调用外部 controller，直接由 worker 内部决定
            pass
        else:
            # 其他策略调用外部 controller
            controller_resp = invoke_controller_lambda(payload, mode=wcp_mode, strategy=strategy)
            if controller_resp and isinstance(controller_resp, dict):
                decision = controller_resp.get('decision', {}) or {}
                controller_should_shed = bool(decision.get('shouldShed') or decision.get('should_shed') or False)

        # --- 执行阶段 ---
        ideal_duration = row['duration']
        task_payload = {
            "task_name": f"TraceReq-{req_id}",
            "simulated_duration_ms": ideal_duration,
            "priority": prio,
            "qos_class": qos_class
        }

        if controller_should_shed and qos_class == "Q3":
            e2e_latency = 0.0
            success = False
            worker_status = "shedded"
        else:
            worker_result = invoke_worker_lambda(
                decision if decision else None,
                task_payload,
                mode='auto',
                strategy=strategy,
                priority=prio,
                metrics=payload['metrics']
            )
            if worker_result is None:
                success = False
                worker_status = "failed"
                e2e_latency = 0.0
            else:
                resp = worker_result.get('response', {}) or {}
                worker_status = resp.get('status', 'unknown')
                success = (worker_status != "failed")
                # 估算延迟：从 worker 返回的 debug 信息或模拟
                if success:
                    debug = resp.get('debug', {})
                    alloc = debug.get('resource_alloc', 1.0)
                    # 简化延迟模型：base_duration / alloc + overhead
                    overhead = 50.0  # 网络 + 调度开销
                    e2e_latency = (ideal_duration / max(0.1, alloc)) + overhead + random.gauss(0, 20)
                else:
                    e2e_latency = 0.0

        # --- SLO 判定 ---
        if worker_status == "shedded":
            if qos_class in ["Q1", "Q2"]:
                met_slo = False
            else:
                met_slo = True  # Q3 可以丢弃
        else:
            met_slo = (e2e_latency <= slo_bound) and success

        is_violation = not met_slo

        # --- 记录结果 ---
        with self.lock:
            self.pending_requests -= 1
            self.slo_violation_window.append(1.0 if is_violation else 0.0)
            if len(self.slo_violation_window) > 100:
                self.slo_violation_window.pop(0)
            if success:
                self.latency_window.append(e2e_latency)
                if len(self.latency_window) > 50:
                    self.latency_window.pop(0)

        # 提取资源分配
        alloc_val = 1.0
        if success:
            try:
                alloc_val = float(resp.get('response', {}).get('debug', {}).get('resource_alloc', 1.0))
            except:
                alloc_val = 1.0

        self.results.append({
            "req_id": req_id,
            "timestamp": time.time() - start_exp,
            "trace_duration": ideal_duration,
            "e2e_latency": e2e_latency,
            "slowdown": e2e_latency / max(1.0, ideal_duration),
            "alloc": alloc_val,
            "slo_violation": is_violation,
            "strategy": strategy,
            "success": success,
            "priority": prio,
            "qos_class": qos_class,
            "worker_status": worker_status,
            "shed_by_worker": (worker_status in ["degraded", "shedded"]),
            "met_slo": met_slo,
            "slo_bound": slo_bound
        })

    def run_experiment(self, strategy, wcp_mode, output_filename):
        """运行一次完整实验"""
        random.seed(42)
        np.random.seed(42)

        print(f"\n{'='*60}")
        print(f"Running Experiment: Strategy='{strategy}', Mode='{wcp_mode}'")
        print(f"{'='*60}")

        self.results = []
        self.slo_violation_window = []
        self.latency_window = []
        self.pending_requests = 0

        # 深拷贝轨迹数据
        self.trace_data = copy.deepcopy(self.raw_trace_data)

        # 应用负载因子（可选）
        load_factor = 1.0
        if load_factor < 1.0:
            self.trace_data = [x for x in self.trace_data if random.random() < load_factor]

        start_exp = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [
                executor.submit(self.run_request, i, row, strategy, wcp_mode, start_exp)
                for i, row in enumerate(self.trace_data)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Thread Error] {e}")

        end_exp = time.time()
        duration = end_exp - start_exp

        output_path = os.path.join(self.output_dir, output_filename)
        print(f"Saving results to {output_path}...")
        pd.DataFrame(self.results).to_csv(output_path, index=False)
        self.analyze_results(strategy, duration)

    def analyze_results(self, strategy, duration=None):
        """分析并打印结果"""
        df = pd.DataFrame(self.results)
        if df.empty:
            print("[Warning] No results to analyze.")
            return

        total_reqs = len(df)
        success_reqs = df['success'].sum()
        failed_reqs = total_reqs - success_reqs

        fail_rate = (failed_reqs / total_reqs * 100) if total_reqs > 0 else 0
        throughput = success_reqs / duration if duration and duration > 0 else 0.0

        print(f"\n=== Strategy: {strategy.upper()} ===")
        print(f"Total Requests: {total_reqs} | Success: {success_reqs} | Failed: {failed_reqs} | Fail Rate: {fail_rate:.2f}%")
        print(f"Duration: {duration:.2f}s | Throughput: {throughput:.2f} RPS")

        if total_reqs > 0:
            # 总体 SLO 违约率
            total_violations = df['slo_violation'].sum()
            real_violation_rate = (total_violations / total_reqs) * 100
            print(f"Overall SLO Violation Rate: {real_violation_rate:.2f}%")

            # 按 QoS 类别统计
            print("\n--- Per-QoS Breakdown ---")
            for qos in ['Q1', 'Q2', 'Q3']:
                d_qos = df[df['qos_class'] == qos]
                if len(d_qos) == 0:
                    continue
                q_total = len(d_qos)
                q_violations = d_qos['slo_violation'].sum()
                q_viol_rate = (q_violations / q_total) * 100
                q_avg_alloc = d_qos['alloc'].mean() * 100.0
                print(f"  {qos}: Count={q_total} | Violation={q_viol_rate:.2f}% | Avg Alloc={q_avg_alloc:.1f}%")

            # 成本指标
            avg_alloc = df['alloc'].mean()
            print(f"\nAverage Resource Allocation (Cost Proxy): {avg_alloc:.3f}x")

            # 延迟统计（仅成功请求）
            df_success = df[df['success'] == True]
            if not df_success.empty:
                p50 = df_success['e2e_latency'].quantile(0.50)
                p90 = df_success['e2e_latency'].quantile(0.90)
                p99 = df_success['e2e_latency'].quantile(0.99)
                print(f"Latency (P50/P90/P99): {p50:.1f}/{p90:.1f}/{p99:.1f} ms")

        print("=" * 60)


def plot_comparison(results_dir, output_dir):
    """绘制对比图表"""
    # 收集所有策略的结果
    strategies = ['baseline_naive', 'baseline_fixed', 'static_conservative', 'static_aggressive', 'mpc']
    summary = []

    for strategy in strategies:
        csv_files = [f for f in os.listdir(results_dir) if f.startswith(f'results_{strategy}') and f.endswith('.csv')]
        if not csv_files:
            continue
        # 合并多次运行
        dfs = [pd.read_csv(os.path.join(results_dir, f)) for f in csv_files]
        df_all = pd.concat(dfs, ignore_index=True)

        total = len(df_all)
        violations = df_all['slo_violation'].sum()
        viol_rate = (violations / total * 100) if total > 0 else 0
        avg_alloc = df_all['alloc'].mean()
        avg_latency = df_all[df_all['success'] == True]['e2e_latency'].mean()

        summary.append({
            'Strategy': strategy,
            'Violation Rate (%)': viol_rate,
            'Avg Cost (alloc)': avg_alloc,
            'Avg Latency (ms)': avg_latency,
            'Total Requests': total
        })

    if not summary:
        print("[Warning] No results found for plotting.")
        return

    df_summary = pd.DataFrame(summary)

    # 打印汇总表
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False, float_format='%.2f'))
    print("="*80)

    # 验证目标
    print("\nValidation against Targets:")
    for _, row in df_summary.iterrows():
        strat = row['Strategy']
        viol = row['Violation Rate (%)']
        cost = row['Avg Cost (alloc)']

        if strat == 'baseline_naive':
            status = "✓" if viol > 10 else "✗ (need >10%)"
            print(f"  {strat}: Violation={viol:.1f}% {status}")
        elif strat == 'mpc':
            status = "✓" if viol <= 10 else "✗ (need ≤10%)"
            print(f"  {strat}: Violation={viol:.1f}% {status} | Cost={cost:.2f}x")
        elif strat == 'static_aggressive':
            # 需要比 MPC 违约率低，且成本 > MPC 2倍
            mpc_row = df_summary[df_summary['Strategy'] == 'mpc']
            if not mpc_row.empty:
                mpc_viol = mpc_row.iloc[0]['Violation Rate (%)']
                mpc_cost = mpc_row.iloc[0]['Avg Cost (alloc)']
                viol_ok = "✓" if viol < mpc_viol else "✗ (need < MPC)"
                cost_ratio = cost / mpc_cost if mpc_cost > 0 else 0
                cost_ok = "✓" if cost_ratio > 2.0 else f"✗ (need >2x, actual {cost_ratio:.2f}x)"
                print(f"  {strat}: Violation={viol:.1f}% {viol_ok} | Cost={cost:.2f}x ({cost_ratio:.2f}x MPC) {cost_ok}")

    # 绘制柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Violation Rate
    axes[0].bar(df_summary['Strategy'], df_summary['Violation Rate (%)'], color=['red', 'orange', 'blue', 'green', 'purple'])
    axes[0].axhline(10, color='red', linestyle='--', label='Target (10%)')
    axes[0].set_ylabel('Violation Rate (%)')
    axes[0].set_title('SLO Violation Rate')
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Cost
    axes[1].bar(df_summary['Strategy'], df_summary['Avg Cost (alloc)'], color=['red', 'orange', 'blue', 'green', 'purple'])
    axes[1].set_ylabel('Avg Resource Allocation')
    axes[1].set_title('Cost (Relative to Full Allocation)')
    axes[1].tick_params(axis='x', rotation=45)

    # 3. Latency
    axes[2].bar(df_summary['Strategy'], df_summary['Avg Latency (ms)'], color=['red', 'orange', 'blue', 'green', 'purple'])
    axes[2].set_ylabel('Avg Latency (ms)')
    axes[2].set_title('Average Latency')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    # 实验配置
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    TRACE_FILE = os.path.join(PROJECT_ROOT, 'datasets', 'processed', 'clean_trace.csv')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'improved_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 策略列表
    STRATEGIES = {
        'baseline_naive': 'baseline',      # 固定 0.6
        'baseline_fixed': 'baseline',      # 固定 0.8
        'static_conservative': 'static',   # 按优先级分配
        'static_aggressive': 'static1',    # 过度分配
        'mpc': 'mpc'                       # MPC
    }

    # 并发数
    THREAD_COUNT = 100

    # 运行实验
    replayer = ImprovedTraceReplayer(trace_file=TRACE_FILE, output_dir=RESULTS_DIR, thread_num=THREAD_COUNT)
    replayer.load_trace()

    # 多次运行取平均
    N_TRIALS = 3
    for trial in range(1, N_TRIALS + 1):
        print(f"\n{'#'*70}")
        print(f"TRIAL {trial}/{N_TRIALS}")
        print(f"{'#'*70}")

        for strategy, wcp_mode in STRATEGIES.items():
            output_filename = f'results_{strategy}_trial{trial}.csv'
            replayer.run_experiment(strategy=strategy, wcp_mode=wcp_mode, output_filename=output_filename)
            time.sleep(2)  # 策略间冷却

    # 汇总并绘图
    plot_comparison(RESULTS_DIR, RESULTS_DIR)

    print("\n所有实验完成！结果已保存到:", RESULTS_DIR)
