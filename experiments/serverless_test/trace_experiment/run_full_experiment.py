"""
全自动化多窗口轨迹回放实验
支持：10个窗口 × 7个策略 × 3次重复 = 210次独立运行

使用方法：
python run_full_experiment.py --mode real  # 真实 AWS 环境
python run_full_experiment.py --mode simulation  # 本地仿真
"""
import sys
import os
import time
import json
import argparse
import pandas as pd
import numpy as np
import concurrent.futures
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 动态路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from experiments.serverless_test.trace_experiment.experiment_config import ExperimentConfig
from experiments.serverless_test.wcp_validation.serverless_utils import (
    invoke_controller_lambda,
    invoke_worker_lambda
)


class WindowManager:
    """管理10个实验窗口（5 stable + 5 bursty）"""
    def __init__(self, traces_dir: str):
        self.traces_dir = Path(traces_dir)
        self.windows = {}
        self._load_windows()

    def _load_windows(self):
        """加载所有窗口的 trace 数据"""
        for trace_file in sorted(self.traces_dir.glob("trace_*.csv")):
            window_id = trace_file.stem.replace('trace_', '')
            df = pd.read_csv(trace_file)
            self.windows[window_id] = {
                'id': window_id,
                'type': window_id.split('_')[0],  # 'stable' or 'bursty'
                'data': df,
                'num_requests': len(df),
                'duration_sec': df['timestamp'].max() / 1000.0 if not df.empty else 0
            }

    def list_windows(self) -> List[str]:
        return list(self.windows.keys())

    def get_window(self, window_id: str) -> Dict:
        return self.windows.get(window_id)


class ExperimentRunner:
    """
    单个策略在单个窗口上的实验运行器
    """
    def __init__(self, window_data: pd.DataFrame,
                 strategy: str,
                 wcp_mode: str,
                 thread_num: int = 100,
                 memory_mb: int = 128):
        self.window_data = window_data
        self.strategy = strategy
        self.wcp_mode = wcp_mode
        self.thread_num = thread_num
        self.memory_mb = memory_mb

        # 线程安全的结果收集
        self.results = []
        self.lock = threading.Lock()

        # 滑动窗口统计（实时反馈用）
        self.slo_violation_window = []
        self.latency_window = []
        self.qos_violation = {'Q1': [], 'Q2': [], 'Q3': []}
        self.pending_requests = 0

    def run_request(self, req_id: int, row: pd.Series,
                    start_exp: float, base_p90_cache: List[float]):
        """
        执行单个请求

        Args:
            base_p90_cache: 滑动窗口缓存，用于动态 SLO
        """
        # 确定优先级
        prio = self._sample_priority()
        qos_class = self._priority_to_qos(prio)

        # 获取实时系统状态（线程安全）
        with self.lock:
            current_backlog = self.pending_requests
            current_slo_violation_rate = (sum(self.slo_violation_window) /
                                          len(self.slo_violation_window)
                                          if self.slo_violation_window else 0.0)

            # 动态 base_p90（使用���动窗口）
            if base_p90_cache:
                current_base_p90 = np.median(base_p90_cache[-50:])
            else:
                current_base_p90 = 100.0

        # SLO 阈值（基于 base_p90 × factor）
        slo_factors = {'Q1': 1.0, 'Q2': 1.8, 'Q3': 3.0}
        slo_bound = current_base_p90 * slo_factors[qos_class]

        # 构建 payload
        payload = {
            "metrics": {
                "queue_backlog": current_backlog,
                "concurrency": min(max(1, current_backlog), ExperimentConfig.CONCURRENCY_BUDGET),
                "slo_violation_rate": current_slo_violation_rate,
                "p90": current_base_p90,
                "latency": current_base_p90,
                "base_p90": current_base_p90  # 用于 worker 计算
            },
            "priority": prio,
            "risk": {},
            "strategy": self.strategy,
            "wcp_mode": self.wcp_mode
        }

        # 等待到达时间
        target_time = start_exp + (row['timestamp'] / 1000.0)
        wait_time = target_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)

        # --- 决策阶段 ---
        controller_should_shed = False
        decision = {}

        if self.strategy not in ['baseline', 'static', 'mpc']:
            # 调用外部 controller Lambda
            controller_resp = invoke_controller_lambda(payload, mode=self.wcp_mode, strategy=self.strategy)
            if controller_resp and isinstance(controller_resp, dict):
                decision = controller_resp.get('decision', {}) or {}
                controller_should_shed = bool(decision.get('shouldShed') or decision.get('should_shed', False))

        # --- 执行阶段 ---
        ideal_duration = row['duration']
        task_payload = {
            "task_name": f"Req-{req_id}",
            "simulated_duration_ms": ideal_duration,
            "priority": prio,
            "qos_class": qos_class,
            "memory_mb": self.memory_mb
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
                strategy=self.strategy,
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

                if success:
                    # 提取实际延迟
                    debug = resp.get('debug', {})
                    alloc = debug.get('resource_alloc', 1.0)

                    # 简化延迟模型
                    overhead = 50.0  # 网络+调度
                    e2e_latency = (ideal_duration / max(0.1, alloc)) + overhead
                else:
                    e2e_latency = 0.0

        # --- SLO 判定 ---
        if worker_status == "shedded":
            met_slo = (qos_class == "Q3")  # Q3 可丢弃
        else:
            met_slo = (e2e_latency <= slo_bound) and success

        is_violation = not met_slo

        # --- 记录结果（线程安全）---
        with self.lock:
            self.pending_requests -= 1

            # 更新滑动窗口
            self.slo_violation_window.append(1.0 if is_violation else 0.0)
            if len(self.slo_violation_window) > 100:
                self.slo_violation_window.pop(0)

            self.qos_violation[qos_class].append(1.0 if is_violation else 0.0)
            if len(self.qos_violation[qos_class]) > 100:
                self.qos_violation[qos_class].pop(0)

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
            "window_id": None,  # 由外层设置
            "trial": None,      # 由外层设置
            "trace_duration": ideal_duration,
            "e2e_latency": e2e_latency,
            "alloc": alloc_val,
            "slo_violation": is_violation,
            "strategy": self.strategy,
            "success": success,
            "priority": prio,
            "qos_class": qos_class,
            "worker_status": worker_status,
            "met_slo": met_slo,
            "slo_bound": slo_bound,
            "base_p90": current_base_p90,
            "memory_mb": self.memory_mb
        })

    def _sample_priority(self) -> str:
        """采样优先级（遵循论文分布：25% critical, 40% high, 35% low）"""
        r = np.random.random()
        if r < 0.25:
            return 'critical'
        elif r < 0.65:
            return 'high'
        else:
            return 'standard'

    def _priority_to_qos(self, prio: str) -> str:
        return {'critical': 'Q1', 'high': 'Q2', 'standard': 'Q3'}.get(prio, 'Q3')

    def run_experiment(self, window_id: str, trial: int) -> pd.DataFrame:
        """
        运行完整实验

        Returns:
            DataFrame 包含所有请求的结果
        """
        print(f"\n[Experiment] Window={window_id} | Strategy={self.strategy} | Trial={trial}")

        # 重置状态
        self.results = []
        self.slo_violation_window = []
        self.latency_window = {}
        for q in ['Q1', 'Q2', 'Q3']:
            self.qos_violation[q] = []
        self.pending_requests = 0

        trace_data = self.window_data.copy()
        start_exp = time.time()

        # 并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [
                executor.submit(self.run_request, i, row, start_exp, [])
                for i, row in trace_data.iterrows()
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Thread Error] {e}")

        # 转换为 DataFrame
        df = pd.DataFrame(self.results)
        df['window_id'] = window_id
        df['trial'] = trial

        return df


def run_full_experiment_suite(traces_dir: str,
                             strategies: List[str] = None,
                             n_trials: int = 3,
                             output_dir: str = None):
    """
    运行完整实验套件

    Args:
        traces_dir: 预处理好的 trace 目录（包含 stable_XX.csv, bursty_XX.csv）
        strategies: 策略列表（默认使用所有7个）
        n_trials: 每个窗口策略重复次数
        output_dir: 结果保存目录
    """
    if strategies is None:
        strategies = ExperimentConfig.STRATEGIES

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(SCRIPT_DIR, f"results_{timestamp}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 初始化窗口管理器
    window_mgr = WindowManager(traces_dir)
    windows = window_mgr.list_windows()

    print(f"\n{'='*70}")
    print(f"全自动实验启动")
    print(f"窗口数: {len(windows)} | 策略数: {len(strategies)} | 重复次数: {n_trials}")
    print(f"总运行数: {len(windows) * len(strategies) * n_trials}")
    print(f"输出目录: {output_path}")
    print(f"{'='*70}\n")

    # 保存实验配置
    config_log = {
        'traces_dir': traces_dir,
        'strategies': strategies,
        'n_trials': n_trials,
        'windows': windows,
        'timestamp': timestamp
    }
    with open(output_path / 'experiment_config.json', 'w') as f:
        json.dump(config_log, f, indent=2)

    # 运行所有实验
    all_results = []

    for window_id in windows:
        window_info = window_mgr.get_window(window_id)
        trace_df = window_info['data']

        for strategy in strategies:
            for trial in range(1, n_trials + 1):
                # 创建 runner
                runner = ExperimentRunner(
                    window_data=trace_df,
                    strategy=strategy,
                    wcp_mode='strict' if strategy == 'mpc' else 'baseline',
                    thread_num=ExperimentConfig.THREAD_POOL_SIZE
                )

                # 运行
                try:
                    result_df = runner.run_experiment(window_id, trial)
                    all_results.append(result_df)

                    # 实时保存（防止中断丢失）
                    interim_path = output_path / 'interim_results.csv'
                    if not result_df.empty:
                        result_df.to_csv(interim_path, mode='a' if interim_path.exists() else 'w',
                                       header=not interim_path.exists(), index=False)

                except Exception as e:
                    print(f"[Error] Window={window_id}, Strategy={strategy}, Trial={trial}: {e}")
                    continue

                # 策略间冷却
                time.sleep(2)

    # 合并所有结果
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_path = output_path / 'all_results.csv'
        final_df.to_csv(final_path, index=False)
        print(f"\n[Success] All results saved to {final_path}")
    else:
        print("[Warning] No results collected!")
        return

    # 生成汇总报告
    generate_summary_report(final_df, output_path)

    print(f"\n[Done] Experiment suite completed!")


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """
    生成学术论文风格的汇总报告
    """
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    # 按策略和窗口聚合
    summary = []

    for strategy in df['strategy'].unique():
        strat_df = df[df['strategy'] == strategy]

        total_reqs = len(strat_df)
        success_reqs = strat_df['success'].sum()
        failed_reqs = total_reqs - success_reqs

        # SLO 违约率（包含失败）
        total_violations = (strat_df['slo_violation'].sum() + failed_reqs)
        viol_rate = (total_violations / total_reqs * 100) if total_reqs > 0 else 0.0

        # 成本计算（简化：用 alloc 作为成本代理）
        avg_alloc = strat_df['alloc'].mean() if 'alloc' in strat_df.columns else 1.0

        # 延迟统计（仅成功）
        success_df = strat_df[strat_df['success'] == True]
        if not success_df.empty:
            p50_lat = success_df['e2e_latency'].quantile(0.50)
            p90_lat = success_df['e2e_latency'].quantile(0.90)
            p99_lat = success_df['e2e_latency'].quantile(0.99)
        else:
            p50_lat = p90_lat = p99_lat = 0.0

        # 按窗口分组统计
        window_stats = {}
        for win in sorted(df['window_id'].unique()):
            win_df = strat_df[strat_df['window_id'] == win]
            win_viol = (win_df['slo_violation'].sum() + (len(win_df) - win_df['success'].sum()))
            win_rate = (win_viol / len(win_df) * 100) if len(win_df) > 0 else 0.0
            window_stats[win] = round(win_rate, 2)

        summary.append({
            'Strategy': strategy,
            'Total_Requests': total_reqs,
            'Success_Rate(%)': round(success_reqs / total_reqs * 100, 2) if total_reqs > 0 else 0.0,
            'Violation_Rate(%)': round(viol_rate, 2),
            'Avg_Cost(alloc)': round(avg_alloc, 3),
            'P50_Latency(ms)': round(p50_lat, 1),
            'P90_Latency(ms)': round(p90_lat, 1),
            'P99_Latency(ms)': round(p99_lat, 1),
            'Window_Violations': str(window_stats)
        })

    summary_df = pd.DataFrame(summary)

    # 打印表格
    print("\n" + "-"*80)
    print(summary_df.to_string(index=False, float_format='%.2f'))
    print("-"*80)

    # 保存
    summary_df.to_csv(output_dir / 'summary.csv', index=False)

    # 验证目标达成情况
    print("\n[Validation] Target Check (Violation Rate ≤ 10%):")
    for _, row in summary_df.iterrows():
        strat = row['Strategy']
        viol = row['Violation_Rate(%)']
        if strat == 'mpc':
            status = "✓ PASS" if viol <= 10.0 else "✗ FAIL"
            print(f"  MPC: {viol:.2f}% {status}")
        elif strat == 'oracle':
            print(f"  Oracle: {viol:.2f}% (theoretical upper bound)")
        else:
            # 基线应该比 MPC 差（违约更高）或成本更高
            print(f"  {strat}: {viol:.2f}% (baseline reference)")

    # 生成 LaTeX 表格代码
    print("\n[LaTeX] Table code for paper:")
    print_latex_table(summary_df)


def print_latex_table(df: pd.DataFrame):
    """生成 LaTeX 表格代码"""
    print("\n% Copy this to your paper.tex")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Experimental Results: SLO Violation Rate and Cost Comparison}")
    print("\\label{tab:results}")
    print("\\begin{tabular}{lccccccc}")
    print("\\toprule")
    print("Strategy & Total Reqs & Success Rate & Violation Rate & Avg Cost & P50 & P90 & P99 \\\\")
    print("\\midrule")

    for _, row in df.iterrows():
        line = f"{row['Strategy']} & {int(row['Total_Requests'])} & "
        line += f"{row['Success_Rate(%)']:.1f}\\% & "
        line += f"\\textbf{{{row['Violation_Rate(%)']:.1f}\\%}}" if row['Strategy'] == 'mpc' else f"{row['Violation_Rate(%)']:.1f}\\% & "
        line += f"{row['Avg_Cost(alloc)']:.3f} & "
        line += f"{row['P50_Latency(ms)']:.0f} & {row['P90_Latency(ms)']:.0f} & {row['P99_Latency(ms)']:.0f} \\\\"
        print(line)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run full experiment suite')
    parser.add_argument('--mode', choices=['real', 'simulation'], default='real',
                       help='Run in real AWS environment or local simulation')
    parser.add_argument('--traces-dir', type=str,
                       default=os.path.join(PROJECT_ROOT, 'datasets', 'processed', 'traces'),
                       help='Directory containing trace CSV files')
    parser.add_argument('--strategies', nargs='+', default=None,
                       help='Strategies to run (default: all)')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials per window/strategy')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')

    args = parser.parse_args()

    # 检查 traces 目录
    traces_path = Path(args.traces_dir)
    if not traces_path.exists():
        print(f"[Error] Traces directory not found: {traces_path}")
        print("Please run select_azure_windows.py first to generate traces.")
        sys.exit(1)

    trace_files = list(traces_path.glob("trace_*.csv"))
    if not trace_files:
        print(f"[Error] No trace files found in {traces_path}")
        sys.exit(1)

    print(f"[Info] Found {len(trace_files)} trace files:")
    for f in sorted(trace_files):
        print(f"  - {f.name}")

    # 确认运行
    print(f"\nAbout to run {len(trace_files)} windows × {len(args.strategies or ExperimentConfig.STRATEGIES)} strategies × {args.trials} trials")
    print(f"Total: {len(trace_files) * len(args.strategies or ExperimentConfig.STRATEGIES) * args.trials} experiments")
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)

    # 运行
    run_full_experiment_suite(
        traces_dir=str(args.traces_dir),
        strategies=args.strategies,
        n_trials=args.trials,
        output_dir=args.output_dir
    )
