import sys
import os
import time
import concurrent.futures
import pandas as pd
import numpy as np

# --- 动态路径设置 ---
# 将项目根目录添加到系统路径，以便导入项目内模块（例如 `from src.utils import ...`）
# 这种方式使得脚本在任何位置都能被正确执行
try:
    # 获取当前脚本的目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 从脚本目录向上回溯三层，找到项目根目录 (trace_experiment -> serverless_test -> experiments -> Serverless-MPC-Guard)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    # 为Jupyter等交互式环境提供备用方案
    PROJECT_ROOT = os.path.abspath('.')
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

# 假设的工具函数导入路径，在实际项目中，这些函数应放在共享模块中
from experiments.serverless_test.wcp_validation.serverless_utils import invoke_controller_lambda, invoke_worker_lambda


class TraceReplayer:
    """
    从CSV文件加载处理过的请求轨迹，并根据不同的策略重放负载，以测量系统性能。
    """
    def __init__(self, trace_file, output_dir, thread_num=50):
        self.trace_file = trace_file
        self.output_dir = output_dir
        self.results = []
        self.thread_num = thread_num
        self.trace_data = []

    def load_trace(self):
        """从指定的CSV文件加载轨迹数据。"""
        if not os.path.exists(self.trace_file):
            print(f"[致命错误] 轨迹文件不存在: '{self.trace_file}'")
            print("请确认文件路径是否正确。")
            sys.exit(1)  # 如果数据文件缺失，则终止程序

        print(f"[信息] 正在从 {self.trace_file} 加载轨迹...")
        self.trace_data = pd.read_csv(self.trace_file).sort_values(by="timestamp").to_dict('records')
        print(f"[信息] 已加载 {len(self.trace_data)} 个请求。")

    def run_request(self, req_id, row, strategy, wcp_mode, start_exp):
        """
        执行单个请求。此函数由线程池并发调用。
        """
        payload = {
            "metrics": {}, "priority": "standard", "risk": {},
            "strategy": strategy, "wcp_mode": wcp_mode
        }

        # 根据轨迹中的时间戳，等待并模拟请求的到达时间
        target_time = start_exp + (row['timestamp'] / 1000.0)
        wait_time = target_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)

        e2e_latency, slowdown, is_violation = 0.0, 0.0, False
        success = True
        ideal_duration = row['duration']

        try:
            start_t = time.time()
            # 1. 调用控制器获取调度决策
            controller_resp = invoke_controller_lambda(payload, mode=wcp_mode)
            decision = controller_resp.get('decision', {}) if controller_resp else {}

            # 2. 使用决策调用工作函数
            task_payload = {"task_name": f"TraceReq-{req_id}", "simulated_duration_ms": ideal_duration}
            invoke_worker_lambda(decision, task_payload, mode='auto')

            end_t = time.time()
            e2e_latency = (end_t - start_t) * 1000.0
            slowdown = e2e_latency / max(1.0, ideal_duration)

            # SLO违约判断：端到端延迟是否超过理想时长的2倍
            is_violation = e2e_latency > (ideal_duration * 2.0)

        except Exception as e:
            success = False
            print(f"[错误] 请求 {req_id} 失败: {str(e)}")

        # 记录本次请求的结果
        self.results.append({
            "req_id": req_id, "trace_duration": ideal_duration, "e2e_latency": e2e_latency,
            "slowdown": slowdown, "slo_violation": is_violation, "strategy": strategy, "success": success
        })

        if req_id % 50 == 0:  # 定期打印进度
            print(f"[{strategy}] Req {req_id}: Ideal={ideal_duration}ms -> Actual={e2e_latency:.1f}ms (Slowdown={slowdown:.2f})")

    def run_experiment(self, strategy, wcp_mode, output_filename):
        """为给定的策略运行一次完整的实验。"""
        print(f"\n>>> 开始实验: Strategy='{strategy}', Mode='{wcp_mode}', Threads={self.thread_num} <<<")
        self.results = []  # 为新实验重置结果
        start_exp = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [executor.submit(self.run_request, i, row, strategy, wcp_mode, start_exp) for i, row in enumerate(self.trace_data)]
            concurrent.futures.wait(futures)

        output_path = os.path.join(self.output_dir, output_filename)
        print(f">>> 实验结束. 正在保存结果到 {output_path}...")
        pd.DataFrame(self.results).to_csv(output_path, index=False)
        self.analyze_results(strategy)

    def analyze_results(self, strategy):
        """分析并打印实验结果摘要。"""
        df = pd.DataFrame(self.results)
        if df.empty:
            print("[警告] 没有可供分析的结果。")
            return

        df_success = df[df['success'] == True]
        total_reqs, success_reqs = len(df), len(df_success)
        fail_rate = (total_reqs - success_reqs) / total_reqs * 100 if total_reqs > 0 else 0

        print(f"\n=== '{strategy}' 策略实验摘要 ===")
        print(f"总请求数: {total_reqs} | 成功: {success_reqs} | 失败率: {fail_rate:.2f}%")
        if success_reqs > 0:
            violation_rate = (df_success['slo_violation'].sum() / success_reqs) * 100
            print(f"平均减速因子: {df_success['slowdown'].mean():.2f}")
            print(f"P99 减速因子: {df_success['slowdown'].quantile(0.99):.2f}")
            print(f"SLO 违约率: {violation_rate:.2f}%")
        print("==============================\n")


if __name__ == "__main__":
    # --- 实验配置 ---
    # 动态定位项目根目录并构建数据集的绝对路径
    # 修正：向上回溯3层即可到达 Serverless-MPC-Guard
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    TRACE_FILE_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'processed', 'clean_trace.csv')
    
    # 定义实验结果的保存目录
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    THREAD_COUNT = 50  # 并发请求数

    # --- 运行实验 ---
    # 1. 初始化Replayer并加载一次数据
    replayer = TraceReplayer(trace_file=TRACE_FILE_PATH, output_dir=RESULTS_DIR, thread_num=THREAD_COUNT)
    replayer.load_trace()

    # 2. 运行基线（Baseline）实验
    replayer.run_experiment(
        strategy='baseline',
        wcp_mode='baseline',
        output_filename='results_baseline.csv'
    )

    # 3. 运行你的MPC实验
    replayer.run_experiment(
        strategy='mpc',
        wcp_mode='strict',
        output_filename='results_mpc.csv'
    )

    print("所有实验已完成。")
