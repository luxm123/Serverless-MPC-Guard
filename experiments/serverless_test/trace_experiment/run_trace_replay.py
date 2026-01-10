import sys
import os
import time
import random
import concurrent.futures
import pandas as pd
import numpy as np

# 添加项目根路径到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from experiments.serverless_test.wcp_validation.serverless_utils import invoke_controller_lambda, invoke_worker_lambda

class TraceReplayer:
    def __init__(self, trace_file="trace_data.csv", output_file="experiment_results.csv", thread_num=50):
        self.trace_file = trace_file
        self.output_file = output_file
        self.results = []
        self.thread_num = thread_num

    def generate_dummy_trace(self, count=100):
        """当真实Trace文件缺失时，生成模拟突发流量的测试数据"""
        print(f"[Info] '{self.trace_file}' not found. Generating DUMMY trace for testing...")
        
        data = []
        current_time = 0
        
        # 模拟三个流量阶段：稳定->突发->冷却
        for i in range(count):
            if 30 < i < 70:
                inter_arrival = random.expovariate(1.0 / 50.0)
            else:
                inter_arrival = random.expovariate(1.0 / 500.0)
                
            current_time += inter_arrival
            
            duration = int(np.random.lognormal(mean=4.0, sigma=0.8))
            duration = max(10, min(duration, 5000))
            
            data.append({
                "timestamp": int(current_time),
                "duration": duration,
                "memory": 128
            })
            
        df = pd.DataFrame(data)
        df.to_csv(self.trace_file, index=False)
        print(f"[Info] Dummy trace generated: {len(data)} requests.")

    def load_trace(self):
        if not os.path.exists(self.trace_file):
            self.generate_dummy_trace()
            
        print(f"[Info] Loading trace from {self.trace_file}...")
        self.trace_data = pd.read_csv(self.trace_file).sort_values(by="timestamp").to_dict('records')
        print(f"[Info] Loaded {len(self.trace_data)} requests.")

    def run_request(self, req_id, row, strategy, wcp_mode, start_exp):
        """执行单个Trace请求"""
        payload = {
            "metrics": {},
            "priority": "standard",
            "risk": {},
            "strategy": strategy,
            "wcp_mode": wcp_mode
        }
        
        # 按Trace时间戳等待请求到达
        target_time = start_exp + (row['timestamp'] / 1000.0)
        now = time.time()
        wait_time = target_time - now
        if wait_time > 0:
            time.sleep(wait_time)

        e2e_latency = 0.0
        slowdown = 0.0
        is_violation = False
        success = True
        ideal = row['duration']

        try:
            start_t = time.time()
            # 调用控制器获取调度决策
            controller_resp = invoke_controller_lambda(payload, mode=wcp_mode)
            decision = controller_resp.get('decision', {}) if controller_resp else {}
            
            # 调用工作函数执行任务
            task_payload = {
                "task_name": f"TraceReq-{req_id}",
                "simulated_duration_ms": row['duration']
            }
            invoke_worker_lambda(decision, task_payload, mode='auto')
            
            end_t = time.time()
            e2e_latency = (end_t - start_t) * 1000.0
            slowdown = e2e_latency / max(1.0, ideal)

            # 计算SLO违例（阈值为任务时长的2倍）
            slo_target = ideal * 2.0
            is_violation = e2e_latency > slo_target

        except Exception as e:
            success = False
            print(f"[Error] Req {req_id} failed: {str(e)}")

        self.results.append({
            "req_id": req_id,
            "trace_duration": ideal,
            "e2e_latency": e2e_latency,
            "slowdown": slowdown,
            "slo_violation": is_violation,
            "strategy": strategy,
            "success": success
        })
        
        if req_id % 20 == 0:
            print(f"[{strategy}] Req {req_id}: Ideal={ideal}ms -> Actual={e2e_latency:.1f}ms (Slowdown={slowdown:.2f}) | Success={success}")

    def run_experiment(self, strategy='mpc', wcp_mode='strict'):
        print(f"\n>>> Starting Experiment: Strategy={strategy}, Mode={wcp_mode}, Threads={self.thread_num} <<<")
        self.results = []
        start_exp = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = []
            for i, row in enumerate(self.trace_data):
                futures.append(executor.submit(self.run_request, i, row, strategy, wcp_mode, start_exp))
            concurrent.futures.wait(futures)
            
        print(f">>> Experiment Finished. Saving results to {self.output_file}...")
        pd.DataFrame(self.results).to_csv(self.output_file, index=False)
        self.analyze_results()

    def analyze_results(self):
        """分析实验结果"""
        df = pd.DataFrame(self.results)
        if df.empty:
            print("No results to analyze.")
            return
        
        # 过滤失败请求
        df_success = df[df['success'] == True]
        fail_rate = (len(df) - len(df_success)) / len(df) * 100
        
        print("\n=== Experiment Summary ===")
        print(f"Total Requests: {len(df)} | Success: {len(df_success)} | Fail Rate: {fail_rate:.2f}%")
        print(f"Avg Slowdown:   {df_success['slowdown'].mean():.2f}")
        print(f"P99 Slowdown:   {df_success['slowdown'].quantile(0.99):.2f}")
        print(f"SLO Violation Rate: {(df_success['slo_violation'].sum() / len(df_success)) * 100:.2f}%")
        print("==========================\n")

if __name__ == "__main__":
    # 统一指定Trace文件（相对路径，Windows/Linux双系统兼容）
    trace_path = "./datasets/processed/clean_trace.csv"
    thread_count = 50

    # 实例化Replayer并加载一次数据，全局复用
    replayer = TraceReplayer(trace_file=trace_path, thread_num=thread_count)
    replayer.load_trace()

    # 第一步：跑 基线方案
    print("\n--- Running Baseline Experiment ---")
    replayer.output_file = os.path.join(os.path.dirname(__file__), "results_baseline.csv")
    replayer.run_experiment(strategy='baseline', wcp_mode='baseline')

    # 第二步：再跑 你的MPC方案
    print("\n--- Running MPC Experiment ---")
    replayer.output_file = os.path.join(os.path.dirname(__file__), "results_mpc.csv")
    replayer.run_experiment(strategy='mpc', wcp_mode='strict')