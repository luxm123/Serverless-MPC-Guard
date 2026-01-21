import sys
import os
import time
import threading
import copy
import concurrent.futures
import pandas as pd
import numpy as np
import random

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
        self.raw_trace_data = []
        self.slo_violation_window = []
        self.latency_window = []
        self.qos_violation_window = {'Q1': [], 'Q2': [], 'Q3': []}
        self.qos_drop_window = {'Q1': [], 'Q2': [], 'Q3': []}
        self.pending_requests = 0  # Global counter for client-side queue depth
        self.lock = threading.Lock()

    def load_trace(self):
        """从指定的CSV文件加载轨迹数据。"""
        if not os.path.exists(self.trace_file):
            print(f"[致命错误] 轨迹文件不存在: '{self.trace_file}'")
            print("请确认文件路径是否正确。")
            sys.exit(1)  # 如果数据文件缺失，则终止程序

        print(f"[信息] 正在从 {self.trace_file} 加载轨迹...")
        self.raw_trace_data = pd.read_csv(self.trace_file).sort_values(by="timestamp").to_dict('records')
        print(f"[信息] 已加载 {len(self.raw_trace_data)} 个请求 (Raw)。")

    def inject_flash_crowd(self, peak_time=5.0, duration=1.0, requests=100):
        """
        Dynamically inject a flash crowd (burst of requests) into the current trace data.
        This modifies self.trace_data in-memory without touching the source file.
        """
        print(f"[Info] Injecting flash crowd (Q3 flood): {requests} requests at t={peak_time}s")
        flash_crowd_data = []
        for _ in range(requests):
            # Generate random timestamp around peak_time (Gaussian distribution)
            timestamp_ms = (peak_time + random.gauss(0, duration/4)) * 1000.0
            flash_crowd_data.append({
                "timestamp": max(0, timestamp_ms),
                "duration": random.randint(30, 100),
                "is_flash": True
            })
        
        self.trace_data.extend(flash_crowd_data)
        self.trace_data.sort(key=lambda x: x['timestamp'])
        print(f"[Info] Trace data size increased to {len(self.trace_data)} requests.")

    def run_request(self, req_id, row, strategy, wcp_mode, start_exp):
        """
        执行单个请求。此函数由线程池并发调用。
        """
        prio = None
        if row.get('is_flash'):
            prio = "low"
        else:
            r = random.random()
            if r < 0.30:
                prio = "critical"
            elif r < 0.65:
                prio = "standard"
            else:
                prio = "low"
        qos_class = "Q1" if prio == "critical" else ("Q2" if prio == "standard" else "Q3")

        current_slo_violation_rate = 0.0
        current_p90_latency = 100.0
        q1_violation_rate = 0.0
        q2_violation_rate = 0.0
        q3_violation_rate = 0.0
        q1_drop_rate = 0.0
        q2_drop_rate = 0.0
        q3_drop_rate = 0.0
        current_backlog = 0
        with self.lock:
            self.pending_requests += 1
            current_backlog = self.pending_requests
            if self.slo_violation_window:
                current_slo_violation_rate = sum(self.slo_violation_window) / len(self.slo_violation_window)
            if self.latency_window:
                sorted_lat = sorted(self.latency_window)
                idx = int(len(sorted_lat) * 0.9)
                if idx < len(sorted_lat):
                    current_p90_latency = sorted_lat[idx]
            q1_v = self.qos_violation_window.get('Q1') or []
            q2_v = self.qos_violation_window.get('Q2') or []
            q3_v = self.qos_violation_window.get('Q3') or []
            q1_d = self.qos_drop_window.get('Q1') or []
            q2_d = self.qos_drop_window.get('Q2') or []
            q3_d = self.qos_drop_window.get('Q3') or []
            if q1_v:
                q1_violation_rate = sum(q1_v) / len(q1_v)
            if q2_v:
                q2_violation_rate = sum(q2_v) / len(q2_v)
            if q3_v:
                q3_violation_rate = sum(q3_v) / len(q3_v)
            if q1_d:
                q1_drop_rate = sum(q1_d) / len(q1_d)
            if q2_d:
                q2_drop_rate = sum(q2_d) / len(q2_d)
            if q3_d:
                q3_drop_rate = sum(q3_d) / len(q3_d)

        payload = {
            "metrics": {
                "queue_backlog": current_backlog,  # Real-time Client-Side Injection
                "slo_violation_rate": current_slo_violation_rate,
                "p90": current_p90_latency,
                "latency": current_p90_latency,
                "q1_violation_rate": q1_violation_rate,
                "q2_violation_rate": q2_violation_rate,
                "q3_violation_rate": q3_violation_rate,
                "q1_worker_drop_rate": q1_drop_rate,
                "q2_worker_drop_rate": q2_drop_rate,
                "q3_worker_drop_rate": q3_drop_rate
            }, 
            "priority": prio, "risk": {},
            "strategy": strategy, "wcp_mode": wcp_mode
        }

        # 根据轨迹中的时间戳，等待并模拟请求的到达时间
        target_time = start_exp + (row['timestamp'] / 1000.0)
        wait_time = target_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)

        e2e_latency, slowdown, is_violation = 0.0, 0.0, False
        ideal_duration = row['duration']
        controller_ok = True
        worker_ok = True
        controller_should_shed = False
        degrade_plan = None
        admit_thr = None
        pred_total_ms = None
        worker_status = "unknown"

        try:
            start_t = time.time()

            decision = {}
            # OPTIMIZATION: For 'mpc' (Integrated), skip external controller call.
            # The Worker runs middleware internally, saving ~200ms RTT.
            if strategy != 'baseline' and strategy != 'mpc':
                controller_resp = invoke_controller_lambda(payload, mode=wcp_mode, strategy=strategy)
                if controller_resp and isinstance(controller_resp, dict):
                    decision = controller_resp.get('decision', {}) or {}
                    controller_should_shed = bool(decision.get('shouldShed') or decision.get('should_shed') or False)
                    degrade_plan = decision.get('degrade_plan')
                    admit_thr = decision.get('admit_threshold_ms')
                    pred_total_ms = decision.get('pred_total_latency_ms')
                else:
                    controller_ok = False

            task_payload = {
                "task_name": f"TraceReq-{req_id}",
                "simulated_duration_ms": ideal_duration,
                "priority": prio
            }
            if controller_should_shed and qos_class == "Q3":
                worker_status = "shedded"
                end_t = time.time()
                e2e_latency = (end_t - start_t) * 1000.0
                slowdown = e2e_latency / max(1.0, ideal_duration)
            else:
                worker_result = invoke_worker_lambda(
                    decision,
                    task_payload,
                    mode='auto',
                    strategy=strategy,
                    priority=prio
                )
                if worker_result is None:
                    worker_ok = False
                else:
                    resp = worker_result.get('response', {}) or {}
                    worker_status = resp.get('status', 'unknown')

                end_t = time.time()
                e2e_latency = (end_t - start_t) * 1000.0
                slowdown = e2e_latency / max(1.0, ideal_duration)

        except Exception as e:
            controller_ok = False
            worker_ok = False
            print(f"[错误] 请求 {req_id} 失败: {str(e)}")
        finally:
            with self.lock:
                self.pending_requests -= 1

        success = controller_ok and worker_ok

        # 更新 SLO 阈值以符合现实网络环境 (Network RTT ~170ms)
        slo_map = {"Q1": 1000.0, "Q2": 1800.0, "Q3": 3000.0}
        slo_bound = slo_map.get(qos_class, 2000.0)
        if controller_should_shed and worker_status == "shedded":
            if qos_class in ["Q1", "Q2"]:
                met_slo = False
            else:
                met_slo = True
        else:
            if qos_class in ["Q1", "Q2"] and worker_status in ["degraded", "shedded"]:
                met_slo = False
            else:
                met_slo = (e2e_latency <= slo_bound) and success
        if qos_class == "Q3":
            shed_by_worker = worker_status in ["degraded", "shedded"] or controller_should_shed
        else:
            shed_by_worker = worker_status == "degraded"
        is_violation = not met_slo

        violation_val = 1.0 if not met_slo else 0.0
        drop_val = 1.0 if shed_by_worker else 0.0
        with self.lock:
            self.slo_violation_window.append(violation_val)
            if len(self.slo_violation_window) > 100:
                self.slo_violation_window.pop(0)
            qos_violation_list = self.qos_violation_window.get(qos_class)
            if qos_violation_list is not None:
                qos_violation_list.append(violation_val)
                if len(qos_violation_list) > 100:
                    qos_violation_list.pop(0)
            qos_drop_list = self.qos_drop_window.get(qos_class)
            if qos_drop_list is not None:
                qos_drop_list.append(drop_val)
                if len(qos_drop_list) > 100:
                    qos_drop_list.pop(0)
            if worker_status != "shedded":
                self.latency_window.append(e2e_latency)
                if len(self.latency_window) > 50:
                    self.latency_window.pop(0)

        self.results.append({
            "req_id": req_id,
            "trace_duration": ideal_duration,
            "e2e_latency": e2e_latency,
            "slowdown": slowdown,
            "slo_violation": is_violation,
            "strategy": strategy,
            "controller_ok": controller_ok,
            "worker_ok": worker_ok,
            "success": success,
            "priority": prio,
            "qos_class": qos_class,
            "is_flash": bool(row.get('is_flash', False)),
            "controller_should_shed": controller_should_shed,
            "worker_status": worker_status,
            "shed_by_worker": shed_by_worker,
            "degrade_plan": degrade_plan,
            "admit_thr_ms": admit_thr,
            "pred_total_ms": pred_total_ms,
            "met_slo": met_slo,
            "slo_bound": slo_bound
        })

        if req_id % 50 == 0:  # 定期打印进度
            print(f"[{strategy}] Req {req_id}: Ideal={ideal_duration}ms -> Actual={e2e_latency:.1f}ms (Slowdown={slowdown:.2f})")

    def run_experiment(self, strategy, wcp_mode, output_filename):
        """为给定的策略运行一次完整的实验。"""
        print(f"\n>>> 开始实验: Strategy='{strategy}', Mode='{wcp_mode}', Threads={self.thread_num} <<<")
        
        # Reset trace data from raw source for every experiment to ensure consistency
        if not hasattr(self, 'raw_trace_data') or not self.raw_trace_data:
             # Fallback if load_trace wasn't called or failed
             self.load_trace()
        
        # Deep copy to ensure fresh start for each run
        self.trace_data = copy.deepcopy(self.raw_trace_data)

        # --- OPTIONAL: LOAD REDUCTION ---
        # If the dataset is too aggressive (physical overload), reduce the load here.
        # Set load_factor < 1.0 to drop requests randomly.
        load_factor = 0.7  # <--- 70% Load: Reduce intensity to avoid physical limit
        if load_factor < 1.0:
            print(f"[Info] Applying Load Factor {load_factor}. Dropping {100*(1-load_factor):.1f}% of requests.")
            self.trace_data = [x for x in self.trace_data if random.random() < load_factor]
        # --------------------------------
        
        # Inject flash crowd to simulate high concurrency
        self.inject_flash_crowd(peak_time=5.0, requests=100)

        self.results = []  # 为新实验重置结果
        start_exp = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [executor.submit(self.run_request, i, row, strategy, wcp_mode, start_exp) for i, row in enumerate(self.trace_data)]
            # Check for exceptions in threads
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Thread Error] {e}")

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
            # QoS 维度评估
            if 'qos_class' in df_success.columns:
                for qos in ['Q1', 'Q2', 'Q3']:
                    d = df_success[df_success['qos_class'] == qos]
                    if len(d) == 0:
                        continue
                    shed_rate = (d['shed_by_worker'].sum() / len(d)) * 100
                    ctrl_shed = (d['controller_should_shed'].sum() / len(d)) * 100
                    met_slo_rate = (d['met_slo'].sum() / len(d)) * 100
                    slo_violation_rate_q = 100.0 - met_slo_rate
                    p50 = d['e2e_latency'].quantile(0.50)
                    p90 = d['e2e_latency'].quantile(0.90)
                    p99 = d['e2e_latency'].quantile(0.99)
                    print(f"- {qos}: 数量={len(d)} | 满足SLO={met_slo_rate:.2f}% | 违约率={slo_violation_rate_q:.2f}% | 触发丢弃(控制器)={ctrl_shed:.2f}% | 实际丢弃(Worker)={shed_rate:.2f}%")
                    print(f"       延迟(P50/P90/P99) = {p50:.1f}/{p90:.1f}/{p99:.1f} ms")
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

    THREAD_COUNT = 50  # 降低并发，避免瞬间打死 Lambda 冷启动

    # --- 运行实验 ---
    # 1. 初始化Replayer并加载一次数据
    replayer = TraceReplayer(trace_file=TRACE_FILE_PATH, output_dir=RESULTS_DIR, thread_num=THREAD_COUNT)
    replayer.load_trace()

    # 2. 运行基线（Baseline）实验
    # replayer.run_experiment(
    #     strategy='baseline',
    #     wcp_mode='baseline',
    #     output_filename='results_baseline.csv'
    # )

    # 3. 运行静态优先级实验
    # replayer.run_experiment(
    #     strategy='static',
    #     wcp_mode='baseline',
    #     output_filename='results_static.csv'
    # )

    # 4. 运行MPC实验
    replayer.run_experiment(
        strategy='mpc',
        wcp_mode='strict',
        output_filename='results_mpc.csv'
    )

    print("所有实验已完成。")
