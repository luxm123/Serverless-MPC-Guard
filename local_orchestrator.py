import time
import requests
import json
import numpy as np
import pandas as pd
import random
import sys
import os
import boto3
from botocore.config import Config

from concurrent.futures import ThreadPoolExecutor

# --- 路径配置 (必须在 import wcp 之前) ---
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from wcp.wcp_update import RLS, build_phi, wcp_update
except ImportError:
    print("Error: Could not import wcp_update. Make sure you are running from the project root.")
    sys.exit(1)

# --- 实验超参数 ---
SLO_TARGET_MS = 800.0  
MAX_CPU = 2.0          
MIN_CPU = 0.2          
LAMBDA_FUNC_NAME = "MPC_BusinessWorker"  
USE_AWS_LAMBDA = True 
CONTROL_LAG_STEPS = 2  
SAMPLES_PER_STEP = 30  
MAX_WORKERS = 2000     # 大幅提升并发处理上限，消除客户端瓶颈
REAL_WORKLOAD_PATH = "real_workload.json" 

# 初始化 AWS Lambda 客户端
lambda_config = Config(
    max_pool_connections=2000, # 匹配 MAX_WORKERS
    retries={'max_attempts': 2},
    connect_timeout=5,
    read_timeout=20            # 稍微延长超时，应对极端排队情况
)
lmb = boto3.client('lambda', region_name='us-east-1', config=lambda_config)

# --- 连通性自检 ---
def check_aws_connectivity():
    print(">>> Checking AWS Connectivity and Lambda Status...", end="", flush=True)
    try:
        lmb.get_function(FunctionName=LAMBDA_FUNC_NAME)
        print(" [OK]")
    except Exception as e:
        print(f" [FAILED]\nError: {e}")
        sys.exit(1)

check_aws_connectivity()

class BaseController:
    def __init__(self, name):
        self.name = name
    def decide(self, obs_p90, current_cpu, concurrency=1, future_concurrency=1, **kwargs):
        raise NotImplementedError

class HeuristicAWSController(BaseController):
    """基准：AWS 风格的步进式调整 (Reactive)"""
    def __init__(self):
        super().__init__("Heuristic-AWS")
    def decide(self, obs_p90, current_cpu, **kwargs):
        if obs_p90 > SLO_TARGET_MS:
            return min(MAX_CPU, current_cpu + 0.2)  # 步进加资源
        elif obs_p90 < SLO_TARGET_MS * 0.5:
            return max(MIN_CPU, current_cpu - 0.1)  # 缓慢缩容
        return current_cpu

class PIDController(BaseController):
    """基准：传统 PID 控制器 (Proportional-Integral-Derivative)"""
    def __init__(self):
        super().__init__("PID")
        self.prev_error = 0
        self.integral = 0
        self.kp, self.ki, self.kd = 0.5, 0.1, 0.05
    def decide(self, obs_p90, current_cpu, **kwargs):
        error = (obs_p90 - SLO_TARGET_MS) / SLO_TARGET_MS
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        new_cpu = current_cpu + output
        return max(MIN_CPU, min(MAX_CPU, new_cpu))

class MPCGuardController(BaseController):
    """
    核心方案：MPC-Guard (论文版实现)
    1. 在线建模：使用 RLS 实时捕捉资源-性能敏感度。
    2. 风险量化：使用 WCP 生成分布无关的置信区间 (Uncertainty Delta)。
    3. 机会约束优化：求解满足 P(Latency <= SLO) >= 1-alpha 的最小资源分配。
    """
    def __init__(self):
        super().__init__("MPC-Guard")
        self.state = {
            'rls_state': None, 
            'last_prediction': 0.0,
            'scores': [], 
            'last_cpu': 1.0, 
            'last_y': 0.0
        }
        
    def decide(self, obs_p90, current_cpu, concurrency=1, future_concurrency=1, task_type='image_processing', **kwargs):
        # 1. 在线系统辨识：更新 RLS 模型并计算 WCP 不确定性边界 (delta)
        # alpha=0.1 意味着我们追求 90% 的置信度满足 SLO
        _, delta, debug = wcp_update(
            self.state, obs_p90, 
            concurrency=concurrency, 
            cpu=current_cpu, 
            backlog=kwargs.get('backlog', 0), # 传递真实积压
            service_time_ms=300,
            task_type=task_type,
            alpha=0.1 
        )
        
        # 2. 机会约束求解 (Chance-Constrained Optimization)
        # 获取最新的 RLS 模型参数 (特征维度已升至 11)
        rls = RLS.from_dict(self.state['rls_state'], n_features=11)
        
        best_cpu = MAX_CPU
        found = False
        
        # 目标：找到最小的 CPU，使得：未来预测延迟 (含积压影响) + WCP不确定性 <= SLO
        safe_slo = SLO_TARGET_MS * 0.95
        
        for test_cpu in np.arange(MIN_CPU, MAX_CPU + 0.01, 0.05):
            # 构造未来时刻特征向量 (包含预判的积压状态)
            future_phi = build_phi(future_concurrency, test_cpu, kwargs.get('backlog', 0), 300, task_type=task_type)
            pred_latency = rls.predict(future_phi)
            
            if pred_latency + delta <= safe_slo:
                best_cpu = test_cpu
                found = True
                break
        
        if not found:
            best_cpu = MAX_CPU
            
        # 3. 动态平滑 (防止资源剧烈震荡导致成本飙升)
        # 允许快速升容，缓慢缩容
        if best_cpu > current_cpu:
            max_change = 1.0  # 允许瞬间拉满
        else:
            max_change = 0.2  # 缩容要谨慎
            
        target_cpu = max(current_cpu - max_change, min(current_cpu + max_change, best_cpu))
        return max(MIN_CPU, min(MAX_CPU, target_cpu))

def run_experiment_for_algorithm(controller, workload_trace, task_type_trace):
    """
    运行基于物理队列状态机的对比实验。
    引入跨步进的积压(Backlog)和排队延迟，真实模拟雪球效应。
    """
    print(f"\n>>> Running STATEFUL-QUEUE Experiment for: {controller.name}", flush=True)
    results = []
    
    cpu_buffer = [1.0] * (CONTROL_LAG_STEPS + 1)
    total_cost = 0.0
    violations = 0
    
    # 物理状态机：当前队列中的积压请求数
    current_backlog = 0
    # 处理能力常数：1.0 CPU 每秒能处理的请求数基准
    # 调低至 100，使得 2.0 CPU 的最大吞吐为 200 RPS，
    # 当 Concurrency 达到 1000+ 时，系统会产生真实的雪崩积压。
    THROUGHPUT_PER_CPU = 100 
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for step, concurrency in enumerate(workload_trace):
            step_start_time = time.time()
            current_effective_cpu = cpu_buffer.pop(0)
            current_task_type = task_type_trace[step]
            
            # 1. 更新物理队列：新请求进入
            current_backlog += concurrency
            
            # 2. 模拟请求的流水线处理
            # 每一秒的处理能力取决于当前的 CPU 分配
            processing_capacity = int(current_effective_cpu * THROUGHPUT_PER_CPU)
            processed_count = min(current_backlog, processing_capacity)
            
            # 采样 20 个代表性请求来测量 P90 (包含排队等待时间)
            def invoke_once(sample_idx):
                # 排队延迟：取决于该请求在队列中的位置
                # 假设采样请求均匀分布在这一秒的处理序列中
                queue_pos = (sample_idx / SAMPLES_PER_STEP) * current_backlog
                wait_time = (queue_pos / (processing_capacity + 1)) * 1000.0 # 毫秒
                
                try:
                    if USE_AWS_LAMBDA:
                        payload = json.dumps({
                            "cpu_limit": current_effective_cpu, 
                            "concurrency": int(concurrency),
                            "task_type": current_task_type
                        })
                        resp = lmb.invoke(FunctionName=LAMBDA_FUNC_NAME, Payload=payload)
                        resp_payload = json.loads(resp['Payload'].read())
                        exec_time = float(resp_payload.get('latency_ms', 200.0))
                        return wait_time + exec_time
                    else:
                        return wait_time + (250.0 / (current_effective_cpu + 0.01))
                except Exception:
                    return wait_time + 1000.0

            # 饱和采样
            sample_latencies = list(executor.map(invoke_once, range(SAMPLES_PER_STEP)))
            
            # 3. 观测物理延迟
            obs_p90 = np.percentile(sample_latencies, 90)
            
            # 4. 更新积压：处理掉的请求离开队列
            current_backlog = max(0, current_backlog - processed_count)
            
            # 统计与记录
            is_violation = obs_p90 > SLO_TARGET_MS
            if is_violation: violations += 1
            total_cost += (current_effective_cpu * 0.000016) * processed_count
            
            results.append({
                "Algorithm": controller.name, "Step": step, "Concurrency": concurrency,
                "CPU": current_effective_cpu, "P90": obs_p90, "Violation": is_violation,
                "TaskType": current_task_type, "Backlog": current_backlog
            })
            
            # 决策：现在必须考虑积压状态
            look_ahead_idx = min(step + CONTROL_LAG_STEPS + 1, len(workload_trace) - 1)
            future_concurrency = workload_trace[look_ahead_idx]
            
            next_cpu = controller.decide(obs_p90, current_effective_cpu, 
                                        concurrency=concurrency, 
                                        future_concurrency=future_concurrency,
                                        task_type=current_task_type,
                                        backlog=current_backlog)
            cpu_buffer.append(next_cpu)
            
            # 强制步进节奏，让实验具有真实的“可观测性”
            elapsed = time.time() - step_start_time
            # 实时吞吐量监控
            real_rps = processed_count / (elapsed if elapsed > 0 else 1.0)
            
            if step % 10 == 0:
                 print(f"Step {step:4d}/{len(workload_trace)} | Task: {current_task_type:15s} | Backlog: {current_backlog:4d} | P90: {obs_p90:7.2f}ms | CPU: {current_effective_cpu:.2f} | RPS: {real_rps:5.1f}", flush=True)

            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

    return pd.DataFrame(results), total_cost, violations

# --- 执行主程序 ---
if __name__ == "__main__":
    # 1. 加载真实的、带有任务类型的负载 (Alibaba Trace 处理结果)
    try:
        with open(REAL_WORKLOAD_PATH, 'r') as f:
            real_workload = json.load(f)
    except FileNotFoundError:
        print(f"Error: {REAL_WORKLOAD_PATH} not found. Running trace_parser.py first...")
        # 自动触发解析
        from trace_parser import parse_alibaba_gpu_trace
        csv_path = "benchmarks/clusterdata/cluster-trace-gpu-v2023/csv/openb_pod_list_cpu100.csv"
        if os.path.exists(csv_path):
            parse_alibaba_gpu_trace(csv_path, REAL_WORKLOAD_PATH)
            with open(REAL_WORKLOAD_PATH, 'r') as f:
                real_workload = json.load(f)
        else:
            print("Critical Error: ClusterData trace file not found.")
            sys.exit(1)

    workload = [item['concurrency'] for item in real_workload]
    task_types = [item['task_type'] for item in real_workload]
    
    TOTAL_STEPS = len(workload)
    
    # 2. 运行对比实验
    controllers = [HeuristicAWSController(), PIDController(), MPCGuardController()]
    all_dfs = []
    report = []

    for ctrl in controllers:
        df, cost, v_count = run_experiment_for_algorithm(ctrl, workload, task_types)
        all_dfs.append(df)
        report.append({
            "Algorithm": ctrl.name,
            "Total Cost ($)": f"{cost:.6f}",
            "Avg P90 (ms)": f"{df['P90'].mean():.2f}",
            "Max P90 (ms)": f"{df['P90'].max():.2f}",
            "SLO Violation (%)": f"{(v_count/TOTAL_STEPS)*100:.2f}"
        })

    # 3. 生成报告
    final_df = pd.concat(all_dfs)
    final_df.to_csv('time_cost_latency_results.csv', index=False)
    
    print("\n" + "="*60)
    print("Final JIAGU-Level Comparative Report (Alibaba Trace + FunctionBench)")
    print("="*60)
    print(pd.DataFrame(report).to_string(index=False))
    print("="*60)
    print("New result saved to 'time_cost_latency_results.csv'.")
    print("Please review and Accept the code changes, then run 'python local_orchestrator.py'.")
