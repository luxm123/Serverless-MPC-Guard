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
LAMBDA_FUNC_NAME = "MPC_BusinessWorker"  # 对标 deploy_infra.py 中的业务函数
USE_AWS_LAMBDA = True # 优先使用真实 Lambda 以获得真实数据
CONTROL_LAG_STEPS = 2  # 引入 2 步的资源生效滞后
MAX_WORKERS = 200       # 提升并发处理能力，支持饱和负载模拟

# 初始化 AWS Lambda 客户端 (预先配置连接池以支持高并发)
lambda_config = Config(max_pool_connections=200, retries={'max_attempts': 0})
lmb = boto3.client('lambda', region_name='us-east-1', config=lambda_config)

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
        
    def decide(self, obs_p90, current_cpu, concurrency=1, future_concurrency=1, **kwargs):
        # 1. 在线系统辨识：更新 RLS 模型并计算 WCP 不确定性边界 (delta)
        # alpha=0.1 意味着我们追求 90% 的置信度满足 SLO
        _, delta, debug = wcp_update(
            self.state, obs_p90, 
            concurrency=concurrency, 
            cpu=current_cpu, 
            backlog=0, 
            service_time_ms=300,
            alpha=0.1 
        )
        
        # 2. 机会约束求解 (Chance-Constrained Optimization)
        # 获取最新的 RLS 模型参数
        rls = RLS.from_dict(self.state['rls_state'], n_features=6)
        
        best_cpu = MAX_CPU
        found = False
        
        # 目标：找到最小的 CPU，使得：未来预测延迟 + WCP不确定性 <= SLO
        # 预留 5% 的控制余量
        safe_slo = SLO_TARGET_MS * 0.95
        
        for test_cpu in np.arange(MIN_CPU, MAX_CPU + 0.01, 0.05):
            # 构造未来时刻特征向量
            future_phi = build_phi(future_concurrency, test_cpu, 0, 300)
            pred_latency = rls.predict(future_phi)
            
            # 核心约束：预测值 + 风险边界 <= SLO
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

def run_experiment_for_algorithm(controller, workload_trace, complexity_trace=None):
    print(f"\n>>> Running Saturation-Load Experiment for: {controller.name}", flush=True)
    results = []
    
    if complexity_trace is None:
        complexity_trace = [1.0] * len(workload_trace)
    
    # 模拟资源生效滞后的缓冲区
    cpu_buffer = [1.0] * (CONTROL_LAG_STEPS + 1)
    total_cost = 0.0
    violations = 0
    
    # 使用线程池模拟真实饱和负载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for step, concurrency in enumerate(workload_trace):
            current_effective_cpu = cpu_buffer.pop(0)
            comp_factor = complexity_trace[step]
            
            # --- 饱和负载逻辑：采样数随并发同步增加 ---
            # 模拟每 5 个并发用户产生一个活跃采样请求
            num_samples = max(10, int(concurrency / 5)) 
            
            def invoke_once(_):
                try:
                    if USE_AWS_LAMBDA:
                        # 真实负载下，Lambda 内部也会感知到并发压力 (通过 concurrency 参数模拟)
                        payload = json.dumps({"cpu_limit": current_effective_cpu, "concurrency": int(concurrency)})
                        resp = lmb.invoke(FunctionName=LAMBDA_FUNC_NAME, Payload=payload)
                        resp_payload = json.loads(resp['Payload'].read())
                        return float(resp_payload.get('latency_ms', 200.0))
                    else:
                        # 仿真：资源竞争导致的非线性延迟增加
                        base = 250.0 / (current_effective_cpu + 0.1)
                        contention = (concurrency / 500.0) ** 2 * 100.0 # 并发越高，竞争越剧烈
                        return base + contention
                except Exception:
                    return 1000.0 # 饱和状态下的请求丢弃惩罚

            # 并发执行饱和采样
            start_step_time = time.time()
            sample_latencies = list(executor.map(invoke_once, range(num_samples)))
            
            # 计算这一步真实的 P90 延迟
            raw_p90 = np.percentile(sample_latencies, 90) * comp_factor
            
            # --- 饱和排队模型：当并发超过 CPU 承载能力时，延迟指数级上升 ---
            # 临界容量：每个 CPU 核心支持约 100 并发
            capacity = current_effective_cpu * 100.0
            if concurrency > capacity:
                # 超过容量的部分产生严重的排队
                queue_delay = ((concurrency - capacity) / capacity) * 500.0 
            else:
                queue_delay = (concurrency / capacity) * 50.0
                
            obs_p90 = raw_p90 + queue_delay + random.uniform(0, 30)
            
            # 记录数据
            is_violation = obs_p90 > SLO_TARGET_MS
            if is_violation: violations += 1
            total_cost += (current_effective_cpu * 0.000016) * num_samples
            
            results.append({
                "Algorithm": controller.name, "Step": step, "Concurrency": concurrency,
                "CPU": current_effective_cpu, "P90": obs_p90, "Violation": is_violation
            })
            
            # 决策 (未来视预判)
            look_ahead_idx = min(step + CONTROL_LAG_STEPS + 1, len(workload_trace) - 1)
            future_concurrency = workload_trace[look_ahead_idx]
            
            next_cpu = controller.decide(obs_p90, current_effective_cpu, 
                                        concurrency=concurrency, 
                                        future_concurrency=future_concurrency)
            cpu_buffer.append(next_cpu)
            
            if step % 10 == 0:
                 elapsed = time.time() - start_step_time
                 print(f"Step {step}/{len(workload_trace)} | P90: {obs_p90:.2f}ms | CPU: {current_effective_cpu:.2f} | Samples: {num_samples} | Time: {elapsed:.2f}s", flush=True)

    return pd.DataFrame(results), total_cost, violations

# --- 执行主程序 ---
if __name__ == "__main__":
    # 1. 加载负载轨迹
    try:
        with open('local_testbed/workload_trace.json', 'r') as f:
            trace_data = json.load(f)
    except FileNotFoundError:
        print("Error: local_testbed/workload_trace.json not found. Run azure_dataset_emulator.py first.")
        sys.exit(1)

    # 构造极其恶劣的饱和负载 (Extremely Adversarial Saturated Workload)
    # 1. 基础负载：大幅提升并发基数 (x10)
    base_workload = [trace_data[i % len(trace_data)]['concurrency'] * 10 for i in range(400, 700)]
    workload = np.array(base_workload, dtype=float)
    
    # 2. 注入剧烈的 Flash Crowds (流量爆发)
    workload[50:80] += 500   # 爆发 1：并发直接拉升 500
    workload[180:220] += 800 # 爆发 2：极高并发，挑战系统极限
    
    # 3. 注入高频抖动 (Jitter)
    jitter = np.random.uniform(-50, 50, size=len(workload))
    workload = np.clip(workload + jitter, 10, 2000) # 最高并发可达 2000
    
    # 4. 注入剧烈的协变量偏移 (Extreme Complexity Drift)
    # 模拟在实验中后期，系统遭遇性能黑洞或外部依赖崩溃，基础耗时飙升 2.5 倍
    complexity_trace = np.ones(len(workload))
    complexity_trace[120:] = 2.5 
    
    TOTAL_STEPS = len(workload)
    
    # 2. 运行对比实验
    controllers = [HeuristicAWSController(), PIDController(), MPCGuardController()]
    all_dfs = []
    report = []

    for ctrl in controllers:
        df, cost, v_count = run_experiment_for_algorithm(ctrl, workload, complexity_trace)
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
    print("Final Comparative Report (Adversarial Environment)")
    print("="*60)
    print(pd.DataFrame(report).to_string(index=False))
    print("="*60)
    print("New result saved to 'time_cost_latency_results.csv'.")
    print("Please review and Accept the code changes, then run 'python local_orchestrator.py'.")
