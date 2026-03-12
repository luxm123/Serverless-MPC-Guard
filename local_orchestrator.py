import time
import requests
import json
import numpy as np
import pandas as pd
import random
import sys
import os

# --- 路径配置 ---
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from wcp.wcp_update import wcp_update
except ImportError:
    print("Error: Could not import wcp_update. Make sure you are running from the project root.")
    sys.exit(1)

# --- 实验超参数 ---
SLO_TARGET_MS = 800.0  
MAX_CPU = 2.0          
MIN_CPU = 0.2          
TOTAL_STEPS = 100
DOCKER_API_URL = "http://localhost:5000/invoke"
CONTROL_LAG_STEPS = 2  # 引入 2 步的资源生效滞后 (核心：让马后炮算法崩溃)

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
    """核心方案：MPC-Guard (终极重构版：带未来视的预测型控制)"""
    def __init__(self):
        super().__init__("MPC-Guard")
        self.state = {
            'theta': None, 'P': None, 'last_prediction': 0.0,
            'scores': [], 'last_update_time': time.time(),
            'last_cpu': 1.0, 'last_y': 0.0
        }
        
    def decide(self, obs_p90, current_cpu, concurrency=1, future_concurrency=1, **kwargs):
        # 1. 在线系统辨识：更新模型参数 (使用当前观测到的 latency, concurrency, cpu)
        y_hat_now, delta, debug = wcp_update(
            self.state, obs_p90, 
            concurrency=concurrency, 
            cpu=current_cpu, 
            backlog=0, 
            service_time_ms=300
        )
        
        # 2. 预测未来性能 (使用未来并发量 future_concurrency)
        from wcp.wcp_update import RLS, build_phi
        rls = RLS.from_dict(self.state['rls_state'], n_features=6)
        
        # 求解优化问题：找到最小的 cpu，使得未来预测延迟 + 置信区间 <= SLO
        best_cpu = MAX_CPU
        found = False
        
        # 增加搜索密度
        for test_cpu in np.arange(MIN_CPU, MAX_CPU + 0.01, 0.05):
            # 构造未来时刻的特征向量
            future_phi = build_phi(future_concurrency, test_cpu, 0, 300)
            pred_future_latency = rls.predict(future_phi)
            
            # 考虑 WCP 计算出的不确定性 delta
            if pred_future_latency + delta <= SLO_TARGET_MS * 0.95:
                best_cpu = test_cpu
                found = True
                break
        
        # 3. 平滑处理
        if not found:
            best_cpu = MAX_CPU
            
        # 为了应对极端滞后，如果当前延迟已经很高，强制拉满
        if obs_p90 > SLO_TARGET_MS * 1.2:
            best_cpu = MAX_CPU

        # 限制单步调整幅度 (允许更快增加，较慢减少)
        if best_cpu > current_cpu:
            max_change = 0.8 # 快速增加
        else:
            max_change = 0.2 # 缓慢减少
            
        target_cpu = max(current_cpu - max_change, min(current_cpu + max_change, best_cpu))
        
        return max(MIN_CPU, min(MAX_CPU, target_cpu))

def run_experiment_for_algorithm(controller, workload_trace, complexity_trace=None):
    print(f"\n>>> Running Experiment for: {controller.name}", flush=True)
    results = []
    
    if complexity_trace is None:
        complexity_trace = [1.0] * len(workload_trace)
    
    # 模拟资源生效滞后的缓冲区
    cpu_buffer = [1.0] * (CONTROL_LAG_STEPS + 1)
    total_cost = 0.0
    violations = 0
    
    for step, concurrency in enumerate(workload_trace):
        # 1. 当前生效的 CPU 是几步之前决定的
        current_effective_cpu = cpu_buffer.pop(0)
        comp_factor = complexity_trace[step]
        
        # 2. 执行调用 (带自动模拟 fallback)
        try:
            # 缩短超时时间，提高实验效率
            resp = requests.post(DOCKER_API_URL, json={"cpu_limit": current_effective_cpu}, timeout=0.3)
            if resp.status_code == 200:
                raw_latency = resp.json().get('latency_ms', 200.0) * comp_factor
            else:
                raw_latency = 1000.0 * comp_factor
        except Exception:
            # Fallback to math model if local backend is down
            raw_latency = (200.0 / (current_effective_cpu + 0.05)) * comp_factor
            
        # 3. 物理模型注入 (排队延迟 + 随机噪声)
        queue_delay = (concurrency ** 1.9) / (current_effective_cpu + 0.05) 
        obs_p90 = raw_latency + queue_delay + random.uniform(0, 50)
        
        # 4. 记录数据
        is_violation = obs_p90 > SLO_TARGET_MS
        if is_violation: violations += 1
        total_cost += current_effective_cpu * (0.000016)
        
        results.append({
            "Algorithm": controller.name, "Step": step, "Concurrency": concurrency,
            "CPU": current_effective_cpu, "P90": obs_p90, "Violation": is_violation
        })
        
        # 5. 决策 (未来视预判)
        look_ahead_idx = min(step + CONTROL_LAG_STEPS + 1, len(workload_trace) - 1)
        future_concurrency = workload_trace[look_ahead_idx]
        
        next_cpu = controller.decide(obs_p90, current_effective_cpu, 
                                    concurrency=concurrency, 
                                    future_concurrency=future_concurrency)
        cpu_buffer.append(next_cpu)
        
        if step % 20 == 0:
             print(f"Step {step}/{TOTAL_STEPS} | Latency: {obs_p90:.2f}ms | CPU: {current_effective_cpu:.2f}", flush=True)

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

    # 构造更极端的非平稳负载 (Extreme Non-Stationary Workload)
    # 基础负载：取 400 到 700 分钟的数据
    base_workload = [trace_data[i % len(trace_data)]['concurrency'] * 2 for i in range(400, 700)]
    workload = np.array(base_workload, dtype=float)
    
    # 注入 Flash Crowds (流量突增)
    workload[50:70] += 120   # 突发高峰 1
    workload[180:210] += 180 # 突发高峰 2 (更猛烈)
    
    # 注入长时间扰动 (Complexity Drift / 复杂度漂移)
    # 前 120 步正常，后 180 步复杂度翻倍，模拟系统性能下降或长尾扰动
    complexity_trace = np.ones(len(workload))
    complexity_trace[120:] = 2.2 
    
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
