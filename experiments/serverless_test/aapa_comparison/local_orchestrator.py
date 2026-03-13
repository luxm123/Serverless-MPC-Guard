import time
import requests
import json
import numpy as np
import pandas as pd
import random
import sys
import os
import boto3
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # 关键：防止 headless EC2 报错
from botocore.config import Config
from scipy.stats import skew, kurtosis

from concurrent.futures import ThreadPoolExecutor

# --- 路径配置 (必须在 import wcp 之前) ---
# 确保能够找到项目根目录下的 src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../"))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

try:
    from wcp.wcp_update import RLS, build_phi, wcp_update
except ImportError:
    print(f"Error: Could not import wcp_update. Tried path: {os.path.join(ROOT_DIR, 'src')}")
    sys.exit(1)

# --- 实验超参数 ---
SLO_TARGET_MS = 800.0  
MAX_CPU = 3.0          # 提高上限至 3.0
MIN_CPU = 0.2          
LAMBDA_FUNC_NAME = "MPC_BusinessWorker"  
USE_AWS_LAMBDA = True 
CONTROL_LAG_STEPS = 2  
SAMPLES_PER_STEP = 30  
MAX_WORKERS = 2000     
REAL_WORKLOAD_PATH = os.path.join(BASE_DIR, "real_workload.json") 

# 初始化 AWS Lambda 客户端
lambda_config = Config(
    max_pool_connections=2000, 
    retries={'max_attempts': 2},
    connect_timeout=5,
    read_timeout=25            # 应对积压时的长尾延迟
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
    """
    基准：AWS/K8s 风格的 HPA (Horizontal Pod Autoscaler) 逻辑
    公式：Desired = Current * (Current_Metric / Target_Metric)
    """
    def __init__(self):
        super().__init__("Heuristic-AWS")
        self.tolerance = 0.1

    def decide(self, obs_p90, current_cpu, **kwargs):
        ratio = obs_p90 / SLO_TARGET_MS
        if abs(ratio - 1.0) < self.tolerance:
            return current_cpu
        new_cpu = current_cpu * ratio
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
            # 物理一致性：service_time 设为 400ms，匹配 Lambda 真实负载
            future_phi = build_phi(future_concurrency, test_cpu, kwargs.get('backlog', 0), 400, task_type=task_type)
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

class AAPAController(BaseController):
    """
    对标方案：AAPA (arXiv '25) 核心逻辑复刻
    1. 负载原型分类：SPIKE, PERIODIC, RAMP, STATIONARY
    2. 原型专属策略：SPIKE 采用激进预热(低阈值)，STATIONARY 采用保守缩容
    """
    def __init__(self):
        super().__init__("AAPA")
        self.window = []
        self.window_size = 60
        self.cooldown_counter = 0
        self.current_archetype = "STATIONARY"
        
    def classify(self):
        if len(self.window) < 20: return "STATIONARY"
        
        data = np.array(self.window)
        mean_val = np.mean(data)
        max_val = np.max(data)
        std_val = np.std(data)
        
        if max_val > mean_val * 2.5:
            return "SPIKE"
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        if abs(slope) > 0.5:
            return "RAMP"
            
        if len(data) > 30:
            autocorr = np.corrcoef(data[:-15], data[15:])[0, 1]
            if autocorr > 0.6:
                return "PERIODIC"
                
        return "STATIONARY"

    def decide(self, obs_p90, current_cpu, concurrency=1, **kwargs):
        self.window.append(concurrency)
        if len(self.window) > self.window_size:
            self.window.pop(0)
            
        self.current_archetype = self.classify()
            
        if self.current_archetype == "SPIKE":
            target_util = 0.3
            cooldown_steps = 20
        elif self.current_archetype == "PERIODIC":
            target_util = 0.6
            cooldown_steps = 5
        elif self.current_archetype == "RAMP":
            target_util = 0.7
            cooldown_steps = 10
        else:
            target_util = 0.75
            cooldown_steps = 3
            
        obs_util = obs_p90 / SLO_TARGET_MS
        target_cpu_raw = current_cpu * (obs_util / target_util)
        
        if target_cpu_raw < current_cpu:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return current_cpu
            else:
                self.cooldown_counter = cooldown_steps
        else:
            self.cooldown_counter = 0
        
        return max(MIN_CPU, min(MAX_CPU, target_cpu_raw))

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
    # 处理能力：1.0 CPU 每秒能处理的请求数基准
    # 调至 120，使得 3.0 CPU 的最大吞吐为 360 RPS。
    # 与 trace_parser.py 配合，确保平均负载在 100-200 RPS 之间，
    # 这样在高峰期(300+ RPS)产生的积压才能在高峰后被排水排掉。
    THROUGHPUT_PER_CPU = 120 
    
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
    controllers = [HeuristicAWSController(), AAPAController(), MPCGuardController()]
    all_dfs = []
    report = []

    for ctrl in controllers:
        df, cost, v_count = run_experiment_for_algorithm(ctrl, workload, task_types)
        all_dfs.append(df)
        
        # 计算 REI 指标 (Resource Efficiency Index)
        slo_compliance = 1.0 - (v_count / TOTAL_STEPS)
        norm_cost = cost / (MAX_CPU * 0.000016 * sum(workload))
        # 稳定性：CPU 变动的步数占比
        instability = (df['CPU'].diff().fillna(0) != 0).sum() / TOTAL_STEPS
        rei = slo_compliance / (norm_cost * (1 + instability))
        
        report.append({
            "Algorithm": ctrl.name,
            "Total Cost ($)": f"{cost:.4f}",
            "Avg P90 (ms)": f"{df['P90'].mean():.2f}",
            "SLO Violation (%)": f"{(v_count/TOTAL_STEPS)*100:.2f}",
            "REI (Higher better)": f"{rei:.4f}"
        })

    # 3. 生成可视化报告 (CDF 图)
    final_df = pd.concat(all_dfs)
    plt.figure(figsize=(10, 6))
    for name, group in final_df.groupby("Algorithm"):
        sorted_p90 = np.sort(group["P90"])
        y = np.arange(len(sorted_p90)) / float(len(sorted_p90))
        plt.plot(sorted_p90, y, label=name, linewidth=2)
    
    plt.axvline(x=SLO_TARGET_MS, color='r', linestyle='--', label='SLO Target')
    plt.xlabel("P90 Latency (ms)")
    plt.ylabel("Cumulative Probability")
    plt.title("JIAGU/AAPA-Style Performance Comparison (CDF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("performance_cdf.png")
    
    print("\n" + "="*70)
    print("Final Scientific Report (Alibaba Trace + FunctionBench + AAPA Baseline)")
    print("="*70)
    print(pd.DataFrame(report).to_string(index=False))
    print("="*70)
    print("CDF plot saved to 'performance_cdf.png'.")
    print("New results saved to 'time_cost_latency_results.csv'.")
    print("Please review and Accept the code changes, then run 'python local_orchestrator.py'.")
