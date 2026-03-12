import numpy as np
import json
import os

def generate_24h_trace():
    minutes = 1440 # 24小时，每分钟一个采样点
    t = np.linspace(0, 24, minutes)
    
    # 1. 基础负载：双峰模式（模拟典型的 Web 应用早晚高峰）
    # 使用两个高斯分布叠加模拟
    peak1 = 8 * np.exp(-(t - 9)**2 / (2 * 2**2))  # 早上 9 点高峰
    peak2 = 12 * np.exp(-(t - 20)**2 / (2 * 3**2)) # 晚上 8 点高峰
    base_load = peak1 + peak2 + 3 # 基础并发为 3
    
    # 2. 随机突发负载 (Flash Crowds)
    # 模拟 24 小时内出现 4 次突发流量
    bursts = np.zeros(minutes)
    np.random.seed(42)
    for _ in range(4):
        start = np.random.randint(0, minutes - 60)
        duration = np.random.randint(20, 50)
        magnitude = np.random.uniform(10, 25)
        bursts[start:start+duration] = magnitude
        
    # 3. 协变量偏移：任务复杂度随时间缓慢漂移 (Non-stationarity)
    # 模拟函数处理的数据量或计算密集度在变化
    complexity_drift = 10**6 * (1 + 0.4 * np.sin(np.pi * t / 12) + 0.2 * np.random.normal(0, 0.1, minutes))
    
    trace = []
    for i in range(minutes):
        # 最终并发数 = 基础 + 突发 + 观测噪声
        concurrency = int(base_load[i] + bursts[i] + np.random.normal(0, 1.5))
        trace.append({
            "minute": i,
            "timestamp": f"{int(t[i]):02d}:{int((t[i]%1)*60):02d}",
            "concurrency": max(1, concurrency),
            "complexity": int(max(10**5, complexity_drift[i]))
        })
    
    output_path = os.path.join(os.path.dirname(__file__), "workload_trace.json")
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Successfully generated 24-hour trace with {minutes} points: {output_path}")

if __name__ == "__main__":
    generate_24h_trace()
