from flask import Flask, request, jsonify
import time
import os
import math
import random

app = Flask(__name__)

# 模拟非平稳环境：随时间漂移的计算复杂度
def get_complexity_factor():
    # 使用正弦波模拟周期性的背景负载干扰 + 随机噪声
    t = time.time()
    drift = 0.2 * math.sin(t / 300)  # 5分钟一个周期的漂移
    noise = random.uniform(-0.1, 0.1)
    return 1.0 + drift + noise

def heavy_computation(n, cpu_limit):
    # 模拟受资源限制的计算过程
    # 增加一个非线性的性能惩罚：当CPU极低时，延迟不成比例增加
    effective_cpu = cpu_limit + 0.05
    effective_n = int(n / effective_cpu)
    
    result = 0.0
    for i in range(effective_n):
        result += math.sqrt(i)
    return result

@app.route('/invoke', methods=['POST'])
def invoke():
    data = request.get_json() or {}
    complexity = data.get('complexity', 10**6)
    cpu_limit = float(data.get('cpu_limit', 0.5))
    
    # 基础计算量 (单位：秒)
    base_workload = 0.05  # 降低基础负载，防止超时
    
    # 引入环境干扰因子 (带保护)
    try:
        factor = get_complexity_factor()
    except:
        factor = 1.0
    
    # 计算实际执行时间 (正比于复杂度，反比于CPU分配)
    # 增加 CPU 限制的下限保护，防止除零错误或执行时间过长
    safe_cpu = max(cpu_limit, 0.1)
    execution_time = (base_workload * factor) / safe_cpu
    
    # 强制限制最大执行时间为 2 秒，防止实验卡死
    execution_time = min(execution_time, 2.0)
    
    # 模拟网络往返和框架开销 (10-20ms)
    overhead = random.uniform(0.01, 0.02)
    total_latency_ms = (execution_time + overhead) * 1000  # 转换为毫秒
    
    print(f"Request: CPU={cpu_limit:.2f}, Factor={factor:.2f}, Latency={total_latency_ms:.2f}ms")
    
    time.sleep(execution_time) # 真实阻塞
    
    return jsonify({
        "status": "success",
        "latency_ms": total_latency_ms,
        "cpu_limit": cpu_limit,
        "complexity": complexity,
        "factor": factor
    })

if __name__ == '__main__':
    print("Function Emulator starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
