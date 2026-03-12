import json
import boto3
import time
import random

def physical_workload(cpu_limit):
    """
    通过矩阵乘法模拟真实的物理计算任务。
    计算量随 cpu_limit 的减少而线性增加，从而模拟资源受限时的真实延迟。
    """
    # 基础计算量：在 1.0 CPU 下，200x200 矩阵乘法约需 150-200ms (取决于 Lambda 实例)
    # 当 cpu_limit=0.2 时，计算量扩大 5 倍
    u = max(0.1, min(2.0, float(cpu_limit)))
    size = int(200 * (1.0 / u))
    
    # 构造随机矩阵并进行乘法运算 (纯 Python 循环模拟 CPU 密集型任务)
    # 这会产生真实的 CPU 占用、内存访问和指令周期消耗
    res = 0
    for i in range(size):
        for j in range(100):
            res += (i * j) % 1234
    return res

def lambda_handler(event, context):
    """
    真实的 Serverless 业务 Worker。
    性能受真实的 CPU 指令周期、内存访问和网络延迟驱动。
    """
    cpu_limit = event.get('cpu_limit', 1.0)
    concurrency = event.get('concurrency', 1)
    
    start_time = time.time()
    
    # 模拟真实计算开销
    result = physical_workload(cpu_limit)
    
    # 模拟随机的 I/O 抖动 (真实网络环境中的波动)
    time.sleep(random.uniform(0.01, 0.05))
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000.0
    
    return {
        'statusCode': 200,
        'latency_ms': latency_ms,
        'cpu_limit': cpu_limit,
        'concurrency': concurrency,
        'result_hash': result % 10000,
        'timestamp': time.time()
    }
