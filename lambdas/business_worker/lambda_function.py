import json
import boto3
import time
import random

def physical_workload(cpu_limit, comp_factor=1.0):
    """
    矩阵式物理负载，加入安全上限防止超时。
    复杂度漂移现在也发生在 Lambda 内部。
    """
    u = max(0.1, min(2.0, float(cpu_limit)))
    # 负载量 = 基础规模 * 复杂度因子 / CPU分配
    size = int(120 * (1.0 / u) * float(comp_factor)) 
    
    res = 0
    for i in range(size):
        for j in range(100):
            res += (i * j) % 1234
    return res

def lambda_handler(event, context):
    """
    真实的 Serverless 业务 Worker。
    """
    cpu_limit = event.get('cpu_limit', 1.0)
    concurrency = event.get('concurrency', 1)
    comp_factor = event.get('comp_factor', 1.0)
    
    start_time = time.time()
    
    # 执行物理功耗，复杂度完全由云端承受
    result = physical_workload(cpu_limit, comp_factor)
    
    # 模拟随机抖动
    time.sleep(random.uniform(0.01, 0.03))
    
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
