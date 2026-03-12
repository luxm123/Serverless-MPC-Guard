import pandas as pd
import json
import os
import numpy as np

def parse_alibaba_gpu_trace(csv_path, output_json, max_steps=300):
    """
    从阿里巴巴 GPU 迹线中提取并发模式。
    使用 openb_pod_list_cpu100.csv，通过 pod 的创建时间分布来模拟并发请求。
    """
    print(f"Reading trace from {csv_path}...")
    # 只读取必要的列：creation_time
    df = pd.read_csv(csv_path, usecols=['creation_time'])
    
    # 将时间归一化到从 0 开始
    min_time = df['creation_time'].min()
    df['normalized_time'] = df['creation_time'] - min_time
    
    # 假设步长为 1000 个时间单位 (对应 1 秒)
    step_size = 1000
    df['step'] = (df['normalized_time'] / step_size).astype(int)
    
    # 按步聚合，统计每步新创建的 pod 数量作为并发请求数
    concurrency_series = df.groupby('step').size()
    
    # 截取一段活跃的迹线
    # 找到第一个并发大于 5 的位置
    start_step = 0
    for step, count in concurrency_series.items():
        if count > 5:
            start_step = step
            break
            
    trace = []
    task_types = ['image_processing', 'pyaes', 'linpack', 'model_serving']
    
    for i in range(max_steps):
        curr_step = start_step + i
        count = int(concurrency_series.get(curr_step, 0))
        
        # 缩放因子：Alibaba 迹线原始并发可能较低，我们将其缩放到 50-200 的压力区间
        # 原始 count 通常在 1-10 之间，我们乘以 15 左右
        scaled_count = int(count * 15) + 10
        scaled_count = min(scaled_count, 500) # 封顶 500
        
        trace.append({
            "step": i,
            "concurrency": scaled_count,
            "task_type": task_types[i % len(task_types)] # 循环切换任务类型，测试自适应性
        })
        
    with open(output_json, 'w') as f:
        json.dump(trace, f, indent=2)
    print(f"Successfully saved {len(trace)} steps to {output_json}")

if __name__ == "__main__":
    csv_path = "benchmarks/clusterdata/cluster-trace-gpu-v2023/csv/openb_pod_list_cpu100.csv"
    output_json = "real_workload.json"
    if os.path.exists(csv_path):
        parse_alibaba_gpu_trace(csv_path, output_json)
    else:
        print(f"Error: {csv_path} not found. Please ensure git clone depth 1 finished.")
