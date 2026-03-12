import pandas as pd
import json
import os
import numpy as np

def parse_alibaba_gpu_trace(csv_path, output_json, max_steps=1200):
    """
    从阿里巴巴 GPU 迹线中提取并发模式，并引入“复杂度漂移”（Task Phase Shift）。
    """
    print(f"Reading trace from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['creation_time'])
    
    min_time = df['creation_time'].min()
    df['normalized_time'] = df['creation_time'] - min_time
    
    step_size = 1000
    df['step'] = (df['normalized_time'] / step_size).astype(int)
    
    concurrency_series = df.groupby('step').size()
    
    # 找到第一个活跃位置
    start_step = 0
    for step, count in concurrency_series.items():
        if count > 3:
            start_step = step
            break
            
    trace = []
    task_types = ['image_processing', 'pyaes', 'linpack', 'model_serving']
    
    # 复杂度漂移逻辑：每隔一段随机长度切换任务类型
    current_task_type = task_types[0]
    phase_remaining = random.randint(50, 150)
    
    for i in range(max_steps):
        curr_step = start_step + i
        count = int(concurrency_series.get(curr_step, 0))
        
        # 极高压缩放：基础并发 100 + 动态并发(80-120倍)
        # 模拟“突发高峰” (Flash Crowds)
        noise = random.uniform(0.8, 1.2)
        scaled_count = int(count * 120 * noise) + 150 
        scaled_count = min(scaled_count, 2000) # 封顶 2000，匹配 Client MAX_WORKERS
        
        # 切换任务类型阶段
        if phase_remaining <= 0:
            current_task_type = random.choice(task_types)
            phase_remaining = random.randint(50, 150)
        phase_remaining -= 1
        
        trace.append({
            "step": i,
            "concurrency": scaled_count,
            "task_type": current_task_type
        })
        
    with open(output_json, 'w') as f:
        json.dump(trace, f, indent=2)
    print(f"Successfully saved {len(trace)} steps (approx 40min) to {output_json}")

if __name__ == "__main__":
    import random # 确保 random 已导入
    csv_path = "benchmarks/clusterdata/cluster-trace-gpu-v2023/csv/openb_pod_list_cpu100.csv"
    output_json = "real_workload.json"
    if os.path.exists(csv_path):
        parse_alibaba_gpu_trace(csv_path, output_json)
    else:
        print(f"Error: {csv_path} not found. Please ensure git clone depth 1 finished.")
