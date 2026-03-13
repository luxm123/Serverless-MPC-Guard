import pandas as pd
import json
import os
import numpy as np
import random

def parse_aapa_kswd_trace(aapa_root, output_json, max_steps=1200):
    """
    直接解析 AAPA 官方的 KSWD (Kubernetes Serverless Workload Dataset) JSON 迹线。
    """
    print(f"Loading AAPA KSWD traces from {aapa_root}...")
    
    # 选取代表性原型 (KSWD 官方分类)
    # Spike: w_0001, Periodic: w_0015, Stationary: w_0005 (示例，需根据实际文件微调)
    workload_ids = {
        "STATIONARY": "w_0005.json",
        "PERIODIC": "w_0003.json", # 根据 ls 结果选择存在的
        "SPIKE": "w_0001.json"
    }
    
    trace = []
    task_types = ['image_processing', 'pyaes', 'linpack', 'model_serving']
    
    step_idx = 0
    # 构造 1200 步的复合迹线：400 步 Stationary -> 400 步 Periodic -> 400 步 Spike
    for archetype, filename in workload_ids.items():
        file_path = os.path.join(aapa_root, "dataset", "workloads", filename)
        if not os.path.exists(file_path):
            # 备选：如果 w_0003 不在，就顺延找一个
            print(f"Warning: {filename} not found, searching fallback...")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            # AAPA 格式: requests_per_second 是一个列表
            rps_list = data['request_trace']['requests_per_second']
            
            for i in range(400):
                if step_idx >= max_steps: break
                
                # 获取原始 RPS 并缩放到我们的实验压强区间 (100-300 RPS)
                raw_rps = rps_list[i % len(rps_list)]
                # 假设 KSWD 原始 RPS 较小，我们乘以 10 左右并加基础负荷
                scaled_count = int(raw_rps * 10) + 80 
                scaled_count = min(scaled_count, 1500)
                
                trace.append({
                    "step": step_idx,
                    "concurrency": scaled_count,
                    "task_type": task_types[step_idx // 300 % len(task_types)],
                    "archetype": archetype # 记录原始原型，用于对标
                })
                step_idx += 1
                
    with open(output_json, 'w') as f:
        json.dump(trace, f, indent=2)
    print(f"Successfully saved {len(trace)} steps (AAPA KSWD-based) to {output_json}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../"))
    
    # EC2 上的路径 (相对于根目录)
    aapa_root = os.path.join(ROOT_DIR, "benchmarks/aapa-simulator")
    output_json = os.path.join(BASE_DIR, "real_workload.json")
    
    if os.path.exists(aapa_root):
        parse_aapa_kswd_trace(aapa_root, output_json)
    else:
        # 本地调试路径 (尝试相对于脚本路径)
        local_aapa = os.path.join(ROOT_DIR, "benchmarks/aapa-simulator")
        if os.path.exists(local_aapa):
            parse_aapa_kswd_trace(local_aapa, output_json)
        else:
            print(f"Error: AAPA root not found at {local_aapa}")
