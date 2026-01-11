import pandas as pd
import os
import numpy as np
import random

def process_azure_trace(input_dir, output_file):
    print(f'Processing Azure 2019 Trace (Lite Version) from {input_dir}...')
    
    if not os.path.exists(input_dir):
        print(f'Error: Input directory {input_dir} not found!')
        return

    # ==========================================
    # 1. 配置：只处理必要的数据范围
    # ==========================================
    TARGET_DAY = 'd01'        # 只看第1天
    MAX_FUNCTIONS = 200       # 只采样前200个函数（足够把单机打满）
    MAX_MINUTES = 60          # 只取前60分钟（足够覆盖30分钟实验）
    
    # ==========================================
    # 2. 读取元数据 (Duration & Memory)
    # ==========================================
    print("Loading metadata (Duration & Memory)...")
    
    # 读取 Duration (只保留必要的列)
    duration_files = [f for f in os.listdir(input_dir) if f.startswith('function_durations_percentiles.anon')]
    df_duration = pd.DataFrame()
    for f in duration_files:
        path = os.path.join(input_dir, f)
        temp = pd.read_csv(path, usecols=['HashFunction', 'Average'])
        df_duration = pd.concat([df_duration, temp], ignore_index=True)
    df_duration.rename(columns={'Average': 'duration'}, inplace=True)
    # 去重，防止同一个函数有多条记录
    df_duration = df_duration.drop_duplicates(subset=['HashFunction'])

    # 读取 Memory
    memory_files = [f for f in os.listdir(input_dir) if f.startswith('app_memory_percentiles.anon')]
    df_memory = pd.DataFrame()
    for f in memory_files:
        path = os.path.join(input_dir, f)
        temp = pd.read_csv(path, usecols=['HashApp', 'AverageAllocatedMb'])
        df_memory = pd.concat([df_memory, temp], ignore_index=True)
    df_memory.rename(columns={'AverageAllocatedMb': 'memory'}, inplace=True)
    df_memory = df_memory.drop_duplicates(subset=['HashApp'])

    # ==========================================
    # 3. 读取调用链 (Invocations) - 核心优化点
    # ==========================================
    print(f"Loading invocations (Day {TARGET_DAY}, First {MAX_FUNCTIONS} funcs, First {MAX_MINUTES} mins)...")
    
    invoke_file = f"invocations_per_function_md.anon.{TARGET_DAY}.csv"
    invoke_path = os.path.join(input_dir, invoke_file)
    
    if not os.path.exists(invoke_path):
        print(f"Error: Target file {invoke_file} not found!")
        return

    # 关键优化：只读取前 N 行 (nrows=MAX_FUNCTIONS)
    # 这样内存占用极低
    df_invokes = pd.read_csv(invoke_path, nrows=MAX_FUNCTIONS)
    
    # ==========================================
    # 4. 数据展开与打散 (Unroll & Jitter)
    # ==========================================
    final_records = []
    
    # 筛选出分钟列（列名为 '1', '2', ... '1440'）
    # 只取前 MAX_MINUTES 列
    minute_cols = [str(i) for i in range(1, MAX_MINUTES + 1) if str(i) in df_invokes.columns]
    
    print("Processing and unrolling data...")
    
    # 预处理元数据索引，加速查找
    dur_map = df_duration.set_index('HashFunction')['duration'].to_dict()
    mem_map = df_memory.set_index('HashApp')['memory'].to_dict()

    for _, row in df_invokes.iterrows():
        func_hash = row['HashFunction']
        app_hash = row['HashApp']
        
        # 获取该函数的元数据
        dur = dur_map.get(func_hash, 100)  # 默认 100ms
        mem = mem_map.get(app_hash, 128)   # 默认 128MB
        
        # 遍历每一分钟
        for col in minute_cols:
            val = row[col]
            if pd.isna(val):
                continue
            count = int(val)
            if count == 0:
                continue
            
            # 这一分钟的起始毫秒数
            base_ms = (int(col) - 1) * 60 * 1000
            
            # 【重要】将 Count 转换为具体的时间戳
            # 比如这一分钟调用了 5 次，我们随机分布这 5 次请求（泊松分布/均匀分布）
            # 避免所有请求都在第 0 毫秒到达
            for _ in range(count):
                # 随机偏移量 0~59999ms
                jitter = random.randint(0, 59999)
                timestamp = base_ms + jitter
                
                final_records.append({
                    'timestamp': timestamp,
                    'duration': int(dur),
                    'memory': int(mem)
                })

    # ==========================================
    # 5. 保存结果
    # ==========================================
    if not final_records:
        print("Warning: No records generated. Check if input data has invocations.")
        return

    df_final = pd.DataFrame(final_records)
    # 按时间排序
    df_final = df_final.sort_values(by='timestamp')
    
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    df_final.to_csv(output_file, index=False)
    print(f'Saved cleaned data to {output_file}')
    print(f'Total Requests: {len(df_final)}')
    print(f'Time Range: 0 to {df_final["timestamp"].max() / 1000 / 60:.1f} minutes')

if __name__ == '__main__':
    # 请确保这个路径是你存放原始大文件的位置
    RAW_DIR = r"G:\datasets\raw\azurefunctions-dataset2019"
    OUTPUT_FILE = r"G:\datasets\processed\clean_trace.csv"
    
    process_azure_trace(RAW_DIR, OUTPUT_FILE)