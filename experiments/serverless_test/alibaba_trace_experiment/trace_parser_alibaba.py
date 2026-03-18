import pandas as pd
import json
import os

def parse_alibaba_trace(trace_path, output_json_path, num_steps=1200):
    """
    从 Alibaba Trace 中解析并生成一个包含平稳期和高峰期的混合负载。
    """
    print(f"Reading Alibaba trace from {trace_path}...")
    df = pd.read_csv(trace_path)

    # 按时间戳排序
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 将时间戳转换为每秒的请求数 (RPS)
    df['time_sec'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds().astype(int)
    rps_series = df.groupby('time_sec').size()

    # 选取一个包含“平稳”和“高峰”的典型片段
    # 比如，我们可以选取第 3000 秒到第 4200 秒的数据
    start_sec = 3000
    end_sec = start_sec + num_steps
    selected_rps = rps_series.reindex(range(start_sec, end_sec), fill_value=0)

    workload = []
    for rps in selected_rps:
        # 将原始 RPS 映射到我们的实验并发量级 (e.g., 50-300)
        concurrency = int(rps * 0.5) + 50
        workload.append({
            "concurrency": min(concurrency, 800), # 设置上限防止过载
            "task_type": "mix" # 假设是混合任务
        })
    
    print(f"Generated workload with {len(workload)} steps.")
    with open(output_json_path, 'w') as f:
        json.dump(workload, f)
    print(f"Saved Alibaba workload to {output_json_path}")

if __name__ == "__main__":
    # 假设 benchmarks 文件夹在项目根目录
    # 注意：你需要先下载 Alibaba Trace v2018 并解压
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "../../../"))
    
    # 原始数据路径
    trace_file = os.path.join(root_dir, "benchmarks/alibaba-cluster-trace-v2018/batch_instance.csv")
    # 输出的负载文件路径
    output_json = os.path.join(base_dir, "alibaba_workload.json")

    if not os.path.exists(trace_file):
        print(f"Error: Alibaba trace file not found at {trace_file}")
        print("Please download and place it in the benchmarks/ directory.")
    else:
        parse_alibaba_trace(trace_file, output_json)
