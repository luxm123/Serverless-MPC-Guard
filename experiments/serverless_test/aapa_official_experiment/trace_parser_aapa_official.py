import pandas as pd
import json
import os

def parse_aapa_official_trace(trace_path, output_json_path, num_steps=1200):
    print(f"Reading AAPA official trace from {trace_path}...")
    df = pd.read_csv(trace_path)
    # 'invocations' 列代表了每分钟的调用次数
    rps_series = df['invocations'] / 60.0
    # 选取一个 1200 秒（20分钟）的典型片段
    selected_rps = rps_series.iloc[100:100+num_steps]
    workload = []
    for rps in selected_rps:
        concurrency = int(rps * 1.5) + 40
        workload.append({
            "concurrency": min(concurrency, 800),
            "task_type": "mix"
        })
    print(f"Generated workload with {len(workload)} steps from AAPA official dataset.")
    with open(output_json_path, 'w') as f:
        json.dump(workload, f)
    print(f"Saved AAPA official workload to {output_json_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "../../../"))
    trace_file = os.path.join(root_dir, "benchmarks/aapa-simulator/dataset/azure-traces/invocations-per-function-app-all.csv")
    output_json = os.path.join(base_dir, "aapa_official_workload.json")
    if not os.path.exists(trace_file):
        print(f"Error: AAPA official trace file not found at {trace_file}")
        print("Please ensure you have cloned the aapa-simulator repo into benchmarks/")
    else:
        parse_aapa_official_trace(trace_file, output_json)
