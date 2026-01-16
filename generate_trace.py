import pandas as pd
import numpy as np

def generate_trace(output_file='datasets/processed/clean_trace.csv', num_requests=1000, duration_s=20):
    """
    生成更真实的 Trace 数据，足够支撑 Flash Crowd 实验。
    """
    print(f"Generating {num_requests} requests over {duration_s} seconds...")
    
    # 1. Base Traffic (Poisson Process)
    timestamps = np.sort(np.random.uniform(0, duration_s * 1000, num_requests))
    
    # 2. Durations (Log-Normal Distribution, mean=50ms, sigma=0.5)
    # 模拟真实业务处理时间
    durations = np.random.lognormal(mean=np.log(50), sigma=0.5, size=num_requests)
    durations = np.clip(durations, 10, 500) # 限制在 10ms - 500ms 之间

    df = pd.DataFrame({
        'timestamp': timestamps,
        'duration': durations
    })
    
    # 3. 确保时间戳是整数
    df['timestamp'] = df['timestamp'].astype(int)
    df['duration'] = df['duration'].astype(int)

    # 4. Save
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}. Preview:")
    print(df.head())

if __name__ == "__main__":
    generate_trace()
