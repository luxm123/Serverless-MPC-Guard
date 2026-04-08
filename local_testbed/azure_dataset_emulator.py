import pandas as pd
import numpy as np
import os

# 模拟真实的 Azure Functions 2019/2021 数据集结构
# 论文引用: "Azure Functions Dataset 2019/2021" - Microsoft
# 这种数据集通常以 CSV 存储，每一行代表一个函数在 1440 分钟内的调用次数

def generate_azure_format_sample():
    minutes = 1440
    # 模拟三个具有不同特征的函数
    # 1. 突发型 (Bursty)
    # 2. 周期型 (Periodic)
    # 3. 持续型 (Stable)
    
    data = []
    t = np.arange(minutes)
    
    # Bursty Function
    bursty = np.zeros(minutes)
    for _ in range(10):
        start = np.random.randint(0, minutes-30)
        bursty[start:start+30] = np.random.randint(20, 100)
    data.append(["app_1", "func_bursty", "Trigger", *bursty])
    
    # Periodic Function
    periodic = 20 * (np.sin(2 * np.pi * t / 360) + 1) + np.random.normal(0, 2, minutes)
    data.append(["app_2", "func_periodic", "Trigger", *np.clip(periodic, 1, 200)])
    
    # Stable Function
    stable = 10 + np.random.normal(0, 1, minutes)
    data.append(["app_3", "func_stable", "Trigger", *np.clip(stable, 1, 200)])
    
    columns = ["HashApp", "HashFunction", "Trigger"] + [str(i+1) for i in range(minutes)]
    df = pd.DataFrame(data, columns=columns)
    
    output_file = os.path.join(os.path.dirname(__file__), "azure_dataset_emulator_sample.csv")
    df.to_csv(output_file, index=False)
    print(f"Generated Azure-style dataset sample: {output_file}")

if __name__ == "__main__":
    generate_azure_format_sample()
