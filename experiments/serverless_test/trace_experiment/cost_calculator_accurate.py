"""
成本计算器（整合到实验中的模块）
"""
import pandas as pd
import numpy as np

# AWS Lambda 定价（us-east-1）
COST_PER_GB_SEC = 0.00001667  # $/GB-s

def compute_cost_per_request(duration_ms: float, memory_mb: int = 128) -> float:
    """
    计算单次请求成本

    Args:
        duration_ms: 执行时间（毫秒）
        memory_mb: 内存分配（MB），默认 128

    Returns:
        成本（美元）
    """
    duration_sec = duration_ms / 1000.0
    memory_gb = memory_mb / 1024.0
    return duration_sec * memory_gb * COST_PER_GB_SEC


def add_cost_column(df: pd.DataFrame, memory_mb: int = 128) -> pd.DataFrame:
    """
    为 DataFrame 添加 cost 列

    Args:
        df: 包含 e2e_latency 列的 DataFrame
        memory_mb: 内存（MB），如果为 None 则使用 df['memory_mb']

    Returns:
        添加了 cost_usd 列的 DataFrame
    """
    df = df.copy()
    if 'memory_mb' in df.columns and memory_mb is None:
        memory_col = df['memory_mb']
    else:
        memory_col = memory_mb

    df['cost_usd'] = df.apply(
        lambda row: compute_cost_per_request(
            duration_ms=row.get('e2e_latency', 0),
            memory_mb=memory_col if isinstance(memory_col, (int, float)) else memory_col.iloc[row.name]
        ),
        axis=1
    )
    return df


def analyze_costs(all_results_dir: str) -> pd.DataFrame:
    """
    分析实验目录中所有结果文件的成本

    Args:
        all_results_dir: 包含 results_*.csv 的目录

    Returns:
        按策略分组的成本汇总 DataFrame
    """
    results_path = Path(all_results_dir)
    csv_files = sorted(results_path.glob("results_*.csv"))

    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df = add_cost_column(df, memory_mb=128)
            df['source_file'] = f.name
            all_data.append(df)
        except Exception as e:
            print(f"[Warning] Could not read {f}: {e}")

    if not all_data:
        print("[Error] No data found!")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # 聚合统计
    summary = []
    for strategy in combined['strategy'].unique():
        s_df = combined[combined['strategy'] == strategy]
        n = len(s_df)
        total_cost = s_df['cost_usd'].sum()
        avg_cost = s_df['cost_usd'].mean()
        cost_per_million = avg_cost * 1e6

        # 每请求成本（美元）
        per_req_cost_usd = avg_cost

        # 每 100 万次请求成本
        per_million_usd = total_cost / n * 1e6 if n > 0 else 0

        summary.append({
            'Strategy': strategy,
            'Total_Requests': n,
            'Total_Cost_USD': round(total_cost, 6),
            'Avg_Cost_per_Req_USD': round(per_req_cost_usd, 10),
            'Cost_per_Million_USD': round(per_million_usd, 4),
            'Cost_per_1M_USD': round(per_million_usd, 2)
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Total_Cost_USD')

    print("\n" + "="*80)
    print("COST ANALYSIS (AWS Lambda Pricing)")
    print("="*80)
    print(f"Pricing: $0.00001667 per GB-second")
    print("-"*80)
    print(summary_df.to_string(index=False, float_format='%.6f'))
    print("="*80)
    print(f"\nTotal Experiment Cost: ${summary_df['Total_Cost_USD'].sum():.6f}")
    print(f"Most Expensive: {summary_df.iloc[-1]['Strategy']} (${summary_df.iloc[-1]['Total_Cost_USD']:.6f})")
    print(f"Least Expensive: {summary_df.iloc[0]['Strategy']} (${summary_df.iloc[0]['Total_Cost_USD']:.6f})")

    return summary_df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python cost_calculator_accurate.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]
    summary = analyze_costs(results_dir)

    # 保存
    out_path = Path(results_dir) / 'cost_analysis.csv'
    summary.to_csv(out_path, index=False)
    print(f"\nCost analysis saved to {out_path}")
