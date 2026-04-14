"""
成本计算器（修正版）
基于真实 AWS Lambda 定价公式
Cost = duration(ms) × memory(GB) × $0.00001667 / 1000
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def compute_request_cost(duration_ms: float, memory_mb: float = 128) -> float:
    """
    计算单次请求成本

    Args:
        duration_ms: 执行时间（毫秒）
        memory_mb: 分配内存（MB）

    Returns:
        成本（美元）
    """
    # AWS Lambda 定价（us-east-1）:
    # $0.0000166667 per GB-second
    duration_sec = duration_ms / 1000.0
    memory_gb = memory_mb / 1024.0
    cost = duration_sec * memory_gb * 0.00001667
    return cost


def compute_experiment_cost(df: pd.DataFrame, memory_mb: float = None) -> Dict[str, float]:
    """
    计算整个实验的��成本

    Args:
        df: 结果 DataFrame
        memory_mb: 如果指定，使用统一内存；否则用每行的 memory_mb

    Returns:
        包含总成本、每请求成本、按策略分组的成本
    """
    if df.empty:
        return {'total_cost': 0.0, 'per_request_cost': 0.0, 'by_strategy': {}}

    # 确定内存列
    if 'memory_mb' in df.columns and memory_mb is None:
        memory_col = df['memory_mb']
    else:
        memory_col = memory_mb if memory_mb else 128.0

    # 计算每行成本
    costs = []
    for idx, row in df.iterrows():
        duration = row.get('e2e_latency', row.get('duration_ms', 0))
        mem = memory_col if isinstance(memory_col, (int, float)) else memory_col.iloc[idx]
        costs.append(compute_request_cost(duration, mem))

    df_with_cost = df.copy()
    df_with_cost['cost_usd'] = costs

    total_cost = sum(costs)
    per_request_cost = total_cost / len(df)

    # 按策略分组
    by_strategy = {}
    for strategy in df['strategy'].unique():
        strat_df = df_with_cost[df_with_cost['strategy'] == strategy]
        by_strategy[strategy] = strat_df['cost_usd'].sum()

    return {
        'total_cost': total_cost,
        'per_request_cost': per_request_cost,
        'by_strategy': by_strategy,
        'cost_per_million': per_request_cost * 1e6
    }


def compute_multi_trial_cost(results_dir: str) -> pd.DataFrame:
    """
    汇总多个 trial 的成本

    Args:
        results_dir: 包含 all_results.csv 的目录

    Returns:
        按策略聚合的成本统计 DataFrame
    """
    results_path = Path(results_dir)
    all_files = list(results_path.glob("results_*.csv"))

    if not all_files:
        print(f"[Warning] No result files in {results_dir}")
        return pd.DataFrame()

    all_dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        if 'strategy' not in df.columns:
            continue
        df['source_file'] = f.name
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # 计算成本
    cost_stats = compute_experiment_cost(combined)

    # 生成汇总表
    summary = []
    for strategy, cost in cost_stats['by_strategy'].items():
        strat_df = combined[combined['strategy'] == strategy]
        n_reqs = len(strat_df)
        avg_cost_per_req = cost / n_reqs if n_reqs > 0 else 0

        summary.append({
            'Strategy': strategy,
            'Total_Requests': n_reqs,
            'Total_Cost_USD': round(cost, 6),
            'Avg_Cost_per_Req': round(avg_cost_per_req, 10),
            'Cost_per_Million': round(avg_cost_per_req * 1e6, 4)
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Total_Cost_USD')

    print("\n" + "="*70)
    print("COST ANALYSIS")
    print("="*70)
    print(summary_df.to_string(index=False, float_format='%.6f'))
    print("="*70)
    print(f"\nTotal Experiment Cost: ${cost_stats['total_cost']:.6f}")
    print(f"Cost per Million Requests: ${cost_stats['cost_per_million']:.4f}")

    return summary_df


if __name__ == "__main__":
    # 测试
    print("Testing cost calculator...")

    # 示例：100ms, 128MB
    cost1 = compute_request_cost(100, 128)
    print(f"100ms @ 128MB = ${cost1:.10f}")

    cost2 = compute_request_cost(500, 512)
    print(f"500ms @ 512MB = ${cost2:.10f}")

    print("\nCost per 1M invocations (128MB, 100ms):")
    print(f"  ${cost1 * 1e6:.4f}")
