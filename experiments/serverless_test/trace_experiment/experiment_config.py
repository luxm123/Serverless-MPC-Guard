"""
实验配置常量
统一管理所有实验参数，确保可重复性
"""
import os
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ExperimentConfig:
    """主实验配置"""
    # 并发预算（固定）
    CONCURRENCY_BUDGET = 10

    # SLO 配置
    SLO_MULTIPLIER = 1.2  # base_p90 × 1.2
    QOS_CLASSES = {
        "Q1": {"name": "critical", "slo_factor": 1.0},      # 1000ms
        "Q2": {"name": "high", "slo_factor": 1.8},          # 1800ms
        "Q3": {"name": "standard", "slo_factor": 3.0},      # 3000ms
    }
    BASE_SLO_MS = 1000.0  # Q1 的基准 SLO

    # Cost 计算（AWS Lambda 定价）
    COST_PER_GB_MS = 0.00001667  # $/GB-ms (0.00001667 $/GB-s)

    # 窗口配置
    WINDOW_MINUTES = 30  # 每个窗口 30 分钟
    MINUTES_PER_DAY = 1440

    # 轨迹筛选条件
    STABLE_CV_THRESHOLD = 0.3
    BURSTY_CV_THRESHOLD = 0.5
    MEAN_RPS_MIN = 8.0
    MEAN_RPS_MAX = 12.0

    # 实验重复次数
    N_TRIALS = 3  # 每个窗口策略运行 3 次

    # 策略列表（按预期性能排序）
    STRATEGIES = [
        'static_0.6',      # 最差基线
        'static_0.8',      # 保守基线
        'static_1.0',      # 高成本基线
        'aws_tt',          # AWS Target Tracking
        'hpa_baseline',    # HPA PID 控制
        'mpc',             # 我们的方法
        'oracle',          # 理论上限
    ]

    # 预期表现（用于结果验证）
    EXPECTED_ORDER = {
        'static_0.6': {'violation': 'highest', 'cost': 'lowest'},
        'static_0.8': {'violation': 'high', 'cost': 'low'},
        'static_1.0': {'violation': 'low', 'cost': 'highest'},
        'aws_tt': {'violation': 'medium', 'cost': 'medium'},
        'hpa_baseline': {'violation': 'medium-high', 'cost': 'medium'},
        'mpc': {'violation': '≤10%', 'cost': 'low-medium'},
        'oracle': {'violation': 'lowest', 'cost': 'lowest'},
    }

    # AWS 区域（用于实验部署）
    AWS_REGION = 'us-east-1'

    # 线程池配置
    THREAD_POOL_SIZE = 200

    # 结果保存
    RESULTS_DIR = 'experiments/serverless_test/trace_experiment/final_results'
    FIGURE_DIR = 'experiments/serverless_test/trace_experiment/figures'


# 成本计算函数
def compute_cost(duration_ms: float, memory_mb: float) -> float:
    """
    计算单次请求的 AWS Lambda 成本
    Cost = duration_ms × memory_GB × $0.00001667 / 1000
    """
    memory_gb = memory_mb / 1024.0
    cost = duration_ms * memory_gb * ExperimentConfig.COST_PER_GB_MS
    return cost


def compute_total_cost(df) -> float:
    """
    计算整个实验的总成本
    df 必须包含：duration_ms 和 memory_mb 列
    """
    if df.empty:
        return 0.0
    total = 0.0
    for _, row in df.iterrows():
        total += compute_cost(
            duration_ms=row.get('duration_ms', row.get('e2e_latency', 0)),
            memory_mb=row.get('memory_mb', 128)  # 默认 128MB
        )
    return total


# SLO 判定函数
def check_slo_violation(latency_ms: float, qos_class: str, base_p90: float = None) -> bool:
    """
    判定是否违反 SLO
    SLO = base_p90_srv × 1.2 (对于 Q1)
    对于 Q2/Q3 使用相对宽松的阈值
    """
    if base_p90 is None:
        base_p90 = ExperimentConfig.BASE_SLO_MS

    slo_map = {
        'Q1': base_p90 * 1.0,      # Critical: 1000ms
        'Q2': base_p90 * 1.8,      # High: 1800ms
        'Q3': base_p90 * 3.0,      # Standard: 3000ms
    }
    slo_bound = slo_map.get(qos_class, base_p90 * 2.0)
    return latency_ms > slo_bound
