"""
Oracle 控制器（离线最优）
已知未来 30 分钟完整负载序列，计算全局最优资源分配
作为理论上限，评估其他策略的接近程度

优化问题：
min Σ_t [cost(u_t) + penalty * violation_t]
s.t.  u_min ≤ u_t ≤ u_max
     latency_t = base_latency / u_t ≤ SLO_t (∀t for Q1/Q2)
"""
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class OracleController:
    """
    离线最优资源分配器
    假设已知未来所有信息，求解松弛优化问题
    """
    def __init__(self, config):
        self.config = config
        self.u_min = 0.4
        self.u_max = 4.0
        self.base_slo = 1000.0  # ms
        self.penalty = 1000.0   # 违约惩罚（高）

    def solve_optimal_allocation(self,
                                  future_load: List[float],
                                  base_service_times: List[float],
                                  slo_limits: List[float],
                                  current_alloc: float = 1.0) -> np.ndarray:
        """
        求解最优资源分配序列（动态规划近似）

        Args:
            future_load: 未来N步的负载（队列深度或并发）
            base_service_times: 基准服务时间（满分配下）
            slo_limits: 每步的 SLO 阈值
            current_alloc: 当前分配

        Returns:
            optimal_allocs: 最优分配序列
        """
        N = len(future_load)
        if N == 0:
            return np.array([])

        # 状态空间离散化（0.4 到 4.0，步长 0.05）
        u_grid = np.arange(self.u_min, self.u_max + 0.05, 0.05)

        # DP 表：dp[t, i] = 从t步开始、分配为u_grid[i]时的最小未来成本
        dp = np.full((N + 1, len(u_grid)), np.inf)
        opt_alloc = np.zeros((N, len(u_grid)))

        # 边界条件：最后一步
        for i, u in enumerate(u_grid):
            dp[N, i] = 0.0  # 无未来成本

        # 反向递推
        for t in range(N - 1, -1, -1):
            backlog = future_load[t]
            base_lat = base_service_times[t]
            slo = slo_limits[t]

            for i, u in enumerate(u_grid):
                # 预估延迟：base_lat / u + queue_delay
                # 队列延迟简化模型：backlog / u
                service_time = base_lat / max(u, 0.1)
                queue_delay = (max(0, backlog - u) / u) * base_lat if backlog > u else 0
                est_latency = service_time + queue_delay

                # 成本：资源成本 + 违约惩罚
                resource_cost = u  # 线性成本（相对）
                violation_penalty = self.penalty if est_latency > slo else 0.0

                step_cost = resource_cost + violation_penalty

                # 寻找下一步最优
                future_costs = dp[t + 1, :]
                best_next = np.argmin(future_costs)
                dp[t, i] = step_cost + future_costs[best_next]
                opt_alloc[t, i] = u_grid[best_next]

        # 前向提取最优路径
        optimal_allocs = np.zeros(N)
        current_idx = np.argmin(dp[0, :])
        optimal_allocs[0] = u_grid[current_idx]

        for t in range(1, N):
            next_idx = np.argmin([dp[t, i] for i in range(len(u_grid))])
            optimal_allocs[t] = u_grid[next_idx]

        # 平滑约束：避免频繁震荡
        for t in range(1, N):
            if abs(optimal_allocs[t] - optimal_allocs[t - 1]) > 0.5:
                # 限制变化幅度
                optimal_allocs[t] = optimal_allocs[t - 1] + 0.5 * np.sign(optimal_allocs[t] - optimal_allocs[t - 1])

        return np.clip(optimal_allocs, self.u_min, self.u_max)

    def decide(self, context: Dict) -> Dict:
        """
        决策接口（与 MPC 兼容）
        在真实环境中，这是离线运行后返回预计算的分配序列

        Args:
            context: 包含未来负载序列和系统状态

        Returns:
            decision: 资源分配决策
        """
        # 提取未来序列（实际是从上下文中获取预计算的 oracle 分配）
        if 'oracle_allocs' in context:
            # 已预计算好的序列
            return {
                'resource_alloc': float(context['oracle_allocs'][0]),
                'should_shed': False,
                'oracle_mode': 'precomputed'
            }

        # 否则现场求解（仅用于测试）
        future_load = context.get('future_load', [1.0] * 10)
        base_times = context.get('base_service_times', [100.0] * len(future_load))
        slo_limits = context.get('slo_limits', [1000.0] * len(future_load))

        optimal = self.solve_optimal_allocation(future_load, base_times, slo_limits)

        return {
            'resource_alloc': float(optimal[0]),
            'should_shed': False,
            'oracle_mode': 'online',
            'future_allocs': optimal.tolist()
        }


def generate_oracle_allocations(trace_df: pd.DataFrame,
                                horizon: int = 30,
                                strategy: str = 'conservative') -> np.ndarray:
    """
    为给定 trace 生成 oracle 分配序列

    Args:
        trace_df: 包含 timestamp, duration, priority 的 trace
        horizon: 预测窗口（步数）
        strategy: 'conservative' (保守) 或 'aggressive' (激进)

    Returns:
        allocations: 每步的最优分配
    """
    oracle = OracleController(None)

    # 提取未来负载（使用滑动窗口统计）
    # 简化为：未来 N 步的预计并发数
    # 实际应从 trace 的 timestamp 推断

    # 生成测试序列
    N = min(horizon, len(trace_df))
    future_load = np.ones(N) * 5.0  # 默认并发
    base_times = np.ones(N) * 100.0  # 默认服务时间
    slo_limits = np.ones(N) * 1000.0

    # 根据 trace 调整
    if 'priority' in trace_df.columns:
        for i, row in trace_df.head(N).iterrows():
            if row['priority'] == 'critical':
                slo_limits[i] = 1000.0
            elif row['priority'] == 'high':
                slo_limits[i] = 1800.0
            else:
                slo_limits[i] = 3000.0

    allocs = oracle.solve_optimal_allocation(future_load, base_times, slo_limits)
    return allocs


if __name__ == "__main__":
    # 测试 Oracle
    oracle = OracleController(None)

    # 模拟场景：10 步，负载逐渐上升
    load = np.array([2, 3, 5, 8, 12, 15, 18, 20, 18, 15])
    base_lat = np.ones(10) * 100.0
    slo = np.ones(10) * 1000.0

    optimal = oracle.solve_optimal_allocation(load, base_lat, slo)

    print("Oracle Optimal Allocation Test")
    print("="*50)
    print(f"{'Step':<6} {'Load':<6} {'BaseLat':<8} {'SLO':<8} {'OptAlloc':<8}")
    print("-"*50)
    for i in range(10):
        print(f"{i:<6} {load[i]:<6.1f} {base_lat[i]:<8.1f} {slo[i]:<8.1f} {optimal[i]:<8.3f}")

    print(f"\nAvg Alloc: {optimal.mean():.3f}")
    print(f"Min Alloc: {optimal.min():.3f}")
    print(f"Max Alloc: {optimal.max():.3f}")
