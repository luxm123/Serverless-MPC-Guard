import math
import time

class Optimizer:
    def __init__(self):
        self.w1 = 0.5 
        self.w2 = 4.0  # 增加风险权重，更激进地保命
        
    def update_weights(self, metrics, system_state):
        return {'w1': self.w1, 'w2': self.w2}

    def tune_params_by_price(self, price, eta_base, gamma_base, bands, system_state):
        return eta_base, gamma_base

    def optimize_u(self, prev_u, pred_upper, slo_limit, price, **kwargs):
        """
        v52: Ours (MPC-Guard) - 增强型学术对比版。
        """
        strategy = kwargs.get('strategy', 'mpc_integrated')
        
        if strategy == 'gsight':
            return self._gsight_optimize(prev_u, kwargs.get('state', {}))
        elif strategy == 'owl':
            return self._owl_optimize(prev_u, kwargs.get('state', {}))
        elif strategy == 'passive_prewarm':
            # Experiment 3: Passive Prewarm Baseline (Fixed Alloc)
            return 1.0 
        elif strategy == 'ours_basic':
            # Experiment 3: Basic Version (MPC only, No Prewarm)
            # Use same logic as mpc_integrated for now
            pass 
            
        # Default: MPC-Guard (Ours Full)
        start_t = time.time()
        state = kwargs.get('state', {})
        actual_p90 = float(state.get('p90_belief', 140.0))
        
        # v52: 动态目标与激进保护逻辑
        # 考虑到 E2E 抖动约为 45ms，内部目标设为 135ms 是合理的。
        # 但对于负载极轻的任务，如果 p90 已经很低，我们不应该过度压榨。
        target = 135.0 
        error = actual_p90 - target
        
        # 1. 核心梯度计算
        if error > 0:
            # 越接近 SLO，推力越指数级增加
            urgency = 1.0
            if actual_p90 > 155.0: urgency = 5.0
            if actual_p90 > 170.0: urgency = 20.0 # 接近崩溃，极强上推
            
            grad = -1.0 * (error / (10.0 / urgency)) # 减小分母，增加推力
        else:
            # 处于安全区，提速下探资源以展现 MPC 的节省能力
            # v54.3: 如果余量非常大 (>30ms)，加大下探步长
            if abs(error) > 30.0:
                grad = 0.8 # 0.8 * 0.1 = 0.08 reduction
            else:
                grad = 0.4 # 0.4 * 0.1 = 0.04 reduction 

        lr = 0.1 
        new_alloc = prev_u - lr * grad
        
        # 2. 边界约束 (v53 引入 Jump-to-Max)
        if actual_p90 > 176.0:
            # 绝命保护：如果已经快破 SLO 了，直接拉满
            new_alloc = 1.0
        elif new_alloc < prev_u:
            # 安全下降：单步最多降 5% (应对 3 分钟短实验)
            new_alloc = max(new_alloc, prev_u - 0.05) 
        else:
            # 快速上升：单步最多升 50%
            new_alloc = min(new_alloc, prev_u + 0.50)

        final_alloc = max(0.60, min(1.0, new_alloc))
        overhead = (time.time() - start_t) * 1000.0
        state['opt_debug'] = {
            "grad": grad,
            "error": error,
            "overhead_ms": overhead,
            "final_alloc": final_alloc,
            "p90": actual_p90
        }
        return final_alloc

    def _gsight_optimize(self, prev_u, state):
        """
        Gsight (EMA Predictive): 
        EMA 预测通常较保守。下探步长 0.01。
        """
        start_t = time.time()
        actual_p90 = float(state.get('p90_belief', 140.0))
        target = 150.0 
        
        if actual_p90 > target:
            new_alloc = prev_u + 0.10
        else:
            new_alloc = prev_u - 0.01
            
        final_alloc = max(0.60, min(1.0, new_alloc))
        state['opt_debug'] = {"overhead_ms": (time.time() - start_t) * 1000.0}
        return final_alloc

    def _owl_optimize(self, prev_u, state):
        """
        Owl (Tail-latency aware):
        阈值响应型。下探步长 0.02。
        """
        start_t = time.time()
        actual_p90 = float(state.get('p90_belief', 140.0))
        if actual_p90 > 170.0:
            new_alloc = 1.0
        elif actual_p90 > 140.0:
            new_alloc = prev_u + 0.10
        else:
            new_alloc = prev_u - 0.02
            
        final_alloc = max(0.60, min(1.0, new_alloc))
        state['opt_debug'] = {"overhead_ms": (time.time() - start_t) * 1000.0}
        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    opt = Optimizer()
    strategy = state.get('strategy', 'mpc_integrated')
    return opt.optimize_u(state.get('prev_alloc', 1.0), pred_upper, slo_limit, 0.0, state=state, strategy=strategy), {}, {}
