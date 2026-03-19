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
        v50: 全局余量版。
        考虑到网络/调度会吃掉约 40ms，我们将内部目标下调到 135ms。
        """
        start_t = time.time()
        state = kwargs.get('state', {})
        
        # 核心：获取真实延迟
        actual_p90 = float(state.get('p90_belief', 140.0))
        
        # 考虑到 E2E 180ms 的限制，Server 必须跑在 135ms 才能稳赢
        target = 135.0 
        error = actual_p90 - target
        
        # 1. 计算调整方向
        if error > 0:
            # 只要超过 135ms，产生强力向上梯度
            # 相比 v49，推力翻倍
            grad = -1.0 * (error / 15.0) 
        else:
            # 只有低于 135ms，才允许非常缓慢地省钱
            grad = 0.05 

        # 2. 执行更新
        lr = 0.08 # 提高学习率，反应更快
        new_alloc = prev_u - lr * grad
        
        # 3. 严格限制下降，允许快速上升
        if new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - 0.01) 
        else:
            new_alloc = min(new_alloc, prev_u + 0.15) # 允许更快的上升

        # 4. 边界保护
        final_alloc = max(0.60, min(1.0, new_alloc))

        # 5. 调试信息
        overhead = (time.time() - start_t) * 1000.0
        state['opt_debug'] = {
            "grad": grad,
            "error": error,
            "overhead_ms": overhead,
            "final_alloc": final_alloc
        }

        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    opt = Optimizer()
    return opt.optimize_u(state.get('prev_alloc', 1.0), pred_upper, slo_limit, 0.0, state=state), {}, {}
