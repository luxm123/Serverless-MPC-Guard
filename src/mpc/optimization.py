import math
import time

class Optimizer:
    def __init__(self):
        self.w1 = 0.5  # 浪费权重
        self.w2 = 2.0  # 风险权重
        
    def update_weights(self, metrics, system_state):
        return {'w1': self.w1, 'w2': self.w2}

    def tune_params_by_price(self, price, eta_base, gamma_base, bands, system_state):
        return eta_base, gamma_base

    def optimize_u(self, prev_u, pred_upper, slo_limit, price, **kwargs):
        """
        v49_No_Priority: 回归极简控制逻辑，彻底移除优先级。
        直接根据延迟误差进行调节。
        """
        start_t = time.time()
        state = kwargs.get('state', {})
        
        # 核心逻辑：获取真实延迟
        actual_p90 = float(state.get('p90_belief', 160.0))
        
        # 计算误差：目标 160ms
        target = 160.0
        error = actual_p90 - target
        
        # 1. 计算调整方向
        if error > 0:
            # 延迟太高，增加资源
            grad = -0.5 * (error / 20.0) 
        else:
            # 延迟较低，缓慢省钱
            grad = 0.1 

        # 2. 执行更新
        lr = 0.05 
        new_alloc = prev_u - lr * grad
        
        # 3. 严格限制下降速度，允许快速上升
        if new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - 0.01) # 限制下降步长
        else:
            new_alloc = min(new_alloc, prev_u + 0.1)  # 快速回弹保命

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
