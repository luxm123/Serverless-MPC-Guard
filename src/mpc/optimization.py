import math
import time

class Optimizer:
    def __init__(self):
        self.w1 = 0.5  # 浪费权重
        self.w2 = 2.0  # 风险权重
        
    def optimize_u(self, prev_u, pred_upper, slo_limit, price, **kwargs):
        """
        v48_Final_Fix: 回归极简控制逻辑。
        不再使用复杂的梯度组合，直接根据延迟误差进行线性调节。
        """
        start_t = time.time()
        state = kwargs.get('state', {})
        
        # 核心逻辑：获取真实延迟
        actual_p90 = float(state.get('p90_belief', 160.0))
        
        # 计算误差：我们希望延迟维持在 160ms 左右
        target = 160.0
        error = actual_p90 - target
        
        # 1. 计算调整方向
        if error > 0:
            # 延迟太高，需要增加资源（负梯度）
            # 误差越大，拉回的力量越强
            grad = -0.5 * (error / 20.0) 
        else:
            # 延迟较低，可以尝试省钱（正梯度）
            # 省钱要慢，所以推力很小
            grad = 0.1 

        # 2. 执行更新
        lr = 0.05 
        new_alloc = prev_u - lr * grad
        
        # 3. 严格限制下降速度，允许快速上升
        if new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - 0.01) # 每次最多降 0.01
        else:
            new_alloc = min(new_alloc, prev_u + 0.1)  # 允许较快上升保命

        # 4. 边界保护
        final_alloc = max(0.60, min(1.0, new_alloc))

        # 5. 修复调试信息：确保不再是 0
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
