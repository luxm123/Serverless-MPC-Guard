import math

class Optimizer:
    def __init__(self):
        self.w1 = 1.0
        self.w2 = 0.5
        self.w3 = 5.0
        self.stable_count = 0
        self.last_grad = 0.0
        self.int_waste = 0.0

    def update_weights(self, metrics, system_state):
        # Placeholder for dynamic weight adjustment if needed
        return {
            'w1': self.w1,
            'w2': self.w2,
            'w3': self.w3
        }

    def tune_params_by_price(self, price, eta_base, gamma_base, bands, system_state):
        # Simplified tuning based on price
        return eta_base, gamma_base

    def optimize_u(self, prev_u, pred_upper, slo_limit, price, **kwargs):
        """
        v46.1: 优化预警阈值，减少过度反应和震荡。
        """
        state = kwargs.get('state', {})
        params = state.get('params', {}) 
        # 目标延迟设为 145ms，给系统留出下探空间
        ref_latency = kwargs.get('ref_latency', 145.0) 
        
        # 1. 梯度计算
        # 1.1. 资源浪费梯度 (Utility/Waste Gradient)
        # 稍微调低权重，避免剧烈下降
        w1 = float(params.get('w1', 0.4)) 
        grad_waste = w1 * (pred_upper - ref_latency)

        # 1.2. SLO风险梯度 (Risk Gradient)
        # 将安全线提高到 165ms (约 92% SLO)，只有真危险才开始推
        w2 = float(params.get('w2', 0.3)) 
        safe_margin = 165.0 
        
        grad_risk = 0.0
        if pred_upper > safe_margin:
            # 使用指数增长，但由于 safe_margin 提高了，只有在接近 180ms 时才会有爆发力
            risk_factor = (pred_upper - safe_margin) / (slo_limit - safe_margin)
            grad_risk = -w2 * math.exp(risk_factor)

        # 1.3. 动态风险权重 (Dynamic Risk Weight)
        p90_ema = float(state.get('p90_belief', 0.0))
        dynamic_risk_weight = 1.0
        if p90_ema > ref_latency:
            dynamic_risk_weight = 1.0 + (p90_ema - ref_latency) / ref_latency

        # 1.4. 梯度追踪 (Gradient Tracking)
        grad_track = float(state.get('grad_track', 0.0))
        # 减少追踪的惯性，防止被历史的一次尖峰长时间误导
        if pred_upper <= safe_margin:
            grad_track *= 0.5 

        # 4. 真实延迟兜底 (Dynamic Reality Check) - v46.1
        actual_p90 = float(state.get('p90_belief', 0.0))
        grad_panic = 0.0
        # 恐慌线设在 175ms，非常接近 SLO
        panic_margin = 175.0 
        if actual_p90 > panic_margin:
            panic_excess = (actual_p90 - panic_margin) / 5.0 # 5ms 差距就产生巨大推力
            grad_panic = -20.0 * (panic_excess ** 2) - 10.0 * panic_excess

        # 5. 终极合力计算
        grad = 1.5 * grad_track + dynamic_risk_weight * grad_risk + grad_waste + grad_panic

        # 6. 更新梯度追踪和分配
        # 稍微放宽下降速度到 0.03，提高响应灵活性
        max_decrease = 0.03
        lr = float(kwargs.get('eta', 0.005)) # 减小学习率，防止从 0.7 瞬移到 1.0

        # 更新梯度追踪 (EMA 方式更新)
        beta = 0.7 
        new_grad_track = beta * grad_track + (1 - beta) * grad
        state['grad_track'] = new_grad_track
        
        # 计算新的分配值
        new_alloc = prev_u - lr * grad
        
        # 施加下降速度限制
        if new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - max_decrease)

        # 7. 应用边界和最终决策
        lower = 0.60
        upper = 1.0
        final_alloc = max(lower, min(upper, new_alloc))

        # 调试信息更新
        state['opt_debug'] = {
            "grad_waste": grad_waste,
            "grad_risk": grad_risk,
            "grad_track": grad_track,
            "grad_panic": grad_panic,
            "actual_p90": actual_p90,
            "final_alloc": final_alloc
        }

        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    """
    Legacy function wrapper for backward compatibility if needed.
    """
    opt = Optimizer()
    return opt.optimize_u(
        state.get('prev_alloc', 1.0),
        pred_upper,
        slo_limit,
        0.0,
        ref_latency=ref_latency,
        state=state
    ), {}, {}
