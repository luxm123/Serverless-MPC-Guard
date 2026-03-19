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
        v46 Logic: Dynamic Reality Check + Gradient Descent
        """
        state = kwargs.get('state', {})
        params = state.get('params', {}) # Actually system_state is passed as state
        ref_latency = kwargs.get('ref_latency', slo_limit * 0.8)
        
        # 1. 梯度计算
        # 1.1. 资源浪费梯度 (Utility/Waste Gradient)
        w1 = float(params.get('w1', 0.5))
        grad_waste = w1 * (pred_upper - ref_latency)

        # 1.2. SLO风险梯度 (Risk Gradient)
        w2 = float(params.get('w2', 0.2)) 
        safe_margin = float(params.get('safe_margin', 0.60)) 
        warning_line = slo_limit * safe_margin
        
        grad_risk = 0.0
        if pred_upper > warning_line:
            risk_factor = (pred_upper - warning_line) / (slo_limit - warning_line)
            grad_risk = -w2 * math.exp(risk_factor)

        # 1.3. 动态风险权重 (Dynamic Risk Weight)
        p90_ema = float(state.get('p90_ema', 0.0))
        dynamic_risk_weight = 1.0
        if p90_ema > ref_latency:
            dynamic_risk_weight = 1.0 + (p90_ema - ref_latency) / ref_latency

        # 1.4. 梯度追踪 (Gradient Tracking)
        grad_track = float(state.get('grad_track', 0.0))
        if pred_upper <= ref_latency:
            grad_track = 0.0 # Only allow upward push from history if prediction is risky

        # 4. 真实延迟兜底 (Dynamic Reality Check) - v46
        actual_p90 = float(state.get('p90_belief', 0.0))
        grad_panic = 0.0
        panic_margin = slo_limit * 0.90 # 162ms
        if actual_p90 > panic_margin:
            panic_excess = (actual_p90 - panic_margin) / max(1.0, slo_limit)
            grad_panic = -50.0 * (panic_excess ** 2) - 5.0 * panic_excess

        # 5. 终极合力计算
        grad = 2.0 * grad_track + dynamic_risk_weight * grad_risk + grad_waste + grad_panic

        # 6. 更新梯度追踪和分配
        max_decrease = float(params.get('max_decrease', 0.02))
        lr = float(kwargs.get('eta', 0.01))
        
        # 更新梯度追踪 EMA (Stored in state for next round)
        beta = float(params.get('beta', 0.5))
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

        # Prepare debug info for the middleware
        state['opt_debug'] = {
            "grad_waste": grad_waste,
            "grad_risk": grad_risk,
            "grad_track": grad_track,
            "grad_panic": grad_panic,
            "grad_total": grad,
            "actual_p90": actual_p90,
            "pred_upper": pred_upper,
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
