import math

class Optimizer:
    def __init__(self):
        self.w1 = 1.0
        self.w2 = 0.5
        self.w3 = 5.0

    def update_weights(self, metrics, system_state):
        return {'w1': self.w1, 'w2': self.w2, 'w3': self.w3}

    def tune_params_by_price(self, price, eta_base, gamma_base, bands, system_state):
        return eta_base, gamma_base

    def optimize_u(self, prev_u, pred_upper, slo_limit, price, **kwargs):
        """
        v46.3: 强力反馈版。
        解决 v46.2 面对高负载任务时反应迟钝、大面积违规的问题。
        """
        state = kwargs.get('state', {})
        params = state.get('params', {}) 
        
        # 1. 梯度计算
        # 1.1. 资源浪费梯度 (Utility/Waste Gradient) - 适中的向下推力
        w1 = 0.5 
        grad_waste = w1 

        # 1.2. SLO风险梯度 (Risk Gradient) - 极强的向上阻力
        # 只要预测延迟超过 155ms (86% SLO)，就开始产生显著阻力
        w2 = 3.0 
        safe_line = 155.0
        grad_risk = 0.0
        if pred_upper > safe_line:
            # 指数增长，在 180ms 附近会产生约 -22 的梯度，远超 0.5 的 grad_waste
            risk_factor = (pred_upper - safe_line) / (slo_limit - safe_line)
            grad_risk = -w2 * math.exp(risk_factor * 2)

        # 1.3. 动态权重 (基于真实 P90)
        actual_p90 = float(state.get('p90_belief', 100.0))
        dynamic_risk_weight = 1.0
        if actual_p90 > 170.0:
            # 真实延迟一旦逼近 180ms，风险梯度权重暴增 5 倍
            dynamic_risk_weight = 5.0

        # 1.4. 梯度追踪 (历史惯性)
        grad_track = float(state.get('grad_track', 0.0))

        # 4. 真实延迟恐慌 (Dynamic Reality Check) - 终极安全网
        grad_panic = 0.0
        panic_line = 175.0 
        if actual_p90 > panic_line:
            # 只要真实延迟超过 175ms，每多 1ms 就产生 -10 的梯度
            grad_panic = -10.0 * (actual_p90 - panic_line)

        # 5. 终极合力计算
        grad = grad_waste + dynamic_risk_weight * grad_risk + grad_panic + 0.3 * grad_track

        # 6. 更新决策
        lr = 0.01 
        
        # 特殊逻辑：如果当前已经处于高风险（actual_p90 > 175），且计算出的 grad 竟然还是正的（想降资源）
        # 强制将 grad 设为负值（改为增资源）
        if actual_p90 > 175.0 and grad > 0:
            grad = -2.0 

        new_alloc = prev_u - lr * grad
        
        # 限制下降速度，但不限制上升速度
        max_decrease = 0.02 # 进一步收紧下降速度，保命要紧
        if new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - max_decrease)

        # 7. 边界
        lower = 0.60
        upper = 1.0
        final_alloc = max(lower, min(upper, new_alloc))

        # 更新状态供下一轮使用
        state['grad_track'] = grad
        state['opt_debug'] = {
            "grad_waste": grad_waste,
            "grad_risk": grad_risk,
            "grad_panic": grad_panic,
            "grad_total": grad,
            "actual_p90": actual_p90,
            "final_alloc": final_alloc
        }

        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    opt = Optimizer()
    return opt.optimize_u(state.get('prev_alloc', 1.0), pred_upper, slo_limit, 0.0, state=state), {}, {}
