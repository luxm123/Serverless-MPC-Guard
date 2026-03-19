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
        v46.2: 修正梯度方向逻辑。
        grad_waste 应该是向下的推力（正梯度），grad_risk 是向上的推力（负梯度）。
        """
        state = kwargs.get('state', {})
        params = state.get('params', {}) 
        
        # 1. 梯度计算
        # 1.1. 资源浪费梯度 (Utility/Waste Gradient) - 恒定向下的推力
        # 这个值越大，系统越倾向于省钱
        w1 = 0.5 
        grad_waste = w1 

        # 1.2. SLO风险梯度 (Risk Gradient) - 向上推力
        # 只有预测延迟超过 160ms 才开始产生阻力
        w2 = 0.8
        safe_line = 160.0
        grad_risk = 0.0
        if pred_upper > safe_line:
            risk_factor = (pred_upper - safe_line) / (slo_limit - safe_line)
            grad_risk = -w2 * math.exp(risk_factor)

        # 1.3. 动态权重
        actual_p90 = float(state.get('p90_belief', 100.0))
        dynamic_risk_weight = 1.0
        if actual_p90 > 170.0:
            dynamic_risk_weight = 2.0 # 延迟一旦过高，风险梯度权重翻倍

        # 1.4. 梯度追踪 (历史惯性)
        grad_track = float(state.get('grad_track', 0.0))

        # 4. 真实延迟恐慌 (Dynamic Reality Check)
        grad_panic = 0.0
        panic_line = 175.0 # 极度接近违规
        if actual_p90 > panic_line:
            panic_excess = actual_p90 - panic_line
            grad_panic = -10.0 * panic_excess # 强力向上拉回

        # 5. 终极合力计算
        # grad = 浪费(下) + 风险(上) + 恐慌(上) + 惯性
        grad = grad_waste + dynamic_risk_weight * grad_risk + grad_panic + 0.5 * grad_track

        # 6. 更新决策
        lr = 0.01 # 稍微加大步长，让它动起来
        new_alloc = prev_u - lr * grad
        
        # 限制下降速度，但不限制上升速度
        max_decrease = 0.05
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
            "final_alloc": final_alloc
        }

        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    opt = Optimizer()
    return opt.optimize_u(state.get('prev_alloc', 1.0), pred_upper, slo_limit, 0.0, state=state), {}, {}
