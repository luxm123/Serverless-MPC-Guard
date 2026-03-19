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
        v47: 稳定最终版。
        根据截图结果，我们的梯度逻辑已成功击败基准线。
        现在进一步平滑决策，减少容器冲突。
        """
        state = kwargs.get('state', {})
        params = state.get('params', {}) 
        
        # 1. 梯度计算
        # 1.1. 资源浪费梯度 (向下推力) - 降低推力，防止剧烈下降
        w1 = 0.3 
        grad_waste = w1 

        # 1.2. SLO风险梯度 (向上阻力) - 保持强力
        w2 = 3.0 
        safe_line = 155.0
        grad_risk = 0.0
        if pred_upper > safe_line:
            risk_factor = (pred_upper - safe_line) / (slo_limit - safe_line)
            grad_risk = -w2 * math.exp(risk_factor * 2)

        # 1.3. 动态权重 (真实 P90)
        actual_p90 = float(state.get('p90_belief', 100.0))
        dynamic_risk_weight = 1.0
        if actual_p90 > 170.0:
            dynamic_risk_weight = 4.0

        # 1.4. 梯度惯性
        grad_track = float(state.get('grad_track', 0.0))

        # 4. 真实延迟恐慌 (终极安全网)
        grad_panic = 0.0
        panic_line = 175.0 
        if actual_p90 > panic_line:
            grad_panic = -15.0 * (actual_p90 - panic_line)

        # 5. 终极合力计算
        grad = grad_waste + dynamic_risk_weight * grad_risk + grad_panic + 0.2 * grad_track

        # 6. 更新决策
        lr = 0.005 # 降低步长，防止震荡
        
        # 安全机制：一旦真实延迟超过 170ms，严禁任何形式的资源削减
        if actual_p90 > 170.0 and grad > 0:
            grad = -1.0 

        new_alloc = prev_u - lr * grad
        
        # 限制下降速度到 0.01 (每 10 个请求才降 0.1)
        max_decrease = 0.01
        if new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - max_decrease)

        # 7. 边界
        lower = 0.60
        upper = 1.0
        final_alloc = max(lower, min(upper, new_alloc))

        # 状态持久化
        state['grad_track'] = grad
        state['opt_debug'] = {
            "grad_total": grad,
            "actual_p90": actual_p90,
            "final_alloc": final_alloc
        }

        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    opt = Optimizer()
    return opt.optimize_u(state.get('prev_alloc', 1.0), pred_upper, slo_limit, 0.0, state=state), {}, {}
