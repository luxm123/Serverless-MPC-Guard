import math
import random

class Optimizer:
    def __init__(self):
        # Default weights (used if not in state)
        self.default_w1 = 1.0
        self.default_w2 = 5.0 # Boosted from 0.5 to fight Shadow Price
        self.default_w3 = 5.0

    def calculate_cost(self, y_ac, y_ref, metrics, state=None):
        w1 = self.default_w1
        w2 = self.default_w2
        w3 = self.default_w3
        
        if state:
            w1 = float(state.get('opt_w1', w1))
            w2 = float(state.get('opt_w2', w2))
            w3 = float(state.get('opt_w3', w3))

        # --- MPC Priority Boosting ---
        # Instead of hard-coding the output, we tune the objective function weights.
        # For Q1 (Mission Critical), the cost of Violation (w3) and Tracking Error (w1)
        # must significantly outweigh the Shadow Price (Cost).
        if qos_class == 'Q1':
            # Remove artificial boost for Q1 to allow Shadow Price to work
            is_fidelity_mode = True
        elif qos_class == 'Q2':
            w1 *= 2.0
            w3 *= 2.0
        # -----------------------------

        error = abs(y_ac - y_ref)
        waste = metrics.get('resource_waste', 0.0)
        is_viol = 1.0 if y_ac > y_ref else 0.0
        
        J = w1 * error + w2 * waste + w3 * is_viol
        return J

    def update_weights(self, metrics, state):
        """
        Adaptive Weight Tuning using PI (Proportional-Integral) Logic.
        Persists state to the 'state' dictionary.
        """
        # Load state
        w1 = float(state.get('opt_w1', self.default_w1))
        w2 = float(state.get('opt_w2', self.default_w2))
        w3 = float(state.get('opt_w3', self.default_w3))
        int_viol_err = float(state.get('opt_int_viol_err', 0.0))
        int_waste_err = float(state.get('opt_int_waste_err', 0.0))
        stable_count = int(state.get('opt_stable_count', 0))
        tense_count = int(state.get('opt_tense_count', 0))

        viol_rate = metrics.get('slo_violation_rate', 0.0)
        waste_rate = metrics.get('resource_waste_rate', 0.0)
        
        # 1. Adapt w3 (SLO Penalty)
        w3_step_inc = float(state.get('opt_w3_step_inc', 20.0))
        w3_step_dec = float(state.get('opt_w3_step_dec', 0.95))
        w3_min = float(state.get('opt_w3_min', 5.0))
        w3_max = float(state.get('opt_w3_max', 50.0)) # Reduced from 500.0 to prevent panic
        
        # 2. Adapt w2 (Waste Penalty) - Define params early
        w2_step_inc = float(state.get('opt_w2_step_inc', 10.0))
        w2_step_dec = float(state.get('opt_w2_step_dec', 0.98))
        w2_min = float(state.get('opt_w2_min', 0.5))
        w2_max = float(state.get('opt_w2_max', 50.0))
        
        # CRITICAL FIX: Tolerance alignment
        # Ignore violations < 1% to prevent w3 from exploding on noise
        if viol_rate > 0.01:
            w3 = w3_min + (viol_rate * w3_step_inc) + (int_viol_err * 10.0)
            # CRITICAL FIX: Clamp w3 to prevent Death Spiral
            w3 = min(w3, w3_max)
        else:
            int_viol_err *= 0.9
            w3 = max(w3_min, w3 * w3_step_dec)

        waste_stable_thr = float(state.get('opt_waste_stable_thr', 0.05))
        waste_tense_thr = float(state.get('opt_waste_tense_thr', 0.2))
        viol_tense_thr = float(state.get('opt_viol_tense_thr', 0.1))
        if viol_rate == 0.0 and waste_rate < waste_stable_thr:
            stable_count += 1
            tense_count = 0
        elif viol_rate > viol_tense_thr or waste_rate > waste_tense_thr:
            tense_count += 1
            stable_count = 0
        else:
            stable_count = max(0, stable_count - 1)
            tense_count = max(0, tense_count - 1)

        stable_thr = int(state.get('opt_stable_thr', 3))
        if stable_count >= stable_thr:
            int_viol_err = 0.0
            int_waste_err *= 0.5
            w3_relax_mul = float(state.get('opt_w3_relax_mul', 0.7))
            w2_relax_mul = float(state.get('opt_w2_relax_mul', 0.9))
            w3 = max(w3_min, w3 * w3_relax_mul)
            w2 = max(w2_min, w2 * w2_relax_mul)
            
        # Adapt w2 (Waste Penalty) logic
        waste_base = float(state.get('opt_waste_err_base', 0.1))
        waste_cap = float(state.get('opt_waste_err_cap', 0.5))
        int_cap = float(state.get('opt_int_waste_cap', 5.0))
        int_decay = float(state.get('opt_int_decay', 0.9))
        viol_int_gain = float(state.get('opt_viol_int_gain', 10.0))
        waste_err = max(0.0, waste_rate - waste_base)
        if waste_err <= waste_cap:
            int_waste_err += waste_err
        int_waste_err = min(int_cap, max(0.0, int_waste_err))
        
        if waste_err > 0.0:
            w2_int_gain = float(state.get('opt_w2_int_gain', 2.0))
            w2 = w2_min + (waste_err * w2_step_inc) + (int_waste_err * w2_int_gain)
        else:
            int_waste_err *= int_decay
            w2 = max(w2_min, w2 * w2_step_dec)
            
        # Limit weights
        w3 = min(w3, w3_max)
        w2 = min(w2, w2_max)
        
        # Save state
        state['opt_w1'] = w1
        state['opt_w2'] = w2
        state['opt_w3'] = w3
        state['opt_int_viol_err'] = int_viol_err
        state['opt_int_waste_err'] = int_waste_err
        state['opt_stable_count'] = stable_count
        state['opt_tense_count'] = tense_count
        
        return {'w1': w1, 'w2': w2, 'w3': w3}

    def allocate(self, prev_u, price, eta=0.05, gamma=0.1, lower=0.0, upper=1.0):
        grad = price
        u_new = prev_u - eta * (grad + gamma * prev_u)
        if u_new < lower:
            u_new = lower
        if u_new > upper:
            u_new = upper
        return u_new

    def optimize_u(self, prev_u, pred_upper, slo_limit, price, eta=0.05, gamma=0.1, risk_comp=None, ku=None, risks=None, tau=1.0, ref_latency=None, state=None, priority=0.5, qos_class=None, current_backlog=None):
        """
        Gradient Descent Step for u (Admission Probability / Resource Alloc).
        """
        # Load adaptive weights
        w1 = self.default_w1
        w2 = self.default_w2
        w3 = self.default_w3
        if state:
            w1 = float(state.get('opt_w1', w1))
            w2 = float(state.get('opt_w2', w2))
            w3 = float(state.get('opt_w3', w3))
        
        # CRITICAL FIX: Force Clamp w3 to prevent state poisoning (e.g. w3=10000 from prev run)
        w3 = min(w3, 50.0)

        # --- MPC Priority Boosting (Explicit QoS) ---
        # All requests are executed fully. u represents resource allocation.
        if qos_class == 'Q1':
            w1 *= 1.5
            w3 *= 1.5
        elif qos_class == 'Q2':
            w1 *= 1.2
            w3 *= 1.2
        # --------------------------------------------

        # 1. Risk Gradient (Risk of violating SLO)
        # Formula: SLO_safe = SLO_target - C_{1-alpha} (Uncertainty Margin)
        # Constraint: y_pred <= SLO_safe  <==>  y_pred + C_{1-alpha} <= SLO_target
        # pred_upper represents (y_pred + C_{1-alpha}) i.e., the (1-alpha) quantile prediction.
        
        # We quantify violation as: max(0, (pred_upper - slo_limit) / slo_limit)
        base_risk = max(0.0, (pred_upper - slo_limit) / max(1.0, slo_limit))
        soft_risk = base_risk
        
        # Softmax aggregation for risk vector
        if isinstance(risks, dict):
            vals = []
            for k in ('latency', 'timeout', 'error', 'memory'):
                v = float(risks.get(k, 0.0))
                vals.append(v)
            
            # Log-Sum-Exp Smooth Max
            mx = max(vals) if vals else 0.0
            if tau <= 0.0: tau = 1.0
            s = 0.0
            for v in vals:
                s += pow(2.718281828, (v - mx) / tau)
            if s > 0.0:
                soft_risk = mx + tau * math.log(s)
            else:
                soft_risk = mx

        if isinstance(risk_comp, (int, float)):
            soft_risk = float(risk_comp)
            
        # CRITICAL FIX: Gradient Direction
        # Current Architecture: u = Resource Allocation (0.01 to 1.0).
        # Higher u -> Higher Load/Latency (Wait, No!)
        # Higher u (Resources) -> LOWER Latency (Fast execution).
        # Therefore, d(Latency)/du is NEGATIVE.
        # d(Risk)/du is NEGATIVE.
        # To reduce Risk, we must INCREASE u.
        # Gradient Descent: u_new = u - eta * grad.
        # So grad should be NEGATIVE for risk to increase u.
        grad_risk = -1.0 * soft_risk 
        if isinstance(ku, (int, float)):
             grad_risk = -float(ku) * soft_risk

        # 2. Utility Gradient (Cost of Resources)
        # v25: 终极重构 - 风险置信度衰减与强制回收
        
        # 1. Tracking Error Gradient
        grad_track = 0.0
        if ref_latency is not None:
            # 对预测值进行物理截断，防止 WCP 中毒
            safe_pred = min(pred_upper, ref_latency * 1.5) # Allow slightly more headroom
            diff = (safe_pred - ref_latency) / max(1.0, slo_limit)
            grad_track = -1.0 * diff
            
        # 2. Utility Gradient (资源回收拉力)
        # v34: 彻底解决 Alloc 掉到 0.4 的问题。
        # 之前的回收拉力 (grad_waste) 仍然存在，导致在没有明显违规时，Alloc 会缓慢下降到 0.4 左右，然后触发高延迟。
        # 修复：
        # 1. 引入“安全底线”概念。如果当前 Alloc 已经很低（例如 < 0.8），则大幅削弱回收拉力。
        # 2. 进一步降低 w2 (2.0 -> 0.5)。
        w2 = 0.5
        waste_factor = 1.0
        if prev_u < 0.8:
            waste_factor = 0.1 # 如果 Alloc 已经低于 0.8，几乎停止回收
        grad_waste = w2 * prev_u * waste_factor
        
        # 3. 风险梯度 (WCP 提供的概率保证)
        slo_viol = 0.0
        actual_metrics = {}
        if isinstance(state, dict):
            actual_metrics = state.get('metrics', {})
            
        if actual_metrics:
            slo_viol = float(actual_metrics.get('slo_violation_rate', 0.0))
        
        # 动态风险权重：
        # v34: 增加对预测延迟的敏感度。即使没有实际违规，只要预测延迟接近 SLO，就应该增加风险权重。
        confidence_gate = 1.0
        if slo_viol < 0.01:
            # 如果预测延迟已经超过 SLO 的 80%，提高警惕
            if pred_upper > slo_limit * 0.8:
                confidence_gate = 0.8
            else:
                confidence_gate = 0.2
            
        dynamic_risk_weight = 3.0 * (slo_viol + 0.1) * confidence_gate
        if prev_u > 0.9:
            dynamic_risk_weight *= 0.5 # 稍微降低高位的质疑程度
            
        forced_recovery = 0.0
            
        # 终极合力计算
        grad = 2.0 * grad_track + dynamic_risk_weight * grad_risk + grad_waste + forced_recovery
        
        if prev_u > 0.8:
            if random.random() < 0.1:
                print(f"[MPC-CORE-v34] U:{prev_u:.3f} | Grad:{grad:.2f} (T:{2.0*grad_track:.2f}, R:{dynamic_risk_weight*grad_risk:.2f}, W:{grad_waste:.2f}) | Gate:{confidence_gate:.2f}")
        
        # Update Step
        # v34: 保持极低的学习率
        step_eta = 0.05
        if grad < -2.0: # 降低向上加速的阈值
            step_eta *= 1.5
            
        step = step_eta * (grad + gamma * prev_u)
        u_new = prev_u - step
        
        # --- Physical Rate Limiting ---
        # v34: 极其严格的限速，彻底杜绝跳跃
        max_increase = 0.10
        max_decrease = 0.05 # 每次最多只允许下降 0.05
        
        if u_new > prev_u + max_increase:
            u_new = prev_u + max_increase
        elif u_new < prev_u - max_decrease:
            u_new = prev_u - max_decrease
            
        # DEBUG: 详情 (v29)
        if random.random() < 0.1:
            print(f"[MPC-DEBUG-v29] u:{prev_u:.2f}->{u_new:.2f} | Total_Grad:{grad:.1f}")
        
        # Projection to Feasible Set U (Box constraints [0, 1])
        lower = 0.0
        upper = 1.0
        if u_new < lower: u_new = lower
        if u_new > upper: u_new = upper
        
        return u_new
    
    def tune_params_by_price(self, price, eta_base, gamma_base, bands=None, state=None):
        """
        Adjust learning rate (eta) and regularization (gamma) based on price bands.
        """
        eta = float(eta_base)
        gamma = float(gamma_base)
        
        if isinstance(bands, list):
            for b in bands:
                lo = float(b.get('lo', 0.0))
                hi = float(b.get('hi', lo))
                e = float(b.get('eta_mul', 1.0))
                g = float(b.get('gamma_mul', 1.0))
                if price >= lo and price < hi:
                    eta = eta_base * e
                    gamma = gamma_base * g
                    break
        else:
            price_high = float(state.get('opt_price_high', 200.0)) if isinstance(state, dict) else 200.0
            price_med = float(state.get('opt_price_med', 50.0)) if isinstance(state, dict) else 50.0
            eta_high_mul = float(state.get('opt_eta_high_mul', 0.5)) if isinstance(state, dict) else 0.5
            gamma_high_mul = float(state.get('opt_gamma_high_mul', 1.5)) if isinstance(state, dict) else 1.5
            eta_med_mul = float(state.get('opt_eta_med_mul', 0.8)) if isinstance(state, dict) else 0.8
            gamma_med_mul = float(state.get('opt_gamma_med_mul', 1.2)) if isinstance(state, dict) else 1.2
            if price >= price_high:
                eta = eta_base * eta_high_mul
                gamma = gamma_base * gamma_high_mul
            elif price >= price_med:
                eta = eta_base * eta_med_mul
                gamma = gamma_base * gamma_med_mul
        return eta, gamma
