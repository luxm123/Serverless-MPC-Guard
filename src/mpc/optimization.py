import math

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
        # v22: 回归理性。使用二次方惩罚，确保在 u 较高时产生足够的向下压力
        # grad_waste = 2 * w2 * u
        w2 = 15.0
        grad_waste = 2.0 * w2 * prev_u
        
        # v24: 系统性重构 - 区分“可控延迟”与“环境噪声”
        # 1. Tracking Error Gradient (风险拉力)
        grad_track = 0.0
        if ref_latency is not None:
            # WCP 提供的 pred_upper。如果 WCP 预测值远超 1.5x SLO，
            # 我们认为 WCP 受到了环境噪声污染，进行强制平滑。
            safe_pred = min(pred_upper, ref_latency * 1.3) 
            diff = (safe_pred - ref_latency) / max(1.0, slo_limit)
            grad_track = -1.0 * diff
            
        # 2. Utility Gradient (资源回收拉力)
        # v24: 非线性回收。当 u 越高，回收压力呈平方级增加
        # w2_base = 15.0
        # u=0.5 -> grad=15.0; u=1.0 -> grad=60.0
        w2_base = 15.0
        grad_waste = 2.0 * w2_base * (prev_u ** 2) * 4.0 
        
        # 3. 风险梯度
        risk_weight = 0.1
        
        # v24: 合力计算
        grad = 2.0 * grad_track + risk_weight * grad_risk + grad_waste
        
        # Update Step
        step = eta * (grad + gamma * prev_u)
        u_new = prev_u - step
        
        # --- Physical Rate Limiting (v24) ---
        # 爬坡上限收紧到 0.05，确保稳定性
        max_increase = 0.05
        if u_new > prev_u + max_increase:
            u_new = prev_u + max_increase
            
        # DEBUG: 详情 (v24)
        print(f"[MPC-DEBUG-v24] u:{prev_u:.2f}->{u_new:.2f} | T:{2.0*grad_track:.2f} W:{grad_waste:.2f} | Total:{grad:.2f}")
        
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
