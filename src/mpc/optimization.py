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
        v58: Ours (MPC-Guard) - 前瞻性突发意识 (Proactive Burst-Awareness) 终极版
        """
        strategy = kwargs.get('strategy', 'mpc_integrated')
        state = kwargs.get('state', {})
        start_t = time.time()

        # --- v58: Shield Protocol (护盾协议) ---
        current_rps = float(state.get('current_rps', 0.0))
        prev_rps = float(state.get('prev_rps', 0.0))
        
        # 一级警报：RPS 增长超过 50% 且基数 > 5
        if current_rps > 1.5 * prev_rps and prev_rps > 5.0:
            state['opt_debug'] = {
                "overhead_ms": (time.time() - start_t) * 1000.0,
                "final_alloc": 1.0,
                "reason": "Shield Protocol Activated: RPS burst detected."
            }
            return 1.0 # 强制拉满资源，硬抗流量洪峰

        # --- 常规 P90 梯度下降控制 ---
        if strategy == 'gsight':
            return self._gsight_optimize(prev_u, state)
        elif strategy == 'owl':
            return self._owl_optimize(prev_u, state)
        elif strategy == 'passive_prewarm':
            return 1.0 
        elif strategy == 'ours_basic':
            pass 
            
        # Default: MPC-Guard (Ours Full)
        actual_p90 = float(state.get('p90_belief', 140.0))
        target = 135.0 
        error = actual_p90 - target
        
        if error > 0:
            urgency = 1.0
            if actual_p90 > 155.0: urgency = 5.0
            if actual_p90 > 170.0: urgency = 20.0
            grad = -1.0 * (error / (10.0 / urgency))
        else:
            if abs(error) > 30.0:
                grad = 0.8
            else:
                grad = 0.4

        lr = 0.1 
        new_alloc = prev_u - lr * grad
        
        if actual_p90 > 176.0:
            new_alloc = 1.0
        elif new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - 0.05) 
        else:
            new_alloc = min(new_alloc, prev_u + 0.50)

        final_alloc = max(0.60, min(1.0, new_alloc))
        overhead = (time.time() - start_t) * 1000.0
        state['opt_debug'] = {
            "grad": grad,
            "error": error,
            "overhead_ms": overhead,
            "final_alloc": final_alloc,
            "p90": actual_p90
        }
        return final_alloc

    def _gsight_optimize(self, prev_u, state):
        """
        Gsight (EMA Predictive): 
        EMA 预测通常较保守。下探步长 0.01。
        """
        start_t = time.time()
        actual_p90 = float(state.get('p90_belief', 140.0))
        target = 150.0 
        
        if actual_p90 > target:
            new_alloc = prev_u + 0.10
        else:
            new_alloc = prev_u - 0.01
            
        final_alloc = max(0.60, min(1.0, new_alloc))
        state['opt_debug'] = {"overhead_ms": (time.time() - start_t) * 1000.0}
        return final_alloc

    def _owl_optimize(self, prev_u, state):
        """
        Owl (Tail-latency aware):
        阈值响应型。下探步长 0.02。
        """
        start_t = time.time()
        actual_p90 = float(state.get('p90_belief', 140.0))
        if actual_p90 > 170.0:
            new_alloc = 1.0
        elif actual_p90 > 140.0:
            new_alloc = prev_u + 0.10
        else:
            new_alloc = prev_u - 0.02
            
        final_alloc = max(0.60, min(1.0, new_alloc))
        state['opt_debug'] = {"overhead_ms": (time.time() - start_t) * 1000.0}
        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    opt = Optimizer()
    strategy = state.get('strategy', 'mpc_integrated')
    return opt.optimize_u(state.get('prev_alloc', 1.0), pred_upper, slo_limit, 0.0, state=state, strategy=strategy), {}, {}
