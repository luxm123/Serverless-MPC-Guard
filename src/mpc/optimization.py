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
        v50: Ours (MPC-Guard) - 全局余量版。
        """
        strategy = kwargs.get('strategy', 'mpc_integrated')
        
        if strategy == 'gsight':
            return self._gsight_optimize(prev_u, kwargs.get('state', {}))
        elif strategy == 'owl':
            return self._owl_optimize(prev_u, kwargs.get('state', {}))
            
        # Default: MPC-Guard (Ours)
        start_t = time.time()
        state = kwargs.get('state', {})
        actual_p90 = float(state.get('p90_belief', 140.0))
        target = 135.0 
        error = actual_p90 - target
        
        if error > 0:
            grad = -1.0 * (error / 15.0) 
        else:
            grad = 0.05 

        lr = 0.08 
        new_alloc = prev_u - lr * grad
        
        if new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - 0.01) 
        else:
            new_alloc = min(new_alloc, prev_u + 0.15)

        final_alloc = max(0.60, min(1.0, new_alloc))
        overhead = (time.time() - start_t) * 1000.0
        state['opt_debug'] = {
            "grad": grad,
            "error": error,
            "overhead_ms": overhead,
            "final_alloc": final_alloc
        }
        return final_alloc

    def _gsight_optimize(self, prev_u, state):
        """
        Gsight (Lightweight Reproduction):
        Predictive EMA-based scaling. If latency > target, increase linearly.
        """
        start_t = time.time()
        actual_p90 = float(state.get('p90_belief', 140.0))
        target = 160.0 # Gsight usually doesn't account for network jitter as much
        
        if actual_p90 > target:
            new_alloc = prev_u + 0.1 # Aggressive increase
        else:
            new_alloc = prev_u - 0.02 # Fixed decrease
            
        final_alloc = max(0.60, min(1.0, new_alloc))
        state['opt_debug'] = {"overhead_ms": (time.time() - start_t) * 1000.0}
        return final_alloc

    def _owl_optimize(self, prev_u, state):
        """
        Owl (Lightweight Reproduction):
        Tail-latency-aware. Uses a threshold-based reactive approach.
        """
        start_t = time.time()
        actual_p90 = float(state.get('p90_belief', 140.0))
        # Owl is more sensitive to spikes
        if actual_p90 > 175.0:
            new_alloc = 1.0 # Jump to max
        elif actual_p90 > 150.0:
            new_alloc = prev_u + 0.05
        else:
            new_alloc = prev_u - 0.05
            
        final_alloc = max(0.60, min(1.0, new_alloc))
        state['opt_debug'] = {"overhead_ms": (time.time() - start_t) * 1000.0}
        return final_alloc

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    opt = Optimizer()
    strategy = state.get('strategy', 'mpc_integrated')
    return opt.optimize_u(state.get('prev_alloc', 1.0), pred_upper, slo_limit, 0.0, state=state, strategy=strategy), {}, {}
