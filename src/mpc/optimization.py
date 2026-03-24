import time

class Optimizer:
    def __init__(self, params):
        pass

    def optimize_u(self, prev_u, pred_upper, slo_limit, price, **kwargs):
        """
        v59: Ours (MPC-Guard) - Proactive Burst-Awareness + Optimistic Locking
        """
        strategy = kwargs.get('strategy', 'mpc_integrated')
        state = kwargs.get('state', {})
        start_t = time.time()

        # --- v58: Shield Protocol (护盾协议) ---
        current_rps = float(state.get('current_rps', 0.0))
        prev_rps = float(state.get('prev_rps', 0.0))
        
        # Activate shield if RPS grows by >50% and the base RPS is meaningful
        if current_rps > 1.5 * prev_rps and prev_rps > 5.0:
            state['opt_debug'] = {
                "overhead_ms": (time.time() - start_t) * 1000.0,
                "final_alloc": 1.0,
                "reason": "Shield Protocol Activated: RPS burst detected."
            }
            return 1.0

        # --- Regular P90-based Gradient Descent Control ---
        actual_p90 = float(state.get('p90_belief', 140.0))
        target = 135.0 
        error = actual_p90 - target
        
        if error > 0:
            urgency = 1.0
            if actual_p90 > 155.0: urgency = 5.0
            if actual_p90 > 170.0: urgency = 20.0
            grad = -1.0 * (error / (10.0 / urgency))
        else:
            # If we are very safe, explore downwards more aggressively
            grad = 0.8 if abs(error) > 30.0 else 0.4

        lr = 0.1 
        new_alloc = prev_u - lr * grad
        
        # Safety clamps
        if actual_p90 > 176.0:
            new_alloc = 1.0 # Jump to max if we are about to violate SLO
        elif new_alloc < prev_u:
            new_alloc = max(new_alloc, prev_u - 0.05) # Max 5% reduction per step
        else:
            new_alloc = min(new_alloc, prev_u + 0.50) # Max 50% increase per step

        final_alloc = max(0.60, min(1.0, new_alloc))
        state['opt_debug'] = {
            "grad": grad,
            "error": error,
            "overhead_ms": (time.time() - start_t) * 1000.0,
            "final_alloc": final_alloc,
            "p90": actual_p90
        }
        return final_alloc
