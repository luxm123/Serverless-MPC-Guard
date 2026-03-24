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
        # v62.4: Target 135ms (Stable)
        # If actual latency is ~120ms, it will be < 135ms, allowing Alloc to drop slowly to save cost.
        target = 135.0 
        error = actual_p90 - target
        
        # v59.4: Increased learning rate and adjusted gradients for more responsive down-scaling
        lr = 0.2 # Back to 0.2 for stability
        
        if error > 0:
            # Positive error (latency too high) -> increase Alloc (grad must be negative)
            # new_alloc = prev_u - lr * grad -> new_alloc = prev_u + lr * |grad|
            urgency = 1.0
            if actual_p90 > 130.0: urgency = 5.0
            if actual_p90 > 150.0: urgency = 20.0
            grad = -1.0 * (error / (5.0 / urgency)) # Doubled gain
        else:
            # Negative error (too safe) -> decrease Alloc (grad must be positive)
            # new_alloc = prev_u - lr * grad
            grad = 0.5 if abs(error) > 20.0 else 0.2

        new_alloc = prev_u - lr * grad
        
        # Safety clamps
        if actual_p90 > 176.0:
            new_alloc = 1.0 # Jump to max if we are about to violate SLO
        elif new_alloc < prev_u:
            # Allow up to 10% reduction if we are very safe
            max_reduction = 0.10 if error < -20.0 else 0.05
            new_alloc = max(new_alloc, prev_u - max_reduction)
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
