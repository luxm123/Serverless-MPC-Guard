import time

class Optimizer:
    def __init__(self, params):
        pass

    def optimize_u(self, prev_u, pred_upper, slo_limit, price, **kwargs):
        """
        v66.0: Robust MPC - Breaking the Performance-Overhead Trade-off
        Uses WCP Upper Bound (pred_upper) for proactive QoS protection.
        """
        strategy = kwargs.get('strategy', 'mpc_integrated')
        state = kwargs.get('state', {})
        start_t = time.time()

        # --- v58: Shield Protocol ---
        current_rps = float(state.get('current_rps', 0.0))
        prev_rps = float(state.get('prev_rps', 0.0))
        if current_rps > 1.5 * prev_rps and prev_rps > 5.0:
            return 1.0

        # --- v66.0: Robust Prediction-Based Control ---
        # pred_upper is (WCP Prediction + Margin). 
        # This represents the 90th percentile worst-case latency.
        # We want to keep this BELOW the slo_limit (180ms).
        
        # Define a safety target slightly below SLO to provide a buffer
        safety_target = slo_limit - 20.0 # 160ms
        
        # Calculate error based on the ROBUST prediction
        error = pred_upper - safety_target
        
        # Aggressive LR for QoS recovery, smoother for efficiency gains
        lr = 0.5 if error > 0 else 0.3
        
        if error > 0:
            # Risk detected -> Increase allocation
            # The higher the risk, the more we increase
            urgency = 1.0
            if pred_upper > slo_limit: urgency = 10.0 # Extreme urgency if bound exceeds SLO
            grad = -1.0 * (error / (10.0 / urgency))
        else:
            # Safe zone -> Decrease allocation to improve density
            # We use a smaller gradient to avoid jitter
            grad = 0.6 if abs(error) > 30.0 else 0.2

        new_alloc = prev_u - lr * grad
        
        # Safety Clamps & Adaptive Bounds
        if pred_upper > slo_limit + 10.0:
            new_alloc = 1.0 # Emergency jump
        elif new_alloc < prev_u:
            # Efficiency gain: allow down-scaling
            max_reduction = 0.15 if error < -40.0 else 0.08
            new_alloc = max(new_alloc, prev_u - max_reduction)
        else:
            # QoS protection: allow up-scaling
            new_alloc = min(new_alloc, prev_u + 0.40)

        # v65.1: Minimum allocation 0.40 for high density
        final_alloc = max(0.40, min(1.0, new_alloc))
        
        state['opt_debug'] = {
            "pred_upper": pred_upper,
            "error": error,
            "grad": grad,
            "final_alloc": final_alloc,
            "overhead_ms": (time.time() - start_t) * 1000.0
        }
        return final_alloc
