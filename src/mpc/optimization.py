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

        overhead_ms = float(state.get('e2e_overhead_ms', 0.0))
        pred_total = float(pred_upper) + overhead_ms
        safety_target = float(slo_limit) - 20.0
        error = pred_total - safety_target

        if pred_total <= (safety_target - 30.0):
            alloc_floor = 0.40
        elif pred_total >= safety_target:
            alloc_floor = 0.60
        else:
            alloc_floor = 0.40 + 0.20 * ((pred_total - (safety_target - 30.0)) / 30.0)
        
        # Aggressive LR for QoS recovery, smoother for efficiency gains
        lr = 0.5 if error > 0 else 0.3
        
        if error > 0:
            # Risk detected -> Increase allocation
            # The higher the risk, the more we increase
            urgency = 1.0
            if pred_total > slo_limit: urgency = 10.0
            grad = -1.0 * (error / (10.0 / urgency))
        else:
            # Safe zone -> Decrease allocation to improve density
            # We use a smaller gradient to avoid jitter
            grad = 0.6 if abs(error) > 30.0 else 0.2

        new_alloc = prev_u - lr * grad

        if pred_total >= (safety_target - 10.0) and new_alloc < prev_u:
            new_alloc = prev_u
        
        # Safety Clamps & Adaptive Bounds
        if pred_total > slo_limit + 10.0:
            new_alloc = 1.0 # Emergency jump
        elif new_alloc < prev_u:
            # Efficiency gain: allow down-scaling
            max_reduction = 0.12 if error < -40.0 else 0.06
            new_alloc = max(new_alloc, prev_u - max_reduction)
        else:
            # QoS protection: allow up-scaling
            new_alloc = min(new_alloc, prev_u + 0.40)

        # v65.1: Minimum allocation 0.40 for high density
        final_alloc = max(alloc_floor, min(1.0, new_alloc))
        
        state['opt_debug'] = {
            "pred_upper": pred_upper,
            "pred_total": pred_total,
            "e2e_overhead_ms": overhead_ms,
            "alloc_floor": alloc_floor,
            "error": error,
            "grad": grad,
            "final_alloc": final_alloc,
            "overhead_ms": (time.time() - start_t) * 1000.0
        }
        return final_alloc
