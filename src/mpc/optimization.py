import time
import math

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

        try:
            prev_u = float(prev_u)
        except Exception:
            prev_u = 1.0
        if not math.isfinite(prev_u):
            prev_u = 1.0

        try:
            pred_upper = float(pred_upper)
        except Exception:
            pred_upper = float(slo_limit)
        if not math.isfinite(pred_upper):
            pred_upper = float(slo_limit)

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
        last_y = float(state.get('last_y', 0.0))
        uncertainty = float(state.get('uncertainty', 0.0))
        pred_total = float(pred_upper) + overhead_ms
        obs_total = last_y + overhead_ms
        safety_target = float(slo_limit) - 20.0
        error = pred_total - safety_target

        safe_streak = int(state.get('safe_streak', 0))
        if pred_total <= (safety_target - 35.0) and obs_total <= (safety_target - 35.0) and uncertainty <= 15.0:
            safe_streak += 1
        else:
            safe_streak = max(0, safe_streak - 1)
        state['safe_streak'] = safe_streak

        if pred_total <= (safety_target - 30.0):
            alloc_floor = 0.40
        elif pred_total >= safety_target:
            alloc_floor = 0.60
        else:
            alloc_floor = 0.40 + 0.20 * ((pred_total - (safety_target - 30.0)) / 30.0)

        if safe_streak >= 30:
            alloc_floor = min(alloc_floor, 0.50)
        if safe_streak >= 60:
            alloc_floor = min(alloc_floor, 0.45)
        
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
            "obs_total": obs_total,
            "e2e_overhead_ms": overhead_ms,
            "alloc_floor": alloc_floor,
            "uncertainty": uncertainty,
            "safe_streak": safe_streak,
            "error": error,
            "grad": grad,
            "final_alloc": final_alloc,
            "overhead_ms": (time.time() - start_t) * 1000.0
        }
        return final_alloc
