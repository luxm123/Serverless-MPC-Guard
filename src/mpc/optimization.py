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

        current_rps = float(state.get('current_rps', 0.0))
        prev_rps = float(state.get('prev_rps', 0.0))
        try:
            budget = float(state.get("budget", 0.0) or 0.0)
        except Exception:
            budget = 0.0
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0
        spike = False
        if math.isfinite(current_rps) and math.isfinite(prev_rps):
            if prev_rps > 0.5 and current_rps > 1.5 * prev_rps and current_rps > 2.0:
                spike = True
            elif budget > 0.0 and current_rps > 1.2 * budget and (current_rps - prev_rps) > 0.3 * budget:
                spike = True
        if spike:
            try:
                jump_max = float(state.get('max_alloc', 1.0) or 1.0)
            except Exception:
                jump_max = 1.0
            if not math.isfinite(jump_max):
                jump_max = 1.0
            jump_max = float(max(0.4, min(4.0, jump_max)))
            return float(min(jump_max, max(prev_u, prev_u + 0.45)))

        # --- v66.0: Robust Prediction-Based Control ---
        # pred_upper is (WCP Prediction + Margin). 
        # This represents the 90th percentile worst-case latency.
        # We want to keep this BELOW the slo_limit (180ms).

        overhead_ms = float(state.get('e2e_overhead_ms', 0.0))
        last_y = float(state.get('last_y', 0.0))
        uncertainty = float(state.get('uncertainty', 0.0))
        concurrency = float(state.get('concurrency', state.get('backlog', 0.0)))
        backlog = float(state.get('backlog', concurrency))
        try:
            unc_scale = float(state.get('unc_scale', 1.0) or 1.0)
        except Exception:
            unc_scale = 1.0
        if not math.isfinite(unc_scale) or unc_scale <= 0.0:
            unc_scale = 1.0
        unc_scale = float(max(0.5, min(3.0, unc_scale)))
        pred_total = max(float(pred_upper), float(last_y) + float(unc_scale) * float(uncertainty))
        obs_total = last_y
        try:
            tight_slo_ms = float(state.get('tight_slo_ms', 80.0) or 80.0)
        except Exception:
            tight_slo_ms = 80.0
        if not math.isfinite(tight_slo_ms) or tight_slo_ms <= 0.0:
            tight_slo_ms = 80.0
        tight_slo_ms = float(max(20.0, min(200.0, tight_slo_ms)))

        if float(slo_limit) <= tight_slo_ms:
            margin_ms = min(4.0, 0.03 * float(slo_limit))
        else:
            margin_ms = min(10.0, 0.10 * float(slo_limit))
        safety_target = max(1.0, float(slo_limit) - float(margin_ms))
        servers = max(1.0, concurrency)
        service_est = max(40.0, last_y)
        queue_depth_per_server = max(0.0, (backlog - servers) / servers)
        queue_delay = min(1500.0, queue_depth_per_server * service_est)
        server_pred = pred_total + queue_delay
        e2e_pred = server_pred + overhead_ms
        error = server_pred - safety_target
        min_alloc = 0.0
        try:
            min_alloc = float(state.get('min_alloc', 0.0) or 0.0)
        except Exception:
            min_alloc = 0.0
        if not math.isfinite(min_alloc):
            min_alloc = 0.0

        max_alloc = 1.0
        try:
            max_alloc = float(state.get('max_alloc', 1.0) or 1.0)
        except Exception:
            max_alloc = 1.0
        if not math.isfinite(max_alloc):
            max_alloc = 1.0
        max_alloc = float(max(0.4, min(4.0, max_alloc)))

        min_alloc = float(max(0.0, min(max_alloc, min_alloc)))

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

        if float(slo_limit) <= tight_slo_ms:
            alloc_floor = max(alloc_floor, 0.85 if error <= 0.0 else 0.95)

        if budget > 0.0:
            if concurrency >= 1.8 * budget:
                alloc_floor = max(alloc_floor, 0.90)
            elif concurrency >= 1.2 * budget:
                alloc_floor = max(alloc_floor, 0.75)
            elif concurrency >= 0.6 * budget:
                alloc_floor = max(alloc_floor, 0.60)
        else:
            if concurrency >= 180.0:
                alloc_floor = max(alloc_floor, 0.90)
            elif concurrency >= 120.0:
                alloc_floor = max(alloc_floor, 0.75)
            elif concurrency >= 60.0:
                alloc_floor = max(alloc_floor, 0.60)

        if safe_streak >= 30:
            alloc_floor = min(alloc_floor, 0.50)
        if safe_streak >= 60:
            alloc_floor = min(alloc_floor, 0.45)
        if float(slo_limit) <= tight_slo_ms:
            safe_relax = False
            if budget > 0.0:
                safe_relax = (safe_streak >= 20 and error <= -20.0 and backlog <= 0.8 * budget and uncertainty <= 15.0)
            else:
                safe_relax = (safe_streak >= 20 and error <= -20.0 and backlog <= 8.0 and uncertainty <= 15.0)
            alloc_floor = max(alloc_floor, 0.65 if safe_relax else 0.80)
        
        server_pred_prev = server_pred
        if safety_target > 1.0:
            u_req = prev_u * (server_pred_prev / safety_target)
        else:
            u_req = prev_u
        if not math.isfinite(u_req):
            u_req = prev_u

        if error > 0.0:
            target_alloc = max(prev_u, u_req)
            lr = 0.65 if concurrency >= 60.0 else 0.55
        else:
            target_alloc = min(prev_u, u_req)
            lr = 0.35 if concurrency >= 30.0 else 0.28

        target_alloc = float(max(alloc_floor, min_alloc, min(max_alloc, target_alloc)))
        new_alloc = float(prev_u + lr * (target_alloc - prev_u))

        if server_pred >= (safety_target - 10.0) and new_alloc < prev_u:
            new_alloc = prev_u
        
        # Safety Clamps & Adaptive Bounds
        emergency_margin = max(20.0, 0.3 * float(slo_limit))
        if server_pred > float(slo_limit) + emergency_margin:
            new_alloc = max_alloc # Emergency jump
        elif new_alloc < prev_u:
            # Efficiency gain: allow down-scaling
            if concurrency < 30.0:
                max_reduction = 0.10 if error < -50.0 else 0.06
            else:
                max_reduction = 0.10 if error < -40.0 else 0.05
            new_alloc = max(new_alloc, prev_u - max_reduction)
        else:
            # QoS protection: allow up-scaling
            if concurrency < 30.0:
                max_inc = 0.35 if error > 0.0 else 0.25
            else:
                max_inc = 0.60 if concurrency >= 120.0 else 0.40
            new_alloc = min(new_alloc, prev_u + max_inc)

        final_alloc = max(alloc_floor, min_alloc, min(max_alloc, new_alloc))
        
        state['opt_debug'] = {
            "pred_upper": pred_upper,
            "pred_total": pred_total,
            "obs_total": obs_total,
            "e2e_overhead_ms": overhead_ms,
            "alloc_floor": alloc_floor,
            "min_alloc": min_alloc,
            "max_alloc": max_alloc,
            "uncertainty": uncertainty,
            "concurrency": concurrency,
            "backlog": backlog,
            "queue_delay": queue_delay,
            "e2e_pred": e2e_pred,
            "safe_streak": safe_streak,
            "error": error,
            "u_req": u_req,
            "final_alloc": final_alloc,
            "overhead_ms": (time.time() - start_t) * 1000.0
        }
        return final_alloc
