from src.wcp.risk import compute_risk
import random

class Scheduler:
    def __init__(self, optimizer=None):
        self.slo_limit = 500.0
        self.optimizer = optimizer
        
    def decide_control_input(self, priority, wcp_constraints, system_state, ref_latency=None, qos_class=None):
        pred = wcp_constraints.get('pred', {})
        unc_raw = wcp_constraints.get('uncertainty', 0.0)
        queue_delay_ms = float(system_state.get('pred_queue_delay_ms', 0.0) or 0.0)
        if isinstance(unc_raw, dict):
            ub_latency = float(pred.get('p90', 0.0)) + float(unc_raw.get('p90', 0.0))
            ub_timeout = float(pred.get('timeout_rate', 0.0)) + float(unc_raw.get('timeout_rate', 0.0))
            ub_error = float(pred.get('error_rate', 0.0)) + float(unc_raw.get('error_rate', 0.0))
            ub_memory = float(pred.get('memory_pressure', 0.0)) + float(unc_raw.get('memory_pressure', 0.0))
        else:
            unc = float(unc_raw)
            ub_latency = float(pred.get('p90', 0.0)) + unc
            ub_timeout = float(pred.get('timeout_rate', 0.0)) + unc
            ub_error = float(pred.get('error_rate', 0.0)) + unc
            ub_memory = float(pred.get('memory_pressure', 0.0)) + unc
        ub_latency_total = ub_latency + queue_delay_ms
        bounds = {
            'latency': {'upper': ub_latency_total},
            'timeout': {'upper': ub_timeout},
            'error': {'upper': ub_error},
            'memory': {'upper': ub_memory}
        }
        if ref_latency is None:
            self.slo_limit = float(system_state.get('slo_limit', self.slo_limit))
        self.slo_timeout_limit = float(system_state.get('slo_timeout_limit', getattr(self, 'slo_timeout_limit', 0.0)))
        self.slo_error_limit = float(system_state.get('slo_error_limit', getattr(self, 'slo_error_limit', 0.0)))
        self.slo_mem_limit = float(system_state.get('slo_mem_limit', getattr(self, 'slo_mem_limit', 0.8)))
        targets = {
            'latency': self.slo_limit,
            'timeout': self.slo_timeout_limit,
            'error': self.slo_error_limit,
            'memory': self.slo_mem_limit
        }
        risks, composite = compute_risk(bounds, targets)
        
        should_shed = False
        degrade_plan = None
        prev_u = float(system_state.get('last_alloc', 1.0))
        price_vec = system_state.get('shadow_price_vector', None)
        vec_weights = system_state.get('sp_vec_weights', None)
        if isinstance(price_vec, dict):
            if isinstance(vec_weights, dict) and vec_weights:
                price = sum(float(price_vec.get(k, 0.0)) * float(vec_weights.get(k, 0.0)) for k in price_vec.keys())
            else:
                price = max(float(v) for v in price_vec.values())
        else:
            price = float(system_state.get('shadow_price', 0.0))
        gamma = float(system_state.get('gamma', 0.1))
        resource_alloc = prev_u
        if self.optimizer:
            eta_base = float(system_state.get('u_eta', 0.05))
            gamma_base = float(system_state.get('gamma', 0.1))
            bands = system_state.get('price_bands', None)
            eta_eff, gamma_eff = self.optimizer.tune_params_by_price(price, eta_base, gamma_base, bands, system_state)
            resource_alloc = self.optimizer.optimize_u(
                prev_u,
                ub_latency_total,
                self.slo_limit,
                price,
                eta=eta_eff,
                gamma=gamma_eff,
                risk_comp=composite,
                ku=system_state.get('sp_ku', None),
                risks=risks,
                tau=float(system_state.get('risk_tau', 1.0)),
                ref_latency=ref_latency,
                state=system_state,
                priority=priority,
                qos_class=qos_class
            )
        max_delta = float(system_state.get('u_max_delta', 0.15))
        
        # Adaptive delta based on price (Dynamic Bands)
        # We use the same 'bands' logic as optimization or simple thresholds from state
        price_high = float(system_state.get('sched_price_high', 200.0))
        price_med = float(system_state.get('sched_price_med', 50.0))
        
        if price >= price_high:
            max_delta *= 0.5
        elif price >= price_med:
            max_delta *= 0.8
            
        delta = resource_alloc - prev_u
        if delta > max_delta:
            resource_alloc = prev_u + max_delta
        elif delta < -max_delta:
            resource_alloc = prev_u - max_delta
        
        # Priority-based Shedding Logic
        prio_high_thr = float(system_state.get('sched_prio_high_thr', 0.8))
        prio_med_thr = float(system_state.get('sched_prio_med_thr', 0.4))

        pred_admit_enabled = bool(system_state.get('pred_admit_enabled', True))
        if pred_admit_enabled:
            thr_high = float(system_state.get('pred_thr_high', self.slo_limit * 1.2))
            thr_med = float(system_state.get('pred_thr_med', self.slo_limit * 1.0))
            thr_low = float(system_state.get('pred_thr_low', self.slo_limit * 0.8))
            if priority >= prio_high_thr:
                if price > price_high and ub_latency_total > thr_high:
                    should_shed = True
                    degrade_plan = "store_to_sqs_recovery"
            elif priority >= prio_med_thr:
                if ub_latency_total > thr_med:
                    should_shed = True
                    degrade_plan = "store_to_sqs_recovery"
            else:
                if ub_latency_total > thr_low:
                    should_shed = True
                    degrade_plan = "store_to_sqs"
        
        if priority >= prio_high_thr:
            if price > price_high and (ub_latency_total > self.slo_limit or composite > 0.0):
                should_shed = True
                degrade_plan = "store_to_sqs_recovery"
        elif priority >= prio_med_thr:
            relax_factor = float(system_state.get('sched_relax_factor', 1.2))
            relaxed_limit = self.slo_limit * relax_factor
            if (ub_latency_total > relaxed_limit or composite > 0.0) and price > price_med:
                # RED-like Probabilistic Shedding
                # Price range [price_med, price_high] -> Prob [0.0, 1.0]
                # This prevents binary on/off shedding and allows smooth degradation
                denom = max(1.0, price_high - price_med)
                prob = min(1.0, max(0.0, (price - price_med) / denom))
                
                if random.random() < prob:
                    should_shed = True
                    degrade_plan = "store_to_sqs_recovery"
        else:
            price_low = float(system_state.get('sched_price_low', 10.0))
            if (ub_latency_total > self.slo_limit or composite > 0.0) and price > price_low:
                should_shed = True
                degrade_plan = "store_to_sqs"
                
        return should_shed, degrade_plan, resource_alloc
