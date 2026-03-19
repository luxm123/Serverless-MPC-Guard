from src.wcp.risk import compute_risk
import random

class Scheduler:
    def __init__(self, optimizer=None):
        self.slo_limit = 180.0
        self.optimizer = optimizer
        
    def decide_control_input(self, wcp_constraints, system_state, ref_latency=None, current_backlog=None):
        """
        Simplified decision logic without priority/QoS levels.
        """
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
            
        targets = {
            'latency': self.slo_limit,
            'timeout': 0.0,
            'error': 0.0,
            'memory': 0.8
        }
        risks, composite = compute_risk(bounds, targets)
        
        prev_u = float(system_state.get('last_alloc', 1.0))
        price = float(system_state.get('shadow_price', 0.0))
        resource_alloc = prev_u
        
        if self.optimizer:
            eta_base = float(system_state.get('u_eta', 0.05))
            gamma_base = float(system_state.get('gamma', 0.1))
            resource_alloc = self.optimizer.optimize_u(
                prev_u,
                ub_latency_total,
                self.slo_limit,
                price,
                eta=eta_base,
                gamma=gamma_base,
                risk_comp=composite,
                risks=risks,
                ref_latency=ref_latency,
                state=system_state,
                current_backlog=current_backlog
            )
            
        max_delta_up = 0.5 if (current_backlog and current_backlog > 50.0) else 0.2
        max_delta_down = 0.1 
            
        delta = resource_alloc - prev_u

        if delta > max_delta_up:
            resource_alloc = prev_u + max_delta_up
        elif delta < -max_delta_down:
            resource_alloc = prev_u - max_delta_down
            
        resource_alloc = max(0.01, min(1.0, resource_alloc))
        
        should_shed = False
        degrade_plan = None
                
        return should_shed, degrade_plan, resource_alloc
