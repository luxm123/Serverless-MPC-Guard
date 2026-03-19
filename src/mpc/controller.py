import os
from .trajectory import TrajectoryGenerator
from .optimization import Optimizer
from .scheduler import Scheduler
from .pricing import update_shadow_price

class MPCController:
    """
    Main MPC Controller Module
    Orchestrates the closed-loop process.
    Simplified: Priority and QoS logic removed.
    """
    def __init__(self):
        self.trajectory_gen = TrajectoryGenerator()
        self.optimizer = Optimizer()
        self.scheduler = Scheduler(self.optimizer)
    
    def hydrate_defaults(self, system_state):
        defaults = {
            # Scheduler & global
            'slo_limit': 180.0, 'slo_timeout_limit': 0.0, 'slo_error_limit': 0.0, 'slo_mem_limit': 0.8,
            'u_max_delta': 0.15, 'sched_price_high': 200.0, 'sched_price_med': 50.0, 'sched_price_low': 10.0,
            # Admission / buffer control
            'pred_admit_enabled': True,
            'queue_delay_model': 'backlog_linear',
            'queue_backlog_ttl_s': 2.0,
            'buffer_servers_default': 1.0,
            # Optimizer thresholds
            'opt_w3_step_inc': 50.0, 'opt_w3_step_dec': 0.95, 'opt_w3_min': 5.0, 'opt_w3_max': 2000.0,
            'opt_int_decay': 0.9, 'opt_price_norm': 100.0,
            # WCP thresholds
            'wcp_alpha_min': 0.01, 'wcp_alpha_dec': 0.9, 'wcp_alpha_step_dec': 0.005,
            'trend_alpha': 0.2,
            # Pricing thresholds
            'sp_B': 0.8, 'sp_risk_thr': 0.03, 'sp_decay': 0.01,
        }

        # [PROFILE OVERRIDE]
        profile_name = os.environ.get('MPC_PROFILE') or system_state.get('mpc_profile')
        
        if profile_name == 'scalability_tuned':
            overrides = {
                'sp_decay': 0.01,
                'sp_backlog_capacity': 1000.0 
            }
            defaults.update(overrides)
            for k, v in overrides.items():
                system_state[k] = v

        for k, v in defaults.items():
            if k not in system_state:
                system_state[k] = v

    def update_feedback(self, metrics, system_state):
        """Part III: Closed-loop parameter update"""
        self.hydrate_defaults(system_state)
        # Update Optimizer Weights
        optimizer_weights = self.optimizer.update_weights(metrics, system_state)
        return {
            'optimizer_weights': optimizer_weights
        }
        
    def update_pricing(self, system_state):
        u = float(system_state.get('last_alloc', 1.0))
        lam, dbg = update_shadow_price(system_state, system_state, u)
        system_state['shadow_price'] = lam
        system_state['congestion_price'] = lam
        system_state['sp_debug'] = dbg
        return lam, dbg
        
    def decide(self, task, wcp_constraints, system_state):
        """
        Main Decision Function
        """
        self.hydrate_defaults(system_state)
        self.update_pricing(system_state)
        
        # Step 1: Reference Trajectory 
        # Simplified: Priority removed.
        ref = self.trajectory_gen.get_reference(system_state)
        ref_latency = float(ref.get('ref_latency', self.scheduler.slo_limit))
        self.scheduler.slo_limit = ref_latency
        
        # Step 2: Scheduling Execution
        # Simplified: priority and qos_class removed.
        should_shed, plan, alloc = self.scheduler.decide_control_input(
            wcp_constraints, system_state, ref_latency=ref_latency,
            current_backlog=system_state.get('queue_backlog', 0.0)
        )
        
        # Step 3: Feedback Update
        if 'metrics' in system_state:
             self.update_feedback(system_state['metrics'], system_state)
        
        # Calculate predicted delay
        q_backlog = float(system_state.get('queue_backlog', 0.0))
        avg_svc = float(system_state.get('avg_service_ms', 100.0))
        pred_delay = q_backlog * avg_svc

        return {
            'decision': {
                'should_shed': should_shed,
                'degrade_plan': plan,
                'resource_alloc': alloc,
                'congestion_price': system_state.get('congestion_price', 0.0),
                'shadow_price': system_state.get('shadow_price', 0.0),
                'pred_queue_delay_ms': pred_delay,
                'queue_backlog': q_backlog
            },
            'meta': {
                'ref_target': ref,
                'wcp_bound': (
                    wcp_constraints.get('pred', {}).get('p90', 0) +
                    (
                        wcp_constraints.get('uncertainty', 0.0).get('p90', 0.0)
                        if isinstance(wcp_constraints.get('uncertainty', 0.0), dict)
                        else wcp_constraints.get('uncertainty', 0.0)
                    )
                )
            }
        }
