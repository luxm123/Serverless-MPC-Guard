import os
from .priority import PriorityManager
from .trajectory import TrajectoryGenerator
from .optimization import Optimizer
from .scheduler import Scheduler
from .pricing import update_shadow_price

class MPCController:
    """
    Main MPC Controller Module
    Orchestrates the 4-step closed-loop process.
    """
    def __init__(self):
        self.priority_mgr = PriorityManager()
        self.trajectory_gen = TrajectoryGenerator()
        self.optimizer = Optimizer()
        self.scheduler = Scheduler(self.optimizer)
    
    def hydrate_defaults(self, system_state):
        defaults = {
            # Scheduler & global
            'slo_limit': 500.0, 'slo_timeout_limit': 0.0, 'slo_error_limit': 0.0, 'slo_mem_limit': 0.8,
            'u_max_delta': 0.15, 'sched_price_high': 200.0, 'sched_price_med': 50.0, 'sched_price_low': 10.0,
            'sched_prio_high_thr': 0.8, 'sched_prio_med_thr': 0.4, 'sched_relax_factor': 1.2,
            # Admission / buffer control
            'pred_admit_enabled': True,
            'admit_thr_platinum_ms': None,
            'admit_thr_gold_ms': None,
            'admit_thr_standard_ms': None,
            'queue_delay_model': 'backlog_linear',
            'queue_backlog_ttl_s': 2.0,
            'buffer_servers_default': 1.0,
            # Optimizer thresholds
            'opt_w3_step_inc': 50.0, 'opt_w3_step_dec': 0.95, 'opt_w3_min': 5.0, 'opt_w3_max': 2000.0,
            'opt_waste_stable_thr': 0.05, 'opt_waste_tense_thr': 0.2, 'opt_viol_tense_thr': 0.1,
            'opt_stable_thr': 3, 'opt_w3_relax_mul': 0.7, 'opt_w2_relax_mul': 0.9,
            'opt_w2_step_inc': 10.0, 'opt_w2_step_dec': 0.98, 'opt_w2_min': 0.5, 'opt_w2_max': 50.0,
            'opt_waste_err_base': 0.1, 'opt_waste_err_cap': 0.5, 'opt_int_waste_cap': 5.0,
            'opt_int_decay': 0.9, 'opt_w2_int_gain': 2.0, 'opt_price_norm': 100.0,
            'opt_price_high': 200.0, 'opt_price_med': 50.0, 'opt_eta_high_mul': 0.5, 'opt_gamma_high_mul': 1.5,
            'opt_eta_med_mul': 0.8, 'opt_gamma_med_mul': 1.2,
            # WCP thresholds
            'wcp_alpha_min': 0.01, 'wcp_alpha_dec': 0.9, 'wcp_alpha_step_dec': 0.005,
            'wcp_alpha_inc': 1.05, 'wcp_alpha_step_inc': 0.002,
            'wcp_risk_thr': 0.05, 'wcp_mem_thr': 0.8, 'trend_alpha': 0.2,
            'wcp_window': 100, 'wcp_win_max': 200, 'wcp_win_min': 50, 'wcp_win_inc': 20, 'wcp_win_dec': 10,
            'wcp_spike_thr': 1.5, 'wcp_drop_thr': 0.5,
            # Pricing thresholds
            'sp_B': 0.8, 'sp_risk_thr': 0.03, 'sp_kr_inc_fac': 1.05, 'sp_kr_inc_step': 0.01,
            'sp_kr_dec_fac': 0.98, 'sp_kr_dec_step': 0.005,
            'sp_queue_thr': 0.3, 'sp_kq_inc_fac': 1.05, 'sp_kq_inc_step': 0.005,
            'sp_kq_dec_fac': 0.98, 'sp_kq_dec_step': 0.003,
            'sp_load_high': 0.88, 'sp_risk_high': 0.04, 'sp_load_exit': 0.75, 'sp_risk_exit': 0.02,
            'sp_eta_norm_dec': 0.9, 'sp_eta_high_inc': 1.2, 'sp_eta_min': 0.01, 'sp_eta_max': 0.2,
            'sp_decay': 0.01, 'sp_risk_low': 0.01, 'sp_load_low': 0.6, 'sp_queue_low': 0.1, 'sp_streak_thr': 3,
            # Trajectory thresholds
            'traj_util_high': 0.9, 'traj_vi_util_relax': 1.1,
            'traj_price_high': 200.0, 'traj_price_med': 50.0,
            'traj_vi_price_relax_high': 1.15, 'traj_vi_price_relax_med': 1.05,
            'traj_prio_high': 0.8, 'traj_prio_low': 0.3,
            'traj_vi_prio_tight': 0.9, 'traj_vi_prio_relax': 1.1,
        }

        # [PROFILE OVERRIDE]
        # Check for scalability profile to apply aggressive tuning without modifying code permanently.
        # Check both Env Var (Lambda Config) and System State (Client Payload)
        profile_name = os.environ.get('MPC_PROFILE') or system_state.get('mpc_profile')
        
        if profile_name == 'scalability_tuned':
            overrides = {
                'sp_queue_thr': 0.1, 'sp_kq_inc_fac': 1.2, 'sp_kq_inc_step': 0.02,
                'sp_kq_dec_fac': 0.95, 'sp_kq_dec_step': 0.005,
                'sp_decay': 0.01, 'sp_risk_low': 0.01, 'sp_load_low': 0.6, 'sp_queue_low': 0.05,
                'traj_price_high': 100.0, 'traj_price_med': 20.0,
                'sp_backlog_capacity': 1000.0 # Default High Capacity
            }
            defaults.update(overrides)
            for k, v in overrides.items():
                system_state[k] = v
                
        elif profile_name == 'flash_crowd':
            # Flash Crowd Profile:
            # Lower capacity to ensure 200 threads trigger congestion logic.
            # Otherwise, 200 threads < 1000 capacity looks like "low load".
            overrides = {
                'sp_backlog_capacity': 150.0, # Trigger congestion at ~150 backlog
                'sp_queue_thr': 0.1,         # React fast to queue
                'sp_kq_inc_fac': 1.5,        # Aggressive price hike
                'traj_price_high': 150.0     # High barrier
            }
            defaults.update(overrides)
            for k, v in overrides.items():
                system_state[k] = v

        for k, v in defaults.items():
            if k not in system_state:
                system_state[k] = v

        slo_limit_ms = float(system_state.get('slo_limit', 500.0))
        if system_state.get('admit_thr_platinum_ms') is None:
            system_state['admit_thr_platinum_ms'] = slo_limit_ms * 1.2
        if system_state.get('admit_thr_gold_ms') is None:
            system_state['admit_thr_gold_ms'] = slo_limit_ms * 1.0
        if system_state.get('admit_thr_standard_ms') is None:
            system_state['admit_thr_standard_ms'] = slo_limit_ms * 0.8
        
    def update_feedback(self, metrics, system_state):
        """Part III: Closed-loop parameter update"""
        self.hydrate_defaults(system_state)
        # Update Optimizer Weights (e.g. w1, w2, w3)
        optimizer_weights = self.optimizer.update_weights(metrics, system_state)
        
        # Update Priority Weights (e.g. lambda1, alpha, beta) - Auto Calibration
        priority_weights = self.priority_mgr.update_params(metrics, system_state)
        
        return {
            'optimizer_weights': optimizer_weights,
            'priority_weights': priority_weights
        }
        
    def update_pricing(self, system_state):
        u = float(system_state.get('last_alloc', 1.0))
        lam, dbg = update_shadow_price(system_state, system_state, u)
        system_state['shadow_price'] = lam
        system_state['congestion_price'] = lam
        system_state['sp_debug'] = dbg
        return lam, dbg
        
    def update_mpc_stats(self, system_state, vec):
        stats = system_state.get('mpc_stats', {})
        n = int(stats.get('n', 0))
        sums = stats.get('sum', [0.0, 0.0])
        sums_sq = stats.get('sum_sq', [0.0, 0.0])
        ema_sum = stats.get('ema_sum', [0.0, 0.0])
        ema_sum_sq = stats.get('ema_sum_sq', [0.0, 0.0])
        alpha = float(system_state.get('mpc_alpha_ema', 0.1))
        for i in range(2):
            v = float(vec[i]) if i < len(vec) else 0.0
            sums[i] = float(sums[i]) + v
            sums_sq[i] = float(sums_sq[i]) + v * v
            ema_sum[i] = (1.0 - alpha) * float(ema_sum[i]) + alpha * v
            ema_sum_sq[i] = (1.0 - alpha) * float(ema_sum_sq[i]) + alpha * (v * v)
        n += 1
        system_state['mpc_stats'] = {'n': n, 'sum': sums, 'sum_sq': sums_sq, 'ema_sum': ema_sum, 'ema_sum_sq': ema_sum_sq}
        
    def decide(self, task, wcp_constraints, system_state):
        """
        Main Decision Function
        
        Args:
            task: Task Profile
            wcp_constraints: Output from WCP {'p90': ..., 'uncertainty': ...}
            system_state: Global system state (prices, history stats)
            
        Returns:
            Dict containing decision and meta-data
        """
        self.hydrate_defaults(system_state)
        self.update_pricing(system_state)
        
        # Step 1: Priority Quantification (Moved up to influence Reference)
        # (Includes Step 3.1 Hierarchical Model & 3.2 Weight Fusion)
        priority, task_vector = self.priority_mgr.calculate_priority(task, system_state)
        self.update_mpc_stats(system_state, task_vector)
        
        # Step 2: Reference Trajectory 
        # (Calculated for observability, implicitly used in scheduler via SLO consts)
        # Now uses priority to adjust vi (reference adjustment factor)
        ref = self.trajectory_gen.get_reference(system_state, priority=priority)
        # We update scheduler's SLO limit to match reference, 
        # effectively making the scheduler track this dynamic reference.
        ref_latency = float(ref.get('ref_latency', self.scheduler.slo_limit))
        self.scheduler.slo_limit = ref_latency
        
        # Step 3: Scheduling Execution
        qos_class = task.get('qos_class', 'Q2')
        should_shed, plan, alloc = self.scheduler.decide_control_input(
            priority, wcp_constraints, system_state, ref_latency=ref_latency, qos_class=qos_class,
            current_backlog=system_state.get('queue_backlog', 0.0)
        )
        
        # Step 4: Feedback Update (Persist Adaptive Params)
        # Note: In a real loop, this might happen AFTER observing effects, 
        # but here we update based on current metrics to prepare for NEXT request.
        # We use the current task's risk/metrics if available, or just system state.
        # Ideally, we should update weights based on the RESULT of the previous action.
        # Assuming system_state contains 'metrics' from the previous step?
        # Let's assume the caller passes fresh metrics in system_state or we use task risk.
        # For now, we update using the input task's risk/metrics as a proxy for "current system status"
        # or we rely on the external loop to call update_feedback.
        # To be safe, we do it here if metrics are present.
        if 'metrics' in system_state:
             self.update_feedback(system_state['metrics'], system_state)
        
        # Calculate predicted delay for Worker Circuit Breaker
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
                'priority': priority,
                'ref_target': ref,
                'task_vector': task_vector,
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
