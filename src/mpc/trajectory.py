class TrajectoryGenerator:
    """
    Step 1: Reference Trajectory Generation
    Formula: r_{i+1} = f(r_i, V_i, theta)
    """
    def __init__(self):
        pass # Stateless initialization

    def update_theta(self, error, state):
        """
        Online Calibration of Theta.
        """
        theta = float(state.get('traj_theta', 0.1))
        
        # Load params
        learning_rate = float(state.get('traj_lr', 0.01))
        err_threshold = float(state.get('traj_err_thresh', 0.05))
        min_theta = float(state.get('traj_min_theta', 0.01))
        max_theta = float(state.get('traj_max_theta', 0.5))
        
        if abs(error) > err_threshold: 
            theta = max(min_theta, theta - learning_rate)
        else:
            # If tracking well, slowly increase bandwidth
            theta = min(max_theta, theta + learning_rate * 0.1)
            
        state['traj_theta'] = theta
        return theta

    def get_reference(self, current_state, priority=None):
        """
        Generate reference trajectory for next step.
        """
        # Load persistent theta
        theta = float(current_state.get('traj_theta', 0.1))
        
        # Current system state
        curr_lat = current_state.get('p90_latency', 100.0)
        curr_util = current_state.get('cpu_util', 0.5)
        
        # Target should come from task or state, defaulting to 500ms
        target_latency = float(current_state.get('slo_target', 500.0))
        target_util = 0.8
        
        vi = self.compute_vi(current_state, priority)
            
        # Update Reference
        # r_{k+1} = (1-theta)*r_k + theta * (Target * Adjustment)
        # Note: We use curr_lat as proxy for r_k (current reference) if r_k is not stored.
        # Ideally we should store r_k. Let's check if 'ref_latency' is in state.
        prev_ref = float(current_state.get('prev_ref_latency', curr_lat))
        
        ref_latency = (1 - theta) * prev_ref + theta * (target_latency * vi)
        ref_util = target_util 
        
        # Store for next step
        current_state['prev_ref_latency'] = ref_latency
        
        # Calculate tracking error (for next update cycle)
        # Use previous reference vs current actual
        # If prev_ref was 500 and we got 600, error is (600-500)/500 = 0.2
        tracking_error = (curr_lat - prev_ref) / max(1.0, prev_ref)
        
        self.update_theta(tracking_error, current_state)
        
        return {
            'ref_latency': ref_latency,
            'ref_util': ref_util
        }

    def compute_vi(self, current_state, priority=None):
        """
        Calculate adjustment factor vi.
        Adapts to system constraints (utilization, price) and task priority.
        """
        util = current_state.get('cpu_util', 0.5)
        price = current_state.get('shadow_price', current_state.get('congestion_price', 0.0))
        vi = 1.0
        
        # 1. System Constraints
        util_high = float(current_state.get('traj_util_high', 0.9))
        vi_util_relax = float(current_state.get('traj_vi_util_relax', 1.1))
        
        if util > util_high:
            vi = vi_util_relax # Relax reference (allow higher latency) if system is busy
            
        price_high = float(current_state.get('traj_price_high', 200.0))
        price_med = float(current_state.get('traj_price_med', 50.0))
        vi_price_relax_high = float(current_state.get('traj_vi_price_relax_high', 1.15))
        vi_price_relax_med = float(current_state.get('traj_vi_price_relax_med', 1.05))
        
        if price >= price_high:
            vi = max(vi, vi_price_relax_high) # Relax more if price is very high
        elif price >= price_med:
            vi = max(vi, vi_price_relax_med)
            
        # 2. Task Priority (New: Design Requirement)
        # High priority -> tighter reference (vi < 1.0)
        # Low priority -> looser reference (vi > 1.0)
        prio_high = float(current_state.get('traj_prio_high', 0.8))
        prio_low = float(current_state.get('traj_prio_low', 0.3))
        vi_prio_tight = float(current_state.get('traj_vi_prio_tight', 0.9))
        vi_prio_relax = float(current_state.get('traj_vi_prio_relax', 1.1))

        if priority is not None:
            if priority >= prio_high:
                vi *= vi_prio_tight # Tighten reference for high priority
            elif priority <= prio_low:
                vi *= vi_prio_relax # Relax reference for low priority
                
        return vi
