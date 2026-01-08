import math

class PriorityManager:
    """
    Step 3: Task Priority Quantification
    Logic: Hierarchical Evaluation + Subjective/Objective Weight Fusion
    """
    def __init__(self):
        pass

    def update_params(self, metrics, state=None):
        """
        Adaptive Priority Parameter Tuning.
        """
        # Defaults
        lambda1 = 0.6
        alpha = 0.7
        beta = 0.3
        phi = [0.6, 0.4]
        
        if state:
            lambda1 = float(state.get('prio_lambda1', lambda1))
            beta = float(state.get('prio_beta', beta))
            # alpha is usually 1-beta
            phi_raw = state.get('prio_phi', phi)
            if isinstance(phi_raw, list) and len(phi_raw) == 2:
                phi = [float(x) for x in phi_raw]

        # 1. Assess Performance
        error_rate = float(metrics.get('error_rate', 0.0))
        err_thr = float(state.get('prio_err_thr', 0.05))
        
        lambda_step = float(state.get('prio_lambda_step', 0.05))
        lambda_min = float(state.get('prio_lambda_min', 0.2))
        lambda_max = float(state.get('prio_lambda_max', 0.8))
        lambda_inc = float(state.get('prio_lambda_inc', 0.01))
        
        if error_rate > err_thr:
            # High error -> Trust objective constraints (shadow price) more
            lambda1 = max(lambda_min, lambda1 - lambda_step)
            beta = min(1.0 - lambda_min, beta + lambda_step)
            alpha = 1.0 - beta
        else:
            # Stable -> Allow more subjective/business influence
            lambda1 = min(lambda_max, lambda1 + lambda_inc)
            beta = max(1.0 - lambda_max, beta - lambda_inc)
            alpha = 1.0 - beta

        # Update Subjective Weights (Fuzzy Group Decision)
        new_phi = self._update_subjective_weights(metrics, phi)

        if state is not None:
            state['prio_lambda1'] = lambda1
            state['prio_beta'] = beta
            state['prio_alpha'] = alpha
            state['prio_phi'] = new_phi

        return {'lambda1': lambda1, 'alpha': alpha, 'beta': beta, 'phi': new_phi}

    def _update_subjective_weights(self, metrics, current_phi):
        """
        Calculate Subjective Weights (phi) using Fuzzy Rough Group Decision.
        """
        ratings_per_indicator = self._get_virtual_expert_ratings(metrics)
        
        centroids = []
        for ratings in ratings_per_indicator:
            agg_fuzzy = self._rough_group_decision(ratings)
            g = self._fuzzy_centroid(agg_fuzzy)
            centroids.append(g)
            
        total = sum(centroids)
        if total > 0:
            return [c / total for c in centroids]
        else:
            return current_phi





    def _fuzzy_centroid(self, fuzzy_num):
        """
        Calculate Centroid of a Trapezoidal/Triangular Fuzzy Number.
        Formula: G = (a + 2b + c) / 4 (for Triangular)
        Args:
            fuzzy_num: tuple (a, b, c)
        """
        a, b, c = fuzzy_num
        return (a + 2 * b + c) / 4.0

    def _rough_group_decision(self, expert_fuzzy_nums):
        """
        Rough Group Decision (Simplified for Implementation).
        
        Logic:
        1. We have N experts, each giving a fuzzy number (a, b, c).
        2. Instead of full Rough Set approximation which is computationally heavy,
           we use an 'Average Fuzzy Number' approach which is mathematically equivalent
           to the centroid of the aggregated fuzzy set in many simplified models.
        3. Aggregated (a_agg, b_agg, c_agg):
           a_agg = min(all a)
           b_agg = mean(all b)
           c_agg = max(all c)
        
        Returns:
            Aggregated Fuzzy Number (a, b, c)
        """
        if not expert_fuzzy_nums:
            return (0.0, 0.0, 0.0)
            
        a_vals = [f[0] for f in expert_fuzzy_nums]
        b_vals = [f[1] for f in expert_fuzzy_nums]
        c_vals = [f[2] for f in expert_fuzzy_nums]
        
        # Using a robust aggregation:
        # Lower bound is min, Upper is max, Middle is average
        return (min(a_vals), sum(b_vals)/len(b_vals), max(c_vals))

    def _get_virtual_expert_ratings(self, metrics):
        """
        Simulate 3 Virtual Experts giving Fuzzy Ratings for [Business Value, Consistency].
        Output: List of 2 lists (one for each indicator), each containing 3 fuzzy numbers.
        """
        # Metrics
        err_rate = float(metrics.get('error_rate', 0.0))
        timeout = float(metrics.get('timeout_rate', 0.0))
        waste = float(metrics.get('resource_waste_rate', 0.0))
        
        # Expert 1: SRE (Stability Focused)
        # If unstable (err > 1%), prioritizes Consistency highly.
        if err_rate > 0.01 or timeout > 0.02:
            sre_bv = (0.3, 0.4, 0.5)
            sre_cl = (0.8, 0.9, 1.0)
        else:
            sre_bv = (0.5, 0.6, 0.7)
            sre_cl = (0.4, 0.5, 0.6)
            
        # Expert 2: FinOps (Cost Focused)
        # If waste is high, prioritizes Business Value (to shed non-core).
        if waste > 0.1:
            fin_bv = (0.7, 0.8, 0.9)
            fin_cl = (0.2, 0.3, 0.4)
        else:
            fin_bv = (0.4, 0.5, 0.6)
            fin_cl = (0.4, 0.5, 0.6)
            
        # Expert 3: PM (Business Focused)
        # Always prioritizes Business Value.
        pm_bv = (0.8, 0.9, 0.95)
        pm_cl = (0.2, 0.3, 0.5)
        
        # Group by Indicator
        # Indicator 0: Business Value
        ratings_bv = [sre_bv, fin_bv, pm_bv]
        # Indicator 1: Consistency
        ratings_cl = [sre_cl, fin_cl, pm_cl]
        
        return [ratings_bv, ratings_cl]

    def get_objective_weights(self, stats):
        """
        Calculate Objective Weights using Coefficient of Variation (CV).
        Formula:
          CV_j = sigma_j / mu_j
          X_j = CV_j / sum(CV)
        """
        n = stats.get('n', 0)
        use_ema = 'ema_sum' in stats and 'ema_sum_sq' in stats
        if not use_ema and n < 2:
            return [0.5, 0.5]
        if use_ema:
            sum_vals = stats.get('ema_sum', [0.0, 0.0])
            sum_sq_vals = stats.get('ema_sum_sq', [0.0, 0.0])
            n_eff = 1.0
        else:
            sum_vals = stats.get('sum', [0.0, 0.0])
            sum_sq_vals = stats.get('sum_sq', [0.0, 0.0])
            n_eff = float(n)
        
        cvs = []
        for i in range(2):
            mu = sum_vals[i] / n_eff
            if mu <= 1e-6:
                cv = 0.0
            else:
                var = max(0.0, sum_sq_vals[i] - (sum_vals[i]**2) / n_eff)
                sigma = math.sqrt(max(0.0, var))
                cv = sigma / mu
            cvs.append(cv)
            
        sum_cv = sum(cvs)
        if sum_cv == 0:
            return [0.5, 0.5]
            
        return [cv / sum_cv for cv in cvs]

    def calculate_priority(self, task, system_state=None):
        """
        Step 3.1 & 3.2: Calculate Final Priority Score.
        """
        # 1. Hierarchical Evaluation (Indicators)
        # Assume task has raw features: 'business_value', 'user_tier', 'latency_sens'
        # Normalize to [0, 1]
        
        # Load weights from state if available
        lambda1 = 0.6
        alpha = 0.7
        beta = 0.3
        phi = [0.6, 0.4]
        
        if system_state:
            lambda1 = float(system_state.get('prio_lambda1', lambda1))
            beta = float(system_state.get('prio_beta', beta))
            # alpha is derived
            alpha = 1.0 - beta
            
            phi_raw = system_state.get('prio_phi', phi)
            if isinstance(phi_raw, list) and len(phi_raw) == 2:
                phi = [float(x) for x in phi_raw]

        # ... (rest of the logic uses lambda1, alpha, beta, phi)
        
        bv = float(task.get('business_value', 0.5))
        # Consistency roughly maps to how stable/reliable we want this to be
        # We can map 'tier' to consistency requirement.
        tier = task.get('tier', 'standard')
        
        cl_plat = 0.9
        cl_gold = 0.7
        cl_std = 0.5
        if system_state:
            cl_plat = float(system_state.get('prio_cl_plat', 0.9))
            cl_gold = float(system_state.get('prio_cl_gold', 0.7))
            cl_std = float(system_state.get('prio_cl_std', 0.5))

        if tier == 'platinum':
            cl = cl_plat
        elif tier == 'gold':
            cl = cl_gold
        else:
            cl = cl_std
            
        # Indicator Vector R_i = [bv, cl]
        R_i = [bv, cl]
        
        # 2. Subjective Score S_i = phi * R_i^T
        # phi is the weight vector from Fuzzy Group Decision
        S_i = sum(phi[k] * R_i[k] for k in range(len(phi)))
        
        # 3. Objective Score O_i (Constraint-based)
        # We use Shadow Price as a proxy for "Resource Scarcity Risk"
        # If Shadow Price is high, we want to prioritize tasks that are "cheaper" or "more critical"?
        # Actually, Objective Weighting usually reflects the "information content" or "discrimination power" of attributes.
        # But here, let's say O_i is based on system load (Shadow Price).
        # If system is stressed, O_i might penalize heavy tasks? 
        # Let's stick to the design: "Entropy/CRITIC method for objective weights" -> That generates 'w_obj' for attributes.
        # But we need a score O_i. 
        # Let's assume O_i is similar to S_i but using Objective Weights w_obj instead of phi.
        # For simplicity, let's use the 'phi' (subjective) vs 'Objective State' (shadow price) fusion at the top level.
        
        # Simplified Fusion:
        # Priority = lambda1 * S_i + (1 - lambda1) * (SystemConstraintFactor)
        # Wait, the formula usually is P = alpha * S_i + beta * O_i
        # where alpha + beta = 1.
        
        # Let's assume O_i comes from "How well does this task fit current constraints?"
        # If Shadow Price is high, O_i is low?
        sp = 0.0
        if system_state:
            sp = float(system_state.get('shadow_price', 0.0))
        
        # Normalize SP to [0, 1] (Assuming max SP ~ 100)
        sp_norm = min(1.0, sp / 100.0)
        
        # If SP is high, O_i (Objective Score) should reflect "Urgency". 
        # Maybe O_i is just S_i? Or maybe O_i includes 'Resource Cost' of the task?
        # Let's define O_i = 1 - (EstimatedCost * sp_norm)
        # For now, let's keep it simple: O_i = S_i (Objective assessment aligns with Subjective)
        # BUT, we use 'beta' (Objective Weight) to modulate the impact of Shadow Price on the final decision.
        
        # Design correction:
        # Priority = alpha * S_i + beta * (1 - sp_norm) ?
        # No, that would mean when loaded, priority drops for everyone. That makes sense (load shedding).
        
        O_i = 1.0 - sp_norm
        
        # Final P_i
        P_i = alpha * S_i + beta * O_i
        
        return P_i, [bv, cl]
