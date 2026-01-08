import math

# --- 1. Pure Python Matrix Helpers (Strict No-Numpy) ---

def vec_dot(u, v):
    """Dot product: s = u . v"""
    return sum(u[i] * v[i] for i in range(len(u)))

def mat_vec(A, x):
    """Matrix-vector multiplication: y = A * x"""
    return [vec_dot(row, x) for row in A]

def vec_outer(u, v):
    """Outer product: A = u * v^T"""
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]

def mat_add(A, B):
    """Matrix addition: C = A + B"""
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]

def mat_scale(A, s):
    """Matrix scalar multiplication: C = A * s"""
    return [[val * s for val in row] for row in A]

def vec_sub(u, v):
    """Vector subtraction: w = u - v"""
    return [u[i] - v[i] for i in range(len(u))]

def vec_add(u, v):
    """Vector addition: w = u + v"""
    return [u[i] + v[i] for i in range(len(u))]

# --- 2. RLS Class (Parameterized Dynamics Model) ---

class RLS:
    """
    Recursive Least Squares filter.
    Model: y = theta^T * phi
    """
    def __init__(self, n_features, lambda_factor=0.99, delta=100.0):
        self.n = n_features
        self.lambda_factor = lambda_factor
        # Initialize theta to zeros
        self.theta = [0.0] * n_features
        # Initialize P to delta * I (large initial covariance)
        self.P = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            self.P[i][i] = delta

    def update(self, phi, y):
        """
        Update RLS parameters based on new observation.
        phi: Feature vector (list)
        y: Observed scalar value
        """
        # 1. Compute gain vector: k = (P * phi) / (lambda + phi^T * P * phi)
        P_phi = mat_vec(self.P, phi)
        phi_P_phi = vec_dot(phi, P_phi)
        denom = self.lambda_factor + phi_P_phi
        
        k = [val / denom for val in P_phi]
        
        # 2. Update parameters: theta = theta + k * (y - phi^T * theta)
        prediction = vec_dot(phi, self.theta)
        error = y - prediction
        
        theta_correction = [val * error for val in k]
        self.theta = vec_add(self.theta, theta_correction)
        
        # 3. Update covariance: P = (P - k * phi^T * P) / lambda
        # Term: k * (phi^T * P) = k * (P_phi)^T (since P is symmetric)
        k_P_phi_T = vec_outer(k, P_phi)
        P_numer = mat_add(self.P, mat_scale(k_P_phi_T, -1.0))
        self.P = mat_scale(P_numer, 1.0 / self.lambda_factor)
        
        return prediction

    def predict(self, phi):
        return vec_dot(phi, self.theta)

    # --- Serialization Helpers ---
    def to_dict(self):
        return {'theta': self.theta, 'P': self.P}

    @staticmethod
    def from_dict(d, n_features=2):
        rls = RLS(n_features)
        if d and 'theta' in d and 'P' in d:
            rls.theta = [float(x) for x in d['theta']]
            # Handle P parsing carefully if it comes from DynamoDB JSON
            raw_P = d['P']
            rls.P = [[float(val) for val in row] for row in raw_P]
        return rls

# --- 3. Main WCP Logic (Strict Implementation) ---

def wcp_update(state, metrics, alpha=0.1):
    """
    Strict WCP Update Implementation.
    
    1. Global L1 Non-conformity Score: s_k = ||y_k - y_hat_k||_1
    2. RLS Prediction: y_hat_{k+1} = f(y_k; theta)
    3. Weighted Quantile with Infinite Mass Point
    
    Args:
        state (dict): Persistent state (RLS params, score history, last_prediction).
        metrics (dict): Current observations.
        alpha (float): Target error rate.
        
    Returns:
        tuple: (prediction_next, uncertainty_radius, debug_info)
    """
    
    # Define metric vector order
    metric_names = ['p90', 'timeout_rate', 'error_rate', 'memory_pressure']
    
    # 1. Extract Current Observation y_k
    # Default values handle missing metrics gracefully, though in strict mode we might want to error out.
    y_k = [
        float(metrics.get('p90', 0.0)),
        float(metrics.get('timeout_rate', 0.0)),
        float(metrics.get('error_rate', 0.0)),
        float(metrics.get('memory_pressure', 0.0))
    ]
    
    # 2. Retrieve last prediction (y_hat_k)
    # If first run, use current metrics as naive prediction (or zeros)
    y_hat_k = state.get('last_prediction', [])
    if not y_hat_k or len(y_hat_k) != len(y_k):
        y_hat_k = [0.0] * len(y_k)
    
    # 3. Calculate Global L2 Non-conformity Score (weighted)
    # Ensure y_hat_k is list of floats
    if not isinstance(y_hat_k, list):
         y_hat_k = [float(y_hat_k)] * len(y_k) # Fallback if state corrupted
    
    # Handle length mismatch gracefully (e.g. if metrics changed)
    min_len = min(len(y_k), len(y_hat_k))
    weights = [1.0] * min_len
    l2_sum = 0.0
    for i in range(min_len):
        diff = y_k[i] - float(y_hat_k[i])
        l2_sum += weights[i] * (diff * diff)
    score_k = math.sqrt(l2_sum)
    
    # 4. Update Score History
    if 'scores_l1' not in state:
        state['scores_l1'] = []
    
    state['scores_l1'].append(score_k)
    if 'scores_dim' not in state:
        state['scores_dim'] = {}
    residuals = []
    for i in range(min_len):
        residuals.append(abs(y_k[i] - float(y_hat_k[i])))
    for i, name in enumerate(metric_names[:min_len]):
        arr = state['scores_dim'].get(name, [])
        arr.append(residuals[i])
        state['scores_dim'][name] = arr
    
    # Adaptive sliding window
    prev_avg = sum(state['scores_l1'][:-1]) / max(1, len(state['scores_l1']) - 1) if len(state['scores_l1']) > 1 else score_k
    window_len = int(state.get('wcp_window', 100))
    
    # Adaptive Thresholds
    spike_thr = float(state.get('wcp_spike_thr', 1.5))
    drop_thr = float(state.get('wcp_drop_thr', 0.5))
    win_inc = int(state.get('wcp_win_inc', 20))
    win_dec = int(state.get('wcp_win_dec', 10))
    win_max = int(state.get('wcp_win_max', 200))
    win_min = int(state.get('wcp_win_min', 50))

    if score_k > spike_thr * prev_avg:
        window_len = min(win_max, window_len + win_inc)
    elif score_k < drop_thr * prev_avg:
        window_len = max(win_min, window_len - win_dec)
    state['wcp_window'] = window_len
    if len(state['scores_l1']) > window_len:
        state['scores_l1'] = state['scores_l1'][-window_len:]
        
    # 5. RLS Update & Prediction for Next Step
    # We use independent RLS for each dimension for simplicity and stability.
    # Feature vector phi = [1, y_{k-1}] (AR(1) process)
    # Note: For the update step, we use the PREVIOUS observation as feature.
    # But to keep it stateless simple:
    # We maintain RLS state. The 'input' to RLS for this step is the PREVIOUS y.
    # Wait, RLS models y_k = theta * phi_k. 
    # If we model y_k based on y_{k-1}, then phi_k = [1, y_{k-1}].
    # We need to store y_{k-1} in state.
    
    last_y = state.get('last_y', y_k) # Default to current if missing
    phi = [1.0, 0.0] # Placeholder
    
    if 'rls_states' not in state:
        init = {}
        for i, name in enumerate(metric_names):
            prev_val = float(last_y[i]) if i < len(last_y) else 0.0
            init[name] = {
                'theta': [0.0, 1.0],
                'P': [[10.0, 0.0], [0.0, 10.0]]
            }
        state['rls_states'] = init
    
    y_hat_next = []
    
    for i, name in enumerate(metric_names):
        # Feature: [1, previous_value_of_this_metric]
        # Using AR(1) per dimension
        
        # Safe access to last_y
        if i < len(last_y):
            phi = [1.0, float(last_y[i])]
        else:
            phi = [1.0, 0.0]
        
        rls_data = state['rls_states'].get(name, {})
        rls = RLS.from_dict(rls_data, n_features=2)
        
        # Update RLS with CURRENT observation y_k[i]
        # The model predicted y_k[i] using last_y[i]
        current_val = float(y_k[i]) if i < len(y_k) else 0.0
        
        rls.update(phi, current_val)
        
        # Predict NEXT value y_{k+1}
        # Feature for next step is CURRENT observation y_k[i]
        phi_next = [1.0, current_val]
        pred_val = rls.predict(phi_next)
        y_hat_next.append(pred_val)
        
        # Save RLS state
        state['rls_states'][name] = rls.to_dict()
        
    # Store current y as last_y for next iteration
    state['last_y'] = y_k
    state['last_prediction'] = y_hat_next
    
    # 6. Weighted Quantile (Strict)
    # Adaptive rho (forgetting factor)
    # If volatility is high (large variance in recent scores), decrease rho to adapt faster.
    # If stable, increase rho to 0.99+ for better statistical power.
    
    scores = state['scores_l1']
    n = len(scores)

    recent_scores = scores[-10:] if n > 10 else scores
    if len(recent_scores) > 1:
        avg_s = sum(recent_scores) / len(recent_scores)
        var_s = sum((x - avg_s)**2 for x in recent_scores) / (len(recent_scores) - 1)
        
        # Heuristic: High variance -> lower rho
        if var_s > 100.0: # Arbitrary threshold for "high volatility"
            rho = 0.90
        elif var_s > 10.0:
            rho = 0.95
        else:
            rho = 0.99
    else:
        rho = 0.98
        
    # Weights w_i = rho^{n-i-1}
    
    if n == 0:
        uncertainty = 0.0
    else:
        weights = [rho**(n - 1 - i) for i in range(n)]
        target_mass = 1.0 - float(state.get('wcp_alpha', alpha))
        from .stats import weighted_quantile
        uncertainty = weighted_quantile(scores, weights, target_mass, inf_weight=1.0)
    unc_dict = {}
    for name in metric_names:
        arr = state['scores_dim'].get(name, [])
        m = len(arr)
        if m == 0:
            unc_dict[name] = 0.0
        else:
            w = [rho**(m - 1 - i) for i in range(m)]
            from .stats import weighted_quantile
            unc_dict[name] = weighted_quantile(arr, w, target_mass, inf_weight=1.0)

    # Dynamic alpha calibration
    current_alpha = float(state.get('wcp_alpha', alpha))
    risk_signal = max(metrics.get('timeout_rate', 0.0), metrics.get('error_rate', 0.0))
    mp = metrics.get('memory_pressure', 0.0)
    risk_thr = float(state.get('wcp_risk_thr', 0.05))
    mem_thr = float(state.get('wcp_mem_thr', 0.8))
    
    # Parametric steps
    alpha_min = float(state.get('wcp_alpha_min', 0.01))
    alpha_dec_factor = float(state.get('wcp_alpha_dec', 0.9))
    alpha_dec_step = float(state.get('wcp_alpha_step_dec', 0.005))
    alpha_inc_factor = float(state.get('wcp_alpha_inc', 1.05))
    alpha_inc_step = float(state.get('wcp_alpha_step_inc', 0.002))
    
    # If we are seeing timeouts or errors, we are under-provisioning.
    # We need to increase the confidence level (1-alpha), i.e., DECREASE alpha.
    # This will push the quantile higher (e.g. p90 -> p95), allocating more resources.
    if risk_signal > risk_thr:
        # Decrease alpha to be safer (min 0.01)
        current_alpha = max(alpha_min, current_alpha * alpha_dec_factor - alpha_dec_step)
    elif mp > mem_thr:
        # If memory pressure is high but NO timeouts yet, we might be close to limit.
        # But if we increase resources, we might OOM?
        # Actually usually 'memory_pressure' means system load.
        # If we are under pressure, maybe we should be safer too?
        # Or maybe we want to shed load (increase alpha)?
        # Let's assume 'memory_pressure' means we are running out of RAM, so we should be careful.
        # But if we reduce resources (increase alpha), we might cause OOM/Timeout?
        # Let's stick to risk_signal for safety adaptation.
        pass
    else:
        # If everything is stable (no risk), we can slowly relax (increase alpha) to save cost.
        target_alpha = alpha
        if current_alpha < target_alpha:
             current_alpha = min(target_alpha, current_alpha * alpha_inc_factor + alpha_inc_step)

    state['wcp_alpha'] = current_alpha

    # 7. Construct Result
    # Return structure matching what MPC expects, but strictly derived.
    # We return prediction vector and scalar uncertainty.
    
    # Map back to dict for readability/compatibility
    pred_dict = {
        'p90': y_hat_next[0],
        'timeout_rate': y_hat_next[1],
        'error_rate': y_hat_next[2],
        'memory_pressure': y_hat_next[3]
    }
    
    return pred_dict, unc_dict if unc_dict else uncertainty, {'scores_len': n, 'rls_updated': True}

def slow_loop_calibration(state, metrics):
    slo = float(metrics.get('slo_violation_rate', 0.0))
    util = float(metrics.get('cpu_util', 0.5))
    err = float(metrics.get('error_rate', 0.0))
    tout = float(metrics.get('timeout_rate', 0.0))
    waste = float(metrics.get('resource_waste_rate', 0.0))
    
    # Calibration Thresholds
    slo_stable = float(state.get('cal_slo_stable', 0.01))
    util_min = float(state.get('cal_util_min', 0.35))
    util_max = float(state.get('cal_util_max', 0.65))
    err_stable = float(state.get('cal_err_stable', 0.02))
    
    util_tense = float(state.get('cal_util_tense', 0.8))
    slo_tense = float(state.get('cal_slo_tense', 0.02))
    err_tense = float(state.get('cal_err_tense', 0.05))
    
    stable = (slo < slo_stable) and (util_min <= util <= util_max) and (err < err_stable) and (tout < err_stable)
    tense = (util > util_tense) or (slo > slo_tense) or (err > err_tense) or (tout > err_tense)
    cooldown = int(state.get('slow_cooldown', 0))
    if cooldown > 0:
        state['slow_cooldown'] = cooldown - 1
        return state
    sc = int(state.get('slow_stable_count', 0))
    tc = int(state.get('slow_tense_count', 0))
    if stable:
        sc += 1
        tc = 0
    elif tense:
        tc += 1
        sc = 0
    else:
        sc = max(0, sc - 1)
        tc = max(0, tc - 1)
    state['slow_stable_count'] = sc
    state['slow_tense_count'] = tc
    alpha_val = float(state.get('wcp_alpha', 0.1))
    window_len = int(state.get('wcp_window', 100))
    eta = float(state.get('sp_eta', 0.05))
    rho = float(state.get('sp_rho', 0.1))
    gamma = float(state.get('gamma', 0.1))
    u_eta = float(state.get('u_eta', 0.05))
    u_max_delta = float(state.get('u_max_delta', 0.15))
    trend = state.get('trend_state', 'flat')
    
    stable_thr = int(state.get('cal_stable_thr', 3))
    tense_thr = int(state.get('cal_tense_thr', 2))

    if sc >= stable_thr:
        # Relax (stable)
        alpha_min = float(state.get('cal_alpha_min', 0.05))
        alpha_dec = float(state.get('cal_alpha_dec', 0.97))
        alpha_step = float(state.get('cal_alpha_step', 0.003))
        
        win_max = int(state.get('wcp_win_max', 200))
        win_inc = int(state.get('cal_win_inc', 10))
        
        eta_min = float(state.get('opt_eta_min', 0.01))
        eta_dec = float(state.get('cal_eta_dec', 0.9))
        
        rho_max = float(state.get('sp_rho_max', 0.5))
        rho_inc = float(state.get('cal_rho_inc', 0.05))
        
        gamma_min = float(state.get('opt_gamma_min', 0.05))
        gamma_dec = float(state.get('cal_gamma_dec', 0.9))
        
        u_eta_min = float(state.get('u_eta_min', 0.01))
        u_eta_dec = float(state.get('cal_u_eta_dec', 0.9))
        
        u_max_delta_max = float(state.get('u_max_delta_max', 0.2))
        u_max_delta_inc = float(state.get('cal_u_max_delta_inc', 0.02))

        alpha_val = max(alpha_min, alpha_val * alpha_dec - alpha_step)
        window_len = min(win_max, window_len + win_inc)
        eta = max(eta_min, eta * eta_dec)
        rho = min(rho_max, rho + rho_inc)
        gamma = max(gamma_min, gamma * gamma_dec)
        u_eta = max(u_eta_min, u_eta * u_eta_dec)
        u_max_delta = min(u_max_delta_max, u_max_delta + u_max_delta_inc)
        state['slow_cooldown'] = stable_thr
        
    elif tc >= tense_thr:
        # Tighten (tense)
        alpha_max = float(state.get('cal_alpha_max', 0.3))
        alpha_inc = float(state.get('cal_alpha_inc', 1.05))
        alpha_step_inc = float(state.get('cal_alpha_step_inc', 0.005))
        
        win_min = int(state.get('wcp_win_min', 50))
        win_dec = int(state.get('cal_win_dec', 10))
        
        eta_max = float(state.get('opt_eta_max', 0.2))
        eta_inc = float(state.get('cal_eta_inc', 1.2))
        
        rho_min = float(state.get('sp_rho_min', 0.05))
        rho_dec = float(state.get('cal_rho_dec', 0.8))
        
        gamma_max = float(state.get('opt_gamma_max', 0.5))
        gamma_inc = float(state.get('cal_gamma_inc', 1.2))
        
        u_eta_max = float(state.get('u_eta_max', 0.2))
        u_eta_inc = float(state.get('cal_u_eta_inc', 1.2))
        
        u_max_delta_min = float(state.get('u_max_delta_min', 0.05))
        u_max_delta_dec = float(state.get('cal_u_max_delta_dec', 0.8))

        alpha_val = min(alpha_max, alpha_val * alpha_inc + alpha_step_inc)
        window_len = max(win_min, window_len - win_dec)
        eta = min(eta_max, eta * eta_inc)
        rho = max(rho_min, rho * rho_dec)
        gamma = min(gamma_max, gamma * gamma_inc)
        u_eta = min(u_eta_max, u_eta * u_eta_inc)
        u_max_delta = max(u_max_delta_min, u_max_delta * u_max_delta_dec)
        state['slow_cooldown'] = tense_thr
        
    if trend == 'up':
        # Trend Up -> Tighten slightly
        alpha_max = float(state.get('cal_alpha_max', 0.3))
        alpha_trend_inc = float(state.get('cal_alpha_trend_inc', 1.03))
        alpha_trend_step = float(state.get('cal_alpha_trend_step', 0.002))
        eta_trend_inc = float(state.get('cal_eta_trend_inc', 1.1))
        u_eta_trend_inc = float(state.get('cal_u_eta_trend_inc', 1.1))
        
        alpha_val = min(alpha_max, alpha_val * alpha_trend_inc + alpha_trend_step)
        eta = min(0.2, eta * eta_trend_inc)
        u_eta = min(0.2, u_eta * u_eta_trend_inc)
        
    elif trend == 'down':
        # Trend Down -> Relax slightly
        alpha_min = float(state.get('cal_alpha_min', 0.05))
        alpha_trend_dec = float(state.get('cal_alpha_trend_dec', 0.98))
        alpha_trend_step_dec = float(state.get('cal_alpha_trend_step_dec', 0.002))
        eta_trend_dec = float(state.get('cal_eta_trend_dec', 0.95))
        u_eta_trend_dec = float(state.get('cal_u_eta_trend_dec', 0.95))
        
        alpha_val = max(alpha_min, alpha_val * alpha_trend_dec - alpha_trend_step_dec)
        eta = max(0.01, eta * eta_trend_dec)
        u_eta = max(0.01, u_eta * u_eta_trend_dec)

    if waste > 0.15:
        # Wasted resources? Tune down w1 (track), tune up w2 (waste penalty)
        # Actually if we are wasting, we are allocating too much.
        # We should increase w2 (waste penalty).
        w2_step = float(state.get('cal_w2_step', 2.0))
        w2_max = float(state.get('opt_w2_max', 50.0))
        w2 = float(state.get('opt_w2', 5.0))
        state['opt_w2'] = min(w2_max, w2 + w2_step)
    state['wcp_alpha'] = alpha_val
    state['wcp_window'] = window_len
    state['sp_eta'] = eta
    state['sp_rho'] = rho
    state['gamma'] = gamma
    state['u_eta'] = u_eta
    state['u_max_delta'] = u_max_delta
    return state

def detect_trend(state, metrics):
    val = float(metrics.get('p90', 0.0))
    ewma = float(state.get('p90_ewma', val))
    alpha = float(state.get('trend_alpha', 0.2))
    ewma = (1 - alpha) * ewma + alpha * val
    prev = float(state.get('p90_prev', val))
    slope = val - prev
    state['p90_prev'] = val
    state['p90_ewma'] = ewma
    thr = float(state.get('trend_thr', 5.0))
    trend_state = 'flat'
    if slope > thr:
        trend_state = 'up'
    elif slope < -thr:
        trend_state = 'down'
    state['trend_state'] = trend_state
    return state
