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

def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def logit(p, eps=1e-6):
    p = clamp01(float(p))
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))

def sigmoid(z):
    if z > 50.0:
        return 1.0
    if z < -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def safe_get_float(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)

def build_phi(concurrency, cpu, backlog, service_time_ms, task_type='image_processing'):
     """
     构建 RLS 模型的特征向量，包含任务类型的 One-Hot 编码和物理积压维度。
     """
     c = float(concurrency)
     u_inv = 1.0 / (float(cpu) + 0.05)
     b = float(backlog) # 引入真实的物理积压特征
     
     # 任务类型的 One-Hot 编码
     is_image = 1.0 if task_type == 'image_processing' else 0.0
     is_pyaes = 1.0 if task_type == 'pyaes' else 0.0
     is_linpack = 1.0 if task_type == 'linpack' else 0.0
     is_model = 1.0 if task_type == 'model_serving' else 0.0
     
     # 11 维特征：[1, c, u^-1, c/u, c^2, u^-2, backlog, image, pyaes, linpack, model]
     return [1.0, c, u_inv, c * u_inv, c**2, u_inv**2, b, is_image, is_pyaes, is_linpack, is_model]

# --- 2. RLS Class (Parameterized Dynamics Model) ---

class RLS:
    """
    Recursive Least Squares filter.
    Model: y = theta^T * phi
    """
    def __init__(self, n_features, lambda_factor=0.98, delta=10000.0):
        """
        初始化 RLS。
        delta 设为 10000.0 以实现高增益冷启动，确保模型在没有预热的情况下也能在前几步快速进化。
        """
        self.n = n_features
        self.lambda_factor = lambda_factor
        self.theta = [0.0] * n_features
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
        # Denominator protection: avoid division by zero or negative values
        denom = max(1e-9, self.lambda_factor + phi_P_phi)
        
        k = [val / denom for val in P_phi]
        
        # 2. Update parameters: theta = theta + k * (y - phi^T * theta)
        prediction = vec_dot(phi, self.theta)
        error = y - prediction
        
        # Huber-like error clipping (optional safety): prevents outliers from blowing up parameters
        # error = max(-500.0, min(500.0, error)) 
        
        theta_correction = [val * error for val in k]
        self.theta = vec_add(self.theta, theta_correction)
        
        # 3. Update covariance: P = (P - k * phi^T * P) / lambda
        # Term: k * (phi^T * P) = k * (P_phi)^T (since P is symmetric)
        k_P_phi_T = vec_outer(k, P_phi)
        P_numer = mat_add(self.P, mat_scale(k_P_phi_T, -1.0))
        new_P = mat_scale(P_numer, 1.0 / self.lambda_factor)
        
        # Numerical stability: enforce symmetry P = 0.5 * (P + P^T)
        rows = len(new_P)
        for i in range(rows):
            for j in range(i + 1, rows):
                avg = 0.5 * (new_P[i][j] + new_P[j][i])
                new_P[i][j] = avg
                new_P[j][i] = avg
        self.P = new_P
        
        return prediction

    def predict(self, phi):
        return vec_dot(phi, self.theta)

    # --- Serialization Helpers ---
    def to_dict(self):
        return {'theta': self.theta, 'P': self.P}

    @staticmethod
    def from_dict(d, n_features=6):
        rls = RLS(n_features)
        if d and 'theta' in d and 'P' in d:
            rls.theta = [float(x) for x in d['theta']]
            # Handle P parsing carefully if it comes from DynamoDB JSON
            raw_P = d['P']
            rls.P = [[float(val) for val in row] for row in raw_P]
        return rls

# --- 3. Main WCP Logic (Strict Implementation) ---

def wcp_update(state, p90_latency, concurrency, cpu, backlog, service_time_ms, task_type='image_processing', alpha=0.1):
    """
    Strict WCP Update for P90 Latency.

    Args:
        state (dict): Persistent state (RLS params, score history, last_prediction).
        p90_latency (float): The current observed P90 latency.
        concurrency (float): Current request concurrency.
        cpu (float): Current CPU limit.
        backlog (float): Current task backlog.
        service_time_ms (float): Current average service time.
        alpha (float): Target error rate.

    Returns:
        tuple: (prediction_next, uncertainty_radius, debug_info)
    """
    y_k = float(p90_latency)
    y_hat_k = float(state.get('last_prediction', 0.0))

    # Non-conformity score is the absolute error
    # v22 ROOT CAUSE FIX: Error Clipping. 
    # 不允许冷启动产生的巨大误差 (如 300ms) 彻底破坏不确定性边界。
    # 我们将单次误差对 Margin 的贡献截断在 50ms。
    raw_score = abs(y_k - y_hat_k)
    score_k = min(50.0, raw_score) 
    
    if 'scores' not in state:
        state['scores'] = []
    state['scores'].append(score_k)

    # --- Adaptive Window for Scores ---
    prev_avg = sum(state['scores'][:-1]) / max(1, len(state['scores']) - 1) if len(state['scores']) > 1 else score_k
    window_len = int(state.get('wcp_window', 100))
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
    if len(state['scores']) > window_len:
        state['scores'] = state['scores'][-window_len:]

    # --- RLS Update and Prediction ---
    # Model: y = f(concurrency, cpu)
    # 构建特征向量 (包含 One-Hot 编码)
    phi = build_phi(concurrency, cpu, backlog, service_time_ms, task_type=task_type)
    feat_len = len(phi)

    # 初始化或恢复 RLS 模型
    rls_data = state.get('rls_state', {})
    rls = RLS.from_dict(rls_data, n_features=feat_len)
    if len(rls.theta) != feat_len or len(rls.P) != feat_len:
        # 修正：冷启动回退也必须使用高增益 delta=10000.0
        rls = RLS(feat_len, lambda_factor=0.98, delta=10000.0)
    
    # Update RLS with current observation
    rls.update(phi, y_k)
    
    # Predict next step using the updated model
    # Note: In the orchestrator, we will call this with future_concurrency
    y_hat_next = rls.predict(phi) # Prediction for current state (for next WCP score)

    # Persist state
    state['rls_state'] = rls.to_dict()
    state['last_y'] = y_k
    state['last_cpu'] = cpu
    state['last_prediction'] = y_hat_next

    # --- Weighted Quantile Calculation ---
    scores = state['scores']
    q_index = 0
    sorted_scores_len = 0
    if len(scores) < 10:
        # 降低冷启动安全边界，从 250ms 降至 30ms，避免初始阶段 Alloc 锁死在 1.0
        uncertainty = 30.0
    else:
        sorted_scores = sorted(scores, reverse=True)
        q_index = math.ceil((len(scores) + 1) * (1 - alpha)) - 1
        q_index = max(0, min(len(scores) - 1, q_index))
        uncertainty = sorted_scores[q_index]
        sorted_scores_len = len(sorted_scores)

    debug_info = {
        'score_k': score_k,
        'wcp_window': window_len,
        'quantile_idx': q_index,
        'sorted_scores_len': sorted_scores_len,
        'rls_theta': rls.theta
    }

    pred_dict = {
        'p90': float(y_hat_next),
        'timeout_rate': 0.0,
        'error_rate': 0.0,
        'memory_pressure': 0.0
    }

    return pred_dict, uncertainty, debug_info

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
