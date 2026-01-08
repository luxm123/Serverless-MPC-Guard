import math
import random

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

# --- 3. WCP Lite B: Streaming Quantile (Frugal) ---

# --- Constants for Cold Start Safety Net ---
WARMUP_STEPS = 10
SAFETY_UNCERTAINTY = 500.0

def wcp_lite_b_update(state, metrics, alpha=0.1):
    """
    WCP Lite B: Streaming Quantile Tracking (Frugal-Style).
    
    Instead of storing history, we directly update the estimate of the (1-alpha) quantile.
    
    Update Rule:
    If score > quantile: quantile += step * (1-alpha)
    If score < quantile: quantile -= step * alpha
    
    (This ensures equilibrium at P(score < Q) = 1-alpha)
    
    ASYNC OPTIMIZATION NOTE:
    This function is pure computation. It modifies the `state` dictionary in-place
    and returns it. The caller (Lambda Handler) should:
    1. Call this function to get prediction and uncertainty.
    2. Return the response to the user immediately.
    3. Asynchronously write the updated `state` to DynamoDB (e.g., using 
       context.callbackWaitsForEmptyEventLoop = false).

    Args:
        state (dict): Persistent state (RLS params, quantile_est, step_size, last_prediction).
        metrics (dict): Current observations.
        alpha (float): Target error rate.
        
    Returns:
        tuple: (prediction_next, uncertainty_radius, debug_info)
    """
    
    # Define metric vector order
    metric_names = ['p90', 'timeout_rate', 'error_rate', 'memory_pressure']
    
    # --- Cold Start Safety Net: Counter ---
    sample_count = state.get('sample_count', 0)
    sample_count += 1
    state['sample_count'] = sample_count

    # 1. Extract Current Observation y_k
    y_k = [
        float(metrics.get('p90', 0.0)),
        float(metrics.get('timeout_rate', 0.0)),
        float(metrics.get('error_rate', 0.0)),
        float(metrics.get('memory_pressure', 0.0))
    ]
    
    # 2. Retrieve last prediction (y_hat_k)
    y_hat_k = state.get('last_prediction', [])
    if not y_hat_k:
        y_hat_k = [0.0] * len(y_k)
    if len(y_hat_k) != len(y_k):
        y_hat_k = [0.0] * len(y_k)

    # 3. Calculate Global L1 Non-conformity Score
    score_k = sum(abs(y_k[i] - float(y_hat_k[i])) for i in range(len(y_k)))
    
    # 4. Update Streaming Quantile
    q_est = state.get('quantile_est', 0.0)
    
    # If first run, initialize q_est with current score
    if 'quantile_est' not in state:
        q_est = score_k * 1.5 # Start slightly pessimistic
    
    # Target quantile tau
    tau = 1.0 - alpha
    
    # Adaptive Step Size (Simple logic: 1% of current estimate or min 0.01)
    # This makes it scale-invariant.
    step = max(0.01, q_est * 0.05)
    
    # Gradient-based update (Deterministic)
    # Logic:
    # If we are UNDER-estimating (score > q), we need to move UP strongly.
    # If we are OVER-estimating (score < q), we need to move DOWN gently.
    # The ratio of up/down moves should match tau/(1-tau).
    
    if score_k > q_est:
        # Increase q
        q_est += step * tau
    else:
        # Decrease q
        q_est -= step * (1.0 - tau)
        
    # Ensure non-negative
    if q_est < 0:
        q_est = 0.0
        
    state['quantile_est'] = q_est
    
    # --- Cold Start Safety Net: Output Logic ---
    if sample_count <= WARMUP_STEPS:
        uncertainty = SAFETY_UNCERTAINTY
    else:
        uncertainty = q_est
    
    # 5. RLS Update & Prediction for Next Step
    # (Same RLS logic as Strict WCP)
    
    last_y = state.get('last_y', y_k)
    if 'rls_states' not in state:
        state['rls_states'] = {}
    
    y_hat_next = []
    
    for i, name in enumerate(metric_names):
        if i < len(last_y):
            phi = [1.0, float(last_y[i])]
        else:
            phi = [1.0, 0.0]
        
        rls_data = state['rls_states'].get(name, {})
        rls = RLS.from_dict(rls_data, n_features=2)
        
        current_val = float(y_k[i]) if i < len(y_k) else 0.0
        rls.update(phi, current_val)
        
        phi_next = [1.0, current_val]
        pred_val = rls.predict(phi_next)
        y_hat_next.append(pred_val)
        
        state['rls_states'][name] = rls.to_dict()
        
    state['last_y'] = y_k
    state['last_prediction'] = y_hat_next
    
    # 6. Return Result
    pred_dict = {
        'p90': y_hat_next[0],
        'timeout_rate': y_hat_next[1],
        'error_rate': y_hat_next[2],
        'memory_pressure': y_hat_next[3]
    }
    
    return pred_dict, uncertainty, {
        'mode': 'lite_b_streaming', 
        'quantile_est': q_est, 
        'score': score_k
    }
