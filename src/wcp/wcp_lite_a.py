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

# --- 3. WCP Lite A: EWMA-based Uncertainty ---

# --- Constants for Cold Start Safety Net ---
WARMUP_STEPS = 10
SAFETY_UNCERTAINTY = 500.0  # Large enough to cover initial fluctuations

def wcp_lite_a_update(state, metrics, alpha=0.1):
    """
    WCP Lite A: Lightweight Adaptive Conformal Prediction (Chebyshev EWMA).
    
    Instead of assuming a Normal distribution, we use Chebyshev's Inequality 
    to determine the multiplier K, making this method distribution-free (though conservative).
    
    Uncertainty = Mean + K * StdDev
    Where K = 1 / sqrt(alpha)
    
    ASYNC OPTIMIZATION NOTE:
    This function is pure computation. It modifies the `state` dictionary in-place
    and returns it. The caller (Lambda Handler) should:
    1. Call this function to get prediction and uncertainty.
    2. Return the response to the user immediately.
    3. Asynchronously write the updated `state` to DynamoDB (e.g., using 
       context.callbackWaitsForEmptyEventLoop = false).
    
    Args:
        state (dict): Persistent state (RLS params, ewma_mean, ewma_var, last_prediction).
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
    # Handle length mismatch (e.g. initial state empty list)
    if len(y_hat_k) != len(y_k):
        y_hat_k = [0.0] * len(y_k)

    # 3. Calculate Global L1 Non-conformity Score
    score_k = sum(abs(y_k[i] - float(y_hat_k[i])) for i in range(len(y_k)))
    
    # 4. Update EWMA Statistics (Mean and Variance of Score)
    # Using a fixed learning rate for statistics tracking
    eta = 0.1 
    
    mu = state.get('ewma_mean', 0.0)
    var = state.get('ewma_var', 0.0)
    
    # Initialize if first run (var is 0, but score might be non-zero)
    if 'ewma_mean' not in state:
        mu = score_k
        var = 0.0 # Initial variance
    else:
        # Standard EWMA updates
        diff = score_k - mu
        incr = eta * diff
        mu = mu + incr
        # Update variance: (1-eta)*var + eta*(diff * (score - new_mu))
        # Or simpler approx: var = (1-eta)*var + eta*(diff**2)
        # We use the standard Welford-like approx for EWMA variance
        var = (1 - eta) * var + eta * (diff * (score_k - mu))

    state['ewma_mean'] = mu
    state['ewma_var'] = var
    
    # 5. Calculate Uncertainty
    # We use Chebyshev's Inequality to be Distribution-Free.
    # P(|X-mu| >= k*sigma) <= 1/k^2
    # We want coverage probability >= 1 - alpha
    # So error probability <= alpha
    # 1/k^2 = alpha  =>  k = 1 / sqrt(alpha)
    
    if alpha <= 0: alpha = 0.001 # Prevent division by zero
    K_factor = 1.0 / math.sqrt(alpha)
    
    # For alpha=0.1, K = 3.16
    # This is more conservative than Normal (1.28) but rigorous.
    
    std_dev = math.sqrt(var) if var > 0 else 0.0
    
    # --- Cold Start Safety Net: Output Logic ---
    # If we are in the warmup phase, we don't trust the EWMA stats yet.
    # Return a safe, large uncertainty.
    if sample_count <= WARMUP_STEPS:
        uncertainty = SAFETY_UNCERTAINTY
        # Optional: Blend with calculated value if needed, but safe is better.
    else:
        uncertainty = mu + K_factor * std_dev
    
    # 6. RLS Update & Prediction for Next Step
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
    
    # 7. Return Result
    pred_dict = {
        'p90': y_hat_next[0],
        'timeout_rate': y_hat_next[1],
        'error_rate': y_hat_next[2],
        'memory_pressure': y_hat_next[3]
    }
    
    return pred_dict, uncertainty, {
        'mode': 'lite_a_ewma', 
        'mean': mu, 
        'std': std_dev, 
        'score': score_k
    }
