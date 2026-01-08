import math
from src.wcp.wcp_update import RLS

# --- Constants for Cold Start Safety Net ---
WARMUP_STEPS = 10
SAFETY_UNCERTAINTY = 500.0

def wcp_simple_update(state, metrics, alpha=0.1):
    """
    WCP Simple: RLS Prediction + Unweighted Quantile.
    
    Uses standard (non-weighted) quantile of the non-conformity score history.
    
    ASYNC OPTIMIZATION NOTE:
    The caller should return the prediction immediately and write state asynchronously.
    """
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
    if not y_hat_k or len(y_hat_k) != len(y_k):
        y_hat_k = [0.0] * len(y_k)
    
    # 3. Calculate Global L1 Non-conformity Score
    min_len = min(len(y_k), len(y_hat_k))
    score_k = sum(abs(y_k[i] - float(y_hat_k[i])) for i in range(min_len))
    
    # 4. Update Score History
    if 'scores_l1' not in state:
        state['scores_l1'] = []
    
    state['scores_l1'].append(score_k)
    
    # Keep history manageable (Standard WCP length)
    max_history = 150
    if len(state['scores_l1']) > max_history:
        state['scores_l1'] = state['scores_l1'][-max_history:]
        
    # 5. RLS Update & Prediction
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
    
    # 6. Calculate Uncertainty (Unweighted Quantile)
    scores = sorted(state['scores_l1'])
    n = len(scores)
    
    if n == 0:
        uncertainty = 0.0
    else:
        # Standard quantile for 1-alpha
        # Index k = ceil((1-alpha)*(n+1)) - 1
        k = math.ceil((1.0 - alpha) * (n + 1)) - 1
        k = max(0, min(k, n - 1))
        uncertainty = scores[k]
    
    # --- Cold Start Safety Net: Output Logic ---
    if sample_count <= WARMUP_STEPS:
        # Override with safety uncertainty if in warmup
        uncertainty = max(uncertainty, SAFETY_UNCERTAINTY)
    
    # 7. Return Result
    pred_dict = {
        'p90': y_hat_next[0],
        'timeout_rate': y_hat_next[1],
        'error_rate': y_hat_next[2],
        'memory_pressure': y_hat_next[3]
    }
    
    return pred_dict, uncertainty, {'mode': 'simple', 'score': score_k}
