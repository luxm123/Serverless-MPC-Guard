from src.wcp.wcp_update import RLS

def wcp_baseline_update(state, metrics, alpha=0.1):
    """
    WCP Baseline: RLS Prediction Only (Uncertainty = 0).
    
    This method only predicts the next values using RLS but provides
    zero uncertainty radius, serving as a lower-bound performance baseline.
    
    ASYNC OPTIMIZATION NOTE:
    The caller should return the prediction immediately and write state asynchronously.
    """
    metric_names = ['p90', 'timeout_rate', 'error_rate', 'memory_pressure']
    
    # 1. Extract Current Observation y_k
    y_k = [
        float(metrics.get('p90', 0.0)),
        float(metrics.get('timeout_rate', 0.0)),
        float(metrics.get('error_rate', 0.0)),
        float(metrics.get('memory_pressure', 0.0))
    ]
    
    # 2. RLS Update & Prediction
    last_y = state.get('last_y', y_k)
    if 'rls_states' not in state:
        state['rls_states'] = {}
        
    y_hat_next = []
    
    for i, name in enumerate(metric_names):
        # Feature: [1, previous_value]
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
    
    # 3. Return Result
    pred_dict = {
        'p90': y_hat_next[0],
        'timeout_rate': y_hat_next[1],
        'error_rate': y_hat_next[2],
        'memory_pressure': y_hat_next[3]
    }
    
    return pred_dict, 0.0, {'mode': 'baseline'}
