def compute_risk(bounds, targets):
    # Use upper bound for risk calculation
    # bounds structure: {'metric': {'lower': val, 'upper': val}}
    
    # Helper to safe access upper bound
    def get_upper(metric):
        b = bounds.get(metric)
        if isinstance(b, dict):
            return b.get('upper', 0.0)
        return float(b) # Fallback for backward compatibility

    n_p90 = max(0.0, (get_upper('latency') - targets['latency']) / max(targets['latency'], 1.0))
    n_timeout = max(0.0, (get_upper('timeout') - targets['timeout']) / max(targets['timeout'], 1e-6))
    n_error = max(0.0, (get_upper('error') - targets['error']) / max(targets['error'], 1e-6))
    n_mem = max(0.0, (get_upper('memory') - targets['memory']) / max(targets['memory'], 1.0))
    
    composite = max(n_p90, n_timeout, n_error, n_mem)
    risks = {
        'latency': n_p90,
        'timeout': n_timeout,
        'error': n_error,
        'memory': n_mem,
        'max': composite
    }
    return risks, composite
