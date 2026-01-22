def clamp(val, lo, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

def update_shadow_price(state, metrics, u):
    """
    Shadow Price Calculation via Dual Gradient Descent.
    
    The shadow price (lambda) reflects the marginal cost of resource constraints.
    It is updated iteratively based on the violation of the resource capacity constraint.
    
    Optimization Problem:
      minimize J(u)  s.t.  g(u) <= B
    Lagrangian:
      L(u, lambda) = J(u) + lambda * (g(u) - B)
    Dual Update:
      lambda_{k+1} = [lambda_k + eta * (g(u_k) - B)]^+
      
    Here:
      g(u) is approximated by current system load (cpu, backlog, etc.)
      B is the target capacity (e.g., 0.8 utilization)
    """
    eta = float(state.get('sp_eta', 0.05))
    rho = float(state.get('sp_rho', 0.1))
    lam_max = float(state.get('sp_lambda_max', 100.0))
    lam = float(state.get('shadow_price', 0.0))
    mu = float(state.get('sp_mu', 0.8))  # momentum
    cpu = float(metrics.get('cpu_util', 0.5))
    backlog = float(metrics.get('queue_backlog', 0.0))
    timeout = float(metrics.get('timeout_rate', 0.0))
    error = float(metrics.get('error_rate', 0.0))
    memp = float(metrics.get('memory_pressure', 0.0))
    slo = float(metrics.get('slo_violation_rate', 0.0))
    # CRITICAL FIX: Tolerance for Noise
    # If SLO violation is very small (e.g., < 1%), treat it as 0.0 to allow Price to decay.
    # This prevents "Price Creep" where 0.75% violation keeps Price high forever, killing Q2.
    if slo < 0.01:
        slo = 0.0
        
    # Capacity baseline
    B = float(state.get('sp_B', 0.8))
    # Dual signals: load side + risk side
    load = max(cpu, min(1.0, memp))
    risk = max(timeout, error, slo)
    queue_term = min(1.0, backlog / 1000.0)
    # Weights (can be auto-calibrated by slow loop)
    kr = float(state.get('sp_kr', 0.5))
    kq = float(state.get('sp_kq', 0.2))
    ku = float(state.get('sp_ku', 0.2))
    u_target = float(state.get('sp_u_target', 0.8))
    # Slow-loop EMA for adaptive weights
    alpha_ema = float(state.get('sp_alpha_ema', 0.1))
    risk_ema = float(state.get('sp_risk_ema', risk))
    queue_ema = float(state.get('sp_queue_ema', queue_term))
    risk_ema = (1.0 - alpha_ema) * risk_ema + alpha_ema * risk
    queue_ema = (1.0 - alpha_ema) * queue_ema + alpha_ema * queue_term
    
    # Adaptive kr/kq parameters
    risk_thr = float(state.get('sp_risk_thr', 0.03))
    kr_inc_fac = float(state.get('sp_kr_inc_fac', 1.05))
    kr_inc_step = float(state.get('sp_kr_inc_step', 0.01))
    kr_dec_fac = float(state.get('sp_kr_dec_fac', 0.98))
    kr_dec_step = float(state.get('sp_kr_dec_step', 0.005))
    
    queue_thr = float(state.get('sp_queue_thr', 0.3))
    kq_inc_fac = float(state.get('sp_kq_inc_fac', 1.05))
    kq_inc_step = float(state.get('sp_kq_inc_step', 0.005))
    kq_dec_fac = float(state.get('sp_kq_dec_fac', 0.98))
    kq_dec_step = float(state.get('sp_kq_dec_step', 0.003))

    if risk_ema > risk_thr:
        kr = min(1.0, kr * kr_inc_fac + kr_inc_step)
    else:
        kr = max(0.2, kr * kr_dec_fac - kr_dec_step)
    if queue_ema > queue_thr:
        kq = min(0.5, kq * kq_inc_fac + kq_inc_step)
    else:
        kq = max(0.1, kq * kq_dec_fac - kq_dec_step)
    timeout_target = float(state.get('sp_timeout_target', 0.0))
    error_target = float(state.get('sp_error_target', 0.0))
    mem_target = float(state.get('sp_mem_target', 0.8))
    grad_lat = (load - B) + kr * risk + kq * queue_term + ku * (u - u_target)
    grad_to = (timeout - timeout_target) + kq * queue_term + ku * (u - u_target)
    grad_er = (error - error_target) + kq * queue_term + ku * (u - u_target)
    grad_mem = (memp - mem_target) + kr * risk + ku * (u - u_target)
    m_lat = float(state.get('sp_m_lat', 0.0))
    m_to = float(state.get('sp_m_to', 0.0))
    m_er = float(state.get('sp_m_er', 0.0))
    m_mem = float(state.get('sp_m_mem', 0.0))
    m_lat = mu * m_lat + (1.0 - mu) * grad_lat
    m_to = mu * m_to + (1.0 - mu) * grad_to
    m_er = mu * m_er + (1.0 - mu) * grad_er
    m_mem = mu * m_mem + (1.0 - mu) * grad_mem
    # Hysteresis: adapt eta based on regime
    mode = state.get('sp_mode', 'normal')
    
    # Hysteresis Thresholds
    load_high_thr = float(state.get('sp_load_high', 0.88))
    risk_high_thr = float(state.get('sp_risk_high', 0.04))
    load_exit_thr = float(state.get('sp_load_exit', 0.75))
    risk_exit_thr = float(state.get('sp_risk_exit', 0.02))
    
    enter_high = (load > load_high_thr) or (risk > risk_high_thr)
    exit_high = (load < load_exit_thr) and (risk < risk_exit_thr)
    
    if mode == 'normal' and enter_high:
        mode = 'high'
    elif mode == 'high' and exit_high:
        mode = 'normal'
    state['sp_mode'] = mode
    
    eta_norm_dec = float(state.get('sp_eta_norm_dec', 0.9))
    eta_high_inc = float(state.get('sp_eta_high_inc', 1.2))
    eta_min = float(state.get('sp_eta_min', 0.01))
    eta_max = float(state.get('sp_eta_max', 0.2))

    if mode == 'high':
        eta = min(eta_max, eta * eta_high_inc)
    else:
        eta = max(eta_min, eta * eta_norm_dec)
    lam_lat = float(state.get('sp_lambda_lat', lam))
    lam_to = float(state.get('sp_lambda_to', lam))
    lam_er = float(state.get('sp_lambda_er', lam))
    lam_mem = float(state.get('sp_lambda_mem', lam))
    lam_next_lat = clamp(lam_lat + eta * m_lat, 0.0, lam_max)
    lam_next_to = clamp(lam_to + eta * m_to, 0.0, lam_max)
    lam_next_er = clamp(lam_er + eta * m_er, 0.0, lam_max)
    lam_next_mem = clamp(lam_mem + eta * m_mem, 0.0, lam_max)
    lam_smooth_lat = clamp((1.0 - rho) * lam_lat + rho * lam_next_lat, 0.0, lam_max)
    lam_smooth_to = clamp((1.0 - rho) * lam_to + rho * lam_next_to, 0.0, lam_max)
    lam_smooth_er = clamp((1.0 - rho) * lam_er + rho * lam_next_er, 0.0, lam_max)
    lam_smooth_mem = clamp((1.0 - rho) * lam_mem + rho * lam_next_mem, 0.0, lam_max)
    price_vec = {
        'latency': lam_smooth_lat,
        'timeout': lam_smooth_to,
        'error': lam_smooth_er,
        'memory': lam_smooth_mem
    }
    lam_smooth = max(price_vec.values())
    decay = float(state.get('sp_decay', 0.01))
    low_streak = int(state.get('sp_low_streak', 0))
    
    # Low regime definition
    risk_low_thr = float(state.get('sp_risk_low', 0.01))
    load_low_thr = float(state.get('sp_load_low', 0.6))
    queue_low_thr = float(state.get('sp_queue_low', 0.1))
    streak_thr = int(state.get('sp_streak_thr', 3))
    
    low_regime = (mode == 'normal') and (risk < risk_low_thr) and (load < load_low_thr) and (queue_term < queue_low_thr)
    if low_regime:
        low_streak += 1
    else:
        low_streak = 0
    if low_regime and low_streak >= streak_thr:
        lam_smooth = lam_smooth * 0.5
    elif low_regime:
        lam_smooth = lam_smooth * (1.0 - decay)
    state['sp_low_streak'] = low_streak
    # Persist
    state['shadow_price'] = lam_smooth
    state['shadow_price_vector'] = price_vec
    state['sp_eta'] = eta
    state['sp_rho'] = rho
    state['sp_lambda_max'] = lam_max
    state['sp_mu'] = mu
    state['sp_m_lat'] = m_lat
    state['sp_m_to'] = m_to
    state['sp_m_er'] = m_er
    state['sp_m_mem'] = m_mem
    state['sp_kr'] = kr
    state['sp_kq'] = kq
    state['sp_ku'] = ku
    state['sp_u_target'] = u_target
    state['sp_risk_ema'] = risk_ema
    state['sp_queue_ema'] = queue_ema
    state['sp_alpha_ema'] = alpha_ema
    state['sp_lambda_lat'] = lam_smooth_lat
    state['sp_lambda_to'] = lam_smooth_to
    state['sp_lambda_er'] = lam_smooth_er
    state['sp_lambda_mem'] = lam_smooth_mem
    state['sp_timeout_target'] = timeout_target
    state['sp_error_target'] = error_target
    state['sp_mem_target'] = mem_target
    return lam_smooth, {
        'B': B,
        'load': load,
        'risk': risk,
        'queue': queue_term,
        'price_vector': price_vec
    }
