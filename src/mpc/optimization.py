import math

def get_optimal_allocation(state, params, ref_latency, slo_limit, pred_upper, pred_lower):
    """
    Calculates the optimal resource allocation using a gradient-based approach.

    Args:
        state (dict): The current system state.
        params (dict): The MPC parameters.
        ref_latency (float): The reference latency.
        slo_limit (float): The SLO latency limit.
        pred_upper (float): The predicted upper bound of latency.
        pred_lower (float): The predicted lower bound of latency.

    Returns:
        float: The calculated optimal allocation.
    """
    # 1. 梯度计算
    # 1.1. 资源浪费梯度 (Utility/Waste Gradient)
    # 当预测延迟远低于参考延迟时，倾向于降低资源，节省成本
    w1 = float(params.get('w1', 0.5))
    grad_waste = w1 * (pred_upper - ref_latency)

    # 1.2. SLO风险梯度 (Risk Gradient)
    # 当预测延迟接近甚至超过SLO时，倾向于增加资源，保证服务质量
    w2 = float(params.get('w2', 0.2)) # v43: Be more paranoid
    # v44: Use a safer margin (60%) to react earlier
    safe_margin = float(params.get('safe_margin', 0.60)) 
    warning_line = slo_limit * safe_margin
    
    grad_risk = 0.0
    if pred_upper > warning_line:
        # 使用指数增长来更强力地响应风险
        risk_factor = (pred_upper - warning_line) / (slo_limit - warning_line)
        grad_risk = -w2 * math.exp(risk_factor)

    # 1.3. 动态风险权重 (Dynamic Risk Weight)
    # 当实际延迟（而非预测延迟）已经很高时，加大风险梯度的权重
    p90_ema = float(state.get('p90_ema', 0.0)) if isinstance(state, dict) else 0.0
    dynamic_risk_weight = 1.0
    if p90_ema > ref_latency:
        dynamic_risk_weight = 1.0 + (p90_ema - ref_latency) / ref_latency

    # 1.4. 梯度追踪 (Gradient Tracking) - v44 fix
    # 目标：平滑决策，防止剧烈波动
    # 错误修复：之前当WCP预测偏低时，grad_track会错误地把分配往下拉。
    # 新逻辑：只在预测值高于参考延迟时（即系统有过热风险时）才允许grad_track产生向上的推力。
    # 这样可以防止在WCP失效（预测值极低）时，grad_track还继续把系统往深渊里推。
    grad_track = 0.0
    if pred_upper > ref_latency:
        grad_track = float(state.get('grad_track', 0.0)) if isinstance(state, dict) else 0.0


    # 4. 真实延迟兜底 (Dynamic Reality Check) - v46
    # 彻底解决硬编码问题。直接读取系统近期的真实 P90 延迟。
    # 如果真实延迟逼近 SLO，产生极其强大的恐慌推力，无视 WCP 的盲目乐观。
    actual_p90 = float(state.get('p90_belief', 0.0)) if isinstance(state, dict) else 0.0
    grad_panic = 0.0
    panic_margin = slo_limit * 0.90 # 162ms
    if actual_p90 > panic_margin:
        # 当真实延迟超过安全线，产生一个与超标程度二次相关的巨大负梯度（向上推力）
        panic_excess = (actual_p90 - panic_margin) / max(1.0, slo_limit)
        grad_panic = -50.0 * (panic_excess ** 2) - 5.0 * panic_excess


    # 5. 终极合力计算
    # v46: 加入 grad_panic
    grad = 2.0 * grad_track + dynamic_risk_weight * grad_risk + grad_waste + grad_panic

    # 6. 更新梯度追踪和分配
    # v44: 严格限制下降速度，给系统更多反应时间
    max_decrease = float(params.get('max_decrease', 0.02))
    lr = float(params.get('lr', 0.01))
    prev_alloc = float(state.get('prev_alloc', 1.0)) if isinstance(state, dict) else 1.0
    
    # 更新梯度追踪 EMA
    beta = float(params.get('beta', 0.5))
    new_grad_track = beta * grad_track + (1 - beta) * grad
    
    # 计算新的分配值
    new_alloc = prev_alloc - lr * grad
    
    # 施加下降速度限制
    if new_alloc < prev_alloc:
        new_alloc = max(new_alloc, prev_alloc - max_decrease)

    # 7. 应用边界和最终决策
    # v46: 移除硬编码底线，恢复动态探索能力。安全网由 grad_panic 提供。
    lower = 0.60
    upper = 1.0
    final_alloc = max(lower, min(upper, new_alloc))

    # 准备返回的状态
    new_state = {
        'grad_track': new_grad_track,
        'prev_alloc': final_alloc,
        'p90_ema': p90_ema, # p90_ema is updated outside
        'p90_belief': actual_p90 # Pass through for next iteration
    }
    
    debug_info = {
        "grad_waste": grad_waste,
        "grad_risk": grad_risk,
        "grad_track": grad_track,
        "grad_panic": grad_panic,
        "grad_total": grad,
        "dynamic_risk_weight": dynamic_risk_weight,
        "pred_upper": pred_upper,
        "ref_latency": ref_latency,
        "p90_ema": p90_ema,
        "actual_p90": actual_p90,
        "prev_alloc": prev_alloc,
        "new_alloc_before_clip": new_alloc,
        "final_alloc": final_alloc
    }

    return final_alloc, new_state, debug_info
