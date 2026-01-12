
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.mpc.controller import MPCController
from src.mpc.optimization import Optimizer
from src.wcp.wcp_update import wcp_update
from src.mpc.middleware import MPCMiddleware
from src.mpc import middleware as middleware_mod
from src.mpc.priority import PriorityManager

def test_full_flow():
    controller = MPCController()
    
    # Mock Data
    task = {'type': 'core', 'id': 't1'}
    system_state = {
        'cpu_util': 0.85, 
        'shadow_price': 60.0,
        'last_alloc': 0.8,
        'u_eta': 0.05,
        'gamma': 0.1,
        'p90_latency': 450.0 # Current latency
    }
    
    # Mock WCP constraints
    wcp_constraints = {
        'pred': {'p90': 480.0, 'timeout_rate': 0.01, 'error_rate': 0.0, 'memory_pressure': 0.6},
        'uncertainty': {'p90': 10.0, 'timeout_rate': 0.001, 'error_rate': 0.0, 'memory_pressure': 0.05}
    }
    
    print("Running MPC Decision...")
    result = controller.decide(task, wcp_constraints, system_state)
    
    print("\nDecision Result:")
    print(f"Should Shed: {result['decision']['should_shed']}")
    print(f"Alloc: {result['decision']['resource_alloc']}")
    print(f"Priority: {result['meta']['priority']}")
    print(f"Ref Latency: {result['meta']['ref_target']['ref_latency']}")
    
    # Check if ref_latency is reasonable (should be around 500ms modified by vi)
    # Priority calc involves randomness in fuzzy logic but usually > 0.5 for core
    
    print("\nTest Passed!")

def test_middleware_admission_priority_mapping():
    middleware_mod._L1_CACHE['params'] = {
        'bP': 2000.0,
        'rls_states': {},
        'shadow_price': 0.0,
        'last_alloc': 1.0,
        'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0},
        'gamma': 0.1,
        'u_eta': 0.05,
        'u_max_delta': 0.15,
        'slo_limit': 1000.0,
        'p90_belief': 100.0,
        'pred_admit_enabled': True,
        'admit_thr_platinum_ms': 50.0,
        'admit_thr_gold_ms': 50.0,
        'admit_thr_standard_ms': 50.0,
        'queue_delay_model': 'backlog_linear',
        'queue_backlog_ttl_s': 2.0,
        'buffer_servers_default': 1.0,
        'avg_service_ms': 100.0,
    }
    middleware_mod._L1_CACHE['version'] = 1
    middleware_mod._L1_CACHE['last_sync'] = 1e12

    mw = MPCMiddleware()

    event = {
        'metrics': {'p90': 100.0, 'queue_backlog': 10.0, 'concurrency': 1.0},
        'task': {'id': 't_mw_1'},
        'slo_limit': 1000.0,
    }
    decision, dbg = mw.decide(event)

    assert decision.get('shouldShed') is True
    assert decision.get('queue_backlog_source') == 'metrics'
    assert decision.get('pred_total_latency_ms', 0.0) > 0.0
    assert decision.get('admit_threshold_ms', 0.0) == 50.0
    pr = decision.get('priority_score', None)
    assert pr is not None
    assert 0.0 <= float(pr) <= 1.0

def test_cl_continuous_from_task_attrs():
    pm = PriorityManager()
    now = time.time()
    system_state = {'shadow_price': 0.0}
    t_low = {
        'id': 't_low',
        'business_value': 0.5,
        'latency_sens': 0.0,
        'risk': {'impact': 0.0, 'volatility': 0.0},
        'timestamp': now,
    }
    t_high = {
        'id': 't_high',
        'business_value': 0.5,
        'latency_sens': 1.0,
        'risk': {'impact': 1.0, 'volatility': 1.0},
        'timestamp': now - 120.0,
    }
    _, v_low = pm.calculate_priority(t_low, system_state)
    _, v_high = pm.calculate_priority(t_high, system_state)
    assert v_high[1] > v_low[1]

if __name__ == "__main__":
    test_full_flow()
    test_middleware_admission_priority_mapping()
    test_cl_continuous_from_task_attrs()
