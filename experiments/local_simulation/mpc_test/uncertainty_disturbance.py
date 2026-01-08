import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.mpc.controller import MPCController
from experiments.local_simulation.mpc_test.sim_utils import TaskGenerator, SimulatorEnv

def run_disturbance_test(mode='mpc', duration_sec=60):
    """
    Scenario 2: Uncertainty Disturbance Test.
    Injects faults (Resource, Cold Start, Burst) periodically.
    """
    print(f"--- Starting Disturbance Test: {mode.upper()} ---")
    
    task_gen = TaskGenerator()
    env = SimulatorEnv()
    controller = MPCController() if mode == 'mpc' else None
    
    start_time = time.time()
    steps = 0
    
    # Tracking for recovery time
    violation_timestamps = []
    disturbance_active_ts = 0
    
    # State
    system_state = {
        'congestion_price': 0.0,
        'mpc_stats': {'n': 0, 'sum': [0.0, 0.0], 'sum_sq': [0.0, 0.0]},
        'p90_latency': 150.0,
        'cpu_util': 0.5
    }
    
    # Base WCP (Simulating Tightened Constraint due to disturbance if MPC is aware)
    # In real integration, WCP would auto-widen. Here we simulate that effect.
    wcp_constraints = {'p90': 150.0, 'uncertainty': 50.0}

    while time.time() - start_time < duration_sec:
        steps += 1
        curr_time = time.time() - start_time
        
        # --- Inject Disturbances ---
        # 1. Resource Fluctuation (every 5s in scaled time)
        if 10 < curr_time < 15 or 40 < curr_time < 45:
            env.inject_disturbance('resource_fluctuation')
            if mode == 'mpc': wcp_constraints['uncertainty'] = 150.0 # WCP reacts
            
        # 2. Cold Start (every 20s)
        elif 20 < curr_time < 22:
            env.inject_disturbance('cold_start')
            if mode == 'mpc': wcp_constraints['uncertainty'] = 300.0 # WCP reacts strongly
            
        # 3. Burst (every 30s)
        elif 30 < curr_time < 35:
            env.inject_disturbance('burst')
            if mode == 'mpc': system_state['congestion_price'] = 200.0 # Price spikes
            
        else:
            env.clear_disturbance()
            # Recovery
            wcp_constraints['uncertainty'] = max(50.0, wcp_constraints['uncertainty'] * 0.95)
            
        # --- Task Generation ---
        # Burst logic: double task rate
        if env.burst_active:
             # Generate 2 tasks
             tasks = [task_gen.generate_task(steps), task_gen.generate_task(steps+9999)]
        else:
             tasks = [task_gen.generate_task(steps)]
             
        for task in tasks:
            # --- Decision ---
            if mode == 'mpc':
                result = controller.decide(task, wcp_constraints, system_state)
                decision = {
                    'should_shed': result['decision']['should_shed'],
                    'resource_alloc': result['decision']['resource_alloc']
                }
                # Update Feedback
                sim_metrics = {
                    'slo_violation_rate': env.stats['slo_violations'] / max(1, env.stats['total_tasks']),
                    'resource_waste_rate': env.stats['resource_waste'] / max(1, env.stats['total_tasks'])
                }
                controller.update_feedback(sim_metrics)
            else:
                # Baseline: No Shedding, Full Alloc
                decision = {'should_shed': False, 'resource_alloc': 1.0}
                
            # --- Execution ---
            obs = env.step(task, decision)
            
            # Record Violation Time
            if obs['latency'] > task['slo']:
                violation_timestamps.append(curr_time)
                
            # Update State
            system_state['congestion_price'] = obs['congestion_price']
            system_state['cpu_util'] = obs['cpu_util']

    # --- Metrics ---
    total = env.stats['total_tasks']
    core_total = env.stats['core_tasks']
    core_viol = env.stats['core_slo_violations']
    
    metrics = {
        'mode': mode,
        'total_violations': env.stats['slo_violations'],
        'core_slo_violation_rate': core_viol / max(1, core_total),
        'shed_count': env.stats['shed_count']
    }
    
    print(f"Completed {mode}: Total Violations={metrics['total_violations']}, Core Rate={metrics['core_slo_violation_rate']:.4f}")
    return metrics

if __name__ == "__main__":
    modes = ['mpc', 'baseline']
    results = {}
    
    duration = 10 # Scaled down from 60 mins to 10s for dev test
    
    print("Running Scenario 2: Uncertainty Disturbance")
    for mode in modes:
        results[mode] = run_disturbance_test(mode, duration)
        
    print("\n--- Final Results (Scenario 2) ---")
    print(f"{'Mode':<10} | {'Total Viol':<12} | {'Core Fail Rate':<15} | {'Shed Count':<10}")
    for mode, m in results.items():
        print(f"{mode:<10} | {m['total_violations']:<12} | {m['core_slo_violation_rate']:.4f}          | {m['shed_count']:<10}")
