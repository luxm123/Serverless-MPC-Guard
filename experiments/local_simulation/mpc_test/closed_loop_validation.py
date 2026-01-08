import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.mpc.controller import MPCController
from experiments.local_simulation.mpc_test.sim_utils import TaskGenerator, SimulatorEnv

def run_closed_loop_test(use_feedback=True, duration_sec=90):
    """
    Scenario 3: Closed-loop Validation.
    Compares MPC with feedback (Adaptive Weights) vs Fixed MPC.
    """
    mode_name = "Adaptive" if use_feedback else "Fixed"
    print(f"--- Starting Closed-Loop Test: {mode_name} ---")
    
    task_gen = TaskGenerator()
    env = SimulatorEnv()
    controller = MPCController()
    
    # Disable feedback manually if needed
    if not use_feedback:
        # Mock the update method to do nothing
        controller.update_feedback = lambda x: {}
        
    start_time = time.time()
    steps = 0
    
    # Track parameter evolution
    weight_history = []
    
    # State
    system_state = {
        'congestion_price': 0.0,
        'mpc_stats': {'n': 0, 'sum': [0.0, 0.0], 'sum_sq': [0.0, 0.0]},
        'p90_latency': 150.0,
        'cpu_util': 0.5
    }
    wcp_constraints = {'p90': 150.0, 'uncertainty': 50.0}

    # Time-series metrics
    slo_rates = []
    
    while time.time() - start_time < duration_sec:
        steps += 1
        curr_time = time.time() - start_time
        
        # Inject Constant Disturbance (e.g., High Load) to force adaptation
        env.inject_disturbance('resource_fluctuation')
        if steps % 10 == 0: env.inject_disturbance('burst')
        
        task = task_gen.generate_task(steps)
        
        # Decision
        result = controller.decide(task, wcp_constraints, system_state)
        decision = {
            'should_shed': result['decision']['should_shed'],
            'resource_alloc': result['decision']['resource_alloc']
        }
        
        # Feedback
        sim_metrics = {
            'slo_violation_rate': env.stats['slo_violations'] / max(1, env.stats['total_tasks']),
            'resource_waste_rate': env.stats['resource_waste'] / max(1, env.stats['total_tasks'])
        }
        
        new_weights = controller.update_feedback(sim_metrics)
        if use_feedback and steps % 100 == 0:
            weight_history.append(new_weights)
            
        # Execution
        obs = env.step(task, decision)
        
        # Record Snapshot
        if steps % 500 == 0:
            rate = 1.0 - (env.stats['core_slo_violations'] / max(1, env.stats['core_tasks']))
            slo_rates.append(rate)
            
        system_state['congestion_price'] = obs['congestion_price']

    # Final Stats
    metrics = {
        'mode': mode_name,
        'final_core_slo': 1.0 - (env.stats['core_slo_violations'] / max(1, env.stats['core_tasks'])),
        'total_waste': env.stats['resource_waste'],
        'weight_evolution': weight_history[-1] if weight_history else "Fixed"
    }
    
    print(f"Completed {mode_name}: Final SLO={metrics['final_core_slo']:.4f}, Waste={metrics['total_waste']:.2f}")
    if use_feedback:
        print(f"Final Weights: {metrics['weight_evolution']}")
        
    return metrics

if __name__ == "__main__":
    duration = 15 # Scaled down
    
    print("Running Scenario 3: Closed-Loop Optimization")
    
    res_adaptive = run_closed_loop_test(use_feedback=True, duration_sec=duration)
    res_fixed = run_closed_loop_test(use_feedback=False, duration_sec=duration)
    
    print("\n--- Final Results (Scenario 3) ---")
    print(f"{'Mode':<10} | {'Final SLO':<10} | {'Total Waste':<12}")
    print(f"{res_adaptive['mode']:<10} | {res_adaptive['final_core_slo']:.4f}     | {res_adaptive['total_waste']:.2f}")
    print(f"{res_fixed['mode']:<10} | {res_fixed['final_core_slo']:.4f}     | {res_fixed['total_waste']:.2f}")
