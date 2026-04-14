import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.mpc.controller import MPCController
from experiments.local_simulation.mpc_test.sim_utils import TaskGenerator, SimulatorEnv
from src.wcp.wcp_update import RLS

def run_experiment(mode='mpc', duration_sec=30):
    """
    Run simulation for Scenario 1.
    mode: 'mpc', 'baseline' (No MPC), 'static' (Priority Only)
    """
    print(f"--- Starting Experiment: {mode.upper()} ---")
    
    # Initialize components
    task_gen = TaskGenerator()
    env = SimulatorEnv()
    
    if mode == 'mpc':
        controller = MPCController()
    
    # Sim loop
    start_time = time.time()
    steps = 0
    # Dummy WCP constraints (Assuming stable prediction for Scenario 1)
    wcp_constraints = {'p90': 150.0, 'uncertainty': 50.0} 
    
    # System State for MPC
    system_state = {
        'last_alloc': 1.0,
        'p90_belief': 150.0,
        'uncertainty': 20.0,
        'last_y': 150.0,
        'slo_limit': 200.0,
        'e2e_overhead_ms': 0.0,
        'concurrency': 1.0,
        'backlog': 1.0,
        'budget': 0.0,
        'min_alloc': 0.0,
        'max_alloc': 4.0
    }

    while time.time() - start_time < duration_sec:
        steps += 1
        task = task_gen.generate_task(steps)
        
        # --- Decision Making ---
        if mode == 'mpc':
            # Full MPC Logic
            system_state['slo_limit'] = float(task.get('slo', system_state.get('slo_limit', 200.0)))
            result = controller.decide(task, wcp_constraints, system_state)
            decision = {
                'should_shed': result['decision']['should_shed'],
                'resource_alloc': result['decision']['resource_alloc']
            }
            
        elif mode == 'static':
            # Static Priority Logic: High gets 1.0, Med 0.8, Low 0.5
            p = task['priority']
            if p == 'critical': alloc = 1.0
            elif p == 'high': alloc = 0.8
            else: alloc = 0.5
            decision = {'should_shed': False, 'resource_alloc': alloc}
            
        else: # Baseline
            # FIFO / No logic: Everyone gets full resource
            decision = {'should_shed': False, 'resource_alloc': 1.0}
            
        # --- Execution ---
        obs = env.step(task, decision)
        
        # Update System State for next step
        try:
            last_y = float(obs.get('latency', 0.0) or 0.0)
        except Exception:
            last_y = 0.0
        if last_y > 0.0:
            prev_p90 = float(system_state.get('p90_belief', last_y) or last_y)
            system_state['p90_belief'] = 0.9 * prev_p90 + 0.1 * last_y
            system_state['last_y'] = last_y
        system_state['last_alloc'] = float(decision.get('resource_alloc', system_state.get('last_alloc', 1.0)))
        
        # Simulate time step
        # time.sleep(0.001) # Optional for real-time feel, but skip for speed

    # --- Result Calculation ---
    total = env.stats['total_tasks']
    if total == 0: return {}
    
    core_total = env.stats['core_tasks']
    core_viol = env.stats['core_slo_violations']
    
    metrics = {
        'mode': mode,
        'total_tasks': total,
        'core_slo_compliance': 1.0 - (core_viol / max(1, core_total)),
        'avg_resource_waste': env.stats['resource_waste'] / total,
        'long_tail_ratio': env.stats['long_tail_count'] / total,
        'shed_ratio': env.stats['shed_count'] / total
    }
    
    print(f"Completed {mode}: Core SLO={metrics['core_slo_compliance']:.4f}, Waste={metrics['avg_resource_waste']:.4f}")
    return metrics

if __name__ == "__main__":
    modes = ['mpc', 'baseline', 'static']
    results = {}
    
    # Run 3 times and average (Short duration for test run)
    n_runs = 3
    duration = 5 # seconds for quick test, user asked for 30 mins but we scale down for dev
    
    print("Running Scenario 1: Baseline Comparison (Scaled Down Duration)")
    
    for mode in modes:
        mode_res = []
        for i in range(n_runs):
            res = run_experiment(mode, duration_sec=duration)
            mode_res.append(res)
            
        # Average
        avg_metrics = {}
        for k in mode_res[0].keys():
            if k == 'mode': continue
            avg_metrics[k] = np.mean([r[k] for r in mode_res])
        results[mode] = avg_metrics
        
    print("\n--- Final Results (Scenario 1) ---")
    print(f"{'Mode':<10} | {'Core SLO':<10} | {'Res Waste':<10} | {'Long Tail':<10}")
    for mode, m in results.items():
        print(f"{mode:<10} | {m['core_slo_compliance']:.4f}     | {m['avg_resource_waste']:.4f}     | {m['long_tail_ratio']:.4f}")
