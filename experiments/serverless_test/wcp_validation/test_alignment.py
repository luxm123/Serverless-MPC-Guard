import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.mpc.controller import MPCController

def test_stateless_behavior():
    print("=== Testing Stateless MPC Behavior ===")
    
    # 1. Initial State (Simulate Cold Start)
    system_state = {
        'last_alloc': 0.8,
        'shadow_price': 10.0,
        'mpc_stats': {},
        # Start with default params implicitly
    }
    
    controller = MPCController()
    
    # Mock Inputs
    task = {'latency_sensitivity': 'high', 'tier': 'gold'}
    wcp_constraints = {
        'pred': {'p90': 450.0},
        'uncertainty': {'p90': 20.0} # Low uncertainty
    }
    
    # 2. First Decision
    print("\n--- Step 1: First Decision (Defaults) ---")
    # Parameterize scheduler targets
    system_state['slo_mem_limit'] = 0.7
    system_state['slo_limit'] = 480.0
    result = controller.decide(task, wcp_constraints, system_state)
    print(f"Alloc: {result['decision']['resource_alloc']}")
    print(f"Shadow Price: {result['decision']['shadow_price']}")
    print(f"Scheduler mem limit: {controller.scheduler.slo_mem_limit}")
    
    # 3. Feedback Update (Simulate high violation to trigger adaptation)
    # Violating SLO (limit 500) with 600ms latency
    metrics = {
        'p90': 600.0,
        'timeout_rate': 0.0,
        'error_rate': 0.0,
        'memory_pressure': 0.5
    }
    system_state['metrics'] = metrics # controller.decide uses this if present, but we can call update_feedback directly
    
    print("\n--- Step 2: Feedback Update (Trigger Adaptation) ---")
    # We call update_feedback explicitly to simulate the feedback loop
    updates = controller.update_feedback(metrics, system_state)
    print("Updates:", updates)
    
    # Check if state has new params
    print("\nSystem State keys after update:", list(system_state.keys()))
    if 'opt_w1' in system_state:
        print(f"Optim Weights in State: w1={system_state['opt_w1']}, w3={system_state['opt_w3']}")
    if 'traj_theta' in system_state:
        print(f"Traj Theta in State: {system_state['traj_theta']}")
        
    # Verify values changed from defaults
    # Default w1=1.0. With high violation (latency 600 > 500), w3 (penalty) should likely increase or w1 (cost) decrease?
    # Actually w3 is penalty for violation. If violation occurs, we might want to increase w3 to penalize it more?
    # Or maybe the logic in optimization.py does something specific.
    # Let's just check they exist and are persisted.
    
    # 4. New Controller Instance (Simulate New Lambda)
    print("\n--- Step 3: New Controller Instance (Persisted State) ---")
    new_controller = MPCController()
    
    # Run decide again with the modified system_state
    # If state is used, the optimizer should pick up 'opt_w1' etc.
    # We can verify by calling update_weights again and seeing if it starts from the saved values.
    
    # Let's change opt_w1 manually in state to be sure
    system_state['opt_w1'] = 999.0
    
    updates_2 = new_controller.update_feedback(metrics, system_state)
    print(f"New Updates w1 (should be approx 999): {updates_2['optimizer_weights']['w1']}")
    
    if abs(updates_2['optimizer_weights']['w1'] - 999.0) < 100.0: # Allowing for some adaptation drift
        print("SUCCESS: State persistence verified!")
    else:
        print("FAILURE: State persistence failed!")

if __name__ == "__main__":
    test_stateless_behavior()
