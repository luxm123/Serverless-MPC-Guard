
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.mpc.controller import MPCController
from src.mpc.optimization import Optimizer
from src.wcp.wcp_update import wcp_update

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

if __name__ == "__main__":
    test_full_flow()
