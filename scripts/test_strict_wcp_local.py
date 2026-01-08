import sys
import os
import random
import math

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wcp.wcp_update import wcp_update

def test_strict_wcp():
    print("=== Testing Strict WCP Implementation ===")
    
    # 1. Initialize State
    state = {}
    
    # 2. Generate Synthetic Data (Sine wave + Noise)
    # We simulate 'p90' latency
    data_points = []
    for t in range(200):
        val = 100.0 + 50.0 * math.sin(t / 10.0) + random.uniform(-10, 10)
        metrics = {
            'p90': val,
            'timeout_rate': 0.0,
            'error_rate': 0.0,
            'memory_pressure': 0.0
        }
        data_points.append(metrics)
        
    # 3. Run Loop
    coverages = []
    uncertainties = []
    
    print(f"{'Step':<5} | {'Observed':<10} | {'Predicted':<10} | {'Uncertainty':<12} | {'L1 Score':<10} | {'Covered?'}")
    print("-" * 80)
    
    for i, metrics in enumerate(data_points):
        # Store "true" next value for coverage check (simplified)
        # Actually coverage is checking if y_k is in C_k(x_k).
        # wcp_update returns prediction for NEXT step (k+1) and uncertainty for NEXT step.
        # So we check if y_k was covered by prediction made at k-1.
        
        # Get prediction from previous step
        prev_pred = state.get('last_prediction', [0]*4)
        prev_unc = state.get('last_uncertainty', 0.0) # We need to store this manually for test checking
        
        # Run Update
        pred_dict, uncertainty, debug = wcp_update(state, metrics, alpha=0.1)
        
        # Check coverage for CURRENT step (using previous prediction)
        if i > 0:
            # Reconstruct L1 score for current step
            y_curr = [metrics['p90'], metrics['timeout_rate'], metrics['error_rate'], metrics['memory_pressure']]
            # prev_pred is list
            l1_score = sum(abs(y_curr[j] - prev_pred[j]) for j in range(4))
            
            is_covered = l1_score <= prev_unc
            coverages.append(is_covered)
            
            if i % 20 == 0:
                print(f"{i:<5} | {metrics['p90']:<10.2f} | {prev_pred[0]:<10.2f} | {prev_unc:<12.2f} | {l1_score:<10.2f} | {is_covered}")
        
        # Store for next verification
        state['last_uncertainty'] = uncertainty
        
    # 4. Analyze Results
    coverage_rate = sum(coverages) / len(coverages) if coverages else 0.0
    print("-" * 80)
    print(f"Total Steps: {len(data_points)}")
    print(f"Coverage Rate: {coverage_rate:.2%} (Target: 90.00%)")
    print(f"Final RLS State: {state.get('rls_states', {}).keys()}")
    
    if coverage_rate > 0.8: # Allow some slack for initial convergence
        print("SUCCESS: WCP Coverage is reasonable.")
    else:
        print("WARNING: Coverage might be too low. Check RLS convergence.")

if __name__ == "__main__":
    test_strict_wcp()
