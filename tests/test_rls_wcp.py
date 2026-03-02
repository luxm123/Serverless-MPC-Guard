
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from wcp.wcp_update import RLS, wcp_update

def test_rls_basic():
    print("--- Testing RLS Basic Update ---")
    rls = RLS(2, lambda_factor=0.99, delta=10.0)
    # y = 2*x1 + 3*x2 (approximately)
    phi = [1.0, 5.0]
    y = 13.0
    for _ in range(20):
        rls.update(phi, y)
    
    print(f"Theta: {rls.theta}")
    print(f"P symmetric: {rls.P[0][1] == rls.P[1][0]}")
    assert abs(rls.predict(phi) - y) < 1.0
    print("RLS Basic Update Success\n")

def test_wcp_api():
    print("--- Testing WCP Update API ---")
    state = {}
    metrics = {
        'p90': 100.0,
        'timeout_rate': 0.01,
        'error_rate': 0.0,
        'memory_pressure': 0.4,
        'concurrency': 10,
        'backlog': 5
    }
    pred, unc, debug = wcp_update(state, metrics)
    print(f"Prediction: {pred}")
    print(f"Uncertainty: {unc}")
    print(f"RLS States keys: {state['rls_states'].keys()}")
    # Check if dimension is 5 as expected
    p90_theta = state['rls_states']['p90']['theta']
    print(f"p90 Theta length: {len(p90_theta)}")
    assert len(p90_theta) == 5
    print("WCP API Success\n")

if __name__ == "__main__":
    try:
        test_rls_basic()
        test_wcp_api()
        print("All Tests Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        sys.exit(1)
