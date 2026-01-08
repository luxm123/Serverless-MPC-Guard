import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from experiments.serverless_test.wcp_validation.serverless_utils import run_experiment_sequence

if __name__ == "__main__":
    print(">>> SERVERLESS TEST: GROUP 4 (LITE A - CHEBYSHEV EWMA) <<<")
    print("Goal: Verify Lite A (Chebyshev EWMA) with Cold Start.")
    # Expect high uncertainty for first 10 steps (Cold Start)
    run_experiment_sequence(mode='lite_a', steps=20)
