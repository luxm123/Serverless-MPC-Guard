import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from experiments.serverless_test.wcp_validation.serverless_utils import run_experiment_sequence

if __name__ == "__main__":
    print(">>> SERVERLESS TEST: GROUP 1 (BASELINE - RLS ONLY) <<<")
    print("Goal: Verify RLS prediction without uncertainty (Unc=0).")
    # Run longer sequence to see behavior
    run_experiment_sequence(mode='baseline', steps=20)
