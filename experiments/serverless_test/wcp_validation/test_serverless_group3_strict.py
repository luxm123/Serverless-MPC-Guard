import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from experiments.serverless_test.wcp_validation.serverless_utils import run_experiment_sequence

if __name__ == "__main__":
    print(">>> SERVERLESS TEST: GROUP 3 (STRICT WCP) <<<")
    print("Goal: Verify Strict WCP (Weighted Quantile + Inf Mass).")
    run_experiment_sequence(mode='strict', steps=20)
