import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from experiments.serverless_test.wcp_validation.serverless_utils import run_experiment_sequence

if __name__ == "__main__":
    print(">>> SERVERLESS TEST: GROUP 2 (SIMPLE WCP) <<<")
    print("Goal: Verify Simple WCP (RLS + Unweighted Quantile).")
    run_experiment_sequence(mode='simple', steps=20)
