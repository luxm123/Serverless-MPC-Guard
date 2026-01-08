#!/usr/bin/env bash
set -e
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install boto3 numpy
export AWS_REGION=${AWS_REGION:-us-east-1}
export MPC_WORKER_NAME=${MPC_WORKER_NAME:-MPC_BusinessWorker}
ts=$(date +%Y%m%d_%H%M%S)
mkdir -p experiments/serverless_test/wcp_validation/logs
python experiments/serverless_test/wcp_validation/run_comparison_experiment.py --levels 50,100 --minutes 10 --region "$AWS_REGION" --function "$MPC_WORKER_NAME" | tee experiments/serverless_test/wcp_validation/logs/run_${ts}.log
