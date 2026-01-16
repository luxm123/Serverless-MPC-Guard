#!/usr/bin/env bash
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"
CONTROLLER_NAME="${MPC_CONTROLLER_NAME:-mpc-controller}"
WORKER_NAME="${MPC_WORKER_NAME:-mpc-worker}"
ROLE_NAME="${LAMBDA_ROLE_NAME:-lambda-basic-exec}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  cat > trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Principal": { "Service": "lambda.amazonaws.com" }, "Action": "sts:AssumeRole" }
  ]
}
JSON
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://trust.json
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
fi
ROLE_ARN="arn:aws:iam::$ACCOUNT:role/$ROLE_NAME"
BASE="${HOME}/lambda"
mkdir -p "$BASE/$CONTROLLER_NAME" "$BASE/$WORKER_NAME"
cat > "$BASE/$CONTROLLER_NAME/lambda_function.py" <<'PY'
def lambda_handler(event, context):
    decision = {"should_shed": False, "degrade_plan": None, "admit_threshold_ms": 10000, "pred_total_latency_ms": None}
    return {"decision": decision}
PY
cat > "$BASE/$WORKER_NAME/lambda_function.py" <<'PY'
import time
def lambda_handler(event, context):
    decision = (event.get("decision", {}) or {})
    task = (event.get("task", {}) or {})
    dur_ms = int(task.get("simulated_duration_ms", 50))
    time.sleep(dur_ms / 1000.0)
    return {"response": {"status": "ok", "latency_ms": dur_ms, "degraded": bool(decision.get("degrade_plan"))}}
PY
cd "$BASE/$CONTROLLER_NAME" && zip -qr "../${CONTROLLER_NAME}.zip" .
cd "$BASE/$WORKER_NAME" && zip -qr "../${WORKER_NAME}.zip" .
if ! aws lambda get-function --region "$REGION" --function-name "$CONTROLLER_NAME" >/dev/null 2>&1; then
  aws lambda create-function --region "$REGION" --function-name "$CONTROLLER_NAME" --runtime python3.11 --role "$ROLE_ARN" --handler lambda_function.lambda_handler --zip-file "fileb://$BASE/${CONTROLLER_NAME}.zip"
else
  aws lambda update-function-code --region "$REGION" --function-name "$CONTROLLER_NAME" --zip-file "fileb://$BASE/${CONTROLLER_NAME}.zip"
fi
if ! aws lambda get-function --region "$REGION" --function-name "$WORKER_NAME" >/dev/null 2>&1; then
  aws lambda create-function --region "$REGION" --function-name "$WORKER_NAME" --runtime python3.11 --role "$ROLE_ARN" --handler lambda_function.lambda_handler --zip-file "fileb://$BASE/${WORKER_NAME}.zip"
else
  aws lambda update-function-code --region "$REGION" --function-name "$WORKER_NAME" --zip-file "fileb://$BASE/${WORKER_NAME}.zip"
fi
aws lambda invoke --region "$REGION" --function-name "$CONTROLLER_NAME" --payload "{}" "$BASE/out1.json" --cli-binary-format raw-in-base64-out >/dev/null
aws lambda invoke --region "$REGION" --function-name "$WORKER_NAME" --payload '{"decision":{},"task":{"simulated_duration_ms":50}}' "$BASE/out2.json" --cli-binary-format raw-in-base64-out >/dev/null
echo "Controller:"; cat "$BASE/out1.json"
echo "Worker:";    cat "$BASE/out2.json"
echo "export AWS_REGION=$REGION"
echo "export MPC_CONTROLLER_NAME=$CONTROLLER_NAME"
echo "export MPC_WORKER_NAME=$WORKER_NAME"
