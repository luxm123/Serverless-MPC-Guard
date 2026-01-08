import boto3
import json
import time
import os

def find_state_machine_arn():
    sfn = boto3.client('stepfunctions', region_name=os.environ.get('AWS_REGION','us-east-1'))
    env_arn = os.environ.get('SFN_ARN')
    if env_arn: return env_arn
    names = ['MPC_Layered_Defense_Workflow_Complex','MPC_Layered_Defense_Workflow']
    try:
        resp = sfn.list_state_machines()
        found = {}
        for sm in resp.get('stateMachines', []):
            if sm.get('name') in names:
                found[sm.get('name')] = sm.get('stateMachineArn')
        
        for n in names:
            if n in found: return found[n]
            
    except Exception as e:
        print(f"List SFN error: {e}")
    return None

def run_one(sfn, arn, payload):
    resp = sfn.start_execution(stateMachineArn=arn, input=json.dumps(payload))
    ex_arn = resp['executionArn']
    status = 'RUNNING'
    while status == 'RUNNING':
        desc = sfn.describe_execution(executionArn=ex_arn)
        status = desc['status']
        if status == 'RUNNING':
            time.sleep(0.2)
    if status != 'SUCCEEDED':
        print(f"Execution failed: {status} {desc.get('error')} {desc.get('cause')}")
        return None
    out = json.loads(desc.get('output','{}'))
    return out.get('mpc_result', {}).get('Payload', {})

def invoke_controller_lambda(payload):
    lmb = boto3.client('lambda', region_name=os.environ.get('AWS_REGION','us-east-1'))
    name = os.environ.get('MPC_CONTROLLER_NAME','MPC_Controller')
    try:
        resp = lmb.invoke(FunctionName=name, InvocationType='RequestResponse', Payload=json.dumps({
            "requestId": "local-test",
            "metrics": payload.get('metrics', {}),
            "task": {
                "priority": payload.get('priority','standard'),
                "id": "local-test",
                "risk": payload.get('risk', {})
            }
        }).encode('utf-8'))
        raw = resp['Payload'].read().decode('utf-8')
        body = json.loads(raw)
        return body
    except Exception as e:
        print(f"Lambda invoke error: {e}")
        return None

def print_result(tag, payload):
    decision = payload.get('decision', {})
    wcp_bounds = payload.get('wcp_bounds', {})
    wcp_risk = payload.get('wcp_risk', {})
    price = decision.get('congestion_price', 0.0)
    uncertainty = decision.get('uncertainty', 0.0)
    shed = bool(decision.get('shouldShed', False))
    res = "SHED" if shed else "EXECUTE"
    
    # Format bounds for readability
    formatted_bounds = {}
    for k, v in wcp_bounds.items():
        if isinstance(v, dict) and 'lower' in v and 'upper' in v:
            formatted_bounds[k] = f"[{v['lower']:.1f}, {v['upper']:.1f}]"
        else:
            formatted_bounds[k] = v
            
    print(f"{tag}: {res} | Price={price:.2f} | Uncertainty(e_k)={uncertainty:.3f} | Bounds={formatted_bounds}")

def main():
    arn = find_state_machine_arn()
    if not arn:
        print("Cannot find Step Function ARN. Set env SFN_ARN or deploy workflow.")
        print("Falling back to direct Lambda invocation.")
    sfn = boto3.client('stepfunctions', region_name=os.environ.get('AWS_REGION','us-east-1'))
    normal = {'task_name':'TEST_NORMAL','priority':'standard','metrics':{'p90':120,'timeout_rate':0.01,'error_rate':0.005,'memory_pressure':50},'risk':{}}
    high_latency = {'task_name':'TEST_HIGH_LAT','priority':'standard','metrics':{'p90':900,'timeout_rate':0.03,'error_rate':0.015,'memory_pressure':90},'risk':{}}
    critical = {'task_name':'TEST_CRITICAL','priority':'critical','metrics':{'p90':700,'timeout_rate':0.02,'error_rate':0.01,'memory_pressure':80},'risk':{}}
    if arn:
        r1 = run_one(sfn, arn, normal) or invoke_controller_lambda(normal)
        if r1: print_result("NORMAL", r1)
        r2 = run_one(sfn, arn, high_latency) or invoke_controller_lambda(high_latency)
        if r2: print_result("HIGH_LAT", r2)
        r3 = run_one(sfn, arn, critical) or invoke_controller_lambda(critical)
        if r3: print_result("CRITICAL", r3)
    else:
        r1 = invoke_controller_lambda(normal); 
        if r1: print_result("NORMAL", r1)
        r2 = invoke_controller_lambda(high_latency); 
        if r2: print_result("HIGH_LAT", r2)
        r3 = invoke_controller_lambda(critical); 
        if r3: print_result("CRITICAL", r3)
    print("Done.")

if __name__ == "__main__":
    main()
