import boto3
import json
import time
import os
from botocore.config import Config

_GLOBAL_LAMBDA_CLIENT = None

def get_lambda_client():
    """
    Returns a cached global boto3 lambda client configured with adaptive retries
    and high connection pool size to handle concurrency.
    """
    global _GLOBAL_LAMBDA_CLIENT
    if _GLOBAL_LAMBDA_CLIENT:
        return _GLOBAL_LAMBDA_CLIENT
        
    config = Config(
        retries = {
            'max_attempts': 10, # Increased to 10 for extreme resilience
            'mode': 'adaptive'  # Use adaptive mode for client-side rate limiting
        },
        connect_timeout=5, 
        read_timeout=6, # Increased to 6s to allow for deeper queues before giving up
        max_pool_connections=200 
    )
    _GLOBAL_LAMBDA_CLIENT = boto3.client('lambda', 
                       region_name=os.environ.get('AWS_REGION','us-east-1'),
                       config=config)
    return _GLOBAL_LAMBDA_CLIENT

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

def invoke_controller_lambda(payload, mode='strict', **kwargs):
    """
    Directly invoke the Lambda controller.
    payload: Dict containing metrics, etc.
    mode: 'baseline', 'simple', 'strict' (Passed in payload if supported by Lambda)
    kwargs: Additional fields to merge into lambda payload (e.g. strategy)
    """
    lmb = get_lambda_client()
    name = os.environ.get('MPC_CONTROLLER_NAME','MPC_Controller')
    
    # Construct Lambda payload
    lambda_payload = {
        "requestId": "serverless-test",
        "metrics": payload.get('metrics', {}),
        "task": {
            "priority": payload.get('priority','standard'),
            "id": "serverless-test",
            "risk": payload.get('risk', {})
        },
        # Future-proofing: Pass mode if Lambda supports it
        "wcp_mode": mode 
    }
    # Merge extra args
    lambda_payload.update(kwargs)
    
    try:
        resp = lmb.invoke(
            FunctionName=name, 
            InvocationType='RequestResponse', 
            Payload=json.dumps(lambda_payload).encode('utf-8')
        )
        raw = resp['Payload'].read().decode('utf-8')
        
        # Handle potential Lambda errors
        if 'FunctionError' in resp:
            print(f"Lambda Error: {raw}")
            return None
            
        body = json.loads(raw)
        return body
    except Exception as e:
        print(f"Lambda invoke error: {e}")
        return None

def invoke_worker_lambda(decision, task, mode='auto', resource_alloc=None, **kwargs):
    """
    Directly invoke the Business Worker Lambda.
    decision: Dict from controller (contains shouldShed, etc.)
    task: Task dict
    mode: 'auto', 'force_shed', 'normal'
    resource_alloc: float override (optional)
    kwargs: Additional fields to merge into lambda payload (e.g. strategy, metrics)
    """
    lmb = get_lambda_client()
    name = os.environ.get('MPC_WORKER_NAME','MPC_BusinessWorker')
    
    # Worker expects resource_alloc at root or inside decision? 
    # Based on code: resource_alloc = float(event.get('resource_alloc', 1.0))
    
    payload = {
        "decision": decision,
        "task": task,
        "mode": mode
    }
    
    # Explicitly pass resource_alloc at root if provided, otherwise check decision
    if resource_alloc is not None:
        payload['resource_alloc'] = resource_alloc
    elif decision and 'resource_alloc' in decision:
        payload['resource_alloc'] = decision['resource_alloc']
        
    # Merge extra args (critical for integrated MPC mode which needs metrics/strategy at root)
    payload.update(kwargs)

        
    try:
        start_t = time.time()
        resp = lmb.invoke(
            FunctionName=name, 
            InvocationType='RequestResponse', 
            Payload=json.dumps(payload).encode('utf-8')
        )
        end_t = time.time()
        
        # LogResult is base64 encoded if we asked for it, but here we just want duration
        # Real execution duration is in the response log or we approximate with client side
        # Ideally we parse 'LogResult' if LogType='Tail', but that requires decoding.
        # For simplicity, we use client-side duration as a proxy for E2E latency.
        
        raw = resp['Payload'].read().decode('utf-8')

        if 'FunctionError' in resp:
            print(f"Worker Lambda Error: {raw}")
            return None

        body = json.loads(raw)

        return {
            'response': body,
            'client_duration': (end_t - start_t) * 1000.0
        }
    except Exception as e:
        print(f"Worker invoke error: {e}")
        return None

def run_experiment_sequence(mode, steps=10, quiet=False):
    """
    Run a sequence of requests to simulate a timeline.
    mode: 'baseline' (ignore unc), 'simple' (strict as proxy), 'strict' (full)
    Returns list of dicts with step data.
    """
    if not quiet:
        print(f"--- Starting Serverless Experiment: {mode.upper()} ---")
    
    history = []
    
    # Simple simulated metric sequence (Stable -> Spike -> Stable)
    metrics_seq = []
    for i in range(steps):
        base_p90 = 100
        if 3 <= i <= 6: base_p90 = 500 # Spike
        
        metrics_seq.append({
            'p90': base_p90,
            'timeout_rate': 0.01,
            'error_rate': 0.005,
            'memory_pressure': 50
        })
        
    for i, m in enumerate(metrics_seq):
        # Now that Lambda supports wcp_mode, we pass it directly.
        payload = {'metrics': m, 'priority': 'standard'}
        result = invoke_controller_lambda(payload, mode=mode)
        
        step_data = {
            'step': i,
            'input': m['p90'],
            'pred': 0.0,
            'unc': 0.0,
            'lower': 0.0,
            'upper': 0.0,
            'covered': False,
            'server_mode': 'unknown'
        }

        if result:
            # print(f"DEBUG: Raw Lambda Response: {json.dumps(result)}")
            # Parse from the actual Lambda response structure
            decision_data = result.get('decision', {})
            meta_data = result.get('meta', {})
            
            # 1. Get Prediction
            pred = decision_data.get('p90_prediction', 0.0)
            
            # 2. Get Uncertainty
            # specific strict wcp returns uncertainty dict inside decision
            unc_dict = decision_data.get('uncertainty', {})
            if isinstance(unc_dict, dict):
                raw_unc = unc_dict.get('p90', 0.0)
            else:
                raw_unc = float(unc_dict)
                
            p90_lower = pred - raw_unc
            p90_upper = pred + raw_unc
            
            # Infer prediction center for display
            pred_p90 = pred
            
            # Check coverage
            is_covered = (p90_lower <= m['p90'] <= p90_upper)
            effective_unc = raw_unc
            debug = result.get('wcp_debug', {})
            mode_in_debug = debug.get('mode', mode)
            
            step_data.update({
                'pred': pred_p90,
                'unc': effective_unc,
                'lower': p90_lower,
                'upper': p90_upper,
                'covered': is_covered,
                'server_mode': mode_in_debug
            })
            
            if not quiet:
                print(f"Step {i}: Input={m['p90']} | Pred~={pred_p90:.1f} | Unc={effective_unc:.3f}")
                print(f"        Bounds: [{p90_lower:.1f}, {p90_upper:.1f}]")
                print(f"        Covered: {is_covered}")
                print(f"        Server Mode: {mode_in_debug}")
            
        else:
            if not quiet:
                print(f"Step {i}: Failed")
        
        history.append(step_data)
        time.sleep(0.5) # Avoid rate limits
        
    return history

def force_cold_start(function_names):
    """
    Forces a cold start by updating a dummy environment variable for the specified functions.
    """
    lmb = get_lambda_client()
    if isinstance(function_names, str):
        function_names = [function_names]
        
    print(f"Forcing cold start for: {function_names}...")
    for fname in function_names:
        try:
            # 1. Get current config
            conf = lmb.get_function_configuration(FunctionName=fname)
            env = conf.get('Environment', {}).get('Variables', {})
            
            # 2. Update a dummy variable
            env['FORCE_COLD_START'] = str(time.time())
            
            # 3. Update function configuration
            lmb.update_function_configuration(
                FunctionName=fname,
                Environment={'Variables': env}
            )
            print(f"  - Updated {fname} successfully.")
        except Exception as e:
            print(f"  - Failed to update {fname}: {e}")
    
    # Wait a bit for the update to propagate
    time.sleep(2)
    print("Cold start forced.")
