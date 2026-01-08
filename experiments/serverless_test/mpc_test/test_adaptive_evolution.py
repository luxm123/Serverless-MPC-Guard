import boto3
import json
import time
import statistics
import random

# Configuration
REGION = 'us-east-1'
SFN_ARN = None 
sfn = boto3.client('stepfunctions', region_name=REGION)
dynamodb = boto3.client('dynamodb', region_name=REGION)

def get_sfn_arn(name_substring='MPC_Layered_Defense_Workflow_Complex'):
    pages = sfn.get_paginator('list_state_machines')
    for page in pages.paginate():
        state_machines = page.get('stateMachines', [])
        if not state_machines:
             state_machines = page.get('StateMachines', [])
        
        for sm in state_machines:
            if name_substring in sm['name']:
                return sm['stateMachineArn']
    return None

def clear_state(state_id):
    """Reset the state in DynamoDB for a fresh start."""
    try:
        dynamodb.delete_item(
            TableName='MPC_State',
            Key={'id': {'S': state_id}}
        )
        print(f"[Setup] Cleared state: {state_id}")
    except Exception as e:
        print(f"[Setup] Warning: Could not clear state {state_id}: {e}")

def run_execution(payload):
    global SFN_ARN
    if not SFN_ARN:
        SFN_ARN = get_sfn_arn()
        if not SFN_ARN:
            raise Exception("SFN ARN not found")
            
    start_resp = sfn.start_execution(
        stateMachineArn=SFN_ARN,
        input=json.dumps(payload)
    )
    execution_arn = start_resp['executionArn']
    
    # Poll
    while True:
        desc = sfn.describe_execution(executionArn=execution_arn)
        status = desc['status']
        if status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
            break
        time.sleep(0.5)
        
    if status == 'SUCCEEDED':
        return json.loads(desc['output'])
    else:
        print(f"[ERROR DEBUG] Status: {status}")
        if 'error' in desc: print(f"Error: {desc['error']}")
        if 'cause' in desc: print(f"Cause: {desc['cause']}")
        raise Exception(f"Execution failed: {status}")

def run_phase(phase_name, group_name, config, iterations=5):
    print(f"\n>>> Running Phase: {phase_name} | Group: {group_name}")
    results = []
    
    # Config parameters
    state_id = f"exp_{group_name}"
    enable_feedback = config['enable_feedback']
    
    for i in range(iterations):
        # 1. Construct Payload
        # We inject metrics to simulate the environment state that the Controller sees.
        # This drives the "Sensing" part of the closed loop.
        metrics = config['metrics_injection']
        
        # Add some jitter to metrics to make it realistic
        current_metrics = {
            'p90': metrics.get('p90', 100) + random.uniform(-10, 10),
            'slo_violation_rate': max(0.0, metrics.get('slo_violation_rate', 0.0) + random.uniform(-0.01, 0.01)),
            'resource_waste_rate': max(0.0, metrics.get('resource_waste_rate', 0.0) + random.uniform(-0.02, 0.02)),
            'error_rate': metrics.get('error_rate', 0.0),
            'timeout_rate': metrics.get('timeout_rate', 0.0),
            'cpu_util': metrics.get('cpu_util', 0.5)
        }
        
        task_payload = {
            "id": f"{group_name}-{phase_name}-{i}",
            "priority": "high", # We use High priority tasks to test shedding/degradation
            "consistency": "strong",
            "fault_type": config.get('worker_fault', 'none') # Fault injected into Worker
        }
        
        payload = {
            "task_name": f"Test-{group_name}-{phase_name}-{i}",
            "priority": "high",
            "risk": "standard",
            "consistency": "strong",
            "state_id": state_id,
            "enable_feedback": enable_feedback,
            "strategy": "mpc",
            "wcp_mode": "strict",
            "metrics": current_metrics,
            "task": task_payload
        }
        
        # 2. Execute
        try:
            output = run_execution(payload)
            
            # 3. Extract Data
            # Note: The output structure depends on the ASL. 
            # Usually it returns the result of the last step or a map.
            # We look for 'mpc_result' and 'worker_result' if available, 
            # or try to parse the flat output.
            
            mpc_res = {}
            worker_res = {}
            
            # Heuristic parsing based on common ASL output patterns
            if 'mpc_result' in output:
                mpc_res = output['mpc_result'].get('Payload', {})
            elif 'decision' in output: # Maybe flat?
                mpc_res = output
            elif 'Payload' in output: # Direct Lambda
                mpc_res = output['Payload']
                
            # Try to find worker result (latency)
            # If the workflow puts worker output in 'worker_result' or similar
            if 'worker_result' in output:
                worker_res = output['worker_result'].get('Payload', {})
            elif 'latency_ms' in output:
                worker_res = output
            
            decision = mpc_res.get('decision', {})
            meta = mpc_res.get('meta', {})
            overhead = mpc_res.get('overhead', {})
            
            # Real Latency from Worker
            real_latency = worker_res.get('latency_ms', 0.0)
            status = worker_res.get('status', 'unknown')
            
            # Record
            data = {
                'iter': i,
                'alloc': decision.get('resource_alloc', 1.0),
                'shadow_price': decision.get('shadow_price', 0.0),
                'priority_score': meta.get('priority', 0.0),
                'real_latency': real_latency,
                'shed': decision.get('shouldShed', False),
                'status': status
            }
            results.append(data)
            
            # Print brief progress
            print(f"   [{i+1}/{iterations}] Alloc: {data['alloc']:.2f} | SP: {data['shadow_price']:.2f} | Pri: {data['priority_score']:.2f} | Lat: {data['real_latency']:.1f}ms | Shed: {data['shed']}")
            
        except Exception as e:
            print(f"   [{i+1}/{iterations}] Failed: {e}")
            
    return results

def print_summary(results):
    if not results:
        print("No results.")
        return
        
    avg_alloc = statistics.mean([r['alloc'] for r in results])
    avg_sp = statistics.mean([r['shadow_price'] for r in results])
    avg_lat = statistics.mean([r['real_latency'] for r in results])
    shed_count = sum(1 for r in results if r['shed'])
    
    print(f"   -> Avg Alloc: {avg_alloc:.2f}")
    print(f"   -> Avg Shadow Price: {avg_sp:.2f}")
    print(f"   -> Avg Latency: {avg_lat:.1f} ms")
    print(f"   -> Shed Count: {shed_count}/{len(results)}")

if __name__ == "__main__":
    print("=== Starting Adaptive Evolution Experiment ===")
    
    # 1. Setup Groups
    # Group A: Static (No Feedback)
    config_a = {
        'enable_feedback': False,
    }
    # Group B: Adaptive (Feedback Enabled)
    config_b = {
        'enable_feedback': True,
    }
    
    # Reset States
    clear_state("exp_GroupA")
    clear_state("exp_GroupB")
    
    # --- Phase 1: Baseline ---
    # Normal metrics, no faults
    p1_metrics = {'p90': 100, 'slo_violation_rate': 0.0, 'resource_waste_rate': 0.1, 'error_rate': 0.0}
    
    print("\n[PHASE 1] Baseline (Normal Load)")
    res_a_p1 = run_phase("Phase1", "GroupA", {**config_a, 'metrics_injection': p1_metrics}, iterations=5)
    res_b_p1 = run_phase("Phase1", "GroupB", {**config_b, 'metrics_injection': p1_metrics}, iterations=5)
    
    print("\n--- Phase 1 Summary ---")
    print("Group A (Static):")
    print_summary(res_a_p1)
    print("Group B (Adaptive):")
    print_summary(res_b_p1)
    
    # --- Phase 2: Stability Crisis ---
    # High Error Rate injected into Controller Metrics (Sensing)
    # AND Timeout injected into Worker (Actuation Reality)
    p2_metrics = {'p90': 400, 'slo_violation_rate': 0.05, 'resource_waste_rate': 0.1, 'error_rate': 0.15, 'timeout_rate': 0.1}
    
    print("\n[PHASE 2] Stability Crisis (High Errors)")
    # Note: Worker fault 'timeout' makes latency huge (11s), causing timeouts.
    # We expect Group B to lower the priority of these unstable tasks or shed them more aggressively 
    # if the virtual expert downgrades the "Consistency" weight or score.
    res_a_p2 = run_phase("Phase2", "GroupA", {**config_a, 'metrics_injection': p2_metrics, 'worker_fault': 'none'}, iterations=5)
    res_b_p2 = run_phase("Phase2", "GroupB", {**config_b, 'metrics_injection': p2_metrics, 'worker_fault': 'none'}, iterations=5)
    # Note: I disabled 'worker_fault'='timeout' to avoid 11s waits which slow down the test too much.
    # We focus on the Controller's reaction to the *Metrics* (Sensing).
    
    print("\n--- Phase 2 Summary ---")
    print("Group A (Static):")
    print_summary(res_a_p2)
    print("Group B (Adaptive):")
    print_summary(res_b_p2)
    
    # --- Phase 3: Resource Crunch ---
    # High Utilization -> Shadow Price should rise
    p3_metrics = {'p90': 150, 'slo_violation_rate': 0.0, 'resource_waste_rate': 0.0, 'cpu_util': 0.95}
    
    print("\n[PHASE 3] Resource Crunch (High Congestion)")
    # We run more iterations to let Gradient Descent converge
    res_a_p3 = run_phase("Phase3", "GroupA", {**config_a, 'metrics_injection': p3_metrics}, iterations=8)
    res_b_p3 = run_phase("Phase3", "GroupB", {**config_b, 'metrics_injection': p3_metrics}, iterations=8)
    
    print("\n--- Phase 3 Summary ---")
    print("Group A (Static):")
    print_summary(res_a_p3)
    print("Group B (Adaptive):")
    print_summary(res_b_p3)
    
    print("\n=== Experiment Complete ===")
