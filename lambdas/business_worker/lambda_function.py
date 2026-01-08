import json
import boto3
import time
import random
try:
    from src.mpc.middleware import MPCMiddleware
    _MIDDLEWARE = MPCMiddleware()
except ImportError:
    print("MPC Middleware not found. Integrated mode disabled.")
    _MIDDLEWARE = None

sqs = boto3.client('sqs')
RECOVERY_QUEUE_URL = None # Will be populated via env var or discovery

def get_queue_url():
    global RECOVERY_QUEUE_URL
    if RECOVERY_QUEUE_URL: return RECOVERY_QUEUE_URL
    try:
        # In a real deploy, pass this as ENV VAR. 
        # Fallback to lookup for demo simplicity.
        resp = sqs.get_queue_url(QueueName='RecoveryQueue')
        RECOVERY_QUEUE_URL = resp['QueueUrl']
        return RECOVERY_QUEUE_URL
    except:
        print("Queue not found")
        return None

def lambda_handler(event, context):
    """
    Input:
    {
        "decision": { "shouldShed": true/false, "degrade_plan": ... },
        "task": { ... },
        "mode": "auto" | "force_shed" | "normal" | "external_api",
        "strategy": "mpc_integrated" | "external" (default)
    }
    """
    # --- Middleware Integration ---
    strategy = event.get('strategy', 'external')
    decision = event.get('decision', {})
    
    mpc_debug = {}
    
    if strategy == 'mpc_integrated' and _MIDDLEWARE:
        # Run MPC logic internally!
        try:
            internal_decision, debug_info = _MIDDLEWARE.decide(event)
            # Override external decision
            decision = internal_decision
            mpc_debug = debug_info
            # Inject alloc into event for penalty logic below
            event['resource_alloc'] = decision.get('resource_alloc', 1.0)
            print(f"[Integrated MPC] Alloc: {event['resource_alloc']:.2f}, Shed: {decision.get('shouldShed')}")
        except Exception as e:
            print(f"[Integrated MPC Error] {e}")
            # Fallback
            decision = {'shouldShed': False, 'resource_alloc': 1.0}
            
    task = event.get('task', {})
    mode = event.get('mode', 'auto')
    
    start_time = time.time()
    status = "success"
    
    # Determine behavior based on mode and decision
    should_shed = False
    
    if mode == 'force_shed':
        should_shed = True
    elif mode == 'normal' or mode == 'external_api':
        should_shed = False
    else:
        # 'auto' or default: rely on MPC decision
        should_shed = decision.get('shouldShed', False)
    
    # Execution Logic
    if should_shed:
        # --- Fault Tolerance Path ---
        print(f"Task {task.get('id')} shed (Mode: {mode}). Sending to Recovery Queue.")
        q_url = get_queue_url()
        if q_url:
            sqs.send_message(
                QueueUrl=q_url,
                MessageBody=json.dumps({
                    'task': task,
                    'original_intent': decision.get('degrade_plan'),
                    'timestamp': time.time(),
                    'reason': f'mpc_shedding_mode_{mode}'
                })
            )
        status = "degraded"
        # Shedding is fast
        time.sleep(0.05) 
        
    else:
        # --- Normal Execution Path ---
        
        # Base latency setup
        base_latency = 0.1 # 100ms
        
        # Handle "external_api" specific logic (e.g., call 3rd party)
        if mode == 'external_api':
            # Simulate External API call which is slower
            base_latency = 0.5 
            provider = event.get('provider', 'Unknown')
            print(f"Calling External Provider: {provider}")
        
        # Apply Resource Allocation Penalty (Simulating resource limits)
        resource_alloc = float(event.get('resource_alloc', 1.0))
        penalty_factor = 1.0
        if resource_alloc < 1.0:
            # e.g., if alloc is 0.8, we are 20% slower? No, maybe more.
            # Sim_utils used: exec_latency *= (1.0 + (1.0 - alloc))
            penalty_factor = 1.0 + (1.0 - resource_alloc)
            print(f"Resource Alloc: {resource_alloc:.2f} -> Penalty Factor: {penalty_factor:.2f}")

        # Check for Fault Injection
        fault_type = task.get('fault_type', 'none')
        injected_delay = 0.0
        
        if fault_type == 'timeout':
            print("!!! SIMULATING TIMEOUT (Sleeping 11s) !!!")
            time.sleep(11) 
        elif fault_type == 'oom':
            print("!!! SIMULATING OOM (Allocating 150MB) !!!")
            big_list = []
            for _ in range(150):
                big_list.append('x' * 1024 * 1024)
            print(f"Allocated {len(big_list)} MB")
        elif fault_type == 'cold_start':
            print("!!! SIMULATING COLD START (Sleeping 3s) !!!")
            injected_delay = 3.0
        elif fault_type == 'latency':
            injected_delay = float(task.get('injected_delay_ms', 0)) / 1000.0
            
        # Add Jitter
        jitter = random.uniform(0, 0.5) 
        if random.random() < 0.1: 
            jitter += 1.0 # Long tail
            
        # Apply penalty to TOTAL active time (base + jitter)
        total_sleep = (base_latency + jitter) * penalty_factor + injected_delay
        time.sleep(total_sleep)
        print(f"Task {task.get('id')} processed. Mode: {mode}. Sleep: {total_sleep:.3f}s (Base: {base_latency}, Jitter: {jitter:.3f}, Pen: {penalty_factor:.2f})")
        
    duration = (time.time() - start_time) * 1000 # ms
    
    # --- POST-EXECUTION: Feedback Update ---
    # Update MPC state with ACTUAL metrics from this execution
    if strategy == 'mpc_integrated' and _MIDDLEWARE:
        try:
            # Construct metrics based on actual execution
            real_metrics = {
                'latency': duration,
                'cpu_usage': 0.8 if duration > 500 else 0.2, # Rough estimation based on duration
                'error_rate': 1.0 if status != 'success' else 0.0,
                'timestamp': time.time()
            }
            # Asynchronously update state (fire-and-forget or sampling)
            _MIDDLEWARE.update_metrics(real_metrics)
        except Exception as e:
            print(f"[Integrated MPC Feedback Error] {e}")

    return {
        "status": status,
        "latency_ms": duration,
        "task_id": task.get('id'),
        "mode": mode,
        "provider": event.get('provider'), # Echo back provider if any
        "debug": mpc_debug, # Return MPC debug info
        "strategy": strategy
    }
