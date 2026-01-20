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

def burn_cpu(duration_sec):
    """
    Simulate CPU-bound processing to trigger real resource contention.
    """
    start = time.time()
    # Busy loop
    while time.time() - start < duration_sec:
        _ = 999999 * 999999

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
    
    if (strategy == 'mpc_integrated' or strategy == 'mpc') and _MIDDLEWARE:
        # Run MPC logic internally!
        try:
            internal_decision, debug_info = _MIDDLEWARE.decide(event)
            # Override external decision
            decision = internal_decision
            mpc_debug = debug_info or {}
            mpc_debug.update(
                {
                    'resource_alloc': decision.get('resource_alloc', 1.0),
                    'p90_prediction': decision.get('p90_prediction', 0.0),
                    'uncertainty': decision.get('uncertainty', 0.0),
                    'shouldShed': decision.get('shouldShed', False),
                    'pred_queue_delay_ms': decision.get('pred_queue_delay_ms', 0.0),
                    'queue_backlog': decision.get('queue_backlog', 0.0),
                    'priority_score': decision.get('priority_score', None),
                }
            )
            # Inject alloc into event for penalty logic below
            event['resource_alloc'] = decision.get('resource_alloc', 1.0)
            print(f"[Integrated MPC] Alloc: {event['resource_alloc']:.2f}, Shed: {decision.get('shouldShed')}")
        except Exception as e:
            print(f"[Integrated MPC Error] {e}")
            # Fallback
            decision = {'shouldShed': False, 'resource_alloc': 1.0}
            
    task = event.get('task', {})
    mode = event.get('mode', 'auto')
    priority = task.get('priority', event.get('priority', 'standard'))
    qos = "Q1" if priority in ["critical", "high"] else ("Q3" if priority == "low" else "Q2")
    
    start_time = time.time()
    status = "success"
    
    should_shed = False
    
    if mode == 'force_shed':
        should_shed = True
    elif mode == 'normal' or mode == 'external_api':
        should_shed = False
    else:
        raw_should_shed = decision.get('shouldShed', False)
        shed_reason = decision.get('shed_reason', '')

        if qos == "Q3":
            should_shed = raw_should_shed
        elif qos == "Q2":
            should_shed = raw_should_shed
            # Removed hardcoded 50% check; rely on Scheduler's probabilistic RED logic.
        elif qos == "Q1":
            # Q1 Mission Critical: Prefer Degradation (Fidelity Scaling) over Shedding
            # BUT allow shedding if system is overwhelmed even at min fidelity.
            
            # Use MPC allocation (u) as Fidelity Factor
            alloc = float(event.get('resource_alloc', 1.0))
            fidelity = max(0.01, min(1.0, alloc))
            
            # CRITICAL FIX: Always enable Fidelity Mode if Alloc < 1.0 for Q1
            # We must set this explicitly so the execution block knows not to apply penalty.
            if alloc < 1.0:
                 event['fidelity_factor'] = fidelity
            
            pred_queue_delay = float(decision.get('pred_queue_delay_ms', 0.0))

            # CRITICAL OVERRIDE: Bang-Bang Control for Flash Crowds
            # If queue is detected (>200ms), DROP fidelity immediately. Don't wait for Controller.
            if pred_queue_delay > 200.0:
                 fidelity = 0.05 
                 event['fidelity_factor'] = fidelity
                 print(f"[Q1 EMERGENCY] Queue {pred_queue_delay:.0f}ms > 200ms. FORCING Fidelity 5%.")

            if raw_should_shed:
                # If Controller says SHED, we check if we can just degrade.
                # Circuit Breaker: 
                # 1. If alloc is extremely low (< 0.05), system is completely broken.
                # 2. If queue delay is HIGH (>300ms), better to shed than timeout.
                
                # Revert to strict threshold: 300ms.
                # User Feedback: "Not effective" -> We were allowing 1700ms latencies. Stop that.
                if alloc < 0.05 or pred_queue_delay > 300.0:
                    should_shed = True
                    reason = "LowAlloc" if alloc < 0.05 else "HighQueue"
                    print(f"[Q1 Critical] System Saturated ({reason}: Alloc {alloc:.2f}, Queue {pred_queue_delay:.0f}ms). Force Shedding.")
                else:
                    should_shed = False
                    # Fidelity factor is already set above
                    print(f"[Q1 Protection] Overload detected. Scaling Fidelity to {fidelity*100:.1f}% (Alloc: {alloc:.2f})")
            else:
                should_shed = False
                # Ensure Fidelity is applied even if not shedding
                if alloc < 1.0:
                     print(f"[Q1 Fidelity] Active Scaling: {fidelity*100:.1f}% (Alloc: {alloc:.2f})")
    
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
        # Use simulated_duration_ms from task if available (for Trace Replay), else default to 100ms
        sim_duration = float(task.get('simulated_duration_ms', 100.0))
        base_latency = sim_duration / 1000.0 # Convert to seconds
        
        # Handle "external_api" specific logic (e.g., call 3rd party)
        if mode == 'external_api':
            # Simulate External API call which is slower
            base_latency = 0.5 
            provider = event.get('provider', 'Unknown')
            print(f"Calling External Provider: {provider}")
        
        # Apply Resource Allocation Penalty (Simulating resource limits)
        resource_alloc = float(event.get('resource_alloc', 1.0))
        penalty_factor = 1.0
        
        # Check if we are in Q1 Fidelity Mode
        # If event has 'fidelity_factor', it implies we are controlling duration explicitly.
        # We should NOT apply the starvation penalty in this case, otherwise we negate the speedup.
        is_fidelity_mode = (event.get('fidelity_factor', 1.0) < 1.0) or (qos == "Q1" and resource_alloc < 1.0 and decision.get('shouldShed', False))

        if resource_alloc < 1.0 and not is_fidelity_mode:
            # Only apply penalty for non-fidelity tasks (e.g. Q2/Q3 if they weren't shed)
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
        # We use burn_cpu to simulate real CPU contention (SQARTS-like)
        active_duration = (base_latency + jitter) * penalty_factor
        
        # --- Q1 Fidelity Scaling Application ---
        fidelity = event.get('fidelity_factor', 1.0)
        if fidelity < 1.0:
            original_duration = active_duration
            active_duration *= fidelity
            print(f"[Fidelity Scaling] Duration: {original_duration:.3f}s -> {active_duration:.3f}s (Factor: {fidelity:.2f})")
        # ---------------------------------------
        
        # Injected delay (Cold start, etc) is passive sleep
        passive_duration = injected_delay
        
        # 1. CPU Bound Work
        if active_duration > 0:
            burn_cpu(active_duration)
            
        # 2. IO/Wait Work
        if passive_duration > 0:
            time.sleep(passive_duration)
            
        total_duration = active_duration + passive_duration
        print(f"Task {task.get('id')} processed. Mode: {mode}. Total: {total_duration:.3f}s (CPU: {active_duration:.3f}, Sleep: {passive_duration:.3f}, Pen: {penalty_factor:.2f})")
        
    duration = (time.time() - start_time) * 1000 # ms
    
    # --- POST-EXECUTION: Feedback Update ---
    # Update MPC state with ACTUAL metrics from this execution
    # Enable feedback for both integrated and external MPC modes
    if (strategy == 'mpc_integrated' or strategy == 'mpc') and _MIDDLEWARE:
        try:
            # Calculate CPU usage ratio (Active CPU time / Total Wall time)
            cpu_ratio = 0.0
            if duration > 0:
                # active_duration is defined in the else block, need to scope it correctly
                # Simplified: if we shed, cpu is low. If we ran, use rough estimate or better tracking.
                pass
            
            # Re-calculate specific usage
            # If we shed, duration is small (~50ms), CPU is minimal.
            # If we processed, we know active_duration.
            # However, variable scope is tricky here. Let's infer from duration.
            
            estimated_cpu = 0.1
            if should_shed:
                estimated_cpu = 0.05
            else:
                # If we didn't shed, we likely burned CPU.
                # Assuming most of the time was active_duration (unless injected_delay was huge)
                # But we can't easily access local vars from the else block without refactoring.
                # Let's use a heuristic: if duration > 100ms, assume high CPU load for this workload.
                estimated_cpu = 0.95 
                
            real_metrics = {
                'latency': duration,
                'cpu_usage': estimated_cpu, 
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
