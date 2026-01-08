import json
import boto3
import time
import math
import os
import random
from decimal import Decimal
from src.wcp.wcp_update import wcp_update
from src.wcp.wcp_update import slow_loop_calibration
from src.wcp.wcp_update import detect_trend
from src.wcp.wcp_baseline import wcp_baseline_update
from src.wcp.wcp_simple import wcp_simple_update
from src.wcp.wcp_lite_a import wcp_lite_a_update
from src.wcp.wcp_lite_b import wcp_lite_b_update
from src.mpc.controller import MPCController
from src.mpc.pricing import update_shadow_price

# --- DynamoDB Helper ---
dynamodb = boto3.client('dynamodb')
TABLE_NAME = 'MPC_State'

def get_state(state_id='global_params'):
    try:
        resp = dynamodb.get_item(TableName=TABLE_NAME, Key={'id': {'S': state_id}})
        if 'Item' in resp:
            item = resp['Item']
            # Parse simple structure safely
            params_map = item.get('params', {}).get('M', {})
            
            # Helper to get float
            def get_float(m, k, default):
                try:
                    return float(m.get(k, {}).get('N', default))
                except:
                    return float(default)
            
            # Helper to get list of floats
            def get_list(m, k):
                try:
                    l = m.get(k, {}).get('L', [])
                    return [float(x.get('N', '0')) for x in l]
                except:
                    return []

            # Parse RLS states (stored as JSON string for simplicity)
            rls_states_json = item.get('rls_states', {}).get('S', '{}')
            try:
                rls_states = json.loads(rls_states_json)
            except:
                rls_states = {}
            
            # Parse Optimizer Weights (New for MPC)
            weights_map = item.get('optimizer_weights', {}).get('M', {})
            optimizer_weights = {
                'w1': get_float(weights_map, 'w1', 1.0),
                'w2': get_float(weights_map, 'w2', 0.5),
                'w3': get_float(weights_map, 'w3', 5.0)
            }
            pr_map = item.get('priority_weights', {}).get('M', {})
            # Helper for list
            def get_phi(m, k):
                try:
                    l = m.get(k, {}).get('L', [])
                    if not l: return [0.6, 0.4]
                    return [float(x.get('N', '0.5')) for x in l]
                except:
                    return [0.6, 0.4]

            priority_weights = {
                'lambda1': get_float(pr_map, 'lambda1', 0.6),
                'alpha': get_float(pr_map, 'alpha', 0.7),
                'beta': get_float(pr_map, 'beta', 0.3),
                'phi': get_phi(pr_map, 'phi')
            }

            params = {
                'bP': get_float(params_map, 'bP', 2000.0),
                # New Strict WCP fields
                'scores_l1': get_list(params_map, 'scores_l1'),
                'rls_states': rls_states,
                'last_prediction': get_list(params_map, 'last_prediction'),
                'last_y': get_list(params_map, 'last_y'),
                'sample_count': int(get_float(params_map, 'sample_count', 0.0)),
                'wcp_alpha': get_float(params_map, 'wcp_alpha', 0.1),
                'wcp_window': int(get_float(params_map, 'wcp_window', 100.0)),
                'shadow_price': get_float(item, 'shadow_price', 0.0),
                'sp_eta': get_float(params_map, 'sp_eta', 0.05),
                'sp_rho': get_float(params_map, 'sp_rho', 0.1),
                'sp_lambda_max': get_float(params_map, 'sp_lambda_max', 100.0),
                'gamma': get_float(params_map, 'gamma', 0.1),
                'last_alloc': get_float(params_map, 'last_alloc', 1.0),
                
                # MPC Weights
                'optimizer_weights': optimizer_weights,
                'priority_weights': priority_weights,
                
                # Legacy fields (kept for safety)
                'congestion_price': get_float(item, 'congestion_price', 0.0),
                'theta': get_float(params_map, 'theta', 1.0),
            }
            version = int(item.get('version', {'N': '0'}).get('N'))
            return params, version
        else:
            print("Item not found in DB")
            # Return defaults
            return {
                'bP': 2000.0,
                'scores_l1': [],
                'rls_states': {},
                'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0},
                'congestion_price': 0.0,
                'error': 'Item not found'
            }, 0
    except Exception as e:
        print(f"DB Read Error: {e}")
        # Return defaults but signal error in logs
        return {
            'bP': 2000.0,
            'scores_l1': [],
            'rls_states': {},
            'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0},
            'congestion_price': 0.0,
            'error': str(e) # Pass error to caller
        }, 0

def save_state(params, current_version, state_id='global_params'):
    # Keep scores window small to fit in 400KB Item limit
    max_window = 100
    
    new_version = current_version + 1
    
    # Serialize RLS states
    rls_json = json.dumps(params.get('rls_states', {}))
    
    weights = params.get('optimizer_weights', {'w1': 1.0, 'w2': 0.5, 'w3': 5.0})
    
    item = {
        'id': {'S': state_id},
        'params': {'M': {
            'bP': {'N': str(params.get('bP', 2000.0))},
            'theta': {'N': str(params.get('theta', 1.0))},
            
            # New Strict WCP fields
            'scores_l1': {'L': [{'N': str(s)} for s in params.get('scores_l1', [])[-max_window:]]},
            'last_prediction': {'L': [{'N': str(p)} for p in params.get('last_prediction', [])]},
            'last_y': {'L': [{'N': str(y)} for y in params.get('last_y', [])]},
            'sample_count': {'N': str(params.get('sample_count', 0))},
            'wcp_alpha': {'N': str(params.get('wcp_alpha', 0.1))},
            'wcp_window': {'N': str(params.get('wcp_window', 100))},
            'sp_eta': {'N': str(params.get('sp_eta', 0.05))},
            'sp_rho': {'N': str(params.get('sp_rho', 0.1))},
            'sp_lambda_max': {'N': str(params.get('sp_lambda_max', 100.0))},
            'gamma': {'N': str(params.get('gamma', 0.1))},
            'last_alloc': {'N': str(params.get('last_alloc', 1.0))},
        }},
        'rls_states': {'S': rls_json},
        
        # Save MPC Weights
        'optimizer_weights': {'M': {
            'w1': {'N': str(weights.get('w1', 1.0))},
            'w2': {'N': str(weights.get('w2', 0.5))},
            'w3': {'N': str(weights.get('w3', 5.0))}
        }},
        'priority_weights': {'M': {
            'lambda1': {'N': str(params.get('priority_weights', {}).get('lambda1', 0.6))},
            'alpha': {'N': str(params.get('priority_weights', {}).get('alpha', 0.7))},
            'beta': {'N': str(params.get('priority_weights', {}).get('beta', 0.3))},
            'phi': {'L': [{'N': str(x)} for x in params.get('priority_weights', {}).get('phi', [0.6, 0.4])]}
        }},
        
        'shadow_price': {'N': str(params.get('shadow_price', 0.0))},
        'updatedAt': {'N': str(time.time())},
        'version': {'N': str(new_version)}
    }
    
    try:
        if current_version == 0:
            # First creation
            dynamodb.put_item(
                TableName=TABLE_NAME, 
                Item=item,
                ConditionExpression='attribute_not_exists(id)'
            )
        else:
            # Optimistic Locking
            dynamodb.put_item(
                TableName=TABLE_NAME, 
                Item=item,
                ConditionExpression='version = :v',
                ExpressionAttributeValues={':v': {'N': str(current_version)}}
            )
        
        # Calculate size
        size_kb = len(json.dumps(item)) / 1024.0
        return True, size_kb
    except dynamodb.exceptions.ConditionalCheckFailedException:
        return False, 0
    except Exception as e:
        print(f"DB Write Error: {e}")
        return False, 0

# --- Core MPC Logic ---
def lambda_handler(event, context):
    """
    Event format expected:
    {
        "metrics": { "p90": 120, "success_rate": 98.5, "slo_violation_rate": 0.01, "resource_waste_rate": 0.1 },
        "task": { "priority": "critical", "id": "123" } 
    }
    """
    print("Received event:", json.dumps(event))
    
    metrics = event.get('metrics', {'p90': 100, 'timeout_rate': 0.0, 'error_rate': 0.0, 'memory_pressure': 0.0})
    task = event.get('task', {'priority': 'standard', 'id': 'unknown'})
    
    # Initialize Controller
    # In a real warm Lambda, we might want to cache this globally, 
    # but for safety with state hydration, we init per request or hydrate it.
    controller = MPCController()
    
    # Check Control Strategy
    strategy = event.get('strategy', 'mpc') # 'mpc', 'static', 'baseline'
    enable_feedback = event.get('enable_feedback', True)
    state_id = event.get('state_id', 'global_params')
    
    # If not MPC, we can return early or mock the result
    if strategy == 'baseline':
        print("Strategy: Baseline (No MPC)")
        return {
            'decision': {
                'shouldShed': False,
                'degrade_plan': None,
                'resource_alloc': 1.0, # Full resource
                'congestion_price': 0.0,
                'p90_prediction': 0.0,
                'uncertainty': 0.0
            },
            'meta': {'mode': 'baseline'}
        }
    elif strategy == 'static':
        print("Strategy: Static Priority")
        p = task.get('priority', 'standard')
        if p == 'critical': alloc = 1.0
        elif p == 'high': alloc = 0.8
        else: alloc = 0.5
        return {
            'decision': {
                'shouldShed': False,
                'degrade_plan': None,
                'resource_alloc': alloc,
                'congestion_price': 0.0,
                'p90_prediction': 0.0,
                'uncertainty': 0.0
            },
            'meta': {'mode': 'static'}
        }

    max_retries = 5
    
    for attempt in range(max_retries):
        start_time = time.time()
        
        # 1. Sensing & State Sync
        state, version = get_state(state_id)
        
        # Hydrate Controller Weights
        weights = state.get('optimizer_weights', {})
        controller.optimizer.w1 = weights.get('w1', 1.0)
        controller.optimizer.w2 = weights.get('w2', 0.5)
        controller.optimizer.w3 = weights.get('w3', 5.0)
        pw = state.get('priority_weights', {})
        controller.priority_mgr.lambda1 = pw.get('lambda1', 0.6)
        controller.priority_mgr.alpha = pw.get('alpha', 0.7)
        controller.priority_mgr.beta = pw.get('beta', 0.3)
        controller.priority_mgr.phi = pw.get('phi', [0.6, 0.4])
        
        # 2. Select WCP Mode and Update
        wcp_mode = event.get('wcp_mode', 'strict')
        print(f"Running WCP Mode: {wcp_mode}")
        
        wcp_start = time.time()
        if wcp_mode == 'baseline':
            pred_dict, uncertainty, debug_info = wcp_baseline_update(state, metrics, alpha=0.1)
        elif wcp_mode == 'simple':
            pred_dict, uncertainty, debug_info = wcp_simple_update(state, metrics, alpha=0.1)
        elif wcp_mode == 'lite_a':
            pred_dict, uncertainty, debug_info = wcp_lite_a_update(state, metrics, alpha=0.1)
        elif wcp_mode == 'lite_b':
            pred_dict, uncertainty, debug_info = wcp_lite_b_update(state, metrics, alpha=0.1)
        elif wcp_mode == 'none':
            # Pure MPC Mode: Use RLS for prediction (via wcp_update), but ignore Uncertainty
            # This allows us to test the "Point Prediction" capability of RLS without WCP safety padding.
            pred_dict, original_uncertainty, debug_info = wcp_update(state, metrics, alpha=0.1)
            uncertainty = 0.0
            debug_info['mode'] = 'pure_mpc_rls_only'
            debug_info['ignored_uncertainty'] = original_uncertainty
        else:
            # Strict (Default)
            pred_dict, uncertainty, debug_info = wcp_update(state, metrics, alpha=0.1)
        wcp_duration = (time.time() - wcp_start) * 1000 # ms
        
        # 3. Prepare Constraints for MPC
        # WCP Output -> MPC Constraints
        wcp_constraints = {'pred': pred_dict, 'uncertainty': uncertainty}
        
        # System State for MPC
        system_state = {
            'shadow_price': state.get('shadow_price', 0.0),
            'cpu_util': metrics.get('cpu_util', 0.5), # Assuming metrics has this or default
            # Pass history for trajectory if needed
            'last_prediction': state.get('last_prediction', []),
            'p90_latency': metrics.get('p90', 0.0),
            'last_alloc': float(event.get('last_alloc', state.get('last_alloc', 1.0))),
            'gamma': state.get('gamma', 0.1),
            'u_eta': state.get('u_eta', 0.05),
            'u_max_delta': state.get('u_max_delta', 0.15),
            'slo_limit': float(event.get('slo_limit', state.get('slo_limit', 500.0)))
        }
        
        lam, lam_dbg = update_shadow_price(state, metrics, system_state['last_alloc'])
        system_state['shadow_price'] = lam
        state['shadow_price'] = lam
        
        # 4. MPC Decision
        result = controller.decide(task, wcp_constraints, system_state)
        
        # 5. Closed-loop Feedback Update
        # Extract feedback metrics from event
        if enable_feedback:
            feedback_metrics = {
                'slo_violation_rate': metrics.get('slo_violation_rate', 0.0),
                'resource_waste_rate': metrics.get('resource_waste_rate', 0.0)
            }
            updates = controller.update_feedback(feedback_metrics, state)
            
            # Update State with new weights
            state['optimizer_weights'] = updates.get('optimizer_weights', {})
            state['priority_weights'] = updates.get('priority_weights', {})
            detect_trend(state, metrics)
            slow_loop_calibration(state, metrics)
            print(f"Feedback Enabled. Updates: {updates}")
        else:
            print("Feedback Disabled. Skipping weight update.")
        
        # Update State with WCP results (already done by wcp_update but we need to ensure state is clean)
        # Note: wcp_update modifies 'state' in place mostly, but let's be sure.
        
        if result and 'decision' in result:
            # Add debug info to response
            result['debug'] = {
                'prev_u': system_state['last_alloc'],
                'slo_limit': system_state.get('slo_limit'),
                'price': system_state.get('shadow_price')
            }
            
            state['last_alloc'] = result['decision']['resource_alloc']
        
        # 6. Persist State (Optimistic Locking)
        success, state_size_kb = save_state(state, version, state_id)
        if success:
            print(f"State saved successfully. Version: {version+1}")
            
            # Return final result with MPC decision
            total_duration = (time.time() - start_time) * 1000 # ms
            
            return {
                'decision': {
                    'shouldShed': result['decision']['should_shed'],
                    'degrade_plan': result['decision']['degrade_plan'],
                    'resource_alloc': result['decision']['resource_alloc'],
                    'shadow_price': system_state['shadow_price'],
                    'p90_prediction': pred_dict.get('p90', 0.0),
                    'uncertainty': uncertainty
                },
                'meta': result['meta'],
                'wcp_debug': debug_info,
                'overhead': {
                    'wcp_compute_ms': wcp_duration,
                    'total_ms': total_duration,
                    'state_size_kb': state_size_kb
                }
            }
        else:
            print(f"Optimistic locking failed (Version {version}). Retrying {attempt+1}/{max_retries}...")
            time.sleep(random.uniform(0.05, 0.2)) # Jitter
            
    # If we get here, DB writes failed repeatedly
    print("Max retries exceeded for DB write.")
    # Return a safe fallback decision
    return {
        'decision': {
            'shouldShed': True, # Fail safe
            'degrade_plan': 'fallback_shed',
            'resource_alloc': 0.5,
            'congestion_price': 999.0
        },
        'error': 'State persistence failed'
    }
