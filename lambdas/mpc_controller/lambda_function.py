import json
import boto3
import time
import math
import os
import random
from decimal import Decimal
from botocore.config import Config
from src.wcp.wcp_update import wcp_update
from src.controllers.sinan_controller import SinanController
from src.controllers.aws_baseline_controller import AwsBaselineController
from src.wcp.wcp_update import slow_loop_calibration
from src.wcp.wcp_update import detect_trend
from src.mpc.controller import MPCController
from src.mpc.pricing import update_shadow_price

# --- DynamoDB Helper ---
dynamodb = boto3.client(
    'dynamodb',
    config=Config(
        connect_timeout=1,
        read_timeout=1,
        retries={'max_attempts': 2, 'mode': 'standard'},
        max_pool_connections=50,
    ),
)
TABLE_NAME = 'MPC_State'

_CONTROLLER = MPCController()

try:
    _CACHE_TTL_SEC = float(os.environ.get('MPC_STATE_CACHE_TTL_SEC', '5.0'))
except Exception:
    _CACHE_TTL_SEC = 5.0

_L1_CACHE = {
    'by_id': {}
}

def _cache_get(state_id):
    now = time.time()
    entry = _L1_CACHE.get('by_id', {}).get(state_id)
    if not entry:
        return None
    last_sync = float(entry.get('last_sync', 0.0))
    if entry.get('params') is not None and (now - last_sync) < _CACHE_TTL_SEC:
        return entry.get('params'), int(entry.get('version', 0))
    return None

def _cache_put(state_id, params, version):
    now = time.time()
    by_id = _L1_CACHE.setdefault('by_id', {})
    by_id[state_id] = {'params': params, 'version': int(version), 'last_sync': now}

def _remaining_ms(context):
    try:
        if context is None:
            return None
        return int(context.get_remaining_time_in_millis())
    except Exception:
        return None

def _should_write_state():
    try:
        rate = float(os.environ.get('MPC_STATE_WRITE_SAMPLE_RATE', '0.2'))
    except Exception:
        rate = 0.2
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate

def _merge_state(latest_state, computed_state):
    merged = dict(latest_state or {})
    allow = [
        'bP',
        'scores_l1',
        'rls_states',
        'last_prediction',
        'last_y',
        'sample_count',
        'wcp_alpha',
        'wcp_window',
        'shadow_price',
        'sp_eta',
        'sp_rho',
        'sp_lambda_max',
        'gamma',
        'last_alloc',
        'optimizer_weights',
        'priority_weights',
        'trend_state',
        'p90_prev',
        'p90_ewma',
        'slow_cooldown',
        'slow_stable_count',
        'slow_tense_count',
        'u_eta',
        'u_max_delta',
    ]
    for k in allow:
        if k in computed_state:
            merged[k] = computed_state[k]
    return merged

def get_state(state_id='global_params', include_scores=True):
    cached = _cache_get(state_id)
    if cached is not None:
        return cached
    try:
        expr_attr_names = {
            '#v': 'version',
            '#p': 'params',
            '#rs': 'rls_states',
            '#ow': 'optimizer_weights',
            '#pw': 'priority_weights',
            '#sp': 'shadow_price',
            '#cp': 'congestion_price',
            '#bP': 'bP',
            '#lp': 'last_prediction',
            '#ly': 'last_y',
            '#sc': 'sample_count',
            '#wa': 'wcp_alpha',
            '#ww': 'wcp_window',
            '#sco': 'scores_l1',
            '#se': 'sp_eta',
            '#sr': 'sp_rho',
            '#slm': 'sp_lambda_max',
            '#g': 'gamma',
            '#la': 'last_alloc',
        }
        projection = (
            "#v, #rs, #ow, #pw, #sp, #cp, "
            "#p.#bP, #p.#lp, #p.#ly, #p.#sc, #p.#wa, #p.#ww, "
            "#p.#se, #p.#sr, #p.#slm, #p.#g, #p.#la"
        )
        if include_scores:
            projection += ", #p.#sco"
        resp = dynamodb.get_item(
            TableName=TABLE_NAME,
            Key={'id': {'S': state_id}},
            ProjectionExpression=projection,
            ExpressionAttributeNames=expr_attr_names,
        )
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
                'congestion_price': get_float(item, 'congestion_price', 0.0),
                'sp_eta': get_float(params_map, 'sp_eta', 0.05),
                'sp_rho': get_float(params_map, 'sp_rho', 0.1),
                'sp_lambda_max': get_float(params_map, 'sp_lambda_max', 100.0),
                'gamma': get_float(params_map, 'gamma', 0.1),
                'last_alloc': get_float(params_map, 'last_alloc', 1.0),
                
                # MPC Weights
                'optimizer_weights': optimizer_weights,
                'priority_weights': priority_weights,
            }
            if not include_scores:
                params['scores_l1'] = []
            version = int(item.get('version', {'N': '0'}).get('N'))
            _cache_put(state_id, params, version)
            return params, version
        else:
            print("Item not found in DB")
            # Return defaults
            params = {
                'bP': 2000.0,
                'scores_l1': [],
                'rls_states': {},
                'priority_weights': {'lambda1': 0.6, 'alpha': 0.7, 'beta': 0.3, 'phi': [0.6, 0.4]},
                'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0},
                'error': 'Item not found'
            }
            _cache_put(state_id, params, 0)
            return params, 0
    except Exception as e:
        print(f"DB Read Error: {e}")
        stale = _L1_CACHE.get('by_id', {}).get(state_id)
        if stale and stale.get('params') is not None:
            return stale.get('params'), int(stale.get('version', 0))
        return {
            'bP': 2000.0,
            'scores_l1': [],
            'rls_states': {},
            'priority_weights': {'lambda1': 0.6, 'alpha': 0.7, 'beta': 0.3, 'phi': [0.6, 0.4]},
            'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0},
            'error': str(e)
        }, 0

def save_state(params, current_version, state_id='global_params'):
    max_window = 100

    try:
        full_snapshot_every = int(os.environ.get('MPC_STATE_FULL_SNAPSHOT_EVERY', '10'))
    except Exception:
        full_snapshot_every = 10
    if full_snapshot_every <= 0:
        full_snapshot_every = 1

    rls_json = json.dumps(params.get('rls_states', {}))

    weights = params.get('optimizer_weights', {'w1': 1.0, 'w2': 0.5, 'w3': 5.0})

    priority_weights = params.get('priority_weights', {})
    scores = params.get('scores_l1', []) or []
    trimmed_scores = scores[-max_window:]
    last_score = trimmed_scores[-1] if trimmed_scores else None
    sample_count = int(params.get('sample_count', 0))
    do_full_snapshot = (sample_count % full_snapshot_every) == 0

    expr_names = {
        '#params': 'params',
        '#bP': 'bP',
        '#scores_l1': 'scores_l1',
        '#last_prediction': 'last_prediction',
        '#last_y': 'last_y',
        '#sample_count': 'sample_count',
        '#wcp_alpha': 'wcp_alpha',
        '#wcp_window': 'wcp_window',
        '#sp_eta': 'sp_eta',
        '#sp_rho': 'sp_rho',
        '#sp_lambda_max': 'sp_lambda_max',
        '#gamma': 'gamma',
        '#last_alloc': 'last_alloc',
    }

    expr_values = {
        ':bP': {'N': str(params.get('bP', 2000.0))},
        ':last_prediction': {'L': [{'N': str(p)} for p in params.get('last_prediction', [])]},
        ':last_y': {'L': [{'N': str(y)} for y in params.get('last_y', [])]},
        ':sample_count': {'N': str(sample_count)},
        ':wcp_alpha': {'N': str(params.get('wcp_alpha', 0.1))},
        ':wcp_window': {'N': str(params.get('wcp_window', 100))},
        ':sp_eta': {'N': str(params.get('sp_eta', 0.05))},
        ':sp_rho': {'N': str(params.get('sp_rho', 0.1))},
        ':sp_lambda_max': {'N': str(params.get('sp_lambda_max', 100.0))},
        ':gamma': {'N': str(params.get('gamma', 0.1))},
        ':last_alloc': {'N': str(params.get('last_alloc', 1.0))},
        ':rls_states': {'S': rls_json},
        ':optimizer_weights': {'M': {
            'w1': {'N': str(weights.get('w1', 1.0))},
            'w2': {'N': str(weights.get('w2', 0.5))},
            'w3': {'N': str(weights.get('w3', 5.0))}
        }},
        ':priority_weights': {'M': {
            'lambda1': {'N': str(priority_weights.get('lambda1', 0.6))},
            'alpha': {'N': str(priority_weights.get('alpha', 0.7))},
            'beta': {'N': str(priority_weights.get('beta', 0.3))},
            'phi': {'L': [{'N': str(x)} for x in priority_weights.get('phi', [0.6, 0.4])]}
        }},
        ':shadow_price': {'N': str(params.get('shadow_price', 0.0))},
        ':congestion_price': {'N': str(params.get('congestion_price', params.get('shadow_price', 0.0)))},
        ':updatedAt': {'N': str(time.time())},
        ':empty_list': {'L': []},
    }

    if do_full_snapshot:
        expr_values[':scores_l1'] = {'L': [{'N': str(s)} for s in trimmed_scores]}
    else:
        if last_score is not None:
            expr_values[':score_one'] = {'L': [{'N': str(last_score)}]}

    try:
        set_parts = [
            "#params.#bP = :bP",
            "#params.#last_prediction = :last_prediction",
            "#params.#last_y = :last_y",
            "#params.#sample_count = :sample_count",
            "#params.#wcp_alpha = :wcp_alpha",
            "#params.#wcp_window = :wcp_window",
            "#params.#sp_eta = :sp_eta",
            "#params.#sp_rho = :sp_rho",
            "#params.#sp_lambda_max = :sp_lambda_max",
            "#params.#gamma = :gamma",
            "#params.#last_alloc = :last_alloc",
            "rls_states = :rls_states",
            "optimizer_weights = :optimizer_weights",
            "priority_weights = :priority_weights",
            "shadow_price = :shadow_price",
            "congestion_price = :congestion_price",
            "updatedAt = :updatedAt",
        ]
        if do_full_snapshot:
            set_parts.append("#params.#scores_l1 = :scores_l1")
            update_expr = "SET " + ", ".join(set_parts)
        else:
            update_expr = "SET " + ", ".join(set_parts)
            if last_score is not None:
                update_expr += ", #params.#scores_l1 = list_append(if_not_exists(#params.#scores_l1, :empty_list), :score_one)"

        resp = dynamodb.update_item(
            TableName=TABLE_NAME,
            Key={'id': {'S': state_id}},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
            ReturnValues='UPDATED_NEW',
        )

        _cache_put(state_id, params, int(current_version))
        return True, 0.0
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
    try:
        debug_every = int(os.environ.get('MPC_DEBUG_EVERY', '1'))
    except Exception:
        debug_every = 1
    if debug_every <= 0:
        debug_every = 1
    
    metrics = event.get('metrics', {'p90': 100, 'timeout_rate': 0.0, 'error_rate': 0.0, 'memory_pressure': 0.0})
    task = event.get('task', {'priority': 'standard', 'id': 'unknown'})
    
    controller = _CONTROLLER
    
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
    elif strategy == 'baseline':
        print("Strategy: AWS Baseline (Target Tracking)")
        # This requires state to be passed between invocations for cooldown logic.
        # For a stateless implementation, we'd need to persist last_scale times.
        # As a simplification for this integration, we instantiate it fresh.
        aws_controller = AwsBaselineController()
        # The baseline controller needs the *current* allocation to make a decision.
        # This is not available in the current event payload by default.
        # We will assume a default of 1.0 for this stateless implementation.
        current_alloc = float(event.get('last_alloc', 1.0))
        decision = aws_controller.get_decision(metrics, current_alloc)
        return {
            'decision': {
                'shouldShed': False,
                'degrade_plan': None,
                'resource_alloc': decision.get('cpu_cores', 1.0),
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

    elif strategy == 'sinan':
        print("Strategy: Sinan (Lit. 1)")
        # Use the simplified Sinan controller logic
        sinan_controller = SinanController(target_slo_p90_ms=float(event.get('slo_limit', 500.0)))
        decision = sinan_controller.get_decision(metrics)
        return {
            'decision': {
                'shouldShed': False,
                'degrade_plan': None,
                'resource_alloc': decision.get('cpu_cores', 1.0),
                'congestion_price': 0.0,
                'p90_prediction': 0.0,
                'uncertainty': 0.0
            },
            'meta': {'mode': 'sinan'}
        }

    wcp_mode = event.get('wcp_mode', 'strict')
    include_scores = (wcp_mode in {'strict'})

    start_time = time.time()
    state, version = get_state(state_id, include_scores=include_scores)

    try:
        state['sample_count'] = int(state.get('sample_count', 0)) + 1
    except Exception:
        state['sample_count'] = 1

    step = int(state.get('sample_count', 0))
    should_log = (step % debug_every) == 0
    if should_log:
        print("Received event")
    
    weights = state.get('optimizer_weights', {})
    controller.optimizer.w1 = weights.get('w1', 1.0)
    controller.optimizer.w2 = weights.get('w2', 0.5)
    controller.optimizer.w3 = weights.get('w3', 5.0)
    pw = state.get('priority_weights', {})
    controller.priority_mgr.lambda1 = pw.get('lambda1', 0.6)
    controller.priority_mgr.alpha = pw.get('alpha', 0.7)
    controller.priority_mgr.beta = pw.get('beta', 0.3)
    controller.priority_mgr.phi = pw.get('phi', [0.6, 0.4])
    
    if should_log:
        print(f"Running WCP Mode: {wcp_mode} (Enforced Strict)")
    
    wcp_start = time.time()
    # Optimization: Enforce strict mode as confirmed by user requirements
    pred_dict, uncertainty, debug_info = wcp_update(state, metrics, alpha=0.1)
    if isinstance(debug_info, dict) and 'mode' not in debug_info:
        debug_info['mode'] = 'strict'
    wcp_duration = (time.time() - wcp_start) * 1000
    
    wcp_constraints = {'pred': pred_dict, 'uncertainty': uncertainty}
    
    system_state = {
        'shadow_price': state.get('shadow_price', 0.0),
        'cpu_util': metrics.get('cpu_util', 0.5),
        'last_prediction': state.get('last_prediction', []),
        'p90_latency': metrics.get('p90', 0.0),
        'last_alloc': float(event.get('last_alloc', state.get('last_alloc', 1.0))),
        'gamma': state.get('gamma', 0.1),
        'u_eta': state.get('u_eta', 0.05),
        'u_max_delta': state.get('u_max_delta', 0.15),
        'slo_limit': float(event.get('slo_limit', state.get('slo_limit', 500.0)))
    }
    
    lam, _ = update_shadow_price(state, metrics, system_state['last_alloc'])
    system_state['shadow_price'] = lam
    state['shadow_price'] = lam
    
    result = controller.decide(task, wcp_constraints, system_state)
    
    if enable_feedback:
        feedback_metrics = {
            'slo_violation_rate': metrics.get('slo_violation_rate', 0.0),
            'resource_waste_rate': metrics.get('resource_waste_rate', 0.0)
        }
        updates = controller.update_feedback(feedback_metrics, state)
        state['optimizer_weights'] = updates.get('optimizer_weights', {})
        state['priority_weights'] = updates.get('priority_weights', {})
        try:
            trend_every = int(os.environ.get('MPC_TREND_EVERY', '10'))
        except Exception:
            trend_every = 10
        if trend_every <= 0:
            trend_every = 1
        if (step % trend_every) == 0:
            detect_trend(state, metrics)

        try:
            slow_every = int(os.environ.get('MPC_SLOW_LOOP_EVERY', '15'))
        except Exception:
            slow_every = 15
        if slow_every <= 0:
            slow_every = 1

        if (step % slow_every) == 0:
            rem = _remaining_ms(context)
            if rem is None or rem > 500:
                slow_loop_calibration(state, metrics)
    else:
        updates = None
    
    if result and 'decision' in result:
        result['debug'] = {
            'prev_u': system_state['last_alloc'],
            'slo_limit': system_state.get('slo_limit'),
            'price': system_state.get('shadow_price')
        }
        state['last_alloc'] = result['decision']['resource_alloc']
    
    persisted = False
    state_size_kb = 0.0
    if _should_write_state():
        rem = _remaining_ms(context)
        if rem is None or rem > 600:
            ok, sz = save_state(state, version, state_id)
            if ok:
                persisted = True
                state_size_kb = sz
    
    total_duration = (time.time() - start_time) * 1000
    out = {
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
            'state_size_kb': state_size_kb,
            'persisted': persisted
        }
    }
    if updates is not None:
        out['overhead']['feedback_applied'] = True
    else:
        out['overhead']['feedback_applied'] = False
    if not persisted:
        out['error'] = 'State persistence skipped_or_failed'
    return out
