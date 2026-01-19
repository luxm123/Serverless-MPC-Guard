import time
import json
import boto3
import os
import random
import threading
from botocore.config import Config
from src.mpc.controller import MPCController
from src.wcp.wcp_update import wcp_update, detect_trend, slow_loop_calibration
from src.mpc.pricing import update_shadow_price

# --- Global L1 Cache (Container Re-use) ---
_L1_CACHE = {
    'params': None,
    'version': 0,
    'last_sync': 0,
    'last_backlog': None,
    'last_backlog_sync': 0,
    'updating_backlog': False
}
CACHE_TTL = 5.0 # seconds. Refresh from DB if older than this.

# AWS Clients
dynamodb = boto3.client(
    'dynamodb',
    config=Config(
        connect_timeout=1,
        read_timeout=1,
        retries={'max_attempts': 2, 'mode': 'standard'},
    ),
)
sqs = boto3.client('sqs')

TABLE_NAME = 'MPC_State'
UPDATE_QUEUE_URL = os.environ.get('MPC_UPDATE_QUEUE_URL') # Optional: For async updates
MAIN_QUEUE_URL = os.environ.get('MPC_MAIN_QUEUE_URL') or os.environ.get('QUEUE_URL')

class MPCMiddleware:
    def __init__(self, state_id='global_params'):
        self.state_id = state_id
        self.controller = MPCController()
        
    def _fetch_backlog_async(self):
        """
        Fetch SQS backlog in a background thread to avoid blocking the main path.
        """
        global _L1_CACHE
        try:
            if not MAIN_QUEUE_URL:
                return
            r = sqs.get_queue_attributes(
                QueueUrl=MAIN_QUEUE_URL,
                AttributeNames=['ApproximateNumberOfMessages'],
            )
            v = r.get('Attributes', {}).get('ApproximateNumberOfMessages', '0')
            val = float(v)
            _L1_CACHE['last_backlog'] = val
            _L1_CACHE['last_backlog_sync'] = time.time()
        except Exception as e:
            print(f"Async Backlog Fetch Error: {e}")
        finally:
            _L1_CACHE['updating_backlog'] = False

    def _load_state(self):
        """
        Load state from L1 Cache or DynamoDB.
        """
        global _L1_CACHE
        now = time.time()
        
        # 1. Try L1 Cache
        if _L1_CACHE['params'] and (now - _L1_CACHE['last_sync'] < CACHE_TTL):
            return _L1_CACHE['params'], _L1_CACHE['version']
            
        # 2. Fetch from DynamoDB
        try:
            resp = dynamodb.get_item(TableName=TABLE_NAME, Key={'id': {'S': self.state_id}})
            if 'Item' in resp:
                item = resp['Item']
                params = self._parse_dynamo_item(item)
                version = int(item.get('version', {'N': '0'}).get('N'))
                
                # Update Cache
                _L1_CACHE['params'] = params
                _L1_CACHE['version'] = version
                _L1_CACHE['last_sync'] = now
                
                return params, version
        except Exception as e:
            print(f"MPC Middleware DB Read Error: {e}")
            if _L1_CACHE['params']:
                return _L1_CACHE['params'], _L1_CACHE['version'] # Fallback to stale cache
                
        # 3. Default Fallback
        return self._get_default_params(), 0

    def decide(self, event):
        """
        Main entry point for decision making.
        Returns: (decision_dict, debug_info)
        """
        start_t = time.time()
        
        # Parse Input
        metrics = event.get('metrics', {})
        task = event.get('task', {})
        # If client didn't send metrics (Real Scenario), use our belief from state
        state, version = self._load_state()
        qos_raw = task.get('qos', task.get('priority'))
        qos = 'Q3'
        if isinstance(qos_raw, str):
            s = qos_raw.strip().lower()
            if s in ('critical', 'q1', 'high'):
                qos = 'Q1'
            elif s in ('standard', 'q2', 'medium'):
                qos = 'Q2'
        task['qos_class'] = qos
        
        if not metrics or 'p90' not in metrics:
             metrics = metrics.copy()
             metrics['p90'] = float(state.get('p90_belief', 100.0))
             metrics['cpu_usage'] = 0.8 if metrics['p90'] > 500 else 0.2
        
        # Hydrate Controller
        self._hydrate_controller(state)
        
        # WCP Update (Prediction)
        # Note: We run WCP locally for prediction, but we don't save RLS state synchronously
        # to avoid high latency. RLS state changes are small and can be approximated or
        # pushed asynchronously.
        pred_dict, uncertainty, debug_info = wcp_update(state, metrics, alpha=0.2)
        
        # MPC Constraints
        wcp_constraints = {'pred': pred_dict, 'uncertainty': uncertainty}

        queue_backlog = metrics.get('queue_backlog', metrics.get('queue', None))
        backlog_source = 'metrics'
        if queue_backlog is None:
            queue_backlog = state.get('queue_backlog_belief', None)
            backlog_source = 'state'
        if queue_backlog is None and MAIN_QUEUE_URL:
            now = time.time()
            ttl_s = float(state.get('queue_backlog_ttl_s', 2.0) or 2.0)
            if _L1_CACHE.get('last_backlog') is not None and (now - float(_L1_CACHE.get('last_backlog_sync', 0) or 0)) < ttl_s:
                queue_backlog = _L1_CACHE.get('last_backlog')
                backlog_source = 'sqs_cache'
            else:
                # Cache Miss/Expired -> Trigger Async Update
                if not _L1_CACHE.get('updating_backlog'):
                    _L1_CACHE['updating_backlog'] = True
                    t = threading.Thread(target=self._fetch_backlog_async)
                    t.daemon = True
                    t.start()
                
                # Use Stale Data (Fail-Soft)
                if _L1_CACHE.get('last_backlog') is not None:
                    queue_backlog = _L1_CACHE.get('last_backlog')
                    backlog_source = 'sqs_stale'
                else:
                    queue_backlog = 0.0 # Cold start
                    backlog_source = 'default_cold'
        queue_backlog = float(queue_backlog or 0.0)

        unc_p90 = 0.0
        if isinstance(uncertainty, dict):
            unc_p90 = float(uncertainty.get('p90', 0.0) or 0.0)
        else:
            try:
                unc_p90 = float(uncertainty or 0.0)
            except Exception:
                unc_p90 = 0.0

        last_alloc = float(state.get('last_alloc', 1.0) or 1.0)
        servers = metrics.get('concurrency', metrics.get('servers', None))
        if servers is None:
            servers = state.get('buffer_servers', state.get('concurrency_belief', state.get('buffer_servers_default', 1.0)))
        servers = max(1.0, float(servers or 1.0))

        base_service_ms = float(state.get('avg_service_ms', 0.0) or 0.0)
        if base_service_ms <= 0.0:
            base_service_ms = float(state.get('p90_belief', 0.0) or 0.0)
        if base_service_ms <= 0.0:
            base_service_ms = float(metrics.get('p90', 0.0) or 0.0)
        if base_service_ms <= 0.0:
            base_service_ms = float(pred_dict.get('p90', 100.0) or 100.0)
        eff_alloc = max(0.1, last_alloc)
        eff_service_ms = base_service_ms / eff_alloc
        queue_delay_model = str(state.get('queue_delay_model', 'backlog_linear') or 'backlog_linear').strip().lower()
        if queue_delay_model == 'backlog_linear':
            queue_delay_ms = (queue_backlog * eff_service_ms) / servers
        else:
            queue_delay_ms = (queue_backlog * eff_service_ms) / servers
        
        # System State
        qos_slo_map = {'Q1': 1000.0, 'Q2': 1800.0, 'Q3': 3000.0}
        slo_limit_ms = float(qos_slo_map.get(qos, float(event.get('slo_limit', state.get('slo_limit', 1000.0)))))
        q1_violation_rate = float(metrics.get('q1_violation_rate', 0.0) or 0.0)
        q2_violation_rate = float(metrics.get('q2_violation_rate', 0.0) or 0.0)
        q3_violation_rate = float(metrics.get('q3_violation_rate', 0.0) or 0.0)
        q1_drop_rate = float(metrics.get('q1_worker_drop_rate', 0.0) or 0.0)
        q2_drop_rate = float(metrics.get('q2_worker_drop_rate', 0.0) or 0.0)
        q3_drop_rate = float(metrics.get('q3_worker_drop_rate', 0.0) or 0.0)

        system_state = {
            'shadow_price': state.get('shadow_price', 0.0),
            'last_alloc': last_alloc,
            'gamma': state.get('gamma', 0.1),
            'u_eta': state.get('u_eta', 0.05),
            'u_max_delta': state.get('u_max_delta', 0.15),
            'slo_limit': slo_limit_ms,
            'pred_queue_delay_ms': queue_delay_ms,
            'prio_cl_w_latency': state.get('prio_cl_w_latency', state.get('priority_weights', {}).get('prio_cl_w_latency', None)),
            'prio_cl_w_risk': state.get('prio_cl_w_risk', state.get('priority_weights', {}).get('prio_cl_w_risk', None)),
            'prio_cl_w_wait': state.get('prio_cl_w_wait', state.get('priority_weights', {}).get('prio_cl_w_wait', None)),
            'q1_violation_rate': q1_violation_rate,
            'q2_violation_rate': q2_violation_rate,
            'q3_violation_rate': q3_violation_rate,
            'q1_worker_drop_rate': q1_drop_rate,
            'q2_worker_drop_rate': q2_drop_rate,
            'q3_worker_drop_rate': q3_drop_rate,
        }
        system_state = {k: v for k, v in system_state.items() if v is not None}

        priority_score, task_vec = self.controller.priority_mgr.calculate_priority(task, system_state)
        try:
            priority_score = float(priority_score)
        except Exception:
            priority_score = 0.5
        if qos == 'Q1':
            if priority_score < 0.95:
                priority_score = 0.95
        elif qos == 'Q2':
            if priority_score < 0.65:
                priority_score = 0.65
        else:
            if priority_score > 0.35:
                priority_score = 0.35
        if priority_score < 0.0:
            priority_score = 0.0
        if priority_score > 1.0:
            priority_score = 1.0

        pred_admit_enabled = bool(state.get('pred_admit_enabled', True))
        if pred_admit_enabled:
            admit_thr = slo_limit_ms

            pred_total_ms = float(pred_dict.get('p90', 0.0) or 0.0) + unc_p90 + queue_delay_ms

            should_shed_early = False
            shed_reason = "latency"

            p90_belief = float(state.get('p90_belief', 100.0))
            current_pred_p90 = float(pred_dict.get('p90', 0.0) or 0.0)
            latency_gradient = current_pred_p90 - p90_belief
            
            if qos == 'Q3' and latency_gradient > 3.0 and current_pred_p90 > (slo_limit_ms * 0.5):
                should_shed_early = True
                shed_reason = "gradient_control"

            if not should_shed_early and qos == 'Q3':
                if (q1_drop_rate > 0.01 or q2_drop_rate > 0.01) or (q1_violation_rate > 0.1 or q2_violation_rate > 0.1):
                    should_shed_early = True
                    shed_reason = "protect_q1q2"

            if not should_shed_early and qos == 'Q2':
                if q1_drop_rate > 0.05 or q1_violation_rate > 0.2:
                    should_shed_early = True
                    shed_reason = "extreme_protect_q1"

            current_price = float(state.get('shadow_price', 0.0))
            if not should_shed_early:
                if current_price > 20.0 and qos == 'Q3':
                    should_shed_early = True
                    shed_reason = "bulkhead_q3"
                elif current_price > 200.0 and qos == 'Q2':
                    should_shed_early = True
                    shed_reason = "bulkhead_q2"

            shed_by_latency = False
            if qos == 'Q3':
                if pred_total_ms > admit_thr * 0.8:
                    shed_by_latency = True

            if qos == 'Q3' and (should_shed_early or shed_by_latency):
                degrade_plan = "store_to_sqs"
                cl_val = None
                try:
                    if isinstance(task_vec, list) and len(task_vec) >= 2:
                        cl_val = float(task_vec[1])
                except Exception:
                    cl_val = None
                if qos == 'Q2' or (cl_val is not None and cl_val >= 0.6):
                    degrade_plan = "store_to_sqs_recovery"
                
                early_decision = {
                    'shouldShed': True,
                    'degrade_plan': degrade_plan,
                    'resource_alloc': last_alloc,
                    'p90_prediction': float(pred_dict.get('p90', 0.0) or 0.0),
                    'uncertainty': uncertainty,
                    'pred_queue_delay_ms': queue_delay_ms,
                    'pred_total_latency_ms': pred_total_ms,
                    'admit_threshold_ms': admit_thr,
                    'queue_backlog': queue_backlog,
                    'queue_backlog_source': backlog_source,
                    'priority_score': priority_score,
                    'qos_class': qos,
                    'shed_reason': shed_reason
                }
                dbg = debug_info or {}
                dbg.update(
                    {
                        'pred_queue_delay_ms': queue_delay_ms,
                        'pred_total_latency_ms': pred_total_ms,
                        'admit_threshold_ms': admit_thr,
                        'queue_backlog': queue_backlog,
                        'queue_backlog_source': backlog_source,
                        'priority_score': priority_score,
                        'servers': servers,
                        'service_ms': eff_service_ms,
                        'queue_delay_model': queue_delay_model,
                        'qos_class': qos,
                        'latency_gradient': latency_gradient,
                        'shadow_price': current_price,
                        'shed_reason': shed_reason
                    }
                )
                return early_decision, dbg
        
        # Shadow Price Update (Local Estimate)
        lam, _ = update_shadow_price(state, metrics, system_state['last_alloc'])
        system_state['shadow_price'] = lam
        
        # Optimization
        result = self.controller.decide(task, wcp_constraints, system_state)
        
        # Update Local Cache Immediate (for next request in same container)
        new_alloc = result['decision']['resource_alloc']
        state['last_alloc'] = new_alloc
        state['shadow_price'] = lam
        _L1_CACHE['params'] = state # Update cache reference
        
        # Async Persistence
        # We push the critical updates (alloc, price, RLS stats) to SQS or Fire-and-Forget
        # For this PoC, we will SKIP synchronous DynamoDB write to maximize speed.
        # We rely on "Soft State" in L1 Cache. 
        # To make it robust, we can write to DB with probability p=0.1 (sampling)
        # or if state changed significantly.
        
        if random.random() < 0.1:
            self._async_save_state(state, version)

        decision_out = result['decision']
        thr_platinum = float(state.get('admit_thr_platinum_ms', slo_limit_ms * 1.2) or (slo_limit_ms * 1.2))
        thr_gold = float(state.get('admit_thr_gold_ms', slo_limit_ms * 1.0) or (slo_limit_ms * 1.0))
        thr_standard = float(state.get('admit_thr_standard_ms', slo_limit_ms * 0.8) or (slo_limit_ms * 0.8))
        if priority_score >= 0.5:
            t = (priority_score - 0.5) / 0.5
            admit_thr = thr_gold + t * (thr_platinum - thr_gold)
        else:
            t = priority_score / 0.5
            admit_thr = thr_standard + t * (thr_gold - thr_standard)
        internal_decision = {
            'shouldShed': bool(decision_out.get('should_shed', False)),
            'degrade_plan': decision_out.get('degrade_plan'),
            'resource_alloc': float(decision_out.get('resource_alloc', 1.0)),
            'p90_prediction': float(pred_dict.get('p90', 0.0)),
            'uncertainty': uncertainty,
            'pred_queue_delay_ms': queue_delay_ms,
            'pred_total_latency_ms': float(pred_dict.get('p90', 0.0) or 0.0) + unc_p90 + queue_delay_ms,
            'admit_threshold_ms': admit_thr,
            'queue_backlog': queue_backlog,
            'queue_backlog_source': backlog_source,
            'priority_score': priority_score,
            'qos_class': qos,
        }
        dbg = debug_info or {}
        dbg.update(
            {
                'pred_queue_delay_ms': queue_delay_ms,
                'queue_backlog': queue_backlog,
                'queue_backlog_source': backlog_source,
                'priority_score': priority_score,
                'servers': servers,
                'service_ms': eff_service_ms,
                'queue_delay_model': queue_delay_model,
            }
        )
        return internal_decision, dbg

    def update_metrics(self, real_metrics):
        """
        Called after execution to update state with realized performance.
        real_metrics: {'latency': ms, 'cpu': %, ...}
        """
        global _L1_CACHE
        
        # 1. Update L1 Cache History (Simple EMA of p90)
        # We try to update the 'p90_belief' in the cached params
        if _L1_CACHE['params']:
            curr_p90 = float(_L1_CACHE['params'].get('p90_belief', 100.0))
            new_val = float(real_metrics.get('latency', 100.0))
            
            # EMA Update
            alpha = 0.2
            updated_p90 = (1 - alpha) * curr_p90 + alpha * new_val
            
            _L1_CACHE['params']['p90_belief'] = updated_p90
            _L1_CACHE['last_sync'] = time.time() # Refresh timestamp
            
            # 2. Async Persist (Sampling 20%)
            if random.random() < 0.2:
                 self._async_update_feedback(updated_p90)

    def _async_update_feedback(self, p90_val):
        try:
            if UPDATE_QUEUE_URL:
                sqs.send_message(
                    QueueUrl=UPDATE_QUEUE_URL,
                    MessageBody=json.dumps(
                        {
                            'type': 'update_feedback',
                            'state_id': self.state_id,
                            'p90_belief': float(p90_val),
                            'ts': time.time(),
                        }
                    ),
                )
        except Exception as e:
            print(f"Async Feedback Error: {e}")

    def _async_save_state(self, params, version):
        """
        Write to DynamoDB. In a real middleware, this would be put on a queue.
        Here we do a quick synchronous write but only 10% of time.
        """
        try:
            if UPDATE_QUEUE_URL:
                sqs.send_message(
                    QueueUrl=UPDATE_QUEUE_URL,
                    MessageBody=json.dumps(
                        {
                            'type': 'save_state',
                            'state_id': self.state_id,
                            'last_alloc': float(params.get('last_alloc', 1.0)),
                            'shadow_price': float(params.get('shadow_price', 0.0)),
                            'version': int(version + 1),
                            'ts': time.time(),
                        }
                    ),
                )
        except Exception as e:
            print(f"Async Write Error: {e}")

    def _hydrate_controller(self, state):
        weights = state.get('optimizer_weights', {})
        self.controller.optimizer.w1 = weights.get('w1', 1.0)
        self.controller.optimizer.w2 = weights.get('w2', 0.5)
        self.controller.optimizer.w3 = weights.get('w3', 5.0)
        pw = state.get('priority_weights', {})
        try:
            self.controller.priority_mgr.lambda1 = float(pw.get('lambda1', self.controller.priority_mgr.lambda1))
        except Exception:
            pass
        try:
            self.controller.priority_mgr.alpha = float(pw.get('alpha', self.controller.priority_mgr.alpha))
        except Exception:
            pass
        try:
            self.controller.priority_mgr.beta = float(pw.get('beta', self.controller.priority_mgr.beta))
        except Exception:
            pass
        phi = pw.get('phi', None)
        if isinstance(phi, list) and len(phi) == 2:
            try:
                self.controller.priority_mgr.phi = [float(phi[0]), float(phi[1])]
            except Exception:
                pass
        
    def _parse_dynamo_item(self, item):
        # ... (Simplified parser based on controller logic) ...
        # Copied essential parts
        params_map = item.get('params', {}).get('M', {})
        
        def get_float(m, k, default):
            try: return float(m.get(k, {}).get('N', default))
            except: return float(default)
        def get_phi(m, k):
            try:
                l = m.get(k, {}).get('L', [])
                if not l:
                    return [0.6, 0.4]
                return [float(x.get('N', '0.5')) for x in l]
            except Exception:
                return [0.6, 0.4]
            
        rls_states_json = item.get('rls_states', {}).get('S', '{}')
        try: rls_states = json.loads(rls_states_json)
        except: rls_states = {}

        pr_map = item.get('priority_weights', {}).get('M', {})
        priority_weights = {
            'lambda1': get_float(pr_map, 'lambda1', 0.6),
            'alpha': get_float(pr_map, 'alpha', 0.7),
            'beta': get_float(pr_map, 'beta', 0.3),
            'phi': get_phi(pr_map, 'phi'),
            'prio_cl_w_latency': get_float(pr_map, 'prio_cl_w_latency', 0.45),
            'prio_cl_w_risk': get_float(pr_map, 'prio_cl_w_risk', 0.35),
            'prio_cl_w_wait': get_float(pr_map, 'prio_cl_w_wait', 0.20),
        }

        return {
            'bP': get_float(params_map, 'bP', 2000.0),
            'rls_states': rls_states,
            'wcp_alpha': get_float(params_map, 'wcp_alpha', 0.1),
            'shadow_price': get_float(item, 'shadow_price', 0.0),
            'last_alloc': get_float(params_map, 'last_alloc', 1.0),
            'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0}, # Defaults
            'priority_weights': priority_weights,
            'prio_cl_w_latency': priority_weights.get('prio_cl_w_latency', 0.45),
            'prio_cl_w_risk': priority_weights.get('prio_cl_w_risk', 0.35),
            'prio_cl_w_wait': priority_weights.get('prio_cl_w_wait', 0.20),
            'gamma': get_float(params_map, 'gamma', 0.1),
            'u_eta': get_float(params_map, 'u_eta', 0.05),
            'u_max_delta': get_float(params_map, 'u_max_delta', 0.15),
            'slo_limit': get_float(params_map, 'slo_limit', 1000.0),
            'p90_belief': get_float(params_map, 'p90_belief', 100.0),
            'pred_admit_enabled': True,
            'admit_thr_platinum_ms': get_float(params_map, 'admit_thr_platinum_ms', 0.0),
            'admit_thr_gold_ms': get_float(params_map, 'admit_thr_gold_ms', 0.0),
            'admit_thr_standard_ms': get_float(params_map, 'admit_thr_standard_ms', 0.0),
            'queue_delay_model': 'backlog_linear',
            'queue_backlog_ttl_s': get_float(params_map, 'queue_backlog_ttl_s', 2.0),
            'buffer_servers_default': get_float(params_map, 'buffer_servers_default', 1.0),
            'avg_service_ms': get_float(params_map, 'avg_service_ms', 0.0),
        }

    def _get_default_params(self):
        return {
            'bP': 2000.0,
            'rls_states': {},
            'shadow_price': 0.0,
            'last_alloc': 1.0,
            'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0},
            'priority_weights': {
                'lambda1': 0.6,
                'alpha': 0.7,
                'beta': 0.3,
                'phi': [0.6, 0.4],
                'prio_cl_w_latency': 0.45,
                'prio_cl_w_risk': 0.35,
                'prio_cl_w_wait': 0.20,
            },
            'prio_cl_w_latency': 0.45,
            'prio_cl_w_risk': 0.35,
            'prio_cl_w_wait': 0.20,
            'gamma': 0.1,
            'u_eta': 0.05,
            'u_max_delta': 0.15,
            'slo_limit': 1000.0,
            'p90_belief': 100.0,
            'pred_admit_enabled': True,
            'admit_thr_platinum_ms': 0.0,
            'admit_thr_gold_ms': 0.0,
            'admit_thr_standard_ms': 0.0,
            'queue_delay_model': 'backlog_linear',
            'queue_backlog_ttl_s': 2.0,
            'buffer_servers_default': 1.0,
            'avg_service_ms': 0.0,
        }
