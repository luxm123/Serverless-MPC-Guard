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
CACHE_TTL = 0.1 # 缩短到 0.1s，强制更频繁地从 DynamoDB 读取全局状态，减少容器间的冲突

# AWS Clients
dynamodb = boto3.client(
    'dynamodb',
    config=Config(
        connect_timeout=1,
        read_timeout=1,
        retries={'max_attempts': 2, 'mode': 'standard'},
    ),
)
sqs = boto3.client(
    'sqs',
    config=Config(
        connect_timeout=0.5,
        read_timeout=0.5,
        retries={'max_attempts': 1}
    )
)

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
        task_type = task.get('task_type', event.get('task_type', 'image_processing'))
        strategy = event.get('strategy', 'mpc_integrated')
        
        # v53: Use per-task state + Asymmetric EMA
        original_state_id = self.state_id
        if self.state_id == 'global_params':
            self.state_id = f"mpc_state_{task_type}"
            
        # Metadata for debugging code version
        current_ver = '20260320_v53_Academic_Final'
        debug_info = {'version': current_ver, 'state_id': self.state_id}

        # Step 1: Get latest state from DynamoDB
        state, version = self._load_state()
        
        # 终极重置逻辑
        if not state or state.get('version') != current_ver or event.get('reset_state'):
            state = self._get_default_params()
            state['version'] = current_ver
            state['last_alloc'] = 1.0 
            state['shadow_price'] = 0.0
            state['queue_backlog_belief'] = 0.0
            state['prev_rps'] = 0.0
            version = None
            debug_info['state_source'] = 'forced_reset'
            print(f"[Middleware-v53] NUCLEAR RESET for {self.state_id}: Academic Final mode.")
        else:
            debug_info['state_source'] = 'dynamodb'
            
        last_alloc = float(state.get('last_alloc', 1.0))
        debug_info['loaded_alloc'] = last_alloc

        # --- Experiment 3: Pre-warming Trigger Logic ---
        current_rps = float(metrics.get('rps', 0.0))
        prev_rps = float(state.get('prev_rps', 0.0))
        trigger_prewarm = False
        
        # 创新点逻辑：RPS 增长 >= 20% 触发预热 (仅在 full 或 prewarm 模式下开启)
        if strategy in ['mpc_integrated', 'passive_prewarm'] and prev_rps > 1.0:
            if current_rps >= 1.2 * prev_rps:
                trigger_prewarm = True
                debug_info['prewarm_triggered'] = True
        
        state['prev_rps'] = current_rps # 更新 RPS 记录
        
        if not metrics or 'p90' not in metrics:
             metrics = metrics.copy()
             metrics['p90'] = float(state.get('p90_belief', 100.0))
             metrics['cpu_usage'] = 0.8 if metrics['p90'] > 500 else 0.2
        
        # Hydrate Controller
        self._hydrate_controller(state)
        
        # --- 2. Resolve Dynamics Metrics (concurrency, cpu, backlog, service_time_ms) ---
        client_backlog = float(metrics.get('queue_backlog', -1.0))
        queue_backlog = None
        backlog_source = 'unknown'

        if client_backlog >= 0.0:
            queue_backlog = client_backlog
            backlog_source = 'client_injected'
            _L1_CACHE['last_backlog'] = queue_backlog
            _L1_CACHE['last_backlog_sync'] = time.time()
        else:
            queue_backlog = metrics.get('queue', None)
            if queue_backlog is None:
                queue_backlog = state.get('queue_backlog_belief', 0.0)
        queue_backlog = float(queue_backlog or 0.0)

        servers = metrics.get('concurrency', metrics.get('servers', None))
        if servers is None or float(servers) <= 1.0:
            if queue_backlog > 10.0:
                servers = queue_backlog # Heuristic
        servers = max(1.0, float(servers or state.get('buffer_servers', 1.0)))

        last_alloc = float(state.get('last_alloc', 1.0) or 1.0)
        
        # Service Time Estimation
        p90_val = float(metrics.get('p90', 100.0))
        if queue_backlog > 1.0:
             service_time_ms = p90_val / (1.0 + queue_backlog / servers)
        else:
             service_time_ms = p90_val
        service_time_ms = min(max(10.0, service_time_ms), 500.0) # Safety clamp

        # WCP Update (Prediction)
        pred_dict, uncertainty, wcp_dbg = wcp_update(
            state, 
            p90_val, 
            servers, 
            last_alloc, 
            queue_backlog, 
            service_time_ms, 
            task_type=task_type,
            alpha=0.2
        )
        debug_info.update(wcp_dbg or {})
        
        # MPC Constraints
        wcp_constraints = {'pred': pred_dict, 'uncertainty': uncertainty}

        queue_backlog = float(queue_backlog or 0.0)
        metrics['queue_backlog'] = queue_backlog

        unc_p90 = 0.0
        if isinstance(uncertainty, dict):
            unc_p90 = float(uncertainty.get('p90', 0.0) or 0.0)
        else:
            try:
                unc_p90 = float(uncertainty or 0.0)
            except Exception:
                unc_p90 = 0.0

        eff_alloc = max(0.1, last_alloc)
        base_service_ms = service_time_ms
        eff_service_ms = base_service_ms / (eff_alloc + 0.01)

        queue_delay_model = str(state.get('queue_delay_model', 'backlog_linear') or 'backlog_linear').strip().lower()
        queue_delay_ms = (queue_backlog * eff_service_ms) / servers
        
        analytic_latency = queue_delay_ms + eff_service_ms
        
        if not isinstance(pred_dict, dict):
            pred_dict = {'p90': float(pred_dict)} 

        wcp_latency = float(pred_dict.get('p90', 100.0) or 100.0)

        if wcp_latency > 2.0 * analytic_latency + 100.0:
            if random.random() < 0.05:
                print(f"[Middleware] Stale WCP Model Override: WCP={wcp_latency:.1f}ms -> Analytic={analytic_latency:.1f}ms (Backlog={queue_backlog}, Servers={servers})")
            pred_dict['p90'] = analytic_latency
            uncertainty = 50.0
            
        slo_limit_ms = 180.0
        slo_viol_rate = float(metrics.get('slo_violation_rate', 0.0) or 0.0)

        system_state = {
            'shadow_price': state.get('shadow_price', 0.0),
            'last_alloc': last_alloc,
            'p90_belief': float(state.get('p90_belief', 100.0)),
            'grad_track': float(state.get('grad_track', 0.0)),
            'strategy': event.get('strategy', 'mpc_integrated'), # Pass strategy to optimizer
            'gamma': state.get('gamma', 0.1),
            'u_eta': state.get('u_eta', 0.05),
            'u_max_delta': state.get('u_max_delta', 0.15),
            'slo_limit': slo_limit_ms,
            'pred_queue_delay_ms': queue_delay_ms,
            'queue_backlog': queue_backlog,
            'metrics': {
                'slo_violation_rate': slo_viol_rate
            }
        }
        system_state = {k: v for k, v in system_state.items() if v is not None}

        # ALL REQUESTS ARE EXECUTED (No Early Shedding)
        should_shed_early = False
        shed_reason = None
        admit_thr = slo_limit_ms
        pred_total_ms = float(pred_dict.get('p90', 0.0) or 0.0) + unc_p90 + queue_delay_ms

        # Shadow Price Update (Local Estimate)
        lam, _ = update_shadow_price(state, metrics, system_state['last_alloc'])
        system_state['shadow_price'] = lam
        
        # Optimization
        system_state['last_alloc'] = last_alloc
        system_state['p90_belief'] = float(state.get('p90_belief', 100.0))
        
        # Step 3: Call Controller
        result = self.controller.decide(task, wcp_constraints, system_state)
        ctrl_dbg = result.get('meta', {})
        debug_info.update(ctrl_dbg)
        
        # 提取优化器耗时
        opt_overhead = system_state.get('opt_debug', {}).get('overhead_ms', 0.0)
        debug_info['scheduling_overhead_ms'] = opt_overhead
        debug_info['latency_gradient'] = system_state.get('opt_debug', {}).get('grad', 0.0)
        
        decision_out = result['decision']
        new_alloc = float(decision_out.get('resource_alloc', last_alloc))
        
        # 强制将 new_alloc 和 grad_track 写回状态
        state['last_alloc'] = new_alloc
        state['grad_track'] = system_state.get('grad_track', 0.0)
        state['shadow_price'] = lam
        _L1_CACHE['params'] = state # Update cache reference
        
        debug_info['new_alloc'] = new_alloc
        debug_info['prev_alloc'] = last_alloc
        
        self._async_save_state(state, version or 0)

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
            'trigger_prewarm': trigger_prewarm, # Experiment 3
        }
        dbg = debug_info or {}
        dbg.update(
            {
                'pred_queue_delay_ms': queue_delay_ms,
                'pred_total_latency_ms': float(pred_dict.get('p90', 0.0) or 0.0) + unc_p90 + queue_delay_ms,
                'admit_threshold_ms': admit_thr,
                'queue_backlog': queue_backlog,
                'queue_backlog_source': backlog_source,
                'servers': servers,
                'service_ms': eff_service_ms,
                'queue_delay_model': queue_delay_model,
                'latency_gradient': debug_info.get('latency_gradient', 0.0),
                'shadow_price': lam,
                'shed_reason': shed_reason
            }
        )
        
        # Restore original state_id for next request in this container if it's reused
        self.state_id = original_state_id
        
        return internal_decision, dbg

    def update_metrics(self, real_metrics):
        """
        Called after execution to update state with realized performance.
        v53: Asymmetric EMA (Fast Rise, Slow Fall) to prevent QoS lag.
        """
        global _L1_CACHE
        
        if _L1_CACHE['params']:
            curr_p90 = float(_L1_CACHE['params'].get('p90_belief', 100.0))
            new_val = float(real_metrics.get('latency', 100.0))
            
            # 非对称 EMA：上涨时反应快 (0.5)，下跌时反应慢 (0.05) 以保命
            if new_val > curr_p90:
                alpha = 0.5 
            else:
                alpha = 0.05 
                
            updated_p90 = (1 - alpha) * curr_p90 + alpha * new_val
            _L1_CACHE['params']['p90_belief'] = updated_p90
            _L1_CACHE['last_sync'] = time.time() # Refresh timestamp
            self._async_save_state(_L1_CACHE['params'], _L1_CACHE['version'])

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
        Directly update DynamoDB to persist Fidelity/ShadowPrice.
        CRITICAL FIX for Distributed Amnesia (Flapping).
        """
        try:
            version_str = params.get('version', 'unknown')
            dynamodb.update_item(
                TableName=TABLE_NAME,
                Key={'id': {'S': self.state_id}},
                UpdateExpression=(
                    "SET #p = if_not_exists(#p, :empty_map), "
                    "#p.last_alloc = :la, #p.shadow_price = :sp, "
                    "shadow_price = :sp_top, #p.p90_belief = :p90, "
                    "version_str = :vs, "
                    "version = :v, last_updated = :t"
                ),
                ExpressionAttributeNames={'#p': 'params'},
                ExpressionAttributeValues={
                    ':la': {'N': str(float(params.get('last_alloc', 1.0)))},
                    ':sp': {'N': str(float(params.get('shadow_price', 0.0)))},
                    ':sp_top': {'N': str(float(params.get('shadow_price', 0.0)))},
                    ':p90': {'N': str(float(params.get('p90_belief', 100.0)))},
                    ':vs': {'S': str(version_str)},
                    ':v': {'N': str(int((version or 0) + 1))},
                    ':t': {'N': str(time.time())},
                    ':empty_map': {'M': {}}
                }
            )
            if random.random() < 0.05:
                print(f"[Middleware] State Saved: ID={self.state_id}, Alloc={params.get('last_alloc', 1.0)}, Ver={version_str}")
        except Exception as e:
            print(f"[Middleware] State Save Error: {e}")

    def _hydrate_controller(self, state):
        weights = state.get('optimizer_weights', {})
        self.controller.optimizer.w1 = weights.get('w1', 1.0)
        self.controller.optimizer.w2 = weights.get('w2', 0.5)
        self.controller.optimizer.w3 = weights.get('w3', 5.0)
        
    def _parse_dynamo_item(self, item):
        params_map = item.get('params', {}).get('M', {})
        def get_float(m, k, default):
            try: return float(m.get(k, {}).get('N', default))
            except: return float(default)
        rls_states_json = item.get('rls_states', {}).get('S', '{}')
        try: rls_states = json.loads(rls_states_json)
        except: rls_states = {}
        version_str = item.get('version_str', {}).get('S', 'unknown')
        return {
            'bP': get_float(params_map, 'bP', 2000.0),
            'rls_states': rls_states,
            'wcp_alpha': get_float(params_map, 'wcp_alpha', 0.1),
            'shadow_price': get_float(item, 'shadow_price', 0.0),
            'last_alloc': get_float(params_map, 'last_alloc', 1.0),
            'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0}, 
            'gamma': get_float(params_map, 'gamma', 0.1),
            'u_eta': get_float(params_map, 'u_eta', 0.05),
            'u_max_delta': get_float(params_map, 'u_max_delta', 0.15),
            'slo_limit': get_float(params_map, 'slo_limit', 1000.0),
            'p90_belief': get_float(params_map, 'p90_belief', 100.0),
            'pred_admit_enabled': True,
            'queue_delay_model': 'backlog_linear',
            'queue_backlog_ttl_s': get_float(params_map, 'queue_backlog_ttl_s', 2.0),
            'buffer_servers_default': get_float(params_map, 'buffer_servers_default', 1.0),
            'avg_service_ms': get_float(params_map, 'avg_service_ms', 0.0),
            'version': version_str
        }

    def _get_default_params(self):
        return {
            'bP': 2000.0,
            'rls_states': {},
            'shadow_price': 0.0,
            'last_alloc': 1.0, 
            'optimizer_weights': {'w1': 1.0, 'w2': 5.0, 'w3': 1.0}, 
            'gamma': 0.05, 
            'u_eta': 0.15, 
            'u_max_delta': 0.5, 
            'slo_limit': 180.0, 
            'p90_belief': 100.0,
            'pred_admit_enabled': True,
            'queue_delay_model': 'backlog_linear',
            'queue_backlog_ttl_s': 2.0,
            'buffer_servers_default': 1.0,
            'avg_service_ms': 0.0,
            'min_alloc_floor': 0.01, 
        }
