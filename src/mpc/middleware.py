import time
import json
import random
import math
import os
import boto3
from botocore.exceptions import ClientError

# --- Globals ---
TABLE_NAME = 'MPC_State'
# Use a session to handle retries and configurations
session = boto3.Session()
dynamodb_client = session.client('dynamodb')

# v59: L1 Cache for state to avoid DynamoDB latency and inconsistency
_L1_CACHE = {
    'params': None,
    'version': None,
    'last_sync': 0,
    'ttl_s': 2.0,
    'last_write': 0.0,
    'state_id': None
}

_LOCAL_STATE = {}

class MPCMiddleware:
    def __init__(self, state_id='global_params'):
        from src.mpc.controller import MPCController
        self.state_id = state_id
        self.controller = MPCController()
        self._state_mode = str(os.environ.get("MPC_STATE_MODE", "dynamodb") or "dynamodb").strip().lower()

    def _json_safe(self, obj, depth=0):
        if depth >= 6:
            return None
        if obj is None:
            return None
        if isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, (int, float)):
            x = float(obj)
            if not math.isfinite(x):
                return 0.0
            return x
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    ks = str(k)
                except Exception:
                    continue
                out[ks] = self._json_safe(v, depth=depth + 1)
            return out
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v, depth=depth + 1) for v in obj]
        try:
            x = float(obj)
            if math.isfinite(x):
                return x
            return 0.0
        except Exception:
            try:
                return str(obj)
            except Exception:
                return None

    def _finite_float(self, val, default):
        try:
            x = float(val)
        except Exception:
            return float(default)
        if not math.isfinite(x):
            return float(default)
        return x

    def _clamp_ms(self, val, lo, hi, default):
        x = self._finite_float(val, default)
        if x < lo:
            return float(lo)
        if x > hi:
            return float(hi)
        return float(x)

    def _sanitize_params(self, params):
        if not isinstance(params, dict):
            return self._json_safe(params)
        out = dict(params)
        out['last_alloc'] = self._finite_float(out.get('last_alloc', 1.0), 1.0)
        out['p90_belief'] = self._clamp_ms(out.get('p90_belief', 110.0), 1.0, 500.0, 110.0)
        out['uncertainty'] = self._clamp_ms(out.get('uncertainty', 30.0), 0.0, 60.0, 30.0)
        out['last_y'] = self._clamp_ms(out.get('last_y', out['p90_belief']), 1.0, 500.0, out['p90_belief'])
        out['e2e_overhead_ms'] = self._clamp_ms(out.get('e2e_overhead_ms', 50.0), 0.0, 120.0, 50.0)
        try:
            out['safe_streak'] = int(out.get('safe_streak', 0))
        except Exception:
            out['safe_streak'] = 0
        if out['safe_streak'] < 0:
            out['safe_streak'] = 0
        for k in list(out.keys()):
            out[k] = self._json_safe(out.get(k), depth=0)
        return out

    def _load_state(self):
        global _L1_CACHE
        now = time.time()
        mode = str(getattr(self, "_state_mode", "dynamodb") or "dynamodb").strip().lower()

        if mode in ["local", "memory", "mem", "inmem"]:
            item = _LOCAL_STATE.get(self.state_id)
            if not item:
                return None, None
            try:
                state = self._sanitize_params(dict(item.get("state") or {}))
            except Exception:
                state = None
            ver = str(item.get("version") or "0")
            return state, ver

        try:
            ttl_s = float(os.environ.get("MPC_STATE_CACHE_TTL_S", _L1_CACHE.get("ttl_s", 2.0)) or 2.0)
        except Exception:
            ttl_s = float(_L1_CACHE.get("ttl_s", 2.0) or 2.0)
        if (not math.isfinite(ttl_s)) or ttl_s <= 0.0:
            ttl_s = 2.0
        ttl_s = float(max(0.05, min(30.0, ttl_s)))
        
        if _L1_CACHE['params'] and (now - _L1_CACHE['last_sync']) < ttl_s and _L1_CACHE['state_id'] == self.state_id:
            return _L1_CACHE['params'], _L1_CACHE['version']

        try:
            consistent = str(os.environ.get("MPC_STATE_CONSISTENT_READ", "0")).strip() == "1"
            response = dynamodb_client.get_item(
                TableName=TABLE_NAME, 
                Key={'id': {'S': self.state_id}},
                ConsistentRead=bool(consistent)
            )
            item = response.get('Item')
            if item:
                # v62: New JSON Blob format takes priority
                if 'state_blob' in item:
                    state = self._sanitize_params(json.loads(item['state_blob']['S']))
                else:
                    # Fallback to legacy parsing for migration
                    state = self._sanitize_params(self._parse_dynamo_item(item))
                
                lock_version = item.get('lock_version', {}).get('N', '0')
                _L1_CACHE['params'] = state
                _L1_CACHE['version'] = lock_version
                _L1_CACHE['last_sync'] = now
                _L1_CACHE['state_id'] = self.state_id
                return state, lock_version
        except Exception as e:
            print(f"Error loading state: {e}")
        
        return None, None

    def decide(self, event):
        start_t = time.time()
        metrics = event.get('metrics', {})
        task = event.get('task', {})
        task_type = task.get('task_type', event.get('task_type', 'image_processing'))
        strategy = event.get('strategy', 'mpc_integrated')

        self._state_mode = str(metrics.get("state_mode", os.environ.get("MPC_STATE_MODE", "dynamodb")) or "dynamodb").strip().lower()
        
        original_state_id = self.state_id
        if self.state_id == 'global_params':
            self.state_id = f"mpc_state_{task_type}"
            
        current_logic_ver = 'v66.0'
        state, lock_ver = self._load_state()
        write_status = "NotAttempted"

        state = self._sanitize_params(state) if state else None
        should_reset = (not state) or (state.get('code_version') != current_logic_ver) or bool(event.get('reset_state'))
        if state:
            if not math.isfinite(float(state.get('last_alloc', 1.0))):
                should_reset = True
            if float(state.get('p90_belief', 110.0)) > 1000.0:
                should_reset = True
            if float(state.get('uncertainty', 30.0)) > 200.0:
                should_reset = True
            if float(state.get('p90_belief', 110.0)) >= 499.0 and float(state.get('last_y', 0.0)) < 250.0:
                should_reset = True

        if should_reset:
            state = self._get_default_params()
            state['code_version'] = current_logic_ver
            lock_ver = '0'
            print(f"[CRITICAL-DEBUG] {current_logic_ver} RESET for {self.state_id}")
            write_status = self._sync_save_state(state, lock_ver, force=True)
            lock_ver = '1'
        
        last_alloc = self._finite_float(state.get('last_alloc', 1.0), 1.0)
        current_rps = self._finite_float(metrics.get('rps', 0.0), 0.0)
        prev_rps = self._finite_float(state.get('prev_rps', 0.0), 0.0)
        state['prev_rps'] = current_rps
        concurrency = self._finite_float(metrics.get('concurrency', metrics.get('backlog', 0.0)), 0.0)
        backlog = self._finite_float(metrics.get('backlog', concurrency), concurrency)
        budget = self._finite_float(metrics.get('budget', state.get('budget', 0.0)), 0.0)
        if not math.isfinite(float(budget)) or float(budget) < 0.0:
            budget = 0.0
        state['budget'] = float(budget)
        slo_limit = self._finite_float(metrics.get('slo_limit', state.get('slo_limit', 180.0)), 180.0)
        if slo_limit <= 0.0 or not math.isfinite(float(slo_limit)):
            slo_limit = 180.0
        slo_limit = float(max(1.0, min(10000.0, slo_limit)))
        state['slo_limit'] = slo_limit
        min_alloc = self._finite_float(metrics.get('min_alloc', state.get('min_alloc', 0.0)), 0.0)
        if not math.isfinite(float(min_alloc)):
            min_alloc = 0.0
        min_alloc = float(max(0.0, min(1.0, min_alloc)))
        state['min_alloc'] = min_alloc

        max_alloc = self._finite_float(metrics.get('max_alloc', state.get('max_alloc', 1.0)), 1.0)
        if not math.isfinite(float(max_alloc)):
            max_alloc = 1.0
        max_alloc = float(max(0.4, min(4.0, max_alloc)))
        state['max_alloc'] = max_alloc

        unc_scale = self._finite_float(metrics.get('unc_scale', state.get('unc_scale', 1.0)), 1.0)
        if not math.isfinite(float(unc_scale)) or float(unc_scale) <= 0.0:
            unc_scale = 1.0
        unc_scale = float(max(0.5, min(3.0, unc_scale)))
        state['unc_scale'] = unc_scale

        tight_slo_ms = self._finite_float(metrics.get('tight_slo_ms', state.get('tight_slo_ms', 80.0)), 80.0)
        if not math.isfinite(float(tight_slo_ms)) or float(tight_slo_ms) <= 0.0:
            tight_slo_ms = 80.0
        tight_slo_ms = float(max(20.0, min(200.0, tight_slo_ms)))
        state['tight_slo_ms'] = tight_slo_ms

        # v64.0: Ensure belief is REAL
        current_p90 = self._clamp_ms(state.get('p90_belief', 110.0), 1.0, 500.0, 110.0)
        if current_p90 < 1.0:
            current_p90 = 110.0

        overhead_ms = self._clamp_ms(metrics.get('e2e_overhead_ms', state.get('e2e_overhead_ms', 50.0)), 0.0, 120.0, 50.0)
        state['e2e_overhead_ms'] = overhead_ms
        state_last_y = self._clamp_ms(state.get('last_y', current_p90), 1.0, 500.0, current_p90)

        self._hydrate_controller(state)
        system_state = {
            'last_alloc': last_alloc,
            'p90_belief': current_p90,
            'uncertainty': float(state.get('uncertainty', 0.0)),
            'e2e_overhead_ms': overhead_ms,
            'last_y': state_last_y,
            'strategy': strategy,
            'current_rps': current_rps,
            'prev_rps': prev_rps,
            'concurrency': concurrency,
            'backlog': backlog,
            'budget': float(budget),
            'slo_limit': slo_limit,
            'min_alloc': min_alloc,
            'max_alloc': max_alloc,
            'unc_scale': unc_scale,
            'tight_slo_ms': tight_slo_ms,
        }

        result = self.controller.decide(task, {}, system_state)
        new_alloc = self._finite_float(result.get('decision', {}).get('resource_alloc', last_alloc), last_alloc)
        
        # v64.0: EXPLICITLY update state with the belief used for decision
        state['last_alloc'] = new_alloc
        state['p90_belief'] = current_p90 
        if 'safe_streak' in system_state:
            state['safe_streak'] = int(system_state.get('safe_streak', 0))
        
        write_status = self._sync_save_state(state, lock_ver)
        self.state_id = original_state_id 
        
        # v64.0: Pack everything into the result for visibility
        debug_info = {
            'version': f"{current_logic_ver}_L{lock_ver}_W{write_status}",
            'code_version': current_logic_ver,
            'state_id': self.state_id,
            'p90_belief': current_p90, # THIS IS THE ONE THE SCRIPT READS
            'uncertainty': float(state.get('uncertainty', 0.0)),
            'unc_scale': unc_scale,
            'tight_slo_ms': tight_slo_ms,
            'prev_alloc': last_alloc,
            'new_alloc': new_alloc
        }
        
        # Ensure result['decision'] also has these for legacy scripts
        result['decision']['p90_belief'] = current_p90
        result['decision']['uncertainty'] = float(state.get('uncertainty', 0.0))
        result['decision']['version'] = debug_info['version']
        result['decision']['prev_alloc'] = last_alloc
        result['decision']['new_alloc'] = new_alloc

        return result['decision'], {**debug_info, **result.get('meta', {})}

    def _sync_save_state(self, params, version, force=False):
        global _L1_CACHE
        try:
            now_t = time.time()
            mode = str(getattr(self, "_state_mode", "dynamodb") or "dynamodb").strip().lower()
            if mode in ["local", "memory", "mem", "inmem"]:
                safe_params = self._sanitize_params(params)
                try:
                    expected_version = int(version)
                except Exception:
                    expected_version = 0
                new_version = expected_version + 1
                _LOCAL_STATE[self.state_id] = {"state": safe_params.copy(), "version": str(new_version), "last_updated": float(time.time())}
                _L1_CACHE['params'] = safe_params.copy()
                _L1_CACHE['version'] = str(new_version)
                _L1_CACHE['last_sync'] = now_t
                _L1_CACHE['last_write'] = now_t
                _L1_CACHE['state_id'] = self.state_id
                return "OK"
            if not force:
                try:
                    min_write_interval_s = float(os.environ.get("MPC_STATE_WRITE_INTERVAL_S", 2.0) or 2.0)
                except Exception:
                    min_write_interval_s = 2.0
                if (not math.isfinite(min_write_interval_s)) or min_write_interval_s <= 0.0:
                    min_write_interval_s = 2.0
                min_write_interval_s = float(max(0.1, min(30.0, min_write_interval_s)))
                last_write = float(_L1_CACHE.get('last_write', 0.0) or 0.0)
                if (now_t - last_write) < min_write_interval_s:
                    safe_params = self._sanitize_params(params)
                    _L1_CACHE['params'] = safe_params.copy()
                    _L1_CACHE['last_sync'] = now_t
                    _L1_CACHE['state_id'] = self.state_id
                    return "Skipped"

            expected_version = int(version)
            new_version = expected_version + 1
            
            # v62: Use a JSON blob to avoid all attribute naming/type issues
            safe_params = self._sanitize_params(params)
            state_json = json.dumps(safe_params, allow_nan=False)
            
            update_params = {
                'TableName': TABLE_NAME,
                'Key': {'id': {'S': self.state_id}},
                'UpdateExpression': "SET state_blob = :b, lock_version = :new_lv, last_updated = :t",
                'ExpressionAttributeValues': {
                    ':b': {'S': state_json},
                    ':new_lv': {'N': str(new_version)},
                    ':t': {'N': str(time.time())}
                }
            }
            
            if not force:
                update_params['ConditionExpression'] = "lock_version = :expected_lv OR attribute_not_exists(lock_version)"
                update_params['ExpressionAttributeValues'][':expected_lv'] = {'N': str(expected_version)}

            dynamodb_client.update_item(**update_params)
            
            _L1_CACHE['version'] = str(new_version)
            _L1_CACHE['params'] = safe_params.copy()
            _L1_CACHE['last_sync'] = time.time()
            _L1_CACHE['last_write'] = time.time()
            _L1_CACHE['state_id'] = self.state_id
            return "OK"
            
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == 'ConditionalCheckFailedException':
                _L1_CACHE['last_sync'] = 0 
                return "LockConflict"
            return f"Err_{code}"
        except Exception as e:
            return f"Err_{type(e).__name__}"

    def update_metrics(self, real_metrics):
        """v66.0: 使用 WCP (Weighted Conformal Prediction) 代替 EMA，实现高精度低开销预测"""
        global _L1_CACHE
        task_type = real_metrics.get('task_type', 'image_processing')
        target_id = f"mpc_state_{task_type}"
        
        try:
            mode = str(getattr(self, "_state_mode", os.environ.get("MPC_STATE_MODE", "dynamodb")) or "dynamodb").strip().lower()
            now = time.time()
            state = None
            ver = None
            if _L1_CACHE.get('params') and _L1_CACHE.get('state_id') == target_id and (now - float(_L1_CACHE.get('last_sync', 0.0) or 0.0)) < float(_L1_CACHE.get('ttl_s', 0.5) or 0.5):
                state = self._sanitize_params(dict(_L1_CACHE.get('params') or {}))
                ver = str(_L1_CACHE.get('version') or "0")
            elif mode in ["local", "memory", "mem", "inmem"]:
                item = _LOCAL_STATE.get(target_id)
                if not item:
                    return
                state = self._sanitize_params(dict(item.get("state") or {}))
                ver = str(item.get("version") or "0")
                _L1_CACHE['params'] = state
                _L1_CACHE['version'] = ver
                _L1_CACHE['last_sync'] = now
                _L1_CACHE['state_id'] = target_id
            else:
                response = dynamodb_client.get_item(
                    TableName=TABLE_NAME,
                    Key={'id': {'S': target_id}},
                    ConsistentRead=False
                )
                item = response.get('Item')
                if not item or 'state_blob' not in item:
                    return
                state = self._sanitize_params(json.loads(item['state_blob']['S']))
                ver = item.get('lock_version', {}).get('N', '0')
                _L1_CACHE['params'] = state
                _L1_CACHE['version'] = ver
                _L1_CACHE['last_sync'] = now
                _L1_CACHE['state_id'] = target_id
            
            # 2. 调用 WCP 核心更新算法
            from src.wcp.wcp_update import wcp_update
            
            # 提取特征
            p90_lat = self._finite_float(real_metrics.get('latency', 100.0), 100.0)
            concurrency = self._finite_float(real_metrics.get('concurrency', 1.0), 1.0)
            cpu = self._finite_float(real_metrics.get('cpu_limit', 1.0), 1.0)
            backlog = self._finite_float(real_metrics.get('backlog', 0.0), 0.0)
            service_time = self._finite_float(real_metrics.get('service_time', 100.0), 100.0)
            
            # WCP 更新：返回预测值、不确定性和调试信息
            pred, uncertainty, debug = wcp_update(
                state, p90_lat, concurrency, cpu, backlog, service_time, 
                task_type=task_type, alpha=0.1
            )
            
            # 3. 更新关键决策字段
            # 我们将预测值和不确定性保存到状态中，供下一次 decide 使用
            pred_p90 = self._clamp_ms(pred.get('p90', 0.0), 1.0, 500.0, state.get('p90_belief', 110.0))
            unc_val = self._clamp_ms(uncertainty, 0.0, 60.0, state.get('uncertainty', 30.0))
            if pred_p90 >= 499.0 and p90_lat < 250.0:
                state['rls_state'] = {}
                state['scores'] = []
                pred_p90 = self._clamp_ms(p90_lat, 1.0, 500.0, 110.0)
                unc_val = min(20.0, unc_val)
            state['p90_belief'] = pred_p90
            state['uncertainty'] = unc_val
            state['last_alloc'] = cpu # 确保状态同步
            state['last_y'] = self._clamp_ms(p90_lat, 1.0, 500.0, pred_p90)
            state['last_prediction'] = pred_p90
            
            if mode in ["local", "memory", "mem", "inmem"]:
                try:
                    new_ver = str(int(ver) + 1)
                except Exception:
                    new_ver = "1"
                safe = self._sanitize_params(state)
                _LOCAL_STATE[target_id] = {"state": safe.copy(), "version": new_ver, "last_updated": float(time.time())}
                _L1_CACHE['params'] = safe.copy()
                _L1_CACHE['version'] = new_ver
                _L1_CACHE['last_sync'] = time.time()
                _L1_CACHE['state_id'] = target_id
            else:
                update_params = {
                    'TableName': TABLE_NAME,
                    'Key': {'id': {'S': target_id}},
                    'UpdateExpression': "SET state_blob = :b, lock_version = :new_lv, last_updated = :t",
                    'ExpressionAttributeValues': {
                        ':b': {'S': json.dumps(self._sanitize_params(state), allow_nan=False)},
                        ':new_lv': {'N': str(int(ver) + 1)},
                        ':t': {'N': str(time.time())}
                    },
                    'ConditionExpression': "lock_version = :expected_lv"
                }
                update_params['ExpressionAttributeValues'][':expected_lv'] = {'N': str(ver)}
                
                dynamodb_client.update_item(**update_params)
                _L1_CACHE['params'] = self._sanitize_params(state)
                _L1_CACHE['version'] = str(int(ver) + 1)
                _L1_CACHE['last_sync'] = time.time()
                _L1_CACHE['state_id'] = target_id
            if str(os.environ.get("WCP_LOG", "0")).strip() == "1":
                print(f"[WCP-v66] {target_id}: Pred={pred['p90']:.1f}, Margin={uncertainty:.1f}, E2E={p90_lat:.1f}")
            
        except Exception as e:
            print(f"[WCP-Update-Error] {e}")

    def update_metrics_ema(self, real_metrics):
        global _L1_CACHE
        try:
            response = dynamodb_client.get_item(
                TableName=TABLE_NAME,
                Key={'id': {'S': self.state_id}},
                ConsistentRead=True
            )
            item = response.get('Item')
            if not item or 'state_blob' not in item:
                state = self._get_default_params()
                state['code_version'] = 'baseline_ema'
                lock_ver = '0'
            else:
                state = self._sanitize_params(json.loads(item['state_blob']['S']))
                lock_ver = item.get('lock_version', {}).get('N', '0')

            curr_p90 = self._clamp_ms(state.get('p90_belief', 110.0), 1.0, 500.0, 110.0)
            new_val = self._clamp_ms(real_metrics.get('latency', 100.0), 1.0, 500.0, 100.0)

            alpha = 0.5 if new_val > curr_p90 else 0.05
            updated_p90 = (1 - alpha) * curr_p90 + alpha * new_val
            updated_p90 = self._clamp_ms(updated_p90, 1.0, 500.0, 110.0)

            state['p90_belief'] = updated_p90
            state['last_y'] = new_val

            self._sync_save_state(state, lock_ver)
            return updated_p90
        except Exception:
            return None

    def _parse_dynamo_item(self, item):
        state = {}
        params_map = item.get('params', {}).get('M', {})
        for key, value in params_map.items():
            if 'N' in value: state[key] = float(value['N'])
            elif 'S' in value: state[key] = value['S']
        for key, value in item.items():
            if key in ['params', 'id', 'lock_version', 'state_blob']: continue
            if 'N' in value: state[key] = float(value['N'])
            elif 'S' in value: state[key] = value['S']
        return state

    def _get_default_params(self):
        return {
            'last_alloc': 1.0,
            'p90_belief': 110.0,
            'uncertainty': 30.0, # Default margin for cold start
            'prev_rps': 0.0,
            'code_version': 'v66.0'
        }

    def _hydrate_controller(self, state):
        pass
