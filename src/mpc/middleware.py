import time
import json
import random
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
    'ttl_s': 0.5, # Cache state for 500ms within a container
    'state_id': None
}

class MPCMiddleware:
    def __init__(self, state_id='global_params'):
        from src.mpc.controller import MPCController
        self.state_id = state_id
        self.controller = MPCController()

    def _load_state(self):
        global _L1_CACHE
        now = time.time()
        
        # v59: Use L1 cache if valid and state_id matches
        if _L1_CACHE['params'] and (now - _L1_CACHE['last_sync']) < _L1_CACHE['ttl_s'] and _L1_CACHE['state_id'] == self.state_id:
            return _L1_CACHE['params'], _L1_CACHE['version']

        try:
            # v59.2: Force ConsistentRead to ensure we get the latest update from other containers
            response = dynamodb_client.get_item(
                TableName=TABLE_NAME, 
                Key={'id': {'S': self.state_id}},
                ConsistentRead=True
            )
            item = response.get('Item')
            if item:
                state = self._parse_dynamo_item(item)
                # v59.3: Numeric lock version for optimistic concurrency
                lock_version = item.get('lock_version', {}).get('N', '0')
                
                # v59: Update cache
                _L1_CACHE['params'] = state
                _L1_CACHE['version'] = lock_version
                _L1_CACHE['last_sync'] = now
                _L1_CACHE['state_id'] = self.state_id
                
                return state, lock_version
        except Exception as e:
            print(f"Error loading state for {self.state_id}: {e}")
        
        return None, None

    def decide(self, event):
        start_t = time.time()
        
        metrics = event.get('metrics', {})
        task = event.get('task', {})
        task_type = task.get('task_type', event.get('task_type', 'image_processing'))
        strategy = event.get('strategy', 'mpc_integrated')
        
        original_state_id = self.state_id
        if self.state_id == 'global_params':
            self.state_id = f"mpc_state_{task_type}"
            
        # v59.6: Final attempt to fix state loop by going back to nested params
        current_logic_ver = 'v59.6_OptimisticLocking'
        debug_info = {'code_version': current_logic_ver, 'version': current_logic_ver, 'state_id': self.state_id}

        state, lock_ver = self._load_state()
        
        if not state or state.get('code_version') != current_logic_ver or event.get('reset_state'):
            state = self._get_default_params()
            state['code_version'] = current_logic_ver
            lock_ver = '0'
            debug_info['state_source'] = 'forced_reset'
            print(f"[Middleware-v59.6] NUCLEAR RESET for {self.state_id}")
        else:
            debug_info['state_source'] = 'dynamodb_or_cache'
            
        last_alloc = float(state.get('last_alloc', 1.0))
        
        current_rps = float(metrics.get('rps', 0.0))
        prev_rps = float(state.get('prev_rps', 0.0))
        state['prev_rps'] = current_rps

        if not metrics or 'p90' not in metrics:
             metrics = metrics.copy()
             metrics['p90'] = float(state.get('p90_belief', 100.0))

        self._hydrate_controller(state)
        
        system_state = {
            'last_alloc': last_alloc,
            'p90_belief': float(state.get('p90_belief', 100.0)),
            'strategy': strategy,
            'current_rps': current_rps,
            'prev_rps': prev_rps,
        }

        result = self.controller.decide(task, {}, system_state)
        
        new_alloc = float(result.get('decision', {}).get('resource_alloc', last_alloc))
        state['last_alloc'] = new_alloc
        state['p90_belief'] = system_state['p90_belief']
        
        # v59.6: Use a dedicated save method for the nested params format
        self._async_save_state_v2(state, lock_ver)

        self.state_id = original_state_id
        
        debug_info['p90_belief'] = system_state['p90_belief']
        debug_info['prev_alloc'] = last_alloc
        debug_info['new_alloc'] = new_alloc
        
        return result['decision'], {**debug_info, **result.get('meta', {})}

    def update_metrics(self, real_metrics):
        global _L1_CACHE
        if _L1_CACHE['params']:
            curr_p90 = float(_L1_CACHE['params'].get('p90_belief', 100.0))
            new_val = float(real_metrics.get('latency', 100.0))
            
            alpha = 0.5 if new_val > curr_p90 else 0.05
            updated_p90 = (1 - alpha) * curr_p90 + alpha * new_val
            _L1_CACHE['params']['p90_belief'] = updated_p90
            
            new_rps = float(real_metrics.get('rps', _L1_CACHE['params'].get('prev_rps', 0.0)))
            _L1_CACHE['params']['prev_rps'] = new_rps
            
            _L1_CACHE['last_sync'] = time.time()
            self._async_save_state(_L1_CACHE['params'], _L1_CACHE['version'])

    def _async_save_state_v2(self, params, version):
        global _L1_CACHE
        try:
            expected_version = int(version)
            new_version = expected_version + 1
            
            # Pack all attributes into a nested 'params' M-type for maximum reliability
            params_m = {}
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    params_m[k] = {'N': str(v)}
                else:
                    params_m[k] = {'S': str(v)}

            dynamodb_client.update_item(
                TableName=TABLE_NAME,
                Key={'id': {'S': self.state_id}},
                UpdateExpression="SET #p = :p, #lv = :new_lv, last_updated = :t",
                ConditionExpression="#lv = :expected_lv OR attribute_not_exists(#lv)",
                ExpressionAttributeNames={'#p': 'params', '#lv': 'lock_version'},
                ExpressionAttributeValues={
                    ':p': {'M': params_m},
                    ':new_lv': {'N': str(new_version)},
                    ':expected_lv': {'N': str(expected_version)},
                    ':t': {'N': str(time.time())}
                }
            )
            
            _L1_CACHE['version'] = str(new_version)
            _L1_CACHE['params'] = params.copy()
            _L1_CACHE['last_sync'] = time.time()
            _L1_CACHE['state_id'] = self.state_id
            
        except Exception as e:
            print(f"v59.6 Save Error: {e}")

    def _async_save_state(self, params, version):
        # Legacy method kept for safety, but decide() now calls _v2
        self._async_save_state_v2(params, version)

    def _parse_dynamo_item(self, item):
        # v59.5: Optimized parsing to handle legacy clobbering
        state = {}
        
        # 1. Fallback to nested params map first (Legacy format)
        params_map = item.get('params', {}).get('M', {})
        for key, value in params_map.items():
            if 'N' in value:
                state[key] = float(value['N'])
            elif 'S' in value:
                state[key] = value['S']
        
        # 2. Top-level attributes take priority and overwrite params (New format)
        for key, value in item.items():
            if key in ['params', 'id']: continue
            if 'N' in value:
                state[key] = float(value['N'])
            elif 'S' in value:
                state[key] = value['S']
        
        return state

    def _get_default_params(self):
        return {
            'last_alloc': 1.0,
            'p90_belief': 100.0,
            'prev_rps': 0.0,
            'code_version': 'v59.6_OptimisticLocking'
        }

    def _hydrate_controller(self, state):
        # This can be used to pass complex objects to the controller if needed
        pass
