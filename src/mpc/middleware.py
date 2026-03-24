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
        
        if _L1_CACHE['params'] and (now - _L1_CACHE['last_sync']) < _L1_CACHE['ttl_s'] and _L1_CACHE['state_id'] == self.state_id:
            return _L1_CACHE['params'], _L1_CACHE['version']

        try:
            response = dynamodb_client.get_item(
                TableName=TABLE_NAME, 
                Key={'id': {'S': self.state_id}},
                ConsistentRead=True
            )
            item = response.get('Item')
            if item:
                # v62: New JSON Blob format takes priority
                if 'state_blob' in item:
                    state = json.loads(item['state_blob']['S'])
                else:
                    # Fallback to legacy parsing for migration
                    state = self._parse_dynamo_item(item)
                
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
        
        original_state_id = self.state_id
        if self.state_id == 'global_params':
            self.state_id = f"mpc_state_{task_type}"
            
        # v62: The "Truth" Version - exposes errors to the logs
        current_logic_ver = 'v62.2'
        state, lock_ver = self._load_state()
        write_status = "NotAttempted"

        if not state or state.get('code_version') != current_logic_ver or event.get('reset_state'):
            state = self._get_default_params()
            state['code_version'] = current_logic_ver
            lock_ver = '0'
            print(f"[Middleware-v62.2] RESET for {self.state_id}")
            write_status = self._sync_save_state(state, lock_ver, force=True)
        
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
        
        # Only attempt second write if reset didn't just happen, or if we want to update with new decision
        write_status = self._sync_save_state(state, lock_ver)

        self.state_id = original_state_id
        
        # v62: We encode the write status into the version string so the user can see it!
        # Format: v62.0_L[LockVersion]_W[WriteStatus]
        debug_info = {
            'version': f"{current_logic_ver}_L{lock_ver}_W{write_status}",
            'code_version': current_logic_ver,
            'state_id': self.state_id,
            'p90_belief': system_state['p90_belief'],
            'prev_alloc': last_alloc,
            'new_alloc': new_alloc
        }
        
        return result['decision'], {**debug_info, **result.get('meta', {})}

    def _sync_save_state(self, params, version, force=False):
        global _L1_CACHE
        try:
            expected_version = int(version)
            new_version = expected_version + 1
            
            # v62: Use a JSON blob to avoid all attribute naming/type issues
            state_json = json.dumps(params)
            
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
            _L1_CACHE['params'] = params.copy()
            _L1_CACHE['last_sync'] = time.time()
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
        """v62.1: Restored update_metrics for worker Lambda"""
        global _L1_CACHE
        if _L1_CACHE['params']:
            curr_p90 = float(_L1_CACHE['params'].get('p90_belief', 100.0))
            new_val = float(real_metrics.get('latency', 100.0))
            
            # Asymmetric EMA: rise fast, fall slow
            alpha = 0.5 if new_val > curr_p90 else 0.05
            updated_p90 = (1 - alpha) * curr_p90 + alpha * new_val
            _L1_CACHE['params']['p90_belief'] = updated_p90
            
            new_rps = float(real_metrics.get('rps', _L1_CACHE['params'].get('prev_rps', 0.0)))
            _L1_CACHE['params']['prev_rps'] = new_rps
            
            _L1_CACHE['last_sync'] = time.time()
            # Synchronous save to ensure persistence
            self._sync_save_state(_L1_CACHE['params'], _L1_CACHE['version'])

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
            'p90_belief': 100.0,
            'prev_rps': 0.0,
            'code_version': 'v62.1'
        }

    def _hydrate_controller(self, state):
        pass
