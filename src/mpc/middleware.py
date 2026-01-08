import time
import json
import boto3
import os
import random
from src.mpc.controller import MPCController
from src.wcp.wcp_update import wcp_update, detect_trend, slow_loop_calibration
from src.mpc.pricing import update_shadow_price

# --- Global L1 Cache (Container Re-use) ---
_L1_CACHE = {
    'params': None,
    'version': 0,
    'last_sync': 0
}
CACHE_TTL = 5.0 # seconds. Refresh from DB if older than this.

# AWS Clients
dynamodb = boto3.client('dynamodb')
sqs = boto3.client('sqs')

TABLE_NAME = 'MPC_State'
UPDATE_QUEUE_URL = os.environ.get('MPC_UPDATE_QUEUE_URL') # Optional: For async updates

class MPCMiddleware:
    def __init__(self, state_id='global_params'):
        self.state_id = state_id
        self.controller = MPCController()
        
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
        # If client didn't send metrics (Real Scenario), use our belief from state
        state, version = self._load_state()
        
        if not metrics or 'p90' not in metrics:
             metrics = metrics.copy()
             metrics['p90'] = float(state.get('p90_belief', 100.0))
             # Also assume CPU load correlates with p90 for now
             metrics['cpu_usage'] = 0.8 if metrics['p90'] > 500 else 0.2
        
        # Hydrate Controller
        self._hydrate_controller(state)
        
        # WCP Update (Prediction)
        # Note: We run WCP locally for prediction, but we don't save RLS state synchronously
        # to avoid high latency. RLS state changes are small and can be approximated or
        # pushed asynchronously.
        pred_dict, uncertainty, debug_info = wcp_update(state, metrics, alpha=0.1)
        
        # MPC Constraints
        wcp_constraints = {'pred': pred_dict, 'uncertainty': uncertainty}
        
        # System State
        system_state = {
            'shadow_price': state.get('shadow_price', 0.0),
            'last_alloc': float(state.get('last_alloc', 1.0)),
            'gamma': state.get('gamma', 0.1),
            'u_eta': state.get('u_eta', 0.05),
            'u_max_delta': state.get('u_max_delta', 0.15),
            'slo_limit': float(state.get('slo_limit', 500.0))
        }
        
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
        
        if random.random() < 0.1: # 10% sampling write to keep DB loosely synced
            self._async_save_state(state, version)
            
        return result['decision'], debug_info

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
             dynamodb.update_item(
                TableName=TABLE_NAME,
                Key={'id': {'S': self.state_id}},
                UpdateExpression="SET params.p90_belief = :p",
                ExpressionAttributeValues={':p': {'N': str(p90_val)}}
            )
        except Exception as e:
            print(f"Async Feedback Error: {e}")

    def _async_save_state(self, params, version):
        """
        Write to DynamoDB. In a real middleware, this would be put on a queue.
        Here we do a quick synchronous write but only 10% of time.
        """
        try:
            # Re-use the save logic logic, but simplified
            # We only update key fields to reduce WCU
            dynamodb.update_item(
                TableName=TABLE_NAME,
                Key={'id': {'S': self.state_id}},
                UpdateExpression="SET params.last_alloc = :a, shadow_price = :p, version = :v",
                ExpressionAttributeValues={
                    ':a': {'N': str(params.get('last_alloc', 1.0))},
                    ':p': {'N': str(params.get('shadow_price', 0.0))},
                    ':v': {'N': str(version + 1)}
                }
            )
        except Exception as e:
            print(f"Async Write Error: {e}")

    def _hydrate_controller(self, state):
        weights = state.get('optimizer_weights', {})
        self.controller.optimizer.w1 = weights.get('w1', 1.0)
        self.controller.optimizer.w2 = weights.get('w2', 0.5)
        self.controller.optimizer.w3 = weights.get('w3', 5.0)
        
    def _parse_dynamo_item(self, item):
        # ... (Simplified parser based on controller logic) ...
        # Copied essential parts
        params_map = item.get('params', {}).get('M', {})
        
        def get_float(m, k, default):
            try: return float(m.get(k, {}).get('N', default))
            except: return float(default)
            
        rls_states_json = item.get('rls_states', {}).get('S', '{}')
        try: rls_states = json.loads(rls_states_json)
        except: rls_states = {}

        return {
            'bP': get_float(params_map, 'bP', 2000.0),
            'rls_states': rls_states,
            'wcp_alpha': get_float(params_map, 'wcp_alpha', 0.1),
            'shadow_price': get_float(item, 'shadow_price', 0.0),
            'last_alloc': get_float(params_map, 'last_alloc', 1.0),
            'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0}, # Defaults
            'gamma': get_float(params_map, 'gamma', 0.1),
            'u_eta': get_float(params_map, 'u_eta', 0.05),
            'u_max_delta': get_float(params_map, 'u_max_delta', 0.15),
            'slo_limit': 500.0, # Default
            'p90_belief': get_float(params_map, 'p90_belief', 100.0)
        }

    def _get_default_params(self):
        return {
            'bP': 2000.0,
            'rls_states': {},
            'shadow_price': 0.0,
            'last_alloc': 1.0,
            'optimizer_weights': {'w1': 1.0, 'w2': 0.5, 'w3': 5.0},
            'gamma': 0.1,
            'u_eta': 0.05,
            'u_max_delta': 0.15,
            'slo_limit': 500.0,
            'p90_belief': 100.0
        }
