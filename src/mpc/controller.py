from src.mpc.optimization import Optimizer

class MPCController:
    def __init__(self):
        self.optimizer = None

    def decide(self, task, constraints, system_state):
        # Hydrate optimizer with the latest parameters from system_state
        self.optimizer = Optimizer(params=system_state)
        
        prev_u = system_state.get('last_alloc', 1.0)
        
        # The core optimization logic
        new_alloc = self.optimizer.optimize_u(
            prev_u=prev_u,
            pred_upper=None, # Not used in this simplified version
            slo_limit=180.0,
            price=0.0, # Not used
            state=system_state # Pass the full state
        )
        
        decision = {
            'resource_alloc': new_alloc,
            'should_shed': False # Shedding logic is disabled
        }
        
        meta = {
            'opt_debug': system_state.get('opt_debug', {})
        }
        
        return {'decision': decision, 'meta': meta}
