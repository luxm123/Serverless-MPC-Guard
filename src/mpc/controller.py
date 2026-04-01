from src.mpc.optimization import Optimizer

class MPCController:
    def __init__(self):
        self.optimizer = None

    def decide(self, task, constraints, system_state):
        # Hydrate optimizer with the latest parameters from system_state
        self.optimizer = Optimizer(params=system_state)
        
        prev_u = system_state.get('last_alloc', 1.0)
        uncertainty = system_state.get('uncertainty', 0.0)
        
        # v66.0: 将 WCP 提供的不确定性（Confidence Interval）传入优化器
        # pred_upper = 预测值 + 不确定性。这是我们要防御的“最坏情况”
        pred_upper = float(system_state.get('p90_belief', 120.0)) + float(uncertainty)
        
        # The core optimization logic
        slo_limit = float(system_state.get('slo_limit', 180.0))
        new_alloc = self.optimizer.optimize_u(
            prev_u=prev_u,
            pred_upper=pred_upper,
            slo_limit=slo_limit,
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
