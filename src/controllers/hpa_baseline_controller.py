
import time

class HpaBaselineController:
    """
    A controller that strictly simulates the HPA (Horizontal Pod Autoscaler) 
    baseline used in Jiagu (ATC '24).
    
    Parameters:
        - target_utilization: 0.8 (80% CPU usage threshold)
        - window_sec: 15s (Decision window)
        - scale_up_limit: 100% (Max scale up per step)
        - scale_down_limit: 50% (Max scale down per step)
    """
    def __init__(self, target_utilization=0.8, window_sec=15):
        self.target_utilization = target_utilization
        self.window_sec = window_sec
        self.last_decision_time = 0
        self.current_alloc = 1.0

    def get_decision(self, metrics, current_alloc):
        """
        Calculates the new resource allocation based on HPA logic.
        """
        now = time.time()
        self.current_alloc = current_alloc
        
        # 1. Check decision window (15s as per Jiagu)
        if (now - self.last_decision_time) < self.window_sec:
            return {'cpu_cores': self.current_alloc}

        actual_utilization = metrics.get('cpu_util', 0.0)
        if actual_utilization == 0:
            return {'cpu_cores': self.current_alloc}

        # 2. Calculate desired allocation: Desired = Current * (Actual / Target)
        # As per Kubernetes HPA formula
        desired_alloc = self.current_alloc * (actual_utilization / self.target_utilization)

        # 3. Apply Jiagu's constraints: Max scale up 100%, Max scale down 50%
        max_up = self.current_alloc * 2.0
        min_down = self.current_alloc * 0.5
        
        final_alloc = max(min_down, min(desired_alloc, max_up))
        
        # 4. Physical boundaries (matching our AWS Lambda worker limits)
        final_alloc = max(0.5, min(final_alloc, 4.0))

        if final_alloc != self.current_alloc:
            self.last_decision_time = now
            self.current_alloc = final_alloc

        return {'cpu_cores': self.current_alloc}
