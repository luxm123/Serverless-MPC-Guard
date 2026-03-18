
class AwsBaselineController:
    """
    A controller that simulates the AWS Target Tracking scaling policy.
    """
    def __init__(self, target_utilization=0.75, scale_up_cooldown_sec=60, scale_down_cooldown_sec=300):
        self.target_utilization = target_utilization
        self.scale_up_cooldown_sec = scale_up_cooldown_sec
        self.scale_down_cooldown_sec = scale_down_cooldown_sec
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0

    def get_decision(self, metrics, current_alloc):
        """
        Calculates the new resource allocation based on target tracking logic.

        Args:
            metrics (dict): Current system metrics, must include 'cpu_util'.
            current_alloc (float): The current resource allocation.

        Returns:
            dict: A decision dictionary with the new CPU allocation.
        """
        import time
        now = time.time()
        actual_utilization = metrics.get('cpu_util', 0.0)

        if actual_utilization == 0:
            return {'cpu_cores': current_alloc} # No data, no change

        # Calculate desired allocation
        desired_alloc = current_alloc * (actual_utilization / self.target_utilization)

        # Apply cooldown periods
        if desired_alloc > current_alloc:
            if (now - self.last_scale_up_time) < self.scale_up_cooldown_sec:
                # Still in scale-up cooldown, do not scale up further
                return {'cpu_cores': current_alloc}
            self.last_scale_up_time = now
        elif desired_alloc < current_alloc:
            if (now - self.last_scale_down_time) < self.scale_down_cooldown_sec:
                # Still in scale-down cooldown, do not scale down
                return {'cpu_cores': current_alloc}
            self.last_scale_down_time = now
        
        # For simplicity, we directly apply the desired allocation.
        # A more complex implementation would use step scaling policies.
        # We also clamp the allocation within a reasonable range, e.g., [0.5, 4.0]
        new_alloc = max(0.5, min(desired_alloc, 4.0))

        return {'cpu_cores': new_alloc}

