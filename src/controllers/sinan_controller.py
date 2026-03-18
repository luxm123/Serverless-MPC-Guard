
import numpy as np

class SinanController:
    """
    A controller that mimics the core logic of the Sinan paper (ASPLOS '21).
    It uses a simplified predictive model to choose the best resource allocation
    from a set of discrete proposals.
    """
    def __init__(self, target_slo_p90_ms=400):
        self.target_slo_p90_ms = target_slo_p90_ms
        # A very simple linear model: predicted_latency = base_latency - cpu_cores * factor
        # This is a placeholder for Sinan's complex CNN+XGBoost model.
        # We assume more CPU leads to lower latency, which is a reasonable simplification.
        self.perf_model_base_ms = 600  # Predicted latency with 0 CPU
        self.perf_model_cpu_factor = 150 # Milliseconds of latency reduction per CPU core

    def _predict_latency(self, cpu_proposal):
        """
        Predicts the p90 latency for a given CPU allocation proposal.
        This is a simplified stand-in for Sinan's ML model.
        """
        predicted_latency = self.perf_model_base_ms - cpu_proposal * self.perf_model_cpu_factor
        return max(predicted_latency, 50) # Assume a minimum possible latency

    def get_decision(self, observation):
        """
        Generates a set of proposals and chooses the cheapest one that meets the SLO.
        
        Args:
            observation (dict): The current system observation. Not used in this simplified
                                version, but kept for interface compatibility.

        Returns:
            dict: A decision dictionary with the chosen CPU allocation.
        """
        # Sinan generates a set of possible actions (proposals).
        # Here we define a static set of CPU allocation proposals.
        cpu_proposals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        valid_proposals = []

        for cpu in cpu_proposals:
            predicted_latency = self._predict_latency(cpu)
            
            # Check if the predicted latency meets the QoS target (SLO)
            if predicted_latency <= self.target_slo_p90_ms:
                valid_proposals.append(cpu)

        # From the proposals that meet the SLO, choose the one with the lowest cost (least CPU).
        if valid_proposals:
            best_cpu_allocation = min(valid_proposals)
        else:
            # If no proposal can meet the SLO, choose the one with the most resources
            # as a best-effort attempt.
            best_cpu_allocation = max(cpu_proposals)
            
        return {'cpu_cores': best_cpu_allocation}

