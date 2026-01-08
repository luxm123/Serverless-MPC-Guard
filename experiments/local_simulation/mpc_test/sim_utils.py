import random
import time
import math
from collections import deque

class TaskGenerator:
    """Generates synthetic tasks with varying priorities and characteristics."""
    def __init__(self, core_ratio=0.3, med_ratio=0.35, low_ratio=0.35):
        self.ratios = [core_ratio, med_ratio, low_ratio] # High, Med, Low
        
    def generate_task(self, task_id):
        rand = random.random()
        if rand < self.ratios[0]:
            priority = 'critical'
            consistency = 'strong'
            base_latency = 100.0 # ms
            slo = 200.0
        elif rand < self.ratios[0] + self.ratios[1]:
            priority = 'high'
            consistency = 'eventual'
            base_latency = 150.0
            slo = 500.0
        else:
            priority = 'low'
            consistency = 'eventual'
            base_latency = 200.0
            slo = 1000.0
            
        return {
            'id': task_id,
            'priority': priority,
            'consistency': consistency,
            'base_latency': base_latency,
            'slo': slo,
            'timestamp': time.time()
        }

class SimulatorEnv:
    """Simulates system state, resource usage, and latency execution."""
    def __init__(self):
        self.cpu_util = 0.5
        self.memory_util = 0.4
        self.congestion_price = 0.0
        self.latency_noise_std = 20.0
        self.cold_start_prob = 0.0
        self.burst_active = False
        
        # Stats tracking
        self.stats = {
            'total_tasks': 0,
            'slo_violations': 0,
            'core_slo_violations': 0,
            'core_tasks': 0,
            'resource_waste': 0.0,
            'long_tail_count': 0,
            'shed_count': 0
        }
        
    def step(self, task, decision):
        """
        Execute one step of simulation based on decision.
        decision: {should_shed, degrade_plan, resource_alloc}
        """
        self.stats['total_tasks'] += 1
        if task['priority'] == 'critical':
            self.stats['core_tasks'] += 1
            
        # 1. Handle Load Shedding
        if decision['should_shed']:
            self.stats['shed_count'] += 1
            # Shedding implies failure for SLO in this simple sim unless handled by queue (ignored here for simplicity)
            # Or we can consider shedding as "failed to serve in time"
            return {
                'latency': 0.0, 
                'success': False, 
                'shed': True,
                'cpu_util': self.cpu_util,
                'memory_util': self.memory_util,
                'congestion_price': self.congestion_price
            }
            
        # 2. Simulate Execution Latency
        # Base latency + Noise + Resource Contention Penalty
        exec_latency = task['base_latency'] + random.gauss(0, self.latency_noise_std)
        
        # Resource Penalty: If alloc < 1.0, latency increases
        alloc = decision.get('resource_alloc', 1.0)
        if alloc < 1.0:
            exec_latency *= (1.0 + (1.0 - alloc)) # e.g. 0.5 alloc -> 1.5x latency
            
        # Cold Start Injection
        if random.random() < self.cold_start_prob:
            exec_latency += random.uniform(500, 2000)
            
        # Burst Penalty
        if self.burst_active:
            exec_latency += 300.0
            
        exec_latency = max(10.0, exec_latency)
        
        # 3. Update System State (Random Walk)
        self.cpu_util = max(0.1, min(1.0, self.cpu_util + random.uniform(-0.05, 0.05)))
        self.memory_util = max(0.1, min(1.0, self.memory_util + random.uniform(-0.05, 0.05)))
        
        # Update Congestion Price (Simple Logic)
        if self.cpu_util > 0.8:
            self.congestion_price += 10.0
        else:
            self.congestion_price = max(0.0, self.congestion_price - 5.0)
            
        # 4. Record Metrics
        # SLO Check
        if exec_latency > task['slo']:
            self.stats['slo_violations'] += 1
            if task['priority'] == 'critical':
                self.stats['core_slo_violations'] += 1
                
        # Long Tail Check (> 2x SLO)
        if exec_latency > task['slo'] * 2:
            self.stats['long_tail_count'] += 1
            
        # Resource Waste (Allocated - Used)
        # Simplified: Assume task needed 0.5 CPU. If we gave 1.0, waste is 0.5.
        # If we gave 0.5 and it needed 0.5, waste is 0.
        needed = 0.5 
        waste = max(0.0, alloc - needed)
        self.stats['resource_waste'] += waste
        
        return {
            'latency': exec_latency,
            'success': True,
            'shed': False,
            'cpu_util': self.cpu_util,
            'memory_util': self.memory_util,
            'congestion_price': self.congestion_price
        }

    def inject_disturbance(self, dist_type):
        if dist_type == 'resource_fluctuation':
            self.cpu_util = min(1.0, self.cpu_util + 0.3)
        elif dist_type == 'cold_start':
            self.cold_start_prob = 0.1
        elif dist_type == 'burst':
            self.burst_active = True
            
    def clear_disturbance(self):
        self.cold_start_prob = 0.0
        self.burst_active = False

