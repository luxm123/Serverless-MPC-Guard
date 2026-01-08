import random
import math
import copy

# --- Simulation Environment ---

class SimulationEnv:
    def __init__(self, steps=1800, burst_interval=30, burst_duration=5):
        self.steps = steps
        self.burst_interval = burst_interval
        self.burst_duration = burst_duration
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        
    def next_step(self):
        t = self.current_step
        self.current_step += 1
        
        # Base metrics (Sine wave)
        # Period ~ 60 steps
        base_p90 = 100.0 + 20.0 * math.sin(t / 10.0)
        base_timeout = 0.01 + 0.005 * math.sin(t / 10.0)
        base_error = 0.005 + 0.002 * math.cos(t / 10.0)
        base_memory = 50.0 + 10.0 * math.sin(t / 20.0)
        
        # Burst Logic
        # Every 30s (steps), burst for 5s
        is_burst = False
        if (t % self.burst_interval) < self.burst_duration:
            is_burst = True
            base_p90 *= 2.0
            base_timeout *= 3.0
            base_error *= 2.0
            base_memory += 30.0
            
        # Noise
        # P90: +/- 50ms
        p90 = max(0.0, base_p90 + random.uniform(-50, 50))
        # Timeout: +/- 0.005
        timeout = max(0.0, base_timeout + random.uniform(-0.005, 0.005))
        # Error: +/- 0.005
        error = max(0.0, base_error + random.uniform(-0.005, 0.005))
        # Memory: +/- 10MB
        memory = max(0.0, base_memory + random.uniform(-10, 10))
        
        metrics = {
            'p90': p90,
            'timeout_rate': timeout,
            'error_rate': error,
            'memory_pressure': memory
        }
        
        return metrics, is_burst

# --- Metrics Recorder ---

class ExperimentRecorder:
    def __init__(self, name):
        self.name = name
        self.records = []
        
    def add_record(self, step, observed, predicted, uncertainty, is_covered, is_burst, bounds=None):
        self.records.append({
            'step': step,
            'observed_p90': observed['p90'],
            'predicted_p90': predicted.get('p90', 0.0),
            'uncertainty': uncertainty,
            'is_covered': is_covered,
            'is_burst': is_burst,
            'bounds_width': (bounds['p90']['upper'] - bounds['p90']['lower']) if bounds else 0.0
        })
        
    def print_summary(self):
        total = len(self.records)
        if total == 0:
            print(f"[{self.name}] No data.")
            return
            
        covered_count = sum(1 for r in self.records if r['is_covered'])
        coverage_rate = covered_count / total
        
        # Burst Analysis
        burst_records = [r for r in self.records if r['is_burst']]
        burst_coverage = sum(1 for r in burst_records if r['is_covered']) / len(burst_records) if burst_records else 0.0
        
        # Width Analysis
        widths = [r['bounds_width'] for r in self.records]
        avg_width = sum(widths) / total
        
        print(f"=== {self.name} Summary ===")
        print(f"Total Steps: {total}")
        print(f"Coverage Rate: {coverage_rate:.2%} (Target: 98%)")
        print(f"Burst Coverage: {burst_coverage:.2%}")
        print(f"Avg Interval Width: {avg_width:.2f}")
        print("==============================\n")
        
        return {
            'coverage': coverage_rate,
            'burst_coverage': burst_coverage,
            'avg_width': avg_width
        }
