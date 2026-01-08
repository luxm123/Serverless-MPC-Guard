import random
import math
import statistics

# --- 1. Simulation Engine (Generating Realistic Data) ---
def simulate_serverless_latency(alloc, base_latency=100.0, noise_level=0.1):
    """
    Simulates latency with non-linear resource scaling and random noise.
    Model: Latency = Base / Alloc + Overhead + Noise
    This represents the 'Ground Truth' that models try to learn.
    """
    # Inverse relationship: More resources -> Lower latency (up to a point)
    # Plus some fixed overhead (cold start, network)
    execution_time = base_latency / (alloc + 0.1) # Avoid div by zero
    overhead = 20.0 
    
    # Add random noise (heteroscedastic: more noise at low resources)
    noise = random.gauss(0, noise_level * execution_time)
    
    return execution_time + overhead + noise

def generate_trace(n_steps=200):
    """Generates a trace of (Alloc, Latency) pairs with shifting patterns."""
    data = []
    base_latency = 100.0
    
    for i in range(n_steps):
        # Simulate varying resource allocation (e.g., MPC exploring or random)
        alloc = random.uniform(0.1, 1.0)
        
        # Simulate pattern shift halfway (e.g., code update or cache warmup)
        if i > n_steps // 2:
            base_latency = 150.0 # Performance degradation
            
        latency = simulate_serverless_latency(alloc, base_latency)
        data.append({'step': i, 'alloc': alloc, 'latency': latency})
        
    return data

# --- 2. Model Implementations ---

class MovingAverageModel:
    def __init__(self, window_size=5):
        self.window = []
        self.window_size = window_size
        
    def predict(self, alloc):
        # Naive prediction: Next latency = Average of recent latencies
        # Note: MA ignores 'alloc' input, which is its fatal flaw in control systems
        if not self.window:
            return 100.0 # Default guess
        return sum(self.window) / len(self.window)
        
    def update(self, alloc, actual_latency):
        self.window.append(actual_latency)
        if len(self.window) > self.window_size:
            self.window.pop(0)

class LinearRLSModel:
    """Standard RLS: y = w1 * alloc + w0"""
    def __init__(self):
        # Features: [alloc, 1.0] (Bias term)
        self.theta = [0.0, 0.0] 
        # Covariance matrix P (2x2), initialized to large identity
        self.P = [[100.0, 0.0], [0.0, 100.0]]
        self.lambda_factor = 0.98 # Forgetting factor
        
    def predict(self, alloc):
        features = [alloc, 1.0]
        return sum(t * f for t, f in zip(self.theta, features))
        
    def update(self, alloc, y):
        x = [alloc, 1.0]
        
        # Pure Python RLS Update (no numpy for clarity/portability)
        # 1. Calculate gain k = P * x / (lambda + x^T * P * x)
        Px = [self.P[0][0]*x[0] + self.P[0][1]*x[1], 
              self.P[1][0]*x[0] + self.P[1][1]*x[1]]
        
        xPx = x[0]*Px[0] + x[1]*Px[1]
        denom = self.lambda_factor + xPx
        k = [p / denom for p in Px]
        
        # 2. Update error
        y_pred = self.predict(alloc)
        error = y - y_pred
        
        # 3. Update theta: theta = theta + k * error
        self.theta = [t + ki * error for t, ki in zip(self.theta, k)]
        
        # 4. Update P: P = (P - k * x^T * P) / lambda
        # k * x^T * P is equivalent to Outer(k, Px) ? No, k * (x^T * P)
        # Actually P_new = (P - Outer(k, x) * P) / lambda
        # Let's do: term = Outer(k, x) -> term * P
        
        # Simplified for 2x2:
        # P_new = (1/lambda) * (P - k * x^T * P)
        # x^T * P is just Px (since P is symmetric)
        
        # k * Px (outer product)
        kPx = [[k[0]*Px[0], k[0]*Px[1]],
               [k[1]*Px[0], k[1]*Px[1]]]
               
        for r in range(2):
            for c in range(2):
                self.P[r][c] = (self.P[r][c] - kPx[r][c]) / self.lambda_factor

class NonLinearRLSModel:
    """Enhanced RLS: y = w1 * (1/alloc) + w0"""
    # This matches the true physics (Latency ~ 1/Alloc) better
    def __init__(self):
        self.theta = [0.0, 0.0] 
        self.P = [[100.0, 0.0], [0.0, 100.0]]
        self.lambda_factor = 0.98
        
    def predict(self, alloc):
        # Feature transformation: 1/alloc
        feat_val = 1.0 / (alloc + 0.01) # Avoid div by zero
        features = [feat_val, 1.0]
        return sum(t * f for t, f in zip(self.theta, features))
        
    def update(self, alloc, y):
        # Same RLS logic, just different feature 'x'
        feat_val = 1.0 / (alloc + 0.01)
        x = [feat_val, 1.0]
        
        Px = [self.P[0][0]*x[0] + self.P[0][1]*x[1], 
              self.P[1][0]*x[0] + self.P[1][1]*x[1]]
        xPx = x[0]*Px[0] + x[1]*Px[1]
        denom = self.lambda_factor + xPx
        k = [p / denom for p in Px]
        
        y_pred = self.predict(alloc)
        error = y - y_pred
        
        self.theta = [t + ki * error for t, ki in zip(self.theta, k)]
        
        kPx = [[k[0]*Px[0], k[0]*Px[1]],
               [k[1]*Px[0], k[1]*Px[1]]]
        for r in range(2):
            for c in range(2):
                self.P[r][c] = (self.P[r][c] - kPx[r][c]) / self.lambda_factor


# --- 3. Experiment Runner ---

def run_experiment():
    print("Generating synthetic Serverless trace data...")
    data = generate_trace(n_steps=300)
    
    models = {
        'Moving Avg': MovingAverageModel(window_size=10),
        'Linear RLS': LinearRLSModel(),
        'Non-Linear RLS': NonLinearRLSModel()
    }
    
    results = {name: {'errors': [], 'predictions': []} for name in models}
    
    print("Running online training comparison...")
    
    for step_data in data:
        alloc = step_data['alloc']
        actual = step_data['latency']
        
        for name, model in models.items():
            # 1. Predict (Before seeing actual)
            pred = model.predict(alloc)
            
            # 2. Record Error
            error = abs(pred - actual)
            results[name]['errors'].append(error)
            results[name]['predictions'].append(pred)
            
            # 3. Learn (Update model)
            model.update(alloc, actual)
            
    # --- 4. Analysis ---
    print("\n>>> Model Comparison Results (Accuracy) <<<")
    print(f"{'Model':<20} | {'RMSE':<10} | {'MAPE (%)':<10} | {'Convergence Speed'}")
    print("-" * 65)
    
    for name, res in results.items():
        errors = res['errors']
        actuals = [d['latency'] for d in data]
        
        mse = statistics.mean([e**2 for e in errors])
        rmse = math.sqrt(mse)
        
        mape = statistics.mean([e / act for e, act in zip(errors, actuals)]) * 100
        
        # Simple convergence proxy: avg error in first 20 steps vs last 20
        early_err = statistics.mean(errors[:20])
        late_err = statistics.mean(errors[-20:])
        conv_speed = "Fast" if late_err < early_err * 0.5 else "Slow"
        
        print(f"{name:<20} | {rmse:<10.2f} | {mape:<10.2f} | {conv_speed}")
    print("-" * 65)
    
    print("\n[Analysis]")
    print("1. Moving Avg: Fails to account for resource changes (alloc), high error.")
    print("2. Linear RLS: Good baseline, adapts to trend, widely used.")
    print("3. Non-Linear RLS: Best physics match (1/alloc), lowest error, but more complex features.")

if __name__ == "__main__":
    run_experiment()
