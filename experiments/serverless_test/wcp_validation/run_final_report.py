import boto3
import time
import os
import sys

# Add current directory to path so we can import serverless_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from serverless_utils import run_experiment_sequence

# Configuration
STEPS = 20
REGION = 'us-east-1'
TABLE_NAME = 'MPC_State'

def clear_db():
    """Resets the global_params in DynamoDB to ensure clean state."""
    dynamodb = boto3.client('dynamodb', region_name=REGION)
    try:
        dynamodb.delete_item(
            TableName=TABLE_NAME,
            Key={'id': {'S': 'global_params'}}
        )
        # print("  [System] State cleared.")
    except Exception as e:
        print(f"  [System] Error clearing state: {e}")

def calculate_stats(history):
    if not history:
        return {'coverage': 0.0, 'burst_coverage': 0.0, 'avg_width': 0.0}
    
    # Total Coverage
    covered_count = sum(1 for step in history if step['covered'])
    coverage = (covered_count / len(history)) * 100
    
    # Burst Coverage (Steps 3-6, indices 3,4,5,6)
    burst_steps = [h for h in history if 3 <= h['step'] <= 6]
    if burst_steps:
        burst_covered = sum(1 for step in burst_steps if step['covered'])
        burst_coverage = (burst_covered / len(burst_steps)) * 100
    else:
        burst_coverage = 0.0
        
    # Average Width
    # Width = upper - lower
    widths = [(step['upper'] - step['lower']) for step in history]
    avg_width = sum(widths) / len(widths) if widths else 0.0
    
    return {
        'coverage': coverage,
        'burst_coverage': burst_coverage,
        'avg_width': avg_width
    }

def run_all():
    print("Running all experiments... (This may take ~1 minute)")
    
    results = {}
    modes = [
        ('Baseline (G1)', 'baseline'),
        ('Simple WCP (G2)', 'simple'),
        ('Strict WCP (G3)', 'strict'),
        ('Lite A (Chebyshev)', 'lite_a'),
        ('Lite B (Streaming)', 'lite_b')
    ]
    
    for label, mode in modes:
        print(f"Executing {label}...")
        clear_db()
        # Initial wait for cold start simulation if needed, but delete_item handles the state reset.
        # We might want a tiny sleep to ensure consistency.
        time.sleep(1) 
        
        hist = run_experiment_sequence(mode=mode, steps=STEPS, quiet=True)
        stats = calculate_stats(hist)
        results[label] = stats
        
    print("\n" + "="*20 + " FINAL REPORT " + "="*20 + "\n")
    
    # --- Comparison 1: WCP Value Add ---
    print("[Comparison 1: WCP Value Add]")
    print("Baseline (G1) provides NO uncertainty interval (Width=0).")
    g3_stats = results['Strict WCP (G3)']
    print(f"Strict WCP (G3) Coverage: {g3_stats['coverage']:.2f}%")
    print(f"Strict WCP (G3) Burst Coverage: {g3_stats['burst_coverage']:.2f}%")
    if g3_stats['coverage'] > 90:
        print("-> WCP successfully quantifies risk where Baseline is blind.")
    else:
        print("-> WCP provides risk quantification.")
    print("")

    # --- Comparison 2: Necessity of Weights (G2 vs G3) ---
    print("[Comparison 2: Necessity of Weights (G2 vs G3)]")
    g2_stats = results['Simple WCP (G2)']
    print(f"Simple WCP (G2) Coverage: {g2_stats['coverage']:.2f}%")
    print(f"Strict WCP (G3) Coverage: {g3_stats['coverage']:.2f}%")
    print(f"Simple WCP (G2) Avg Width: {g2_stats['avg_width']:.2f}")
    print(f"Strict WCP (G3) Avg Width: {g3_stats['avg_width']:.2f}")
    
    if g3_stats['coverage'] > g2_stats['coverage']:
        print("SUCCESS: Strict WCP handles bursts better than Simple WCP.")
    elif g3_stats['avg_width'] > g2_stats['avg_width']:
        print("NOTE: Strict WCP uses wider intervals to maintain coverage.")
    print("")

    # --- Comparison 3: Serverless Adaptation (Lite A/B) ---
    print("[Comparison 3: Serverless Adaptation (Lite A vs Lite B)]")
    g4_stats = results['Lite A (Chebyshev)']
    g5_stats = results['Lite B (Streaming)']
    
    print(f"Lite A (Chebyshev) Coverage: {g4_stats['coverage']:.2f}% (Avg Width: {g4_stats['avg_width']:.2f})")
    print(f"Lite B (Streaming) Coverage: {g5_stats['coverage']:.2f}% (Avg Width: {g5_stats['avg_width']:.2f})")
    
    if g5_stats['coverage'] >= 90:
        print("SUCCESS: Lite B (Streaming) achieves target coverage with constant memory.")
    else:
        print("WARNING: Lite B coverage below target.")
        
    print("")
    
    # --- Final Summary ---
    if g3_stats['coverage'] >= 90:
        print("OVERALL SUCCESS: Strict WCP meets coverage target (>90%).")
    else:
        print("OVERALL WARNING: Strict WCP coverage below target.")

if __name__ == "__main__":
    run_all()
