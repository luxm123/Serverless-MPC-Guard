import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import TraceReplayer from the sibling directory
from experiments.serverless_test.trace_experiment.run_trace_replay import TraceReplayer
from experiments.serverless_test.wcp_validation.serverless_utils import force_cold_start

def run_scalability_experiment():
    # Configuration
    # Scalability Test: Iterate through increasing concurrency levels
    concurrency_levels = [200, 500, 800] 
    
    trace_file = os.path.join(PROJECT_ROOT, 'datasets', 'processed', 'clean_trace.csv')
    base_output_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(base_output_dir, exist_ok=True)
    
    summary_data = []

    target_funcs = [
        os.environ.get('MPC_CONTROLLER_NAME', 'MPC_Controller'),
        os.environ.get('MPC_WORKER_NAME', 'MPC_BusinessWorker')
    ]

    for concurrency in concurrency_levels:
        print(f"\n{'='*50}")
        print(f"Running Experiment with Concurrency Level: {concurrency}")
        print(f"{'='*50}")
        
        current_output_dir = os.path.join(base_output_dir, f'concurrency_{concurrency}')
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Initialize Replayer with specific thread count
        # Note: We create a new instance for each level to reset state if needed
        replayer = TraceReplayer(trace_file=trace_file, output_dir=current_output_dir, thread_num=concurrency)
        replayer.load_trace()
        
        # Run 3 Trials per concurrency level
        for trial in range(1, 4):
            trial_suffix = f"_run{trial}"
            print(f"\n--- Trial {trial}/3 (Concurrency={concurrency}) ---")
            
            # 1. Run MPC
            print(f"Running MPC...")
            force_cold_start(target_funcs)
            replayer.run_experiment(strategy='mpc', wcp_mode='strict', output_filename=f'results_mpc{trial_suffix}.csv', mpc_profile='scalability_tuned')
            
            # 2. Run Baseline
            print(f"Running Baseline...")
            force_cold_start(target_funcs)
            replayer.run_experiment(strategy='baseline', wcp_mode='baseline', output_filename=f'results_baseline{trial_suffix}.csv')
            
            # Analyze results for this trial
            try:
                df_mpc = pd.read_csv(os.path.join(current_output_dir, f'results_mpc{trial_suffix}.csv'))
                df_base = pd.read_csv(os.path.join(current_output_dir, f'results_baseline{trial_suffix}.csv'))
                
                summary_data.append(get_metrics(df_mpc, 'MPC', concurrency))
                summary_data.append(get_metrics(df_base, 'Baseline', concurrency))
            except Exception as e:
                print(f"Error analyzing results for concurrency {concurrency}, trial {trial}: {e}")
            
            time.sleep(2) # Brief cooldown

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(base_output_dir, 'scalability_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")
    
    # Plotting
    plot_scalability_results(summary_df, base_output_dir)

def get_metrics(df, strategy_name, concurrency):
    # Violation Rate
    violation_rate = df['slo_violation'].mean() * 100
    # Success Rate (Goodput)
    success_rate = (df['success'] & ~df['slo_violation']).mean() * 100
    # P99 Latency
    p99 = df['e2e_latency'].quantile(0.99)
    # Average Latency
    avg_lat = df['e2e_latency'].mean()
    
    return {
        'Concurrency': concurrency,
        'Strategy': strategy_name,
        'SLO Violation Rate (%)': violation_rate,
        'Success Rate (%)': success_rate,
        'P99 Latency (ms)': p99,
        'Average Latency (ms)': avg_lat
    }

def plot_scalability_results(df, output_dir):
    # Set style to match user preference (Whitegrid + Pastel)
    sns.set_theme(style="whitegrid")
    
    # Custom pastel palette
    custom_palette = {'MPC': '#CCEBC5', 'Baseline': '#FBB4AE'} # Green-ish for MPC, Red-ish for Baseline
    
    # Plot 1: SLO Violation Rate vs Concurrency
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Concurrency', y='SLO Violation Rate (%)', hue='Strategy', 
                 palette=custom_palette, marker='o', linewidth=3, markersize=10)
    plt.title('Scalability: SLO Violation Rate vs. Concurrency', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Concurrency Level (Threads)', fontsize=14)
    plt.ylabel('SLO Violation Rate (%)', fontsize=14)
    plt.xticks(df['Concurrency'].unique())
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_slo_violation.png'), dpi=300)
    plt.close()
    
    # Plot 2: P99 Latency vs Concurrency
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Concurrency', y='P99 Latency (ms)', hue='Strategy', 
                 palette=custom_palette, marker='s', linewidth=3, markersize=10)
    plt.title('Scalability: Tail Latency (P99) vs. Concurrency', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Concurrency Level (Threads)', fontsize=14)
    plt.ylabel('P99 Latency (ms)', fontsize=14)
    plt.xticks(df['Concurrency'].unique())
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_p99_latency.png'), dpi=300)
    plt.close()
    
    print("\nScalability plots generated in:", output_dir)

if __name__ == "__main__":
    run_scalability_experiment()
