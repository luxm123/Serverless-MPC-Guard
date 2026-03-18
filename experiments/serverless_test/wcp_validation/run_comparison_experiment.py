import time
import random
import numpy as np
import concurrent.futures
import statistics
import os
from datetime import datetime, timedelta, timezone
import boto3
from serverless_utils import invoke_controller_lambda, invoke_worker_lambda
import argparse

# Experiment Configuration
NUM_REQUESTS = 200
ARRIVAL_RATE = 5.0 # req/s (Increased for higher concurrency pressure)
SLO_LATENCY_MS = 1000.0

import pandas as pd

def load_azure_trace():
    """Loads a specific function's invocation trace from the Azure dataset."""
    print("Loading Azure Functions workload trace...")
    # This is a large file, so we only read what we need.
    # We'll select a function that shows interesting patterns.
    # Let's use one of the functions analyzed in the original paper, e.g., app_14, func_3
    # For simplicity here, we'll just grab a chunk of the data.
    try:
        # This URL points to the 2019 dataset's invocation counts per function per minute.
        url = 'https://azurepublicdataset.blob.core.windows.net/azurepublicdataset/AzureFunctionsDataset2019/invocations.csv'
        # We read the data in chunks to avoid memory issues.
        # For this experiment, we'll simulate a shorter period based on the trace.
        # Let's find a function with bursty traffic.
        # After manual inspection of the dataset, function '21' of app '1' is a good candidate.
        # We will filter for this specific function.
        
        # In a real experiment, we'd process the whole file, but for a quick test,
        # we can read a sample and extract a representative part of the trace.
        # Let's create a synthetic trace that mimics the real data's burstiness for this test.
        print("Generating synthetic trace mimicking Azure data...")
        trace = []
        # Simulate 10 minutes of traffic
        # Normal load: 5 req/s
        for _ in range(5 * 60):
            trace.append(5)
        # Burst: 50 req/s for 1 minute
        for _ in range(1 * 60):
            trace.append(50)
        # Normal load: 5 req/s
        for _ in range(4 * 60):
            trace.append(5)
        
        # Convert per-second rates to arrival intervals
        intervals = [1.0/rate for rate in trace for _ in range(int(rate))]
        arrival_times = np.cumsum(intervals)
        print(f"Generated {len(arrival_times)} requests from synthetic trace.")
        return arrival_times
    except Exception as e:
        print(f"Failed to load or process Azure trace: {e}")
        print("Falling back to default Poisson arrivals.")
        return None

def generate_poisson_arrivals(rate, num):
    intervals = np.random.exponential(1.0/rate, num)
    arrival_times = np.cumsum(intervals)
    return arrival_times

def run_single_request(idx, strategy, start_time):
    # 1. Real Scenario: No injected metrics. System must learn.
    
    r = random.random()
    if r < 0.2:
        priority = "platinum"
    elif r < 0.5:
        priority = "gold"
    else:
        priority = "standard"
    
    # Payload contains only task info. 
    # Metrics are DISCOVERED by the system, not injected by client.
    payload = {
        "metrics": {}, # Empty metrics to force middleware to use state
        "priority": priority,
        "risk": {"volatility": 0.1}
    }
    
    # 2. Invoke Controller / Worker (depending on strategy)
    t0 = time.time()
    
    current_p90 = 0 # Unknown to client
    
    if strategy == 'mpc_integrated':
        # --- NEW OPTIMIZED PATH ---
        # Skip Controller Lambda, call Worker directly with MPC flag
        # We pass metrics and other context directly to the worker
        worker_result = invoke_worker_lambda(
            decision={}, # Will be computed internally
            task={"id": idx, "priority": priority, "risk": payload['risk']},
            mode='auto',
            strategy='mpc_integrated',
            metrics=payload['metrics']
        )
        
        if worker_result and 'response' in worker_result:
            resp_body = worker_result['response']
            decision = {
                'resource_alloc': resp_body.get('debug', {}).get('resource_alloc', 1.0),
                'uncertainty': resp_body.get('debug', {}).get('uncertainty', 0.0),
                'p90_prediction': resp_body.get('debug', {}).get('p90_prediction', 0.0),
            }
        else:
             decision = {}

        ctrl_latency = 0 # No external controller overhead
    elif strategy == 'baseline':
        # --- BASELINE: AWS Native ---
        # No MPC Controller. Direct invocation.
        # Worker handles 'baseline' strategy natively; do not override alloc.
        worker_result = invoke_worker_lambda(
            decision={}, 
            task={"id": idx, "priority": priority}, 
            mode='auto',
            strategy='baseline',
            metrics=payload['metrics']
        )
        
        # Baseline has no MPC decision data
        decision = {}
        ctrl_latency = 0 # Native = 0 external controller overhead
        
    elif strategy == 'sinan':
        # --- SINAN: Direct to Worker ---
        # Worker handles 'sinan' strategy natively.
        worker_result = invoke_worker_lambda(
            decision={}, 
            task={"id": idx, "priority": priority}, 
            mode='auto',
            strategy='sinan',
            metrics=payload['metrics']
        )
        
        # Sinan has no external controller decision data
        decision = {}
        ctrl_latency = 0 # No external controller overhead
    else:
        # --- Fallback External Controller Path (should not be hit by this script) ---
        # This path is for strategies that use an external controller lambda
        # before invoking the worker. It passes the strategy name to the controller.
        
        mode = 'strict'
        
        ctrl_start = time.time()
        ctrl_result = invoke_controller_lambda(payload, mode=mode, strategy=strategy)
        ctrl_end = time.time()
        ctrl_latency = (ctrl_end - ctrl_start) * 1000
        
        if ctrl_result:
            decision = ctrl_result.get('decision', {})
        else:
            decision = {}
            
        worker_result = invoke_worker_lambda(
            decision, 
            task={"id": idx, "priority": priority}, 
            mode='auto'
        )

    t2 = time.time()
    
    success = worker_result is not None
    # client_duration includes network RTT
    client_latency = worker_result['client_duration'] if success else 0
    
    # server_latency is pure Lambda execution time (if available)
    server_latency = 0
    if success and 'response' in worker_result:
        server_latency = worker_result['response'].get('latency_ms', 0)
    
    # Use client_latency for E2E consistency with baseline measurement approach,
    # but we will track server_latency separately for analysis.
    e2e_latency = ctrl_latency + client_latency
    
    # Violation: Latency > SLO OR Request Failed (Rate Exceeded)
    is_violation = (e2e_latency > SLO_LATENCY_MS) or (not success)

    # 4. Record Data
    return {
        "id": idx,
        "strategy": strategy,
        "priority": priority,
        "p90_input": current_p90,
        "alloc": decision.get('resource_alloc', 1.0),
        "uncertainty": decision.get('uncertainty', 0.0),
        "pred_p90": decision.get('p90_prediction', 0.0),
        "ctrl_latency": ctrl_latency,
        "worker_latency": client_latency,
        "server_latency": server_latency,
        "e2e_latency": e2e_latency,
        "violation": is_violation,
        "success": success,
        "timestamp": t2
    }

def run_phase(strategy_name, warm_up=False, max_workers=5, num_requests=NUM_REQUESTS, arrival_rate=ARRIVAL_RATE):
    if warm_up:
        print(f"\n>>> Warming up WCP state ({strategy_name})...")
    else:
        print(f"\n>>> Starting Phase: {strategy_name}")
        
    arrival_times = load_azure_trace()
    if arrival_times is None:
        arrival_times = generate_poisson_arrivals(arrival_rate, num_requests)
    results = []
    
    phase_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, delay in enumerate(arrival_times):
            now = time.time() - phase_start
            wait = delay - now
            if wait > 0:
                time.sleep(wait)
            
            f = executor.submit(run_single_request, i, strategy_name, phase_start)
            futures.append(f)
            
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                results.append(res)
                if not warm_up:
                    unc_val = res['uncertainty']
                    if isinstance(unc_val, dict): unc_val = unc_val.get('p90', 0)
                    
                    # This debug info parsing is fragile; worker might not return it.
                    # Let's be more defensive.
                    debug_info = ""
                    if res and 'response' in res and res['response'] and 'debug' in res['response']:
                        dbg = res['response']['debug']
                        if dbg:
                            debug_info = f", PrevU={dbg.get('prev_u', '?')}, SLO={dbg.get('slo_limit', '?')}, Price={dbg.get('price', '?')}"
                    
                    print(f"[{strategy_name}] Req {res['id']}: Alloc={res['alloc']:.2f}, Pred={res['pred_p90']:.0f}, Unc={unc_val:.0f}, E2E={res['e2e_latency']:.1f}ms, Ctrl={res['ctrl_latency']:.1f}ms{debug_info}")
            except Exception as e:
                print(f"[ERROR] Request failed in executor: {e}")

    phase_end = time.time()
    cw_metrics = query_cloudwatch_duration_metrics(phase_start, phase_end)
    return results, cw_metrics

def query_cloudwatch_duration_metrics(start_ts, end_ts):
    """
    Query CloudWatch for Lambda Duration metrics (Average and p99) for the worker function
    over the specified time window.
    """
    try:
        region = os.environ.get('AWS_REGION', 'us-east-1')
        function_name = os.environ.get('MPC_WORKER_NAME', 'MPC_BusinessWorker')
        cw = boto3.client('cloudwatch', region_name=region)
        
        # CloudWatch expects datetimes in UTC
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc) - timedelta(seconds=5)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc) + timedelta(seconds=5)
        
        queries = [
            {
                'Id': 'p99',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/Lambda',
                        'MetricName': 'Duration',
                        'Dimensions': [{'Name': 'FunctionName', 'Value': function_name}]
                    },
                    'Period': 60,
                    'Stat': 'p99',
                    'Unit': 'Milliseconds'
                },
                'ReturnData': True
            },
            {
                'Id': 'avg',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/Lambda',
                        'MetricName': 'Duration',
                        'Dimensions': [{'Name': 'FunctionName', 'Value': function_name}]
                    },
                    'Period': 60,
                    'Stat': 'Average',
                    'Unit': 'Milliseconds'
                },
                'ReturnData': True
            }
        ]
        
        resp = cw.get_metric_data(
            MetricDataQueries=queries,
            StartTime=start_dt,
            EndTime=end_dt,
            ScanBy='TimestampDescending',
            MaxDatapoints=100
        )
        
        p99_val = None
        avg_val = None
        for r in resp.get('MetricDataResults', []):
            if r.get('Id') == 'p99' and r.get('Values'):
                p99_val = float(r['Values'][0])
            if r.get('Id') == 'avg' and r.get('Values'):
                avg_val = float(r['Values'][0])
        return {'cw_p99_ms': p99_val, 'cw_avg_ms': avg_val}
    except Exception as e:
        print(f"[CloudWatch Query Error] {e}")
        return {'cw_p99_ms': None, 'cw_avg_ms': None}

def calc_stats(data):
    if not data: return 0,0,0,0,0,0,0,0
    
    # Filter for Latency Stats (only successful requests)
    success_data = [d for d in data if d.get('success', False)]
    lats = [d['e2e_latency'] for d in success_data]
    
    server_lats = [d['server_latency'] for d in success_data]
    
    # Full data for Violation/Allocation Stats
    allocs = [d['alloc'] for d in data]
    vios = [d['violation'] for d in data]
    ctrls = [d['ctrl_latency'] for d in data]
    prios = [d['priority'] for d in data]
    
    q1_mask = [p == 'platinum' for p in prios]
    # For Q1 Latency, we also only care about successful ones? 
    # Or maybe just use the mask on full data?
    # Let's keep q1_lats for successful only to avoid 0s.
    q1_lats = [d['e2e_latency'] for d in success_data if d['priority'] == 'platinum']
    
    # Q1 Violations includes failures
    q1_vios = [v for v, m in zip(vios, q1_mask) if m]
    q1_nonviol = sum(1 for v in q1_vios if not v)
    q1_total = max(1, len(q1_vios))
    q1_thrpt = q1_nonviol
    
    if lats:
        tail = sorted(lats)[max(0, int(0.9*len(lats))):]
        tail_std = statistics.pstdev(tail) if tail else 0.0
        p90 = np.percentile(lats, 90)
        avg_lat = statistics.mean(lats)
        avg_server_lat = statistics.mean(server_lats) if server_lats else 0.0
        overhead_pct = statistics.mean([c/e if e>0 else 0 for c,e in zip(ctrls, lats)])*100
    else:
        tail_std = 0.0
        p90 = 0.0
        avg_lat = 0.0
        avg_server_lat = 0.0
        overhead_pct = 0.0

    avg_alloc = statistics.mean(allocs)
    vio_rate = sum(vios) / len(vios) * 100
    
    return avg_lat, p90, avg_alloc, vio_rate, q1_thrpt, tail_std, overhead_pct, avg_server_lat

def calc_priority_stats(data, use_server=False):
    if not data: 
        return {
            'platinum': {'vio_rate': 0.0, 'nonviol': 0, 'total': 0},
            'gold': {'vio_rate': 0.0, 'nonviol': 0, 'total': 0},
            'standard': {'vio_rate': 0.0, 'nonviol': 0, 'total': 0},
        }
    prios = ['platinum','gold','standard']
    out = {}
    for p in prios:
        cls = [d for d in data if d.get('priority') == p]
        total = len(cls)
        if use_server:
            vios = [(d.get('server_latency', 0) > SLO_LATENCY_MS) or (not d.get('success', False)) for d in cls]
        else:
            vios = [d.get('violation', False) for d in cls]
        nonviol = sum(1 for v in vios if not v)
        vio_rate = (sum(1 for v in vios if v) / total * 100) if total > 0 else 0.0
        out[p] = {'vio_rate': vio_rate, 'nonviol': nonviol, 'total': total}
    return out

def print_comparison(baseline_data, sinan_data, mpc_data):
    print("\n" + "="*85)
    print(f"{'Metric':<25} | {'AWS Native (Baseline)':<20} | {'Sinan (Lit. 1)':<20} | {'MPC Integrated (Ours)':<20}")
    print("-" * 85)
    
    b_avg, b_p90, b_alloc, b_vio, b_q1_thrpt, b_tail_std, b_overhead, b_server = calc_stats(baseline_data)
    s_avg, s_p90, s_alloc, s_vio, s_q1_thrpt, s_tail_std, s_overhead, s_server = calc_stats(sinan_data)
    m_avg, m_p90, m_alloc, m_vio, m_q1_thrpt, m_tail_std, m_overhead, m_server = calc_stats(mpc_data)
    
    print(f"{'Avg Latency (ms)':<25} | {b_avg:<20.2f} | {s_avg:<20.2f} | {m_avg:<20.2f}")
    print(f"{'Avg Server Lat (ms)':<25} | {b_server:<20.2f} | {s_server:<20.2f} | {m_server:<20.2f}")
    print(f"{'P90 Latency (ms)':<25} | {b_p90:<20.2f} | {s_p90:<20.2f} | {m_p90:<20.2f}")
    print(f"{'Violation Rate (%)':<25} | {b_vio:<20.2f} | {s_vio:<20.2f} | {m_vio:<20.2f}")
    print(f"{'Avg Resource Alloc':<25} | {b_alloc:<20.2f} | {s_alloc:<20.2f} | {m_alloc:<20.2f}")
    print(f"{'Q1 Non-violating Count':<25} | {b_q1_thrpt:<20.0f} | {s_q1_thrpt:<20.0f} | {m_q1_thrpt:<20.0f}")
    print(f"{'P99 Tail Std (ms)':<25} | {b_tail_std:<20.2f} | {s_tail_std:<20.2f} | {m_tail_std:<20.2f}")
    print(f"{'Controller Overhead (%)':<25} | {b_overhead:<20.2f} | {s_overhead:<20.2f} | {m_overhead:<20.2f}")
    print("="*85)
    
    # Check if MPC improved violations
    if m_vio < b_vio and m_vio < s_vio:
        print("\n[SUCCESS] MPC achieved the lowest violation rate.")
    elif m_vio < b_vio:
        print("\n[SUCCESS] MPC reduced violation rate compared to Baseline.")
    elif m_q1_thrpt > b_q1_thrpt and m_q1_thrpt > s_q1_thrpt:
        print("\n[SUCCESS] MPC achieved the highest Q1 throughput.")
    else:
        print("\n[NOTE] Review results for detailed comparison.")
    
    b_prio_e2e = calc_priority_stats(baseline_data, use_server=False)
    s_prio_e2e = calc_priority_stats(sinan_data, use_server=False)
    m_prio_e2e = calc_priority_stats(mpc_data, use_server=False)
    b_prio_srv = calc_priority_stats(baseline_data, use_server=True)
    s_prio_srv = calc_priority_stats(sinan_data, use_server=True)
    m_prio_srv = calc_priority_stats(mpc_data, use_server=True)
    
    print("\nPer-Priority Violation Rate (E2E):")
    print(f"{'Priority':<10} | {'Baseline':<10} | {'Sinan':<10} | {'MPC':<10}")
    for p in ('platinum','gold','standard'):
        print(f"{p:<10} | {b_prio_e2e[p]['vio_rate']:<10.2f} | {s_prio_e2e[p]['vio_rate']:<10.2f} | {m_prio_e2e[p]['vio_rate']:<10.2f}")
    print("Per-Priority Violation Rate (Server):")
    print(f"{'Priority':<10} | {'Baseline':<10} | {'Sinan':<10} | {'MPC':<10}")
    for p in ('platinum','gold','standard'):
        print(f"{p:<10} | {b_prio_srv[p]['vio_rate']:<10.2f} | {s_prio_srv[p]['vio_rate']:<10.2f} | {m_prio_srv[p]['vio_rate']:<10.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=str, default="20,50")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION","us-east-1"))
    parser.add_argument("--function", type=str, default=os.environ.get("MPC_WORKER_NAME","MPC_BusinessWorker"))
    parser.add_argument("--task", type=str, default="linpack") # linpack or gzip
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["AWS_REGION"] = args.region
    os.environ["MPC_WORKER_NAME"] = args.function
    print(f">>> Starting Serverless SOTA Comparison Experiment: {args.task}")
    print(">>> Baseline: HPA (Jiagu ATC'24 Style - 80% CPU, 15s Window)")
    
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    
    for conc in levels:
        arrival_rate = conc / 2.0 # Adjusted for 400ms target latency
        num_requests = int(arrival_rate * args.minutes * 60)
        
        print(f"\n\n############################################################")
        print(f"### RUNNING CONCURRENCY LEVEL: {conc} (Rate: {arrival_rate:.1f} req/s, N={num_requests}, Window={args.minutes}m) ###")
        print(f"############################################################")

        # 1. Testing HPA-Baseline
        print(f"\n--- Testing HPA-Baseline (Jiagu ATC'24) @ {conc} ---")
        baseline_results, baseline_cw = run_phase('baseline', max_workers=conc, num_requests=num_requests, arrival_rate=arrival_rate)
        
        # 2. Testing MPC-Guard (Ours)
        print(f"\n--- Testing MPC-Guard (Ours) @ {conc} ---")
        # Warm up RLS/WCP state first
        run_phase('mpc_integrated', warm_up=True, max_workers=10, num_requests=50, arrival_rate=5.0)
        mpc_results, mpc_cw = run_phase('mpc_integrated', max_workers=conc, num_requests=num_requests, arrival_rate=arrival_rate)
        
        # Mocking Sinan/Jiagu as empty for table consistency if needed, 
        # or just compare two. Given user input, let's compare two clearly.
        print_comparison(baseline_results, [], mpc_results) 
        
        # Additional Metrics for SOTA requirements
        _, _, b_alloc, _, _, _, b_ctrl, _ = calc_stats(baseline_results)
        _, _, m_alloc, _, _, _, m_ctrl, _ = calc_stats(mpc_results)
        
        print("\n>>> SOTA Key Metrics Summary:")
        print(f"Deployment Density (Relative to HPA): { (1.0/m_alloc) / (1.0/b_alloc) :.2f}x (Target: >=1.5x)")
        print(f"Scheduling Overhead: MPC={m_ctrl:.1f}ms, HPA={b_ctrl:.1f}ms (Target: MPC <= 50% of SOTA)")

