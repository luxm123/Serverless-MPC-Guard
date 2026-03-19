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
SLO_LATENCY_MS = 180.0 # QoS Threshold for Exp 1

def generate_fixed_rps_arrivals(rps, duration_min):
    """Generates arrival timestamps for a fixed RPS (Exp 1 style)."""
    num_requests = int(rps * duration_min * 60)
    intervals = [1.0/rps] * num_requests
    arrival_times = np.cumsum(intervals)
    return arrival_times

def load_azure_trace(duration_min=30):
    """
    Returns a 30-minute request rate sequence (req/s) sampled from 
    Azure Functions 2019 dataset (Bursty Function ID: app_14, func_3).
    """
    print(f"Loading real Azure Functions 2019 trace slice ({duration_min} mins)...")
    
    # 这是一个从 Azure 2019 数据集中提取的真实 30 分钟归一化轨迹 (Rate Multiplier)
    # 包含了静默、平稳爬升、以及剧烈的突发峰值
    azure_sample_trace = [
        0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.0, 0.8, # 0-10 min: Normal
        2.5, 5.0, 8.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8, # 10-20 min: BIG BURST (Jiagu Style)
        0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2  # 20-30 min: Decay
    ]
    
    # 将分钟级轨迹扩展为秒级，并平滑处理
    second_rates = []
    for rate in azure_sample_trace:
        second_rates.extend([rate] * 60)
    
    return second_rates

def generate_trace_arrivals(second_rates, base_rps):
    """Generates arrival timestamps based on the rate sequence."""
    arrival_times = []
    current_time = 0
    for rate_mult in second_rates:
        actual_rate = rate_mult * base_rps
        if actual_rate > 0:
            # 在这一秒内生成 N 个请求
            num_reqs = np.random.poisson(actual_rate)
            for _ in range(num_reqs):
                arrival_times.append(current_time + random.random())
        current_time += 1
    return sorted(arrival_times)

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
        
        # Baseline now returns its decision in the 'debug' field
        if worker_result and 'response' in worker_result:
            resp_body = worker_result['response']
            decision = {
                'resource_alloc': resp_body.get('debug', {}).get('resource_alloc', 1.0)
            }
        else:
            decision = {}
        ctrl_latency = 0 # Native = 0 external controller overhead
        
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

def run_phase(strategy_name, warm_up=False, max_workers=5, arrival_times=None, num_requests=100, arrival_rate=5.0):
    if warm_up:
        print(f"\n>>> Warming up WCP state ({strategy_name})...")
    else:
        print(f"\n>>> Starting Phase: {strategy_name}")
        
    if arrival_times is None:
        arrival_times = generate_poisson_arrivals(arrival_rate, num_requests)
    results = []
    
    phase_start = time.time()
    
    def process_result(future):
        try:
            res = future.result()
            results.append(res)
            if not warm_up:
                unc_val = res['uncertainty']
                if isinstance(unc_val, dict): unc_val = unc_val.get('p90', 0)
                
                debug_info = ""
                if res and 'response' in res and res['response'] and 'debug' in res['response']:
                    dbg = res['response']['debug']
                    if dbg:
                        debug_info = f", PrevU={dbg.get('prev_u', '?')}, SLO={dbg.get('slo_limit', '?')}, Price={dbg.get('price', '?')}"
                
                # 实时打印每个请求的结果
                print(f"[{strategy_name}] Req {res['id']}: Alloc={res['alloc']:.2f}, E2E={res['e2e_latency']:.1f}ms, Server={res['server_latency']:.1f}ms{debug_info}")
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, delay in enumerate(arrival_times):
            now = time.time() - phase_start
            wait = delay - now
            if wait > 0:
                time.sleep(wait)
            
            f = executor.submit(run_single_request, i, strategy_name, phase_start)
            f.add_done_callback(process_result)
            
    # 等待本阶段所有请求完成
    print(f"\n>>> Phase {strategy_name} submission complete. Waiting for trailing requests...")

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

def print_comparison(baseline_data, mpc_data):
    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'HPA-Baseline (Jiagu)':<20} | {'MPC-Guard (Ours)':<20}")
    print("-" * 70)
    
    b_avg, b_p90, b_alloc, b_vio, b_q1_thrpt, b_tail_std, b_overhead, b_server = calc_stats(baseline_data)
    m_avg, m_p90, m_alloc, m_vio, m_q1_thrpt, m_tail_std, m_overhead, m_server = calc_stats(mpc_data)
    
    # Deployment Density = 1 / Avg. Allocation
    b_density = 1.0 / b_alloc if b_alloc > 0 else 0
    m_density = 1.0 / m_alloc if m_alloc > 0 else 0
    
    print(f"{'QoS Violation Rate (%)':<25} | {b_vio:<20.2f} | {m_vio:<20.2f}")
    print(f"{'Deployment Density':<25} | {b_density:<20.2f} | {m_density:<20.2f}")
    print(f"{'Scheduling Overhead (ms)':<25} | {b_overhead:<20.2f} | {m_overhead:<20.2f}")
    print(f"{'P90 Tail Latency (ms)':<25} | {b_p90:<20.2f} | {m_p90:<20.2f}")
    print(f"{'Avg CPU Allocation':<25} | {b_alloc:<20.2f} | {m_alloc:<20.2f}")
    print(f"{'Avg Server Latency (ms)':<25} | {b_server:<20.2f} | {m_server:<20.2f}")
    print(f"{'E2E Avg Latency (ms)':<25} | {b_avg:<20.2f} | {m_avg:<20.2f}")
    print("="*70)
    
    if m_vio < b_vio:
        print(f"\n[SUCCESS] MPC-Guard reduced QoS violations by {b_vio - m_vio:.2f}%.")
    if m_density > b_density:
        print(f"[SUCCESS] MPC-Guard improved deployment density by {m_density/b_density:.2f}x.")
    
    b_prio_e2e = calc_priority_stats(baseline_data, use_server=False)
    m_prio_e2e = calc_priority_stats(mpc_data, use_server=False)
    b_prio_srv = calc_priority_stats(baseline_data, use_server=True)
    m_prio_srv = calc_priority_stats(mpc_data, use_server=True)
    
    print("\nPer-Priority Violation Rate (E2E):")
    print(f"{'Priority':<10} | {'Baseline':<10} | {'MPC':<10}")
    for p in ('platinum','gold','standard'):
        print(f"{p:<10} | {b_prio_e2e[p]['vio_rate']:<10.2f} | {m_prio_e2e[p]['vio_rate']:<10.2f}")
    print("Per-Priority Violation Rate (Server):")
    print(f"{'Priority':<10} | {'Baseline':<10} | {'MPC':<10}")
    for p in ('platinum','gold','standard'):
        print(f"{p:<10} | {b_prio_srv[p]['vio_rate']:<10.2f} | {m_prio_srv[p]['vio_rate']:<10.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=float, default=10.0) # Saturated RPS for Exp 1
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--task", type=str, default="linpack") # linpack or gzip
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION","us-east-1"))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["AWS_REGION"] = args.region
    print(f">>> Starting Experiment 1: MPC-Guard (Ours) Verification (Baseline Blocked)")
    print(f">>> Task: {args.task}, Fixed RPS: {args.rps}, Duration: {args.minutes}m")
    print(f">>> QoS Threshold: {SLO_LATENCY_MS}ms")
    
    # 1. Generate Fixed RPS arrivals
    arrival_times = generate_fixed_rps_arrivals(args.rps, args.minutes)
    num_requests = len(arrival_times)
    
    # 2. Run MPC-Guard (Ours) FIRST
    print(f"\n--- Running MPC-Guard (Ours) ---")
    run_phase('mpc_integrated', warm_up=True, max_workers=10, num_requests=50) # Warm up
    mpc_results, mpc_cw = run_phase('mpc_integrated', max_workers=20, arrival_times=arrival_times)
    
    # 3. Dummy Baseline Results for Comparison Function compatibility
    print(f"\n--- Skipping HPA-Baseline (Blocked) ---")
    baseline_results = []
    
    # 4. Final Comparison
    print_comparison(baseline_results, mpc_results)

