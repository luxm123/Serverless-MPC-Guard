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
SERVER_SLO_MS = 180.0
E2E_SLO_MS = 180.0
CURRENT_TASK = "linpack" # Global to be updated by args

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
    
    # For a pure vertical scaling experiment, priority is not a variable.
    # We remove it entirely from the payload to avoid confusion.
    
    # Define missing variables
    task_name = CURRENT_TASK
    req_id = idx
    priority = "standard"
    risk = {}
    metrics = {
        "p90": 0.0,
        "backlog": 0,
        "cpu_util": 0.5,
        "error_rate": 0.0,
        "e2e_overhead_ms": 50.0
    }
    
    # Payload contains only task info. 
    payload = {
        "task": task_name,
        "req_id": req_id,
        "strategy": strategy,
        "timestamp": time.time(),
        "risk": risk,
        "metrics": metrics
    }
    
    # 2. Invoke Controller / Worker (depending on strategy)
    t0 = time.time()
    
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
            debug_data = resp_body.get('debug', {})
            decision = {
                'resource_alloc': debug_data.get('resource_alloc', 1.0),
                'uncertainty': debug_data.get('uncertainty', 0.0),
                'p90_prediction': debug_data.get('p90_prediction', 0.0),
                'p90_belief': debug_data.get('p90_belief', 0.0),
                'version': debug_data.get('version', 'UNKNOWN'),
                'source': debug_data.get('state_source', 'UNKNOWN'),
                'prev_alloc': debug_data.get('prev_alloc', '?'),
                'new_alloc': debug_data.get('new_alloc', '?'),
                'shadow_price': debug_data.get('shadow_price', 0.0),
                'scheduling_overhead_ms': debug_data.get('scheduling_overhead_ms', 0.0)
            }
        else:
             decision = {'version': 'FAILED'}

        ctrl_latency = 0 # No external controller overhead
    elif strategy == 'baseline':
        # --- BASELINE: AWS Native ---
        # No MPC Controller. Direct invocation.
        # Worker handles 'baseline' strategy natively; do not override alloc.
        worker_result = invoke_worker_lambda(
            decision={},
            task={"id": idx, "priority": priority, "risk": payload['risk']},
            mode='auto',
            strategy='baseline',
            metrics=payload['metrics']
        )
        
        if worker_result and 'response' in worker_result:
            resp_body = worker_result['response']
            debug_data = resp_body.get('debug', {})
            decision = {
                'resource_alloc': debug_data.get('resource_alloc', 1.0),
                'version': 'BASELINE',
                'scheduling_overhead_ms': debug_data.get('scheduling_overhead_ms', 0.0)
            }
        else:
            decision = {'resource_alloc': 1.0, 'version': 'BASELINE'}
        ctrl_latency = 0
    else:
        # --- CLASSIC PATH: External Controller ---
        # 1. Invoke Controller
        controller_result = invoke_controller_lambda(payload, mode=strategy, strategy=strategy)
        t1 = time.time()
        ctrl_latency = (t1 - t0) * 1000.0
        
        if controller_result and 'decision' in controller_result:
            decision = controller_result['decision']
        else:
            decision = {'resource_alloc': 1.0, 'version': 'CTRL_FAILED'}
            
        # 2. Invoke Worker with decision
        worker_result = invoke_worker_lambda(
            decision=decision,
            task={"id": idx, "priority": priority, "risk": payload['risk']},
            mode='auto',
            strategy=strategy,
            metrics=payload['metrics']
        )

    # 3. Process Results
    e2e_latency = (time.time() - t0) * 1000.0
    
    if worker_result:
        server_latency = worker_result['response'].get('latency_ms', 0) if 'response' in worker_result else 0
        res = {
            'id': idx,
            'strategy': strategy,
            'priority': priority,
            'e2e_latency': e2e_latency,
            'ctrl_latency': ctrl_latency,
            'worker_latency': worker_result['client_duration'],
            'server_latency': server_latency,
            'scheduling_overhead_ms': decision.get('scheduling_overhead_ms', 0.0),
            'alloc': decision.get('resource_alloc', 1.0),
            'uncertainty': decision.get('uncertainty', 0.0),
            'p90_prediction': decision.get('p90_prediction', 0.0),
            'p90_belief': decision.get('p90_belief', 0.0),
            'version': decision.get('version', 'UNKNOWN'),
            'prev_alloc': decision.get('prev_alloc', '?'),
            'new_alloc': decision.get('new_alloc', '?'),
            'shadow_price': decision.get('shadow_price', 0.0),
            'violation_e2e': (e2e_latency > E2E_SLO_MS),
            'violation_srv': (server_latency > SERVER_SLO_MS),
            'violation': (e2e_latency > E2E_SLO_MS),
            'success': True,
            'timestamp': time.time()
        }
    else:
        res = {
            'id': idx,
            'strategy': strategy,
            'priority': priority,
            'e2e_latency': e2e_latency,
            'violation': True,
            'success': False,
            'timestamp': time.time()
        }
    return res

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
                # 实时打印每个请求的结果 (v29.2 - 修复 Baseline 格式化错误)
                try:
                    prev_a = res.get('prev_alloc', '?')
                    new_a = res.get('new_alloc', '?')
                    
                    # 尝试转换，如果是 '?' 则保持原样
                    prev_str = f"{float(prev_a):.3f}" if isinstance(prev_a, (int, float)) or (isinstance(prev_a, str) and prev_a.replace('.','',1).isdigit()) else str(prev_a)
                    new_str = f"{float(new_a):.3f}" if isinstance(new_a, (int, float)) or (isinstance(new_a, str) and new_a.replace('.','',1).isdigit()) else str(new_a)
                    
                    print(f"[{strategy_name}] Req {res['id']:2d}: Alloc={res['alloc']:.2f}, E2E={res['e2e_latency']:.1f}ms, Srv={res.get('server_latency', 0.0):.1f}ms, Ver={res['version']}, PrevA={prev_str}, NewA={new_str}, P90_B={res.get('p90_belief', 0.0):.1f}, Price={res.get('shadow_price', 0.0):.1f}")
                except Exception as fmt_e:
                    # Fallback print if formatting fails
                    print(f"[{strategy_name}] Req {res['id']:2d}: Alloc={res['alloc']}, E2E={res['e2e_latency']:.1f}ms (Fmt Error: {fmt_e})")
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
            vios = [(d.get('server_latency', 0) > SERVER_SLO_MS) or (not d.get('success', False)) for d in cls]
        else:
            vios = [d.get('violation', False) for d in cls]
        nonviol = sum(1 for v in vios if not v)
        vio_rate = (sum(1 for v in vios if v) / total * 100) if total > 0 else 0.0
        out[p] = {'vio_rate': vio_rate, 'nonviol': nonviol, 'total': total}
    return out

def print_comparison(baseline_results, mpc_results):
    print("\n======================================================================")
    print(f"{'Metric':<25} | {'HPA-Baseline (Jiagu)':<20} | {'MPC-Guard (Ours)':<20}")
    print("----------------------------------------------------------------------")
    
    def calc_metrics(results):
        if not results:
            return 0, 0, 0, 0, 0, 0, 0, 0
        
        total = len(results)
        # E2E Violations
        e2e_violations = sum(1 for r in results if r.get('e2e_latency', 0) > E2E_SLO_MS)
        e2e_viol_rate = (e2e_violations / total) * 100
        
        # Server Violations
        server_violations = sum(1 for r in results if r.get('server_latency', 0) > SERVER_SLO_MS)
        server_viol_rate = (server_violations / total) * 100
        
        avg_alloc = sum(r.get('alloc', 1.0) for r in results) / total
        avg_server_lat = sum(r.get('server_latency', 0) for r in results) / total
        avg_e2e_lat = sum(r.get('e2e_latency', 0) for r in results) / total
        avg_overhead = sum(r.get('scheduling_overhead_ms', 0.0) for r in results) / total
        
        latencies = sorted([r.get('e2e_latency', 0) for r in results])
        p90 = latencies[int(total * 0.9)] if total > 0 else 0
        
        # Deployment Density (Theoretical: 1.0 / avg_alloc)
        # Higher density is better (means we can pack more functions)
        density = 1.0 / (avg_alloc + 0.001)
        
        return e2e_viol_rate, server_viol_rate, density, p90, avg_alloc, avg_server_lat, avg_e2e_lat, avg_overhead

    b_e2e_viol, b_srv_viol, b_dens, b_p90, b_alloc, b_srv_lat, b_e2e_lat, b_overhead = calc_metrics(baseline_results)
    m_e2e_viol, m_srv_viol, m_dens, m_p90, m_alloc, m_srv_lat, m_e2e_lat, m_overhead = calc_metrics(mpc_results)
    
    print(f"{'QoS Violation Rate (E2E) %':<25} | {b_e2e_viol:<20.2f} | {m_e2e_viol:<20.2f}")
    print(f"{'QoS Violation Rate (Srv) %':<25} | {b_srv_viol:<20.2f} | {m_srv_viol:<20.2f}")
    print(f"{'Deployment Density':<25} | {b_dens:<20.2f} | {m_dens:<20.2f}")
    print(f"{'Scheduling Overhead (ms)':<25} | {b_overhead:<20.2f} | {m_overhead:<20.2f}")
    print(f"{'P90 Tail Latency (ms)':<25} | {b_p90:<20.2f} | {m_p90:<20.2f}")
    print(f"{'Avg CPU Allocation':<25} | {b_alloc:<20.2f} | {m_alloc:<20.2f}")
    print(f"{'Avg Server Latency (ms)':<25} | {b_srv_lat:<20.2f} | {m_srv_lat:<20.2f}")
    print(f"{'E2E Avg Latency (ms)':<25} | {b_e2e_lat:<20.2f} | {m_e2e_lat:<20.2f}")
    print("======================================================================\n")
    
    if m_e2e_viol < b_e2e_viol:
        print(f"[SUCCESS] MPC-Guard reduced E2E QoS violations by {b_e2e_viol - m_e2e_viol:.2f}%.")
    else:
        print(f"[WARNING] MPC-Guard did not improve E2E QoS violations.")
        
    if m_srv_viol < b_srv_viol:
        print(f"[SUCCESS] MPC-Guard reduced Server QoS violations by {b_srv_viol - m_srv_viol:.2f}%.")
        
    if m_dens > b_dens:
        print(f"[SUCCESS] MPC-Guard improved deployment density by {m_dens/b_dens:.2f}x.")
    elif m_dens < b_dens:
        print(f"[WARNING] MPC-Guard deployment density is {b_dens/m_dens:.2f}x LOWER than baseline.")

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
    CURRENT_TASK = args.task
    print(f">>> Starting Experiment 1: MPC-Guard (Ours) vs HPA-Baseline (Jiagu)")
    print(f">>> Task: {args.task}, Base RPS: {args.rps}, Duration: {args.minutes}m")
    print(f">>> Mode: Real Azure Bursty Trace (Jiagu-Style stress test)")
    print(f">>> QoS Thresholds: Server={SERVER_SLO_MS}ms, E2E={E2E_SLO_MS}ms")
    
    # 1. Generate Bursty arrivals using real Azure Trace
    # This creates the "real" violations you saw in your previous screenshots
    second_rates = load_azure_trace(duration_min=int(args.minutes))
    arrival_times = generate_trace_arrivals(second_rates, base_rps=args.rps)
    num_requests = len(arrival_times)
    print(f"Generated {num_requests} requests with dynamic bursts.")
    
    # 2. Run MPC-Guard (Ours) FIRST
    print(f"\n--- Running MPC-Guard (Ours) ---")
    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='mpc_integrated', reset_state=True)
    run_phase('mpc_integrated', warm_up=True, max_workers=10, num_requests=50) # Warm up
    # v56: 极度压缩并发（15），彻底消除客户端排队，让“执行耗时”成为 E2E 的唯一变量
    mpc_results, mpc_cw = run_phase('mpc_integrated', max_workers=15, arrival_times=arrival_times)
    
    # 3. Run HPA-Baseline (Jiagu Style)
    # We must reset the state to ensure a fair comparison
    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='baseline', reset_state=True)
    print(f"\n--- Running HPA-Baseline (Jiagu-ATC'24) ---")
    baseline_results, baseline_cw = run_phase('baseline', max_workers=15, arrival_times=arrival_times)
    
    # 4. Final Comparison
    print_comparison(baseline_results, mpc_results)
