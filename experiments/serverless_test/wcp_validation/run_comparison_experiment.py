import time
import random
import numpy as np
import concurrent.futures
import statistics
import math
import os
import threading
from datetime import datetime, timedelta, timezone
import boto3
from serverless_utils import invoke_controller_lambda, invoke_worker_lambda
import argparse

# Experiment Configuration
SERVER_SLO_MS = 180.0
E2E_SLO_MS = 180.0
CURRENT_TASK = "linpack" # Global to be updated by args
BASE_RPS = 10.0
_E2E_OVERHEAD_EMA = 50.0
_OVERHEAD_LOCK = threading.Lock()
_MPC_MIN_ALLOC_LOCK = threading.Lock()
_MPC_MIN_ALLOC = 0.0
_MAX_ALLOC = 1.0

def _p90(values):
    if not values:
        return 0.0
    arr = np.array([float(v) for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 90))

def _pctl(values, p):
    if not values:
        return 0.0
    try:
        arr = np.array([float(v) for v in values if v is not None], dtype=float)
    except Exception:
        return 0.0
    if arr.size == 0:
        return 0.0
    try:
        return float(np.percentile(arr, float(p)))
    except Exception:
        return 0.0

def calibrate_qos_threshold(task_name, factor=1.2, warmup_requests=15, sample_requests=80):
    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='baseline', reset_state=True)
    server_lats = []
    e2e_lats = []

    metrics = {
        "p90": 0.0,
        "backlog": 1,
        "concurrency": 1.0,
        "cpu_util": 0.5,
        "error_rate": 0.0,
        "rps": 0.0,
        "e2e_overhead_ms": float(_E2E_OVERHEAD_EMA)
    }

    for i in range(max(0, int(warmup_requests))):
        invoke_worker_lambda(
            decision={},
            task={"id": f"cal-warm-{i}", "priority": "standard", "risk": {}, "task_type": task_name},
            mode='auto',
            strategy='baseline',
            task_type=task_name,
            resource_alloc=1.0,
            metrics=metrics
        )

    for i in range(max(1, int(sample_requests))):
        res = invoke_worker_lambda(
            decision={},
            task={"id": f"cal-{i}", "priority": "standard", "risk": {}, "task_type": task_name},
            mode='auto',
            strategy='baseline',
            task_type=task_name,
            resource_alloc=1.0,
            metrics=metrics
        )
        if not res:
            continue
        try:
            e2e_lats.append(float(res.get('client_duration', 0.0)))
        except Exception:
            pass
        try:
            body = res.get('response', {}) or {}
            server_lats.append(float(body.get('latency_ms', 0.0) or 0.0))
        except Exception:
            pass

    base_p90_e2e = _p90([v for v in e2e_lats if v and v > 0.0])
    base_p90_srv = _p90([v for v in server_lats if v and v > 0.0])

    qos_e2e = float(max(1.0, base_p90_e2e * float(factor)))
    qos_srv = float(max(1.0, base_p90_srv * float(factor)))
    return {
        "base_p90_e2e_ms": base_p90_e2e,
        "base_p90_srv_ms": base_p90_srv,
        "qos_e2e_ms": qos_e2e,
        "qos_srv_ms": qos_srv
    }

def get_lambda_account_concurrency(region):
    try:
        lam = boto3.client('lambda', region_name=region)
        resp = lam.get_account_settings()
        limits = resp.get('AccountLimit', {}) or {}
        usage = resp.get('AccountUsage', {}) or {}
        return {
            "concurrent_limit": float(limits.get('ConcurrentExecutions', 0.0) or 0.0),
            "unreserved_concurrent": float(limits.get('UnreservedConcurrentExecutions', 0.0) or 0.0),
            "concurrent_in_use": float(usage.get('ConcurrentExecutions', 0.0) or 0.0),
        }
    except Exception:
        return None

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

def run_single_request(idx, strategy, start_time, inflight=1):
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
        "backlog": int(inflight),
        "concurrency": float(inflight),
        "cpu_util": 0.5,
        "error_rate": 0.0,
        "rps": float(BASE_RPS),
        "e2e_overhead_ms": float(_E2E_OVERHEAD_EMA),
        "slo_limit": float(SERVER_SLO_MS),
        "max_alloc": float(_MAX_ALLOC)
    }
    if strategy == 'mpc_integrated':
        with _MPC_MIN_ALLOC_LOCK:
            metrics["min_alloc"] = float(_MPC_MIN_ALLOC)
    
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
            task={"id": idx, "priority": priority, "risk": payload['risk'], "task_type": task_name},
            mode='auto',
            strategy='mpc_integrated',
            task_type=task_name,
            reset_state=(idx == 0),
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
    elif strategy in ['baseline', 'aws_tt'] or str(strategy).startswith('static'):
        worker_result = invoke_worker_lambda(
            decision={},
            task={"id": idx, "priority": priority, "risk": payload['risk'], "task_type": task_name},
            mode='auto',
            strategy=strategy,
            task_type=task_name,
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
            task={"id": idx, "priority": priority, "risk": payload['risk'], "task_type": task_name},
            mode='auto',
            strategy=strategy,
            task_type=task_name,
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

def run_phase(strategy_name, warm_up=False, max_workers=5, arrival_times=None, num_requests=100, arrival_rate=5.0, max_inflight=0):
    if warm_up:
        print(f"\n>>> Warming up WCP state ({strategy_name})...")
    else:
        print(f"\n>>> Starting Phase: {strategy_name}")
        
    if arrival_times is None:
        arrival_times = generate_poisson_arrivals(arrival_rate, num_requests)
    results = []
    
    phase_start = time.time()
    inflight_lock = threading.Lock()
    inflight_count = 0
    inflight_sem = threading.Semaphore(int(max_inflight)) if int(max_inflight) > 0 else None
    
    def process_result(future):
        nonlocal inflight_count
        with inflight_lock:
            inflight_count = max(0, inflight_count - 1)
        if inflight_sem is not None:
            inflight_sem.release()
        try:
            res = future.result()
            results.append(res)
            if res.get('success') and ('server_latency' in res) and ('e2e_latency' in res):
                overhead = float(res.get('e2e_latency', 0.0)) - float(res.get('server_latency', 0.0))
                if overhead > 0.0 and overhead < 200.0:
                    overhead = max(20.0, min(90.0, overhead))
                    global _E2E_OVERHEAD_EMA
                    with _OVERHEAD_LOCK:
                        _E2E_OVERHEAD_EMA = 0.95 * float(_E2E_OVERHEAD_EMA) + 0.05 * overhead
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
            
            if inflight_sem is not None:
                inflight_sem.acquire()
            with inflight_lock:
                inflight_count += 1
                inflight_snapshot = inflight_count
            f = executor.submit(run_single_request, i, strategy_name, phase_start, inflight_snapshot)
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
    print(f"{'Metric':<25} | {'HPA Baseline':<20} | {'MPC-Guard (Ours)':<20}")
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
        
        success = [r for r in results if r.get('success', False)]
        denom = max(1, len(success))
        avg_alloc = sum(r.get('alloc', 1.0) for r in results) / total
        avg_server_lat = sum(r.get('server_latency', 0) for r in success) / denom
        avg_e2e_lat = sum(r.get('e2e_latency', 0) for r in success) / denom
        avg_overhead = sum(r.get('scheduling_overhead_ms', 0.0) for r in results) / total
        
        latencies = sorted([r.get('e2e_latency', 0) for r in success])
        p90 = latencies[int(len(latencies) * 0.9)] if latencies else 0
        
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

def _calc_metrics(results):
    if not results:
        return {
            "e2e_vio": 0.0,
            "srv_vio": 0.0,
            "density": 0.0,
            "p90_e2e": 0.0,
            "avg_alloc": 0.0,
            "avg_srv": 0.0,
            "avg_e2e": 0.0,
            "avg_overhead": 0.0,
            "achieved_rps": 0.0,
            "achieved_success_rps": 0.0,
            "util_pct": 0.0,
            "cpu_ms_per_success": 0.0,
            "alloc_p50": 0.0,
            "alloc_p90": 0.0,
            "alloc_std": 0.0,
            "alloc_churn": 0.0,
        }
    total = len(results)
    e2e_violations = sum(1 for r in results if r.get('e2e_latency', 0) > E2E_SLO_MS)
    e2e_viol_rate = (e2e_violations / total) * 100
    server_violations = sum(1 for r in results if r.get('server_latency', 0) > SERVER_SLO_MS)
    server_viol_rate = (server_violations / total) * 100

    success = [r for r in results if r.get('success', False)]
    denom = max(1, len(success))
    ts = [r.get('timestamp') for r in results if r.get('timestamp') is not None]
    if ts:
        duration_s = max(0.001, float(max(ts) - min(ts)))
    else:
        duration_s = 1.0
    achieved_rps = float(total) / duration_s
    achieved_success_rps = float(len(success)) / duration_s
    avg_alloc = sum(r.get('alloc', 1.0) for r in results) / total
    avg_srv = sum(r.get('server_latency', 0) for r in success) / denom
    avg_e2e = sum(r.get('e2e_latency', 0) for r in success) / denom
    avg_overhead = sum(r.get('scheduling_overhead_ms', 0.0) for r in results) / total
    latencies = sorted([r.get('e2e_latency', 0) for r in success])
    p90 = latencies[int(len(latencies) * 0.9)] if latencies else 0.0
    density = 1.0 / (avg_alloc + 0.001)

    allocs = [float(r.get('alloc', 1.0) or 1.0) for r in results]
    alloc_p50 = _pctl(allocs, 50)
    alloc_p90 = _pctl(allocs, 90)
    try:
        alloc_std = float(np.std(np.array(allocs, dtype=float))) if allocs else 0.0
    except Exception:
        alloc_std = 0.0

    churn_vals = []
    for r in results:
        try:
            prev_a = float(r.get('prev_alloc'))
            new_a = float(r.get('new_alloc'))
            if math.isfinite(prev_a) and math.isfinite(new_a):
                churn_vals.append(abs(new_a - prev_a))
        except Exception:
            continue
    alloc_churn = float(sum(churn_vals) / max(1, len(churn_vals))) if churn_vals else 0.0

    util_pct = 0.0
    if avg_alloc > 0.0 and SERVER_SLO_MS > 0.0:
        util_pct = float((avg_srv / (avg_alloc * SERVER_SLO_MS)) * 100.0)
        util_pct = float(max(0.0, min(100.0, util_pct)))

    cpu_ms = 0.0
    for r in success:
        try:
            a = float(r.get('alloc', 1.0) or 1.0)
            s = float(r.get('server_latency', 0.0) or 0.0)
            if math.isfinite(a) and math.isfinite(s) and a > 0.0 and s > 0.0:
                cpu_ms += a * s
        except Exception:
            continue
    cpu_ms_per_success = float(cpu_ms / max(1, len(success)))

    return {
        "e2e_vio": e2e_viol_rate,
        "srv_vio": server_viol_rate,
        "density": density,
        "p90_e2e": p90,
        "avg_alloc": avg_alloc,
        "avg_srv": avg_srv,
        "avg_e2e": avg_e2e,
        "avg_overhead": avg_overhead,
        "achieved_rps": achieved_rps,
        "achieved_success_rps": achieved_success_rps,
        "util_pct": util_pct,
        "cpu_ms_per_success": cpu_ms_per_success,
        "alloc_p50": alloc_p50,
        "alloc_p90": alloc_p90,
        "alloc_std": alloc_std,
        "alloc_churn": alloc_churn,
    }

def print_summary(results_by_name):
    print("\n======================================================================")
    print(f"{'Strategy':<22} | {'E2E Viol %':<10} | {'Srv Viol %':<10} | {'AvgU':<6} | {'Dens':<6} | {'P90 E2E':<10} | {'AvgSrv':<8} | {'AvgE2E':<8} | {'Overhead':<8} | {'AchRPS':<7}")
    print("----------------------------------------------------------------------")
    for name, results in results_by_name.items():
        m = _calc_metrics(results)
        print(f"{name:<22} | {m['e2e_vio']:<10.2f} | {m['srv_vio']:<10.2f} | {m['avg_alloc']:<6.2f} | {m['density']:<6.2f} | {m['p90_e2e']:<10.2f} | {m['avg_srv']:<8.2f} | {m['avg_e2e']:<8.2f} | {m['avg_overhead']:<8.2f} | {m['achieved_success_rps']:<7.2f}")
    print("======================================================================\n")

def print_efficiency_summary(results_by_name):
    print("\n==================== EFFICIENCY / STABILITY SUMMARY ====================")
    print(f"{'Strategy':<22} | {'Util%':<6} | {'CPU-ms/succ':<11} | {'AllocP50':<8} | {'AllocP90':<8} | {'AllocStd':<8} | {'Churn':<7}")
    print("-------------------------------------------------------------------------")
    for name, results in results_by_name.items():
        m = _calc_metrics(results)
        print(
            f"{name:<22} | {m['util_pct']:<6.1f} | {m['cpu_ms_per_success']:<11.1f} | "
            f"{m['alloc_p50']:<8.2f} | {m['alloc_p90']:<8.2f} | {m['alloc_std']:<8.3f} | {m['alloc_churn']:<7.3f}"
        )
    print("=========================================================================\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=float, default=10.0) # Saturated RPS for Exp 1
    parser.add_argument("--rps_list", type=str, default="")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--task", type=str, default="linpack") # linpack or gzip
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION","us-east-1"))
    parser.add_argument("--workers", type=int, default=200)
    parser.add_argument("--max_inflight", type=int, default=0)
    parser.add_argument("--baselines", type=str, default="hpa,aws_tt,static")
    parser.add_argument("--static_allocs", type=str, default="0.6,0.8,1.0")
    parser.add_argument("--mode", type=str, default="fixed")
    parser.add_argument("--pareto_min_allocs", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--include_baselines_in_pareto", type=int, default=1)
    parser.add_argument("--qos_factor", type=float, default=1.2)
    parser.add_argument("--server_slo_ms", type=float, default=0.0)
    parser.add_argument("--e2e_slo_ms", type=float, default=0.0)
    parser.add_argument("--print_efficiency", type=int, default=1)
    parser.add_argument("--max_alloc", type=float, default=1.0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    BASE_RPS = float(args.rps)
    os.environ["AWS_REGION"] = args.region
    CURRENT_TASK = args.task
    _MAX_ALLOC = float(args.max_alloc)
    if _MAX_ALLOC <= 0.0:
        _MAX_ALLOC = 1.0
    _MAX_ALLOC = float(max(0.4, min(4.0, _MAX_ALLOC)))
    print(f">>> Starting Experiment 1: MPC-Guard (Ours) vs Baselines")
    print(f">>> Task: {args.task}, Base RPS: {args.rps}, Duration: {args.minutes}m")
    print(f">>> Mode: Real Azure Bursty Trace (Jiagu-Style stress test)")
    print(f">>> Max Alloc: {_MAX_ALLOC:.2f}")

    fixed_srv = float(args.server_slo_ms or 0.0)
    fixed_e2e = float(args.e2e_slo_ms or 0.0)
    if fixed_srv > 0.0 or fixed_e2e > 0.0:
        if fixed_srv <= 0.0:
            fixed_srv = fixed_e2e
        if fixed_e2e <= 0.0:
            fixed_e2e = fixed_srv
        SERVER_SLO_MS = float(fixed_srv)
        E2E_SLO_MS = float(fixed_e2e)
        print(f">>> QoS Thresholds (fixed): Server={SERVER_SLO_MS:.1f}ms, E2E={E2E_SLO_MS:.1f}ms")
    else:
        cal = calibrate_qos_threshold(args.task, factor=float(args.qos_factor), warmup_requests=15, sample_requests=80)
        SERVER_SLO_MS = float(cal["qos_srv_ms"])
        E2E_SLO_MS = float(cal["qos_e2e_ms"])
        print(f">>> QoS Thresholds (auto): Server={SERVER_SLO_MS:.1f}ms, E2E={E2E_SLO_MS:.1f}ms (BaseP90: Srv={cal['base_p90_srv_ms']:.1f}ms, E2E={cal['base_p90_e2e_ms']:.1f}ms)")
    
    baselines = [x.strip() for x in str(args.baselines).split(',') if x.strip()]
    static_allocs = []
    for seg in str(args.static_allocs).split(','):
        seg = seg.strip()
        if not seg:
            continue
        try:
            static_allocs.append(float(seg))
        except Exception:
            pass
    pareto_mins = []
    for seg in str(args.pareto_min_allocs).split(','):
        seg = seg.strip()
        if not seg:
            continue
        try:
            pareto_mins.append(float(seg))
        except Exception:
            pass

    rps_list = []
    for seg in str(args.rps_list).split(','):
        seg = seg.strip()
        if not seg:
            continue
        try:
            rps_list.append(float(seg))
        except Exception:
            pass
    if not rps_list:
        rps_list = [float(args.rps)]

    acct = get_lambda_account_concurrency(args.region)
    if acct is not None:
        print(f">>> Lambda Account Concurrency: limit={int(acct['concurrent_limit'])}, unreserved={int(acct['unreserved_concurrent'])}, in_use={int(acct['concurrent_in_use'])}")

    max_inflight = int(args.max_inflight)
    if max_inflight <= 0 and acct is not None and acct["concurrent_limit"] > 0:
        max_inflight = max(1, int(acct["concurrent_limit"]) - 1)
    if max_inflight > 0:
        print(f">>> Client inflight cap: {max_inflight} (prevents Lambda throttling dominating E2E)")

    sweep_rows = []
    for rps in rps_list:
        BASE_RPS = float(rps)
        print(f"\n>>> Running RPS={rps:.2f} <<<")

        second_rates = load_azure_trace(duration_min=int(args.minutes))
        arrival_times = generate_trace_arrivals(second_rates, base_rps=rps)
        num_requests = len(arrival_times)
        print(f"Generated {num_requests} requests with dynamic bursts.")

        results_by_name = {}

        if str(args.mode).lower() == "pareto":
            for mi in pareto_mins:
                with _MPC_MIN_ALLOC_LOCK:
                    _MPC_MIN_ALLOC = float(mi)
                name = f"mpc(min={mi:.2f})"
                print(f"\n--- Running {name} ---")
                invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='mpc_integrated', reset_state=True)
                run_phase('mpc_integrated', warm_up=True, max_workers=10, num_requests=50, max_inflight=max_inflight)
                res, _ = run_phase('mpc_integrated', max_workers=args.workers, arrival_times=arrival_times, max_inflight=max_inflight)
                results_by_name[name] = res
        else:
            with _MPC_MIN_ALLOC_LOCK:
                _MPC_MIN_ALLOC = 0.0
            print(f"\n--- Running MPC-Guard (Ours) ---")
            invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='mpc_integrated', reset_state=True)
            run_phase('mpc_integrated', warm_up=True, max_workers=10, num_requests=50, max_inflight=max_inflight)
            mpc_results, _ = run_phase('mpc_integrated', max_workers=args.workers, arrival_times=arrival_times, max_inflight=max_inflight)
            results_by_name["mpc_integrated"] = mpc_results

        if str(args.mode).lower() != "pareto" or int(args.include_baselines_in_pareto) == 1:
            for b in baselines:
                if b == "hpa":
                    print(f"\n--- Running HPA Baseline ---")
                    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='baseline', reset_state=True)
                    run_phase('baseline', warm_up=True, max_workers=10, num_requests=50, max_inflight=max_inflight)
                    res, _ = run_phase('baseline', max_workers=args.workers, arrival_times=arrival_times, max_inflight=max_inflight)
                    results_by_name["hpa_baseline"] = res
                elif b == "aws_tt":
                    print(f"\n--- Running AWS Target Tracking ---")
                    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='aws_tt', reset_state=True)
                    run_phase('aws_tt', warm_up=True, max_workers=10, num_requests=50, max_inflight=max_inflight)
                    res, _ = run_phase('aws_tt', max_workers=args.workers, arrival_times=arrival_times, max_inflight=max_inflight)
                    results_by_name["aws_tt"] = res
                elif b == "static":
                    for u in static_allocs:
                        name = f"static_{u:.2f}"
                        print(f"\n--- Running {name} ---")
                        run_phase(f"static_{u}", warm_up=True, max_workers=10, num_requests=50, max_inflight=max_inflight)
                        res, _ = run_phase(f"static_{u}", max_workers=args.workers, arrival_times=arrival_times, max_inflight=max_inflight)
                        results_by_name[name] = res

        print_summary(results_by_name)
        if int(args.print_efficiency) == 1:
            print_efficiency_summary(results_by_name)

        for name, results in results_by_name.items():
            m = _calc_metrics(results)
            sweep_rows.append({
                "rps": float(rps),
                "strategy": name,
                **m
            })

    if len(rps_list) > 1 and str(args.mode).lower() != "pareto":
        print("\n==================== FINAL SUMMARY (ALL RPS) ====================")
        print(f"{'RPS':<6} | {'Strategy':<16} | {'E2E Viol %':<10} | {'Srv Viol %':<10} | {'AvgU':<6} | {'Dens':<6} | {'P90 E2E':<10} | {'AvgSrv':<8} | {'AvgE2E':<8} | {'Overhead':<8} | {'AchRPS':<7}")
        print("-----------------------------------------------------------------")
        for row in sweep_rows:
            print(f"{row['rps']:<6.0f} | {row['strategy']:<16} | {row['e2e_vio']:<10.2f} | {row['srv_vio']:<10.2f} | {row['avg_alloc']:<6.2f} | {row['density']:<6.2f} | {row['p90_e2e']:<10.2f} | {row['avg_srv']:<8.2f} | {row['avg_e2e']:<8.2f} | {row['avg_overhead']:<8.2f} | {row['achieved_success_rps']:<7.2f}")
        print("=================================================================\n")
