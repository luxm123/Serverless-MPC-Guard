import os
import time
from datetime import datetime, timedelta, timezone
import boto3
from src.mpc.controller import MPCController
from src.wcp.wcp_lite_a import wcp_lite_a_update

def _latest_datapoint(vals, key):
    if not vals:
        return 0.0
    vals = sorted(vals, key=lambda x: x.get('Timestamp'))
    v = vals[-1].get(key, 0.0)
    try:
        if isinstance(v, dict):
            return float(v.get('p90', 0.0))
        return float(v)
    except:
        return 0.0

def _cw_get(cw, namespace, metric, dimensions, period, start, end, statistics=None, extended=None):
    params = {
        'Namespace': namespace,
        'MetricName': metric,
        'Dimensions': dimensions,
        'StartTime': start,
        'EndTime': end,
        'Period': period
    }
    if statistics:
        params['Statistics'] = statistics
    if extended:
        params['ExtendedStatistics'] = extended
    r = cw.get_metric_statistics(**params)
    return r.get('Datapoints', [])

def _get_lambda_metrics(cw, fn_name, period, start, end):
    dims = [{'Name': 'FunctionName', 'Value': fn_name}]
    inv = _cw_get(cw, 'AWS/Lambda', 'Invocations', dims, period, start, end, statistics=['Sum'])
    err = _cw_get(cw, 'AWS/Lambda', 'Errors', dims, period, start, end, statistics=['Sum'])
    thr = _cw_get(cw, 'AWS/Lambda', 'Throttles', dims, period, start, end, statistics=['Sum'])
    dur = _cw_get(cw, 'AWS/Lambda', 'Duration', dims, period, start, end, extended=['p90'])
    conc = _cw_get(cw, 'AWS/Lambda', 'ConcurrentExecutions', dims, period, start, end, statistics=['Maximum'])
    inv_s = _latest_datapoint(inv, 'Sum')
    err_s = _latest_datapoint(err, 'Sum')
    thr_s = _latest_datapoint(thr, 'Sum')
    dur_p90 = _latest_datapoint(dur, 'ExtendedStatistics')
    conc_max = _latest_datapoint(conc, 'Maximum')
    return {
        'invocations': inv_s,
        'errors': err_s,
        'throttles': thr_s,
        'p90_ms': dur_p90,
        'concurrency': conc_max
    }

def _get_sqs_backlog(sqs, queue_url):
    if not queue_url:
        return 0.0
    r = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages'])
    v = r.get('Attributes', {}).get('ApproximateNumberOfMessages', '0')
    try:
        return float(v)
    except:
        return 0.0

def run_live(duration_minutes=5, period_seconds=60, print_csv=True):
    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or 'us-east-1'
    fn_name = os.environ.get('LAMBDA_FUNCTION_NAME', '')
    queue_url = os.environ.get('SQS_QUEUE_URL', '')
    slo_latency_ms = float(os.environ.get('SLO_LATENCY_MS', '500'))
    u_target = float(os.environ.get('U_TARGET', '0.8'))
    pc_baseline = float(os.environ.get('PC_BASELINE', '100'))
    backlog_target = float(os.environ.get('BACKLOG_TARGET', '1000'))
    vec_weights_raw = os.environ.get('SP_VEC_WEIGHTS', '')
    price_bands_raw = os.environ.get('PRICE_BANDS', '')
    c = boto3.client('cloudwatch', region_name=region)
    s = boto3.client('sqs', region_name=region)
    ctrl = MPCController()
    system_state = {}
    wcp_state = {}
    if vec_weights_raw:
        try:
            parts = vec_weights_raw.split(',')
            w = {}
            for p in parts:
                k, v = p.split(':')
                w[k.strip()] = float(v.strip())
            system_state['sp_vec_weights'] = w
        except:
            pass
    if price_bands_raw:
        try:
            bands = []
            for seg in price_bands_raw.split(';'):
                lo, hi, e, g = seg.split(',')
                bands.append({'lo': float(lo), 'hi': float(hi), 'eta_mul': float(e), 'gamma_mul': float(g)})
            system_state['price_bands'] = bands
        except:
            pass
    rows = []
    steps = int(max(1, duration_minutes * 60 // period_seconds))
    if print_csv:
        print('step,inv,err,thr,conc,p90_ms,cpu_util,queue,timeout_rate,error_rate,mem_pressure,price,alloc,shed,plan')
    for i in range(steps):
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=period_seconds * 5)
        lm = _get_lambda_metrics(c, fn_name, period_seconds, start, end)
        backlog = _get_sqs_backlog(s, queue_url)
        inv = lm['invocations']
        err = lm['errors']
        p90_ms = lm['p90_ms']
        conc = lm['concurrency']
        error_rate = (err / inv) if inv > 0 else 0.0
        timeout_rate = 0.0
        cpu_util = min(1.0, conc / pc_baseline) if pc_baseline > 0 else 0.0
        mem_pressure = 0.0
        if backlog_target > 0:
            mem_pressure = min(1.0, backlog / backlog_target)
        metrics = {
            'p90': p90_ms,
            'timeout_rate': timeout_rate,
            'error_rate': error_rate,
            'memory_pressure': mem_pressure,
            'cpu_util': cpu_util,
            'queue_backlog': backlog,
            'slo_violation_rate': 1.0 if p90_ms > slo_latency_ms else 0.0,
            'last_alloc': float(system_state.get('last_alloc', 1.0)),
            'u_eta': float(system_state.get('u_eta', 0.05)),
            'u_max_delta': float(system_state.get('u_max_delta', 0.15)),
            'gamma': float(system_state.get('gamma', 0.1))
        }
        # Update WCP (Lite A)
        pred, unc_val, dbg = wcp_lite_a_update(wcp_state, metrics)
        
        # Construct WCP Constraints
        wcp = {
            'pred': pred, 
            'uncertainty': {'p90': unc_val, 'timeout_rate': 0.0, 'error_rate': 0.0, 'memory_pressure': 0.0}
        }
        
        res = ctrl.decide({'priority': 'high', 'consistency': 'strict'}, wcp, metrics)
        system_state.update(metrics)
        system_state['last_alloc'] = res['decision']['resource_alloc']
        row = {
            'step': i,
            'inv': inv,
            'err': err,
            'thr': lm['throttles'],
            'conc': conc,
            'p90_ms': p90_ms,
            'cpu_util': cpu_util,
            'queue': backlog,
            'timeout_rate': timeout_rate,
            'error_rate': error_rate,
            'mem_pressure': mem_pressure,
            'price': float(system_state.get('shadow_price', 0.0)),
            'alloc': float(res['decision']['resource_alloc']),
            'shed': int(res['decision']['should_shed']),
            'plan': res['decision']['degrade_plan'] or ''
        }
        rows.append(row)
        if print_csv:
            print(f"{row['step']},{row['inv']},{row['err']},{row['thr']},{row['conc']},{row['p90_ms']},{row['cpu_util']},{row['queue']},{row['timeout_rate']},{row['error_rate']},{row['mem_pressure']},{row['price']},{row['alloc']},{row['shed']},{row['plan']}")
        time.sleep(period_seconds)
    return rows
