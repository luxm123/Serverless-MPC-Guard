import time
import random
import numpy as np
import concurrent.futures
import statistics
import math
import os
import csv
import sys
import json
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
_SRV_LAT_EMA_MS = {}
_SRV_LAT_LOCK = threading.Lock()
_MPC_MIN_ALLOC_LOCK = threading.Lock()
_MPC_MIN_ALLOC = 0.0
_MAX_ALLOC = 4.0
_BUDGET = 10
_UNC_SCALE = 1.0
_TIGHT_SLO_MS = 80.0
_CPU_SCALE_EXP = 0.85
_MPC_STATE_MODE = "dynamodb"
_LAST_AZURE_TRACE_META = None
LAMBDA_MEMORY_MB = 1024
PRICE_PER_GB_S_USD = 0.00001667

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

def calibrate_qos_threshold(task_name, factor=1.2, warmup_requests=30, sample_requests=150, budget=10, include_cold_start=True, percentile=90.0):
    budget = int(budget) if int(budget) > 0 else 10

    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='static_1.0', reset_state=True)

    warm_arrivals = generate_poisson_arrivals(rate=max(1.0, float(budget) * 20.0), num=max(1, int(warmup_requests)))
    run_phase('static_1.0', warm_up=True, max_workers=budget, arrival_times=warm_arrivals, max_inflight=budget)

    sample_arrivals = generate_poisson_arrivals(rate=max(1.0, float(budget) * 20.0), num=max(1, int(sample_requests)))
    results, _ = run_phase('static_1.0', warm_up=True, max_workers=budget, arrival_times=sample_arrivals, max_inflight=budget)

    success = [r for r in results if r.get('success', False)]
    if include_cold_start:
        cal_success = success
    else:
        cal_success = [r for r in success if not bool(r.get('is_cold_start', False))]
        if not cal_success:
            cal_success = success
    base_e2e = _pctl([r.get('e2e_latency', 0.0) for r in cal_success], percentile)
    base_srv = _pctl([r.get('server_latency', 0.0) for r in cal_success], percentile)

    qos_e2e = float(max(1.0, base_e2e * float(factor)))
    qos_srv = float(max(1.0, base_srv * float(factor)))
    return {
        "calibration_percentile": float(percentile),
        "base_pXX_e2e_ms": float(base_e2e),
        "base_pXX_srv_ms": float(base_srv),
        "qos_e2e_ms": float(qos_e2e),
        "qos_srv_ms": float(qos_srv)
    }

def _subsample_arrivals(arrival_times, n, offset=0):
    try:
        n = int(n)
    except Exception:
        n = 0
    if n <= 0:
        return []
    if not arrival_times:
        return []
    m = len(arrival_times)
    if offset < 0:
        offset = 0
    if offset >= m:
        offset = 0
    span = arrival_times[offset:]
    if len(span) <= n:
        return list(span)
    idx = np.linspace(0, len(span) - 1, n).astype(int)
    return [span[int(i)] for i in idx]

def _compact_arrivals(arrival_times, speedup=60.0):
    if not arrival_times:
        return []
    try:
        speedup = float(speedup)
    except Exception:
        speedup = 60.0
    if (not math.isfinite(speedup)) or speedup <= 0.0:
        speedup = 60.0
    speedup = float(max(1.0, min(3600.0, speedup)))
    try:
        t0 = float(arrival_times[0])
    except Exception:
        t0 = 0.0
    out = []
    for t in arrival_times:
        try:
            x = float(t) - t0
        except Exception:
            x = 0.0
        if not math.isfinite(x) or x < 0.0:
            x = 0.0
        out.append(x / speedup)
    return out

def calibrate_qos_threshold_on_azure_trace(task_name, trace_file, app_id, func_id, day, start_min, duration_min, scale, factor=1.2, warmup_requests=30, sample_requests=150, budget=10, include_cold_start=True, percentile=90.0):
    budget = int(budget) if int(budget) > 0 else 10
    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='static_1.0', reset_state=True)

    second_rates, _is_real = load_azure_trace(
        duration_min=int(duration_min),
        trace_file=str(trace_file),
        start_min=int(start_min),
        app_id=str(app_id),
        func_id=str(func_id),
        day=int(day),
        scale=float(scale),
        pick_most_bursty=False,
        auto_shift_empty_window=False,
    )
    arrival_times = generate_trace_arrivals(second_rates, base_rps=1.0)
    if not arrival_times:
        return calibrate_qos_threshold(task_name, factor=factor, warmup_requests=warmup_requests, sample_requests=sample_requests, budget=budget, include_cold_start=include_cold_start, percentile=percentile)

    warm_arrivals = _subsample_arrivals(arrival_times, int(warmup_requests), offset=0)
    sample_arrivals = _subsample_arrivals(arrival_times, int(sample_requests), offset=max(0, len(arrival_times) // 4))
    warm_arrivals = _compact_arrivals(warm_arrivals, speedup=120.0)
    sample_arrivals = _compact_arrivals(sample_arrivals, speedup=120.0)

    run_phase('static_1.0', warm_up=True, max_workers=budget, arrival_times=warm_arrivals, max_inflight=budget, replay_speedup=1.0, second_rates=second_rates)
    results, _ = run_phase('static_1.0', warm_up=True, max_workers=budget, arrival_times=sample_arrivals, max_inflight=budget, replay_speedup=1.0, second_rates=second_rates)

    success = [r for r in results if r.get('success', False)]
    if include_cold_start:
        cal_success = success
    else:
        cal_success = [r for r in success if not bool(r.get('is_cold_start', False))]
        if not cal_success:
            cal_success = success
    base_e2e = _pctl([r.get('e2e_latency', 0.0) for r in cal_success], percentile)
    base_srv = _pctl([r.get('server_latency', 0.0) for r in cal_success], percentile)

    qos_e2e = float(max(1.0, base_e2e * float(factor)))
    qos_srv = float(max(1.0, base_srv * float(factor)))
    return {
        "calibration_percentile": float(percentile),
        "base_pXX_e2e_ms": float(base_e2e),
        "base_pXX_srv_ms": float(base_srv),
        "qos_e2e_ms": float(qos_e2e),
        "qos_srv_ms": float(qos_srv)
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

def load_azure_trace_sample(duration_min=30):
    raise RuntimeError("Hardcoded Azure trace sample has been removed. Provide --azure_trace_file/--azure_app/--azure_func instead.")

def _resolve_trace_path(trace_file):
    if not trace_file:
        return ""
    trace_path = str(trace_file)
    if not os.path.isabs(trace_path):
        trace_path = os.path.abspath(trace_path)
    return trace_path

def _is_feather_trace_file(trace_path):
    try:
        p = str(trace_path).lower().strip()
    except Exception:
        return False
    return p.endswith(".feather")

def _iter_feather_record_batches(trace_path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    source = pa.memory_map(trace_path, "r")
    reader = ipc.open_file(source)
    for i in range(reader.num_record_batches):
        yield reader.get_batch(i)

def _read_azure_minute_series_from_feather(trace_file, app_id, func_id, day=0):
    trace_path = _resolve_trace_path(trace_file)
    if not trace_path or not os.path.exists(trace_path):
        return None, trace_path, None

    want_app = str(app_id).strip() if app_id is not None else ""
    want_func = str(func_id).strip() if func_id is not None else ""
    if not want_app or not want_func:
        return None, trace_path, None

    try:
        import pyarrow.compute as pc
    except Exception as e:
        raise RuntimeError("Reading .feather Azure trace requires pyarrow.") from e

    try:
        day_int = int(day)
    except Exception:
        day_int = 0
    if day_int < 0:
        day_int = 0

    minute_cols = [str(i) for i in range(1, 1441)]
    need_cols = ["HashApp", "HashFunction", "day"] + minute_cols

    for batch in _iter_feather_record_batches(trace_path):
        names = list(batch.schema.names)
        missing = [c for c in need_cols if c not in names]
        if missing:
            raise RuntimeError(f"Feather schema missing columns: {missing[:10]}")

        b = batch.select(need_cols)
        mask = pc.and_(pc.equal(b.column("HashApp"), want_app), pc.equal(b.column("HashFunction"), want_func))
        if day_int > 0:
            mask = pc.and_(mask, pc.equal(b.column("day"), day_int))

        mask_np = mask.to_numpy(zero_copy_only=False)
        idx = np.nonzero(mask_np)[0]
        if idx.size <= 0:
            continue

        r = int(idx[0])
        row = b.slice(r, 1).to_pydict()
        picked_day = int(row["day"][0]) if row.get("day") else None
        series = []
        for c in minute_cols:
            v = row.get(c, [0.0])[0]
            try:
                x = float(v)
            except Exception:
                x = 0.0
            if not math.isfinite(x) or x < 0.0:
                x = 0.0
            series.append(x)
        return series, trace_path, picked_day

    return None, trace_path, None

def _read_azure_minute_series(trace_file, app_id, func_id, day=0):
    trace_path = _resolve_trace_path(trace_file)
    if _is_feather_trace_file(trace_path):
        return _read_azure_minute_series_from_feather(trace_file, app_id, func_id, day=day)
    series, tp = _read_azure_minute_series_from_csv(trace_file, app_id, func_id)
    return series, tp, None

def _read_azure_minute_series_from_csv(trace_file, app_id, func_id):
    trace_path = _resolve_trace_path(trace_file)
    if not trace_path or not os.path.exists(trace_path):
        return None, trace_path

    want_app = str(app_id).strip() if app_id is not None else ""
    want_func = str(func_id).strip() if func_id is not None else ""
    if not want_app or not want_func:
        return None, trace_path

    with open(trace_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, trace_path

        is_wide = False
        if len(header) >= 6:
            tail = header[3:8]
            if all(str(x).strip().isdigit() for x in tail):
                is_wide = True

        if is_wide:
            numeric_start_idx = 3
            for row in reader:
                if not row or len(row) <= numeric_start_idx:
                    continue
                app = str(row[0]).strip()
                func = str(row[1]).strip()
                if app != want_app or func != want_func:
                    continue
                series = []
                for v in row[numeric_start_idx:]:
                    try:
                        x = float(v)
                    except Exception:
                        x = 0.0
                    if not math.isfinite(x) or x < 0.0:
                        x = 0.0
                    series.append(x)
                return series, trace_path

        header_lower = [str(h).strip().lower() for h in header]
        def _find_col(cands):
            for c in cands:
                if c in header_lower:
                    return header_lower.index(c)
            return -1

        app_idx = _find_col(["hashapp", "app", "app_id", "application", "application_id"])
        func_idx = _find_col(["hashfunction", "function", "func", "func_id", "function_id"])
        minute_idx = _find_col(["minute", "min", "minute_index", "t", "time", "time_min"])
        inv_idx = _find_col(["invocations", "invocation", "count", "value"])
        if min(app_idx, func_idx, minute_idx, inv_idx) < 0:
            return None, trace_path

        minute_map = {}
        max_min = -1
        for row in reader:
            if not row or len(row) <= max(app_idx, func_idx, minute_idx, inv_idx):
                continue
            app = str(row[app_idx]).strip()
            func = str(row[func_idx]).strip()
            if app != want_app or func != want_func:
                continue
            try:
                m = int(float(row[minute_idx]))
            except Exception:
                continue
            try:
                x = float(row[inv_idx])
            except Exception:
                x = 0.0
            if not math.isfinite(x) or x < 0.0:
                x = 0.0
            minute_map[m] = minute_map.get(m, 0.0) + x
            if m > max_min:
                max_min = m

        if max_min < 0:
            return None, trace_path
        series = [0.0] * (max_min + 1)
        for m, x in minute_map.items():
            if 0 <= m < len(series):
                series[m] = float(x)
        return series, trace_path
    return None, trace_path

def _window_avg_rps_from_minute_series(series, start_min, duration_min, scale):
    if not series:
        return 0.0, 0.0, 0
    try:
        start_min = int(start_min)
    except Exception:
        start_min = 0
    if start_min < 0:
        start_min = 0
    try:
        duration_min = int(duration_min)
    except Exception:
        duration_min = 30
    if duration_min <= 0:
        duration_min = 30
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    end = min(len(series), start_min + duration_min)
    window = series[start_min:end]
    if not window:
        return 0.0, 0.0, 0
    total_inv = float(sum(window)) * float(scale)
    avg_rps = float(total_inv) / float(60.0 * len(window))
    return avg_rps, total_inv, int(len(window))

def select_azure_windows_by_avg_rps(trace_file, app_id, func_id, day, duration_min, stride_min, scale, target_rps_min, target_rps_max, max_windows):
    series, trace_path, picked_day = _read_azure_minute_series(trace_file, app_id, func_id, day=day)
    if not series:
        return [], trace_path

    try:
        duration_min = int(duration_min)
    except Exception:
        duration_min = 30
    if duration_min <= 0:
        duration_min = 30
    try:
        stride_min = int(stride_min)
    except Exception:
        stride_min = duration_min
    if stride_min <= 0:
        stride_min = duration_min

    try:
        target_rps_min = float(target_rps_min)
    except Exception:
        target_rps_min = 0.0
    try:
        target_rps_max = float(target_rps_max)
    except Exception:
        target_rps_max = float("inf")
    if not math.isfinite(target_rps_min):
        target_rps_min = 0.0
    if not math.isfinite(target_rps_max):
        target_rps_max = float("inf")
    if target_rps_max < target_rps_min:
        target_rps_min, target_rps_max = target_rps_max, target_rps_min

    candidates = []
    max_start = max(0, len(series) - duration_min)
    for s in range(0, max_start + 1, stride_min):
        avg_rps, total_inv, used_min = _window_avg_rps_from_minute_series(series, s, duration_min, scale)
        if used_min <= 0 or total_inv <= 0.0:
            continue
        if avg_rps < target_rps_min or avg_rps > target_rps_max:
            continue
        candidates.append(s)

    if not candidates:
        return [], trace_path

    try:
        max_windows = int(max_windows)
    except Exception:
        max_windows = len(candidates)
    if max_windows <= 0 or max_windows >= len(candidates):
        return candidates, trace_path

    idxs = np.linspace(0, len(candidates) - 1, num=max_windows)
    picked = [candidates[int(round(i))] for i in idxs]
    dedup = []
    seen = set()
    for s in picked:
        if s in seen:
            continue
        seen.add(s)
        dedup.append(s)
    if picked_day is not None:
        print(f">>> Azure trace day used: {picked_day}")
    return dedup, trace_path

def scan_azure_windows_any_function_wide_csv(trace_file, duration_min, stride_min, scale, target_rps_min, target_rps_max, top_funcs):
    trace_path = _resolve_trace_path(trace_file)
    if not trace_path or not os.path.exists(trace_path):
        return [], trace_path

    try:
        duration_min = int(duration_min)
    except Exception:
        duration_min = 30
    if duration_min <= 0:
        duration_min = 30
    try:
        stride_min = int(stride_min)
    except Exception:
        stride_min = duration_min
    if stride_min <= 0:
        stride_min = duration_min
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    try:
        target_rps_min = float(target_rps_min)
    except Exception:
        target_rps_min = 0.0
    try:
        target_rps_max = float(target_rps_max)
    except Exception:
        target_rps_max = float("inf")
    if not math.isfinite(target_rps_min):
        target_rps_min = 0.0
    if not math.isfinite(target_rps_max):
        target_rps_max = float("inf")
    if target_rps_max < target_rps_min:
        target_rps_min, target_rps_max = target_rps_max, target_rps_min

    results = []
    with open(trace_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 6:
            return [], trace_path
        tail = header[3:8]
        if not all(str(x).strip().isdigit() for x in tail):
            return [], trace_path

        numeric_start_idx = 3
        for row in reader:
            if not row or len(row) <= numeric_start_idx:
                continue
            app = str(row[0]).strip()
            func = str(row[1]).strip()
            series = []
            for v in row[numeric_start_idx:]:
                try:
                    x = float(v)
                except Exception:
                    x = 0.0
                if not math.isfinite(x) or x < 0.0:
                    x = 0.0
                series.append(x)

            if len(series) < duration_min:
                continue
            prefix = [0.0]
            acc = 0.0
            for x in series:
                acc += float(x)
                prefix.append(acc)
            max_start = len(series) - duration_min
            hits = 0
            example = None
            for s in range(0, max_start + 1, stride_min):
                seg_sum = float(prefix[s + duration_min] - prefix[s])
                if seg_sum <= 0.0:
                    continue
                avg_rps = (seg_sum * float(scale)) / float(60.0 * duration_min)
                if avg_rps < target_rps_min or avg_rps > target_rps_max:
                    continue
                hits += 1
                if example is None:
                    example = (s, avg_rps, seg_sum * float(scale))
            if hits > 0:
                results.append((hits, app, func, example))

    results.sort(key=lambda x: (-x[0], x[1], x[2]))
    try:
        top_funcs = int(top_funcs)
    except Exception:
        top_funcs = 10
    if top_funcs <= 0:
        top_funcs = 10
    return results[:top_funcs], trace_path

def scan_azure_windows_any_function_wide_feather(trace_file, duration_min, stride_min, scale, target_rps_min, target_rps_max, top_funcs, max_rows=20000, row_chunk=128, day_filter=None):
    trace_path = _resolve_trace_path(trace_file)
    if not trace_path or not os.path.exists(trace_path):
        return [], trace_path

    try:
        import pyarrow as pa
    except Exception as e:
        raise RuntimeError("Reading .feather Azure trace requires pyarrow.") from e

    try:
        duration_min = int(duration_min)
    except Exception:
        duration_min = 30
    if duration_min <= 0:
        duration_min = 30
    try:
        stride_min = int(stride_min)
    except Exception:
        stride_min = duration_min
    if stride_min <= 0:
        stride_min = duration_min
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    try:
        target_rps_min = float(target_rps_min)
    except Exception:
        target_rps_min = 0.0
    try:
        target_rps_max = float(target_rps_max)
    except Exception:
        target_rps_max = float("inf")
    if not math.isfinite(target_rps_min):
        target_rps_min = 0.0
    if not math.isfinite(target_rps_max) or target_rps_max <= 0.0:
        target_rps_max = float("inf")
    if target_rps_max < target_rps_min:
        target_rps_min, target_rps_max = target_rps_max, target_rps_min

    try:
        top_funcs = int(top_funcs)
    except Exception:
        top_funcs = 10
    if top_funcs <= 0:
        top_funcs = 10

    dur = int(duration_min)
    starts = list(range(0, max(0, 1440 - dur) + 1, int(stride_min)))
    if not starts:
        starts = [0]
    starts_arr = np.array(starts, dtype=np.int32)

    minute_cols = [str(i) for i in range(1, 1441)]
    need_cols = ["HashApp", "HashFunction", "day"] + minute_cols

    import heapq
    heap = []
    scanned = 0
    try:
        max_rows = int(max_rows)
    except Exception:
        max_rows = 20000
    if max_rows < 0:
        max_rows = 0
    try:
        row_chunk = int(row_chunk)
    except Exception:
        row_chunk = 128
    if row_chunk <= 0:
        row_chunk = 128

    day_allow = None
    if day_filter is not None:
        try:
            day_allow = int(day_filter)
        except Exception:
            day_allow = None

    for batch in _iter_feather_record_batches(trace_path):
        names = list(batch.schema.names)
        missing = [c for c in need_cols if c not in names]
        if missing:
            raise RuntimeError(f"Feather schema missing columns: {missing[:10]}")

        b = batch.select(need_cols)
        nrows = int(b.num_rows)
        if nrows <= 0:
            continue

        for off in range(0, nrows, row_chunk):
            if max_rows > 0 and scanned >= max_rows:
                break
            take = min(row_chunk, nrows - off)
            if max_rows > 0:
                take = min(take, max_rows - scanned)
            if take <= 0:
                continue

            rb = b.slice(off, take)
            try:
                apps = rb.column(0).to_pylist()
                funcs = rb.column(1).to_pylist()
                days = rb.column(2).to_pylist()
            except Exception:
                continue

            if day_allow is not None:
                keep_idx = [i for i, d in enumerate(days) if int(d) == int(day_allow)]
                if not keep_idx:
                    scanned += take
                    continue
                try:
                    rb = rb.take(pa.array(keep_idx, type=pa.int32()))
                except Exception:
                    scanned += take
                    continue
                try:
                    apps = rb.column(0).to_pylist()
                    funcs = rb.column(1).to_pylist()
                    days = rb.column(2).to_pylist()
                except Exception:
                    scanned += take
                    continue

            scanned += take
            rr = int(rb.num_rows)
            if rr <= 0:
                continue

            try:
                cols = []
                for ci in range(3, rb.num_columns):
                    arr = rb.column(ci).to_numpy(zero_copy_only=False)
                    cols.append(np.asarray(arr, dtype=np.float32))
                mat = np.column_stack(cols) if cols else np.zeros((rr, 0), dtype=np.float32)
            except Exception:
                continue
            if mat.shape[1] < dur:
                continue
            mat = np.where(np.isfinite(mat) & (mat > 0.0), mat, 0.0).astype(np.float32, copy=False)

            prefix = np.concatenate([np.zeros((mat.shape[0], 1), dtype=np.float32), np.cumsum(mat, axis=1, dtype=np.float32)], axis=1)
            seg = prefix[:, starts_arr + dur] - prefix[:, starts_arr]
            avg = (seg * float(scale)) / float(60.0 * dur)
            mask = (seg > 0.0) & (avg >= float(target_rps_min)) & (avg <= float(target_rps_max))
            hits = mask.sum(axis=1).astype(int)

            rows = np.nonzero(hits > 0)[0]
            for r in rows:
                w = int(np.argmax(mask[r]))
                ex = (int(starts_arr[w]), float(avg[r, w]), float(seg[r, w] * float(scale)))
                item = (int(hits[r]), str(apps[int(r)]), str(funcs[int(r)]), int(days[int(r)]), ex)
                if len(heap) < top_funcs:
                    heapq.heappush(heap, item)
                else:
                    if item > heap[0]:
                        heapq.heapreplace(heap, item)

        if max_rows > 0 and scanned >= max_rows:
            break

    heap.sort(reverse=True)
    return heap, trace_path

def load_azure_trace_from_csv(trace_file, duration_min=30, start_min=0, app_id=None, func_id=None, day=0, scale=1.0, pick_most_bursty=False, auto_shift_empty_window=False):
    global _LAST_AZURE_TRACE_META
    if not trace_file:
        return None
    trace_path = _resolve_trace_path(trace_file)
    if not os.path.exists(trace_path):
        print(f"[WARN] Azure trace file not found: {trace_path}")
        return None

    try:
        start_min = int(start_min)
    except Exception:
        start_min = 0
    if start_min < 0:
        start_min = 0
    try:
        duration_min = int(duration_min)
    except Exception:
        duration_min = 30
    if duration_min <= 0:
        duration_min = 30

    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    series = None
    picked = None
    picked_day = None

    if _is_feather_trace_file(trace_path):
        if bool(pick_most_bursty):
            raise RuntimeError("--azure_pick_most_bursty is not supported for .feather traces.")
        series, _tp, picked_day = _read_azure_minute_series_from_feather(trace_file, app_id, func_id, day=day)
        if series is None:
            print(f"[WARN] Azure trace row not found for app={app_id} func={func_id} day={day} in {trace_path}")
            return None
        picked = (str(app_id).strip(), str(func_id).strip())
    else:
        with open(trace_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or len(header) < 5:
                print(f"[WARN] Azure trace header invalid: {trace_path}")
                return None

            numeric_start_idx = 3

            want_app = str(app_id).strip() if app_id is not None else ""
            want_func = str(func_id).strip() if func_id is not None else ""
            pick_most_bursty = bool(pick_most_bursty) and (not want_app) and (not want_func)

            best_score = -1.0
            best_row = None
            best_pick = None

            for row in reader:
                if not row or len(row) <= numeric_start_idx:
                    continue
                app = str(row[0]).strip()
                func = str(row[1]).strip()

                if not pick_most_bursty:
                    if want_app and app != want_app:
                        continue
                    if want_func and func != want_func:
                        continue
                    picked = (app, func)
                    series = row[numeric_start_idx:]
                    break

                values = row[numeric_start_idx:]
                start_idx = int(start_min)
                end_idx = min(len(values), start_idx + int(duration_min))
                if start_idx >= len(values):
                    continue
                window = values[start_idx:end_idx]
                if not window:
                    continue
                nums = []
                for v in window:
                    try:
                        x = float(v)
                    except Exception:
                        x = 0.0
                    if not math.isfinite(x) or x < 0.0:
                        x = 0.0
                    nums.append(x)
                if not nums:
                    continue
                peak = max(nums)
                mean = float(sum(nums) / max(1, len(nums)))
                score = float(peak / (mean + 1e-9))
                if score > best_score:
                    best_score = score
                    best_row = values
                    best_pick = (app, func)

            if pick_most_bursty and best_row is not None:
                series = best_row
                picked = best_pick

        if series is None:
            print(f"[WARN] Azure trace row not found for app={app_id} func={func_id} in {trace_path}")
            return None

    start_idx = int(start_min)
    requested_start_idx = int(start_idx)
    end_idx = start_idx + int(duration_min)
    if start_idx >= len(series):
        start_idx = max(0, len(series) - int(duration_min))
        end_idx = len(series)
    end_idx = min(len(series), end_idx)
    window = series[start_idx:end_idx]

    try:
        dur = int(duration_min)
    except Exception:
        dur = len(window)
    if dur <= 0:
        dur = len(window)

    shifted = False
    if bool(auto_shift_empty_window) and window:
        numeric_series = []
        for v in series:
            try:
                x = float(v)
            except Exception:
                x = 0.0
            if not math.isfinite(x) or x < 0.0:
                x = 0.0
            numeric_series.append(x)

        wsum = float(sum(numeric_series[start_idx:end_idx]))
        if wsum <= 0.0 and len(numeric_series) >= dur:
            prefix = [0.0]
            acc = 0.0
            for x in numeric_series:
                acc += float(x)
                prefix.append(acc)

            best_sum = -1.0
            best_start = 0
            max_start = max(0, len(numeric_series) - dur)
            for s in range(0, max_start + 1):
                e = s + dur
                seg_sum = float(prefix[e] - prefix[s])
                if seg_sum > best_sum:
                    best_sum = seg_sum
                    best_start = s

            if best_sum > 0.0:
                start_idx = int(best_start)
                end_idx = min(len(series), start_idx + dur)
                window = series[start_idx:end_idx]
                shifted = True

    second_rps = []
    total_inv = 0.0
    for v in window:
        try:
            per_min = float(v)
        except Exception:
            per_min = 0.0
        if not math.isfinite(per_min) or per_min < 0.0:
            per_min = 0.0
        total_inv += float(per_min)
        rps = (per_min / 60.0) * float(scale)
        second_rps.extend([rps] * 60)
    total_inv = float(total_inv) * float(scale)
    avg_rps = float(total_inv) / float(60.0 * max(1, len(window)))

    src = "Feather" if _is_feather_trace_file(trace_path) else "CSV"
    shift_note = ", auto_shifted_start=1" if shifted else ""
    req_note = f", requested_start_min={requested_start_idx}"
    day_note = f", day={picked_day}" if picked_day is not None else ""
    if picked:
        print(f"Loading Azure trace from {src}: file={trace_path}, app={picked[0]}, func={picked[1]}{day_note}{req_note}, start_min={start_idx}, duration_min={len(window)}, scale={scale:.3f}{shift_note}")
    else:
        print(f"Loading Azure trace from {src}: file={trace_path}{day_note}{req_note}, start_min={start_idx}, duration_min={len(window)}, scale={scale:.3f}{shift_note}")
    _LAST_AZURE_TRACE_META = {
        "trace_file": trace_path,
        "app": picked[0] if picked else None,
        "func": picked[1] if picked else None,
        "day": int(picked_day) if picked_day is not None else None,
        "requested_start_min": int(requested_start_idx),
        "start_min": int(start_idx),
        "duration_min": int(len(window)),
        "scale": float(scale),
        "window_total_invocations": float(total_inv),
        "window_avg_rps": float(avg_rps),
        "auto_shifted": bool(shifted),
        "auto_shift_enabled": bool(auto_shift_empty_window),
    }
    return second_rps

def load_azure_trace(duration_min=30, trace_file=None, start_min=0, app_id=None, func_id=None, day=0, scale=1.0, pick_most_bursty=False, auto_shift_empty_window=False):
    second_rps = load_azure_trace_from_csv(
        trace_file=trace_file,
        duration_min=duration_min,
        start_min=start_min,
        app_id=app_id,
        func_id=func_id,
        day=day,
        scale=scale,
        pick_most_bursty=pick_most_bursty,
        auto_shift_empty_window=auto_shift_empty_window,
    )
    if second_rps is not None:
        return second_rps, True
    raise RuntimeError("Failed to load Azure trace from file. Provide a valid --azure_trace_file and matching --azure_app/--azure_func.")

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

def _apply_burst_profile(second_rates, profile, total_seconds):
    if not second_rates or not profile:
        return second_rates
    try:
        total_seconds = int(total_seconds)
    except Exception:
        return second_rates
    if total_seconds <= 0:
        return second_rates
    segs = []
    for raw in str(profile).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            a, b = raw.split(":", 1)
        elif "@" in raw:
            a, b = raw.split("@", 1)
        else:
            continue
        try:
            mult = float(a.strip())
        except Exception:
            continue
        try:
            secs = int(float(b.strip()))
        except Exception:
            continue
        if (not math.isfinite(mult)) or mult <= 0.0 or secs <= 0:
            continue
        segs.append((float(mult), int(secs)))
    if not segs:
        return second_rates
    out = []
    for mult, secs in segs:
        out.extend([float(mult)] * int(secs))
        if len(out) >= total_seconds:
            break
    if not out:
        return second_rates
    if len(out) < total_seconds:
        out.extend([out[-1]] * int(total_seconds - len(out)))
    elif len(out) > total_seconds:
        out = out[:total_seconds]
    base = list(second_rates)
    if len(base) < total_seconds:
        base = base + [0.0] * int(total_seconds - len(base))
    elif len(base) > total_seconds:
        base = base[:total_seconds]
    return [float(r) * float(m) for r, m in zip(base, out)]

def run_single_request(idx, strategy, start_time, inflight=1, queue_delay_ms=0.0, rps_hint=0.0, backlog_hint=None, budget_hint=None):
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
        "backlog": int(backlog_hint) if backlog_hint is not None else int(inflight),
        "concurrency": float(backlog_hint) if backlog_hint is not None else float(inflight),
        "cpu_util": 0.5,
        "error_rate": 0.0,
        "rps": float(rps_hint or 0.0),
        "e2e_overhead_ms": float(_E2E_OVERHEAD_EMA),
        "slo_limit": float(SERVER_SLO_MS),
        "budget": float(budget_hint) if budget_hint is not None else float(_BUDGET),
        "max_alloc": float(_MAX_ALLOC),
        "unc_scale": float(_UNC_SCALE),
        "tight_slo_ms": float(_TIGHT_SLO_MS),
        "cpu_scale_exp": float(_CPU_SCALE_EXP),
        "state_mode": str(_MPC_STATE_MODE),
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
    try:
        qd = float(queue_delay_ms or 0.0)
    except Exception:
        qd = 0.0
    if not math.isfinite(qd) or qd < 0.0:
        qd = 0.0
    e2e_latency = ((time.time() - t0) * 1000.0) + qd
    
    if worker_result:
        try:
            is_cold_start = bool((worker_result.get('response') or {}).get('is_cold_start', False))
        except Exception:
            is_cold_start = False
        server_latency = 0.0
        try:
            if 'response' in worker_result and isinstance(worker_result.get('response'), dict):
                server_latency = float(worker_result['response'].get('latency_ms', 0.0) or 0.0)
            if (not math.isfinite(server_latency)) or server_latency <= 0.0:
                server_latency = float(worker_result.get('client_duration', 0.0) or 0.0)
            if (not math.isfinite(server_latency)) or server_latency < 0.0:
                server_latency = 0.0
        except Exception:
            try:
                server_latency = float(worker_result.get('client_duration', 0.0) or 0.0)
            except Exception:
                server_latency = 0.0
        res = {
            'id': idx,
            'strategy': strategy,
            'priority': priority,
            'e2e_latency': e2e_latency,
            'queue_delay_ms': qd,
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
            'is_cold_start': is_cold_start,
            'timestamp': time.time()
        }
    else:
        res = {
            'id': idx,
            'strategy': strategy,
            'priority': priority,
            'e2e_latency': e2e_latency,
            'queue_delay_ms': qd,
            'violation': True,
            'success': False,
            'is_cold_start': False,
            'timestamp': time.time()
        }
    return res

def run_phase(strategy_name, warm_up=False, max_workers=5, arrival_times=None, num_requests=100, arrival_rate=5.0, max_inflight=0, replay_speedup=1.0, second_rates=None):
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
                try:
                    srv_ms = float(res.get("server_latency", 0.0) or 0.0)
                except Exception:
                    srv_ms = 0.0
                if math.isfinite(srv_ms) and srv_ms > 0.0:
                    with _SRV_LAT_LOCK:
                        prev = float(_SRV_LAT_EMA_MS.get(strategy_name, 40.0) or 40.0)
                        _SRV_LAT_EMA_MS[strategy_name] = 0.95 * prev + 0.05 * srv_ms
            if not warm_up:
                # 实时打印每个请求的结果 (v29.2 - 修复 Baseline 格式化错误)
                try:
                    prev_a = res.get('prev_alloc', '?')
                    new_a = res.get('new_alloc', '?')
                    
                    # 尝试转换，如果是 '?' 则保持原样
                    prev_str = f"{float(prev_a):.3f}" if isinstance(prev_a, (int, float)) or (isinstance(prev_a, str) and prev_a.replace('.','',1).isdigit()) else str(prev_a)
                    new_str = f"{float(new_a):.3f}" if isinstance(new_a, (int, float)) or (isinstance(new_a, str) and new_a.replace('.','',1).isdigit()) else str(new_a)
                    
                    print(f"[{strategy_name}] Req {res['id']:2d}: Alloc={res['alloc']:.2f}, E2E={res['e2e_latency']:.1f}ms, Srv={res.get('server_latency', 0.0):.1f}ms, Ver={res['version']}, PrevA={prev_str}, NewA={new_str}, P90_B={res.get('p90_belief', 0.0):.1f}, Unc={res.get('uncertainty', 0.0):.1f}, Price={res.get('shadow_price', 0.0):.1f}")
                except Exception as fmt_e:
                    # Fallback print if formatting fails
                    print(f"[{strategy_name}] Req {res['id']:2d}: Alloc={res['alloc']}, E2E={res['e2e_latency']:.1f}ms (Fmt Error: {fmt_e})")
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            spd = float(replay_speedup)
        except Exception:
            spd = 1.0
        if not math.isfinite(spd) or spd < 0.0:
            spd = 1.0
        for i, delay in enumerate(arrival_times):
            if spd > 0.0:
                now_trace = (time.time() - phase_start) * spd
                wait_trace = float(delay) - float(now_trace)
                if wait_trace > 0:
                    time.sleep(wait_trace / spd)
            
            planned_wall_ts = None
            if spd > 0.0:
                planned_wall_ts = phase_start + (float(delay) / spd)

            if inflight_sem is not None:
                inflight_sem.acquire()
            submit_ts = time.time()
            qd_ms = 0.0
            if planned_wall_ts is not None:
                qd_ms = float(max(0.0, (submit_ts - planned_wall_ts) * 1000.0))
            with inflight_lock:
                inflight_count += 1
                inflight_snapshot = inflight_count
            if second_rates is not None:
                sec_idx = int(float(delay)) if float(delay) >= 0.0 else 0
                if sec_idx < 0:
                    sec_idx = 0
                if sec_idx >= len(second_rates):
                    sec_idx = len(second_rates) - 1 if second_rates else 0
                try:
                    rps_hint = float(second_rates[sec_idx] or 0.0)
                except Exception:
                    rps_hint = 0.0
            else:
                rps_hint = float(arrival_rate or 0.0)

            with _SRV_LAT_LOCK:
                srv_ema = float(_SRV_LAT_EMA_MS.get(strategy_name, 40.0) or 40.0)
            if (not math.isfinite(srv_ema)) or srv_ema <= 0.0:
                srv_ema = 40.0
            try:
                qlen_est = int(float(qd_ms) / max(1.0, float(srv_ema)))
            except Exception:
                qlen_est = 0
            if qlen_est < 0:
                qlen_est = 0
            backlog_hint = int(inflight_snapshot + qlen_est)

            f = executor.submit(run_single_request, i, strategy_name, phase_start, inflight_snapshot, qd_ms, rps_hint, backlog_hint, _BUDGET)
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
        e2e_violations = sum(1 for r in results if (not r.get('success', False)) or (r.get('e2e_latency', 0) > E2E_SLO_MS))
        e2e_viol_rate = (e2e_violations / total) * 100
        
        # Server Violations
        server_violations = sum(1 for r in results if (not r.get('success', False)) or (r.get('server_latency', 0) > SERVER_SLO_MS))
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
            "gb_s_per_success": 0.0,
            "cost_per_success_usd": 0.0,
            "cost_per_1m_success_usd": 0.0,
            "gb_s_per_request": 0.0,
            "cost_per_request_usd": 0.0,
            "cost_per_1m_request_usd": 0.0,
            "alloc_p50": 0.0,
            "alloc_p90": 0.0,
            "alloc_std": 0.0,
            "alloc_churn": 0.0,
        }
    total = len(results)
    e2e_violations = sum(1 for r in results if (not r.get("success", False)) or (r.get("e2e_latency", 0) > E2E_SLO_MS))
    e2e_viol_rate = (e2e_violations / total) * 100
    server_violations = sum(1 for r in results if (not r.get("success", False)) or (r.get("server_latency", 0) > SERVER_SLO_MS))
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

    mem_gb = float(LAMBDA_MEMORY_MB) / 1024.0
    gb_s_acc_success = 0.0
    for r in success:
        try:
            a = float(r.get("alloc", 1.0) or 1.0)
            if not math.isfinite(a) or a <= 0.0:
                a = 1.0
            s = float(r.get('server_latency', 0.0) or 0.0)
            if math.isfinite(mem_gb) and mem_gb > 0.0 and math.isfinite(s) and s > 0.0:
                gb_s_acc_success += (mem_gb * a) * (s / 1000.0)
        except Exception:
            continue
    gb_s_per_success = float(gb_s_acc_success / max(1, len(success)))
    cost_per_success_usd = float(gb_s_per_success * float(PRICE_PER_GB_S_USD))
    cost_per_1m_success_usd = float(cost_per_success_usd * 1_000_000.0)

    gb_s_acc_req = 0.0
    for r in results:
        try:
            a = float(r.get("alloc", 1.0) or 1.0)
            if not math.isfinite(a) or a <= 0.0:
                a = 1.0
            s = float(r.get("server_latency", 0.0) or 0.0)
            if math.isfinite(mem_gb) and mem_gb > 0.0 and math.isfinite(s) and s > 0.0:
                gb_s_acc_req += (mem_gb * a) * (s / 1000.0)
        except Exception:
            continue
    gb_s_per_request = float(gb_s_acc_req / max(1, total))
    cost_per_request_usd = float(gb_s_per_request * float(PRICE_PER_GB_S_USD))
    cost_per_1m_request_usd = float(cost_per_request_usd * 1_000_000.0)

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
        "gb_s_per_success": gb_s_per_success,
        "cost_per_success_usd": cost_per_success_usd,
        "cost_per_1m_success_usd": cost_per_1m_success_usd,
        "gb_s_per_request": gb_s_per_request,
        "cost_per_request_usd": cost_per_request_usd,
        "cost_per_1m_request_usd": cost_per_1m_request_usd,
        "alloc_p50": alloc_p50,
        "alloc_p90": alloc_p90,
        "alloc_std": alloc_std,
        "alloc_churn": alloc_churn,
    }

def print_summary(results_by_name, paper_mode=True, paper_qos_metric="e2e"):
    print("\n======================================================================")
    qos_key = "srv_vio"
    qos_label = "Srv Viol %"
    if str(paper_qos_metric).strip().lower() in ["e2e", "e2e_vio", "end2end"]:
        qos_key = "e2e_vio"
        qos_label = "E2E Viol %"
    if paper_mode:
        print(f"{'Strategy':<22} | {qos_label:<10} | {'Cost($/1M req)':<14}")
    else:
        print(f"{'Strategy':<22} | {'E2E Viol %':<10} | {'Srv Viol %':<10} | {'AvgU':<6} | {'Dens':<6} | {'P90 E2E':<10} | {'AvgSrv':<8} | {'AvgE2E':<8} | {'GB-s':<7} | {'Cost($/1M req)':<14} | {'Overhead':<8} | {'AchRPS':<7}")
    print("----------------------------------------------------------------------")
    for name, results in results_by_name.items():
        if paper_mode and name == "static_0.60":
            continue
        m = _calc_metrics(results)
        if paper_mode:
            print(f"{name:<22} | {m.get(qos_key, 0.0):<10.2f} | {m['cost_per_1m_request_usd']:<14.4f}")
        else:
            print(f"{name:<22} | {m['e2e_vio']:<10.2f} | {m['srv_vio']:<10.2f} | {m['avg_alloc']:<6.2f} | {m['density']:<6.2f} | {m['p90_e2e']:<10.2f} | {m['avg_srv']:<8.2f} | {m['avg_e2e']:<8.2f} | {m['gb_s_per_request']:<7.3f} | {m['cost_per_1m_request_usd']:<14.2f} | {m['avg_overhead']:<8.2f} | {m['achieved_success_rps']:<7.2f}")
    print("======================================================================\n")

def print_aggregate_summary(metrics_by_strategy, paper_mode=True, paper_qos_metric="e2e"):
    def _mean_std(vals):
        xs = [float(v) for v in vals if v is not None]
        if not xs:
            return 0.0, 0.0
        if len(xs) == 1:
            return float(xs[0]), 0.0
        arr = np.array(xs, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    if paper_mode:
        qos_key = "srv_vio"
        qos_label = "Srv Viol %"
        if str(paper_qos_metric).strip().lower() in ["e2e", "e2e_vio", "end2end"]:
            qos_key = "e2e_vio"
            qos_label = "E2E Viol %"
        cols = [
            (qos_label, qos_key),
            ("Cost($/1M req)", "cost_per_1m_request_usd"),
        ]
    else:
        cols = [
            ("E2E Viol %", "e2e_vio"),
            ("Srv Viol %", "srv_vio"),
            ("AvgU", "avg_alloc"),
            ("Dens", "density"),
            ("P90 E2E", "p90_e2e"),
            ("AvgSrv", "avg_srv"),
            ("AvgE2E", "avg_e2e"),
            ("GB-s", "gb_s_per_request"),
            ("Cost($/1M req)", "cost_per_1m_request_usd"),
            ("Overhead", "avg_overhead"),
            ("AchRPS", "achieved_success_rps"),
        ]

    print("\n==================== AGGREGATE SUMMARY (MEAN±STD) ====================")
    header = f"{'Strategy':<22}"
    for label, _k in cols:
        header += f" | {label:<12}"
    print(header)
    print("-" * len(header))
    for name in sorted(metrics_by_strategy.keys()):
        if paper_mode and name == "static_0.60":
            continue
        row = f"{name:<22}"
        for _label, k in cols:
            mean, std = _mean_std([m.get(k) for m in metrics_by_strategy[name]])
            if k == "cost_per_1m_request_usd":
                row += f" | {mean:>8.4f}±{std:<7.4f}"
            else:
                row += f" | {mean:>6.2f}±{std:<5.2f}"
        print(row)
    print("=" * len(header))

def print_efficiency_summary(results_by_name):
    print("\n==================== EFFICIENCY / STABILITY SUMMARY ====================")
    print(f"{'Strategy':<22} | {'Util%':<6} | {'GB-s/succ':<10} | {'AllocP50':<8} | {'AllocP90':<8} | {'AllocStd':<8} | {'Churn':<7}")
    print("-------------------------------------------------------------------------")
    for name, results in results_by_name.items():
        m = _calc_metrics(results)
        print(
            f"{name:<22} | {m['util_pct']:<6.1f} | {m['gb_s_per_success']:<10.3f} | "
            f"{m['alloc_p50']:<8.2f} | {m['alloc_p90']:<8.2f} | {m['alloc_std']:<8.3f} | {m['alloc_churn']:<7.3f}"
        )
    print("=========================================================================\n")

def _ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def _safe_tag(s):
    s = str(s or "").strip()
    if not s:
        return ""
    out = []
    for ch in s:
        if ch.isalnum() or ch in ["-", "_", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)

def _mean_std(vals):
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    arr = np.array(xs, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))

def _aggregate_metrics(per_window_metrics):
    agg = {}
    for name, ms in (per_window_metrics or {}).items():
        keys = set()
        for m in ms:
            if isinstance(m, dict):
                keys.update(m.keys())
        out = {}
        for k in sorted(keys):
            mean, std = _mean_std([m.get(k) for m in ms if isinstance(m, dict)])
            out[k] = {"mean": float(mean), "std": float(std)}
        agg[name] = out
    return agg

def _write_report(report_dir, report_tag, report):
    report_dir = str(report_dir or "").strip()
    if not report_dir:
        return ""
    _ensure_dir(report_dir)
    tag = _safe_tag(report_tag) or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(report_dir, f"report_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f">>> Saved report: {path}")
    return path

def _load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _plot_from_reports(report_paths, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting.") from e

    reports = []
    for p in report_paths:
        p = str(p or "").strip()
        if not p:
            continue
        reports.append(_load_report(p))
    if not reports:
        raise RuntimeError("No reports provided for plotting.")

    _ensure_dir(out_dir)

    def _pick_metric(rep, strategy, key):
        try:
            return float(rep["aggregate"][strategy][key]["mean"]), float(rep["aggregate"][strategy][key]["std"])
        except Exception:
            return 0.0, 0.0

    def _strategy_order(strategies):
        pref = ["mpc", "static_0.80", "static_1.00", "aws_tt", "hpa_baseline"]
        out = []
        seen = set()
        for p in pref:
            if p in strategies and p not in seen:
                out.append(p)
                seen.add(p)
        for s in sorted(strategies):
            if s not in seen:
                out.append(s)
        return out

    all_strats = set()
    for rep in reports:
        all_strats.update((rep.get("aggregate") or {}).keys())
    strategies = _strategy_order(all_strats)

    labels = []
    for rep in reports:
        labels.append(str(rep.get("tag") or rep.get("label") or "run"))

    def plot_grouped(metric_key, ylabel, filename, accept_line=None):
        x = np.arange(len(strategies), dtype=float)
        w = 0.8 / max(1, len(reports))
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for i, rep in enumerate(reports):
            means = []
            stds = []
            for s in strategies:
                m, sd = _pick_metric(rep, s, metric_key)
                means.append(m)
                stds.append(sd)
            pos = x - 0.4 + (i + 0.5) * w
            ax.bar(pos, means, width=w, yerr=stds, capsize=3, label=labels[i], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=0)
        ax.set_ylabel(ylabel)
        if accept_line is not None:
            ax.axhline(float(accept_line), color="gray", linestyle="--", linewidth=1.5)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f">>> Saved figure: {out_path}")

    paper_metric = None
    for rep in reports:
        paper_metric = rep.get("paper_qos_metric")
        if paper_metric:
            break
    pm = str(paper_metric or "e2e").strip().lower()
    vio_key = "srv_vio"
    vio_label = "Srv QoS Violation (%)"
    if pm in ["e2e", "e2e_vio", "end2end"]:
        vio_key = "e2e_vio"
        vio_label = "E2E QoS Violation (%)"

    plot_grouped(vio_key, vio_label, "fig_qos_vio.png", accept_line=10.0)
    plot_grouped("cost_per_1m_success_usd", "Cost per 1M Successes ($)", "fig_cost.png", accept_line=None)

    fig, axes = plt.subplots(1, len(reports), figsize=(6 * len(reports), 4.8), sharey=True)
    if len(reports) == 1:
        axes = [axes]
    for ax, rep, lab in zip(axes, reports, labels):
        xs = []
        ys = []
        for s in strategies:
            x0, _ = _pick_metric(rep, s, vio_key)
            y0, _ = _pick_metric(rep, s, "cost_per_1m_success_usd")
            xs.append(x0)
            ys.append(y0)
        ax.scatter(xs, ys, s=60)
        for s, x0, y0 in zip(strategies, xs, ys):
            ax.text(x0, y0, s, fontsize=9, ha="left", va="bottom")
        ax.axvline(10.0, color="gray", linestyle="--", linewidth=1.5)
        ax.set_xlabel(vio_label)
        ax.set_title(lab)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Cost per 1M Successes ($)")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_tradeoff.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f">>> Saved figure: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=float, default=10.0)
    parser.add_argument("--rps_list", type=str, default="")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--task", type=str, default="linpack")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION","us-east-1"))
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max_inflight", type=int, default=0)
    parser.add_argument("--baselines", type=str, default="hpa,aws_tt,static")
    parser.add_argument("--static_allocs", type=str, default="0.6,0.8,1.0")
    parser.add_argument("--workload", type=str, default="fixed")
    parser.add_argument("--mode", type=str, default="compare")
    parser.add_argument("--min_alloc", type=float, default=0.0)
    parser.add_argument("--pareto_min_allocs", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--include_baselines_in_pareto", type=int, default=1)
    parser.add_argument("--qos_factor", type=float, default=1.2)
    parser.add_argument("--calibration_percentile", type=float, default=90.0)
    parser.add_argument("--calibration_include_cold_start", type=int, default=0)
    parser.add_argument("--server_slo_ms", type=float, default=0.0)
    parser.add_argument("--e2e_slo_ms", type=float, default=0.0)
    parser.add_argument("--print_efficiency", type=int, default=0)
    parser.add_argument("--enable_phase_warmup", type=int, default=1)
    parser.add_argument("--phase_warmup_requests", type=int, default=50)
    parser.add_argument("--cpu_scale_exp", type=float, default=0.85)
    parser.add_argument("--mpc_state_mode", type=str, default="dynamodb")
    parser.add_argument("--max_alloc", type=float, default=4.0)
    parser.add_argument("--unc_scale", type=float, default=1.0)
    parser.add_argument("--tight_slo_ms", type=float, default=80.0)
    parser.add_argument("--azure_trace_file", type=str, default="")
    parser.add_argument("--azure_app", type=str, default="")
    parser.add_argument("--azure_func", type=str, default="")
    parser.add_argument("--azure_day", type=int, default=0)
    parser.add_argument("--azure_start_min", type=int, default=0)
    parser.add_argument("--azure_start_mins", type=str, default="")
    parser.add_argument("--azure_num_windows", type=int, default=1)
    parser.add_argument("--azure_stride_min", type=int, default=30)
    parser.add_argument("--azure_skip_empty_windows", type=int, default=1)
    parser.add_argument("--azure_scale", type=float, default=1.0)
    parser.add_argument("--burst_profile", type=str, default="")
    parser.add_argument("--azure_pick_most_bursty", type=int, default=0)
    parser.add_argument("--azure_auto_shift_empty_window", type=int, default=0)
    parser.add_argument("--azure_filter_windows_by_avg_rps", type=int, default=0)
    parser.add_argument("--azure_target_avg_rps_min", type=float, default=0.0)
    parser.add_argument("--azure_target_avg_rps_max", type=float, default=0.0)
    parser.add_argument("--azure_scan_only", type=int, default=0)
    parser.add_argument("--azure_scan_top", type=int, default=20)
    parser.add_argument("--azure_scan_find_any_function", type=int, default=0)
    parser.add_argument("--azure_scan_top_functions", type=int, default=10)
    parser.add_argument("--azure_scan_max_rows", type=int, default=20000)
    parser.add_argument("--azure_scan_row_chunk", type=int, default=128)
    parser.add_argument("--azure_scan_day", type=int, default=-1)
    parser.add_argument("--lambda_memory_mb", type=int, default=1024)
    parser.add_argument("--gb_s_price_usd", type=float, default=0.00001667)
    parser.add_argument("--replay_speedup", type=float, default=1.0)
    parser.add_argument("--paper_mode", type=int, default=1)
    parser.add_argument("--paper_qos_metric", type=str, default="e2e")
    parser.add_argument("--strategies", type=str, default="static,mpc_integrated")
    parser.add_argument("--report_dir", type=str, default="")
    parser.add_argument("--report_tag", type=str, default="")
    parser.add_argument("--plot_reports", type=str, default="")
    parser.add_argument("--plot_out_dir", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if str(args.plot_reports).strip():
        paths = [p.strip() for p in str(args.plot_reports).split(",") if p.strip()]
        out_dir = str(args.plot_out_dir or "").strip()
        if not out_dir:
            if paths:
                out_dir = os.path.dirname(paths[0]) or os.getcwd()
            else:
                out_dir = os.getcwd()
        _plot_from_reports(paths, out_dir)
        sys.exit(0)
    BASE_RPS = float(args.rps)
    os.environ["AWS_REGION"] = args.region
    CURRENT_TASK = args.task
    _MAX_ALLOC = float(args.max_alloc)
    if _MAX_ALLOC <= 0.0:
        _MAX_ALLOC = 1.0
    _MAX_ALLOC = float(max(0.4, min(4.0, _MAX_ALLOC)))

    _UNC_SCALE = float(args.unc_scale)
    if not math.isfinite(_UNC_SCALE) or _UNC_SCALE <= 0.0:
        _UNC_SCALE = 1.0
    _UNC_SCALE = float(max(0.5, min(3.0, _UNC_SCALE)))

    _TIGHT_SLO_MS = float(args.tight_slo_ms)
    if not math.isfinite(_TIGHT_SLO_MS) or _TIGHT_SLO_MS <= 0.0:
        _TIGHT_SLO_MS = 80.0
    _TIGHT_SLO_MS = float(max(20.0, min(200.0, _TIGHT_SLO_MS)))
    _CPU_SCALE_EXP = float(args.cpu_scale_exp)
    if not math.isfinite(_CPU_SCALE_EXP) or _CPU_SCALE_EXP <= 0.0:
        _CPU_SCALE_EXP = 0.85
    _CPU_SCALE_EXP = float(max(0.5, min(1.0, _CPU_SCALE_EXP)))
    _MPC_STATE_MODE = str(args.mpc_state_mode or "dynamodb").strip().lower()
    if _MPC_STATE_MODE not in ["dynamodb", "local", "memory", "mem", "inmem"]:
        _MPC_STATE_MODE = "dynamodb"
    _BUDGET = int(args.budget)
    if _BUDGET <= 0:
        _BUDGET = 10
    print(f">>> Starting Experiment 1: MPC-Guard (Ours) vs Baselines")
    print(f">>> Task: {args.task}, Base RPS: {args.rps}, Duration: {args.minutes}m")
    print(f">>> Workload: {str(args.workload)}")
    print(f">>> Mode: {str(args.mode)}")
    print(f">>> Max Alloc: {_MAX_ALLOC:.2f}")
    print(f">>> Concurrency Budget: {_BUDGET}")
    print(f">>> MPC Unc Scale: {_UNC_SCALE:.2f}")
    print(f">>> MPC Tight SLO (ms): {_TIGHT_SLO_MS:.1f}")
    print(f">>> Phase Warmup: {int(args.enable_phase_warmup)} (requests={int(args.phase_warmup_requests)})")
    try:
        _REPLAY_SPEEDUP = float(args.replay_speedup)
    except Exception:
        _REPLAY_SPEEDUP = 1.0
    if (not math.isfinite(_REPLAY_SPEEDUP)) or _REPLAY_SPEEDUP <= 0.0:
        _REPLAY_SPEEDUP = 0.0
    print(f">>> Replay Speedup: {_REPLAY_SPEEDUP if _REPLAY_SPEEDUP > 0.0 else 0.0}x (1.0x=real-time, 0=disable sleep)")
    if _REPLAY_SPEEDUP != 1.0:
        print(">>> NOTE: replay_speedup != 1.0 changes the effective arrival process (debug/iteration only).")
    _PAPER_QOS_METRIC = str(args.paper_qos_metric or "e2e").strip().lower()
    if _PAPER_QOS_METRIC not in ["e2e", "srv"]:
        _PAPER_QOS_METRIC = "e2e"
    if int(args.paper_mode) == 1:
        print(f">>> Paper QoS Metric: {_PAPER_QOS_METRIC}")
    try:
        LAMBDA_MEMORY_MB = int(args.lambda_memory_mb)
    except Exception:
        LAMBDA_MEMORY_MB = 1024
    try:
        PRICE_PER_GB_S_USD = float(args.gb_s_price_usd)
    except Exception:
        PRICE_PER_GB_S_USD = 0.00001667
    print(f">>> Cost Model: effective_memory_GB = alloc * ({LAMBDA_MEMORY_MB}/1024), price_per_GB-s=${PRICE_PER_GB_S_USD}")
    if int(args.azure_filter_windows_by_avg_rps) == 1:
        lo = float(args.azure_target_avg_rps_min or 0.0)
        hi = float(args.azure_target_avg_rps_max or 0.0)
        if hi <= 0.0:
            hi = float("inf")
        print(f">>> Azure Window Filter: enabled (avg_rps in [{lo:.2f}, {hi if math.isfinite(hi) else float('inf'):.2f}])")
    else:
        print(f">>> Azure Window Filter: disabled")

    workload = str(args.workload).lower().strip()
    if workload in ["trace", "azure", "bursty"]:
        trace_path = _resolve_trace_path(str(args.azure_trace_file))
        if not trace_path:
            raise RuntimeError("Azure workload selected but --azure_trace_file is empty.")
        if not os.path.exists(trace_path):
            raise RuntimeError(f"Azure trace file not found: {trace_path}")
        if int(args.azure_scan_only) != 1 or int(args.azure_scan_find_any_function) != 1:
            if not str(args.azure_app).strip() or not str(args.azure_func).strip():
                raise RuntimeError("Azure workload selected but --azure_app/--azure_func not set.")

    if int(args.azure_scan_only) == 1:
        if int(args.azure_scan_find_any_function) == 1:
            lo = float(args.azure_target_avg_rps_min or 0.0)
            hi = float(args.azure_target_avg_rps_max or 0.0)
            if hi <= 0.0:
                hi = float("inf")
            tp = _resolve_trace_path(str(args.azure_trace_file))
            if _is_feather_trace_file(tp):
                res, trace_path = scan_azure_windows_any_function_wide_feather(
                    trace_file=str(args.azure_trace_file),
                    duration_min=int(args.minutes),
                    stride_min=int(args.azure_stride_min),
                    scale=float(args.azure_scale),
                    target_rps_min=lo,
                    target_rps_max=hi,
                    top_funcs=int(args.azure_scan_top_functions),
                    max_rows=int(args.azure_scan_max_rows),
                    row_chunk=int(args.azure_scan_row_chunk),
                    day_filter=(None if int(args.azure_scan_day) < 0 else int(args.azure_scan_day)),
                )
            else:
                res, trace_path = scan_azure_windows_any_function_wide_csv(
                    trace_file=str(args.azure_trace_file),
                    duration_min=int(args.minutes),
                    stride_min=int(args.azure_stride_min),
                    scale=float(args.azure_scale),
                    target_rps_min=lo,
                    target_rps_max=hi,
                    top_funcs=int(args.azure_scan_top_functions),
                )
            print("\n==================== AZURE WINDOW SCAN (ANY FUNCTION) ====================")
            print(f"file={trace_path}")
            print(f"duration_min={int(args.minutes)} stride_min={int(args.azure_stride_min)} scale={float(args.azure_scale):.3f}")
            print(f"target_avg_rps=[{lo:.2f}, {hi if math.isfinite(hi) else float('inf'):.2f}]")
            if not res:
                print("found_functions=0 (file may be non-wide format or no functions match target range)")
                print("=========================================================================")
                sys.exit(0)
            print(f"found_functions={len(res)} (showing top)")
            for i, item in enumerate(res, start=1):
                if len(item) == 4:
                    hits, app, func, ex = item
                    day = None
                else:
                    hits, app, func, day, ex = item
                if ex:
                    s, avg_rps, total_inv = ex
                    if day is None:
                        print(f"{i:02d}. app={app} func={func} windows_in_range={hits} example_start_min={int(s)} avg_rps={avg_rps:.2f} total_inv={total_inv:.1f}")
                    else:
                        print(f"{i:02d}. app={app} func={func} day={int(day)} windows_in_range={hits} example_start_min={int(s)} avg_rps={avg_rps:.2f} total_inv={total_inv:.1f}")
                else:
                    if day is None:
                        print(f"{i:02d}. app={app} func={func} windows_in_range={hits}")
                    else:
                        print(f"{i:02d}. app={app} func={func} day={int(day)} windows_in_range={hits}")
            print("=========================================================================")
            sys.exit(0)

        series, trace_path, picked_day = _read_azure_minute_series(str(args.azure_trace_file), args.azure_app, args.azure_func, day=int(args.azure_day))
        if not series:
            print(f"[ERROR] Unable to load minute series for app={args.azure_app} func={args.azure_func} day={int(args.azure_day)} from {trace_path}")
            sys.exit(2)

        try:
            duration_min = int(args.minutes)
        except Exception:
            duration_min = 30
        if duration_min <= 0:
            duration_min = 30
        try:
            stride_min = int(args.azure_stride_min)
        except Exception:
            stride_min = duration_min
        if stride_min <= 0:
            stride_min = duration_min

        try:
            lo = float(args.azure_target_avg_rps_min or 0.0)
        except Exception:
            lo = 0.0
        try:
            hi = float(args.azure_target_avg_rps_max or 0.0)
        except Exception:
            hi = 0.0
        if hi <= 0.0:
            hi = float("inf")

        max_windows = max(1, int(args.azure_num_windows) if int(args.azure_num_windows) > 0 else 1)
        top_n = max(1, int(args.azure_scan_top) if int(args.azure_scan_top) > 0 else 20)

        max_start = max(0, len(series) - duration_min)
        scanned = 0
        nonempty = 0
        inrange = []
        avg_rps_nonempty = []
        for s in range(0, max_start + 1, stride_min):
            scanned += 1
            avg_rps, total_inv, used_min = _window_avg_rps_from_minute_series(series, s, duration_min, float(args.azure_scale))
            if used_min <= 0 or total_inv <= 0.0:
                continue
            nonempty += 1
            avg_rps_nonempty.append(avg_rps)
            if avg_rps >= lo and avg_rps <= hi:
                inrange.append((abs(avg_rps - (0.5 * (lo + hi) if math.isfinite(hi) else lo)), avg_rps, total_inv, s))

        print("\n==================== AZURE WINDOW SCAN ====================")
        print(f"file={trace_path}")
        print(f"app={str(args.azure_app).strip()} func={str(args.azure_func).strip()}")
        print(f"duration_min={duration_min} stride_min={stride_min} scale={float(args.azure_scale):.3f}")
        print(f"target_avg_rps=[{lo:.2f}, {hi if math.isfinite(hi) else float('inf'):.2f}]")
        print(f"scanned_windows={scanned} nonempty_windows={nonempty}")
        if avg_rps_nonempty:
            arr = np.array(avg_rps_nonempty, dtype=float)
            print(f"nonempty_avg_rps: min={float(np.min(arr)):.2f} p50={float(np.percentile(arr, 50)):.2f} p90={float(np.percentile(arr, 90)):.2f} max={float(np.max(arr)):.2f}")
        else:
            print("nonempty_avg_rps: (none)")

        if not inrange:
            print("windows_in_target_range=0")
            print("==========================================================")
            sys.exit(0)

        inrange.sort(key=lambda x: x[0])
        print(f"windows_in_target_range={len(inrange)}")
        print(f"showing_top={min(top_n, len(inrange))} (closest to target band center)")
        for i, (_dist, avg_rps, total_inv, start_min) in enumerate(inrange[:top_n], start=1):
            print(f"{i:02d}. start_min={int(start_min)} avg_rps={float(avg_rps):.2f} total_inv={float(total_inv):.1f}")
        print("==========================================================")
        sys.exit(0)

    cal_info = {}
    fixed_srv = float(args.server_slo_ms or 0.0)
    fixed_e2e = float(args.e2e_slo_ms or 0.0)
    if fixed_srv > 0.0 or fixed_e2e > 0.0:
        if fixed_srv <= 0.0:
            fixed_srv = fixed_e2e
        if fixed_e2e <= 0.0:
            fixed_e2e = fixed_srv
        SERVER_SLO_MS = float(fixed_srv)
        E2E_SLO_MS = float(fixed_e2e)
        cal_info = {
            "mode": "fixed",
            "server_slo_ms": float(SERVER_SLO_MS),
            "e2e_slo_ms": float(E2E_SLO_MS),
        }
        print(f">>> QoS Thresholds (fixed): Server={SERVER_SLO_MS:.1f}ms, E2E={E2E_SLO_MS:.1f}ms")
    else:
        include_cold = bool(int(args.calibration_include_cold_start) == 1)
        workload = str(args.workload).lower().strip()
        use_trace_cal = bool(workload in ["trace", "azure", "bursty"] and str(args.azure_trace_file).strip() and str(args.azure_app).strip() and str(args.azure_func).strip())
        if use_trace_cal:
            cal_start = int(args.azure_start_min)
            raw = str(args.azure_start_mins).strip()
            if raw:
                try:
                    cal_start = int(raw.split(",")[0].strip())
                except Exception:
                    cal_start = int(args.azure_start_min)
            cal = calibrate_qos_threshold_on_azure_trace(
                args.task,
                trace_file=str(args.azure_trace_file),
                app_id=str(args.azure_app).strip(),
                func_id=str(args.azure_func).strip(),
                day=int(args.azure_day),
                start_min=int(cal_start),
                duration_min=int(max(1, int(args.minutes))),
                scale=float(args.azure_scale),
                factor=float(args.qos_factor),
                warmup_requests=30,
                sample_requests=150,
                budget=_BUDGET,
                include_cold_start=include_cold,
                percentile=float(args.calibration_percentile),
            )
        else:
            cal = calibrate_qos_threshold(
                args.task,
                factor=float(args.qos_factor),
                warmup_requests=30,
                sample_requests=150,
                budget=_BUDGET,
                include_cold_start=include_cold,
                percentile=float(args.calibration_percentile),
            )
        SERVER_SLO_MS = float(cal["qos_srv_ms"])
        E2E_SLO_MS = float(cal["qos_e2e_ms"])
        cal_info = dict(cal)
        if use_trace_cal:
            cal_info["mode"] = "auto_trace_with_cold" if include_cold else "auto_trace_warm_only"
        else:
            cal_info["mode"] = "auto_with_cold" if include_cold else "auto_warm_only"
        cal_tag = "with_cold" if include_cold else "warm_only"
        pxx = float(cal.get("calibration_percentile", 90.0) or 90.0)
        base_srv = float(cal.get("base_pXX_srv_ms", 0.0) or 0.0)
        base_e2e = float(cal.get("base_pXX_e2e_ms", 0.0) or 0.0)
        print(f">>> QoS Thresholds (auto/{cal_tag}): Server={SERVER_SLO_MS:.1f}ms, E2E={E2E_SLO_MS:.1f}ms (BaseP{pxx:.0f}: Srv={base_srv:.1f}ms, E2E={base_e2e:.1f}ms)")
    
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
    if int(args.paper_mode) == 1:
        static_allocs = [u for u in static_allocs if u >= 0.8]

    user_strategies = [x.strip() for x in str(args.strategies).split(',') if x.strip()]
    if "--strategies" in sys.argv:
        baselines = []
        if "hpa" in user_strategies:
            baselines.append("hpa")
        if "aws_tt" in user_strategies:
            baselines.append("aws_tt")
        if any(s.startswith("static") for s in user_strategies):
            baselines.append("static")
            specific_statics = []
            for s in user_strategies:
                if not s.startswith("static_"):
                    continue
                try:
                    specific_statics.append(float(s.split("_", 1)[1]))
                except Exception:
                    continue
            if specific_statics:
                static_allocs = specific_statics

    static_allocs = sorted(list(set(static_allocs)))
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
    if max_inflight <= 0:
        max_inflight = int(_BUDGET)
    if max_inflight > 0:
        print(f">>> Client inflight cap: {max_inflight} (prevents throttling dominating E2E)")

    with _MPC_MIN_ALLOC_LOCK:
        _MPC_MIN_ALLOC = float(args.min_alloc or 0.0)
        if not math.isfinite(_MPC_MIN_ALLOC):
            _MPC_MIN_ALLOC = 0.0
        _MPC_MIN_ALLOC = float(max(0.0, min(_MAX_ALLOC, _MPC_MIN_ALLOC)))
    print(f">>> MPC min_alloc (fixed): {_MPC_MIN_ALLOC:.2f}")

    sweep_rows = []
    for rps in rps_list:
        BASE_RPS = float(rps)
        print(f"\n>>> Running RPS={rps:.2f} <<<")

        workload = str(args.workload).lower().strip()
        warm_workers = max(1, min(10, int(args.workers)))
        do_phase_warmup = bool(int(args.enable_phase_warmup) == 1 and int(args.phase_warmup_requests) > 0)
        phase_warmup_n = int(args.phase_warmup_requests) if int(args.phase_warmup_requests) > 0 else 0

        start_mins = []
        if workload in ["trace", "azure", "bursty"]:
            raw = str(args.azure_start_mins).strip()
            if raw:
                for seg in raw.split(","):
                    seg = seg.strip()
                    if not seg:
                        continue
                    try:
                        start_mins.append(int(seg))
                    except Exception:
                        continue
            if not start_mins:
                nwin = int(args.azure_num_windows) if int(args.azure_num_windows) > 0 else 1
                stride = int(args.azure_stride_min) if int(args.azure_stride_min) > 0 else int(args.minutes)
                if int(args.azure_filter_windows_by_avg_rps) == 1:
                    lo = float(args.azure_target_avg_rps_min or 0.0)
                    hi = float(args.azure_target_avg_rps_max or 0.0)
                    if hi <= 0.0:
                        hi = float("inf")
                    start_mins, _tp = select_azure_windows_by_avg_rps(
                        trace_file=str(args.azure_trace_file),
                        app_id=str(args.azure_app) if str(args.azure_app).strip() else None,
                        func_id=str(args.azure_func) if str(args.azure_func).strip() else None,
                        day=int(args.azure_day),
                        duration_min=int(args.minutes),
                        stride_min=stride,
                        scale=float(args.azure_scale),
                        target_rps_min=lo,
                        target_rps_max=hi,
                        max_windows=nwin,
                    )
                    if start_mins:
                        print(f">>> Selected {len(start_mins)} windows by avg_rps filter.")
                if not start_mins:
                    base = int(args.azure_start_min)
                    start_mins = [base + i * stride for i in range(nwin)]
        else:
            start_mins = [0]

        per_window_metrics = {}
        windows_run = 0
        window_meta = []

        for w_i, start_min in enumerate(start_mins):
            second_rates = None
            if workload in ["trace", "azure", "bursty"]:
                print(f"\n>>> Window {w_i+1}/{len(start_mins)}: azure_start_min={int(start_min)} <<<")
                second_rates, is_real = load_azure_trace(
                    duration_min=int(args.minutes),
                    trace_file=str(args.azure_trace_file),
                    start_min=int(start_min),
                    app_id=str(args.azure_app) if str(args.azure_app).strip() else None,
                    func_id=str(args.azure_func) if str(args.azure_func).strip() else None,
                    day=int(args.azure_day),
                    scale=float(args.azure_scale),
                    pick_most_bursty=bool(int(args.azure_pick_most_bursty) == 1),
                    auto_shift_empty_window=bool(int(args.azure_auto_shift_empty_window) == 1),
                )
                total_seconds = int(max(1, int(args.minutes)) * 60)
                if str(args.burst_profile).strip():
                    second_rates = _apply_burst_profile(second_rates, str(args.burst_profile), total_seconds)
                if _LAST_AZURE_TRACE_META:
                    avg_rps = float(_LAST_AZURE_TRACE_META.get("window_avg_rps", 0.0) or 0.0)
                    if not math.isfinite(avg_rps):
                        avg_rps = 0.0
                    if int(args.azure_filter_windows_by_avg_rps) == 1:
                        lo = float(args.azure_target_avg_rps_min or 0.0)
                        hi = float(args.azure_target_avg_rps_max or 0.0)
                        if hi <= 0.0:
                            hi = float("inf")
                        if avg_rps < lo or avg_rps > hi:
                            print(f"[WARN] Window avg_rps={avg_rps:.2f} outside target range; skipping this window.")
                            continue
                arrival_times = generate_trace_arrivals(second_rates, base_rps=1.0)
                num_requests = len(arrival_times)
                src = "azure_trace_file"
                print(f"Generated {num_requests} requests with dynamic bursts ({src}).")
                if _LAST_AZURE_TRACE_META:
                    window_meta.append(dict(_LAST_AZURE_TRACE_META))
                if num_requests <= 0:
                    if int(args.azure_skip_empty_windows) == 1:
                        print("[WARN] Window produced 0 requests; skipping this window.")
                        continue
                    raise RuntimeError("Azure trace produced 0 requests.")
            else:
                arrival_times = generate_fixed_rps_arrivals(rps, args.minutes)
                num_requests = len(arrival_times)
                print(f"Generated {num_requests} requests with fixed rate.")

            results_by_name = {}

            if str(args.mode).lower() == "pareto":
                for mi in pareto_mins:
                    with _MPC_MIN_ALLOC_LOCK:
                        _MPC_MIN_ALLOC = float(mi)
                    name = f"mpc(min={mi:.2f})"
                    print(f"\n--- Running {name} ---")
                    invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='mpc_integrated', reset_state=True)
                    if do_phase_warmup:
                        run_phase('mpc_integrated', warm_up=True, max_workers=warm_workers, num_requests=phase_warmup_n, max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP)
                    res, _ = run_phase('mpc_integrated', max_workers=args.workers, arrival_times=arrival_times, arrival_rate=float(rps), max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP, second_rates=second_rates)
                    results_by_name[name] = res
            elif "mpc_integrated" in user_strategies or "mpc" in user_strategies:
                print(f"\n--- Running MPC-Guard (Ours) ---")
                invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='mpc_integrated', reset_state=True)
                if do_phase_warmup:
                    run_phase('mpc_integrated', warm_up=True, max_workers=warm_workers, num_requests=phase_warmup_n, max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP)
                mpc_results, _ = run_phase('mpc_integrated', max_workers=args.workers, arrival_times=arrival_times, arrival_rate=float(rps), max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP, second_rates=second_rates)
                results_by_name["mpc"] = mpc_results

            if str(args.mode).lower() != "pareto" or int(args.include_baselines_in_pareto) == 1:
                for b in baselines:
                    if b == "hpa":
                        print(f"\n--- Running HPA Baseline ---")
                        invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='baseline', reset_state=True)
                        if do_phase_warmup:
                            run_phase('baseline', warm_up=True, max_workers=warm_workers, num_requests=phase_warmup_n, max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP)
                        res, _ = run_phase('baseline', max_workers=args.workers, arrival_times=arrival_times, arrival_rate=float(rps), max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP, second_rates=second_rates)
                        results_by_name["hpa_baseline"] = res
                    elif b == "aws_tt":
                        print(f"\n--- Running AWS Target Tracking ---")
                        invoke_worker_lambda(decision={}, task={"id": "reset"}, mode='auto', strategy='aws_tt', reset_state=True)
                        if do_phase_warmup:
                            run_phase('aws_tt', warm_up=True, max_workers=warm_workers, num_requests=phase_warmup_n, max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP)
                        res, _ = run_phase('aws_tt', max_workers=args.workers, arrival_times=arrival_times, arrival_rate=float(rps), max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP, second_rates=second_rates)
                        results_by_name["aws_tt"] = res
                    elif b == "static":
                        for u in static_allocs:
                            name = f"static_{u:.2f}"
                            print(f"\n--- Running {name} ---")
                            if do_phase_warmup:
                                run_phase(f"static_{u}", warm_up=True, max_workers=warm_workers, num_requests=phase_warmup_n, max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP)
                            res, _ = run_phase(f"static_{u}", max_workers=args.workers, arrival_times=arrival_times, arrival_rate=float(rps), max_inflight=max_inflight, replay_speedup=_REPLAY_SPEEDUP, second_rates=second_rates)
                            results_by_name[name] = res

            windows_run += 1
            print_summary(results_by_name, paper_mode=bool(int(args.paper_mode) == 1), paper_qos_metric=_PAPER_QOS_METRIC)
            if int(args.print_efficiency) == 1:
                print_efficiency_summary(results_by_name)

            for name, results in results_by_name.items():
                m = _calc_metrics(results)
                sweep_rows.append({
                    "rps": float(rps),
                    "strategy": name,
                    **m
                })
                per_window_metrics.setdefault(name, []).append(m)

        if workload in ["trace", "azure", "bursty"] and windows_run <= 0:
            raise RuntimeError("All Azure windows produced 0 requests; nothing to evaluate.")

        if workload in ["trace", "azure", "bursty"] and len(start_mins) > 1 and windows_run > 0:
            print_aggregate_summary(per_window_metrics, paper_mode=bool(int(args.paper_mode) == 1), paper_qos_metric=_PAPER_QOS_METRIC)
            if window_meta:
                print("\n==================== WINDOWS USED ====================")
                for i, m in enumerate(window_meta, start=1):
                    print(
                        f"{i:02d}. file={m.get('trace_file')} app={m.get('app')} func={m.get('func')} "
                        f"requested_start_min={m.get('requested_start_min')} start_min={m.get('start_min')} "
                        f"duration_min={m.get('duration_min')} scale={m.get('scale'):.3f} "
                        f"avg_rps={float(m.get('window_avg_rps', 0.0) or 0.0):.2f} total_inv={float(m.get('window_total_invocations', 0.0) or 0.0):.1f} "
                        f"auto_shift_enabled={int(bool(m.get('auto_shift_enabled')))} auto_shifted={int(bool(m.get('auto_shifted')))}"
                    )
                print("======================================================")

            report_dir = str(args.report_dir or "").strip()
            if report_dir:
                report_tag = str(args.report_tag or "").strip()
                if not report_tag:
                    report_tag = f"{workload}_{args.task}_{int(args.minutes)}m_rps{float(rps):.2f}"
                report = {
                    "tag": report_tag,
                    "generated_utc": datetime.utcnow().isoformat() + "Z",
                    "workload": str(workload),
                    "task": str(args.task),
                    "minutes": float(args.minutes),
                    "base_rps": float(rps),
                    "budget": int(_BUDGET),
                    "workers": int(args.workers),
                    "max_inflight": int(max_inflight),
                    "replay_speedup": float(_REPLAY_SPEEDUP),
                    "paper_qos_metric": str(_PAPER_QOS_METRIC),
                    "qos": dict(cal_info or {}),
                    "azure": {
                        "trace_file": _resolve_trace_path(str(args.azure_trace_file)),
                        "app": str(args.azure_app),
                        "func": str(args.azure_func),
                        "day": int(args.azure_day),
                        "scale": float(args.azure_scale),
                        "start_mins_requested": [int(x) for x in start_mins],
                    },
                    "windows_used": list(window_meta),
                    "aggregate": _aggregate_metrics(per_window_metrics),
                    "per_window_metrics": per_window_metrics,
                }
                saved = _write_report(report_dir, report_tag, report)
                try:
                    _plot_from_reports([saved], report_dir)
                except Exception as e:
                    print(f"[WARN] Plotting failed: {e}")

    if len(rps_list) > 1 and str(args.mode).lower() != "pareto":
        print("\n==================== FINAL SUMMARY (ALL RPS) ====================")
        print(f"{'RPS':<6} | {'Strategy':<16} | {'E2E Viol %':<10} | {'Srv Viol %':<10} | {'AvgU':<6} | {'Dens':<6} | {'P90 E2E':<10} | {'AvgSrv':<8} | {'AvgE2E':<8} | {'Overhead':<8} | {'AchRPS':<7}")
        print("-----------------------------------------------------------------")
        for row in sweep_rows:
            print(f"{row['rps']:<6.0f} | {row['strategy']:<16} | {row['e2e_vio']:<10.2f} | {row['srv_vio']:<10.2f} | {row['avg_alloc']:<6.2f} | {row['density']:<6.2f} | {row['p90_e2e']:<10.2f} | {row['avg_srv']:<8.2f} | {row['avg_e2e']:<8.2f} | {row['avg_overhead']:<8.2f} | {row['achieved_success_rps']:<7.2f}")
        print("=================================================================\n")
