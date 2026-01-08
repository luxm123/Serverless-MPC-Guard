from src.mpc.controller import MPCController
from src.wcp.wcp_lite_a import wcp_lite_a_update

def run_replay(metrics_list, alpha=0.1, vec_weights=None, price_bands=None, print_csv=False):
    c = MPCController()
    wcp_state = {}
    system_state = {}
    if isinstance(vec_weights, dict):
        system_state['sp_vec_weights'] = vec_weights
    if isinstance(price_bands, list):
        system_state['price_bands'] = price_bands
    rows = []
    step = 0
    for m in metrics_list:
        pred, unc, dbg = wcp_lite_a_update(wcp_state, m, alpha=alpha)
        wcp = {'pred': pred, 'uncertainty': unc}
        res = c.decide({'priority': m.get('priority', 'standard'), 'consistency': m.get('consistency', 'eventual')}, wcp, system_state)
        row = {
            'step': step,
            'cpu': float(m.get('cpu_util', 0.0)),
            'queue': float(m.get('queue_backlog', 0.0)),
            'timeout': float(m.get('timeout_rate', 0.0)),
            'error': float(m.get('error_rate', 0.0)),
            'mem': float(m.get('memory_pressure', 0.0)),
            'p90': float(m.get('p90', 0.0)),
            'price': float(system_state.get('shadow_price', 0.0)),
            'alloc': float(res['decision']['resource_alloc']),
            'shed': bool(res['decision']['should_shed']),
            'plan': res['decision']['degrade_plan'] or ''
        }
        rows.append(row)
        step += 1
    if print_csv:
        print('step,cpu,queue,timeout,error,mem,p90,price,alloc,shed,plan')
        for r in rows:
            print(f"{r['step']},{r['cpu']},{r['queue']},{r['timeout']},{r['error']},{r['mem']},{r['p90']},{r['price']},{r['alloc']},{int(r['shed'])},{r['plan']}")
    return rows
