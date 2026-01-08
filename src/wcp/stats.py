import math

def weighted_quantile(values, weights, q, inf_weight=0.0):
    if not values: return 0.0
    if len(values) != len(weights): return 0.0
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(weights) + inf_weight
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc / total_w >= q:
            return v
    return pairs[-1][0]
