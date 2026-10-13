"""CacheMind v2: synthetic multi-objective proxy, NOT measured Neuron performance."""
from checker import Config, blocked_matmul
from workloads import matrices, Workload
import numpy as np
from math import ceil

TILES = (8, 16, 32, 64)
BUDGET = 6144
WORKLOADS_V2 = (
    Workload('square', 128, 128, 128),
    Workload('skinny', 48, 256, 160),
    Workload('ragged', 73, 119, 65),
    Workload('wide', 128, 40, 257),
)

def capacity(c):
    return c.tile_m*c.tile_k + c.tile_k*c.tile_n + c.tile_m*c.tile_n

def validate_v2(c):
    for x in (c.tile_m,c.tile_n,c.tile_k):
        if x not in TILES:
            return f'Unsupported tile size {x}; allowed {TILES}'
    if capacity(c) > BUDGET:
        return f'MEMORY_CAPACITY: {capacity(c)} > {BUDGET} elements; shrink one tile dimension.'
    return None

def metrics(w,c):
    tm,tn,tk = c.tile_m,c.tile_n,c.tile_k
    bm,bn,bk=ceil(w.m/tm),ceil(w.n/tn),ceil(w.k/tk)
    # Explicit synthetic model: padded tile loads + writes, tiled MAC cost,
    # software per-reduction-loop overhead. These are model assumptions.
    input_bytes=4*bm*bn*bk*(tm*tk+tk*tn)
    output_bytes=4*bm*bn*tm*tn
    padded_macs=bm*bn*bk*tm*tn*tk
    useful_macs=w.m*w.n*w.k
    padding_ratio=padded_macs/useful_macs
    block_overhead=5000*bm*bn*bk  # synthetic equivalent cost units
    cost=(input_bytes+output_bytes)+0.04*padded_macs+block_overhead
    return {'workload':w.name,'shape':[w.m,w.k,w.n], 'input_bytes':input_bytes,
            'output_bytes':output_bytes,'padded_macs':padded_macs,
            'useful_macs':useful_macs,'padding_ratio':round(padding_ratio,4),
            'blocks':bm*bn*bk,'proxy_cost':round(cost,3)}

def evaluate_v2(c, *, verify=True):
    bad=validate_v2(c)
    if bad:
        return {'valid':False,'correct':False,'score':0.,'aggregate_cost':None,'reason':bad,'per_workload':[]}
    per=[metrics(w,c) for w in WORKLOADS_V2]
    if verify:
        for w,p in zip(WORKLOADS_V2,per):
            a,b=matrices(w)
            actual=blocked_matmul(a,b,c)
            expected=a@b
            p['correct']=bool(np.allclose(actual,expected,rtol=1e-3,atol=1e-3))
            p['max_abs_error']=float(np.max(np.abs(actual-expected)))
    else:
        for p in per:p['correct']=True
    passed=all(p['correct'] for p in per)
    cost=sum(p['proxy_cost'] for p in per)
    worst_pad=max(per,key=lambda p:p['padding_ratio'])
    worst_blocks=max(per,key=lambda p:p['blocks'])
    reason=(f'VALID: live={capacity(c)}/{BUDGET}; worst padded MAC ratio={worst_pad["padding_ratio"]} '
            f'on {worst_pad["workload"]}; most tile iterations={worst_blocks["blocks"]} '
            f'on {worst_blocks["workload"]}. Compare boundary padding and tile-count '
            'overhead; changing K changes both padded work and loop count.')
    return {'valid':True,'correct':passed,'score':round(1e6/cost,7) if passed else 0.,
            'aggregate_cost':cost if passed else None,
            'reason':reason if passed else 'NUMERIC_MISMATCH: inspect per-workload errors',
            'per_workload':per}
