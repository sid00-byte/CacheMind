"""Verifiable blocked matrix multiply plus transparent memory-traffic proxy.

This is a CPU simulation, NOT an NKI compiler, hardware measurement, or a
cycle-accurate model of AWS Trainium memory hierarchy.
"""
from dataclasses import dataclass, asdict
from math import ceil
import numpy as np
from workloads import WORKLOADS, matrices

TILES = (8, 16, 32, 64)
MAX_LIVE_ELEMENTS = 8192  # explicit synthetic constraint, not a hardware limit

@dataclass(frozen=True)
class Config:
    tile_m: int
    tile_n: int
    tile_k: int

    @classmethod
    def from_dict(cls, obj):
        if not isinstance(obj, dict):
            raise ValueError('Configuration must be a JSON object')
        try:
            return cls(*(int(obj[k]) for k in ('tile_m', 'tile_n', 'tile_k')))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('Configuration requires integer tile_m, tile_n, tile_k') from exc

    def to_dict(self):
        return asdict(self)


def validate(cfg):
    for key, value in cfg.to_dict().items():
        if value not in TILES:
            return f'{key}={value} is invalid; allowed values are {list(TILES)}'
    live = cfg.tile_m * cfg.tile_k + cfg.tile_k * cfg.tile_n + cfg.tile_m * cfg.tile_n
    if live > MAX_LIVE_ELEMENTS:
        return (f'Synthetic working-set limit exceeded: {live} elements > '
                f'{MAX_LIVE_ELEMENTS}; reduce one or more tile dimensions')
    return None


def blocked_matmul(a, b, cfg):
    """CPU reference tiling implementation, using NumPy for block arithmetic."""
    m, k = a.shape
    kb, n = b.shape
    if kb != k:
        raise ValueError('Matrix dimensions do not match')
    c = np.zeros((m, n), dtype=np.float32)
    for i in range(0, m, cfg.tile_m):
        for j in range(0, n, cfg.tile_n):
            out = c[i:i+cfg.tile_m, j:j+cfg.tile_n]
            for p in range(0, k, cfg.tile_k):
                out += a[i:i+cfg.tile_m, p:p+cfg.tile_k] @ b[p:p+cfg.tile_k, j:j+cfg.tile_n]
    return c


def traffic_bytes(m, k, n, cfg):
    """Explicit proxy: each (i,j,p) iteration loads both operand tiles once;
    each output tile is written once. Assumes NO cross-tile caching, ignores
    storage of partial sums, alignment, transactions, and Neuron architecture.
    """
    words = 0
    for i in range(0, m, cfg.tile_m):
        im = min(cfg.tile_m, m - i)
        for j in range(0, n, cfg.tile_n):
            jn = min(cfg.tile_n, n - j)
            for p in range(0, k, cfg.tile_k):
                pk = min(cfg.tile_k, k - p)
                words += im * pk + pk * jn
            words += im * jn
    return words * 4  # float32


def diagnose(cfg, workloads, best_bytes=None):
    message = []
    live = cfg.tile_m*cfg.tile_k + cfg.tile_k*cfg.tile_n + cfg.tile_m*cfg.tile_n
    message.append(f'Estimated live tile elements: {live}/{MAX_LIVE_ELEMENTS}.')
    message.append('In this proxy, increasing tile_m reduces how often B is loaded, '
                   'while increasing tile_n reduces how often A is loaded.')
    if cfg.tile_m < 64 and cfg.tile_n < 64:
        message.append('Try a larger tile_m or tile_n within the synthetic working-set limit.')
    elif cfg.tile_m < 64:
        message.append('Try increasing tile_m; check working-set constraints first.')
    elif cfg.tile_n < 64:
        message.append('Try increasing tile_n; check working-set constraints first.')
    else:
        message.append('Large spatial tiles are already selected; explore tile_k or retain this candidate.')
    message.append('tile_k may change intermediate work but does not change this simplified traffic estimate.')
    if best_bytes is not None:
        message.append(f'Best previous aggregate proxy traffic: {best_bytes} bytes.')
    return ' '.join(message)


def evaluate(cfg, workset=WORKLOADS, *, correctness=True):
    err = validate(cfg)
    if err:
        return {'valid': False, 'correct': False, 'score': 0.0,
                'aggregate_bytes': None, 'reason': err, 'per_workload': []}
    per = []
    for w in workset:
        transferred = traffic_bytes(w.m, w.k, w.n, cfg)
        okay = True
        maximum_error = 0.0
        if correctness:
            a, b = matrices(w)
            actual = blocked_matmul(a, b, cfg)
            expected = a @ b
            okay = bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-3))
            maximum_error = float(np.max(np.abs(actual - expected)))
        per.append({'workload': w.name, 'shape': [w.m, w.k, w.n],
                    'proxy_bytes': transferred, 'correct': okay,
                    'max_abs_error': maximum_error})
    aggregate = sum(w['proxy_bytes'] for w in per)
    passed = all(w['correct'] for w in per)
    return {'valid': True, 'correct': passed,
            'score': round(1_000_000 / aggregate, 6) if passed else 0.0,
            'aggregate_bytes': aggregate,
            'reason': diagnose(cfg, workset) if passed else 'Incorrect result on one or more workloads; inspect per-workload errors.',
            'per_workload': per}
