"""Deterministic matrix multiplication evaluation workloads."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Workload:
    name: str
    m: int
    k: int
    n: int

WORKLOADS = (
    Workload('square_small', 64, 64, 64),
    Workload('rectangular', 96, 128, 48),
    Workload('irregular', 70, 93, 55),
)

def matrices(w: Workload, seed: int = 7):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(w.m, w.k)).astype(np.float32),
            rng.normal(size=(w.k, w.n)).astype(np.float32))
