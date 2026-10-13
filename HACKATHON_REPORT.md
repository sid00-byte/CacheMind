# CacheMind: Checker-Guided Matrix Tiling Optimization

**Hack the Chip — NYU × Annapurna Labs | October 2026**

**Platform:** Qwen3-8B inference on AWS Trainium; deterministic Python/NumPy checker on CPU.

## Research question

Can structured checker diagnostics help an 8B language model find lower-cost matrix-multiplication tiling configurations than numeric feedback alone, under the same proposal budget?

## Implementation and checker

Qwen proposes integer tile dimensions `(tile_m, tile_n, tile_k)` from `{8,16,32,64}`. A Python checker rejects unsupported dimensions, duplicates, and configurations exceeding a **synthetic 6,144-element live-tile budget**, calculated as `m*k + k*n + m*n`. Valid candidates use blocked matrix multiplication verified against NumPy on square, skinny, ragged, and wide workloads. The analytical objective combines padded input/output traffic, weighted multiply-accumulates, and tile-loop overhead. The checker reports cost, memory usage, worst-case padding, and iteration counts.

The agent logs every proposal and keeps the best valid configuration. No hidden repair calls occur in v3.

## Experimental design and results

Three methods each received **five trials × eight proposals**: random search over unseen valid configurations; Qwen receiving scores and status; and Qwen receiving those values plus detailed diagnostics.

| Method | Mean best modeled cost ↓ | Mean round to best | Valid / attempted |
|---|---:|---:|---:|
| Random | 3,777,023.49 | 3.8 | 40/40 |
| Score-only | 2,952,612.61 | 3.6 | 40/40 |
| **Diagnostic** | **2,710,585.86** | **5.2** | **40/40** |

Diagnostic feedback yielded **28.2% lower mean modeled cost than random search** and **8.2% lower than score-only** in the observed trials, but needed more rounds to reach its best result. Its best observed configuration was `(64,64,16)`, at **2,589,572.48** synthetic cost units.

## Interpretation and limitations

This demonstrates a working model–checker–feedback loop and a preliminary advantage in modeled solution quality. Some model trajectories were repeated across seeds, reducing trial independence. Random search was restricted to valid candidates. The cost objective and capacity are modeling assumptions, **not Neuron hardware specifications or measured accelerator latency**. The original v3 log includes an incorrect score-mode display label for some valid candidates; the source and tests were subsequently corrected without changing the historical experiment records.

**Artifacts:** `agent_v3.py`, `checker_v2.py`, `test_v3.py`, `results/v3_comparison.json`, `results/v3_comparison.log`, and `results/v3_feedback_test.json`.
