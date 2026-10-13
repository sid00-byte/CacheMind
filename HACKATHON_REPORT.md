# CacheMind: Checker-Guided Matrix Tiling Optimization
**Hack the Chip — NYU × Annapurna Labs | October 2026**  
**System:** Qwen3-8B inference on AWS Trainium; Python/NumPy checker on CPU

### Question
Can specific checker diagnostics help an 8B language model find lower-cost matrix-multiplication tile configurations than numeric feedback alone, under the same proposal budget?

### Checker and agent
The model proposes integer tile dimensions `(tile_m, tile_n, tile_k)` drawn from `{8,16,32,64}`. A deterministic checker rejects unsupported sizes, duplicates, and configurations exceeding a **synthetic 6,144-element** live-tile budget (`m*k + k*n + m*n`). Valid configurations execute trusted blocked matrix multiplication on four workloads (square, skinny, ragged, wide), with outputs compared against NumPy. A synthetic objective sums padded input/output traffic, weighted padded multiply-accumulates, and per-tile-loop overhead. Detailed feedback includes memory use, worst-case padding, and tile iteration counts. Every proposal—including a rejected or repeated proposal—consumes one attempt. The agent retains the best valid candidate.

### Experiment
Three methods, **five trials each**, **eight candidate slots per trial**: (1) random search over unseen legal configurations; (2) Qwen with numeric cost/status history; (3) Qwen with the same history plus structured checker diagnostics. Both AI modes receive the same initial constraints. No hidden repair requests are made. Each attempted configuration and checker response is recorded in the original experiment JSON and log.

| Method | Mean best modeled cost ↓ | Mean round to best | Valid/attempted |
|---|---:|---:|---:|
| Random | 3,777,023.49 | 3.8 | 40/40 |
| Score-only | 2,952,612.61 | 3.6 | 40/40 |
| **Diagnostic** | **2,710,585.86** | **5.2** | **40/40** |

The diagnostic method achieved **28.2% lower modeled cost than random search** and **8.2% lower modeled cost than score-only** on average, but took more attempts to reach its best configuration. The best observed diagnostic configuration `(64,64,16)` scored **2,589,572.48** modeled cost units. The initial v3 logs contain a misleading score-mode display label (“Rejected (score 0)”) even for `VALID` candidates; source and tests were subsequently corrected. The numeric status/cost feedback used in the score-only model prompt was unaffected. Original records are preserved rather than rewritten.

### Interpretation and limitations
The experiment demonstrates a working model → checker → feedback loop and an **observed**, not statistically established, diagnostic-feedback advantage. Several repeated trajectories limit the independence of five trials. Random search was restricted to valid candidates. The objective, weights, and capacity are **analytical assumptions**, not AWS Neuron limits or calibrated hardware costs; candidate kernels were **not benchmarked on Trainium**. Next steps: test a broader held-out workload suite, compare against exhaustive search, improve diagnostic specificity, and measure selected kernels on-device if available.

**Artifacts:** `checker_v2.py`, `agent_v3.py`, `test_v3.py`; `v3_comparison.json`, `v3_comparison.log`; `v3_feedback_test.json`.
