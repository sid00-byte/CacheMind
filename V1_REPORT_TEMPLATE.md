# CacheMind — One-Page Experiment Report

**Team:** ______  **Date:** ______  **Hardware / serving:** ______

## Research question
Does structured, checker-generated memory-use feedback improve Qwen3-8B's selection of tiling parameters compared with score-only feedback and random search, at a fixed attempt budget?

## Method
Three matrix multiplication shapes: 64×64×64, 96×128×48, 70×93×55. Candidates use tile dimensions from {8,16,32,64} with synthetic limit `m*k + k*n + m*n <= 8192` tile elements. Correctness: actual NumPy blocked multiply versus NumPy dense multiply (rtol=1e-3, atol=1e-3). Optimization metric: analytic float32 input-reload + output-write traffic proxy, **not measured Trainium bandwidth**. Modes: random search, score-only agent, diagnostic agent. Each trial: ___ attempts, ___ repeats, model temperature ___, context ___, model ___, seed protocol ___.

## Results (fill from all.json)
| Condition | Trials | Valid-response rate | Successful trials | Best proxy bytes (median [min,max]) |
|---|---:|---:|---:|---:|
| Random | | | | |
| Score-only | | | | |
| Diagnostic | | | | |

**Illustrative diagnostic and resulting repair (from actual logs):** ______

## Conclusion
Observed improvement or failure: ______. Alternative explanations and sample variability: ______.

**Limitations:** proxy omits Neuron memory hierarchy, compiler effects, and end-to-end latency; only a small discrete configuration space; checker feedback may be imperfect; no hardware kernel measurements unless separately performed and documented.
