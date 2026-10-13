# CacheMind — Feedback-Guided Matrix Tiling Optimization

**NYU × Annapurna Labs · Hack the Chip · October 2026**

CacheMind is a verification-guided AI optimization agent. Qwen3-8B, served on AWS Trainium during the hackathon, proposes matrix-multiplication tile dimensions (`tile_m`, `tile_n`, `tile_k`). A deterministic Python checker validates configurations, verifies blocked multiplication against NumPy, and estimates a **synthetic** cost reflecting memory traffic, padded arithmetic, and iteration overhead.

> **Scope:** Trainium runs language-model inference. Candidate correctness checks and analytical cost evaluations run on the CPU. These results are *not* measured Trainium performance improvements.

## Results — CacheMind v3

Five trials per method, eight proposals per trial. Lower modeled cost is better.

| Strategy | Mean best synthetic cost | Mean round to best | Valid proposals |
|---|---:|---:|---:|
| Random search | 3,777,023.49 | 3.8 | 40/40 |
| Qwen, score-only feedback | 2,952,612.61 | 3.6 | 40/40 |
| **Qwen, diagnostic feedback** | **2,710,585.86** | **5.2** | **40/40** |

Diagnostic feedback achieved a **28.2% lower mean modeled cost than random search** and **8.2% lower than score-only** in these trials. It **did not converge faster**: the mean round to best was 5.2, versus 3.6 for score-only. Several trial trajectories repeat, limiting generalizability.

## Architecture

```text
Qwen3-8B on AWS Trainium
    │ proposes {tile_m, tile_n, tile_k}
    ▼
Python checker on CPU
    ├── synthetic live-tile capacity constraint
    ├── blocked matrix multiplication checked against NumPy
    ├── padding and iteration overhead
    └── modeled optimization cost
    │
    ▼
Attempt ledger → model feedback → next proposal
```

The allowed tile dimensions are `8, 16, 32, 64`. The synthetic capacity is **6,144 elements**, with live elements calculated as `m*k + k*n + m*n`. Four workload shapes are evaluated: square, skinny, ragged and wide.

The objective combines padded traffic, weighted padded multiply-accumulates and per-tile iteration overhead. These are engineering-model assumptions, not AWS Neuron hardware limits.

## Experimental controls

- **Random search:** samples valid unseen configurations.
- **Score-only:** receives each preceding configuration's status and numeric cost.
- **Diagnostic:** receives the same information plus utilization, worst padding and tile-iteration diagnostics.
- The two AI modes receive identical starting constraints, with one attempt counted for every proposal (including invalid or duplicate proposals).
- The final agent uses an attempt ledger and retains the best valid candidate.

## Running the project

Python 3.10+ and NumPy are required.

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_v3 test_v2 test_checker test_agent
python agent_v3.py --backend random --mode random --rounds 8 --repeat 5 --out results/local_random.json
```

For a compatible OpenAI-style Qwen endpoint, use `KERNEL_AGENT_BASE_URL` and `KERNEL_AGENT_MODEL` or the corresponding CLI overrides:

```bash
python agent_v3.py --backend llm --mode diagnostic --rounds 3 --repeat 1 --out results/smoke.json
python agent_v3.py --backend llm --mode all --rounds 8 --repeat 5 --out results/comparison.json
```

## Important files

| File | Purpose |
|---|---|
| `agent_v3.py` | Final controlled experiment runner |
| `checker_v2.py` | Final checker |
| `workloads.py` | Workload specifications |
| `test_v3.py`, `test_v2.py`, `test_checker.py`, `test_agent.py` | Automated tests |
| `results/v3_comparison.json`, `results/v3_comparison.log` | Original experiment evidence |
| `results/v3_feedback_test.json` | Post-fix feedback-label verification |
| `HACKATHON_REPORT.md` | Experimental report |

The original v3 log contains a *display-label-only* error in score mode: some valid candidates show “Rejected (score 0)”. The final source corrects this label; the score-only model used status and proxy cost independently. The historical log has intentionally not been rewritten.

## Limitations

The search space is deliberately small; exhaustive search remains an important missing baseline. The cost is an uncalibrated analytical proxy. Five partly repetitive trials are preliminary evidence rather than a statistically established advantage. Future work includes held-out workloads and on-device NKI benchmarking.

Developed for **Hack the Chip — NYU × Annapurna Labs**.
