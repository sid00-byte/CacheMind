# CacheMind — Feedback-Guided Matrix Tiling Optimization

**NYU × Annapurna Labs · Hack the Chip · October 2026**

CacheMind is a verification-guided optimization agent. **Qwen3-8B**, served on AWS Trainium during the hackathon, proposes matrix-multiplication tile dimensions (`tile_m`, `tile_n`, `tile_k`). A deterministic Python checker validates candidate configurations, verifies blocked matrix multiplication against NumPy, and estimates a **synthetic** cost for memory traffic, padded arithmetic, and tiled-loop overhead. Feedback is returned to the agent, which proposes its next candidate without changing model weights.

> **Scope:** Trainium ran *model inference*. The checker and blocked matrix operations ran on the **CPU**. CacheMind does **not** compile/benchmark NKI kernels, measure Trainium memory traffic, or establish accelerator speedups.

## Results: controlled v3 experiment

Five trials each, eight proposals per trial (40 candidate slots per method). Lower synthetic cost is better.

| Method | Mean best synthetic cost | Mean round to best | Valid proposals |
|---|---:|---:|---:|
| Random search | 3,777,023.49 | 3.8 | 40/40 |
| Qwen, score-only | 2,952,612.61 | 3.6 | 40/40 |
| **Qwen, diagnostic feedback** | **2,710,585.86** | **5.2** | **40/40** |

Diagnostic feedback achieved **28.2% lower average modeled cost than random search** and **8.2% lower than score-only** in these trials. This does **not** show faster convergence: its average best result occurred later (round 5.2 vs 3.6 for score-only). Several AI trajectories were repeated, limiting trial independence and generalizability. Results are preliminary, and the proxy is not a calibrated hardware performance model.

Full original records: [`results/v3_comparison.json`](results/v3_comparison.json) and [`results/v3_comparison.log`](results/v3_comparison.log). The **original log predates a feedback-label-only fix**: in score mode, some `status=VALID` entries were printed as `Rejected (score 0).`. That label was incorrect; v3's LLM prompt uses the recorded `status` and `proxy_cost`, not that score-mode label. See [`results/v3_feedback_test.json`](results/v3_feedback_test.json) for the later verification. Original evidence is deliberately unchanged.

## How it works

```text
Qwen3-8B (Trainium inference) -> {tile_m, tile_n, tile_k}
       -> Python checker (CPU): capacity + correctness + synthetic cost
       -> attempt ledger: score, status, diagnostic information
       -> Qwen3-8B proposes a new candidate (8-attempt budget)
```

The synthetic memory capacity is **6,144 float elements**, based on `m*k + k*n + m*n`. Permitted tile dimensions are `8, 16, 32, 64`. The checker covers four workload shapes: square `(128,128,128)`, skinny `(48,256,160)`, ragged `(73,119,65)`, and wide `(128,40,257)`; tuples are `(M,K,N)`.

For each workload, the cost is **padded input and output bytes + 0.04 × padded multiply-accumulates + 5,000 × tile iterations**, summed across workloads. These weights and the capacity are chosen modeling assumptions, **not Neuron hardware specifications**. See [`checker_v2.py`](checker_v2.py) for exact calculations and validity criteria.

### Comparison modes

- **Random:** chooses unseen valid configurations, providing a baseline. Because its samples are prefiltered for validity, do not interpret its validity rate as evidence of superior feedback.
- **Score-only:** Qwen sees prior configurations, validity status, and the synthetic cost (or zero on failure).
- **Diagnostic:** Qwen sees the same plus checker diagnostics (capacity, padding ratio, and tile iteration count).

Both AI modes receive identical starting constraints. Every proposed candidate counts toward the budget; no hidden repair requests. Invalid, duplicate, and model-error proposals are logged. Diagnostic prompts are longer and carry additional information by design.

## Run locally (CPU tests / random baseline)

Requires Python 3.10+ (recommended) and NumPy.

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_v3 test_v2 test_checker test_agent
python agent_v3.py --backend random --mode random --rounds 8 --repeat 5 --out results/local_random.json
```

## Run with Qwen3-8B on a workshop Trainium pod

With the workshop's model already running on its OpenAI-compatible endpoint:

```bash
cd /workspace/projects/03-cachemind
python agent_v3.py --backend llm --mode diagnostic --rounds 3 --repeat 1 --out results/v3_smoke_new.json
nohup python agent_v3.py --backend llm --mode all --rounds 8 --repeat 5 \
  --out results/v3_comparison_new.json > results/v3_comparison_new.log 2>&1 < /dev/null &
tail -f results/v3_comparison_new.log
```

By default the client reads `KERNEL_AGENT_BASE_URL` (fallback `http://localhost:8000/v1`) and `KERNEL_AGENT_MODEL` (fallback `Qwen/Qwen3-8B`). Override them using `--base-url` and `--model`. For reproducibility, preserve the original result files rather than overwriting them.

## Repository guide

| Path | Role |
|---|---|
| `agent_v3.py` | **Final agent**, matched-budget, logged experiment |
| `checker_v2.py` | **Final checker**, validity, NumPy comparison, and proxy cost |
| `workloads.py`, `checker.py`, `agent.py` | Workloads and supporting/v1 implementation; v3 reuses types and model client |
| `test_v3.py`, `test_v2.py`, `test_checker.py`, `test_agent.py` | 16 automated tests |
| `results/v3_comparison.json`, `.log` | Original hackathon experiment evidence |
| `results/v3_feedback_test.json` | Verification after feedback-label correction |
| `HACKATHON_REPORT.md` | One-page experimental report |
| `V2_README.md`, `V3_README.md`, `V1_REPORT_TEMPLATE.md`, `agent_v2.py` | Development history |

## Known limitations and future directions

The candidate space is small and bounded; exhaustive search is a useful missing baseline. The findings do not validate real Neuron speedups. Trial sequences are partially deterministic despite changing seeds. The model may identify lower modeled costs without learning a reusable policy; more workloads, diversified sampling, and on-device NKI measurements would be needed to evaluate generalization.

## Attribution

Original work developed during the NYU × Annapurna Labs Hack the Chip workshop. The code here is presented as the CacheMind team project; workshop-provided model weights and external dependencies retain their respective licenses. No license for this repository is asserted until the contributors choose one and confirm the provenance of any reused material.
