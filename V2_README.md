# CacheMind v2 — constraint-aware experiment

Additive extension: original `agent.py` and `checker.py` are unchanged. Version 2 uses `agent_v2.py`, `checker_v2.py`, and `test_v2.py`.

## What's new

- Four heterogeneous matrix workloads (square, skinny, ragged, wide).
- Synthetic capacity budget 6144 elements; valid tiles drawn from 8,16,32,64.
- Synthetic cost combines *padded* input/output bytes, padded multiply-accumulate work, and a per-block overhead; `tile_k` now matters.
- Feedback identifies capacity failures and worst padding/iteration workload.
- Duplicate/invalid candidate rejection **before checker evaluation** with at most 2 repair requests. Each failed repair is logged; report extra model calls for honest comparison.
- Random, score-only, and diagnostic methods each receive exactly 8 **evaluated-candidate slots** per trial, although AI may consume more model requests repairing proposals. This is not a matched *LLM inference budget* experiment. If desired, also compare total requests, time and token use.

## Run in your existing pod

```bash
cd /workspace/projects/03-cachemind
python -m unittest -v test_v2 test_checker test_agent
python agent_v2.py --mode random --rounds 8 --repeat 5 --out v2_random.json
python agent_v2.py --backend llm --mode score --rounds 3 --repeat 1 --out v2_smoke.json
nohup python agent_v2.py --backend llm --mode all --rounds 8 --repeat 5 --out v2_comparison.json > v2_comparison.log 2>&1 < /dev/null &
tail -f v2_comparison.log
```

## Scientific caveats

This is an analytical *toy model*, not true NKI compilation or measured Trainium latency. Score is `1_000_000 / synthetic_cost` and is not a physical performance metric. Shapes and synthetic weights were designed for this prototype, not calibrated to Neuron hardware. Four matrix shapes are used jointly to evaluate each candidate, not held out. All measures of improvement must be reported as **proxy** improvements. `get_model_cfg` retains the existing Qwen-compatible parser from `agent.py`.

## Further extension

Use a verified NKI kernel template and Neuron runtime benchmark to measure actual execution latency; consult event engineers on core isolation and permitted benchmarking. Do not claim hardware speedups from this simulator.
