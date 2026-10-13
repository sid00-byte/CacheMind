# CacheMind v3 — controlled checker-feedback experiment

This version adds `agent_v3.py` and `test_v3.py` without changing v1/v2 artifacts.

## What changed

- **Identical initial constraints** for score and diagnostic agents.
- **One model request per round**: no invisible retries. Rejections consume budget.
- **Complete attempt ledger**: stores every proposal, including invalid, duplicate, and unparseable responses.
- **Treatment difference**: score gets only rejection / score; diagnostic gets the checker's reason. Both see candidate histories and problem constraints.
- **Transparent summary**: valid and rejected proposal counts, best synthetic proxy cost, and round of best.

## In your Trainium pod

```bash
cd /workspace/projects/03-cachemind
python -m unittest -v test_v3 test_v2 test_checker test_agent
python agent_v3.py --backend llm --mode diagnostic --rounds 3 --repeat 1 --out results/v3_smoke.json
nohup python agent_v3.py --backend llm --mode all --rounds 8 --repeat 5 --out results/v3_comparison.json > results/v3_comparison.log 2>&1 < /dev/null &
tail -f results/v3_comparison.log
```

After completion:

```bash
python - <<'PY'
import json
with open('results/v3_comparison.json') as f: d=json.load(f)
print(json.dumps(d['summary'], indent=2))
PY
```

## Interpretation and caveats

- The cost is a **synthetic proxy**, not actual Neuron memory traffic, NKI performance, or Trainium hardware latency.
- A `VALID` result means a candidate passed the implemented checker; it does not prove hardware legality.
- Current trials may remain deterministic even when different nominal random seeds are passed; report exact per-trial results.
- The two AI arms consume the same number of LLM requests, but rejection modes naturally have different prompt lengths. Model response sampling may vary.
- The score-only arm still knows the complete problem specification. Diagnostic messages are the intervention.
- Random search samples only legal configurations. This makes random stronger at validity, so compare solution quality and AI invalid-proposal rates separately rather than conflating validity success.
- If neither approach improves over another, that is a legitimate result.
