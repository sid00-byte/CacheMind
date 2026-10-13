"""CacheMind v3: matched-budget diagnostic-feedback experiment.

Each round consumes exactly one model request (or one random candidate).
No silent retries. Every proposal and rejection is recorded. Proxy scores are
synthetic and do not represent Trainium latency.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from agent import query_model
from checker import Config
from checker_v2 import BUDGET, TILES, WORKLOADS_V2, evaluate_v2, validate_v2

LEGAL = [Config(m, n, k) for m in TILES for n in TILES for k in TILES
         if validate_v2(Config(m, n, k)) is None]

BASE_SYSTEM = (
    'You optimize a SYNTHETIC matrix-tiling benchmark. '
    f'tile_m,tile_n,tile_k must each be from {list(TILES)}. '
    f'Working set = tile_m*tile_k + tile_k*tile_n + tile_m*tile_n; must be <= {BUDGET}. '
    'Goal: minimize summed proxy cost across workloads. '
    'The proxy includes padded data traffic, padded operations, and tile-loop overhead. '
    'Reply ONLY one JSON object: {"tile_m":32,"tile_n":32,"tile_k":32}. '
    'Do not repeat any previous configuration.'
)


def messages(history, mode):
    assert mode in ('score', 'diagnostic')
    lines = ['Workloads (M,K,N): ' + ', '.join(
        f'{w.name}=({w.m},{w.k},{w.n})' for w in WORKLOADS_V2)]
    if history:
        lines.append('Attempt ledger, oldest to newest:')
    for h in history:
        line = f"round {h['round']}: proposed={h['proposed']} status={h['status']}"
        if h['cost'] is not None:
            line += f" proxy_cost={h['cost']}"
        else:
            line += ' proxy_score=0'
        if mode == 'diagnostic':
            line += f" diagnosis={h['feedback']}"
        lines.append(line)
    lines.append('Propose a NEW valid configuration different from every proposal above.')
    return [{'role': 'system', 'content': BASE_SYSTEM},
            {'role': 'user', 'content': '\n'.join(lines)}]


def rejection(mode, status, detail):
    # Keep all problem constraints identical; only per-attempt feedback changes.
    if mode == 'diagnostic':
        return detail
    if status == 'VALID':
        return 'Valid configuration; numerical cost recorded in attempt ledger.'
    return 'Invalid proposal (score 0).'


def run(mode, seed, rounds, url, model, timeout, temperature, query_fn=None):
    assert mode in ('random', 'score', 'diagnostic')
    rng = random.Random(seed)
    seen = set()
    attempts = []
    best = None
    query_fn = query_fn or query_model
    for i in range(1, rounds + 1):
        start = time.monotonic()
        cfg = None
        model_error = None
        if mode == 'random':
            remaining = [c for c in LEGAL if c not in seen]
            if not remaining:
                break
            cfg = rng.choice(remaining)
        else:
            try:
                cfg = query_fn(messages(attempts, mode), url, model, timeout, temperature)
            except Exception as exc:
                model_error = f'{type(exc).__name__}: {str(exc)[:240]}'
        proposed = cfg.to_dict() if cfg is not None else None
        if cfg is None:
            status = 'MODEL_ERROR'
            cost = None
            diagnostic = model_error or 'Missing model proposal'
        elif cfg in seen:
            status = 'DUPLICATE'
            cost = None
            diagnostic = 'DUPLICATE: this exact tile configuration was already tried; select a different triple.'
        else:
            seen.add(cfg)  # Record even invalid proposals to prevent repeated failure.
            err = validate_v2(cfg)
            if err is not None:
                status = 'INVALID'
                cost = None
                diagnostic = err
            else:
                result = evaluate_v2(cfg)
                if result['correct']:
                    status = 'VALID'
                    cost = result['aggregate_cost']
                    diagnostic = result['reason']
                    if best is None or cost < best['cost']:
                        best = {'round': i, 'config': proposed, 'cost': cost}
                else:
                    status = 'WRONG_RESULT'
                    cost = None
                    diagnostic = result['reason']
        entry = {'round': i, 'proposed': proposed, 'status': status,
                 'cost': cost, 'feedback': rejection(mode, status, diagnostic),
                 'checker_detail': diagnostic, 'seconds': round(time.monotonic()-start, 3)}
        attempts.append(entry)
        print(f'[{mode} seed={seed} round={i}] proposed={proposed} status={status} '
              f'cost={cost} feedback={entry["feedback"]}', flush=True)
    return {'mode': mode, 'seed': seed, 'attempts': attempts,
            'best': best, 'best_cost': best['cost'] if best else None,
            'valid_proposals': sum(a['status'] == 'VALID' for a in attempts),
            'invalid_proposals': sum(a['status'] != 'VALID' for a in attempts)}


def summarize(trials, modes):
    out = {}
    for mode in modes:
        t = [x for x in trials if x['mode'] == mode]
        all_costs = [x['best_cost'] for x in t]
        success_costs = [v for v in all_costs if v is not None]
        # Failures stay in the success-rate denominator; do not hide them in the mean.
        out[mode] = {'trials': len(t), 'trials_with_valid_candidate': len(success_costs),
                     'trial_best_cost': all_costs,
                     'mean_best_cost_successful_only': (sum(success_costs)/len(success_costs)
                                                        if success_costs else None),
                     'mean_round_to_best_successful_only': (sum(x['best']['round'] for x in t if x['best']) /
                                                           len(success_costs) if success_costs else None),
                     'valid_proposals': sum(x['valid_proposals'] for x in t),
                     'invalid_or_duplicate_or_error_proposals': sum(x['invalid_proposals'] for x in t),
                     'total_evaluation_slots': sum(len(x['attempts']) for x in t)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['all', 'random', 'score', 'diagnostic'], default='all')
    p.add_argument('--backend', choices=['llm', 'random'], default='llm')
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--base-url', default=os.getenv('KERNEL_AGENT_BASE_URL', 'http://localhost:8000/v1'))
    p.add_argument('--model', default=os.getenv('KERNEL_AGENT_MODEL', 'Qwen/Qwen3-8B'))
    p.add_argument('--timeout', type=float, default=90)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--out', default='results/v3_comparison.json')
    args = p.parse_args()
    if args.rounds < 1 or args.repeat < 1:
        p.error('rounds and repeat must be positive')
    modes = ['random', 'score', 'diagnostic'] if args.mode == 'all' else [args.mode]
    if args.backend == 'random' and any(m != 'random' for m in modes):
        p.error('AI modes require --backend llm')
    trials = []
    for mode in modes:
        for j in range(args.repeat):
            trials.append(run(mode, 100+j, args.rounds, args.base_url, args.model,
                              args.timeout, args.temperature))
    summary = summarize(trials, modes)
    payload = {'metadata': {'model': args.model if args.backend == 'llm' else None,
                            'temperature': args.temperature, 'rounds': args.rounds,
                            'repeat': args.repeat, 'budget_type': 'one candidate per round; '
                            'one LLM call per AI round, no retries',
                            'all_constraints_in_both_prompts': True,
                            'evaluation_is_synthetic_not_trainium_latency': True},
               'summary': summary, 'trials': trials}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print('SUMMARY\n' + json.dumps(summary, indent=2), flush=True)
    print('Wrote', path, flush=True)


if __name__ == '__main__':
    main()
