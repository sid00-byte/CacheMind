"""CacheMind experiment: random, score-only, and diagnostic feedback.

LLM backend uses the OpenAI-compatible /chat/completions protocol exposed by
most workshop servers. Endpoint naming and auth can vary; configure via CLI/env.
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from checker import Config, TILES, evaluate, validate, MAX_LIVE_ELEMENTS


def all_configs():
    return [Config(m, n, k) for m in TILES for n in TILES for k in TILES
            if validate(Config(m, n, k)) is None]


def get_model_cfg(response):
    message = response['choices'][0]['message']
    content = message.get('content') or ''
    if isinstance(content, list):
        content = ''.join(x.get('text', '') for x in content if isinstance(x, dict))
    match = re.search(r'\{[^{}]*\}', content, flags=re.S)
    if not match:
        excerpt = str(content)[:300] or '[empty content]'
        reason = message.get('reasoning_content') or ''
        raise ValueError(f'Model returned no JSON configuration; content={excerpt!r}; reasoning_chars={len(reason)}')
    return Config.from_dict(json.loads(match.group()))


def query_model(messages, base_url, model, timeout, temperature):
    endpoint = base_url.rstrip('/')
    if endpoint.endswith('/chat/completions'):
        url = endpoint
    elif endpoint.endswith('/v1'):
        url = endpoint + '/chat/completions'
    else:
        url = endpoint + '/v1/chat/completions'
    payload = {'model': model, 'messages': messages, 'temperature': temperature,
               'max_tokens': 400,
               'chat_template_kwargs': {'enable_thinking': False}}
    headers = {'Content-Type': 'application/json'}
    if os.getenv('CACHEMIND_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['CACHEMIND_API_KEY']
    request = Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urlopen(request, timeout=timeout) as result:
            return get_model_cfg(json.load(result))
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f'LLM request or JSON parse failed: {exc}') from exc


def prompt(history, mode):
    system = ('You are choosing CPU-simulated tiled matrix-multiply parameters. '
              'Reply ONLY one JSON object with integer keys tile_m, tile_n, tile_k; '
              'each value must be 8,16,32,64. No prose. '
              f'The synthetic live-element budget is {MAX_LIVE_ELEMENTS}: '
              'tile_m*tile_k + tile_k*tile_n + tile_m*tile_n <= budget. '
              'Lower aggregate proxy traffic is better. The proxy counts all '
              'input tile loads without cross-tile caching, plus output writes. '
              'Do not repeat previous configurations.')
    user = 'Optimize across these shapes: (64,64,64), (96,128,48), (70,93,55).'
    if history:
        recent = history[-8:]
        lines = []
        for h in recent:
            result = h['result']
            line = (f"config={h['config']}; valid={result['valid']}; correct={result['correct']}; "
                    f"score={result['score']}; traffic_bytes={result['aggregate_bytes']}")
            if mode == 'diagnostic':
                line += '; diagnosis=' + result['reason']
            lines.append(line)
        user += '\nPast attempts:\n' + '\n'.join(lines)
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def run_once(mode, trials, seed, base_url, model, timeout, temperature):
    rng = random.Random(seed)
    configs = all_configs()
    history, seen = [], set()
    best = None
    for round_id in range(1, trials + 1):
        error = None
        started = time.monotonic()
        if mode == 'random':
            choices = [c for c in configs if tuple(c.to_dict().values()) not in seen]
            if not choices:
                break
            cfg = rng.choice(choices)
        else:
            try:
                cfg = query_model(prompt(history, mode), base_url, model, timeout, temperature)
            except RuntimeError as exc:
                error = str(exc)
                cfg = None
        if cfg is None:
            result = {'valid': False, 'correct': False, 'score': 0.0,
                      'aggregate_bytes': None, 'reason': error, 'per_workload': []}
            entry = {'round': round_id, 'config': None, 'result': result,
                     'seconds': round(time.monotonic()-started, 3)}
        else:
            signature = tuple(cfg.to_dict().values())
            if signature in seen:
                result = {'valid': False, 'correct': False, 'score': 0.0,
                          'aggregate_bytes': None,
                          'reason': 'Repeated configuration; propose an unseen candidate.',
                          'per_workload': []}
            else:
                result = evaluate(cfg)
                seen.add(signature)
            if (result['correct'] and
                (best is None or result['aggregate_bytes'] < best['result']['aggregate_bytes'])):
                best = {'round': round_id, 'config': cfg.to_dict(), 'result': result}
            entry = {'round': round_id, 'config': cfg.to_dict(), 'result': result,
                     'seconds': round(time.monotonic()-started, 3)}
        history.append(entry)
        print(f"[{mode} seed={seed} round={round_id}] cfg={entry['config']} "
              f"traffic={result['aggregate_bytes']} score={result['score']} "
              f"reason={result['reason']}", flush=True)
    return {'mode': mode, 'seed': seed, 'attempts': history,
            'best': best, 'best_proxy_bytes': best['result']['aggregate_bytes'] if best else None}


def main():
    parser = argparse.ArgumentParser(description='CacheMind: checker-driven tiling experiment')
    parser.add_argument('--mode', choices=['random', 'score', 'diagnostic', 'all'], default='random')
    parser.add_argument('--backend', choices=['random', 'llm'], default='random',
                        help='random runs without server; llm permits score/diagnostic modes')
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--base-url', default=(os.getenv('CACHEMIND_BASE_URL') or
                        os.getenv('KERNEL_AGENT_BASE_URL') or 'http://localhost:8000'))
    parser.add_argument('--model', default=(os.getenv('CACHEMIND_MODEL') or
                        os.getenv('KERNEL_AGENT_MODEL') or 'Qwen/Qwen3-8B'))
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--out', default='results/experiment.json')
    args = parser.parse_args()
    if args.rounds < 1 or args.repeat < 1:
        parser.error('--rounds and --repeat must be positive')
    modes = ['random', 'score', 'diagnostic'] if args.mode == 'all' else [args.mode]
    if args.backend == 'random' and any(m != 'random' for m in modes):
        parser.error('score and diagnostic need --backend llm and a working model endpoint')
    output = []
    for mode in modes:
        for seed in range(args.repeat):
            output.append(run_once(mode, args.rounds, seed + 100, args.base_url,
                                   args.model, args.timeout, args.temperature))
    by_mode = {}
    for mode in modes:
        values = [r['best_proxy_bytes'] for r in output if r['mode'] == mode]
        passing = [v for v in values if v is not None]
        by_mode[mode] = {'successful_trials': len(passing), 'total_trials': len(values),
                         'best_bytes_across_trials': min(passing) if passing else None,
                         'mean_best_bytes_successful_trials':
                             sum(passing)/len(passing) if passing else None,
                         'trial_best_bytes': values}
    payload = {'metadata': {'backend': args.backend, 'base_url': args.base_url if args.backend == 'llm' else None,
                            'model': args.model if args.backend == 'llm' else None,
                            'rounds': args.rounds, 'repeat': args.repeat,
                            'proxy_is_hardware_measurement': False},
               'summary': by_mode, 'trials': output}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print('\nSUMMARY\n' + json.dumps(by_mode, indent=2))
    print(f'Wrote {path}')

if __name__ == '__main__':
    main()
