"""Independent CacheMind v2 fair-budget experiment. Reuses original Qwen client."""
import argparse, json, os, random, time
from pathlib import Path
from checker import Config
from checker_v2 import TILES, BUDGET, WORKLOADS_V2, evaluate_v2, validate_v2
from agent import query_model

VALID=[Config(m,n,k) for m in TILES for n in TILES for k in TILES if not validate_v2(Config(m,n,k))]

def prompt(history, mode, excluded):
    system=(f'Optimize a SYNTHETIC tiled-matmul objective with tile_m,tile_n,tile_k in {TILES}. '
      f'Capacity: m*k+k*n+m*n <= {BUDGET}. Objective: minimum aggregate proxy cost including '
      'padded tile transfers, padded compute work, and tile-loop overhead. '
      'Tradeoffs differ across matrix shapes. Reply ONLY one JSON object with three integer fields '
      'tile_m, tile_n, tile_k. Do not use excluded configurations.')
    user='Workloads (M,K,N): '+', '.join(f'{w.name}=({w.m},{w.k},{w.n})' for w in WORKLOADS_V2)
    user+='\nExcluded configurations: '+json.dumps([c.to_dict() for c in excluded])
    for e in history[-7:]:
        r=e['result']
        user+=f"\nCandidate={e['config']} cost={r['aggregate_cost']} score={r['score']}"
        if mode=='diagnostic':user+=' diagnosis='+r['reason']
    return [{'role':'system','content':system},{'role':'user','content':user}]

def run(mode, seed, rounds, url, model, timeout, temperature, max_repairs):
    rng=random.Random(seed); history=[]; seen=set(); best=None
    for iteration in range(1,rounds+1):
        started=time.monotonic(); repairs=0; failure=None
        if mode=='random':
            remaining=[c for c in VALID if c not in seen]
            if not remaining:break
            c=rng.choice(remaining)
        else:
            c=None
            while repairs<=max_repairs:
                try:
                    candidate=query_model(prompt(history,mode,list(seen)),url,model,timeout,temperature)
                    if validate_v2(candidate):failure=validate_v2(candidate)
                    elif candidate in seen:failure='Duplicate proposal (not evaluated)'
                    else:
                        c=candidate;break
                except Exception as e:
                    failure=str(e)
                repairs+=1
        if c is None:
            r={'valid':False,'correct':False,'score':0.,'aggregate_cost':None,
               'reason':'REPAIR_FAILED: '+str(failure),'per_workload':[]}
            cfg=None
        else:
            seen.add(c)
            r=evaluate_v2(c)
            cfg=c.to_dict()
            if r['correct'] and (best is None or r['aggregate_cost']<best['result']['aggregate_cost']):
                best={'round':iteration,'config':cfg,'result':r}
        e={'round':iteration,'config':cfg,'result':r,'repair_requests':repairs,
           'seconds':round(time.monotonic()-started,3)}
        history.append(e)
        print(f'[{mode} seed={seed} round={iteration}] cfg={cfg} cost={r["aggregate_cost"]} '
              f'repairs={repairs} reason={r["reason"]}',flush=True)
    return {'mode':mode,'seed':seed,'attempts':history,'best':best,
            'best_cost':best['result']['aggregate_cost'] if best else None}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['all','random','score','diagnostic'],default='random')
    p.add_argument('--backend',choices=['random','llm'],default='random')
    p.add_argument('--rounds',type=int,default=8);p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--max-repairs',type=int,default=2)
    p.add_argument('--base-url',default=os.getenv('KERNEL_AGENT_BASE_URL','http://localhost:8000/v1'))
    p.add_argument('--model',default=os.getenv('KERNEL_AGENT_MODEL','Qwen/Qwen3-8B'))
    p.add_argument('--temperature',type=float,default=0.7)
    p.add_argument('--timeout',type=float,default=90.)
    p.add_argument('--out',default='results/v2_comparison.json')
    args=p.parse_args()
    if args.rounds<1 or args.repeat<1 or args.max_repairs<0:p.error('Invalid trial settings')
    modes=['random','score','diagnostic'] if args.mode=='all' else [args.mode]
    if args.backend=='random' and any(m!='random' for m in modes):p.error('Use --backend llm for AI modes')
    trials=[run(m,100+i,args.rounds,args.base_url,args.model,args.timeout,args.temperature,args.max_repairs)
            for m in modes for i in range(args.repeat)]
    summary={}
    for m in modes:
        t=[x for x in trials if x['mode']==m]
        success=[x['best_cost'] for x in t if x['best_cost'] is not None]
        summary[m]={'successful_trials':len(success),'total_trials':len(t),
          'mean_best_cost':sum(success)/len(success) if success else None,
          'trial_best_cost':[x['best_cost'] for x in t],
          'mean_round_to_best':sum(x['best']['round'] for x in t if x['best'])/len(success) if success else None,
          'total_repair_requests':sum(a['repair_requests'] for x in t for a in x['attempts'])}
    result={'metadata':{'model':args.model if args.backend=='llm' else None,
            'rounds':args.rounds,'repeat':args.repeat,'max_repairs':args.max_repairs,
            'synthetic_not_hardware':True,'proxy_cost_units':'synthetic cost units',
            'candidate_budget_excludes_repair_requests':True},'summary':summary,'trials':trials}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print('SUMMARY\n'+json.dumps(summary,indent=2));print('Wrote',path)
if __name__=='__main__':main()
