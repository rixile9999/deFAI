"""Run real optimization experiments; all output files distinguish simulated from real data."""
import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from cohort import ROOT,calibrate,make_world,training_schedule,World,sha256
from learning import Learner,run_training,evaluate,DEVICE


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S')+' '+str(message),flush=True)


def config_id(method,lr,mu):
    return f'{method}_lr{lr:g}_mu{mu:g}'


def candidates():
    result=[]
    for method in ('fedavg','fedprox','scaffold','fedavg_bn','fedbn'):
        rates=(.001,.003,.01)
        if method=='scaffold':rates=(.0001,.0003,*rates)
        if method in ('fedavg_bn','fedbn'):rates=(.00001,.00003,.0001,.0003,*rates)
        for lr in rates:
            for mu in ((.001,.01,.1) if method=='fedprox' else (0.,)):
                result.append({'method':method,'lr':lr,'mu':mu})
    return result


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=200000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261016,20261017,20261018,20261019,20261020])
    parser.add_argument('--all-worlds',action='store_true')
    parser.add_argument('--cpu-seeds',type=int,nargs='*',default=[])
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/fl_comparison/results/main')
    args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    calibration=calibrate(out)
    manifest={'n':args.n,'seeds':args.seeds,'days':730,'training_days':[90,499],
        'validation_days':[500,589],'test_days':[590,729],'K':2000,'epochs':5,
        'methods':['fedavg','fedprox','scaffold','fedavg_bn','fedbn'],
        'candidate_count_per_seed':len(candidates()),'aggregation':'equal_completed_client',
        'grid_extension_reason':'Before any test evaluation: SCAFFOLD and BN validation training diverged at original rates; smaller rates added, original failures retained.',
        'clinical_validation':False,'population_type':'synthetic_with_source_calibration',
        'profile':'E1_daily_moments_not_full_original_design',
        'all_worlds':args.all_worlds,'source_code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    manifest['cpu_worker_seeds']=args.cpu_seeds
    write_json(out/'manifest.json',manifest)
    tune_rows=[]
    for seed in args.seeds:
        seed_out=out/f'seed_{seed}';seed_out.mkdir(exist_ok=True)
        if seed in args.cpu_seeds:
            log(f'WAIT CPU workers seed={seed}; all candidate records required')
            while True:
                ready=True
                for params in candidates():
                    folder=seed_out/config_id(**params)
                    path=folder/'validation.json' if (folder/'validation.json').exists() else folder/'failure.json'
                    try:read_json(path)
                    except (OSError,json.JSONDecodeError):ready=False;break
                if ready:break
                time.sleep(5)
        world_path=seed_out/'world.npz'
        if world_path.exists():
            log(f'WORLD load {seed}');world=World.load(world_path)
        else:
            log(f'WORLD generate n={args.n} seed={seed}')
            world=make_world(args.n,seed,0,.7,.98,calibration,progress=print)
            training_schedule(world)
            world.save(world_path)
        schedule=training_schedule(world)
        log(f'WORLD ready seed={seed} updates={world.metadata["accepted_updates"]}')
        for params in candidates():
            run_out=seed_out/config_id(**params)
            if (run_out/'validation.json').exists():
                row=read_json(run_out/'validation.json')
            elif (run_out/'failure.json').exists():
                row=read_json(run_out/'failure.json')
            else:
                log(f'TUNE start seed={seed} {config_id(**params)}')
                try:
                    learner,row=run_training(world,schedule,**params,out=run_out,seed=seed)
                    del learner
                    row['status']='success';write_json(run_out/'validation.json',row)
                    log(f'TUNE done {config_id(**params)} seed={seed} val={row["validation"]["all"]["mae_pp"]:.6f} seconds={row["training_seconds"]:.2f}')
                except (FloatingPointError,RuntimeError) as error:
                    run_out.mkdir(exist_ok=True,parents=True)
                    row={**params,'seed':seed,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
                    write_json(run_out/'failure.json',row)
                    log(f'TUNE failed {config_id(**params)} seed={seed}: {error}')
                gc.collect()
                if DEVICE.type=='cuda':torch.cuda.empty_cache()
            tune_rows.append(row)
        del world;gc.collect()
    selection=[]
    for params in candidates():
        rows=[r for r in tune_rows if all(r.get(k)==v for k,v in params.items())]
        good=[r for r in rows if r.get('status','success')=='success']
        score=float(np.mean([r['validation']['all']['mae_pp'] for r in good])) if len(good)==len(args.seeds) else None
        selection.append({**params,'successful_seeds':len(good),'validation_mean_mae_pp':score})
    winners={}
    for method in manifest['methods']:
        valid=[r for r in selection if r['method']==method and r['validation_mean_mae_pp'] is not None]
        if not valid:raise RuntimeError(f'All candidates failed for {method}; do not invent results')
        winners[method]=min(valid,key=lambda r:r['validation_mean_mae_pp'])
    write_json(out/'optimization_selection.json',{'criterion':'mean person MAE on validation, all five seeds required',
        'test_scores_used_for_selection':False,'candidates':selection,'selected':winners})
    log('SELECTION '+json.dumps(winners))
    final=[]
    worlds=list(itertools.product((-.6931471805599453,0.,.6931471805599453),(.4,.7,.95),(.9,.98,.995))) if args.all_worlds else [(0.,.7,.98)]
    for wi,(beta,attendance,phi) in enumerate(worlds):
        for seed in args.seeds:
            if (beta,attendance,phi)==(0.,.7,.98):
                world=World.load(out/f'seed_{seed}'/'world.npz')
            else:
                log(f'WORLD full {wi+1}/{len(worlds)} seed={seed}')
                world=make_world(args.n,seed,beta,attendance,phi,calibration,progress=print)
            schedule=training_schedule(world)
            for method,params in winners.items():
                chosen={k:params[k] for k in ('method','lr','mu')}
                if (beta,attendance,phi)==(0.,.7,.98):
                    folder=out/f'seed_{seed}'/config_id(**chosen)
                    row=read_json(folder/'validation.json')
                    learner=Learner(world,method,params['lr'],params['mu'],seed)
                    learner.load(folder/'model.npz')
                else:
                    folder=out/'worlds'/f'w{wi:02}_seed{seed}'/method
                    learner,row=run_training(world,schedule,**chosen,out=folder,seed=seed)
                if (folder/'test.json').exists():test=read_json(folder/'test.json')
                else:
                    test=evaluate(learner,world,'test',folder/'test_person_errors.npz')
                    write_json(folder/'test.json',test)
                entry={**row,'test':test,'beta':beta,'attendance':attendance,'phi':phi,'folder':str(folder)}
                final.append(entry)
                log(f'TEST seed={seed} {method} all={test["all"]["mae_pp"]:.6f} seen={test["seen"]["mae_pp"]:.6f} new={test["new"]["mae_pp"]:.6f}')
                del learner;gc.collect()
                if DEVICE.type=='cuda':torch.cuda.empty_cache()
            del world;gc.collect()
    write_json(out/'final_results.json',{'manifest':manifest,'selected':winners,'runs':final,
        'unexplored':['27-world grid' if not args.all_worlds else 'other structural generators',
            'K=200 or 10000','linear models','full CGM waveforms','upload delays','clinical validation','cryptographic protocols']})
    log(f'COMPLETE selected_runs={len(final)} candidates={len(tune_rows)} output={out}')


if __name__=='__main__':main()
