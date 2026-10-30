"""Compute two disjoint seeds on CPU while the main process uses the shared GPU.

Initial parameters are exported from the same CUDA generator used by the main
experiment. CPU arithmetic differs slightly, so every output records its device.
The main process later reuses complete validation records for these seeds.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--export-initial',action='store_true')
parser.add_argument('--seeds',type=int,nargs='+',default=[20261020,20261019])
args=parser.parse_args()
if not args.export_initial:
    os.environ['CUDA_VISIBLE_DEVICES']='-1'

import numpy as np
import torch
from cohort import ROOT,World,calibrate,make_world,training_schedule,sha256
import learning
from run_comparison import candidates,config_id,read_json,write_json,log

OUT=ROOT/'experiments/fl_comparison/results/main'
SEEDS=tuple(args.seeds)
INIT=OUT/'cpu_initial_parameters'

if args.export_initial:
    INIT.mkdir(exist_ok=True)
    for seed in SEEDS:
        arrays=learning.initial_parameters(seed,bn=True)
        np.savez(INIT/f'{seed}.npz',**{f'p_{j}':p.cpu().numpy() for j,p in enumerate(arrays)})
    write_json(INIT/'provenance.json',{'initial_generator':'same CUDA generator as main process',
        'seeds':SEEDS,'files':{p.name:sha256(p) for p in INIT.glob('*.npz')},
        'worker_sha256':sha256(Path(__file__))})
    log('CPU initial parameter export complete')
else:
    assert learning.DEVICE.type=='cpu'
    def same_initial(seed,hidden=64,bn=False,device=learning.DEVICE,dtype=learning.DTYPE):
        assert hidden==64
        with np.load(INIT/f'{seed}.npz') as z:
            return [torch.as_tensor(z[f'p_{j}'],device=device,dtype=dtype).clone()
                    for j in range(6 if bn else 4)]
    learning.initial_parameters=same_initial
    calibration=calibrate(OUT)
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=True)
        path=folder/'world.npz'
        if path.exists():world=World.load(path)
        else:
            log(f'CPU WORLD generate {seed}')
            world=make_world(200000,seed,0.,.7,.98,calibration,progress=print)
            training_schedule(world);world.save(path)
        schedule=training_schedule(world)
        for params in candidates():
            run_out=folder/config_id(**params)
            if (run_out/'validation.json').exists() or (run_out/'failure.json').exists():continue
            log(f'CPU TUNE start seed={seed} {config_id(**params)}')
            try:
                learner,row=learning.run_training(world,schedule,**params,out=run_out,seed=seed)
                row['status']='success'
                row['execution_note']='CPU worker; exact CUDA-generator initial arrays, CPU arithmetic'
                row['cpu_worker_sha256']=sha256(Path(__file__))
                write_json(run_out/'validation.json',row)
                log(f'CPU TUNE done seed={seed} {config_id(**params)} val={row["validation"]["all"]["mae_pp"]:.6f} seconds={row["training_seconds"]:.2f}')
                del learner
            except (FloatingPointError,RuntimeError) as error:
                run_out.mkdir(exist_ok=True,parents=True)
                row={**params,'seed':seed,'status':'failed','error':str(error),'traceback':traceback.format_exc(),'device':'cpu'}
                write_json(run_out/'failure.json',row)
                log(f'CPU TUNE failed seed={seed} {config_id(**params)}: {error}')
            gc.collect()
        del world;gc.collect()
    log('CPU WORKER COMPLETE')
