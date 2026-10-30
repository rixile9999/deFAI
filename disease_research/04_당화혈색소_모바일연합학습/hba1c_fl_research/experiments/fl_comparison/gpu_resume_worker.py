"""Resume only cached seeds 19/20 when the shared GPU becomes available."""
import gc
import traceback
import torch
from cohort import ROOT,World,training_schedule,sha256
from learning import DEVICE,run_training
from run_comparison import candidates,config_id,write_json,log
from pathlib import Path

assert DEVICE.type=='cuda'
out=ROOT/'experiments/fl_comparison/results/main'
for seed in (20261019,20261020):
    folder=out/f'seed_{seed}'
    world=World.load(folder/'world.npz');schedule=training_schedule(world)
    for params in candidates():
        run_out=folder/config_id(**params)
        if (run_out/'validation.json').exists() or (run_out/'failure.json').exists():continue
        log(f'GPU RESUME start seed={seed} {config_id(**params)}')
        try:
            learner,row=run_training(world,schedule,**params,out=run_out,seed=seed)
            row['status']='success';row['execution_note']='GPU resumed after resource availability changed'
            row['resume_worker_sha256']=sha256(Path(__file__))
            write_json(run_out/'validation.json',row)
            log(f'GPU RESUME done seed={seed} {config_id(**params)} val={row["validation"]["all"]["mae_pp"]:.6f}')
            del learner
        except (FloatingPointError,RuntimeError) as error:
            run_out.mkdir(exist_ok=True,parents=True)
            write_json(run_out/'failure.json',{**params,'seed':seed,'status':'failed','error':str(error),'traceback':traceback.format_exc(),'device':'cuda'})
            log(f'GPU RESUME failed seed={seed} {config_id(**params)}: {error}')
        gc.collect();torch.cuda.empty_cache()
    del world;gc.collect()
log('GPU RESUME COMPLETE')
