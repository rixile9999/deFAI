"""Precompute disjoint later worlds on CPU; train only in the main runner."""
import gc
from cohort import ROOT,calibrate,make_world,training_schedule

out=ROOT/'experiments/fl_comparison/results/main'
calibration=calibrate(out)
for seed in (20261019,20261018):
    folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
    path=folder/'world.npz'
    if path.exists():continue
    world=make_world(200000,seed,0.,.7,.98,calibration,progress=print)
    training_schedule(world)
    world.save(path)
    print(f'PREFETCH complete seed={seed}',flush=True)
    del world;gc.collect()
