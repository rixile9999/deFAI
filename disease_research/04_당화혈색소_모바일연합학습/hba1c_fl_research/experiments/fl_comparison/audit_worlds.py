"""Audit time eligibility, source separation and denominators after generation."""
import json
from pathlib import Path
import numpy as np
from cohort import ROOT,World,training_schedule,sha256

OUT=ROOT/'experiments/fl_comparison/results/main'

def audit(path):
    world=World.load(path)
    schedule=training_schedule(world)
    assert world.x.shape==(200000,9,20)
    assert np.isfinite(world.x).all()
    assert not np.intersect1d(world.groups['donor_id'][:world.dev_n],
                             world.groups['donor_id'][world.dev_n:]).size
    assert np.all(world.arrival[world.draw>=0]-world.draw[world.draw>=0]==2)
    last=np.full(world.dev_n,-1)
    counts=[]
    for day,ids in schedule:
        assert len(ids)<=2000 and np.all(ids<world.dev_n)
        assert len(np.unique(ids))==len(ids)
        assert world.available[ids,day].all()
        arr=world.arrival[ids]
        latest=np.where(np.isfinite(world.y[ids])&(arr<=day),arr,-1).max(1)
        assert np.all((latest>last[ids])&(latest>=day-30))
        last[ids]=latest
        mask=np.isfinite(world.y[ids])&(arr<=day)&(arr>=day-365)
        assert mask.any(1).all()
        counts.extend(mask.sum(1).tolist())
    # Check previous-label feature against the known earlier label wherever that
    # earlier record survives final model-eligibility masking. A clipped feature
    # is explicitly allowed because clipping is part of the recorded transform.
    known=0;max_error=0.
    cal=json.loads((OUT/'calibration.json').read_text())
    for j in range(1,9):
        prior=np.where((world.arrival[:,:j]<world.draw[:,j,None])&
                       (world.draw[:,:j]>=0),world.draw[:,:j],-1).argmax(1)
        person=np.arange(len(world.x))
        good=(world.draw[:,j]>=0)&np.isfinite(world.y[person,prior])&\
             (world.arrival[person,prior]<world.draw[:,j])
        target=np.clip((world.y[person,prior]-cal['x_mean'][12])/cal['x_scale'][12],-10,10)
        error=np.abs(world.x[good,j,12]-target[good])
        known+=len(error)
        if len(error):max_error=max(max_error,float(error.max()))
    assert max_error<2e-6
    splits={}
    for name,lo,hi in [('train',90,499),('validation',500,589),('test',590,729)]:
        mask=np.isfinite(world.y)&(world.draw>=lo)&(world.draw<=hi)
        if name=='validation':mask &= world.arrival<=hi
        splits[name]={'observed_labels':int(mask.sum()),'observed_people':int(mask.any(1).sum()),
            'new_people':int(mask[world.dev_n:].any(1).sum())}
    histogram={str(k):int(v) for k,v in zip(*np.unique(counts,return_counts=True))}
    return {'seed':world.metadata['seed'],'population':len(world.x),'development':world.dev_n,
        'planned_training_opportunities':len(schedule),'actual_nonempty_aggregations':sum(bool(len(ids)) for _,ids in schedule),
        'source_donor_overlap':0,'all_input_values_finite':True,
        'eligible_fresh_selected_updates':len(counts),'maximum_selected_per_day':max(len(ids) for _,ids in schedule),
        'selected_local_label_count_histogram':histogram,
        'prior_label_feature_checked_records':known,'prior_label_feature_max_abs_scaled_error':max_error,
        'evaluation_denominators':splits,'world_sha256':sha256(path)}

if __name__=='__main__':
    rows=[audit(path) for path in sorted(OUT.glob('seed_*/world.npz'))]
    output={'audit_scope':'artifact invariants, not clinical realism validation','worlds':rows}
    (OUT/'world_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
