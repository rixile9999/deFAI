"""Independent artifact checks: saved errors, selected scores, complete scope."""
import csv
import json
from pathlib import Path
import numpy as np
from cohort import ROOT,World,sha256
from run_comparison import candidates,config_id

OUT=ROOT/'experiments/fl_comparison/results/main'
final=json.loads((OUT/'final_results.json').read_text(encoding='utf-8'))
selection=json.loads((OUT/'optimization_selection.json').read_text(encoding='utf-8'))
manifest=final['manifest'];seeds=manifest['seeds']
assert manifest['n']==200000 and len(seeds)==5
assert not selection['test_scores_used_for_selection']
assert len(final['runs'])==25
statuses=[]
for seed in seeds:
    for params in candidates():
        folder=OUT/f'seed_{seed}'/config_id(**params)
        paths=[p for p in (folder/'validation.json',folder/'failure.json') if p.exists()]
        assert len(paths)==1,(folder,paths)
        row=json.loads(paths[0].read_text(encoding='utf-8'))
        statuses.append(row.get('status','success'))
for method,chosen in final['selected'].items():
    eligible=[r for r in selection['candidates'] if r['method']==method and r['successful_seeds']==5]
    assert chosen==min(eligible,key=lambda r:r['validation_mean_mae_pp'])
    runs=[r for r in final['runs'] if r['method']==method]
    assert set(r['seed'] for r in runs)==set(seeds)
    assert all(0<r['actual_aggregations']<=410 for r in runs)
    observed=np.mean([r['validation']['all']['mae_pp'] for r in runs])
    assert abs(observed-chosen['validation_mean_mae_pp'])<1e-12
    assert all(r['lr']==chosen['lr'] and r['mu']==chosen['mu'] for r in runs)
max_error=0.
model_hashes={}
for seed in seeds:
    world=World.load(OUT/f'seed_{seed}'/'world.npz')
    masks={'all':np.ones(200000,bool),'seen':np.arange(200000)<160000,
           'new':np.arange(200000)>=160000,'high_a1c':world.groups['high_a1c'],
           'low_engagement':world.groups['low_engagement']}
    for r in [row for row in final['runs'] if row['seed']==seed]:
        with np.load(Path(r['folder'])/'test_person_errors.npz') as z:
            observed=z['counts']>0;errors=z['mae'].astype(np.float64)
        for group,mask in masks.items():
            valid=observed&mask
            assert int(valid.sum())==r['test'][group]['people']
            difference=abs(float(errors[valid].mean())-r['test'][group]['mae_pp'])
            max_error=max(max_error,difference)
            assert difference<1e-7
        with np.load(Path(r['folder'])/'model.npz') as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
        model_hashes[str((Path(r['folder'])/'model.npz').relative_to(ROOT))]=sha256(Path(r['folder'])/'model.npz')
for name in ('cohort.py','learning.py'):
    assert sha256(Path(__file__).parent/name)==manifest['source_code_sha256'][name]
with (ROOT/'report/HbA1c_FL_기법비교_최종평가.csv').open(encoding='utf-8-sig',newline='') as stream:
    csv_rows=list(csv.DictReader(stream))
assert len(csv_rows)==25*8
for row in csv_rows:
    r=next(r for r in final['runs'] if r['seed']==int(row['seed']) and r['method']==row['method'])
    for metric in ('mae_pp','rmse_pp','bias_pp','p90_person_mae_pp'):
        if r['test'][row['group']][metric] is not None:
            assert abs(float(row[metric])-r['test'][row['group']][metric])<1e-12
audit=json.loads((OUT/'world_audit.json').read_text(encoding='utf-8'))
assert {r['seed'] for r in audit['worlds']}==set(seeds)
for r in final['runs']:
    a=next(a for a in audit['worlds'] if a['seed']==r['seed'])
    assert r['actual_aggregations']==a['actual_nonempty_aggregations']
output={'scope':'completed basic 200000-person profile, 5 seeds',
    'finalized_experiment_cells':len(statuses),'completed':statuses.count('success'),'failed':statuses.count('failed'),
    'selected_models':25,'csv_group_rows':len(csv_rows),
    'saved_person_error_mae_max_difference_pp':max_error,
    'selection_recomputed':True,'frozen_training_source_hashes_verified':True,
    'saved_models_finite':True,'all_five_worlds_audited':True,'selected_model_sha256':model_hashes}
(OUT/'results_verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
