"""SYNTHETIC data for FL communication/training-flow smoke tests ONLY.
Not derived from any real person; coefficients are arbitrary; must never be used as ground truth,
for performance claims, or mixed with real datasets."""
import numpy as np, pandas as pd
rng=np.random.default_rng(20261016)
rows=[]
for u in range(1,51):
    n=int(rng.integers(1,13)); off=rng.normal(0,0.3); base_fpg=rng.normal(110,25); base_wt=rng.normal(75,14)
    day=0
    for k in range(n):
        day+=int(rng.integers(60,120))
        fpg=max(60,base_fpg+rng.normal(0,12)); wt=max(35,base_wt+rng.normal(0,1.5))
        a1c=round(2.0+0.03*fpg+0.004*wt+off+rng.normal(0,0.25),1)
        rows.append(dict(synthetic_client_id=f'SYN{u:03d}',record_seq=k+1,day_offset=day,glucose_mgdl=round(fpg,1),weight_kg=round(wt,1),hba1c_pct=a1c,is_synthetic=1))
pd.DataFrame(rows).to_csv('synthetic_fl_smoketest.csv',index=False)
print(len(rows),'rows')
