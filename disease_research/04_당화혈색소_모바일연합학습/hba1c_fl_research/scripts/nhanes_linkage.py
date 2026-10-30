from pathlib import Path
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
(RESEARCH_ROOT / "report").mkdir(exist_ok=True)
import pandas as pd, numpy as np, json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
base=str(RESEARCH_ROOT / 'data/public')
out={}
rows=[]
for cyc,s in [('NHANES_2021-2023','L'),('NHANES_2017-2018','J')]:
    r=lambda f: pd.read_sas(f'{base}/{cyc}/{f}_{s}.xpt',format='xport')
    demo=r('DEMO'); ghb=r('GHB'); glu=r('GLU'); bmx=r('BMX'); diq=r('DIQ'); cbc=r('CBC'); fq=r('FASTQX')
    df=demo[['SEQN','RIDSTATR','RIAGENDR','RIDAGEYR','RIDRETH3','RIDEXPRG']].merge(ghb[['SEQN','LBXGH']],on='SEQN',how='left')\
      .merge(glu[['SEQN','LBXGLU','WTSAF2YR']],on='SEQN',how='left').merge(bmx[['SEQN','BMXWT','BMXHT','BMXBMI']],on='SEQN',how='left')\
      .merge(diq[['SEQN','DIQ010','DIQ050','DIQ070']],on='SEQN',how='left').merge(cbc[['SEQN','LBXHGB']],on='SEQN',how='left')\
      .merge(fq[['SEQN','PHAFSTHR']],on='SEQN',how='left')
    assert df.SEQN.is_unique
    res={'demo_N_total':len(demo),'exam_N(RIDSTATR=2)':int((demo.RIDSTATR==2).sum()),
         'GHB_file_rows':len(ghb),'HbA1c_nonmissing':int(ghb.LBXGH.notna().sum()),
         'GLU_file_rows(fasting_subsample)':len(glu),'FPG_nonmissing':int(glu.LBXGLU.notna().sum()),
         'BMX_rows':len(bmx),'weight_nonmissing':int(bmx.BMXWT.notna().sum())}
    ex=df[(df.RIDSTATR==2)&(df.RIDAGEYR>=12)]
    res['examined_age12plus']=len(ex)
    res['missing_rate_HbA1c_among_exam12+']=round(ex.LBXGH.isna().mean(),4)
    res['missing_rate_weight_among_exam12+']=round(ex.BMXWT.isna().mean(),4)
    gsub=ex[ex.SEQN.isin(glu.SEQN)]
    res['fasting_subsample_assigned_age12+']=len(gsub)
    res['missing_rate_FPG_within_fasting_subsample']=round(gsub.LBXGLU.isna().mean(),4)
    res['missing_rate_HbA1c_within_fasting_subsample']=round(gsub.LBXGH.isna().mean(),4)
    res['missing_rate_weight_within_fasting_subsample']=round(gsub.BMXWT.isna().mean(),4)
    full=df[df.LBXGH.notna()&df.LBXGLU.notna()&df.BMXWT.notna()].copy()
    res['linked_HbA1c+FPG+weight_N']=len(full)
    res['linked_and_fasted_8to24h_N']=int(((full.PHAFSTHR>=8)&(full.PHAFSTHR<24)).sum())
    res['linked_age18plus_N']=int((full.RIDAGEYR>=18).sum())
    res['linked_age12to17_N']=int(((full.RIDAGEYR>=12)&(full.RIDAGEYR<18)).sum())
    res['linked_selfreport_diabetes(DIQ010=1)_N']=int((full.DIQ010==1).sum())
    res['linked_insulin_users(DIQ050=1)_N']=int((full.DIQ050==1).sum())
    res['linked_HbA1c>=6.5_N']=int((full.LBXGH>=6.5).sum())
    res['linked_HbA1c>=8.0_N']=int((full.LBXGH>=8.0).sum())
    res['linked_pregnant(RIDEXPRG=1)_N']=int((full.RIDEXPRG==1).sum())
    anem=((full.RIAGENDR==1)&(full.LBXHGB<13))|((full.RIAGENDR==2)&(full.LBXHGB<12))
    res['linked_anemia_WHO_Hb_cut_N']=int(anem.sum())
    res['linked_Hb_missing_N']=int(full.LBXHGB.isna().sum())
    res['records_per_person']='1 (cross-sectional; SEQN unique; no repeat HbA1c)'
    res['HbA1c_%_mean_sd_min_max']=[round(full.LBXGH.mean(),3),round(full.LBXGH.std(),3),full.LBXGH.min(),full.LBXGH.max()]
    res['FPG_mgdl_mean_sd_min_max']=[round(full.LBXGLU.mean(),2),round(full.LBXGLU.std(),2),full.LBXGLU.min(),full.LBXGLU.max()]
    res['weight_kg_mean_sd_min_max']=[round(full.BMXWT.mean(),2),round(full.BMXWT.std(),2),full.BMXWT.min(),full.BMXWT.max()]
    res['pearson_r_HbA1c_FPG']=round(full[['LBXGH','LBXGLU']].corr().iloc[0,1],3)
    res['pearson_r_HbA1c_weight']=round(full[['LBXGH','BMXWT']].corr().iloc[0,1],3)
    # group distributions
    full['sex']=full.RIAGENDR.map({1:'male',2:'female'})
    full['agegrp']=pd.cut(full.RIDAGEYR,[11,17,39,59,200],labels=['12-17','18-39','40-59','60+'])
    full['race']=full.RIDRETH3.map({1:'Mexican American',2:'Other Hispanic',3:'NH White',4:'NH Black',6:'NH Asian',7:'Other/Multi'})
    full['dm']=full.DIQ010.map({1:'self-report DM',2:'no',3:'borderline'}).fillna('unknown/refused')
    for g in ['sex','agegrp','race','dm']:
        t=full.groupby(g,observed=True).LBXGH.agg(['count','mean','std']).round(3)
        for k,v in t.iterrows():
            rows.append({'cycle':cyc,'group_var':g,'group':k,'N_linked':int(v['count']),'HbA1c_mean_%':v['mean'],'HbA1c_sd':v['std']})
    # exploratory feasibility only: unweighted 5-fold CV, adults 18+
    a=full[full.RIDAGEYR>=18]
    X=a[['LBXGLU','BMXWT']].values; y=a.LBXGH.values
    kf=KFold(5,shuffle=True,random_state=0); mae_m=[];mae_l=[];rm_m=[];rm_l=[]
    for tr,te in kf.split(X):
        m=y[tr].mean(); lr=LinearRegression().fit(X[tr],y[tr]); p=lr.predict(X[te])
        mae_m.append(np.abs(y[te]-m).mean()); mae_l.append(np.abs(y[te]-p).mean())
        rm_m.append(np.sqrt(((y[te]-m)**2).mean())); rm_l.append(np.sqrt(((y[te]-p)**2).mean()))
    res['EXPLORATORY_5foldCV_adults_unweighted']={'N':len(a),'MAE_mean_pred_%p':round(np.mean(mae_m),3),'MAE_linear(FPG,weight)_%p':round(np.mean(mae_l),3),
        'RMSE_mean_pred_%p':round(np.mean(rm_m),3),'RMSE_linear_%p':round(np.mean(rm_l),3),
        'note':'Within-NHANES random person split; not clinical validation; survey weights not applied'}
    out[cyc]=res
json.dump(out,open(str(RESEARCH_ROOT / 'report/NHANES_linkage_summary.json'),'w'),indent=1,default=float,ensure_ascii=False)
pd.DataFrame(rows).to_csv(str(RESEARCH_ROOT / 'report/NHANES_group_distribution.csv'),index=False)
print(json.dumps(out,indent=1,default=float))
