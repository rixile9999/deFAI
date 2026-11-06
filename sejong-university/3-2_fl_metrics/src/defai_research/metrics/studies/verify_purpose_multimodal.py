from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from ._paths import ROOT, source_path, relative_path
import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import f1_score,log_loss
from threadpoolctl import threadpool_limits
_THREADS=threadpool_limits(limits=1)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def forward(arrays,x):
    h=np.maximum(0,np.matmul(x,arrays[0])+arrays[1]);logits=np.matmul(h,arrays[2])+arrays[3]
    logits=logits-np.max(logits,axis=1,keepdims=True);e=np.exp(logits);return e/np.sum(e,axis=1,keepdims=True)
def inputs(x,arm,fault=False):
    a=x.copy()
    if arm=='gyro_only':a[:,:36]=0
    if arm=='acc_only' or fault:a[:,36:]=0
    return a
def f1(y,p):return float(f1_score(y,p.argmax(1),labels=list(range(6)),average='macro',zero_division=0))
def h(counts):
    c=np.asarray(counts,float).flatten();p=c[c>0]/c.sum();return float(np.sum(-p*np.log2(p)))
def cmi(counts):
    p=np.asarray(counts,float);p=p/p.sum();pac=p.sum(1);pbc=p.sum(0);pc=p.sum((0,1));total=0.
    for a,b,c in np.argwhere(p>0):total+=p[a,b,c]*np.log2(p[a,b,c]*pc[c]/(pac[a,c]*pbc[b,c]))
    return float(total)
def ci(values):
    a=np.asarray(values,float);m=a.mean();half=student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    return {'mean':float(m),'sd':float(a.std(ddof=1)),'ci95':[float(m-half),float(m+half)]}
def recovery(values,start,end,target):
    for r in range(start,end-1):
        if all(values[k-1]>=target-.02 for k in range(r,r+3)):return r-start
    return None


def audit(root):
    protocol=read_json(root/'protocol.json');split=read_json(root/'data'/'subject_split.json');summary=read_json(root/'summary.json')
    errors=[];checks={'model_forward_arrays':0,'trajectory_validation_checkpoints':0,'round_work_rows':0,
                      'adaptation_subject_values':0,'information_values':0,'feature_entropy_rows':0,'transfer_subject_rows':0,'statistical_intervals':0}
    def check(condition,note):
        if not condition:errors.append(note)
    def near(got,want,note,tol=1e-10):check(bool(np.allclose(got,want,rtol=0,atol=tol)),note)
    for name,digest in protocol['source_sha256'].items():
        check(sha(source_path(name))==digest,f'current source{name}')
        check(sha(root/'source_snapshot'/name)==digest,f'source snapshot{name}')
    check(sha(root/'data'/'uci_har_official.zip')==split['archive_sha256'],'official archive hash')
    check(sha(root/'data'/'features_and_subject_splits.npz')==split['feature_file_sha256'],'prepared feature hash')
    with np.load(root/'data'/'features_and_subject_splits.npz') as z:data={k:z[k] for k in z.files}
    x,y,sid=data['x'],data['y'],data['subject_id'];ti=data['test_indices'];vi=data['validation_indices'];tr=data['train_indices']
    tsub,vsub,rsub=map(set,(split['test_subjects'],split['validation_subjects'],split['train_subjects']))
    check(not (tsub&vsub or tsub&rsub or vsub&rsub),'subject overlap')
    for key,subjects in (('train_indices',rsub),('validation_indices',vsub),('test_indices',tsub)):
        check(set(sid[data[key]])==subjects,f'{key} subjects')
    check(np.array_equal(np.sort(np.concatenate((tr,vi,ti))),np.arange(len(x))),'complete disjoint windows')
    check(np.all(data['official_test'][ti]==1) and np.all(data['official_test'][tr]==0) and np.all(data['official_test'][vi]==0),'official heldout subjects')
    raw=data['raw_sensor_features'];mean=raw[tr].mean(0);scale=np.maximum(raw[tr].std(0),1e-6)
    near(data['feature_mean'],mean,'train-only feature mean',0);near(data['feature_scale'],scale,'train-only feature std',0)
    near(x,np.clip((raw-mean)/scale,-6,6),'train-only transformation',0)
    lookup={};probs={};models={};seeds=protocol['seeds'];arms=protocol['arms'];rounds=protocol['rounds']
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for seed in seeds:
        for arm in arms:
            run=root/'runs'/f's{seed}_{arm}';final=read_json(run/'final.json');lookup[seed,arm]=final
            check(final['protocol_sha256']==digest,f'protocol run{seed}{arm}');check(final['test_evaluated_after_training'],f'test lock{seed}{arm}')
            with np.load(run/'models.npz') as z:mm={k:z[k] for k in z.files}
            with np.load(run/'test_probabilities.npz') as z:pp={k:z[k] for k in z.files}
            models[seed,arm]=mm;probs[seed,arm]=pp
            for name,p in pp.items():
                check(bool(np.isfinite(p).all() and (p>=0).all()),f'finite probabilities{seed}{arm}{name}');near(p.sum(-1),1,f'probability sum{seed}{arm}{name}',2e-6)
            clients=final['training_subjects'];n=[int(np.sum(sid==s)) for s in clients];work=read_csv(run/'rounds.csv')
            check(len(work)==rounds,f'roundcount{seed}{arm}')
            expected_bytes=4*sum(mm[k].size for k in mm if k.startswith('selected_')) if arm!='local_only' else 0
            for row in work:
                rnd=int(row['round']);check(int(row['examples_processed'])==sum(n),f'examples{seed}{arm}{rnd}')
                check(int(row['optimizer_steps'])==sum((v+63)//64 for v in n),f'steps{seed}{arm}{rnd}')
                shock=arm=='adaptive_fusion' and 11<=rnd<=20
                check(int(row['actual_train_gyro_coverage'])==int(not shock and arm!='acc_only'),f'actual gyro training coverage{seed}{arm}{rnd}')
                check(float(row['fedavg_reference_max_abs_gap'])<=2e-6,f'aggregation{seed}{arm}{rnd}');checks['round_work_rows']+=1
            local_scores=[]
            for prefix in ('selected','final'):
                if arm=='local_only':
                    for i,s in enumerate(clients):
                        aa=[mm[f'{prefix}_c{s}_a{j}'] for j in range(4)];computed=forward(aa,inputs(x[ti],arm))
                        near(pp[prefix][i],computed,f'forward{seed}{arm}{prefix}{s}',2e-7);checks['model_forward_arrays']+=1
                        if prefix=='selected':local_scores.append(f1(y[ti],computed))
                else:
                    aa=[mm[f'{prefix}_a{j}'] for j in range(4)];computed=forward(aa,inputs(x[ti],arm))
                    near(pp[prefix],computed,f'forward{seed}{arm}{prefix}',2e-7);checks['model_forward_arrays']+=1
            near(final['macro_f1'],np.mean(local_scores) if arm=='local_only' else f1(y[ti],pp['selected']),f'final F1{seed}{arm}')
            if arm!='local_only':
                losses=[]
                for rnd in range(1,rounds+1):
                    aa=[mm[f'trajectory_a{j}'][rnd-1] for j in range(4)]
                    for condition in ('clean','fault'):
                        computed=forward(aa,inputs(x[ti],arm,condition=='fault' and 11<=rnd<=20))
                        near(pp[f'trajectory_{condition}'][rnd-1],computed,f'trajectoryforward{seed}{arm}{condition}{rnd}',2e-7);checks['model_forward_arrays']+=1
                    vp=forward(aa,inputs(x[vi],arm));losses.append(log_loss(y[vi],vp,labels=np.arange(6)))
                check(int(final['selected_round'])==int(np.argmin(losses))+1,f'validation-only checkpoint{seed}{arm}');checks['trajectory_validation_checkpoints']+=1
                for condition,fault in (('clean',False),('fault',True)):
                    aa=[mm[f'trajectory_a{j}'][9] for j in range(4)];fr=forward(aa,inputs(x[ti],arm,fault))
                    for rnd in range(1,rounds+1):
                        if (11<=rnd<=20)==fault:near(pp['r10_frozen_fault'][rnd-1],fr,f'r10frozen{seed}{arm}{rnd}',2e-7)
                aa=[mm[f'trajectory_a{j}'][19] for j in range(4)]
                near(pp['r20_frozen_return_clean'],forward(aa,inputs(x[ti],arm)),f'r20frozen{seed}{arm}',2e-7)
    information=read_csv(root/'sensor_information.csv')
    with np.load(root/'empirical_joint_counts.npz') as z:joints={k:z[k] for k in z.files}
    for row in information:
        seed=int(row['seed']);a,g,f=[probs[seed,arm]['selected'] for arm in ('acc_only','gyro_only','full_fusion')]
        pa,pg,pf=a.argmax(1),g.argmax(1),f.argmax(1)
        joint=np.bincount(((y[ti]*6+pa)*6+pg)*6+pf,minlength=6**4).reshape((6,)*4)
        node=np.searchsorted(split['test_subjects'],sid[ti]);nj=np.bincount((node*6+y[ti])*6+pf,minlength=9*36).reshape(9,6,6)
        near(joints[f's{seed}_Y_acc_gyro_fused'],joint,f'joint counts{seed}',0);near(joints[f's{seed}_subject_Y_fused'],nj,f'node counts{seed}',0)
        q=(a.astype(float)+g)/2;jsd=np.mean(np.sum(.5*(a*np.log2(np.maximum(a,1e-30)/q)+g*np.log2(np.maximum(g,1e-30)/q)),1))
        values={'MUI_real_sensor_predictive_JSD_bits':jsd,'H_Y_accPred_gyroPred_fusedPred_bits':h(joint),
                'I_Y_fused_given_acc_gyro_bits':cmi(joint.transpose(0,3,1,2).reshape(6,6,36)),
                'I_Y_gyro_given_acc_bits':cmi(joint.sum(3).transpose(0,2,1)),
                'H_subject_Y_fusedPred_bits':h(nj),'I_Y_fused_given_subject_bits':cmi(nj.transpose(1,2,0)),
                'I_subject_fused_bits':cmi(nj.sum(1)[:,:,None])}
        for key,value in values.items():near(float(row[key]),value,f'information{seed}{key}');checks['information_values']+=1
    for row in read_csv(root/'heldout_transfer.csv'):
        seed=int(row['seed']);s=int(row['heldout_subject']);mask=sid[ti]==s
        gf=f1(y[ti][mask],probs[seed,'full_fusion']['selected'][mask]);lf=np.mean([f1(y[ti][mask],p[mask]) for p in probs[seed,'local_only']['selected']])
        lo=f1(y[ti][mask],probs[seed,'leave_node_out']['selected'][mask])
        for key,value in {'global_macro_f1':gf,'isolated_local_model_mean_macro_f1':lf,'KID_global_minus_isolated_macro_f1':gf-lf,
                          'leave_one_training_node_out_macro_f1':lo,'global_minus_leave_node_out_macro_f1':gf-lo}.items():near(float(row[key]),value,f'transfer{seed}{s}{key}')
        checks['transfer_subject_rows']+=1
    adaptation={}
    for row in read_csv(root/'adaptation_summary.csv'):
        seed=int(row['seed']);s=int(row['heldout_subject']);arm=row['arm'];mask=sid[ti]==s;p=probs[seed,arm]
        clean=np.array([f1(y[ti][mask],v[mask]) for v in p['trajectory_clean']]);fault=np.array([f1(y[ti][mask],v[mask]) for v in p['trajectory_fault']])
        frozen=np.array([f1(y[ti][mask],v[mask]) for v in probs[seed,'full_fusion']['r10_frozen_fault']]);fr20=f1(y[ti][mask],p['r20_frozen_return_clean'][mask])
        target=clean[9];expected={'immediate_fault_F1_drop':clean[10]-fault[10],'shock_mean_fault_F1':fault[10:20].mean(),
                'shock_AUC_positive_deficit_F1_rounds':np.maximum(clean-fault,0)[10:20].sum(),'pre_fault_r10_clean_macro_f1':target,
                'shock_cumulative_F1_gain_vs_r10_frozen':(fault-frozen)[10:20].sum(),
                'return_cumulative_update_gain_vs_r20_frozen_F1_rounds':(fault-fr20)[20:30].sum()}
        for key,value in expected.items():near(float(row[key]),value,f'adaptation{seed}{s}{arm}{key}');checks['adaptation_subject_values']+=1
        for key,start,end in (('recovery_rounds_after_dropout',11,20),('return_recovery_rounds',21,30)):
            got=None if row[key]=='' else int(row[key]);check(got==recovery(fault,start,end,target),f'recovery{seed}{s}{arm}{key}');checks['adaptation_subject_values']+=1
        adaptation[seed,s,arm]=expected
    for row in read_csv(root/'local_train_entropy.csv'):
        s=int(row['training_subject']);j=int(row['feature_index']);mask=sid[tr]==s;counts=np.histogram(x[tr][mask,j],bins=data['entropy_bin_edges'])[0]
        labels=np.bincount(y[tr][mask],minlength=6)
        near(json.loads(row['feature_bin_counts_json']),counts,f'feature count{s}{j}',0);near(json.loads(row['label_counts_json']),labels,f'label count{s}{j}',0)
        near(float(row['marginal_feature_entropy_bits']),h(counts),f'feature entropy{s}{j}');near(float(row['node_label_entropy_bits']),h(labels),f'label entropy{s}{j}');checks['feature_entropy_rows']+=1
    pairs={'fusion_minus_acc':'acc_only','fusion_minus_gyro':'gyro_only','fusion_minus_isolated_local':'local_only','fusion_minus_leave_node_out':'leave_node_out'}
    expected_effects={key:[lookup[s,'full_fusion']['macro_f1']-lookup[s,arm]['macro_f1'] for s in seeds] for key,arm in pairs.items()}
    expected_effects['adaptive_minus_stale_shock_F1']=[np.mean([adaptation[s,t,'adaptive_fusion']['shock_mean_fault_F1']-adaptation[s,t,'full_fusion']['shock_mean_fault_F1'] for t in split['test_subjects']]) for s in seeds]
    expected_effects['stale_minus_adaptive_shock_AUC_deficit']=[np.mean([adaptation[s,t,'full_fusion']['shock_AUC_positive_deficit_F1_rounds']-adaptation[s,t,'adaptive_fusion']['shock_AUC_positive_deficit_F1_rounds'] for t in split['test_subjects']]) for s in seeds]
    for key,values in expected_effects.items():
        expected=ci(values)
        for field,want in expected.items():near(summary['paired_effects'][key][field],want,f'paired statistic{key}{field}')
        checks['statistical_intervals']+=1
    for arm in arms:
        expected=ci([lookup[s,arm]['macro_f1'] for s in seeds])
        for field,want in expected.items():near(summary['arm_test_macro_f1'][arm][field],want,f'arm statistic{arm}{field}')
        checks['statistical_intervals']+=1
    check(len(lookup)==30,'all30predeclaredruns');check(summary['source_unchanged'],'source reported unchanged')
    result={'status':'passed' if not errors else 'failed','errors':errors,'checks':checks,'runs':len(lookup),'rounds':rounds*len(lookup),
            'source_sha256':protocol['source_sha256'],'protocol_sha256':digest,'verifier_sha256':sha(__file__),
            'audit_scope':'Separate implementation self-audit: actual saved model forwards, subject split and train-only transforms, validation checkpoints, actual work and modality event logs, entropy/MI/joint counts, heldout transfer, adaptation/censoring, paired seed intervals.',
            'limits':'No independent third-party review or production deployment;5training seeds/fixed subject split and unadjusted95%intervals describe this study, not a population guarantee.'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=ROOT/'results'/'purpose_multimodal');p.add_argument('--output',type=Path);args=p.parse_args()
    result=audit(args.results);dest=args.output or args.results/'self_verification.json';dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
