from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from ._paths import ROOT, source_path, relative_path
import numpy as np
from scipy.stats import entropy,t
from sklearn.metrics import precision_recall_fscore_support,log_loss,mutual_info_score


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b,atol=1e-8):np.testing.assert_allclose(a,b,atol=atol,rtol=1e-7)
def csvrows(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig',newline='')))


def predict(a,x):
    h=np.maximum(x@a[0]+a[1],0);z=h@a[2]+a[3];z-=z.max(1,keepdims=True)
    p=np.exp(z);return p/p.sum(1,keepdims=True)


def view(x,arm,fault=False):
    result=x.copy()
    if arm=='gyro_only':result[:,:36]=0
    if arm=='acc_only' or fault:result[:,36:]=0
    return result


def macro(y,p):
    return float(precision_recall_fscore_support(y,p.argmax(1),labels=np.arange(6),average='macro',zero_division=0)[2])


def interval(values):
    x=np.asarray(values,float);m=float(x.mean());sd=float(x.std(ddof=1));h=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return {'mean':m,'sd':sd,'ci95':[m-h,m+h],'n_training_seeds':len(x),'values':x.tolist()}


def conditional_mi(counts):
    c=np.asarray(counts).reshape(counts.shape[0],counts.shape[1],-1)
    totals=c.sum(axis=(0,1));n=totals.sum()
    return float(sum(float(totals[k])/n*mutual_info_score(None,None,contingency=c[:,:,k])/np.log(2)
                     for k in range(c.shape[2]) if totals[k]))


def multivariate_count(*variables,sizes):
    flat=np.ravel_multi_index(tuple(variables),sizes)
    return np.bincount(flat,minlength=int(np.prod(sizes))).reshape(sizes)


def raw_features(sample):
    values=[]
    for fn in (lambda a:np.mean(a,axis=1),lambda a:np.std(a,axis=1),
               lambda a:np.sqrt(np.mean(a*a,axis=1)),lambda a:np.min(a,axis=1),
               lambda a:np.max(a,axis=1),lambda a:np.median(a,axis=1),
               lambda a:np.percentile(a,75,axis=1)-np.percentile(a,25,axis=1),
               lambda a:np.mean(np.abs(np.diff(a,axis=1)),axis=1)):
        values.append(fn(sample))
    power=(np.abs(np.fft.rfft(sample,axis=1))/128)**2
    hz=np.arange(65)*50/128
    for left,right in ((0,2.5),(2.5,7.5),(7.5,25.0001)):
        values.append(power[:,(hz>=left)&(hz<right),:].sum(1))
    magnitude=np.sqrt((sample*sample).sum(2))
    values.append(np.column_stack((magnitude.mean(1),magnitude.std(1),np.sqrt((magnitude*magnitude).mean(1)))))
    return np.column_stack(values).astype(np.float32)


def audit_data(study,counts):
    split=load(study/'data/subject_split.json')
    assert sha(study/'data/uci_har_official.zip')==split['archive_sha256']
    assert sha(study/'data/features_and_subject_splits.npz')==split['feature_file_sha256']
    with np.load(study/'data/features_and_subject_splits.npz',allow_pickle=False) as saved:
        data={key:saved[key] for key in saved.files}
    subject_sets=[set(split[k]) for k in ('train_subjects','validation_subjects','test_subjects')]
    assert list(map(len,subject_sets))==[17,4,9]
    assert all(not subject_sets[i]&subject_sets[j] for i in range(3) for j in range(i+1,3))
    for index,key in enumerate(('train_indices','validation_indices','test_indices')):
        expected=np.flatnonzero(np.isin(data['subject_id'],list(subject_sets[index])))
        np.testing.assert_array_equal(data[key],expected)
    assert set.union(*subject_sets)==set(range(1,31))
    train=data['train_indices'];mean=data['raw_sensor_features'][train].mean(0)
    scale=np.maximum(data['raw_sensor_features'][train].std(0),1e-6)
    close(mean,data['feature_mean']);close(scale,data['feature_scale'])
    np.testing.assert_array_equal(np.clip((data['raw_sensor_features']-mean)/scale,-6,6).astype(np.float32),data['x'])
    counts['full_train_only_standardization_and_subject_split']+=1
    archive=zipfile.ZipFile(study/'data/uci_har_official.zip')
    nested=[n for n in archive.namelist() if n.endswith('UCI HAR Dataset.zip')]
    inner=zipfile.ZipFile(io.BytesIO(archive.read(nested[0]))) if nested else archive
    prefix=min([n.rsplit('README.txt',1)[0] for n in inner.namelist() if n.endswith('/README.txt')],key=len)
    offset=0;rng=np.random.default_rng(417)
    for partition in ('train','test'):
        def raw_file(name):
            value=inner.read(prefix+name)
            assert hashlib.sha256(value).hexdigest()==split['archive_input_file_sha256'][name]
            counts['official_archive_file_hashes']+=1
            return value
        labels=np.loadtxt(io.BytesIO(raw_file(f'{partition}/y_{partition}.txt'))).astype(int)-1
        sid=np.loadtxt(io.BytesIO(raw_file(f'{partition}/subject_{partition}.txt'))).astype(int)
        np.testing.assert_array_equal(data['y'][offset:offset+len(labels)],labels)
        np.testing.assert_array_equal(data['subject_id'][offset:offset+len(sid)],sid)
        chosen=np.sort(rng.choice(len(labels),12,replace=False))
        sensors=[]
        for sensor in ('total_acc','body_gyro'):
            axes=[]
            for axis in 'xyz':
                payload=raw_file(f'{partition}/Inertial Signals/{sensor}_{axis}_{partition}.txt')
                lines=payload.splitlines()
                axes.append(np.loadtxt(io.BytesIO(b'\n'.join(lines[int(i)] for i in chosen))))
            sensors.append(raw_features(np.stack(axes,axis=2)))
        close(np.column_stack(sensors),data['raw_sensor_features'][offset+chosen],2e-6)
        counts['raw_sensor_windows_recomputed']+=len(chosen)
        offset+=len(labels)
    assert offset==len(data['y'])
    return data,split


def recovery(gaps,start,end):
    for rnd in range(start,end-1):
        if np.all(np.asarray(gaps[rnd-1:rnd+2])<=.02):return rnd-start
    return None


def audit(study,out):
    counts=Counter();protocol=load(study/'protocol.json');summary=load(study/'summary.json')
    data,split=audit_data(study,counts)
    for name,expected in protocol['source_sha256'].items():
        assert sha(study/'source_snapshot'/name)==sha(source_path(name))==expected
        counts['frozen_training_source_integrity']+=1
    x,y,sid=data['x'],data['y'],data['subject_id'];test=data['test_indices'];val=data['validation_indices']
    test_y=y[test];test_subject=sid[test];nodes=np.searchsorted(split['test_subjects'],test_subject)
    empirical=np.load(study/'empirical_joint_counts.npz',allow_pickle=False)
    cross=np.load(study/'crossnode_information_joint_counts.npz',allow_pickle=False)
    supplement=load(study/'crossnode_information_supplement.json')
    adaptation={(int(r['seed']),int(r['heldout_subject']),r['arm']):r for r in csvrows(study/'adaptation_summary.csv')}
    transfers={(int(r['seed']),int(r['heldout_subject'])):r for r in csvrows(study/'heldout_transfer.csv')}
    information={int(r['seed']):r for r in csvrows(study/'sensor_information.csv')}
    seed_info=[];seed_delta=[];seed_ablation=[];effects={k:[] for k in summary['paired_effects']};loaded_by_seed={}
    for seed in protocol['seeds']:
        saved={};scores={}
        for arm in protocol['arms']:
            run=study/'runs'/f's{seed}_{arm}';final=load(run/'final.json')
            assert final['test_evaluated_after_training'] is True
            assert set(final['training_subjects'])<=set(split['train_subjects'])
            with np.load(run/'test_probabilities.npz',allow_pickle=False) as z:saved[arm]={k:z[k] for k in z.files}
            with np.load(run/'models.npz',allow_pickle=False) as models:
                if arm=='local_only':
                    fs=[]
                    for i,source in enumerate(split['train_subjects']):
                        for stage in ('selected','final'):
                            a=[models[f'{stage}_c{source}_a{j}'] for j in range(4)]
                            close(predict(a,x[test]),saved[arm][stage][i],2e-7);counts['independent_saved_model_forwards']+=1
                        fs.append(macro(test_y,saved[arm]['selected'][i]))
                    scores[arm]=float(np.mean(fs));close(scores[arm],final['macro_f1'])
                else:
                    for stage in ('selected','final'):
                        a=[models[f'{stage}_a{j}'] for j in range(4)]
                        close(predict(a,view(x[test],arm)),saved[arm][stage],2e-7);counts['independent_saved_model_forwards']+=1
                    scores[arm]=macro(test_y,saved[arm]['selected']);close(scores[arm],final['macro_f1'])
                    prs=precision_recall_fscore_support(test_y,saved[arm]['selected'].argmax(1),labels=np.arange(6),average='macro',zero_division=0)
                    assert np.isfinite(prs[:3]).all();counts['independent_P_R_F1_checks']+=3
                    losses=[]
                    for r in range(30):
                        a=[models[f'trajectory_a{j}'][r] for j in range(4)]
                        pv=predict(a,view(x[val],arm));losses.append(float(log_loss(y[val],pv,labels=np.arange(6))))
                        if arm in ('full_fusion','adaptive_fusion'):
                            for fault,label in ((False,'trajectory_clean'),(11<=r+1<21,'trajectory_fault')):
                                close(predict(a,view(x[test],arm,fault)),saved[arm][label][r],2e-7)
                                counts['independent_fault_clean_trajectory_forwards']+=1
                    assert final['selected_round']==int(np.argmin(losses))+1
                    rows=csvrows(run/'rounds.csv');close(losses,[float(row['validation_clean_log_loss']) for row in rows])
                    expected_gyro=[0 if arm=='acc_only' or arm=='adaptive_fusion' and 11<=r+1<21 else 1 for r in range(30)]
                    assert [int(row['actual_train_gyro_coverage']) for row in rows]==expected_gyro
                    counts['validation_checkpoint_and_training_fault_timeline']+=1
            counts['run_manifest_membership']+=1
        a,g,f=(saved[arm]['selected'] for arm in ('acc_only','gyro_only','full_fusion'))
        pa,pg,pf=(p.argmax(1) for p in (a,g,f))
        joint=multivariate_count(test_y,pa,pg,pf,sizes=(6,6,6,6))
        node_joint=multivariate_count(nodes,test_y,pf,sizes=(9,6,6))
        np.testing.assert_array_equal(joint,empirical[f's{seed}_Y_acc_gyro_fused'])
        np.testing.assert_array_equal(node_joint,empirical[f's{seed}_subject_Y_fused'])
        info=information[seed]
        close(float(info['H_Y_accPred_gyroPred_fusedPred_bits']),entropy(joint.ravel(),base=2))
        close(float(info['I_Y_fused_given_acc_gyro_bits']),conditional_mi(joint.transpose(0,3,1,2)))
        close(float(info['I_Y_gyro_given_acc_bits']),conditional_mi(joint.sum(3).transpose(0,2,1)))
        close(float(info['I_Y_fused_given_subject_bits']),conditional_mi(node_joint.transpose(1,2,0)))
        close(float(info['I_subject_fused_bits']),mutual_info_score(None,None,contingency=node_joint.sum(1))/np.log(2))
        counts['independent_real_sensor_joint_and_MI_checks']+=7
        local_extra=[];local_delta=[]
        for i,source in enumerate(split['train_subjects']):
            lp=saved['local_only']['selected'][i].argmax(1)
            table=multivariate_count(nodes,test_y,pf,lp,sizes=(9,6,6,6))
            np.testing.assert_array_equal(table,cross[f's{seed}_local_subject_Y_global_local'][i])
            local_extra.append(conditional_mi(table.transpose(1,2,0,3)))
            global_mi=conditional_mi(table.sum(3).transpose(1,2,0))
            local_mi=conditional_mi(table.sum(2).transpose(1,2,0))
            local_delta.append(global_mi-local_mi);counts['independent_crossnode_conditional_MI']+=3
        seed_info.append(float(np.mean(local_extra)));seed_delta.append(float(np.mean(local_delta)))
        leave=saved['leave_node_out']['selected'].argmax(1)
        table=multivariate_count(nodes,test_y,pf,leave,sizes=(9,6,6,6))
        np.testing.assert_array_equal(table,cross[f's{seed}_subject_Y_global_leave_node_out'])
        seed_ablation.append(conditional_mi(table.transpose(1,2,0,3)))
        for key,reference in (('fusion_minus_acc','acc_only'),('fusion_minus_gyro','gyro_only'),
                              ('fusion_minus_isolated_local','local_only'),('fusion_minus_leave_node_out','leave_node_out')):
            effects[key].append(scores['full_fusion']-scores[reference])
        shock_by_arm={};auc_by_arm={}
        for subject in split['test_subjects']:
            keep=test_subject==subject;transfer=transfers[seed,subject]
            gf=macro(test_y[keep],f[keep]);lf=float(np.mean([macro(test_y[keep],p[keep]) for p in saved['local_only']['selected']]))
            of=macro(test_y[keep],saved['leave_node_out']['selected'][keep])
            for field,value in (('global_macro_f1',gf),('isolated_local_model_mean_macro_f1',lf),('KID_global_minus_isolated_macro_f1',gf-lf),
                                ('leave_one_training_node_out_macro_f1',of),('global_minus_leave_node_out_macro_f1',gf-of)):
                close(float(transfer[field]),value);counts['heldout_subject_transfer_values']+=1
            for arm in ('full_fusion','adaptive_fusion'):
                raw=saved[arm];cf=np.array([macro(test_y[keep],p[keep]) for p in raw['trajectory_clean']])
                df=np.array([macro(test_y[keep],p[keep]) for p in raw['trajectory_fault']])
                ff=np.array([macro(test_y[keep],p[keep]) for p in saved['full_fusion']['r10_frozen_fault']])
                row=adaptation[seed,subject,arm];target=cf[9]
                frozen_return=macro(test_y[keep],raw['r20_frozen_return_clean'][keep])
                expected={'immediate_fault_F1_drop':cf[10]-df[10],'shock_mean_fault_F1':df[10:20].mean(),
                    'shock_AUC_positive_deficit_F1_rounds':np.maximum(cf-df,0)[10:20].sum(),
                    'pre_fault_r10_clean_macro_f1':target,'shock_cumulative_F1_gain_vs_r10_frozen':(df-ff)[10:20].sum(),
                    'return_cumulative_update_gain_vs_r20_frozen_F1_rounds':(df-frozen_return)[20:30].sum()}
                for field,value in expected.items():close(float(row[field]),float(value));counts['independent_adaptation_endpoints']+=1
                for field,start,end in (('recovery_rounds_after_dropout',11,20),('return_recovery_rounds',21,30)):
                    actual=int(row[field]) if row[field] else None
                    assert actual==recovery(target-df,start,end);counts['recovery_censor_and_three_round_rules']+=1
                shock_by_arm.setdefault(arm,[]).append(expected['shock_mean_fault_F1'])
                auc_by_arm.setdefault(arm,[]).append(expected['shock_AUC_positive_deficit_F1_rounds'])
        effects['adaptive_minus_stale_shock_F1'].append(float(np.mean(shock_by_arm['adaptive_fusion'])-np.mean(shock_by_arm['full_fusion'])))
        effects['stale_minus_adaptive_shock_AUC_deficit'].append(float(np.mean(auc_by_arm['full_fusion'])-np.mean(auc_by_arm['adaptive_fusion'])))
    for field,values in effects.items():
        expected=interval(values);actual=summary['paired_effects'][field]
        close(expected['mean'],actual['mean']);close(expected['ci95'],actual['ci95']);close(expected['sd'],actual['sd'])
        counts['independent_paired_seed_statistics']+=1
    for values,actual in ((seed_info,supplement['metric16_global_additional_information']),
                          (seed_delta,supplement['seed_aggregated_information']['delta_MI_global_minus_other_given_subject_bits']),
                          (seed_ablation,supplement['leave_one_node_out_information']['I_Y_global_given_other_subject_bits'])):
        expected=interval(values);close(expected['mean'],actual['mean']);close(expected['ci95'],actual['ci95'])
        counts['independent_crossnode_seed_statistics']+=1
    empirical.close();cross.close()
    result={'status':'passed','checks':dict(counts),'fixed_subject_split':{'train':17,'validation':4,'test':9},
        'metric16_global_additional_information':interval(seed_info),'delta_MI_given_subject':interval(seed_delta),
        'global_given_leave_one_node_out':interval(seed_ablation),
        'paired_effects':{key:interval(value) for key,value in effects.items()},
        'verifier_sha256':sha(__file__),
        'scope':'Independent raw-sensor samples, full split/preprocessing, saved-model forward, contingency/conditional-MI, transfer and adaptation arithmetic. No training-module imports and no model retraining.',
        'limits':['One fixed17/4/9subject split and five training seeds; intervals reflect training randomness, not repeated subject/population sampling.',
            'Local-only85selected/final models independently forward-checked; full per-node30round validation trajectories were not saved, so local minimum-validation selection is source-inspected rather than independently replayed.',
            'MI is empirical label/prediction information with finite-cell upward-bias risk; utility and ablation evidence remain necessary.']}
    out.mkdir(parents=True,exist_ok=True)
    (out/'multimodal_independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'results/purpose_multimodal')
    parser.add_argument('--output',type=Path,default=ROOT/'results/purpose_cross_audit')
    args=parser.parse_args();audit(args.study,args.output)
