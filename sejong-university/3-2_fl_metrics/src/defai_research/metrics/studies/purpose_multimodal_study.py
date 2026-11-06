from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['FLWR_TELEMETRY_ENABLED'] = '0'
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

from ._paths import ROOT, source_path, relative_path
import numpy as np
import scipy
from scipy.stats import t as student_t
import sklearn
from sklearn.metrics import f1_score, log_loss
import flwr
from flwr.client import NumPyClient
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from threadpoolctl import threadpool_limits
from defai_research.metrics import classification_metrics
_THREADS = threadpool_limits(limits=1)

OFFICIAL_URL = 'https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip'
OFFICIAL_PAGE = 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones'
SEEDS = tuple(range(8101, 8106))
ARMS = ('full_fusion', 'acc_only', 'gyro_only', 'local_only', 'leave_node_out', 'adaptive_fusion')
ROUNDS, DROP_START, RETURN_START = 30, 11, 21
FEATURES_PER_SENSOR = 36
BIN_EDGES = np.asarray([-np.inf, -3, -2, -1, -.5, 0, .5, 1, 2, 3, np.inf])


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, keys); writer.writeheader(); writer.writerows(rows)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy_counts(counts):
    p = np.asarray(counts, dtype=np.float64).ravel()
    if p.sum() == 0: return 0.
    p = p[p > 0] / p.sum()
    return float(-np.sum(p * np.log2(p)))


def categorical_joint(*variables, sizes):
    out = np.zeros(sizes, np.int64)
    np.add.at(out, tuple(variables), 1)
    return out


def conditional_mi(joint, left_axes, right_axes, condition_axes):
    def h(axes):
        axes = tuple(sorted(set(axes)))
        drop = tuple(axis for axis in range(joint.ndim) if axis not in axes)
        return entropy_counts(joint.sum(axis=drop))
    return float(max(0., h((*left_axes, *condition_axes)) + h((*right_axes, *condition_axes))
                     - h(condition_axes) - h((*left_axes, *right_axes, *condition_axes))))


def sensor_features(signal):
    signal = np.asarray(signal, np.float64)
    if signal.ndim != 3 or signal.shape[1:] != (128, 3): raise ValueError('Expected128x3 sensor windows')
    stats = [signal.mean(1), signal.std(1), np.sqrt(np.mean(signal**2, 1)), signal.min(1), signal.max(1),
             np.median(signal, 1), np.quantile(signal, .75, axis=1)-np.quantile(signal, .25, axis=1),
             np.mean(np.abs(np.diff(signal, axis=1)), axis=1)]
    power = np.abs(np.fft.rfft(signal, axis=1) / 128)**2
    frequency = np.fft.rfftfreq(128, d=1/50)
    for low, high in ((0, 2.5), (2.5, 7.5), (7.5, 25.0001)):
        stats.append(power[:, (frequency >= low) & (frequency < high), :].sum(1))
    magnitude = np.linalg.norm(signal, axis=2)
    stats.append(np.stack((magnitude.mean(1), magnitude.std(1), np.sqrt(np.mean(magnitude**2, 1))), axis=1))
    out = np.concatenate(stats, axis=1).astype(np.float32)
    assert out.shape == (len(signal), FEATURES_PER_SENSOR)
    return out


def prepare_data(output, download=False):
    directory = output / 'data'; directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / 'uci_har_official.zip'
    if not archive_path.exists():
        if not download: raise FileNotFoundError(f'Download official archive to {archive_path} or use--download')
        with urllib.request.urlopen(OFFICIAL_URL, timeout=120) as response, archive_path.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    provenance = {'download_url': OFFICIAL_URL, 'official_dataset_page': OFFICIAL_PAGE, 'doi': '10.24432/C54S4K',
                  'license_on_official_page': 'CC BY4.0; original archived README also preserved',
                  'archive_sha256': sha(archive_path), 'archive_bytes': archive_path.stat().st_size}
    feature_path = directory / 'features_and_subject_splits.npz'
    if feature_path.exists():
        split = json.loads((directory/'subject_split.json').read_text(encoding='utf-8'))
        if split['archive_sha256'] != provenance['archive_sha256']: raise RuntimeError('Archive changed')
        if sha(feature_path) != split['feature_file_sha256']: raise RuntimeError('Prepared dataset changed')
        with np.load(feature_path) as loaded: data = {key: loaded[key] for key in loaded.files}
        return data, split
    with zipfile.ZipFile(archive_path) as outer:
        nested = [name for name in outer.namelist() if name.endswith('UCI HAR Dataset.zip')]
        source = zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
        names = source.namelist(); prefixes = [name.rsplit('README.txt', 1)[0] for name in names if name.endswith('/README.txt')]
        prefix = min(prefixes, key=len)
        manifest = {}; acc, gyro, labels, subjects, official = [], [], [], [], []
        def load(name):
            payload = source.read(prefix + name); manifest[name] = hashlib.sha256(payload).hexdigest()
            return np.loadtxt(io.BytesIO(payload))
        for partition in ('train', 'test'):
            a = np.stack([load(f'{partition}/Inertial Signals/total_acc_{axis}_{partition}.txt') for axis in 'xyz'], axis=2)
            g = np.stack([load(f'{partition}/Inertial Signals/body_gyro_{axis}_{partition}.txt') for axis in 'xyz'], axis=2)
            acc.append(sensor_features(a)); gyro.append(sensor_features(g))
            labels.append(load(f'{partition}/y_{partition}.txt').astype(np.int64)-1)
            subjects.append(load(f'{partition}/subject_{partition}.txt').astype(np.int64))
            official.append(np.full(len(a), int(partition == 'test'), np.int8))
        (directory/'UCI_HAR_README.txt').write_bytes(source.read(prefix+'README.txt'))
        (directory/'UCI_HAR_features_info.txt').write_bytes(source.read(prefix+'features_info.txt'))
    y, sid, official = np.concatenate(labels), np.concatenate(subjects), np.concatenate(official)
    raw = np.concatenate((np.concatenate(acc), np.concatenate(gyro)), axis=1)
    official_train = sorted(np.unique(sid[official == 0]).tolist())
    val_subjects, train_subjects = official_train[-4:], official_train[:-4]
    test_subjects = sorted(np.unique(sid[official == 1]).tolist())
    train = np.flatnonzero(np.isin(sid, train_subjects)); val = np.flatnonzero(np.isin(sid, val_subjects)); test = np.flatnonzero(np.isin(sid, test_subjects))
    assert not set(train_subjects)&set(val_subjects) and not set(official_train)&set(test_subjects)
    mean, scale = raw[train].mean(0), raw[train].std(0)
    scale = np.maximum(scale, 1e-6)
    x = np.clip((raw-mean)/scale, -6, 6).astype(np.float32)
    data = {'x':x, 'y':y, 'subject_id':sid, 'official_test':official, 'source_window_id':np.arange(len(y)),
            'train_indices':train, 'validation_indices':val, 'test_indices':test, 'feature_mean':mean, 'feature_scale':scale,
            'raw_sensor_features':raw, 'entropy_bin_edges':BIN_EDGES}
    np.savez_compressed(feature_path, **data)
    split = {**provenance, 'archive_input_file_sha256':manifest, 'train_subjects':train_subjects,
             'validation_subjects':val_subjects, 'test_subjects':test_subjects, 'n_train':len(train), 'n_validation':len(val), 'n_test':len(test),
             'feature_file_sha256':sha(feature_path), 'sensors':['actual_total_acc_xyz', 'actual_body_gyro_xyz'],
             'features_per_sensor':36, 'sensor_window':'50Hz;128samples;2.56seconds;50%overlap in official collection',
             'leakage_control':'All windows of a subject assigned to exactly one split. No time-window row split or test/calibration training.',
             'standardization':'Mean/std fitted on17training subjects only; clip standardized values to[-6,6]; missing sensor imputed training mean(standardized0).',
             'features':'peraxis mean,std,RMS,min,max,median,IQR,meanabsdiff and FFTenergy bands[0,2.5),[2.5,7.5),[7.5,25];magnitude mean,std,RMS',
             'subject_counts':{str(s):int(np.sum(sid == s)) for s in np.unique(sid)}}
    write_json(directory/'download_provenance.json', provenance); write_json(directory/'subject_split.json', split)
    return data, split


def model(seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 981]))
    return [rng.normal(0, math.sqrt(2/72), (72,32)).astype(np.float32), np.zeros(32,np.float32),
            rng.normal(0, math.sqrt(1/32), (32,6)).astype(np.float32), np.zeros(6,np.float32)]


def predict(arrays, x):
    logits = np.maximum(x @ arrays[0]+arrays[1], 0) @ arrays[2]+arrays[3]
    logits = logits-logits.max(axis=1, keepdims=True); p = np.exp(logits); return p/p.sum(1,keepdims=True)


class SensorClient(NumPyClient):
    def __init__(self, cid, x, y, seed): self.cid,self.x,self.y,self.seed=cid,x,y,seed
    def fit(self, parameters, config):
        arrays = [array.copy().astype(np.float32) for array in parameters]
        rnd = int(config['round']); lr=.08/math.sqrt(1+(rnd-1)/20); steps=0
        order=np.random.default_rng(np.random.SeedSequence([self.seed,self.cid,rnd,661])).permutation(len(self.y))
        for start in range(0,len(order),64):
            ids=order[start:start+64]; x,y=self.x[ids],self.y[ids]
            pre=x@arrays[0]+arrays[1]; h=np.maximum(pre,0); q=h@arrays[2]+arrays[3]
            q-=q.max(1,keepdims=True); q=np.exp(q);q/=q.sum(1,keepdims=True);q[np.arange(len(y)),y]-=1;q/=len(y)
            dh=(q@arrays[2].T)*(pre>0)
            grads=[x.T@dh+1e-4*arrays[0],dh.sum(0),h.T@q+1e-4*arrays[2],q.sum(0)]
            arrays=[(a-lr*g).astype(np.float32) for a,g in zip(arrays,grads)];steps+=1
        return arrays,len(self.y),{'optimizer_steps':steps,'examples_processed':len(self.y)}


def view(x, arm, fault=False):
    out=x.copy()
    if arm == 'gyro_only': out[:,:36]=0
    if arm == 'acc_only' or fault: out[:,36:]=0
    return out


def score(y,p):
    measured = classification_metrics(y, p)
    return {key: measured[key] for key in ('macro_precision', 'macro_recall', 'macro_f1', 'accuracy')} | {
        'log_loss': float(log_loss(y,p,labels=np.arange(6)))}


def freeze(output, split, seeds, rounds):
    source_hash={Path(__file__).name:sha(__file__), 'api.py':sha(source_path('api.py'))}
    protocol={'schema':1,'study':'purpose-specific actual synchronized sensor evaluation','seeds':list(seeds),'arms':list(ARMS),
              'rounds':rounds,'drop_start':DROP_START,'return_start':RETURN_START,'source_sha256':source_hash,
              'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'flwr':flwr.__version__},
              'split':split,'budget':{'run_count':len(seeds)*len(ARMS),'epochs_per_round':1,'batch_size':64,'hidden':32,'base_lr':.08,'l2':.0001},
              'checkpoint':'Minimum CLEAN heldout-validation-subject log-loss. local_only each isolated model independently uses the same validation subjects. No test selection.',
              'test_lock':'No tuning or candidate selection; all six predeclared arms and all five seeds retained regardless of outcome. Evaluate test only after each complete training trajectory/checkpoint selection.',
              'adaptation':'adaptive_fusion actually retrains on gyro-missing inputs at ALL17training nodes rounds11-20, return21-30. Heldout9subjects never train. full_fusion normally observed training is stale-update comparator under same inference fault; r10 frozen model gives no-update comparator.',
              'comparators':'Same initialization/minibatch RNG/72dimMLP/rounds across arms. Single sensors zero the other physical group. local_only has17independent models, no aggregation/ensemble; report mean isolated model F1. leave_node_out omits smallest training subject ID only; sensitivity not exhaustive Shapley contribution.',
              'MUI':'Actual synchronized acc/gyro-only predictive distributions: per-window Jensen-Shannon divergence(bits), modality coverage indicators and heldout modal/Fusion F1. Divergence alone has no better direction.',
              'KID':'Same heldout subject global_F1 minus mean isolated-source-model F1; paired global minus leave-node-out F1 sensitivity. Direct transfer utility, not knowledge amount or exact original proposal formula.',
              'entropy':'Training-node label entropy and each standardized physical-feature marginal entropy in common fixed z bins[-inf,-3,-2,-1,-.5,0,.5,1,2,3,inf]. Not high-dimensional joint feature entropy.',
              'joint_MI':'Save Yx(accPred,gyroPred,fusedPred) 6^4 counts and subject-node x Y x fusedPred. Empirical entropies, I(Y;F|A,G), I(Y;G|A), I(Y;F|subject), I(subject;F). Link to actual fusion/ablation heldout utility; MI is not a performance substitute.',
              'adaptation_endpoints':'Per-heldout-subject paired fault/clean trajectories; immediate F1drop, shock integrated positive clean-fault deficit(sum F1-round), recovery first3consecutive rounds at faultF1>=ownr10cleanF1-.02(censored if absent), return recovery against same pre-fault target; adaptive minus stale and r10frozen under identical fault. Return cumulative update gain compared with ownr20frozen model under returned sensors.',
              'failure_criteria':'All outcomes retained. Fusion superiority unsupported unless paired seed95%CI lower>0 vs BOTH single sensors. Transfer unsupported if CI lower<=0 vs isolated local. Adaptation benefit unsupported if adaptive-minus-stale faultF1 CI lower<=0 OR stale-minus-adaptive shock deficit CI lower<=0. MI nonzero alone never success. Intervals use5training seeds on one fixed subject split; no new-population inference.',
              'actual_flower':'NumPyClient.to_client().fit(FitIns), FedAvg.aggregate_fit; unquantized serial host execution; bytes are real Flower serialized tensors, not WAN measurement.',
              'privacy':'No differential privacy; no privacy-budget claim.'}
    path=output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=protocol: raise RuntimeError('Frozen protocol/source changed; use a new output directory')
    else:
        write_json(path,protocol); dest=output/'source_snapshot';dest.mkdir()
        for name in source_hash: shutil.copyfile(source_path(name),dest/name)
    return protocol,hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run_one(output,data,split,seed,arm,rounds,protocol_hash):
    out=output/'runs'/f's{seed}_{arm}';out.mkdir(parents=True,exist_ok=True)
    if (out/'final.json').exists():
        saved=json.loads((out/'final.json').read_text(encoding='utf-8'))
        if saved['protocol_sha256']!=protocol_hash:raise RuntimeError('Resume mismatch')
        return saved
    x,y,sid=data['x'],data['y'],data['subject_id'];train_subjects=split['train_subjects']
    clients=[s for s in train_subjects if arm!='leave_node_out' or s!=min(train_subjects)]
    indices={s:np.flatnonzero(sid==s) for s in clients};vi=data['validation_indices'];ti=data['test_indices']
    arrays=model(seed); local={s:[a.copy() for a in arrays] for s in clients}
    strategy=FedAvg(accept_failures=False,inplace=False,fit_metrics_aggregation_fn=lambda _: {})
    trajectories=[];selected=None;best_loss=np.inf;best_round=0
    local_best={s:np.inf for s in clients};local_round={s:0 for s in clients};local_selected={}
    history=[];start_wall,start_cpu=time.perf_counter(),time.process_time()
    for rnd in range(1,rounds+1):
        shock=DROP_START<=rnd<RETURN_START and arm=='adaptive_fusion';fit_results=[];models=[];weights=[];steps=0;byte_count=0
        for s in clients:
            ids=indices[s];vx=view(x[ids],arm,shock);old=local[s] if arm=='local_only' else arrays
            request=FitIns(ndarrays_to_parameters(old),{'round':rnd})
            result=SensorClient(s,vx,y[ids],seed).to_client().fit(request)
            fitted=parameters_to_ndarrays(result.parameters);fit_results.append((None,result));models.append(fitted);weights.append(result.num_examples)
            steps+=int(result.metrics['optimizer_steps']);byte_count+=sum(len(t) for t in request.parameters.tensors)+sum(len(t) for t in result.parameters.tensors)
            if arm=='local_only':
                local[s]=fitted;vp=predict(fitted,view(x[vi],arm));loss=score(y[vi],vp)['log_loss']
                if loss<local_best[s]:local_best[s]=loss;local_round[s]=rnd;local_selected[s]=[a.copy() for a in fitted]
        if arm=='local_only':
            trajectories.append([[a.copy() for a in local[s]] for s in clients])
            validation=float(np.mean([score(y[vi],predict(local[s],view(x[vi],arm)))['log_loss'] for s in clients]))
            gap=0.
        else:
            aggregate,_=strategy.aggregate_fit(rnd,fit_results,failures=[]);arrays=parameters_to_ndarrays(aggregate)
            exact=[sum(n*m[j].astype(np.float64) for n,m in zip(weights,models))/sum(weights) for j in range(4)]
            gap=max(float(np.max(np.abs(a-e))) for a,e in zip(arrays,exact))
            if gap>2e-6:raise AssertionError('Flower weighted aggregation discrepancy')
            trajectories.append([a.copy() for a in arrays]);validation=score(y[vi],predict(arrays,view(x[vi],arm)))['log_loss']
            if validation<best_loss:best_loss,best_round,selected=validation,rnd,[a.copy() for a in arrays]
        history.append({'round':rnd,'actual_train_gyro_coverage':0 if shock or arm=='acc_only' else 1,
                        'actual_train_acc_coverage':0 if arm=='gyro_only' else 1,'validation_clean_log_loss':validation,
                        'examples_processed':sum(weights),'optimizer_steps':steps,'flower_tensor_bytes':byte_count,'fedavg_reference_max_abs_gap':gap})
    train_cpu,train_wall=time.process_time()-start_cpu,time.perf_counter()-start_wall
    saved_models={}; probabilities={};node_rows=[];curve=[]
    if arm=='local_only':
        selected_p=[];final_p=[]
        for s in clients:
            for j,a in enumerate(local_selected[s]):saved_models[f'selected_c{s}_a{j}']=a
            for j,a in enumerate(local[s]):saved_models[f'final_c{s}_a{j}']=a
            sp=predict(local_selected[s],view(x[ti],arm));fp=predict(local[s],view(x[ti],arm));selected_p.append(sp);final_p.append(fp)
            for ts in split['test_subjects']:
                mask=sid[ti]==ts;node_rows.append({'seed':seed,'arm':arm,'source_subject':s,'test_subject':ts,'selected_round':local_round[s],**score(y[ti][mask],sp[mask])})
        probabilities={'selected':np.stack(selected_p),'final':np.stack(final_p)}
        metrics={key:float(np.mean([score(y[ti],p)[key] for p in selected_p])) for key in ('macro_precision','macro_recall','macro_f1','accuracy','log_loss')}
    else:
        for j in range(4):saved_models[f'trajectory_a{j}']=np.stack([a[j] for a in trajectories])
        for prefix,aa in (('selected',selected),('final',arrays)):
            for j,a in enumerate(aa):saved_models[f'{prefix}_a{j}']=a
            probabilities[prefix]=predict(aa,view(x[ti],arm))
        clean=[];fault=[];frozen=[]
        frozen_arrays=trajectories[min(9,rounds-1)]
        for rnd,aa in enumerate(trajectories,1):
            is_fault=DROP_START<=rnd<RETURN_START
            cp=predict(aa,view(x[ti],arm));dp=predict(aa,view(x[ti],arm,is_fault));fr=predict(frozen_arrays,view(x[ti],arm,is_fault))
            clean.append(cp);fault.append(dp);frozen.append(fr)
            for ts in split['test_subjects']:
                mask=sid[ti]==ts
                curve.append({'seed':seed,'arm':arm,'round':rnd,'test_subject':ts,'gyro_coverage':int(not is_fault and arm!='acc_only'),
                              'acc_coverage':int(arm!='gyro_only'),'clean_macro_f1':score(y[ti][mask],cp[mask])['macro_f1'],
                              'fault_macro_f1':score(y[ti][mask],dp[mask])['macro_f1'],'r10_frozen_fault_macro_f1':score(y[ti][mask],fr[mask])['macro_f1']})
        probabilities.update(trajectory_clean=np.stack(clean),trajectory_fault=np.stack(fault),r10_frozen_fault=np.stack(frozen),
                             r20_frozen_return_clean=predict(trajectories[min(19,rounds-1)],view(x[ti],arm)))
        metrics=score(y[ti],probabilities['selected'])
        for ts in split['test_subjects']:
            mask=sid[ti]==ts;node_rows.append({'seed':seed,'arm':arm,'source_subject':'global','test_subject':ts,'selected_round':best_round,**score(y[ti][mask],probabilities['selected'][mask])})
    selected_for_classification = probabilities['selected']
    matrices = [classification_metrics(y[ti], p) for p in selected_for_classification] if arm == 'local_only' else [classification_metrics(y[ti], selected_for_classification)]
    write_json(out/'classification_metrics.json', {'scope':'Macro over six fixed activity classes; individual class denominators retained', 'models':matrices})
    np.savez_compressed(out/'models.npz',**saved_models);np.savez_compressed(out/'test_probabilities.npz',**probabilities)
    write_csv(out/'rounds.csv',history);write_csv(out/'heldout_subject_metrics.csv',node_rows);write_csv(out/'adaptation_curves.csv',curve)
    result={'seed':seed,'arm':arm,'protocol_sha256':protocol_hash,'test_evaluated_after_training':True,'training_subjects':clients,
            'selected_round':local_round if arm=='local_only' else best_round,'rounds':rounds,'train_pipeline_cpu_seconds':train_cpu,'train_pipeline_wall_seconds':train_wall,
            'sum_examples_processed':sum(r['examples_processed'] for r in history),'sum_optimizer_steps':sum(r['optimizer_steps'] for r in history),
            'sum_flower_tensor_bytes':sum(r['flower_tensor_bytes'] for r in history),**metrics}
    write_json(out/'final.json',result);return result


def mean_ci(values):
    a=np.asarray(values,np.float64);m=float(a.mean());half=float(student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
    return {'mean':m,'sd':float(a.std(ddof=1)) if len(a)>1 else 0.,'ci95':[m-half,m+half],'n_training_seeds':len(a),'values':a.tolist()}


def recovery_round(gaps,start,end,threshold=.02):
    for r in range(start,end-1):
        if all(gaps[k-1]<=threshold for k in range(r,r+3)):return r-start
    return None


def summarize(output,data,split,rows,seeds,rounds):
    y=data['y'][data['test_indices']];sid=data['subject_id'][data['test_indices']]
    unique_subjects=split['test_subjects'];node=np.searchsorted(unique_subjects,sid);info=[];transfers=[];adapt=[];joint_saved={}
    lookup={(r['seed'],r['arm']):r for r in rows}
    for seed in seeds:
        loaded={}
        for arm in ARMS:
            with np.load(output/'runs'/f's{seed}_{arm}'/'test_probabilities.npz') as z:loaded[arm]={k:z[k] for k in z.files}
        a,g,f=(loaded[arm]['selected'] for arm in ('acc_only','gyro_only','full_fusion'));pa,pg,pf=a.argmax(1),g.argmax(1),f.argmax(1)
        joint=categorical_joint(y,pa,pg,pf,sizes=(6,6,6,6));node_joint=categorical_joint(node,y,pf,sizes=(len(unique_subjects),6,6))
        joint_saved[f's{seed}_Y_acc_gyro_fused']=joint;joint_saved[f's{seed}_subject_Y_fused']=node_joint
        mid=(a.astype(float)+g)/2;jsd=float(np.mean(np.sum((a*np.log2(np.maximum(a,1e-30)/mid)+g*np.log2(np.maximum(g,1e-30)/mid))/2,axis=1)))
        info.append({'seed':seed,'MUI_real_sensor_predictive_JSD_bits':jsd,'acc_sensor_coverage':1,'gyro_sensor_coverage':1,
                     'acc_macro_f1':lookup[seed,'acc_only']['macro_f1'],'gyro_macro_f1':lookup[seed,'gyro_only']['macro_f1'],'fused_macro_f1':lookup[seed,'full_fusion']['macro_f1'],
                     'H_Y_accPred_gyroPred_fusedPred_bits':entropy_counts(joint),'I_Y_fused_given_acc_gyro_bits':conditional_mi(joint,(0,),(3,),(1,2)),
                     'I_Y_gyro_given_acc_bits':conditional_mi(joint.sum(3),(0,),(2,),(1,)),
                     'H_subject_Y_fusedPred_bits':entropy_counts(node_joint),'I_Y_fused_given_subject_bits':conditional_mi(node_joint,(1,),(2,),(0,)),
                     'I_subject_fused_bits':conditional_mi(node_joint.sum(1),(0,),(1,),())})
        for ts in unique_subjects:
            mask=sid==ts;gf1=score(y[mask],f[mask])['macro_f1'];localscores=[score(y[mask],p[mask])['macro_f1'] for p in loaded['local_only']['selected']]
            lf1=score(y[mask],loaded['leave_node_out']['selected'][mask])['macro_f1']
            transfers.append({'seed':seed,'heldout_subject':ts,'global_macro_f1':gf1,'isolated_local_model_mean_macro_f1':float(np.mean(localscores)),
                              'KID_global_minus_isolated_macro_f1':gf1-float(np.mean(localscores)),
                              'leave_one_training_node_out_macro_f1':lf1,'global_minus_leave_node_out_macro_f1':gf1-lf1})
            for arm in ('full_fusion','adaptive_fusion'):
                clean=loaded[arm]['trajectory_clean'];fault=loaded[arm]['trajectory_fault'];freeze_p=loaded['full_fusion']['r10_frozen_fault']
                cf=np.asarray([score(y[mask],p[mask])['macro_f1'] for p in clean]);df=np.asarray([score(y[mask],p[mask])['macro_f1'] for p in fault])
                ff=np.asarray([score(y[mask],p[mask])['macro_f1'] for p in freeze_p]);gaps=np.maximum(cf-df,0)
                target=float(cf[min(9,rounds-1)]);target_gaps=target-df
                return_frozen=score(y[mask],loaded[arm]['r20_frozen_return_clean'][mask])['macro_f1']
                adapt.append({'seed':seed,'heldout_subject':ts,'arm':arm,'immediate_fault_F1_drop':float(cf[DROP_START-1]-df[DROP_START-1]),
                              'shock_mean_fault_F1':float(df[DROP_START-1:min(RETURN_START-1,rounds)].mean()),
                              'shock_AUC_positive_deficit_F1_rounds':float(gaps[DROP_START-1:min(RETURN_START-1,rounds)].sum()),
                              'pre_fault_r10_clean_macro_f1':target,
                              'recovery_rounds_after_dropout':recovery_round(target_gaps,DROP_START,min(RETURN_START-1,rounds)),
                              'return_recovery_rounds':recovery_round(target_gaps,RETURN_START,rounds) if rounds>=RETURN_START else None,
                              'shock_cumulative_F1_gain_vs_r10_frozen':float((df-ff)[DROP_START-1:min(RETURN_START-1,rounds)].sum()),
                              'return_cumulative_update_gain_vs_r20_frozen_F1_rounds':float((df-return_frozen)[RETURN_START-1:rounds].sum()),
                              'recovery_censor_note':'null=not within2F1pp of pre-fault r10cleanF1 for3consecutive observations; immediate sensor-return improvement does not alone demonstrate learned adaptation'})
    entropy_rows=[];train_x=data['x'][data['train_indices']];train_y=data['y'][data['train_indices']];train_sid=data['subject_id'][data['train_indices']]
    for s in split['train_subjects']:
        mask=train_sid==s;counts=np.bincount(train_y[mask],minlength=6)
        for j in range(72):
            hist=np.histogram(train_x[mask,j],bins=BIN_EDGES)[0]
            entropy_rows.append({'training_subject':s,'modality':'acc' if j<36 else 'gyro','feature_index':j,'marginal_feature_entropy_bits':entropy_counts(hist),
                                 'node_label_entropy_bits':entropy_counts(counts),'feature_bin_counts_json':json.dumps(hist.tolist()),'label_counts_json':json.dumps(counts.tolist())})
    write_csv(output/'sensor_information.csv',info);write_csv(output/'heldout_transfer.csv',transfers);write_csv(output/'adaptation_summary.csv',adapt);write_csv(output/'local_train_entropy.csv',entropy_rows)
    np.savez_compressed(output/'empirical_joint_counts.npz',**joint_saved)
    effects={}
    for key,reference in (('fusion_minus_acc','acc_only'),('fusion_minus_gyro','gyro_only'),('fusion_minus_isolated_local','local_only'),('fusion_minus_leave_node_out','leave_node_out')):
        effects[key]=mean_ci([lookup[s,'full_fusion']['macro_f1']-lookup[s,reference]['macro_f1'] for s in seeds])
    paired=[]
    for seed in seeds:
        aa=[r for r in adapt if r['seed']==seed and r['arm']=='adaptive_fusion'];ss=[r for r in adapt if r['seed']==seed and r['arm']=='full_fusion']
        paired.append({'seed':seed,'adaptive_minus_stale_shock_F1':float(np.mean([r['shock_mean_fault_F1'] for r in aa])-np.mean([r['shock_mean_fault_F1'] for r in ss])),
                       'stale_minus_adaptive_shock_AUC_deficit':float(np.mean([r['shock_AUC_positive_deficit_F1_rounds'] for r in ss])-np.mean([r['shock_AUC_positive_deficit_F1_rounds'] for r in aa]))})
    effects['adaptive_minus_stale_shock_F1']=mean_ci([r['adaptive_minus_stale_shock_F1'] for r in paired])
    effects['stale_minus_adaptive_shock_AUC_deficit']=mean_ci([r['stale_minus_adaptive_shock_AUC_deficit'] for r in paired])
    summary={'status':'complete','runs':len(rows),'rounds':rounds,'seeds':list(seeds),'arm_test_macro_f1':{arm:mean_ci([lookup[s,arm]['macro_f1'] for s in seeds]) for arm in ARMS},'arm_test_macro_precision':{arm:mean_ci([lookup[s,arm]['macro_precision'] for s in seeds]) for arm in ARMS},'arm_test_macro_recall':{arm:mean_ci([lookup[s,arm]['macro_recall'] for s in seeds]) for arm in ARMS},'paired_effects':effects,
             'claim_gates':{'fusion_superiority_vs_both_modalities':all(effects[k]['ci95'][0]>0 for k in ('fusion_minus_acc','fusion_minus_gyro')),
                            'positive_transfer_vs_isolated':effects['fusion_minus_isolated_local']['ci95'][0]>0,
                            'learning_adaptation_benefit':all(effects[k]['ci95'][0]>0 for k in ('adaptive_minus_stale_shock_F1','stale_minus_adaptive_shock_AUC_deficit'))},
             'seed_level_adaptation_effects':paired,'limits':'Purpose-specific improved definitions, not exact restoration of ambiguous original formulas.5training seeds on one fixed subject split; repeated/overlapping test windows not independent subjects; no MI-to-performance/causality/privacy guarantee.',
             'source_unchanged':sha(__file__)==json.loads((output/'protocol.json').read_text(encoding='utf-8'))['source_sha256'][Path(__file__).name]}
    write_json(output/'summary.json',summary)
    export={'study':'UCI HAR actual synchronized accelerometer+gyroscope;5training seeds;fixed17/4/9subjectsplit',
            'scope':'Purpose-specific improved definitions; empirical categorical/marginal entropy, no high-dimensional entropy/DP claim. Confidence intervals reflect training randomness on this one fixed dataset split.',
            'source_summary':'summary.json','rows':[
                {'metric_index':6,'name':'MUI actual sensor complementarity','unit':'bit;F1 proportion;sensor coverage',
                 'value':mean_ci([r['MUI_real_sensor_predictive_JSD_bits'] for r in info]),
                 'conditions':{'acc_F1':summary['arm_test_macro_f1']['acc_only'],'gyro_F1':summary['arm_test_macro_f1']['gyro_only'],
                               'fusion_F1':summary['arm_test_macro_f1']['full_fusion'],'acc_coverage':1,'gyro_coverage':1},
                 'interpretation':'Actual sensor predictive JSD has no universally favorable direction; utility requires fusion-vs-single-sensor F1.'},
                {'metric_index':7,'name':'Adaptation after actual gyro-loss training and return','unit':'F1 proportion;F1-round;round',
                 'value':effects['adaptive_minus_stale_shock_F1'],'conditions':{'stale_minus_adaptive_shock_AUC_deficit':effects['stale_minus_adaptive_shock_AUC_deficit'],
                   'shock_rounds':[DROP_START,RETURN_START-1],'return_rounds':[RETURN_START,rounds],
                   'raw_per_subject_recovery_and_r10_r20_frozen_controls':'adaptation_summary.csv'},
                 'interpretation':'Actual Flower training on faulted source-node inputs; heldout subject labels never train. Censored recovery remains null.'},
                {'metric_index':8,'name':'KID heldout-subject transfer utility','unit':'F1 proportion difference','value':effects['fusion_minus_isolated_local'],
                 'conditions':{'global_minus_leave_one_node_out':effects['fusion_minus_leave_node_out'],'local_models':17,'raw_subject_effects':'heldout_transfer.csv'},
                 'interpretation':'Global transfer vs independent local source models on identical heldout records; one prespecified omitted node is sensitivity, not exhaustive causal attribution.'},
                {'metric_index':11,'name':'Local training empirical entropy','unit':'bit per label/marginal feature',
                 'value':{'mean_node_label_entropy_bits':float(np.mean([entropy_counts(np.bincount(train_y[train_sid==s],minlength=6)) for s in split['train_subjects']])),
                          'mean_acc_marginal_feature_entropy_bits':float(np.mean([r['marginal_feature_entropy_bits'] for r in entropy_rows if r['modality']=='acc'])),
                          'mean_gyro_marginal_feature_entropy_bits':float(np.mean([r['marginal_feature_entropy_bits'] for r in entropy_rows if r['modality']=='gyro']))},
                 'conditions':{'nodes':17,'features_per_sensor':36,'common_bins':'[-inf,-3,-2,-1,-.5,0,.5,1,2,3,inf] after train-only normalization'},
                 'interpretation':'Fixed-bin feature marginals and categorical labels; not full multivariate feature entropy.'},
                {'metric_index':12,'name':'Real multimodal joint information','unit':'bit','value':mean_ci([r['H_Y_accPred_gyroPred_fusedPred_bits'] for r in info]),
                 'conditions':{'I_Y_fused_given_acc_gyro_bits':mean_ci([r['I_Y_fused_given_acc_gyro_bits'] for r in info]),
                               'I_Y_gyro_given_acc_bits':mean_ci([r['I_Y_gyro_given_acc_bits'] for r in info]),'joint_axes':['true_activity','acc_prediction','gyro_prediction','fusion_prediction']},
                 'interpretation':'Empirical6^4categorical count table linked to actual modality ablations; information alone is not utility.'},
                {'metric_index':16,'name':'Subject-node conditional mutual information','unit':'bit','value':mean_ci([r['I_Y_fused_given_subject_bits'] for r in info]),
                 'conditions':{'I_subject_fused_bits':mean_ci([r['I_subject_fused_bits'] for r in info]),'H_subject_Y_fused_bits':mean_ci([r['H_subject_Y_fusedPred_bits'] for r in info]),
                               'joint_axes':['heldout_subject_node','true_activity','fusion_prediction']},
                 'interpretation':'Node-conditional predictive information with direct cross-node transfer/leave-node-out F1 alongside; not proof of integration.'}]}
    export['rows'] = [
        {'metric_index': i, 'name': name, 'unit':'proportion',
         'value': summary[field]['full_fusion'],
         'conditions': {'arm': 'full_fusion', 'classes':6, 'heldout_subjects':len(split['test_subjects']), 'comparison_arms':{arm:summary[field][arm] for arm in ('acc_only','gyro_only')}, 'classwise_evidence':'runs/<run>/classification_metrics.json'},
         'interpretation': 'Computed from actual held-out predictions and explicit confusion-matrix denominators'}
        for i,name,field in ((1,'Precision','arm_test_macro_precision'), (2,'Recall','arm_test_macro_recall'), (3,'F1','arm_test_macro_f1'))
    ] + export['rows']
    write_json(output/'report_export.json',export);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results'/'purpose_multimodal');parser.add_argument('--download',action='store_true')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data,split=prepare_data(args.output,args.download)
    if args.prepare_only:print(json.dumps({'status':'prepared',**split}));return
    if args.smoke:
        args.output=args.output/'smoke';args.output.mkdir(parents=True,exist_ok=True)
        data={**data,'test_indices':data['validation_indices']}
        split={**split,'test_subjects':split['validation_subjects'],'n_test':split['n_validation'],
               'smoke_only':'Validation subjects reused for diagnostic forwards; no official test forward.'}
    seeds=SEEDS[:1] if args.smoke else SEEDS;rounds=3 if args.smoke else ROUNDS
    protocol,digest=freeze(args.output,split,seeds,rounds);rows=[]
    for seed in seeds:
        for arm in ARMS:
            rows.append(run_one(args.output,data,split,seed,arm,rounds,digest));write_csv(args.output/'final_metrics.csv',rows)
            print(json.dumps({'completed':len(rows),'total':len(seeds)*len(ARMS),'seed':seed,'arm':arm}),flush=True)
    summary=summarize(args.output,data,split,rows,seeds,rounds) if not args.smoke else {'status':'smoke_passed','runs':len(rows),'source_unchanged':sha(__file__)==protocol['source_sha256'][Path(__file__).name]}
    write_json(args.output/'completed.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
