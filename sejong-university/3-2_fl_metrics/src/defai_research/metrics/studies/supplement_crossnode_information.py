from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
from ._paths import ROOT, source_path, relative_path
import numpy as np
from scipy.stats import t as student_t


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ci(values):
    values=np.asarray(values,float);mean=float(values.mean());sd=float(values.std(ddof=1));half=float(student_t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)))
    return {'mean':mean,'sd':sd,'ci95':[mean-half,mean+half],'n_training_seeds':len(values),'values':values.tolist()}
def conditional_information(counts):
    counts=np.asarray(counts,float).reshape(counts.shape[0],counts.shape[1],-1)
    p=counts/counts.sum();pac=p.sum(axis=1,keepdims=True);pbc=p.sum(axis=0,keepdims=True);pc=p.sum(axis=(0,1),keepdims=True)
    numerator=p*pc;denominator=pac*pbc;valid=p>0
    return float(np.sum(p[valid]*np.log2(numerator[valid]/denominator[valid])))
def joint_by_subject(y,global_prediction,other_prediction,node,nodes):
    flat=((node*6+y)*6+global_prediction)*6+other_prediction
    return np.bincount(flat,minlength=nodes*6**3).reshape(nodes,6,6,6)
def information_from_joint(joint):
    extra_global=conditional_information(joint.transpose(1,2,0,3))
    extra_local=conditional_information(joint.transpose(1,3,0,2))
    mi_global=conditional_information(joint.sum(axis=3).transpose(1,2,0))
    mi_local=conditional_information(joint.sum(axis=2).transpose(1,2,0))
    if abs((extra_global-extra_local)-(mi_global-mi_local))>1e-10:raise AssertionError('Conditional information identity failed')
    return {'I_Y_global_given_other_subject_bits':extra_global,'I_Y_other_given_global_subject_bits':extra_local,
            'I_Y_global_given_subject_bits':mi_global,'I_Y_other_given_subject_bits':mi_local,
            'delta_MI_global_minus_other_given_subject_bits':mi_global-mi_local}


def run(root):
    original_protocol=sha(root/'protocol.json');protocol=load_json(root/'protocol.json');summary=load_json(root/'summary.json');split=load_json(root/'data'/'subject_split.json')
    with np.load(root/'data'/'features_and_subject_splits.npz') as z:
        index=z['test_indices'];y=z['y'][index];sid=z['subject_id'][index]
    node=np.searchsorted(split['test_subjects'],sid);nsub=len(split['test_subjects']);counts={};seed_rows=[];subject_rows=[];local_rows=[];ablation_rows=[]
    input_hashes={}
    for seed in protocol['seeds']:
        paths={arm:root/'runs'/f's{seed}_{arm}'/'test_probabilities.npz' for arm in ('full_fusion','local_only','leave_node_out')}
        input_hashes[str(seed)]={arm:sha(path) for arm,path in paths.items()}
        loaded={}
        for arm,path in paths.items():
            with np.load(path) as z:loaded[arm]=z['selected'].argmax(axis=-1)
        g=loaded['full_fusion'];locals=loaded['local_only'];global_local=[];subject_values={s:[] for s in split['test_subjects']}
        check=len(locals)==len(split['train_subjects'])
        if not check:raise AssertionError('Exactly17local source models required')
        seed_counts=[]
        for source_subject,lp in zip(split['train_subjects'],locals):
            joint=joint_by_subject(y,g,lp,node,nsub);seed_counts.append(joint)
            values=information_from_joint(joint);global_local.append(values)
            local_rows.append({'seed':seed,'training_source_subject':source_subject,**values})
            for k,s in enumerate(split['test_subjects']):
                value=information_from_joint(joint[k:k+1]);subject_values[s].append(value)
        counts[f's{seed}_local_subject_Y_global_local']=np.stack(seed_counts)
        average={key:float(np.mean([r[key] for r in global_local])) for key in global_local[0]}
        seed_rows.append({'seed':seed,**average})
        for s,values in subject_values.items():
            subject_rows.append({'seed':seed,'heldout_subject':s,**{key:float(np.mean([r[key] for r in values])) for key in values[0]}})
        ablation_joint=joint_by_subject(y,g,loaded['leave_node_out'],node,nsub)
        counts[f's{seed}_subject_Y_global_leave_node_out']=ablation_joint
        ablation_rows.append({'seed':seed,**information_from_joint(ablation_joint)})
    fields=list(seed_rows[0]);fields.remove('seed')
    estimates={field:ci([r[field] for r in seed_rows]) for field in fields}
    ablation_estimates={field:ci([r[field] for r in ablation_rows]) for field in fields}
    count_path=root/'crossnode_information_joint_counts.npz';np.savez_compressed(count_path,**counts)
    result={'schema':1,'analysis_type':'post-hoc additional measurement requested after main30runs;no selection/training/main-gate changes',
            'main_protocol_sha256':original_protocol,'supplement_source_sha256':sha(__file__),'prediction_input_sha256':input_hashes,
            'seeds':protocol['seeds'],'train_subjects':split['train_subjects'],'heldout_subjects':split['test_subjects'],
            'definitions':{'representative_metric16':'Arithmetic mean across17isolated local source models of empirical I(Y;G_global|L_local,heldout_subject), then paired training-seed summary.',
                           'delta_MI':'I(Y;G_global|heldout_subject)-mean_local I(Y;L_local|heldout_subject); equals extra-global minus extra-local conditional information.',
                           'weighting':'Equal17local-model average; categorical joint uses actual heldout-window mass across9subjects. Per-subject values separately preserved.',
                           'ablation':'Same joint construction comparing global17-source model and model trained with smallest training source subject removed. One node only, not exhaustive causal attribution.',
                           'CI':'Unadjusted95%t intervals across5training seeds on one fixed subject split; overlapping windows/9subjects are not treated as independent replicated experiments.'},
            'metric16_global_additional_information':estimates['I_Y_global_given_other_subject_bits'],
            'seed_aggregated_information':estimates,'leave_one_node_out_information':ablation_estimates,
            'seed_values':seed_rows,'heldout_subject_values':subject_rows,'local_source_values':local_rows,'leave_node_out_seed_values':ablation_rows,
            'utility_evidence_from_main':{'fusion_minus_isolated_local_F1':summary['paired_effects']['fusion_minus_isolated_local'],
                                          'fusion_minus_leave_node_out_F1':summary['paired_effects']['fusion_minus_leave_node_out']},
            'joint_counts':{'path':count_path.name,'sha256':sha(count_path),'axes_local':['local_training_source','heldout_subject','true_activity','global_prediction','local_prediction'],
                            'axes_ablation':['heldout_subject','true_activity','global_prediction','leave_one_node_out_prediction']},
            'limits':'Label-based predictive information; direct F1gain and node-ablation utility remain necessary. Finite categorical plug-in estimates can be biased upward; no causal/DP/full-feature mutual-information interpretation. Main original success gates unchanged.',
            'main_protocol_unchanged':sha(root/'protocol.json')==original_protocol}
    path=root/'crossnode_information_supplement.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    export={'metric_index':16,'name':'Additional global predictive information beyond isolated local source models','unit':'bit',
            'value':result['metric16_global_additional_information'],'definition':result['definitions']['representative_metric16'],
            'conditions':{'delta_MI_given_subject':estimates['delta_MI_global_minus_other_given_subject_bits'],
                          'global_vs_leave_node_out_additional_information':ablation_estimates['I_Y_global_given_other_subject_bits'],
                          **result['utility_evidence_from_main']},'posthoc':True,'scope':result['limits'],'source':path.name}
    (root/'report_export_metric16_supplement.json').write_text(json.dumps(export,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'metric16':result['metric16_global_additional_information'],'delta_MI':estimates['delta_MI_global_minus_other_given_subject_bits'],
                      'leave_node_out':ablation_estimates['I_Y_global_given_other_subject_bits'],'main_protocol_unchanged':result['main_protocol_unchanged']}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,default=ROOT/'results'/'purpose_multimodal');args=parser.parse_args();run(args.results)
