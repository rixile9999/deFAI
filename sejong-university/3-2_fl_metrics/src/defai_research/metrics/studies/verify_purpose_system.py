from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

from ._paths import ROOT, source_path, relative_path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score,log_loss


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b,atol=1e-9):np.testing.assert_allclose(a,b,atol=atol,rtol=1e-7)


def frame_decode(raw):
    assert len(raw)>=16
    assert struct.unpack_from('<Q',raw)[0]==len(raw)-8
    size=struct.unpack_from('<I',raw,8)[0];meta=json.loads(raw[12:12+size])
    count=struct.unpack_from('<I',raw,12+size)[0];offset=16+size
    arrays=[];tensor_bytes=0
    for _ in range(count):
        length=struct.unpack_from('<Q',raw,offset)[0];offset+=8
        value=raw[offset:offset+length];assert len(value)==length
        arrays.append(np.load(io.BytesIO(value),allow_pickle=False));offset+=length;tensor_bytes+=length
    assert offset==len(raw)
    return meta,arrays,tensor_bytes


def codec_decode(arrays,bits,n):
    if bits==32:
        assert len(arrays)==1 and arrays[0].dtype==np.float32
        return arrays[0]
    assert len(arrays)==2
    codes,scales=arrays
    assert scales.dtype==np.float32 and len(scales)==(n+255)//256
    if bits==4:
        assert codes.dtype==np.uint8 and len(codes)==(n+1)//2
        unpacked=np.column_stack((codes%16,codes//16)).reshape(-1).astype(np.int16)-7
        assert np.max(np.abs(unpacked))<=7
        codes=unpacked[:n]
    else:
        assert bits==8 and codes.dtype==np.int8 and len(codes)==n
    return (codes.astype(np.float32)*np.repeat(scales,256)[:n]).astype(np.float32)


def flatten(arrays):return np.concatenate([a.reshape(-1) for a in arrays])


def predicted(arrays,x):
    hidden=np.maximum(x@arrays[0]+arrays[1],0)
    logits=hidden@arrays[2]+arrays[3];logits-=logits.max(1,keepdims=True)
    p=np.exp(logits);return p/p.sum(1,keepdims=True)


def weighted_model(old,updates,weights):
    locals=[]
    for update in updates:
        vector=flatten(old).astype(np.float64)+update.astype(np.float64)
        offset=0;arrays=[]
        for reference in old:
            arrays.append(vector[offset:offset+reference.size].reshape(reference.shape).astype(np.float32));offset+=reference.size
        locals.append(arrays)
    return [sum(arrays[i]*n for arrays,n in zip(locals,weights))/sum(weights) for i in range(4)]


def verify_run(run,protocol,counts):
    final=load(run/'final.json');split=load(run/'source_split.json')
    assert final['source_sha256']==protocol['source_sha256']
    assert final['seed'] in protocol['seeds'] and final['codec'] in protocol['codecs']
    assert final['rounds']==protocol['rounds'] and final['clients']==3
    train,val,test=map(set,(split['train_ids'],split['validation_ids'],split['test_ids']))
    assert not (train&val or train&test or val&test)
    for ids in split['phase_source_ids']['initial']:assert set(ids)<=train
    x=load_digits().data.astype(np.float32)/16;y=load_digits().target.astype(np.int64)
    assert train|val|test==set(range(len(y)))
    counts['source_disjointness']+=1
    events=load(run/'wire_events.json')
    rounds=list(csv.DictReader((run/'rounds.csv').open(encoding='utf-8-sig',newline='')))
    assert len(rounds)==final['rounds']
    parsed={};measured_by_round=Counter();tensors_by_round=Counter();bodies_by_round=Counter();pids=[]
    with np.load(run/'wire_frames.npz',allow_pickle=False) as frames:
        assert len(frames.files)==len(events)
        for event in events:
            raw=frames[event['key']].tobytes()
            assert hashlib.sha256(raw).hexdigest()==event['sha256']
            assert len(raw)==event['frame_bytes']==event['body_bytes']+8
            meta,arrays,tensor_bytes=frame_decode(raw)
            assert meta['kind']==event['kind'] and tensor_bytes==event['tensor_bytes']
            cid=event['cid']
            if event['phase']=='training':
                rnd=event['round'];assert meta['round']==rnd and meta['cid']==cid
                measured_by_round[rnd]+=len(raw);bodies_by_round[rnd]+=len(raw)-8;tensors_by_round[rnd]+=tensor_bytes
                key=(rnd,cid,meta['kind']);assert key not in parsed
                parsed[key]=(meta,arrays)
            elif meta['kind']=='ready':
                pids.append(meta['pid']);assert meta['n']==len(split['phase_source_ids']['initial'][cid])
            counts['independent_frame_decodes']+=1
    assert len(pids)==len(set(pids))==3
    weights=[len(ids) for ids in split['phase_source_ids']['initial']]
    worker_updates=[np.load(run/f'worker{cid}_updates.npz',allow_pickle=False) for cid in range(3)]
    distortions=list(csv.DictReader((run/'update_distortion.csv').open(encoding='utf-8-sig',newline='')))
    with np.load(run/'models_by_round.npz',allow_pickle=False) as models, np.load(run/'validation_predictions.npz',allow_pickle=False) as validation:
        assert np.array_equal(validation['source_ids'],split['validation_ids'])
        assert np.array_equal(validation['y'],y[validation['source_ids']])
        val_losses=[];val_f1=[];relative=[];mses=[];cosines=[]
        for index,row in enumerate(rounds):
            rnd=index+1
            assert int(row['round'])==rnd
            old=[models[f'r{rnd:02d}_old_{i}'] for i in range(4)]
            applied=[models[f'r{rnd:02d}_applied_{i}'] for i in range(4)]
            decoded=[];base_decoded=[];raw_updates=[]
            for cid in range(3):
                request,arrays=parsed[rnd,cid,'fit'];assert request['codec']==final['codec']
                for a,b in zip(arrays,old):np.testing.assert_array_equal(a,b)
                header,encoded=parsed[rnd,cid,'update'];assert header['n']==weights[cid]
                assert header['examples_processed']==2*weights[cid]
                assert header['optimizer_steps']==2*((weights[cid]+31)//32)
                base=codec_decode(encoded,header['bits'],header['length'])
                current=base.copy()
                if final['codec']=='int4_repair16':
                    repair,residual=parsed[rnd,cid,'repair']
                    assert residual[0].dtype==np.float16 and residual[0].shape==base.shape
                    current=(base.astype(np.float64)+residual[0].astype(np.float64)).astype(np.float32)
                    np.testing.assert_array_equal(residual[0],worker_updates[cid][f'r{rnd:02d}_residual16'])
                np.testing.assert_array_equal(base,worker_updates[cid][f'r{rnd:02d}_base'])
                np.testing.assert_array_equal(current,worker_updates[cid][f'r{rnd:02d}_restored'])
                raw_updates.append(worker_updates[cid][f'r{rnd:02d}_raw']);decoded.append(current);base_decoded.append(base)
                counts['client_update_wire_reconstructions']+=1
            expected=weighted_model(old,decoded,weights)
            for a,b in zip(applied,expected):close(a,b,2e-6)
            p=predicted(applied,x[validation['source_ids']]);close(p,validation['probabilities'][index],2e-7)
            ce=float(log_loss(validation['y'],p,labels=np.arange(10)))
            score=float(f1_score(validation['y'],p.argmax(1),average='macro',zero_division=0))
            val_losses.append(ce);val_f1.append(score)
            close(ce,float(row['validation_log_loss']));close(score,float(row['validation_macro_F1']))
            assert int(row['framed_bytes'])==measured_by_round[rnd]
            assert int(row['body_bytes'])==bodies_by_round[rnd] and int(row['tensor_bytes'])==tensors_by_round[rnd]
            assert int(row['examples_processed'])==2*sum(weights)
            assert int(row['optimizer_steps'])==sum(2*((n+31)//32) for n in weights)
            original=np.concatenate(raw_updates).astype(np.float64)
            difference=np.concatenate(decoded).astype(np.float64)-original
            damage=np.concatenate(base_decoded).astype(np.float64)-original
            norm=float(np.linalg.norm(difference)/max(np.linalg.norm(original),1e-12))
            mse=float(np.mean(difference**2))
            cosine=float(np.dot(original,original+difference)/max(np.linalg.norm(original)*np.linalg.norm(original+difference),1e-20))
            close(norm,float(distortions[index]['relative_L2']));close(mse,float(distortions[index]['MSE']))
            close(cosine,float(distortions[index]['cosine']))
            relative.append(norm);mses.append(mse);cosines.append(cosine)
            if rnd==final['selected_round']:
                recovery=final['selected_recovery']
                denominator=np.linalg.norm(damage)
                iri=(denominator-np.linalg.norm(difference))/denominator if denominator>1e-12 else None
                assert (iri is None)==(recovery['vector_recovery_fraction'] is None)
                if iri is not None:close(iri,recovery['vector_recovery_fraction'])
                with np.load(run/'selected_predictions.npz',allow_pickle=False) as stored:
                    for split_name,ids in (('val',split['validation_ids']),('test',split['test_ids'])):
                        measures={}
                        for name,values in (('reference',raw_updates),('damaged',base_decoded),('restored',decoded)):
                            counterfactual=weighted_model(old,values,weights)
                            prob=predicted(counterfactual,x[ids]);close(prob,stored[f'{split_name}_{name}'],3e-7)
                            f=float(f1_score(y[ids],prob.argmax(1),average='macro',zero_division=0))
                            ce=float(log_loss(y[ids],prob,labels=np.arange(10)))
                            close(f,recovery[f'{split_name}_{name}_macro_F1']);close(ce,recovery[f'{split_name}_{name}_log_loss'],3e-7)
                            measures[name]=(f,ce)
                        for metric,column in (('F1',0),('logloss',1)):
                            if metric=='F1':
                                numerator=measures['restored'][0]-measures['damaged'][0]
                                headroom=measures['reference'][0]-measures['damaged'][0]
                            else:
                                numerator=measures['damaged'][1]-measures['restored'][1]
                                headroom=measures['damaged'][1]-measures['reference'][1]
                            fraction=recovery[f'{split_name}_{metric}_recovery_fraction']
                            assert (fraction is None)==(headroom<=1e-8)
                            if fraction is not None:close(fraction,numerator/headroom,1e-4)
                            counts['headroom_sensitive_utility_recovery']+=1
            counts['round_model_and_loss_recomputations']+=1
        assert final['selected_round']==int(np.argmin(val_losses))+1
        for target in final['TC_targets']:
            hit=None
            for index in range(2,len(rounds)):
                if all(f>=target['threshold'] for f in val_f1[index-2:index+1]):hit=index;break
            assert target['reached']==(hit is not None)
            assert target['completion_round']==(hit+1 if hit is not None else None)
            if hit is not None:
                close(target['TC_loopback_wall_seconds'],float(rounds[hit]['cumulative_wall_seconds']))
                assert target['framed_bytes_to_target']==sum(int(r['framed_bytes']) for r in rounds[:hit+1])
            else:assert target['TC_loopback_wall_seconds'] is None and target['framed_bytes_to_target'] is None
            counts['sustained_target_TC_and_censoring']+=1
        close(final['mean_update_relative_L2'],np.mean(relative));close(final['mean_update_MSE'],np.mean(mses));close(final['mean_update_cosine'],np.mean(cosines))
    for z in worker_updates:z.close()
    assert final['training_framed_bytes']==sum(measured_by_round.values())
    assert final['session_framed_bytes']==sum(event['frame_bytes'] for event in events)
    close(final['loopback_pipeline_application_Bps'],final['training_framed_bytes']/final['measured_loopback_pipeline_wall_seconds'])
    close(final['test_F1_per_MiB'],final['test_macro_F1']/(final['training_framed_bytes']/2**20))
    resources=load(run/'worker_resources.json')
    assert {r['pid'] for r in resources}==set(pids)
    for resource in resources:
        assert resource['process_CPU_seconds']>=0 and len(resource['sampled_RSS_MiB'])==final['rounds']
        close(resource['sampled_peak_RSS_MiB'],max(resource['sampled_RSS_MiB']))
        assert resource['OS_lifetime_peak_RSS_MiB']>=resource['sampled_peak_RSS_MiB']
        counts['separate_process_CPU_and_RSS_observations']+=1
    close(final['worker_CPU_seconds'],sum(r['process_CPU_seconds'] for r in resources))
    close(final['all_process_CPU_seconds'],final['coordinator_CPU_seconds']+final['worker_CPU_seconds'])
    return pids


def verify_echo(study,echo):
    protocol=load(echo/'protocol.json');main=load(study/'protocol.json');aggregate=load(echo/'summary.json')
    assert protocol['source_sha256']==sha(source_path('purpose_transport_echo.py'))
    assert protocol['main_protocol_sha256']==sha(study/'protocol.json')
    assert protocol['jobs']==main['ordered_jobs']
    assert protocol['warmup_cycles']==10 and protocol['measured_cycles']==200
    assert aggregate['status']=='completed' and aggregate['count']==len(protocol['jobs'])==20
    pids=[];total_frames=0
    for job,session in zip(protocol['jobs'],aggregate['sessions']):
        assert all(session[key]==value for key,value in job.items())
        run=study/'runs'/f's{job["seed"]}_{job["codec"]}'
        events=load(run/'wire_events.json')
        original=[e for e in events if e['phase']=='training' and e['round']==1]
        assert session['frame_shapes_bytes']==[e['frame_bytes'] for e in original]
        assert session['replayed_frame_sha256']==[e['sha256'] for e in original]
        assert session['all_received_frames_exact'] is True
        expected=2*sum(e['frame_bytes'] for e in original)*200
        assert session['measured_roundtrip_framed_bytes']==expected
        assert session['server_observed_frames_with_warmup']==len(original)*210
        close(session['loopback_echo_application_bits_per_second'],expected*8/session['measured_elapsed_seconds'])
        pids.append(session['pid']);total_frames+=len(original)*200
    assert len(pids)==len(set(pids))==20
    result={'status':'passed','sessions':20,'fresh_worker_PIDs':len(set(pids)),
        'measured_roundtrip_frame_exchanges':total_frames,
        'all_main_payload_hashes_and_frame_lengths_match':True,
        'all_warmup_exclusions_and_bits_per_second_arithmetic_match':True,
        'scope':'Checks stored replay provenance and exact-transfer assertion/arithmetic; not external observation or physical link capacity.'}
    (echo/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'results/purpose_system/main')
    parser.add_argument('--echo',type=Path)
    args=parser.parse_args();protocol=load(args.study/'protocol.json')
    counts=Counter();failures=[];pids=[]
    for name,expected in protocol['source_sha256'].items():
        assert sha(args.study/'source_snapshot'/name)==sha(source_path(name))==expected
        counts['frozen_source_integrity']+=1
    planned={(r['seed'],r['codec']) for r in protocol['ordered_jobs']}
    actual=set()
    for run in sorted((args.study/'runs').iterdir()):
        if not (run/'final.json').exists():continue
        try:
            pids.extend(verify_run(run,protocol,counts))
            final=load(run/'final.json');actual.add((final['seed'],final['codec']))
        except Exception as exc:failures.append({'run':run.name,'error':f'{type(exc).__name__}: {exc}'})
    if actual!=planned:failures.append({'scope':'manifest','error':'Missing, failed or unplanned runs'})
    assert len(pids)==len(set(pids)),'PID reuse requires separate process-instance evidence'
    result={'status':'passed' if not failures else 'failed','verified_runs':len(actual),'expected_runs':len(planned),
        'checks':dict(counts),'unique_worker_PIDs':len(set(pids)),'failures':failures,
        'verifier_sha256':sha(__file__),'scope':'Independent stored-frame, reconstruction, heldout model and arithmetic audit; loopback TCP custom Flower transport; no external timing or WAN replication.'}
    if args.echo:result['echo_verification']=verify_echo(args.study,args.echo)
    (args.study/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
