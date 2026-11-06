from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import struct
import sys
import threading
import time
from ._paths import ROOT, source_path, relative_path
import numpy as np
from scipy.stats import t as student_t
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score,log_loss
from threadpoolctl import threadpool_limits
_THREADS=threadpool_limits(limits=1)

SEEDS=tuple(range(8201,8206));SHAPES=((64,32),(32,),(32,10),(10,));N=sum(int(np.prod(s)) for s in SHAPES)
ERASE_START,ERASE_END=0,32


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def vector(arrays):return np.concatenate([a.ravel() for a in arrays])
def shapes(v):
    out=[];offset=0
    for shape in SHAPES:
        size=int(np.prod(shape));out.append(v[offset:offset+size].reshape(shape).astype(np.float32));offset+=size
    assert offset==len(v);return out
def array_hash(arrays):return hashlib.sha256(vector(arrays).astype('<f4').tobytes()).hexdigest()
def predict(arrays,x):
    hidden=np.maximum(x@arrays[0]+arrays[1],0);q=hidden@arrays[2]+arrays[3];q-=q.max(1,keepdims=True)
    q=np.exp(q);return q/q.sum(1,keepdims=True)
def metrics(y,p):return {'macro_F1':float(f1_score(y,p.argmax(1),labels=np.arange(10),average='macro',zero_division=0)),
                         'log_loss':float(log_loss(y,p,labels=np.arange(10)))}
def ratio(numerator,denominator,tolerance=1e-8):
    return (float(numerator/denominator),'defined') if denominator>tolerance else (None,'nonpositive_or_tiny_reference_headroom')
def npy(a):
    stream=io.BytesIO();np.save(stream,a,allow_pickle=False);return stream.getvalue()
def un_npy(payload):return np.load(io.BytesIO(payload),allow_pickle=False)


def frame(meta,arrays=()):
    header=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();tensors=[npy(a) for a in arrays]
    body=struct.pack('<I',len(header))+header+struct.pack('<I',len(tensors))
    body+=b''.join(struct.pack('<Q',len(t))+t for t in tensors)
    return struct.pack('<Q',len(body))+body
def exact_recv(conn,n):
    value=bytearray()
    while len(value)<n:
        chunk=conn.recv(n-len(value))
        if not chunk:raise EOFError('Truncated frame')
        value.extend(chunk)
    return bytes(value)
def receive(conn):
    header=exact_recv(conn,8);size=struct.unpack('<Q',header)[0]
    if size>1_000_000:raise ValueError('Oversized frame')
    return header+exact_recv(conn,size)
def decode(raw):
    if struct.unpack_from('<Q',raw)[0]!=len(raw)-8:raise ValueError('Invalid frame length')
    hlen=struct.unpack_from('<I',raw,8)[0];meta=json.loads(raw[12:12+hlen]);count=struct.unpack_from('<I',raw,12+hlen)[0];offset=16+hlen;tensors=[]
    for _ in range(count):
        size=struct.unpack_from('<Q',raw,offset)[0];offset+=8;tensors.append(raw[offset:offset+size]);offset+=size
    if offset!=len(raw):raise ValueError('Trailing/truncated tensor')
    return meta,[un_npy(t) for t in tensors],[len(t) for t in tensors]


def source_inputs(study):
    inputs={}
    for seed in SEEDS:
        run=study/'runs'/f's{seed}_fp32';final=read_json(run/'final.json')
        inputs[str(seed)]={'selected_round':final['selected_round'],'final_json_sha256':sha(run/'final.json'),
                           'models_by_round_sha256':sha(run/'models_by_round.npz'),'source_split_sha256':sha(run/'source_split.json')}
    return inputs


def freeze(output,study):
    inputs=source_inputs(study)
    protocol={'schema':1,'study':'Recovery sensitivity positive-loss block-erasure control','seeds':list(SEEDS),'cases':5,
              'source_sha256':{Path(__file__).name:sha(__file__)},'parent_system_protocol_sha256':sha(study/'protocol.json'),'parent_input_sha256':inputs,
              'source_models':'Actual existing FP32 validation-selected checkpoint; same source split/labels. No retraining, checkpoint reselection, or test-based damage selection.',
              'fixed_loss':{'matrix':'first hidden weight W1','shape':[64,32],'erase_input_row_start_inclusive':ERASE_START,'erase_input_row_end_exclusive':ERASE_END,
                            'erase_value':0,'erased_parameter_count':1024,'total_model_parameter_count':N},
              'repair':'FP16 residual only at1024erased positions plus explicit packed little-bit loss bitmap; receiver adds actual received residual to damaged FP32 model. No clipping of recovery fractions.',
              'transport':'Three request frames(reference model,damaged model,repair16+loss bitmap), each with actual receiver ACK over127.0.0.1TCP.8-byte body length,4-byte JSON size,JSON,4-byte tensor count,8-byte tensor lengths,actual NPY tensors. Loss injection is controlled application-layer stress over reliable TCP, not observed natural packet/WAN loss.',
              'vector_IRI':'1-L2(restored-reference)/L2(damaged-reference);denominator<=1e-12 ->NA',
              'utility_IRI':'(Urestored-Udamaged)/(Ureference-Udamaged), U=macroF1 or -logloss;positive reference headroom>1e-8required,elseNA;all cases retained,unclipped.',
              'exact_repair_control':'FP32 exact residual arithmetic control only,not transmitted; not claimed as the experimental repair.',
              'budget':'Exactly5fixedseed cases,one session each,zero new fit calls; no alternate damaged block if headroom fails.',
              'limits':'Positive-loss sensitivity stress complements preserved normal INT4 steady-state NA cases; does not establish reliability/probability under natural network failures or production repair policy.'}
    path=output/'protocol.json'
    if path.exists():
        if read_json(path)!=protocol:raise RuntimeError('Frozen stress source/input changed; use a new output directory')
    else:
        write_json(path,protocol);dest=output/'source_snapshot';dest.mkdir();shutil.copyfile(__file__,dest/Path(__file__).name)
    return protocol


def transport(seed,reference,damaged,mask,residual16):
    listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM);listener.settimeout(20);listener.bind(('127.0.0.1',0));listener.listen(1)
    port=listener.getsockname()[1];received={};raw_frames={};events=[];errors=[]
    def record(direction,raw):
        meta,arrays,sizes=decode(raw);key=f'f{len(events):02d}_{direction}_{meta["kind"]}';raw_frames[key]=np.frombuffer(raw,np.uint8).copy()
        events.append({'key':key,'direction':direction,'kind':meta['kind'],'frame_bytes':len(raw),'body_bytes':len(raw)-8,
                       'NPY_tensor_bytes':sum(sizes),'tensor_sizes':sizes,'sha256':hashlib.sha256(raw).hexdigest()})
    def receiver():
        try:
            conn,address=listener.accept()
            if address[0]!='127.0.0.1':raise AssertionError('Only127.0.0.1allowed')
            with conn:
                conn.settimeout(20)
                for expected in ('reference','damaged','repair16'):
                    raw=receive(conn);record('client_to_receiver',raw);meta,aa,_=decode(raw)
                    if meta['kind']!=expected or meta['seed']!=seed:raise AssertionError('Request mismatch')
                    if expected!='repair16':
                        if len(aa)!=4 or any(a.dtype!=np.float32 or a.shape!=shape for a,shape in zip(aa,SHAPES)):raise AssertionError('Model tensor mismatch')
                        received[expected]=aa
                    else:
                        bitmap,correction=aa
                        if bitmap.dtype!=np.uint8 or len(bitmap)!=(N+7)//8 or correction.dtype!=np.float16:raise AssertionError('Repair dtype/shape')
                        bits=np.unpackbits(bitmap,bitorder='little');location=bits[:N].astype(bool)
                        if bits[N:].any() or location.sum()!=1024 or len(correction)!=1024:raise AssertionError('Loss bitmap mismatch')
                        expected_mask=np.zeros(N,bool);expected_mask[:1024]=True
                        if not np.array_equal(location,expected_mask):raise AssertionError('Predeclared loss changed')
                        if meta['damaged_model_sha256']!=array_hash(received['damaged']):raise AssertionError('Repair target changed')
                        restored=vector(received['damaged']).astype(np.float64);restored[location]+=correction.astype(np.float64)
                        received['restored']=shapes(restored)
                    target=received['restored' if expected=='repair16' else expected]
                    ack=frame({'kind':'ack','request_kind':expected,'seed':seed,'applied_model_sha256':array_hash(target)})
                    conn.sendall(ack);record('receiver_to_client',ack)
        except BaseException as exc:errors.append(repr(exc))
        finally:listener.close()
    thread=threading.Thread(target=receiver,daemon=True);thread.start();client=socket.socket();client.settimeout(20)
    started=time.perf_counter()
    try:
        client.connect(('127.0.0.1',port));client.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        requests=[frame({'kind':'reference','seed':seed,'model_dtype':'float32'},reference),
                  frame({'kind':'damaged','seed':seed,'application_loss':'W1rows0:32erased'},damaged),
                  frame({'kind':'repair16','seed':seed,'damaged_model_sha256':array_hash(damaged),'bitmap_bitorder':'little','parameter_count':N,
                         'residual_dtype':'float16','loss_count':1024},[np.packbits(mask,bitorder='little'),residual16])]
        for request in requests:
            client.sendall(request);ack=receive(client);meta,_,_=decode(ack)
            if meta['kind']!='ack' or meta['seed']!=seed:raise AssertionError('Invalid receiver acknowledgement')
    finally:client.close();thread.join(25)
    if thread.is_alive() or errors:raise RuntimeError(f'Receiver failed:{errors}')
    return received,raw_frames,events,time.perf_counter()-started


def run_one(seed,output,study):
    dest=output/'cases'/f's{seed}';dest.mkdir(parents=True,exist_ok=True);parent=study/'runs'/f's{seed}_fp32';info=read_json(parent/'final.json');rnd=info['selected_round']
    with np.load(parent/'models_by_round.npz') as z:reference=[z[f'r{rnd:02d}_applied_{j}'] for j in range(4)]
    damaged=[a.copy() for a in reference];damaged[0][ERASE_START:ERASE_END,:]=0
    mask=np.zeros(N,bool);mask[:1024]=True;rv,dv=vector(reference),vector(damaged);residual16=(rv[mask].astype(np.float64)-dv[mask].astype(np.float64)).astype(np.float16)
    received,frames,events,elapsed=transport(seed,reference,damaged,mask,residual16)
    for name,expected in (('reference',reference),('damaged',damaged)):
        for a,b in zip(received[name],expected):np.testing.assert_array_equal(a,b)
    exact_vector=dv+(rv-dv);np.testing.assert_array_equal(exact_vector,rv);received['exact_FP32_aux']=shapes(exact_vector)
    split=read_json(parent/'source_split.json');digits=load_digits();x=digits.data.astype(np.float32)/16;y=digits.target.astype(np.int64)
    shutil.copyfile(parent/'source_split.json',dest/'source_split.json')
    saved_models={f'{name}_a{j}':a for name,arrays in received.items() for j,a in enumerate(arrays)};np.savez_compressed(dest/'received_models.npz',**saved_models)
    np.savez_compressed(dest/'wire_frames.npz',**frames);write_json(dest/'wire_events.json',events)
    prediction_arrays={};measures={}
    for label,id_key in (('val','validation_ids'),('test','test_ids')):
        ids=np.asarray(split[id_key],np.int64);prediction_arrays[f'{label}_source_ids']=ids;prediction_arrays[f'{label}_y']=y[ids]
        for name,arrays in received.items():
            p=predict(arrays,x[ids]);prediction_arrays[f'{label}_{name}']=p;measures[f'{label}_{name}']=metrics(y[ids],p)
    np.savez_compressed(dest/'predictions.npz',**prediction_arrays)
    before=float(np.linalg.norm(dv.astype(float)-rv));after=float(np.linalg.norm(vector(received['restored']).astype(float)-rv))
    iri,status=ratio(before-after,before,1e-12);fractions={'vector_IRI':iri,'vector_status':status}
    for label in ('val','test'):
        ref,dam,rep=(measures[f'{label}_{name}'] for name in ('reference','damaged','restored'))
        value,status=ratio(rep['macro_F1']-dam['macro_F1'],ref['macro_F1']-dam['macro_F1'])
        fractions[f'{label}_F1_IRI']=value;fractions[f'{label}_F1_status']=status;fractions[f'{label}_F1_reference_headroom']=ref['macro_F1']-dam['macro_F1']
        value,status=ratio(dam['log_loss']-rep['log_loss'],dam['log_loss']-ref['log_loss'])
        fractions[f'{label}_logloss_IRI']=value;fractions[f'{label}_logloss_status']=status;fractions[f'{label}_logloss_reference_headroom']=dam['log_loss']-ref['log_loss']
    repair=next(e for e in events if e['kind']=='repair16');ack=next(e for e in events if e['kind']=='ack' and e['key'].startswith('f05_'))
    cost={'repair_request_framed_bytes':repair['frame_bytes'],'repair_ACK_framed_bytes':ack['frame_bytes'],
          'repair_total_application_bytes':repair['frame_bytes']+ack['frame_bytes'],'repair_NPY_tensor_bytes':repair['NPY_tensor_bytes'],
          'loss_bitmap_NPY_bytes':repair['tensor_sizes'][0],'loss_bitmap_raw_bytes':(N+7)//8,'FP16_residual_NPY_bytes':repair['tensor_sizes'][1],
          'FP16_residual_raw_bytes':residual16.nbytes,'repair_request_metadata_lengthframing_bytes':repair['frame_bytes']-repair['NPY_tensor_bytes'],
          'all_three_requests_and_ACK_application_bytes':sum(e['frame_bytes'] for e in events),'loopback_session_elapsed_seconds':elapsed}
    result={'seed':seed,'parent_validation_selected_round':rnd,'new_training_fit_calls':0,'received_via_actual_loopback':True,
            'damage_L2':before,'repaired_error_L2':after,'exact_FP32_aux_matches_reference':True,'exact_FP32_aux_transmitted':False,
            'measures':measures,'recovery':fractions,'wire_cost':cost}
    write_json(dest/'final.json',result);return result


def describe(values):
    v=[float(a) for a in values if a is not None]
    if not v:return {'n_total':len(values),'n_valid':0,'n_NA':len(values),'mean':None,'sd':None,'ci95':None}
    a=np.asarray(v,float);sd=float(a.std(ddof=1)) if len(a)>1 else None;half=float(student_t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))) if sd is not None else None
    return {'n_total':len(values),'n_valid':len(a),'n_NA':len(values)-len(a),'mean':float(a.mean()),'sd':sd,'ci95':[float(a.mean()-half),float(a.mean()+half)] if half is not None else None}


def verify(output):
    protocol=read_json(output/'protocol.json');checks={'decoded_frames':0,'independent_utility_values':0,'recovery_values':0};digits=load_digits();xx=digits.data.astype(np.float32)/16;yy=digits.target
    for seed in SEEDS:
        case=output/'cases'/f's{seed}';result=read_json(case/'final.json');events=read_json(case/'wire_events.json');decoded={}
        with np.load(case/'wire_frames.npz') as frames:
            assert len(events)==len(frames.files)==6
            for event in events:
                raw=frames[event['key']].tobytes();assert hashlib.sha256(raw).hexdigest()==event['sha256'];assert len(raw)==event['frame_bytes']
                assert int.from_bytes(raw[:8],'little')==len(raw)-8
                size=int.from_bytes(raw[8:12],'little');meta=json.loads(raw[12:12+size]);count=int.from_bytes(raw[12+size:16+size],'little');offset=16+size;aa=[];sizes=[]
                for _ in range(count):
                    length=int.from_bytes(raw[offset:offset+8],'little');offset+=8;aa.append(np.load(io.BytesIO(raw[offset:offset+length]),allow_pickle=False));sizes.append(length);offset+=length
                assert offset==len(raw) and sum(sizes)==event['NPY_tensor_bytes'];decoded[event['key']]=(meta,aa);checks['decoded_frames']+=1
        req={meta['kind']:aa for meta,aa in decoded.values() if meta['kind']!='ack'}
        with np.load(case/'received_models.npz') as z:stored={key:z[key] for key in z.files}
        for name in ('reference','damaged'):
            for j,a in enumerate(req[name]):np.testing.assert_array_equal(a,stored[f'{name}_a{j}'])
        mask=np.unpackbits(req['repair16'][0],bitorder='little')[:N].astype(bool);assert mask.sum()==1024 and mask[:1024].all() and not mask[1024:].any()
        received_damaged=np.concatenate([a.ravel() for a in req['damaged']]);reference=np.concatenate([a.ravel() for a in req['reference']]);expected=received_damaged.astype(float)
        expected[mask]+=req['repair16'][1].astype(float);actual=np.concatenate([stored[f'restored_a{j}'].ravel() for j in range(4)])
        np.testing.assert_array_equal(expected.astype(np.float32),actual)
        np.testing.assert_array_equal(req['reference'][0][32:],req['damaged'][0][32:]);assert np.count_nonzero(req['damaged'][0][:32])==0
        for j in range(1,4):np.testing.assert_array_equal(req['reference'][j],req['damaged'][j])
        split=read_json(case/'source_split.json');independent={}
        with np.load(case/'predictions.npz') as predictions:
            for label,ids_key in (('val','validation_ids'),('test','test_ids')):
                ids=np.asarray(split[ids_key]);y=yy[ids];np.testing.assert_array_equal(predictions[f'{label}_source_ids'],ids);np.testing.assert_array_equal(predictions[f'{label}_y'],y)
                for name in ('reference','damaged','restored','exact_FP32_aux'):
                    a=[stored[f'{name}_a{j}'] for j in range(4)];hidden=np.clip(np.dot(xx[ids],a[0])+a[1],0,None);logits=np.dot(hidden,a[2])+a[3];logits-=np.max(logits,1,keepdims=True)
                    exp=np.exp(logits);p=exp/np.sum(exp,1,keepdims=True);np.testing.assert_allclose(p,predictions[f'{label}_{name}'],atol=1e-7,rtol=0)
                    cm=np.bincount(y*10+p.argmax(1),minlength=100).reshape(10,10);den=cm.sum(0)+cm.sum(1);f1=float(np.mean(np.divide(2*np.diag(cm),den,out=np.zeros(10,float),where=den!=0)))
                    clipped=np.clip(p,np.finfo(np.float32).eps,1-np.finfo(np.float32).eps);clipped=clipped/clipped.sum(1,keepdims=True)
                    ll=float(-np.log(clipped[np.arange(len(y)),y].astype(float)).mean());independent[label,name]={'macro_F1':f1,'log_loss':ll}
                    for key,want in independent[label,name].items():assert abs(result['measures'][f'{label}_{name}'][key]-want)<2e-7
                    checks['independent_utility_values']+=2
        loss=np.linalg.norm(received_damaged.astype(float)-reference);error=np.linalg.norm(actual.astype(float)-reference)
        assert abs(result['recovery']['vector_IRI']-(1-error/loss))<1e-12;checks['recovery_values']+=1
        for label in ('val','test'):
            r,d,p=(independent[label,name] for name in ('reference','damaged','restored'))
            for metric,num,den in (('F1',p['macro_F1']-d['macro_F1'],r['macro_F1']-d['macro_F1']),('logloss',d['log_loss']-p['log_loss'],d['log_loss']-r['log_loss'])):
                want=None if den<=1e-8 else num/den;got=result['recovery'][f'{label}_{metric}_IRI']
                assert (got is None and want is None) or (got is not None and want is not None and abs(got-want)<2e-6);checks['recovery_values']+=1
    for name,digest in protocol['source_sha256'].items():assert sha(source_path(name))==digest and sha(output/'source_snapshot'/name)==digest
    result={'status':'passed','checks':checks,'cases':5,'source_unchanged':True,'scope':'Separate arithmetic self-audit of actual received frames, exact damage/repair masks, received model forward probabilities, independent confusion-matrixF1/crossentropy/recovery mathematics; not an independent third-party review.'}
    write_json(output/'verification.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results'/'purpose_recovery_stress');parser.add_argument('--study',type=Path,default=ROOT/'results'/'purpose_system'/'main')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.verify_only:print(json.dumps(verify(args.output)));return
    protocol=freeze(args.output,args.study)
    if args.prepare_only:print(json.dumps({'status':'frozen','source_sha256':protocol['source_sha256'],'cases':5,'fixed_loss':protocol['fixed_loss']}));return
    rows=[]
    for seed in SEEDS:
        rows.append(run_one(seed,args.output,args.study));print(json.dumps({'completed':len(rows),'total':5,'seed':seed}),flush=True)
    recovery={key:describe([r['recovery'][key] for r in rows]) for key in ('vector_IRI','val_F1_IRI','test_F1_IRI','val_logloss_IRI','test_logloss_IRI')}
    utility={f'{label}_{name}_{metric}':describe([r['measures'][f'{label}_{name}'][metric] for r in rows]) for label in ('val','test') for name in ('reference','damaged','restored') for metric in ('macro_F1','log_loss')}
    summary={'status':'completed','cases':5,'seeds':list(SEEDS),'new_training_fit_calls':0,'recovery':recovery,'utility':utility,
             'wire':{key:describe([r['wire_cost'][key] for r in rows]) for key in rows[0]['wire_cost']},'runs':rows,
             'source_unchanged':sha(__file__)==protocol['source_sha256'][Path(__file__).name],
             'limits':protocol['limits'],'normal_steady_state_evidence_preserved':'Parent INT4normaloperatingutilityheadroom is often zero/NA; this fixedblockstress is an additional positive-loss sensitivity control,not replacement.'}
    write_json(args.output/'summary.json',summary)
    export={'metric_index':14,'name':'IRI fixed positive-loss recovery sensitivity control','unit':'recovery fraction;validn;bytes',
            'value':recovery['test_F1_IRI'],'conditions':{'vector_IRI':recovery['vector_IRI'],'test_logloss_IRI':recovery['test_logloss_IRI'],
                        'test_reference_F1':utility['test_reference_macro_F1'],'test_damaged_F1':utility['test_damaged_macro_F1'],'test_restored_F1':utility['test_restored_macro_F1'],
                        'repair_framed_bytes_including_ACK':summary['wire']['repair_total_application_bytes'],'repair_lossbitmap_NPY_bytes':summary['wire']['loss_bitmap_NPY_bytes'],
                        'fixed_loss':protocol['fixed_loss'],'headroom_tolerance':1e-8},
            'scope':'Actual framed127.0.0.1receiver applies FP16residual+lossbitmap after controlled application-layer modelblockerasure. Not naturalWANpacketloss; exactFP32auxnottransmitted. Same5Bseeds,validationselectedreference,no retraining/testdamage selection.',
            'steady_state_NA_preserved':True,'source':'summary.json'}
    write_json(args.output/'report_export.json',export);verification=verify(args.output)
    print(json.dumps({'summary':{key:summary[key] for key in ('status','cases','recovery','source_unchanged')},'verification':verification}))


if __name__=='__main__':main()
