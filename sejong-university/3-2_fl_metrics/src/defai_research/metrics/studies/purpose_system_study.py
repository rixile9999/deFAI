from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['FLWR_TELEMETRY_ENABLED']='0'
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time

from ._paths import ROOT, source_path, relative_path
import numpy as np
import flwr
from flwr.common import FitIns,FitRes,Status,Code,ndarrays_to_parameters,parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from sklearn.metrics import f1_score,log_loss
from ._helpers import BenchmarkClient,prepare_data,make_model,reshape_like
from ._helpers import pack,predict,json_write,csv_write
from ._resources import memory_bytes

SEEDS=tuple(range(8201,8206))
CODECS=('fp32','int8','int4','int4_repair16')
CLIENTS=3
ROUNDS=30
BLOCK=256
TARGETS=(.90,.95)
STREAK=3
SOURCE_NAMES=(Path(__file__).name,'_helpers.py','learning.py','api.py','graph.py','_resources.py')


def hashes():
    return {name:hashlib.sha256(source_path(name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def serialize(meta,tensors=()):
    header=json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return (struct.pack('<I',len(header))+header+struct.pack('<I',len(tensors))+
            b''.join(struct.pack('<Q',len(t))+t for t in tensors))


def parse(body):
    if len(body)<8:raise ValueError('Truncated message')
    n=struct.unpack_from('<I',body)[0]
    meta=json.loads(body[4:4+n]);count=struct.unpack_from('<I',body,4+n)[0]
    offset=8+n;tensors=[]
    for _ in range(count):
        size=struct.unpack_from('<Q',body,offset)[0];offset+=8
        tensor=body[offset:offset+size]
        if len(tensor)!=size:raise ValueError('Truncated tensor')
        tensors.append(tensor);offset+=size
    if offset!=len(body):raise ValueError('Trailing bytes')
    return meta,tensors


def send(sock,body):
    frame=struct.pack('<Q',len(body))+body
    began=time.perf_counter();sock.sendall(frame)
    return time.perf_counter()-began


def exact_recv(sock,n):
    result=bytearray()
    while len(result)<n:
        part=sock.recv(n-len(result))
        if not part:raise EOFError('Connection closed before frame completion')
        result.extend(part)
    return bytes(result)


def receive(sock):
    length=struct.unpack('<Q',exact_recv(sock,8))[0]
    if length>100_000_000:raise ValueError('Oversized message')
    return exact_recv(sock,length)


def npy(a):
    out=io.BytesIO();np.save(out,a,allow_pickle=False);return out.getvalue()


def unnpy(value):
    return np.load(io.BytesIO(value),allow_pickle=False)


def encode(vector,bits):
    vector=np.asarray(vector,np.float32)
    if bits==32:return [npy(vector)],vector.copy()
    limit=127 if bits==8 else 7
    codes=np.empty(len(vector),np.int8);scales=[];decoded=np.empty_like(vector)
    for start in range(0,len(vector),BLOCK):
        block=vector[start:start+BLOCK]
        scale=np.float32(max(float(np.max(np.abs(block)))/limit,1e-12))
        code=np.clip(np.rint(block/scale),-limit,limit).astype(np.int8)
        codes[start:start+len(block)]=code
        decoded[start:start+len(block)]=code.astype(np.float32)*scale
        scales.append(scale)
    if bits==4:
        shifted=(codes.astype(np.int16)+7).astype(np.uint8)
        if len(shifted)%2:shifted=np.r_[shifted,np.uint8(7)]
        codes=(shifted[::2] | shifted[1::2]<<4).astype(np.uint8)
    return [npy(codes),npy(np.asarray(scales,np.float32))],decoded


def decode(tensors,bits,length):
    if bits==32:
        result=unnpy(tensors[0]);assert result.dtype==np.float32 and result.shape==(length,)
        return result
    codes,scales=map(unnpy,tensors)
    if bits==4:
        unpacked=np.empty(len(codes)*2,np.int8)
        unpacked[::2]=(codes&15).astype(np.int8)-7
        unpacked[1::2]=(codes>>4).astype(np.int8)-7
        codes=unpacked[:length]
    assert len(codes)==length and len(scales)==math.ceil(length/BLOCK)
    return codes.astype(np.float32)*np.repeat(scales,BLOCK)[:length]


def dataset(seed):
    return prepare_data('digits',seed,.3,0.,clients=CLIENTS)


def loss(y,p):
    return float(log_loss(y,p,labels=np.arange(p.shape[1])))


def f1(y,p):
    return float(f1_score(y,p.argmax(1),labels=np.arange(p.shape[1]),average='macro',zero_division=0))


def aggregate(old,deltas,weights):
    fit_results=[]
    for delta,n in zip(deltas,weights):
        local=reshape_like(pack(old).astype(np.float64)+delta.astype(np.float64),old)
        fit_results.append((None,FitRes(Status(Code.OK,'OK'),ndarrays_to_parameters(local),int(n),{})))
    strategy=FedAvg(inplace=False,fit_metrics_aggregation_fn=lambda _: {})
    merged,_=strategy.aggregate_fit(1,fit_results,[])
    return parameters_to_ndarrays(merged)


def worker(args):
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    assert protocol['source_sha256']==hashes()
    data=dataset(args.seed);ids=data['phases']['initial'][args.cid]
    client=BenchmarkClient(args.cid,data['x'][ids],data['y'][ids],args.seed).to_client()
    sock=socket.socket();sock.settimeout(60);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    sock.connect(('127.0.0.1',args.port))
    send(sock,serialize({'kind':'ready','cid':args.cid,'pid':os.getpid(),'n':len(ids)}))
    raw_arrays={};rss=[];first_cpu=None;first_wall=None
    while True:
        body=receive(sock);meta,tensors=parse(body)
        if first_cpu is None:first_cpu=time.process_time();first_wall=time.perf_counter()
        if meta['kind']=='stop':
            current,ospeak=memory_bytes()
            summary={'kind':'summary','cid':args.cid,'pid':os.getpid(),
                'process_CPU_seconds':time.process_time()-first_cpu,
                'session_wall_seconds':time.perf_counter()-first_wall,
                'sampled_RSS_MiB':rss,'sampled_peak_RSS_MiB':max(rss),
                'OS_lifetime_peak_RSS_MiB':ospeak/2**20,'iterations':len(rss)}
            send(sock,serialize(summary));sock.close()
            np.savez_compressed(args.run/f'worker{args.cid}_updates.npz',**raw_arrays)
            json_write(args.run/f'worker{args.cid}_resources.json',summary)
            return
        assert meta['kind']=='fit' and meta['cid']==args.cid
        old=[unnpy(t) for t in tensors]
        cfg={'round':meta['round'],'local_epochs':2,'batch_size':32,'l2':.0001,
             'learning_rate':.2/math.sqrt(1+(meta['round']-1)/20)}
        before=time.process_time()
        result=client.fit(FitIns(ndarrays_to_parameters(old),cfg))
        fit_cpu=time.process_time()-before
        local=parameters_to_ndarrays(result.parameters)
        raw=(pack(local).astype(np.float64)-pack(old).astype(np.float64)).astype(np.float32)
        bits=32 if args.codec=='fp32' else 8 if args.codec=='int8' else 4
        coded,base=encode(raw,bits)
        raw_arrays[f'r{meta["round"]:02d}_raw']=raw
        raw_arrays[f'r{meta["round"]:02d}_base']=base
        header={'kind':'update','cid':args.cid,'round':meta['round'],'bits':bits,'length':len(raw),
                'n':result.num_examples,'optimizer_steps':int(result.metrics['optimizer_steps']),
                'examples_processed':int(result.metrics['examples_processed']),'fit_CPU_seconds':fit_cpu}
        send(sock,serialize(header,coded))
        if args.codec=='int4_repair16':
            residual=(raw.astype(np.float64)-base.astype(np.float64)).astype(np.float16)
            send(sock,serialize({'kind':'repair','cid':args.cid,'round':meta['round'],'dtype':'float16'},[npy(residual)]))
            restored=(base.astype(np.float64)+residual.astype(np.float64)).astype(np.float32)
            raw_arrays[f'r{meta["round"]:02d}_residual16']=residual
        else:restored=base
        raw_arrays[f'r{meta["round"]:02d}_restored']=restored
        rss.append(memory_bytes()[0]/2**20)


def target_summary(rounds,threshold):
    reached=None
    for index in range(STREAK-1,len(rounds)):
        if all(r['validation_macro_F1']>=threshold for r in rounds[index-STREAK+1:index+1]):
            reached=index;break
    return {'threshold':threshold,'sustained_rounds':STREAK,'reached':reached is not None,
        'completion_round':reached+1 if reached is not None else None,
        'TC_loopback_wall_seconds':rounds[reached]['cumulative_wall_seconds'] if reached is not None else None,
        'framed_bytes_to_target':sum(r['framed_bytes'] for r in rounds[:reached+1]) if reached is not None else None}


def ratio(numerator,denominator,tolerance=1e-8):
    if denominator<=tolerance:return None,'nonpositive_or_tiny_reference_headroom'
    return float(numerator/denominator),'defined'


def run_one(seed,codec,out,protocol_path,rounds=ROUNDS):
    run=out/f's{seed}_{codec}'
    if (run/'final.json').exists():return json.loads((run/'final.json').read_text(encoding='utf-8'))
    if run.exists():raise RuntimeError('Partial run exists; preserve it and select a new output directory')
    run.mkdir(parents=True)
    data=dataset(seed);json_write(run/'source_split.json',data['audit'])
    initial=make_model(seed,data['x'].shape[1],data['classes'])
    arrays=[a.copy() for a in initial];wire={};events=[];processes=[];streams=[];sockets={}
    saved_models={};validation=[];history=[];resources=[]
    listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(CLIENTS);listener.settimeout(60)
    port=listener.getsockname()[1];setup_begin=time.perf_counter();frame_index=0
    def record(direction,cid,body,phase,round_id=0):
        nonlocal frame_index
        key=f'f{frame_index:05d}';frame_index+=1
        wire[key]=np.frombuffer(struct.pack('<Q',len(body))+body,np.uint8).copy()
        meta,tensors=parse(body)
        events.append({'key':key,'direction':direction,'cid':cid,'phase':phase,'round':round_id,
            'kind':meta['kind'],'body_bytes':len(body),'frame_bytes':len(body)+8,
            'tensor_bytes':sum(map(len,tensors)),'sha256':hashlib.sha256(wire[key].tobytes()).hexdigest()})
    try:
        for cid in range(CLIENTS):
            log=(run/f'worker{cid}.log').open('w',encoding='utf-8');streams.append(log)
            cmd=[sys.executable,'-m', 'defai_research.metrics.studies.purpose_system_study','--stage','worker','--seed',str(seed),
                 '--codec',codec,'--cid',str(cid),'--port',str(port),'--run',str(run),'--protocol',str(protocol_path)]
            processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
        for _ in range(CLIENTS):
            connection,_=listener.accept();connection.settimeout(60);connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            body=receive(connection);meta,_=parse(body);assert meta['kind']=='ready'
            cid=meta['cid'];assert cid not in sockets
            sockets[cid]=connection;record('up',cid,body,'setup')
        startup=time.perf_counter()-setup_begin
        cpu_begin=time.process_time();start=time.perf_counter();server_rss=[]
        best_loss=float('inf');selected_round=0
        for rnd in range(1,rounds+1):
            began=time.perf_counter();old=[a.copy() for a in arrays]
            for i,a in enumerate(old):saved_models[f'r{rnd:02d}_old_{i}']=a
            event_begin=len(events)
            for cid in range(CLIENTS):
                body=serialize({'kind':'fit','cid':cid,'round':rnd,'codec':codec},ndarrays_to_parameters(old).tensors)
                send(sockets[cid],body);record('down',cid,body,'training',rnd)
            deltas=[];weights=[];steps=examples=0
            for cid in range(CLIENTS):
                body=receive(sockets[cid]);record('up',cid,body,'training',rnd)
                meta,tensors=parse(body);assert meta['kind']=='update' and meta['cid']==cid and meta['round']==rnd
                delta=decode(tensors,meta['bits'],meta['length'])
                if codec=='int4_repair16':
                    body=receive(sockets[cid]);record('up',cid,body,'training',rnd)
                    repair,values=parse(body);assert repair['kind']=='repair' and repair['cid']==cid and repair['round']==rnd
                    residual=unnpy(values[0]);assert residual.dtype==np.float16 and residual.shape==delta.shape
                    delta=(delta.astype(np.float64)+residual.astype(np.float64)).astype(np.float32)
                deltas.append(delta);weights.append(meta['n']);steps+=meta['optimizer_steps'];examples+=meta['examples_processed']
            arrays=aggregate(old,deltas,weights)
            vp=predict(arrays,data['x'][data['val']]);validation.append(vp.copy())
            val_loss=loss(data['y'][data['val']],vp)
            if val_loss<best_loss:best_loss,selected_round=val_loss,rnd
            for i,a in enumerate(arrays):saved_models[f'r{rnd:02d}_applied_{i}']=a
            costs=events[event_begin:]
            elapsed=time.perf_counter()-began
            current_rss=memory_bytes()[0]/2**20;server_rss.append(current_rss)
            history.append({'round':rnd,'wall_seconds':elapsed,'cumulative_wall_seconds':time.perf_counter()-start,
                'validation_log_loss':val_loss,'validation_macro_F1':f1(data['y'][data['val']],vp),
                'framed_bytes':sum(r['frame_bytes'] for r in costs),'body_bytes':sum(r['body_bytes'] for r in costs),
                'tensor_bytes':sum(r['tensor_bytes'] for r in costs),'optimizer_steps':steps,'examples_processed':examples,
                'coordinator_sampled_RSS_MiB':current_rss})
        measured_wall=time.perf_counter()-start;coordinator_cpu=time.process_time()-cpu_begin
        for cid in range(CLIENTS):
            body=serialize({'kind':'stop'});send(sockets[cid],body);record('down',cid,body,'shutdown')
        for cid in range(CLIENTS):
            body=receive(sockets[cid]);record('up',cid,body,'shutdown');meta,_=parse(body)
            assert meta['kind']=='summary';resources.append(meta)
        for child in processes:
            child.wait(timeout=60)
            if child.returncode:raise RuntimeError(f'Worker failed ({child.returncode})')
    finally:
        listener.close()
        for connection in sockets.values():connection.close()
        for child in processes:
            if child.poll() is None:child.terminate()
        for log in streams:log.close()
    audit_updates=[np.load(run/f'worker{cid}_updates.npz',allow_pickle=False) for cid in range(CLIENTS)]
    distortions=[];recovery=[];selected_predictions={}
    weights=[len(data['phases']['initial'][cid]) for cid in range(CLIENTS)]
    for rnd in range(1,rounds+1):
        raw=[z[f'r{rnd:02d}_raw'] for z in audit_updates]
        base=[z[f'r{rnd:02d}_base'] for z in audit_updates]
        repaired=[z[f'r{rnd:02d}_restored'] for z in audit_updates]
        raw_all=np.concatenate(raw).astype(np.float64)
        error=np.concatenate([b.astype(np.float64)-a for a,b in zip(raw,repaired)])
        damage=np.concatenate([b.astype(np.float64)-a for a,b in zip(raw,base)])
        den=float(np.dot(raw_all,raw_all))
        dist={'round':rnd,'raw_squared_norm':den,'repaired_error_squared_norm':float(np.dot(error,error)),
              'damaged_error_squared_norm':float(np.dot(damage,damage)),
              'relative_L2':float(np.linalg.norm(error)/max(np.linalg.norm(raw_all),1e-12)),
              'MSE':float(np.mean(error**2)),
              'cosine':float(np.dot(raw_all,raw_all+error)/max(np.linalg.norm(raw_all)*np.linalg.norm(raw_all+error),1e-20))}
        distortions.append(dist)
        if rnd!=selected_round:continue
        old=[saved_models[f'r{rnd:02d}_old_{i}'] for i in range(4)]
        models={name:aggregate(old,values,weights) for name,values in [('reference',raw),('damaged',base),('restored',repaired)]}
        actual=[saved_models[f'r{rnd:02d}_applied_{i}'] for i in range(4)]
        for a,b in zip(models['restored'],actual):np.testing.assert_array_equal(a,b)
        original=np.concatenate(raw).astype(np.float64);corrupted=np.concatenate(base).astype(np.float64)
        exact=corrupted+(original-corrupted)
        assert np.array_equal(exact,original),'Exact-residual diagnostic failed'
        vector_iri,status=ratio(np.linalg.norm(damage)-np.linalg.norm(error),np.linalg.norm(damage),1e-12)
        values={'round':rnd,'vector_recovery_fraction':vector_iri,'vector_status':status,
                'exact_FP64_residual_restores_FP32_reference':True,'exact_residual_transmitted':False}
        for split in ('val','test'):
            ids=data[split];y=data['y'][ids]
            probs={name:predict(model,data['x'][ids]) for name,model in models.items()}
            for name,p in probs.items():
                values[f'{split}_{name}_macro_F1']=f1(y,p)
                values[f'{split}_{name}_log_loss']=loss(y,p)
                selected_predictions[f'{split}_{name}']=p
            utility,status=ratio(values[f'{split}_restored_macro_F1']-values[f'{split}_damaged_macro_F1'],
                                 values[f'{split}_reference_macro_F1']-values[f'{split}_damaged_macro_F1'])
            values[f'{split}_F1_recovery_fraction']=utility;values[f'{split}_F1_status']=status
            utility,status=ratio(values[f'{split}_damaged_log_loss']-values[f'{split}_restored_log_loss'],
                                 values[f'{split}_damaged_log_loss']-values[f'{split}_reference_log_loss'])
            values[f'{split}_logloss_recovery_fraction']=utility;values[f'{split}_logloss_status']=status
        recovery.append(values)
    for z in audit_updates:z.close()
    best_model=[saved_models[f'r{selected_round:02d}_applied_{i}'] for i in range(4)]
    test=predict(best_model,data['x'][data['test']]);test_f1=f1(data['y'][data['test']],test)
    training_bytes=sum(r['framed_bytes'] for r in history)
    result={'seed':seed,'codec':codec,'dataset':'digits','clients':CLIENTS,'rounds':rounds,
        'selected_round':selected_round,'selected_validation_log_loss':best_loss,
        'test_macro_F1':test_f1,'test_accuracy':float(np.mean(test.argmax(1)==data['y'][data['test']])),
        'training_framed_bytes':training_bytes,'session_framed_bytes':sum(r['frame_bytes'] for r in events),
        'training_body_bytes':sum(r['body_bytes'] for r in history),'training_tensor_bytes':sum(r['tensor_bytes'] for r in history),
        'measured_loopback_pipeline_wall_seconds':measured_wall,'startup_wall_seconds':startup,
        'loopback_pipeline_application_Bps':training_bytes/measured_wall,
        'test_F1_per_MiB':test_f1/(training_bytes/2**20),
        'coordinator_CPU_seconds':coordinator_cpu,'worker_CPU_seconds':sum(r['process_CPU_seconds'] for r in resources),
        'all_process_CPU_seconds':coordinator_cpu+sum(r['process_CPU_seconds'] for r in resources),
        'coordinator_sampled_peak_RSS_MiB':max(server_rss),
        'worker_sampled_peak_RSS_MiB_by_PID':{str(r['pid']):r['sampled_peak_RSS_MiB'] for r in resources},
        'TC_targets':[target_summary(history,target) for target in TARGETS],
        'mean_update_relative_L2':float(np.mean([d['relative_L2'] for d in distortions])),
        'mean_update_MSE':float(np.mean([d['MSE'] for d in distortions])),
        'mean_update_cosine':float(np.mean([d['cosine'] for d in distortions])),
        'repair_applicable':codec=='int4_repair16','selected_recovery':recovery[0],
        'test_used_for_selection':False,'source_sha256':hashes(),
        'scope':'Real custom framed loopback TCP carrying Flower tensors/fit results; not gRPC/WAN. Byte/s includes fit+aggregation+validation pipeline, not link bandwidth. CPU/RSS have separate units.'}
    np.savez_compressed(run/'models_by_round.npz',**saved_models)
    np.savez_compressed(run/'wire_frames.npz',**wire)
    np.savez_compressed(run/'validation_predictions.npz',y=data['y'][data['val']],source_ids=data['val'],probabilities=np.stack(validation))
    np.savez_compressed(run/'selected_predictions.npz',**selected_predictions,test_y=data['y'][data['test']],val_y=data['y'][data['val']])
    csv_write(run/'rounds.csv',history);csv_write(run/'update_distortion.csv',distortions)
    json_write(run/'wire_events.json',events);json_write(run/'worker_resources.json',resources)
    json_write(run/'recovery.json',recovery);json_write(run/'final.json',result)
    return result


def freeze(out,smoke=False):
    seeds=(8199,) if smoke else SEEDS
    codecs=('fp32','int4_repair16') if smoke else CODECS
    rng=np.random.default_rng(972)
    jobs=[]
    for seed in rng.permutation(seeds):
        for codec in rng.permutation(codecs):jobs.append({'seed':int(seed),'codec':str(codec)})
    value={'schema':1,'source_sha256':hashes(),'flower_version':flwr.__version__,'seeds':list(seeds),
        'codecs':list(codecs),'rounds':5 if smoke else ROUNDS,'clients':CLIENTS,'ordered_jobs':jobs,
        'randomized_order_seed':972,'local_epochs':2,'batch_size':32,'learning_rate':'.2/sqrt(1+(round-1)/20)',
        'train_validation_test_split':'source-disjoint60/20/20; training-only standard preparation; no duplicates or drift; no test selection',
        'TC':'fixed validation macroF1>=.90 and .95 for3consecutive rounds; completion round wall and actual framed bytes; NR stays null',
        'wire':'real localhost TCP; application body+8-byte length header; no fabricated WAN/TCPIP/TLS framing or bandwidth claim',
        'Flower':'NumPyClient.to_client.fit(FitIns) in3separateworker processes; server FedAvg.aggregate_fit onreceived reconstructedtensors; custom transport not FlowergRPC',
        'L':'same originalFP32update vs receivedcodec delta relativeL2/MSE/cosine; labelMI not used as reconstruction loss',
        'IRI':'INT4damaged/FP16residualrepaired/FP32reference fromsameoldmodel andupdates atvalidation-selectedcheckpoint; vectorrecovery and heldoutF1/logloss recovery fractions; nonpositive/tinyheadroomNA withvalidcounts; exact FP64residualdiagnosticnottransmitted',
        'R':'actual framed application bytes divided by measured fulltrainingpipelinewallseconds; reportasloopbackpipelinegoodput not physical link capacity',
        'RU':'perprocess measuredCPU seconds andsampledRSSMiB separately; neveraddtimeandmemory; workerprocessstartup excludedfromCPU TCprimarybutstartupwallrecorded',
        'timing':'workersfreshforeachrun; imports/readinessoutsideTC; measuredloopincludesTCP/actualfit/codec/FedAvg/validation/inmemorywireaudit; artifactwriteandheldoutdiagnosticsaftertiming',
        'statistics':'descriptive5splitseedmean/SD andpairedcostdifferences, noallmetricsdominanceclaim; public datasetresplitsnotexternalvalidation'}
    out.mkdir(parents=True,exist_ok=True);path=out/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==value,'Frozen protocol/source mismatch'
    else:
        json_write(path,value)
        for name in SOURCE_NAMES:
            target=out/'source_snapshot'/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(source_path(name),target)
    return path,value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('protocol','main','smoke','worker'),required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/purpose_system/main')
    parser.add_argument('--seed',type=int);parser.add_argument('--codec',choices=CODECS)
    parser.add_argument('--cid',type=int);parser.add_argument('--port',type=int)
    parser.add_argument('--run',type=Path);parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    if args.stage=='worker':worker(args);return
    protocol_path,protocol=freeze(args.output,args.stage=='smoke')
    if args.stage=='protocol':print(json.dumps({'status':'frozen','jobs':len(protocol['ordered_jobs'])}));return
    rows=[]
    for index,job in enumerate(protocol['ordered_jobs']):
        row=run_one(job['seed'],job['codec'],args.output/'runs',protocol_path,protocol['rounds']);rows.append(row)
        json_write(args.output/'progress.json',{'completed':len(rows),'planned':len(protocol['ordered_jobs'])})
        print(json.dumps({'completed':len(rows),'seed':row['seed'],'codec':row['codec'],'testF1':row['test_macro_F1']},ensure_ascii=False),flush=True)
    assert hashes()==protocol['source_sha256']
    json_write(args.output/'summary.json',{'status':'completed','runs':rows,'count':len(rows),'source_unchanged':True,
        'scope':protocol['statistics']})


if __name__=='__main__':main()
