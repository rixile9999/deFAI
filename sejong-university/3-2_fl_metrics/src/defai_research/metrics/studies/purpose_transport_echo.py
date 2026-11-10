from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

from ._paths import ROOT, source_path, relative_path
WARMUP=10
REPETITIONS=200


def read_exact(sock,n):
    result=bytearray()
    while len(result)<n:
        piece=sock.recv(n-len(result))
        if not piece:raise EOFError('Incomplete echo frame')
        result.extend(piece)
    return bytes(result)


def received(sock):
    header=read_exact(sock,8)
    length=struct.unpack('<Q',header)[0]
    return header+read_exact(sock,length)


def write(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def worker(port):
    sock=socket.socket();sock.settimeout(60);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    sock.connect(('127.0.0.1',port))
    import os
    ready=json.dumps({'pid':os.getpid()}).encode();sock.sendall(struct.pack('<Q',len(ready))+ready)
    count=total=0
    while True:
        frame=received(sock)
        if len(frame)==8:break
        sock.sendall(frame);count+=1;total+=2*len(frame)
    body=json.dumps({'frames':count,'framed_bytes_both_directions':total}).encode()
    sock.sendall(struct.pack('<Q',len(body))+body);sock.close()


def driver(args):
    import numpy as np
    main=json.loads((args.study/'protocol.json').read_text(encoding='utf-8'))
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs=main['ordered_jobs']
    protocol={'schema':1,'source_sha256':source_sha,'main_protocol_sha256':hashlib.sha256((args.study/'protocol.json').read_bytes()).hexdigest(),
        'sessions':len(jobs),'warmup_cycles':WARMUP,'measured_cycles':REPETITIONS,'jobs':jobs,
        'payload':'Byte-exact original framed application messages from all three clients in training round1 of corresponding frozen main run.',
        'rate':'Both-direction framed application bits / measured replay elapsed seconds; includesPython echo and byte equality checks; no training, no WAN or physical link capacity claim.',
        'randomized_order':'same prespecified main order; fresh process per session'}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==protocol
    else:write(path,protocol)
    results=[]
    for index,job in enumerate(jobs):
        out=args.output/f's{job["seed"]}_{job["codec"]}.json'
        if out.exists():results.append(json.loads(out.read_text(encoding='utf-8')));continue
        run=args.study/'runs'/f's{job["seed"]}_{job["codec"]}'
        events=json.loads((run/'wire_events.json').read_text(encoding='utf-8'))
        with np.load(run/'wire_frames.npz',allow_pickle=False) as stored:
            frames=[stored[e['key']].tobytes() for e in events if e['phase']=='training' and e['round']==1]
        assert len(frames)==(9 if job['codec']=='int4_repair16' else 6)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(60)
            port=listener.getsockname()[1]
            child=subprocess.Popen([sys.executable,'-m', 'defai_research.metrics.studies.purpose_transport_echo','--stage','worker','--port',str(port)],
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            connection=None
            try:
                connection,_=listener.accept();connection.settimeout(60)
                connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                pid=json.loads(received(connection)[8:])['pid']
                for _ in range(WARMUP):
                    for frame in frames:
                        connection.sendall(frame);assert received(connection)==frame
                start=time.perf_counter()
                for _ in range(REPETITIONS):
                    for frame in frames:
                        connection.sendall(frame);assert received(connection)==frame
                elapsed=time.perf_counter()-start
                connection.sendall(struct.pack('<Q',0));summary=json.loads(received(connection)[8:])
                child.wait(timeout=30);assert child.returncode==0
                bytes_measured=2*sum(map(len,frames))*REPETITIONS
                assert summary['frames']==len(frames)*(WARMUP+REPETITIONS)
                assert summary['framed_bytes_both_directions']==2*sum(map(len,frames))*(WARMUP+REPETITIONS)
                result={**job,'pid':pid,'frame_shapes_bytes':list(map(len,frames)),
                    'replayed_frame_sha256':[hashlib.sha256(f).hexdigest() for f in frames],
                    'warmup_cycles':WARMUP,'measured_cycles':REPETITIONS,
                    'measured_roundtrip_framed_bytes':bytes_measured,'measured_elapsed_seconds':elapsed,
                    'loopback_echo_application_bits_per_second':bytes_measured*8/elapsed,
                    'all_received_frames_exact':True,'server_observed_frames_with_warmup':summary['frames']}
                write(out,result);results.append(result)
                print(json.dumps({'echo_sessions':len(results),'planned':len(jobs)}),flush=True)
            finally:
                if connection is not None:connection.close()
                if child.poll() is None:child.terminate()
    assert len({r['pid'] for r in results})==len(results)
    write(args.output/'summary.json',{'status':'completed','sessions':results,'count':len(results),'scope':protocol['rate']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('main','worker'),default='main')
    parser.add_argument('--port',type=int)
    parser.add_argument('--study',type=Path,default=ROOT/'results/purpose_system/main')
    parser.add_argument('--output',type=Path,default=ROOT/'results/purpose_system/transport_echo')
    args=parser.parse_args()
    if args.stage=='worker':worker(args.port)
    else:driver(args)
