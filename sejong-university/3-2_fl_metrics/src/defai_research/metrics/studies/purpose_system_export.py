from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

from ._paths import ROOT, source_path, relative_path
import numpy as np
from scipy.stats import t


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def describe(values):
    valid=[float(x) for x in values if x is not None]
    if not valid:return {'n_total':len(values),'n_valid':0,'n_NA':len(values),'mean':None,'sd':None,'ci95':None}
    a=np.asarray(valid,float);sd=float(a.std(ddof=1)) if len(a)>1 else None
    half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))) if len(a)>1 else None
    return {'n_total':len(values),'n_valid':len(valid),'n_NA':len(values)-len(valid),'mean':float(a.mean()),'sd':sd,
        'ci95':[float(a.mean()-half),float(a.mean()+half)] if half is not None else None}


def main(args):
    protocol=load(args.study/'protocol.json');aggregate=load(args.study/'summary.json')
    assert aggregate['status']=='completed' and aggregate['count']==20
    rows=aggregate['runs'];echo=load(args.echo/'summary.json')
    assert echo['status']=='completed' and echo['count']==20
    echo_lookup={(r['seed'],r['codec']):r for r in echo['sessions']}
    baseline={r['seed']:r for r in rows if r['codec']=='fp32'}
    summaries={};pairs={};new_metrics=[]
    for codec in protocol['codecs']:
        subset=[r for r in rows if r['codec']==codec];assert len(subset)==5
        summary={field:describe([r[field] for r in subset]) for field in (
            'test_macro_F1','test_accuracy','training_framed_bytes','training_body_bytes','training_tensor_bytes',
            'session_framed_bytes','measured_loopback_pipeline_wall_seconds','startup_wall_seconds',
            'loopback_pipeline_application_Bps','test_F1_per_MiB','all_process_CPU_seconds',
            'coordinator_CPU_seconds','worker_CPU_seconds','coordinator_sampled_peak_RSS_MiB',
            'mean_update_relative_L2','mean_update_MSE','mean_update_cosine')}
        summary['mean_client_sampled_peak_RSS_MiB']=describe([np.mean(list(r['worker_sampled_peak_RSS_MiB_by_PID'].values())) for r in subset])
        summary['echo_application_Mbps']=describe([echo_lookup[r['seed'],codec]['loopback_echo_application_bits_per_second']/1e6 for r in subset])
        summary['pipeline_application_Mbps']=describe([r['loopback_pipeline_application_Bps']*8/1e6 for r in subset])
        summary['TC_targets']={}
        for target in (.90,.95):
            selected=[next(t for t in r['TC_targets'] if t['threshold']==target) for r in subset]
            summary['TC_targets'][str(target)]={'n_seeds':5,'n_reached':sum(t['reached'] for t in selected),
                'n_NR':sum(not t['reached'] for t in selected),'sustained_rounds':3,
                'completion_round':describe([t['completion_round'] for t in selected]),
                'TC_seconds':describe([t['TC_loopback_wall_seconds'] for t in selected]),
                'framed_bytes_to_target':describe([t['framed_bytes_to_target'] for t in selected])}
        summary['IRI']={'repair_applicable':codec=='int4_repair16'}
        for field in ('vector_recovery_fraction','val_F1_recovery_fraction','test_F1_recovery_fraction',
                      'val_logloss_recovery_fraction','test_logloss_recovery_fraction'):
            summary['IRI'][field]=describe([r['selected_recovery'][field] if codec=='int4_repair16' else None for r in subset])
        for field in ('test_damaged_macro_F1','test_restored_macro_F1','test_reference_macro_F1',
                      'test_damaged_log_loss','test_restored_log_loss','test_reference_log_loss'):
            summary['IRI'][field]=describe([r['selected_recovery'][field] for r in subset])
        summaries[codec]=summary
        pairs[codec]={'relative_training_framed_byte_savings':describe([1-r['training_framed_bytes']/baseline[r['seed']]['training_framed_bytes'] for r in subset]),
            'paired_test_macro_F1_difference':describe([r['test_macro_F1']-baseline[r['seed']]['test_macro_F1'] for r in subset])}
    metric_specs=[
        (4,'TC','검증 macro-F1≥0.90/0.95를 3연속 라운드 충족한 완료시점의 실제 loopback 경과시간. 미도달은 NR.','s; rounds; n_NR','lower at same prespecified quality target'),
        (5,'통신 효율','실제 송수신 application frame bytes, 목표 품질까지 bytes 및 F1/MiB. 모델·JSON·NPY·잔차·8-byte framing 포함.','bytes; F1/MiB','lower bytes at same quality, alongside actual utility'),
        (9,'RU','coordinator와 각 worker PID의 실제 process CPU초 및 측정기간 RSS MiB를 별도 보고. 시간과 메모리는 더하지 않음.','CPU-second; MiB','report separate resource dimensions; no mixed-unit score'),
        (13,'L','같은 FP32 학습 update와 실제 수신 복원 update 사이 relative-L2/MSE/cosine. MI를 압축 손실로 대용하지 않음.','unitless; squared parameter units','lower reconstruction error; higher cosine'),
        (14,'IRI','동일 checkpoint의 INT4 손상/FP16잔차 복원/FP32참조를 비교한 벡터 복원율 및 heldout 효용 회복률. headroom≤tolerance는 NA.','recovered error/utility fraction; n_valid','report validity count and denominator; no forced [0,1] clipping'),
        (15,'R','학습없는 실제 frame echo application bits/s와 학습·집계·검증을 포함한 pipeline bits/s를 구분.','application Mbit/s','measured transport rate; not WAN/link capacity or learning efficiency ranking'),
    ]
    for metric_id,name,definition,unit,direction in metric_specs:
        new_metrics.append({'id':metric_id,'name':name,'definition':definition,'unit':unit,'interpretation':direction,
            'purpose_status':'direct operational measurement under declared loopback/public-data scope'})
    paths=[args.study/'protocol.json',args.study/'summary.json',args.study/'verification.json',args.echo/'protocol.json',args.echo/'summary.json']
    result={'status':'completed','n_seeds':5,'seeds':protocol['seeds'],'codecs':protocol['codecs'],'run_count':20,
        'rounds_per_run':30,'flower_fit_calls':20*30*3,'protocol_scope':protocol,'metric_definitions':new_metrics,
        'codec_summary':summaries,'paired_vs_FP32':pairs,
        'limits':['Fresh dataset resplits, not external validation or deployment. Custom framed localhost transport invokes real Flower fit/FedAvg; not Flower gRPC.',
            'Application bytes include model/codec/residual/JSON/NPY/frame headers; TCP/IP/TLS kernel overhead is not captured.',
            'Echo goodput includes Python echo, framing and integrity checks. Pipeline goodput includes training. Neither is physical bandwidth capacity.',
            'CPU measured separately per process; sampled RSS cannot prove instantaneous peak. Coordinator excludes process startup; worker startup time recorded separately.',
            'Nonpositive or tiny recovery headroom is undefined; F1 ties can yield NA even when vector reconstruction succeeds.',
            'Any compressed-vs-FP32 performance contrast is descriptive across five paired seeds; no all-metric dominance claim.'],
        'sources':[{'path':relative_path(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (args.study/'report_export.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'completed','metrics':[r['id'] for r in new_metrics],'codecs':list(summaries)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'results/purpose_system/main')
    parser.add_argument('--echo',type=Path,default=ROOT/'results/purpose_system/transport_echo')
    main(parser.parse_args())
