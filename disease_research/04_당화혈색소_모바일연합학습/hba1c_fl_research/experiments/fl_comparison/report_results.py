"""Append measured experiments to the existing standalone design HTML."""
from collections import defaultdict
from datetime import datetime
from html import escape
import csv
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np
import torch
from scipy.stats import t

from cohort import ROOT,World,training_schedule,sha256
from learning import evaluate,DEVICE

OUT=ROOT/'experiments/fl_comparison/results/main'
REPORT=ROOT/'report'
HTML=REPORT/'HbA1c_FL_20만명_세부계산설계.html'
NAMES={'fedavg':'FedAvg','fedprox':'FedProx','scaffold':'SCAFFOLD',
       'fedavg_bn':'FedAvg + BN','fedbn':'FedBN'}
COLORS={'fedavg':'#718391','fedprox':'#087f81','scaffold':'#326bb6',
        'fedavg_bn':'#a56a24','fedbn':'#7954a1'}
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
report_font = next((name for name in ('Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR') if name in available_fonts), 'DejaVu Sans')
plt.rcParams.update({'font.family':report_font,'font.size':12,
    'axes.unicode_minus':False,'svg.fonttype':'none','svg.hashsalt':'hba1c-fl-20261017','axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#172839',
    'text.color':'#172839','xtick.color':'#536779','ytick.color':'#172839'})


def report_base(html_path, backup_path):
    """Preserve a supplied design, or create an independent repeatable report base."""
    html_path, backup_path = Path(html_path), Path(backup_path)
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if html_path.exists():
            shutil.copy2(html_path, backup_path)
        else:
            backup_path.write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>당화혈색소 연합학습 연구 결과</title><style>'
                'body{font:16px/1.7 sans-serif;margin:0;color:#172839}main{max-width:1100px;margin:auto;padding:24px}'
                'table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd4dc}'
                'svg,img{max-width:100%;height:auto}.svg-scroll{overflow:auto}</style></head>'
                '<body><main><p>연구 단계·실전 미적용. 과거 설계 문서 없이 실험 산출물에서 생성한 독립 보고서입니다.</p>'
                '</main></body></html>', encoding='utf-8')
    return backup_path.read_text(encoding='utf-8')


class Baseline:
    device=DEVICE
    def __init__(self,name,cal):self.name=name;self.cal=cal
    def predict(self,x,people):
        if self.name=='mean':return torch.zeros(len(x),device=x.device)
        if self.name=='previous_a1c':return torch.where(x[:,19]>.5,0.,x[:,12]*self.cal['x_scale'][12])
        c=self.cal['glucose_weight_linear_coefficients']
        rawf=x[:,0]*self.cal['x_scale'][0]+self.cal['x_mean'][0]
        raww=x[:,5]*self.cal['x_scale'][5]+self.cal['x_mean'][5]
        return c[0]+c[1]*rawf+c[2]*raww-self.cal['y_mean']


def number(value,places=4):return f'{value:.{places}f}' if value is not None else '—'
def difference_number(value):return f'{value:+.2e}' if 0<abs(value)<.00001 else f'{value:+.5f}'
def mean(values):return float(np.mean(values))
def sd(values):return float(np.std(values,ddof=1)) if len(values)>1 else 0.
def figure_artifact(fig,name,title):
    folder=REPORT/'figures';folder.mkdir(exist_ok=True)
    vector=folder/f'{name}.svg';raster=folder/f'{name}.png'
    fig.savefig(vector,facecolor='#fafcfb',metadata={'Date':None})
    fig.savefig(raster,facecolor='#fafcfb',dpi=180)
    plt.close(fig)
    svg=vector.read_text(encoding='utf-8');svg=svg[svg.index('<svg'):]
    # Each Matplotlib file has common IDs such as figure_1; prefix for inline use.
    for identifier in re.findall(r'id="([^"]+)"',svg):
        prefixed=name+'-'+identifier
        svg=svg.replace(f'id="{identifier}"',f'id="{prefixed}"').replace(f'url(#{identifier})',f'url(#{prefixed})').replace(f'href="#{identifier}"',f'href="#{prefixed}"')
    svg=svg.replace('<svg ',f'<svg class="chart" style="min-width:800px" role="img" aria-label="{escape(title)}" ',1)
    end=svg.index('>')+1;svg=svg[:end]+f'<title>{escape(title)}</title>'+svg[end:]
    return f'<figure class="chart-box"><div class="svg-scroll">{svg}</div><figcaption>{escape(title)} · <a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.png">PNG</a></figcaption></figure>'


def error_chart(rows):
    values=[[r['test']['all']['mae_pp'] for r in rows if r['method']==m] for m in NAMES]
    avgs=np.array([mean(v) for v in values]);stds=np.array([sd(v) for v in values])
    fig,ax=plt.subplots(figsize=(10,4.7));ax.set_facecolor('#fafcfb')
    ax.barh(np.arange(5),avgs,xerr=stds,color=list(COLORS.values()),height=.57,
        error_kw={'ecolor':'#172839','capsize':3,'elinewidth':1.1})
    ax.set_yticks(np.arange(5),list(NAMES.values()));ax.invert_yaxis()
    limit=float((avgs+stds).max())*1.48;ax.set_xlim(0,limit)
    for j,(value,std) in enumerate(zip(avgs,stds)):
        ax.text(value+std+limit*.015,j,f'{value:.4f} ± {std:.4f}',va='center',fontsize=11)
    ax.set_xlabel('사람별 평균절대오차 (%p)');ax.set_title('5회 반복 · 평균 ± 표준편차',loc='left',pad=16)
    ax.xaxis.grid(True,color='#dce3e5',linewidth=.7);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.18,right=.98,top=.86,bottom=.23)
    fig.text(.02,.035,'가상 인구 20만 명 · 관측된 검사 정답 평가 · 임상 검증 아님',fontsize=9,color='#536779')
    return figure_artifact(fig,'fl_comparison_mae','기법별 최종 평가 오차 — 막대는 평균, 검정 선은 반복 표준편차')


def convergence_chart(rows):
    curves={}
    for method in NAMES:
        values=defaultdict(list)
        for r in rows:
            if r['method']==method:
                for point in r.get('validation_curve',[]):
                    values[point['round']].append(point['offline_validation_mae_pp'])
        curves[method]=sorted((k,mean(v)) for k,v in values.items())
    if not any(curves.values()):return ''
    fig,ax=plt.subplots(figsize=(10,5.2));ax.set_facecolor('#fafcfb')
    for method,curve in curves.items():
        ax.plot([r for r,_ in curve],[v for _,v in curve],marker='o',markersize=4,
                linewidth=1.8,color=COLORS[method],label=NAMES[method])
    ax.set_xticks([50,150,250,350,410]);ax.set_ylim(bottom=0)
    ax.set_xlabel('가상 학습 기회 (회)');ax.set_ylabel('검증 평균절대오차 (%p)')
    ax.set_title('선택된 설정의 오프라인 고정 검증 · 5회 평균',loc='left',pad=16)
    ax.grid(True,color='#dce3e5',linewidth=.7);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False)
    fig.subplots_adjust(left=.11,right=.98,top=.86,bottom=.30)
    fig.text(.02,.025,'미래 검증 정답을 활용한 사후 진단 · 개인 학습 입력으로 사용하지 않음',fontsize=9,color='#536779')
    return figure_artifact(fig,'fl_comparison_convergence','동일 검증 표본에서의 학습 진행 비교')


def main():
    final=json.loads((OUT/'final_results.json').read_text(encoding='utf-8'))
    selection=json.loads((OUT/'optimization_selection.json').read_text(encoding='utf-8'))
    cal=json.loads((OUT/'calibration.json').read_text(encoding='utf-8'))
    rows=final['runs'];seeds=final['manifest']['seeds']
    successful_attempts=sum(r['successful_seeds'] for r in selection['candidates'])
    failure_rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('seed_*/*/failure.json'))]
    failures=''.join(f'<tr><td>{r["seed"]}</td><td>{NAMES[r["method"]]}</td><td>{r["lr"]:g}</td><td>{escape(r["error"])}</td></tr>' for r in failure_rows)
    # Compute transparent no-training baselines only after hyperparameters were fixed.
    baseline_rows=[]
    domain_rows=[]
    for seed in seeds:
        world=World.load(OUT/f'seed_{seed}'/'world.npz');training_schedule(world)
        for name in ('mean','previous_a1c','source_linear'):
            baseline_rows.append({'seed':seed,'baseline':name,'test':evaluate(Baseline(name,cal),world,'test')})
        for row in [r for r in rows if r['seed']==seed]:
            with np.load(Path(row['folder'])/'test_person_errors.npz') as errors:
                values=errors['mae'];observed=errors['counts']>0
            for group,mask in [('domain_negative',world.groups['domain']==0),
                               ('domain_zero',world.groups['domain']==1),
                               ('domain_positive',world.groups['domain']==2),
                               ('sensor',world.groups['sensor']),('no_sensor',~world.groups['sensor'])]:
                valid=observed&mask
                domain_rows.append({'seed':seed,'method':row['method'],'group':group,
                    'people':int(valid.sum()),'population':int(mask.sum()),
                    'mae_pp':float(values[valid].mean())})
    (OUT/'baseline_results.json').write_text(json.dumps(baseline_rows,indent=2),encoding='utf-8')
    (OUT/'domain_diagnostics.json').write_text(json.dumps(domain_rows,indent=2),encoding='utf-8')
    metrics=[]
    for r in rows:
        for group,m in r['test'].items():
            metrics.append({'method':r['method'],'seed':r['seed'],'group':group,
                            'lr':r['lr'],'mu':r['mu'],**m})
    metric_path=REPORT/'HbA1c_FL_기법비교_최종평가.csv'
    with metric_path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    aggregate={}
    for method in NAMES:
        subset=[r for r in rows if r['method']==method]
        aggregate[method]={g:{'mean_mae_pp':mean([r['test'][g]['mae_pp'] for r in subset]),
                                  'sd_mae_pp':sd([r['test'][g]['mae_pp'] for r in subset])}
                           for g in ('all','seen','new','high_a1c','low_engagement')}
        aggregate[method]['mean_execution_seconds']=mean([r['training_seconds'] for r in subset])
        aggregate[method]['payload_gb_decimal']=mean([r['payload_bytes_theoretical']/1e9 for r in subset])
    paired={}
    for method,base in (('fedprox','fedavg'),('scaffold','fedavg'),('fedbn','fedavg_bn')):
        diffs=[]
        for seed in seeds:
            a=mean([r['test']['all']['mae_pp'] for r in rows if r['method']==method and r['seed']==seed])
            b=mean([r['test']['all']['mae_pp'] for r in rows if r['method']==base and r['seed']==seed])
            diffs.append(b-a)
        margin=float(t.ppf(.975,len(diffs)-1)*sd(diffs)/np.sqrt(len(diffs))) if len(diffs)>1 else 0
        paired[method]={'baseline':base,'improvement_pp':mean(diffs),'ci95_low':mean(diffs)-margin,
                        'ci95_high':mean(diffs)+margin,'paired_seed_differences':diffs}
    selected_method=min(final['selected'],key=lambda m:final['selected'][m]['validation_mean_mae_pp'])
    observed_best=min(NAMES,key=lambda m:aggregate[m]['all']['mean_mae_pp'])
    summary={'status':'completed_simulation_not_clinical_validation','population_per_seed':final['manifest']['n'],
             'finalized_experiment_cells':len(selection['candidates'])*len(seeds),
             'completed_training_cells':successful_attempts,'diverged_cells':len(failure_rows),
             'seeds':seeds,'selected_by_validation':selected_method,'lowest_observed_test_mae':observed_best,
             'aggregate':aggregate,'paired_comparisons':paired,'selected':final['selected'],
             'scope':'one daily-moment behavior profile; not 27 worlds or original 4860-run matrix',
             'unexplored':final['unexplored']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    performance=''
    tails=''
    for method in NAMES:
        a=aggregate[method];chosen=final['selected'][method]
        performance+=f'<tr><td>{NAMES[method]}</td><td>{a["all"]["mean_mae_pp"]:.4f} ± {a["all"]["sd_mae_pp"]:.4f}</td><td>{a["seen"]["mean_mae_pp"]:.4f}</td><td>{a["new"]["mean_mae_pp"]:.4f}</td><td>{chosen["validation_mean_mae_pp"]:.4f}</td><td>η={chosen["lr"]:g}'+(f', μ={chosen["mu"]:g}' if method=='fedprox' else '')+'</td></tr>'
        rr=[r['test']['all'] for r in rows if r['method']==method]
        tails+=f'<tr><td>{NAMES[method]}</td><td>{mean([r["rmse_pp"] for r in rr]):.4f}</td><td>{mean([r["bias_pp"] for r in rr]):+.4f}</td><td>{mean([r["p90_person_mae_pp"] for r in rr]):.4f}</td></tr>'
    improvement=''
    for method,d in paired.items():
        interpretation='반복 변동 범위와 구분되는 방향' if d['ci95_low']>0 or d['ci95_high']<0 else '0을 포함 — 방향 불확실'
        if abs(d['improvement_pp'])<.0001:interpretation='0.0001 %p 미만의 차이 — 수치상 근접'
        improvement+=f'<tr><td>{NAMES[method]} 대 {NAMES[d["baseline"]]}</td><td>{difference_number(d["improvement_pp"])}</td><td>[{difference_number(d["ci95_low"])}, {difference_number(d["ci95_high"])}]</td><td>'+interpretation+'</td></tr>'
    costs=''
    for method in NAMES:
        rr=[r for r in rows if r['method']==method]
        state_mb=160000*1409*4/1e6 if method=='scaffold' else (200000*64*4*4/1e6 if method=='fedbn' else 0)
        devices=', '.join(f'{d} {sum(r["device"]==d for r in rr)}회' for d in sorted(set(r['device'] for r in rr)))
        costs+=f'<tr><td>{NAMES[method]}</td><td>{aggregate[method]["mean_execution_seconds"]:.1f}</td><td>{devices}</td><td>{aggregate[method]["payload_gb_decimal"]:.3f}</td><td>{state_mb:.1f}</td><td>{max(r["peak_cuda_allocated_bytes"] for r in rr)/1e6:.1f}</td><td>{max(r["process_rss_bytes"] for r in rr)/1e9:.2f}</td></tr>'
    baselines=''
    for name,label in (('mean','보정 자료의 전체 평균'),('previous_a1c','직전 검사 입력값 유지, 없으면 평균'),('source_linear','보정 자료의 공복혈당·체중 선형모형')):
        vals=[r['test']['all']['mae_pp'] for r in baseline_rows if r['baseline']==name]
        baselines+=f'<tr><td>{label}</td><td>{mean(vals):.4f} ± {sd(vals):.4f}</td></tr>'
    hpo=''
    for row in selection['candidates']:
        selected=final['selected'][row['method']]
        mark='선택' if row['lr']==selected['lr'] and row['mu']==selected['mu'] else ''
        hpo+=f'<tr><td>{NAMES[row["method"]]}</td><td>{row["lr"]:g}</td><td>{row["mu"]:g}</td><td>{number(row["validation_mean_mae_pp"],6)}</td><td>{row["successful_seeds"]}/{len(seeds)}</td><td>{mark}</td></tr>'
    groups=''
    for g,label in (('high_a1c','기본 당화혈색소 8% 이상'),('low_engagement','낮은 관심도 q&lt;0.3'),('seen_never_trained','개발 집단이지만 학습 반영 0회')):
        groups+='<tr><td>'+label+'</td>'
        for method in NAMES:
            vals=[r['test'][g]['mae_pp'] for r in rows if r['method']==method and r['test'][g]['mae_pp'] is not None]
            groups+='<td>'+number(mean(vals) if vals else None)+'</td>'
        groups+='</tr>'
    for g,label in (('domain_negative','입력 측정 편차 −4%'),('domain_zero','입력 측정 편차 0%'),
                    ('domain_positive','입력 측정 편차 +4%'),('sensor','연속혈당 관측 대상'),
                    ('no_sensor','연속혈당 관측 비대상')):
        groups+='<tr><td>'+label+'</td>'
        for method in NAMES:
            groups+='<td>'+number(mean([r['mae_pp'] for r in domain_rows if r['method']==method and r['group']==g]))+'</td>'
        groups+='</tr>'
    all_update_counts=[r['world']['accepted_updates'] for r in rows if r['method']=='fedavg']
    all_ever=[r['world']['ever_trained_people'] for r in rows if r['method']=='fedavg']
    observed_people=[r['test']['all']['people'] for r in rows if r['method']=='fedavg']
    single_fraction=mean([r['single_sample_updates']/r['world']['accepted_updates'] for r in rows if r['method']=='fedbn'])
    scaffold_error=max(r['scaffold_control_mean_max_error'] for r in rows if r['method']=='scaffold')
    naive=mean([r['test']['all']['mae_pp'] for r in baseline_rows if r['baseline']=='previous_a1c'])
    best_test=aggregate[observed_best]['all']['mean_mae_pp']
    naive_statement=('가장 낮은 신경망 오차도 직전 검사값 유지 기준보다 높았다. 따라서 알고리즘 간 상대 개선만으로 추가 예측 가치를 주장할 수 없다.'
                     if best_test>naive else '가장 낮은 신경망 오차는 직전 검사값 유지 기준보다 낮았다. 다만 이 합성 생성 가정 안에서의 결과이며 실제 임상 효과가 아니다.')
    stamp=datetime.now().strftime('%Y.%m.%d %H:%M')
    chosen_setting=f'학습률 {final["selected"][selected_method]["lr"]:g}'
    if selected_method=='fedprox':chosen_setting+=f', 근접 계수 {final["selected"][selected_method]["mu"]:g}'
    chosen_test=aggregate[selected_method]['all']
    section=f'''
<section id="optimization-results" data-training-cells="{len(selection['candidates'])*len(seeds)}" data-test-models="{len(rows)}" data-population="{final['manifest']['n']}">
<div class="section-no">추가 실험 / {stamp} KST · 실제 실행 결과</div>
<h2>FedProx · SCAFFOLD · FedBN: 20만 명에서의 비교</h2>
<p class="lede">검증 기준으로 선택된 구성은 <strong>{NAMES[selected_method]} · {chosen_setting}</strong>이다. 이 구성의 최종 평균절대오차는 <strong>{chosen_test['mean_mae_pp']:.4f} ± {chosen_test['sd_mae_pp']:.4f} %p</strong>였다. 직전 검사 입력값 유지 기준은 {naive:.4f} %p였다. 모두 아래 가상 조건에서 실제 계산한 결과다.</p>
<p>기본 가상 세계에서 설정 {len(selection['candidates'])}개와 난수 씨앗 {len(seeds)}개의 조합을 평가했다. {successful_attempts}개 조합은 훈련을 완료했고 {len(failure_rows)}개는 수치 발산으로 중단했다. <strong>5회 모두 완료한 설정만 검증셋으로 선택한 뒤</strong> 선택된 {len(rows)}개 모델을 최종 평가했다.</p>
<div class="note warn"><strong>범위:</strong> 실제 환자 실험이 아니라 20만 명의 <strong>가상 인구</strong> 실험이다. 기존 4,860회 설계 목록을 모두 실행한 결과가 아니다. 일별 요약 실행프로파일 E1의 한 가지 행태·검사·변화 조건을 비교했다. 고밀도 파형·업로드 지연·장기 임상 변화의 현실성은 아직 검증하지 않았다.</div>
<div class="grid3"><div class="metric"><b>{final['manifest']['n']:,}명</b><span>반복마다 개발 16만 / 신규 4만</span></div><div class="metric"><b>{len(selection['candidates'])*len(seeds)}개 조합</b><span>성공 훈련은 각 410일의 학습 기회 완료</span></div><div class="metric"><b>{len(rows)}개</b><span>선택된 모델의 최종 평가</span></div></div>
<p class="small">조합 수에는 별도의 소규모 검산과 CPU 동치 점검을 넣지 않았다. 자원 배치 변경 중 중단·재시작한 계산도 있었으며 실행 로그에 보존했다. 조합별 결과는 완료한 마지막 기록 또는 발산 기록이고, 중간 재시작은 별도 독립 반복으로 세지 않았다.</p>
<div class="flow" aria-label="공통 가상 세계 생성, 설정 탐색, 검증 선택, 최종 평가 순서"><div class="node"><b>① 같은 조건</b>20만 명 × 5회<br>기법마다 같은 개인·참여</div><div class="arrow">→</div><div class="node"><b>② 설정 탐색</b>{len(selection['candidates'])}개 설정 전부<br>발산 기록도 보존</div><div class="arrow">→</div><div class="node"><b>③ 검증으로 선택</b>5회 완료 설정 중<br>평균 오차 최소</div><div class="arrow">→</div><div class="node"><b>④ 최종 평가</b>기법별 5개 모델<br>합계 {len(rows)}개 평가</div></div>
<h3>무엇을 같게 맞췄는가</h3>
<p><strong>예측 과제는 현재 검사값 추정</strong>이다. 검사 직전까지의 자료로 해당 검사값을 추정하며, 90일 후의 미래 검사값 예측은 이번에 실행하지 않았다. 설정 선택의 기준은 관측된 검사값이 있는 사람들의 평균절대오차다. 참여가 적어 정답이 없는 사람의 성능까지 이 지표가 대표하지는 않는다.</p>
<p>이번 조건은 건강 상태에 따른 추가 기록·가용성 계수 β=0, 검사 참석 기본확률 0.70, 일별 로그혈당 지속성 φ=0.98이다. β=0이어도 개인 관심도·적극/간헐/휴면 상태·이탈에 따른 차이는 유지한다. 이 행태·장기 변화 수치들은 실측으로 확정한 값이 아니라 이번에 고정한 가정이며, 27개 조건 전체를 탐색하지 않았다.</p>
<p>같은 난수 씨앗에서는 개인 상태·검사·관측·초대·완료 참여자와 초기 공유 가중치가 동일하다. 730일 관찰, 90–499일 학습, 500–589일 검증, 590–729일 최종 평가, 상한 K=2,000명, 로컬 5회, 사람 동일 집계를 사용했다. 실제 반영 업데이트는 반복당 평균 {mean(all_update_counts):,.0f}회, 한 번 이상 반영된 개발 참여자는 평균 {mean(all_ever):,.0f}명이었다.</p>
<p class="small">410일은 학습을 시도할 수 있는 날짜 수다. 준비기 종료 직후에는 도착한 적격 검사 결과가 없어 집계를 건너뛰므로 실제 집계 횟수는 반복당 평균 {mean([r['actual_aggregations'] for r in rows]):.0f}회였다. 날짜 수를 실제 모델 갱신 횟수와 혼동하지 않는다.</p>
<p class="small">가상 완료 여부와 마감 지연은 같은 난수로 미리 정한 참여 스케줄이다. 실제 컴퓨터의 CPU·GPU 속도는 이 선정에 영향을 주지 않는다. SCAFFOLD의 추가 전송량이 완료율을 바꾸는 등의 비용 되먹임은 이번 비교에 넣지 않았다. 따라서 결과는 <strong>참여 조건을 같게 고정한 계산 알고리즘 비교</strong>다.</p>
<p>기본 분포는 NHANES 공복 성인 {cal['nhanes_fasting_adults']:,}명에서 적합했다. 생성 전 원본 사람을 개발용 {cal['development_donors']:,}명·보류 {cal['heldout_donors']:,}명으로 분리했다. Hall {cal['hall_bridge_donors']}명의 짧은 기록으로 평균·변동의 연결을 보조했다. 이 제한된 원본 수가 가상 인원만큼 증가한 것은 아니다.</p>
<p class="small">NHANES는 2017–2018 주기의 만 18세 이상, 8시간 이상·24시간 미만 공복, 양의 공복 하위표본 가중치, 필요한 변수의 완전 관측 조건으로 연결했다. 공복 가중치로 원본을 재표집·적합했지만 복합표본 층화·군집을 이용한 국가 수준 추론이나 완전 사례의 추가 무응답 보정은 하지 않았다. 변수·단위 근거는 <a href="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.htm">공복혈당 공식 코드북</a>과 <a href="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GHB_J.htm">당화혈색소 공식 코드북</a>을 따른다.</p>
<div class="table-wrap"><table><thead><tr><th>기법</th><th>실제 적용</th><th>비교할 기준</th></tr></thead><tbody>
<tr><td>FedProx</td><td>개인 손실에 μ/2·‖w−w서버‖² 추가</td><td>같은 20→64→1 구조의 FedAvg</td></tr>
<tr><td>SCAFFOLD</td><td>개인 경사에 c−cᵢ 보정, Option II로 cᵢ 갱신. 서버 c는 개발 인구 전체 cᵢ의 평균 관계 유지</td><td>같은 구조·같은 사람 가중치의 FedAvg</td></tr>
<tr><td>FedBN</td><td>은닉층 BN의 γ·β·이동 평균·분산을 개인별로 보관하고 집계에서 제외</td><td>동일 BN 구조의 FedAvg + BN을 필수 대조군으로 추가</td></tr>
</tbody><caption>일반 신경망은 1,409개, BN 신경망은 학습 매개변수 1,537개다. FedBN과 일반 FedAvg의 차이를 알고리즘 효과만으로 해석하지 않는다.</caption></table></div>
<details><summary>세 방법의 계산식과 용어 해설</summary>
<p><strong>FedProx</strong>는 개인 모델이 서버에서 받은 모델과 너무 멀어지는 것을 벌점으로 제어한다. <strong>SCAFFOLD</strong>는 사람마다 다른 경사 방향을 보정량으로 상쇄한다. <strong>FedBN</strong>은 배치 정규화(Batch Normalization)의 개인 상태를 서버 평균에 섞지 않는다.</p>
<div class="formula">FedProx: Lᵢ(w) + (μ/2)‖w−wʳ‖²<br>SCAFFOLD 개인 갱신: w ← w − η[∇Lᵢ(w) + c − cᵢ]<br>cᵢ⁺ = cᵢ − c + (wʳ−wᵢ⁺)/(Eη)<br>c⁺ = c + (1/N)Σᵢ∈S(cᵢ⁺−cᵢ)<br>wʳ⁺¹ = (1/|S|)Σᵢ∈S wᵢ⁺<span class="explain">Lᵢ는 개인 유효 검사 표본의 평균제곱오차, μ는 벌점 강도, η는 학습률, E=5는 로컬 갱신 횟수다. c는 서버 보정량, cᵢ는 개인 보정량, N=160,000은 개발 집단 전체 인원, S는 그날 완료한 사람 집합이다. 서버 보정량 분모는 당일 참여자 수가 아니라 전체 개발 인원이다.</span></div>
<div class="formula">정규화 출력 = γᵢ·(h−mᵢ)/√(vᵢ+10⁻⁵) + βᵢ<span class="explain">h는 은닉층 값, mᵢ·vᵢ는 개인의 이동 평균·분산, γᵢ·βᵢ는 학습 가능한 크기·위치다. FedBN은 이 네 상태를 모두 개인별로 유지한다. 표본 2개 이상일 때 현재 배치의 평균·분산으로 학습하고, 이동 통계는 계수 0.1로 갱신한다. 표본이 1개면 아래의 별도 규칙을 쓴다.</span></div>
<p>평균절대오차(Mean Absolute Error, MAE)는 예측과 검사값 차이의 절댓값 평균이다. 예를 들어 검사값 7.0%, 예측 7.3%의 오차는 0.3 %p다. 난수 씨앗은 같은 가상 세계를 재생성하는 번호이며, 5회 반복은 서로 다른 가상 세계 5개를 뜻한다.</p>
</details>
<h3 id="comparison-performance">최종 평가 오차</h3>
<p>“개발 집단”은 학습 후보인 16만 명이며 실제 학습 반영 0회인 사람도 포함한다. “학습 제외 집단”은 연합학습에 한 번도 쓰지 않은 4만 명이다. 여기서 신규란 <strong>연합학습이 보지 못한 사용자</strong>라는 뜻이다. 이들도 과거 관측·검사 이력은 있으므로 가입 직후 아무 기록도 없는 상황과는 다르다.</p>
<p class="small">설정 선택에는 두 집단의 <strong>검증 기간</strong> 관측 정답을 함께 사용했다. 따라서 4만 명은 개인 업데이트에서 제외한 집단이며, 모델 선택 전 과정에서 완전히 숨겨 둔 외부 사용자 집단은 아니다. 최종 평가 기간의 정답은 설정 선택에 쓰지 않았다. 사용자까지 완전히 독립시킨 외부 검증은 미실행이다.</p>
<p><strong>평가 범위:</strong> 최종 검사 정답이 관측된 사람은 반복당 평균 {mean(observed_people):,.0f}명으로 등록 인구의 {mean(observed_people)/final['manifest']['n']*100:.1f}%였다. 표의 “전체”는 이 사람들에 대한 평균이며, 나머지 사람의 미관측 정답 오차까지 계산한 값이 아니다.</p>
<div class="table-wrap"><table><thead><tr><th>방법</th><th>전체 MAE 평균 ± 표준편차 (%p)</th><th>개발 집단 (%p)</th><th>학습 제외 집단 (%p)</th><th>검증 MAE (%p)</th><th>검증에서 선택한 설정</th></tr></thead><tbody>{performance}</tbody><caption>MAE는 사람별 평균절대오차를 먼저 구한 뒤 사람들을 평균한다. ±는 5회 가상 세계 반복의 표준편차이며 실제 환자 모집단의 신뢰구간이 아니다.</caption></table></div>
{error_chart(rows)}
<details><summary>큰 오차와 과대·과소 추정 확인</summary><div class="table-wrap"><table><thead><tr><th>방법</th><th>제곱평균제곱근오차 (%p)</th><th>평균 예측 − 정답 (%p)</th><th>사람별 오차의 90백분위 (%p)</th></tr></thead><tbody>{tails}</tbody><caption>모두 5회 반복 평균이다. 제곱평균제곱근오차(Root Mean Squared Error)는 큰 오차에 더 민감하다. 평균 예측 − 정답이 양수면 과대 추정 경향이고, 90백분위는 평가된 사람의 90%가 그 값 이하의 개인 평균절대오차를 갖는 경계다. 평균 오차가 낮더라도 일부 사람의 큰 오차가 사라지는 것은 아니다.</caption></table></div></details>
<div class="note"><strong>선택과 평가 구분:</strong> 검증 평균이 가장 낮아 사전에 선택되는 방법은 <strong>{NAMES[selected_method]}</strong>이다. 최종 평가에서 관측된 가장 낮은 오차는 <strong>{NAMES[observed_best]} {best_test:.4f} %p</strong>였다. 후자의 순위를 보고 학습 설정을 다시 선택하지 않았다. “최적”은 이번 탐색 범위·생성 가정 안의 의미다.</div>
<h3 id="comparison-paired">기준 방식 대비 개선과 반복 변동</h3>
<div class="table-wrap"><table><thead><tr><th>짝지은 비교</th><th>기준 오차 − 기법 오차 (%p)</th><th>반복 차이의 95% t 구간</th><th>해석</th></tr></thead><tbody>{improvement}</tbody><caption>양수는 오차 감소다. 같은 씨앗의 차이 5개로 계산했으며 자유도 4다. 작은 반복 수와 공유 원본 자료 때문에 임상적 확실성이나 모든 행태에서의 우월성을 뜻하지 않는다. 아주 작은 차이는 과학 표기로 보존했으며 0.0001 %p 표시는 수치 설명 기준일 뿐 임상적 유의성 기준이 아니다.</caption></table></div>
<h3>단순한 방법보다 추가 가치가 있는가</h3>
<div class="table-wrap"><table><thead><tr><th>별도 무훈련 기준</th><th>최종 MAE 평균 ± 표준편차 (%p)</th></tr></thead><tbody>{baselines}</tbody><caption>이 기준들은 최종 평가를 보고 적합하지 않았다. 직전 검사값은 현재 검사 전에 공개된 입력만 사용하고 없으면 사전 평균을 사용한다. 신경망과 같은 표준화·범위 제한을 거친 입력값을 원단위로 복원한 기준이므로 원자료의 무제한 직전 값과 구분한다.</caption></table></div>
<p><strong>{naive_statement}</strong></p>
<h3>어떤 사람에게 오차가 남는가</h3>
<div class="table-wrap"><table><thead><tr><th>집단</th>{''.join('<th>'+NAMES[m]+'</th>' for m in NAMES)}</tr></thead><tbody>{groups}</tbody><caption>모든 값은 사람별 평균절대오차의 반복 평균, 단위 %p. 검사 결과가 관측된 사람만 평가할 수 있어 미관측 집단 전체의 오차는 아직 알 수 없다.</caption></table></div>
<p class="small">측정 편차 −4%·0%·+4%는 사람별로 고정한 세 가지 입력 환경의 가정이며, 특정 기기의 실측 정확도가 아니다. 연속혈당 관측 대상은 가상 인구의 20%로 무작위 배정했다. 집단 진단은 최종 평가 후의 설명용이며 설정 선택에 사용하지 않았다.</p>
<h3 id="comparison-convergence">학습 진행과 비용</h3>
{convergence_chart(rows)}
<p class="small">곡선은 선택된 설정의 공통 고정 검증 표본에 대한 오프라인 평가다. 매 시점 실제로 그 미래 정답을 이미 알았다는 의미가 아니며, 이 점검값은 개인 학습 입력으로 전달하지 않았다.</p>
<div class="table-wrap"><table><thead><tr><th>방법</th><th>실행 평균 (초)</th><th>훈련 장치</th><th>이론적 모델·보정량 왕복 (GB)</th><th>영구 개인 상태 (MB)</th><th>최대 GPU 텐서 할당 (MB)</th><th>종료 시 프로세스 상주 메모리 최대 (GB)</th></tr></thead><tbody>{costs}</tbody><caption>시간은 완료한 마지막 훈련 기록으로, 중간 검증 평가는 포함하지만 세계 생성·체크포인트 저장·재시작 전 소요 시간은 제외된다. CPU와 GPU를 함께 사용했고 다른 작업과 자원도 공유했으므로 독립적인 속도 순위로 해석할 수 없다. GPU 최대치는 GPU 훈련 반복에서만 측정했다. 상주 메모리는 캐시·공유 라이브러리를 포함한 프로세스 종료 시점 관측값이며 학습 중 최고점이 아니다. 전송량은 실제 통신이 아닌 32비트 배열 길이 계산이며 헤더·암호화는 제외했다. GB=10억 바이트, MB=100만 바이트.</caption></table></div>
<p class="small">실행 장비: Ryzen 5 3600X, 메모리 약 64 GiB, RTX 4080 약 16 GiB. 개인 상태는 CPU에 두고 선택된 사람의 배열만 GPU에 올렸다. GPU 텐서 할당 수치는 드라이버·실행 문맥·다른 프로세스의 메모리를 제외한다. <a href="../experiments/fl_comparison/results/main/runtime_environment.json">Python·라이브러리·드라이버 버전 및 자원 공유 기록</a>을 보관했다.</p>
<h3>FedBN과 희소한 검사값의 관계</h3>
<p>반영된 개인 업데이트 중 표본 1개인 비율은 평균 <strong>{single_fraction*100:.1f}%</strong>였다. 이 경우 BN의 이동 통계는 고정하고 기존 통계로 학습했으며 γ·β는 갱신할 수 있게 했다. 이는 단일 표본에서 분산 0으로 정규화하는 잘못된 구현을 피하는 명시적 예외처리다.</p>
<p>신규 사용자는 학습한 BN 상태가 없어 γ=1, β=0, 평균=0, 분산=1로 시작했다. 따라서 신규 사용자 성능에는 <strong>개인 상태 부재</strong>가 포함된다. 새 사용자의 비지도 BN 보정은 아직 비교하지 않았고, 이 결과로 충분한 로컬 자료가 있는 원논문의 환경까지 결론을 확대하지 않는다.</p>
<h3>구현과 검산 근거</h3>
<ul><li>독립 개인 학습과 묶음 계산 동치, FedProx μ=0 동치, BN 경사·불편 이동분산의 PyTorch 기준 동치, 단일 표본 처리, FedBN 상태 격리 검사를 통과했다.</li><li>마스크로 제외한 미래 슬롯의 입력·정답을 크게 바꿔도 개인 가중치와 정규화 통계 갱신이 바뀌지 않는 검산을 통과했다. 이는 계산상 제외가 실제로 유지되는지 확인하는 검사이며 물리적 접근 권한 격리를 입증하는 것은 아니다.</li><li>SCAFFOLD의 서버 c와 전체 개인 cᵢ 평균의 최대 차이는 <strong>{scaffold_error:.3g}</strong>였다. 관측이 늘어 개인 목적함수가 바뀌는 환경이므로 원논문의 수렴 보장을 그대로 주장하지 않는다.</li><li>검사 결과는 채혈 2일 후부터 학습에 사용했고, 현재 검사 정답은 해당 예측 입력에 넣지 않았다. 원본 제공자 분리는 가상 복제 전에 수행했다.</li></ul>
<details><summary>전체 설정 탐색 결과 {len(selection['candidates'])}개와 실패 기록 보기</summary><div class="table-wrap"><table><thead><tr><th>방법</th><th>학습률 η</th><th>근접 계수 μ</th><th>검증 MAE (%p)</th><th>성공 반복</th><th>선택</th></tr></thead><tbody>{hpo}</tbody><caption>일반 모델은 학습률 0.001·0.003·0.01, FedProx는 μ=0.001·0.01·0.1과 전부 교차했다. 발산 확인 후 최종 평가를 열기 전에 SCAFFOLD에 0.0001·0.0003, 두 BN 모델 모두에 0.00001·0.00003·0.0001·0.0003을 추가했다. —는 5회 모두 성공하지 않아 선택 대상에서 제외한 설정이다. 성공한 반복만 골라 평균하지 않았다.</caption></table></div>
<div class="table-wrap"><table><thead><tr><th>난수 씨앗</th><th>방법</th><th>학습률</th><th>중단 이유</th></tr></thead><tbody>{failures or '<tr><td colspan="4">중단된 설정 없음</td></tr>'}</tbody><caption>Non-finite local update는 개인 가중치에 유효한 유한수가 아닌 값이 발생했다는 뜻이다. 실패도 결과로 남겼다. SCAFFOLD 서버 학습률 1, 정규화 이동통계 계수 0.1 등은 고정했으며 추가 탐색하지 않았다.</caption></table></div></details>
<h3>실행프로파일 E1의 범위</h3>
<p>아래 01–22절은 원래의 전체 설계다. 그 안의 “미구현·미적합·예정” 표시는 작성 당시의 전체 설계 상태를 설명하며, 이번에 실제 실행한 부분은 이 추가 절과 다음 표를 기준으로 읽는다. 계산기의 과거 커널 실측도 이번 학습 전체의 실측 시간과 다르다.</p>
<div class="table-wrap"><table><thead><tr><th>항목</th><th>원래 전체 설계</th><th>이번 실행프로파일 E1</th></tr></thead><tbody>
<tr><td>자료 보정</td><td>출처별 연결·단위 점검 후 관계 적합 예정</td><td>NHANES 2017–2018 공복 성인과 Hall 자료를 실제로 읽어 원본 제공자 분리 후 보정. 2021–2023 자료를 사용했다고 간주하지 않음</td></tr>
<tr><td>관측 해상도</td><td>분 단위 관측·업로드 사건, 연속혈당 파형·연속 누락 블록</td><td>일별 기록·가용성, 연속혈당의 평균·분산 요약과 일별 독립 10% 센서 누락. 기록은 당일 공개된 것으로 단순화</td></tr>
<tr><td>개인 로컬 표본</td><td>유효 표본 최대 8개, 개인 상태별로 가변</td><td>9개 예약 검사 슬롯을 마스크 처리. 학습에는 결과가 도착한 과거 365일의 유효 표본만 사용</td></tr>
<tr><td>모형과 개인화</td><td>선형·신경망, 잔차 개인 보정 등 비교</td><td>일반 신경망과 정규화 신경망을 비교. 잔차 개인 보정은 적용하지 않음</td></tr>
<tr><td>검사 정답</td><td>실제 자료 보정 관계와 개인효과, 독립 생성기 민감도</td><td>원본 기본 검사값 + 잠재 평균혈당 변화에 비례한 변화 + 가정한 검사 잡음. 독립 생성기는 미실행</td></tr>
</tbody><caption>계산을 실행 가능하게 만든 범위와 원래의 확장 설계를 분리했다. 이 결과만으로 전체 시뮬레이터의 현실성이 검증되었다고 결론 내리지 않는다.</caption></table></div>
<p>개인의 기본 당화혈색소는 실제 자료의 결합분포를 보존하고, 이후 변화는 일별 잠재 혈당의 가중 평균 변화와 가정한 검사 잡음 0.12 %p로 생성했다. 90일 전의 잠재 사전 이력은 개인 기본 상태로 시작했다. 활동 입력은 결측으로 두고 연속혈당은 14일 요약 모멘트만 계산했다. 원래 설계의 업로드 지연·고밀도 파형·독립 G2 생성기까지 구현·검증한 것은 아니다.</p>
<div class="formula">가상 검사값 Yᵢ,d = 원본 기본 검사값 Aᵢ + b·(Bᵢ,d−Mᵢ) + εᵢ,d<br>b = {cal['glucose_weight_linear_coefficients'][1]:.6f}, εᵢ,d ~ 정규분포(0, 0.12²)<br>Mᵢ = max(40, {cal['fasting_to_mean_bridge'][0]:.4f} + {cal['fasting_to_mean_bridge'][1]:.6f}·기본 공복혈당ᵢ)<span class="explain">B는 과거 90일의 잠재 일별 평균혈당을 시간상수 45일로 가중한 평균이다. b는 개발용 원자료의 가중 선형 적합 계수이며, 이를 장기 변화의 기울기로 옮긴 것은 검증되지 않은 가정이다. 식의 혈당 단위는 mg/dL, Y와 ε의 단위는 당화혈색소 % 및 그 차이 %p다. 현실적 장기 치료 효과·적혈구 수명 변화는 적합하지 않았다.</span></div>
<div class="table-wrap"><table><thead><tr><th>한 것</th><th>안 한 것 / 미탐색</th></tr></thead><tbody><tr><td>기본 20만 명 × 5회 반복, 3개 기법과 두 대조군의 실제 훈련·평가</td><td>27개 세계 전체, K=200·10,000, 선형 모델 비교, 원래 4,860회 목록 실행</td></tr><tr><td>설정 전수 탐색과 검증셋 선택, 최종 평가 분리</td><td>더 넓은 학습률·근접 계수, SCAFFOLD 서버 학습률, BN 통계 계수 탐색</td></tr><tr><td>관측된 검사값의 개인·집단별 오차, 실제 계산량·이론적 전송량 기록</td><td>미관측 정답 감사, 신규 사용자 BN 보정, 실제 업로드 지연·암호 통신 비용</td></tr><tr><td>실제 자료로 기본 분포 보정, 가정과 실험 수치 분리</td><td>실제 장기 임상자료 검증·한국인 모바일 인구 대표성·개인정보 보호 입증</td></tr></tbody><caption>방법이 개선되지 않거나 불안정한 경우에도 이 자료·이 일별 생성 방식·이 설정 범위 한정의 결과다. 미탐색 항목을 포함한 전반적 우열로 닫지 않는다.</caption></table></div>
<h3>원자료와 재현 파일</h3>
<div class="actions"><a class="btn primary" href="../experiments/fl_comparison/results/main/optimization_selection.json" download>실행한 최적화 설정·검증 결과 JSON</a><a class="btn" href="HbA1c_FL_기법비교_최종평가.csv" download>기법 비교 결과 CSV</a></div>
<p><a href="HbA1c_FL_기법비교_최종평가.csv">개인 집단별 최종 평가 CSV</a> · <a href="../experiments/fl_comparison/results/main/summary.json">결과 요약 JSON</a> · <a href="../experiments/fl_comparison/results/main/optimization_selection.json">설정 선택 전체 기록</a> · <a href="../experiments/fl_comparison/results/main/manifest.json">실행 매니페스트·소스 해시</a> · <a href="../experiments/fl_comparison/results/main/world_audit.json">가상 세계의 시간·출처·분모 점검</a> · <a href="../experiments/fl_comparison/unit_checks.json">구현 검산 7개</a> · <a href="../experiments/fl_comparison/results/main/artifact_checksums.json">최종 산출물 해시</a></p>
<details><summary>동일 실험을 다시 실행하는 방법</summary><p>연구 폴더를 작업 경로로 두고 기존 Python 3.13 환경에서 아래 순서로 실행한다. 이미 완료된 설정은 기록을 재사용하므로, 완전 재훈련은 별도 결과 경로를 지정해야 한다. 본 문서 생성기는 기본 결과 경로를 읽는다.</p><pre>python experiments/fl_comparison/test_learning.py
python experiments/fl_comparison/run_comparison.py
python experiments/fl_comparison/audit_worlds.py
python experiments/fl_comparison/report_results.py</pre><p class="small">주 실행기는 선택 전에는 최종 평가를 하지 않는다. 개별 모델 파일과 가상 세계 파일을 보존했으며, CPU에 개인 상태를 저장하므로 20만 개 모델 전체를 GPU에 상주시킬 필요는 없다.</p></details>
<p class="small">근거: <a href="https://github.com/litian96/FedProx">FedProx 공식 구현</a>, <a href="https://proceedings.mlr.press/v119/karimireddy20a.html">SCAFFOLD 원논문·Algorithm 1 Option II</a>, <a href="https://github.com/med-air/FedBN">FedBN 공식 구현</a>. 추가 의존성 설치 없이 기존 NumPy·pandas·SciPy·PyTorch 환경에서 독립 구현했다.</p>
</section>
'''
    backup=REPORT/'HbA1c_FL_20만명_세부계산설계_기법비교추가전_20261017.html'
    # Rebuild from the preserved design, so updating results never duplicates sections.
    source=report_base(HTML, backup)
    source=source.replace('<main>','<main>'+section,1)
    source=source.replace('<b>계산 설계 안내</b>','<b>계산 설계 안내</b><a href="#optimization-results">추가 · 실제 기법 비교 결과</a>',1)
    source=source.replace('계산 설계 v1.0 · 연구 단계 / 실전 미적용',f'계산 설계 v1.1 · {stamp} 기법 비교 추가 · 연구 단계 / 실전 미적용',1)
    source=source.replace('[2026.10.16 14:16 KST]',f'[{stamp} KST]',1)
    source=source.replace('<title>20만 명, 하나의 가상 연구실 — 당화혈색소 연합학습 계산 설계</title>',
                          '<title>20만 명의 당화혈색소 연합학습 — 계산 설계와 실제 기법 비교 결과</title>')
    source=source.replace('이번 문서 작성에서는 원자료 표본 수를 다시 계산하지 않았다. 계산 엔진 구현 전에 출처별 적재·연결·단위 검증을 재실행한다.',
        f'최초 설계 작성 시에는 원자료를 재집계하지 않았으나, 추가 E1 실험에서 NHANES 2017–2018 공복 성인 {cal["nhanes_fasting_adults"]:,}명을 다시 연결·보정했다. 다른 출처 전체의 재집계와 종단 현실성 검증은 미완료다.')
    source=source.replace('미적합. 실행 전 보정 산출물이 있어야 현실성 실험을 허용',
        'E1 기본 분포 적합 완료. 장기 변화·참여의 실측 적합과 확장 조건 검증은 미완료')
    source=source.replace('위 의사코드는 순서 설명이며 아직 구현된 훈련 엔진이 아니다.',
        '위 의사코드는 전체 설계의 순서 설명이다. 일별 실행프로파일 E1의 구현·실행 결과는 문서 첫 추가 절에 제시했다.')
    source=source.replace('배치 정규화·클라이언트 간 손실 혼합을 쓰지 않는다.',
        '기본 신경망은 배치 정규화를 쓰지 않는다. 추가 기법 비교에서는 별도의 정규화 신경망도 사용했다. 사람 간 손실을 섞지는 않는다.')
    source=source.replace('건강·참여 생성기의 실제 계수 적합과 장기 현실성 검증',
        '장기 건강·참여 계수의 실제 자료 적합과 현실성 검증 (E1의 기본 결합분포만 적합)')
    source=source.replace('전체 훈련 엔진 구현·실행, 확장 간 모든 조합 탐색',
        'E1 외 전체 엔진(분 단위 업로드 등) 구현, 확장 간 모든 조합 탐색')
    source=source.replace('이 문서 작성 중 원자료 재집계, 실제 종단자료 신규 확보',
        '실제 종단자료 신규 확보·장기 현실성 검증 (E1 보정용 재집계는 추가 절에서 완료)')
    source=source.replace('이 문서는 연구 계산 설계의 완성본이다. 구현·보정·실험이 이미 끝났다는 의미가 아니다. 미확인 고리는 다음 단계의 합격 기준으로 남긴다.',
        '전체 계산 설계와 부분 실행 결과를 구분한다. E1 기본 자료 보정과 기법 비교는 추가 절에서 완료했고, 전체 확장 설계의 미확인 고리는 다음 단계의 합격 기준으로 남긴다.')
    source=source.replace('문서 생성 시각: 2026-10-16 KST.',f'최초 설계: 2026-10-16 KST. 기법 비교 갱신: {stamp} KST.')
    source=source.replace('기본 계산 설정 JSON 내려받기','전체 설계 기본 설정 JSON 내려받기')
    source=source.replace('</style>','@media print{#optimization-results svg.chart{min-width:0!important}#optimization-results .svg-scroll{overflow:visible}}\n</style>',1)
    source=source.replace('JSON은 문서의 기본 설정·근거 상태를 기계가 읽도록 정리한 설계 파일이다.',
        '이 JSON은 원래 전체 설계의 기본 설정이다. 이번 E1 실험에서 선택한 학습률·기법은 첫 추가 절의 최적화 설정 JSON에 별도로 보관했다.')
    HTML.write_text(source,encoding='utf-8')
    checks=Path(__file__).parent/'unit_checks.json'
    files=list(Path(__file__).parent.glob('*.py'))+[HTML,metric_path]+([checks] if checks.is_file() else [])+list(OUT.glob('*.json'))+list((REPORT/'figures').glob('fl_comparison_*'))
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in files if p.name!='artifact_checksums.json'}
    (OUT/'artifact_checksums.json').write_text(json.dumps({'built_at_kst':stamp,'sha256':hashes,
        'missing_optional_artifacts':[] if checks.is_file() else ['unit_checks.json']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
