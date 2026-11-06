import hashlib
import html
import json
import math
from pathlib import Path

UPSTREAM_COMMIT = 'b21d628b8f5042a00d46da5a2280987c6469f281'
METRICS = {
    'global_accuracy': 'test_acc',
    'global_loss': 'test_loss',
    'client_weighted_accuracy': 'eval_acc',
    'client_weighted_loss': 'eval_loss',
    'cumulative_upload_mib': 'comm-mb-up',
    'cumulative_download_mib': 'comm-mb-down',
    'cumulative_total_mib': 'comm-mb-total',
    'dp_epsilon': 'dp-epsilon',
}


def read_history(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get('config'), dict) or not isinstance(data.get('history'), list):
        raise ValueError('Expected a deFAI history.json object with config and history')
    records, previous = [], -1
    for row in data['history']:
        if not isinstance(row, dict):
            raise ValueError('Each history row must be an object')
        r = row.get('round')
        if isinstance(r, bool) or not isinstance(r, int) or r <= previous:
            raise ValueError('Rounds must be unique, increasing nonnegative integers')
        previous = r
        record = {'round': r}
        for target, source in METRICS.items():
            value = row.get(source)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Invalid numeric metric: ' + source)
                if 'accuracy' in target and not 0 <= value <= 1:
                    raise ValueError('Accuracy must be a fraction in [0, 1]')
                if target.startswith('cumulative_') and value < 0:
                    raise ValueError('Communication volume cannot be negative')
            record[target] = value
        record.update(global_f1=None, local_f1=None, precision=None, recall=None)
        records.append(record)
    return {
        'schema': 'defai-addons.host-history.v1',
        'source': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'verified_contract_commit': UPSTREAM_COMMIT,
        'config': data['config'], 'backend': data.get('backend'),
        'clients': data.get('clients'),
        'network_transport_measured': data.get('network_transport_measured', False),
        'records': records,
        'limitations': [
            'F1, precision, recall, client topology and matching are unavailable in the host history contract.',
            'eval_acc is sample-weighted client evaluation accuracy, not mean local F1.',
            'Communication is recorded in MiB; in-process traffic is an estimate, not measured network traffic.',
            'The host writes history.json after completion; this import is not live monitoring.',
        ],
    }


def write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)


def _chart(reports, metric, title):
    colors = ['#2563eb', '#b45309', '#15803d', '#9333ea']
    points = [(row['round'], row[metric]) for r in reports for row in r['records'] if row[metric] is not None]
    if not points:
        return '<section><h2>' + title + '</h2><p>해당 지표가 기록되지 않았습니다.</p></section>'
    xmax = max(1, max(x for x, y in points))
    ymax = 1 if 'accuracy' in metric else max(1e-12, max(y for x, y in points))
    svg = '<svg viewBox="0 0 700 260" role="img" aria-label="' + title + '">'
    for i in range(5):
        yy = 220 - i * 45
        svg += f'<path d="M60 {yy}H680" stroke="#e2e8f0"/><text x="5" y="{yy+4}" font-size="12">{ymax*i/4:.3g}</text>'
    for i, report in enumerate(reports):
        segment = []
        for row in report['records'] + [{metric: None}]:
            if row[metric] is None:
                if segment:
                    svg += f'<polyline points="{" ".join(segment)}" fill="none" stroke="{colors[i%4]}" stroke-width="2.5"/>'
                segment = []
            else:
                x, y = 60 + 620 * row['round']/xmax, 220 - 180 * row[metric]/ymax
                segment.append(f'{x:.2f},{y:.2f}')
                svg += f'<circle cx="{x}" cy="{y}" r="2" fill="{colors[i%4]}"/>'
    svg += f'<text x="60" y="246">0</text><text x="620" y="246">{xmax} rounds</text></svg>'
    return '<section><h2>' + title + '</h2>' + svg + '</section>'


def render_histories(paths, output):
    reports = [read_history(p) for p in paths]
    if not reports:
        raise ValueError('At least one history is required')
    e = lambda x: html.escape(str(x))
    body = '<h1>deFAI 실험 결과 연동</h1><p>원본 결과를 읽어 생성한 평가 화면 | 정확도: 0~1 | 통신량: MiB</p>'
    body += '<p>범례: ' + ' / '.join(f'{i+1}: {e(r["config"].get("strategy", "unknown"))} ({e(Path(r["source"]).parent.name)})' for i, r in enumerate(reports)) + '</p>'
    for metric, title in [('global_accuracy', '글로벌 정확도'), ('client_weighted_accuracy', '클라이언트 표본 수 가중 정확도'), ('cumulative_total_mib', '누적 학습 통신량')]:
        body += _chart(reports, metric, title)
    body += '<h2>최종 기록 및 출처</h2><table><tr><th>전략</th><th>라운드</th><th>글로벌 정확도</th><th>클라이언트 가중 정확도</th><th>누적 MiB</th><th>F1</th></tr>'
    for r in reports:
        last = r['records'][-1] if r['records'] else {}
        values = [r['config'].get('strategy'), last.get('round'), last.get('global_accuracy'), last.get('client_weighted_accuracy'), last.get('cumulative_total_mib'), None]
        body += '<tr>' + ''.join('<td>' + (e(v) if v is not None else '기록 없음') + '</td>' for v in values) + '</tr>'
    body += '</table><p>서로 다른 데이터, 분할, 모델, 학습 설정의 결과는 공정한 성능 순위로 해석할 수 없습니다. 지역 분산 결과는 별도의 비교 모듈에서 생성합니다.</p>'
    for r in reports:
        body += '<details><summary>' + e(r['source']) + '</summary><pre>' + e(json.dumps({k: v for k,v in r.items() if k != 'records'}, ensure_ascii=False, indent=2)) + '</pre></details>'
    style = 'body{font-family:Segoe UI,Malgun Gothic,sans-serif;max-width:1100px;margin:32px auto;color:#172033;padding:20px}section{display:inline-block;width:48%;vertical-align:top}svg{width:100%}h2{font-size:18px}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #cbd5e1}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
    write_new(output, '<!doctype html><html lang="ko"><meta charset="utf-8"><title>deFAI 결과 연동</title><style>' + style + '</style><body>' + body + '</body></html>')
    return reports
