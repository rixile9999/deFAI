# -*- coding: utf-8 -*-
"""
[E3] 연합의 이득: 시군구 단독(격리) 적합 vs 연합 적합
======================================================
각 시군구가 자기 감염쌍만으로 Burr XII를 적합(격리)했을 때의 모수 산포·실패율을
연합(=풀링 등가) 추정과 비교해 정량화한다.
산출: results/e3_benefit.json, e3_isolated_fits.csv, e3_isolated_vs_federated.png
"""
from __future__ import annotations
import sys, io, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fl_pipeline.federated_burr import load_district_pairs

RESULTS = ROOT / 'results'
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ''   # 예: '_nonsat'
manifest = json.loads((RESULTS / f'abm_run_manifest{SUFFIX}.json').read_text(encoding='utf-8'))
e1 = json.loads((RESULTS / f'e1_correctness{SUFFIX}.json').read_text(encoding='utf-8'))
fed = e1['federated']

district_T = load_district_pairs(manifest['gi_csv'])
MIN_N = 30                      # 이보다 작으면 적합 시도 자체가 무의미

rows = []
for sgg, T in sorted(district_T.items()):
    row = {'sigungu': sgg, 'n': len(T)}
    if len(T) < MIN_N:
        row['status'] = 'too_small'
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                c, k, loc, s = stats.burr12.fit(T, floc=0)
            # 발산/퇴화 판정: 경계 근접 또는 비유한
            if not all(np.isfinite([c, k, s])) or c > 80 or k < 0.01 or s > 200:
                row['status'] = 'degenerate'
            else:
                row.update(status='ok', c=c, k=k, sigma=s)
        except Exception:
            row['status'] = 'error'
    rows.append(row)

df = pd.DataFrame(rows)
ok = df[df['status'] == 'ok']
n_total = len(df)
n_ok = len(ok)
fail_rate = 1 - n_ok / n_total

# 산포: 격리 추정치의 IQR/중앙값, 연합 대비 상대오차 분포
rel_err = {p: (ok[p] - fed[p]).abs() / fed[p] for p in ('c', 'k', 'sigma')}
summary = {
    'n_districts': n_total, 'min_n_threshold': MIN_N,
    'isolated_fit': {
        'ok': n_ok, 'too_small': int((df['status'] == 'too_small').sum()),
        'degenerate': int((df['status'] == 'degenerate').sum()),
        'error': int((df['status'] == 'error').sum()),
        'unusable_rate': round(fail_rate, 4)},
    'federated': {p: fed[p] for p in ('c', 'k', 'sigma')},
    'isolated_median': {p: float(ok[p].median()) for p in ('c', 'k', 'sigma')},
    'isolated_rel_err_vs_federated': {
        p: {'median': float(rel_err[p].median()), 'p90': float(rel_err[p].quantile(0.9)),
            'max': float(rel_err[p].max())} for p in ('c', 'k', 'sigma')},
}
print(json.dumps(summary, indent=2, ensure_ascii=False))

df.to_csv(RESULTS / f'e3_isolated_fits{SUFFIX}.csv', index=False, encoding='utf-8-sig')
(RESULTS / f'e3_benefit{SUFFIX}.json').write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

# ── 그림: 격리 추정 산포 vs 연합 단일점 ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, p, lab in zip(axes, ('c', 'k', 'sigma'), ('c (shape)', 'k (tail)', 'sigma (scale, days)')):
    vals = ok[p]
    lo, hi = vals.quantile(0.02), vals.quantile(0.98)
    ax.hist(vals[(vals >= lo) & (vals <= hi)], bins=40, color='#93c5fd', edgecolor='white')
    ax.axvline(fed[p], color='#dc2626', lw=2, label=f'federated={fed[p]:.3f}')
    ax.set_title(f'{lab}\nisolated per-district fits (n={n_ok})', fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle('Isolated district-only Burr fits vs Federated estimate', fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / f'e3_isolated_vs_federated{SUFFIX}.png', dpi=130)
print(f'saved: {RESULTS / f"e3_isolated_vs_federated{SUFFIX}.png"}')
print('=== E3 DONE ===')
