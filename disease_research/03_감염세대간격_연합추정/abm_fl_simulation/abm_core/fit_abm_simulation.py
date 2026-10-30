"""ABM 시뮬레이션의 sim_gen_interval.csv에 5모형 적합 (한국 데이터 분석과 동일 절차).

사용: python fit_abm_simulation.py <input.csv> [output_dir]
"""
from __future__ import annotations
import sys
import io
import math
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

if len(sys.argv) < 2:
    print('Usage: python fit_abm_simulation.py <sim_gen_interval.csv> [out_dir]')
    sys.exit(1)

INPUT  = Path(sys.argv[1])
OUTDIR = Path(sys.argv[2]) if len(sys.argv) >= 3 else INPUT.parent
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---- Load ----
print(f'\n=== Load: {INPUT}', flush=True)
df = pd.read_csv(INPUT, encoding='utf-8-sig')
print(f'  total rows: {len(df)}')
T_all = pd.to_numeric(df['T_days'], errors='coerce')
T = T_all[(T_all.notna()) & (T_all > 0)].to_numpy(dtype=float)
print(f'  T>0 entries: {len(T)}')
if len(T) < 100:
    print('  WARN: 표본 너무 작음. 적합 결과 불안정.')
    sys.exit(1)

print(f'  T stats: mean={T.mean():.3f}, median={np.median(T):.2f}, '
      f'min={T.min()}, max={T.max()}, p99={np.percentile(T,99):.2f}')

n = len(T)
def aic(ll, k): return 2*k - 2*ll
def bic(ll, k): return k*math.log(n) - 2*ll


# ---- Fit ----
print(f'\n=== Fit 5 models (n={n}) ===', flush=True)
MODELS = [
    ('Exponential', stats.expon),
    ('Weibull',     stats.weibull_min),
    ('Lognormal',   stats.lognorm),
    ('Pareto',      stats.pareto),
    ('BurrXII',     stats.burr12),
]
results = []
for name, dist in MODELS:
    try:
        params = dist.fit(T, floc=0)
        ll = float(np.sum(dist.logpdf(T, *params)))
        k_free = len(params) - 1
        ks_stat, _ = stats.kstest(T, lambda x: dist.cdf(x, *params))
        results.append({
            'model': name, 'params': params, 'll': ll,
            'AIC': aic(ll, k_free), 'BIC': bic(ll, k_free), 'KS': ks_stat,
        })
        print(f'  {name:11s} ll={ll:11.1f}  AIC={aic(ll,k_free):11.1f}  KS={ks_stat:.4f}  params={[f"{x:.3f}" for x in params]}')
    except Exception as e:
        print(f'  {name}: FAIL ({e})')

# Rank
sorted_by_ks  = sorted(results, key=lambda r: r['KS'])
sorted_by_aic = sorted(results, key=lambda r: r['AIC'])
print(f'\nRank by KS:  {[r["model"] for r in sorted_by_ks]}')
print(f'Rank by AIC: {[r["model"] for r in sorted_by_aic]}')


# ---- Save summary ----
summary_path = OUTDIR / 'fit_abm_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f'Input: {INPUT}\n')
    f.write(f'n: {n}\n')
    f.write(f'T: mean={T.mean():.3f}, sd={T.std():.3f}, min={T.min()}, max={T.max()}\n\n')
    f.write('model       | log-lik     | AIC         | BIC         | KS     | params\n')
    f.write('-' * 100 + '\n')
    for r in sorted(results, key=lambda r: r['KS']):
        ps = ', '.join(f'{x:.4f}' for x in r['params'])
        f.write(f'{r["model"]:11s} | {r["ll"]:11.1f} | {r["AIC"]:11.1f} | {r["BIC"]:11.1f} | {r["KS"]:.4f} | {ps}\n')
print(f'\nSummary -> {summary_path}')


# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
xs_lin = np.arange(1, max(60, int(T.max()) + 5))
xs_log = np.logspace(np.log10(max(1, T.min())), np.log10(T.max()), 200)

# (a) linear PDF
bins = np.arange(0.5, min(60, T.max()+1) + 1)
axes[0].hist(T, bins=bins, density=True, alpha=0.5, color='#888', label='data')
for r in results:
    dist = dict(MODELS)[r['model']]
    axes[0].plot(xs_lin, dist.pdf(xs_lin, *r['params']), label=r['model'])
axes[0].set_xlim(0, min(60, T.max()+2)); axes[0].set_title(f'PDF (linear, first {min(60,int(T.max())+2)}d)')
axes[0].set_xlabel('T (days)'); axes[0].set_ylabel('density'); axes[0].legend(fontsize=8)

# (b) log-y
axes[1].hist(T, bins=np.arange(0.5, T.max()+1.5), density=True, alpha=0.5, color='#888')
for r in results:
    dist = dict(MODELS)[r['model']]
    axes[1].plot(xs_log, dist.pdf(xs_log, *r['params']), label=r['model'])
axes[1].set_yscale('log'); axes[1].set_title('PDF (log-y)'); axes[1].set_xlabel('T (days)')
axes[1].legend(fontsize=8)

# (c) log-log survival
sorted_T = np.sort(T); surv = 1.0 - np.arange(len(sorted_T)) / len(sorted_T)
axes[2].loglog(sorted_T, surv, '.', color='#888', label='empirical S(T)')
for r in results:
    dist = dict(MODELS)[r['model']]
    axes[2].plot(xs_log, dist.sf(xs_log, *r['params']), label=r['model'])
axes[2].set_title('Survival S(T) (log-log)'); axes[2].set_xlabel('T'); axes[2].set_ylabel('S(T)')
axes[2].legend(fontsize=8)

fig.suptitle(f'ABM sim fit (n={n}, mean T={T.mean():.2f}d)')
fig.tight_layout()
png_path = OUTDIR / 'fit_abm_plot.png'
fig.savefig(png_path, dpi=120)
plt.close(fig)
print(f'Plot -> {png_path}')

print('\n=== DONE ===')
