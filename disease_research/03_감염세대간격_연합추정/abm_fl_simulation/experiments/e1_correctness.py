# -*- coding: utf-8 -*-
"""
[E1] 정합성: 연합 추정 == 중앙집중 풀링 적합
=============================================
같은 ABM 데이터에서
  (a) 연합: 244개 시군구가 (n, ℓ, ∇ℓ)만 전송 → 중앙 L-BFGS
  (b) 풀링: 모든 감염쌍을 한곳에 모아 scipy.stats.burr12.fit
성공 기준: 표본당 NLL 차이 < 1e-6, 모수 상대차 < 0.5%
"""
from __future__ import annotations
import sys, io, json
from pathlib import Path
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl_pipeline.federated_burr import (
    load_district_pairs, build_hierarchy, federated_fit, pooled_reference)

RESULTS = ROOT / 'results'
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ''   # 예: '_nonsat'
manifest = json.loads((RESULTS / f'abm_run_manifest{SUFFIX}.json').read_text(encoding='utf-8'))

district_T = load_district_pairs(manifest['gi_csv'])
T_all = np.concatenate(list(district_T.values()))
districts, provinces = build_hierarchy(district_T)
print(f'districts={len(districts)}, provinces={len(provinces)}, N={len(T_all)}')

(cf, kf, sf), nll_f, rounds, res = federated_fit(
    districts, T_hint_median=float(np.median(T_all)))
(cp_, kp, sp), nll_p = pooled_reference(T_all)

dn = abs(nll_f - nll_p) / len(T_all)
rel = [abs(a - b) / b for a, b in ((cf, cp_), (kf, kp), (sf, sp))]
print(f'federated: c={cf:.5f} k={kf:.5f} s={sf:.5f}  NLL={nll_f:.3f}  rounds={rounds}')
print(f'pooled   : c={cp_:.5f} k={kp:.5f} s={sp:.5f}  NLL={nll_p:.3f}')
print(f'per-sample NLL diff = {dn:.3e}, param rel diff = '
      f'c {rel[0]:.2%}, k {rel[1]:.2%}, s {rel[2]:.2%}')

ok = dn < 1e-6 and max(rel) < 0.005
out = {
    'N': int(len(T_all)), 'n_districts': len(districts), 'n_provinces': len(provinces),
    'federated': {'c': cf, 'k': kf, 'sigma': sf, 'nll': nll_f, 'rounds': rounds},
    'pooled': {'c': cp_, 'k': kp, 'sigma': sp, 'nll': nll_p},
    'per_sample_nll_diff': dn, 'param_rel_diff': {'c': rel[0], 'k': rel[1], 'sigma': rel[2]},
    'bytes_per_round_per_district': 4 * 8,   # (n은 최초 1회) ll+grad3 = 4 float64
    'pass': bool(ok),
}
(RESULTS / f'e1_correctness{SUFFIX}.json').write_text(
    json.dumps(out, indent=2), encoding='utf-8')
print('=== E1 PASS ===' if ok else '=== E1 FAIL ===')
sys.exit(0 if ok else 1)
