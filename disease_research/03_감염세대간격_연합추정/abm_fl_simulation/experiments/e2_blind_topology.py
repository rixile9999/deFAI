# -*- coding: utf-8 -*-
"""
[E2] 3-tier blind aggregation: 시도(Tier 1)가 내용을 못 본 채 합산해도 결과 동일
==============================================================================
(a) 평문 계층 합산: 시군구 → 시도 부분합 → 중앙 총합 == 직접 총합 (부동소수 오차 수준)
(b) Paillier 동형 합산: 시군구가 공개키로 (ℓ, ∇ℓ) 암호화 → 시도는 암호문 곱셈(=평문 합)만
    → 중앙이 비밀키로 복호화 == 평문 총합 (고정소수점 오차 < 1e-6)
(c) 종단: 동형 합산 경로만으로 연합 L-BFGS 전 라운드를 돌려 E1 연합 추정치와 일치 확인
성공 기준: (a) 상대차 < 1e-12, (b) 절대차 < 1e-5, (c) 모수 상대차 < 1e-6
"""
from __future__ import annotations
import sys, io, json, random
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl_pipeline.federated_burr import load_district_pairs, build_hierarchy
from fl_pipeline import paillier

RESULTS = ROOT / 'results'
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ''   # 예: '_nonsat'
manifest = json.loads((RESULTS / f'abm_run_manifest{SUFFIX}.json').read_text(encoding='utf-8'))
district_T = load_district_pairs(manifest['gi_csv'])
districts, provinces = build_hierarchy(district_T)
e1 = json.loads((RESULTS / f'e1_correctness{SUFFIX}.json').read_text(encoding='utf-8'))

eta_test = np.log([1.5, 1.5, float(np.median(np.concatenate(list(district_T.values()))))])

# ── (a) 평문 계층 합산 == 직접 합산 ────────────────────────────────
ll_direct, g_direct = 0.0, np.zeros(3)
for d in districts:
    ll_d, g_d = d.stats(eta_test)
    ll_direct += ll_d; g_direct += g_d
ll_hier, g_hier = 0.0, np.zeros(3)
for p in provinces:
    ll_p, g_p = p.stats(eta_test)
    ll_hier += ll_p; g_hier += g_p
rel_a = abs(ll_hier - ll_direct) / abs(ll_direct)
print(f'(a) hierarchical vs direct: ll rel diff = {rel_a:.2e}')

# ── (b)+(c) Paillier blind 경로 ────────────────────────────────────
pub, sec = paillier.keygen(bits=96, seed=42)   # setup: 공개키→지자체, 비밀키→중앙
enc_rng = random.Random(7)

def blind_round(eta):
    """시군구 암호화 → 시도 blind 합(공개키만) → 중앙 복호화. 4값(ll,g1,g2,g3)."""
    province_ciphers = []
    for p in provinces:
        acc = None
        for d in p.districts:                     # Tier 2: 로컬 계산 후 암호화
            ll_d, g_d = d.stats(eta)
            cs = [paillier.encrypt(pub, v, enc_rng) for v in (ll_d, *g_d)]
            acc = cs if acc is None else [paillier.he_add(a, c, pub) for a, c in zip(acc, cs)]
        province_ciphers.append(acc)              # Tier 1은 복호화 없이 여기서 끝
    total = province_ciphers[0]
    for cs in province_ciphers[1:]:               # 중앙: 시도 암호문 최종 합산 후 복호화
        total = [paillier.he_add(a, c, pub) for a, c in zip(total, cs)]
    vals = [paillier.decrypt(sec, c) for c in total]
    return vals[0], np.array(vals[1:])

ll_he, g_he = blind_round(eta_test)
diff_b = max(abs(ll_he - ll_direct) / max(abs(ll_direct), 1),
             float(np.max(np.abs(g_he - g_direct) / np.maximum(np.abs(g_direct), 1))))
print(f'(b) Paillier blind vs direct: max rel diff = {diff_b:.2e}')

# ── (c) 동형 경로만으로 전체 연합 적합 ─────────────────────────────
rounds = {'count': 0}
def nll_he(eta):
    rounds['count'] += 1
    ll, g = blind_round(eta)
    return -ll, -g

res = minimize(nll_he, eta_test.copy(), jac=True, method='L-BFGS-B',  # E1과 동일 초기값(중앙값 힌트)
               bounds=[(np.log(0.05), np.log(60.0))] * 3,
               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
c_he, k_he, s_he = np.exp(res.x)
fed = e1['federated']
rel_c = [abs(a - b) / b for a, b in ((c_he, fed['c']), (k_he, fed['k']), (s_he, fed['sigma']))]
print(f'(c) HE-only fit: c={c_he:.5f} k={k_he:.5f} s={s_he:.5f} rounds={rounds["count"]}')
print(f'    vs E1 federated rel diff: c {rel_c[0]:.2e}, k {rel_c[1]:.2e}, s {rel_c[2]:.2e}')

ok = rel_a < 1e-12 and diff_b < 1e-5 and max(rel_c) < 1e-6
out = {
    'a_plain_hier_rel_diff': rel_a,
    'b_paillier_one_round_max_rel_diff': diff_b,
    'c_he_only_fit': {'c': float(c_he), 'k': float(k_he), 'sigma': float(s_he),
                      'rounds': rounds['count'], 'max_rel_diff_vs_e1': float(max(rel_c))},
    'key_bits': 96, 'tier1_has_secret_key': False,
    'pass': bool(ok),
}
(RESULTS / f'e2_blind_topology{SUFFIX}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print('=== E2 PASS ===' if ok else '=== E2 FAIL ===')
sys.exit(0 if ok else 1)
