# -*- coding: utf-8 -*-
"""
[E0] ABM 실행 → 시군구별 감염쌍(GI pair) 데이터 생성
=====================================================
패치된 abm_simulation_rev31_timescale.py([FL-EXT] 컬럼 확장·시드 고정)를 로드해
한국 보정 시나리오 K_target을 실행하고, 결과 sim_gen_interval.csv에
child_sigungu/child_age 컬럼이 있는지 검증한다.

실행: python experiments/run_abm_district_pairs.py  (abm_fl_simulation/ 기준 상대경로)
출력: results/abm_run_manifest.json + abm_core/outputs_simplified_flext_*/
"""
from __future__ import annotations
import sys, io, os, json, time, importlib.util
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HERE = Path(__file__).resolve().parent          # experiments/
ROOT = HERE.parent                              # abm_fl_simulation/
CORE = ROOT / 'abm_core'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

os.chdir(CORE)                                  # ABM은 ./inputs 등 상대경로 가정

import pandas as pd

DISEASE = 'K_target'          # 논문 F10 한국 보정 시나리오 (lognormal 잠복기)
DAYS = 20                     # 포화 완화를 위해 30일 대신 20일
SEED = 42

# ── 패치된 ABM 로드 ────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    'abm_flext', CORE / 'abm_simulation_rev31_timescale.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ── 요율 파일 생성 (시나리오 1회) ──────────────────────────────────
scenario_dir = f'./inputs/scenarios/{DISEASE}'
os.makedirs(scenario_dir, exist_ok=True)
print(f'[STEP 1] rate files for {DISEASE}...', flush=True)
gen = mod.generate_scenario_rates_for_disease(
    disease_name=DISEASE, base_input_dir='./inputs',
    scenario_output_dir=scenario_dir)

# ── 시뮬레이션 실행 ────────────────────────────────────────────────
print(f'[STEP 2] run {DAYS} days, seed={SEED}...', flush=True)
t0 = time.time()
result_dir = mod.run_simulation(
    policy_name=f'flext_{DISEASE}',
    policy_file='./inputs/policy.xml',
    disease_name=DISEASE,
    simulation_days=DAYS,
    force_gpu=True,
    severity_rates_file=gen.get('severity'),
    fatality_rates_file=gen.get('fatality'),
    travel_config_file='travel_config.xml',
    extra_config={'rng_seed': SEED},
)
elapsed = time.time() - t0
print(f'  elapsed: {elapsed:.1f}s, result_dir: {result_dir}', flush=True)
if result_dir is None:
    print('FAIL: simulation returned None'); sys.exit(1)

# ── 산출 검증 ──────────────────────────────────────────────────────
gi_csv = os.path.join(result_dir, 'sim_gen_interval.csv')
df = pd.read_csv(gi_csv, encoding='utf-8-sig')
need = ['child_id', 'parent_id', 'T_days', 'child_sigungu', 'parent_sigungu',
        'child_age', 'child_exposure_day']
missing = [c for c in need if c not in df.columns]
if missing:
    print(f'FAIL: missing columns {missing}'); sys.exit(1)

valid = df.dropna(subset=['T_days'])
pos = valid[valid['T_days'] > 0]
n_district = pos['child_sigungu'].nunique()
per_d = pos.groupby('child_sigungu').size()
print(f'[STEP 3] pairs={len(df)}, valid T={len(valid)}, T>0={len(pos)}')
print(f'  districts={n_district}, per-district n: median={per_d.median():.0f}, '
      f'min={per_d.min()}, max={per_d.max()}')
print(f'  T>0 stats: mean={pos["T_days"].mean():.3f}, median={pos["T_days"].median():.2f}, '
      f'p99={pos["T_days"].quantile(0.99):.2f}, max={pos["T_days"].max():.2f}')

manifest = {
    'disease': DISEASE, 'days': DAYS, 'seed': SEED, 'elapsed_s': round(elapsed, 1),
    'result_dir': str(result_dir), 'gi_csv': gi_csv,
    'n_pairs': int(len(df)), 'n_valid_T': int(len(valid)), 'n_T_pos': int(len(pos)),
    'n_districts': int(n_district),
    'per_district_n': {'median': float(per_d.median()), 'min': int(per_d.min()), 'max': int(per_d.max())},
}
with open(RESULTS / 'abm_run_manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f'[OK] manifest -> {RESULTS / "abm_run_manifest.json"}')
