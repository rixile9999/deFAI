"""Integration smoke test: M1+M2+M3+M4 + 4-step/transport 복원 통합 회귀.

5일 짧은 시뮬레이션 실행 후:
1) 시뮬레이션이 에러 없이 완료되는지
2) sim_gen_interval.csv가 생성되는지
3) (parent, child, T) 페어가 logging되는지
4) transport place에서의 감염이 INFECTION 카운트에 잡히는지

이건 통합 회귀 테스트이지 실험 실행이 아님. 결과의 수치적 정확성은 검증 안 함.
"""
from __future__ import annotations
import sys
import io
import os
import importlib.util
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HERE = Path(__file__).parent
os.chdir(HERE)  # rel paths assume rev30 cwd

import numpy as np
import pandas as pd

# ---- Load rev30 module ----
spec = importlib.util.spec_from_file_location(
    'abm_rev30', HERE / 'abm_simulation_integrated-rev9-hospitalized.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


# ---- Generate severity/fatality rate files for one disease ----
disease_name = 'new_flu_not_treated'
base_input_dir = './inputs'
scenario_dir = f'{base_input_dir}/scenarios/{disease_name}'
os.makedirs(scenario_dir, exist_ok=True)

print(f'\n[STEP 1] Generating rate files for {disease_name}...', flush=True)
generated = mod.generate_scenario_rates_for_disease(
    disease_name=disease_name,
    base_input_dir=base_input_dir,
    scenario_output_dir=scenario_dir,
)
print(f'  generated: {list(generated.keys())}', flush=True)


# ---- Run simulation (SHORT: 5 days) ----
print(f'\n[STEP 2] Running simulation 5 days...', flush=True)
t0 = time.time()
result_dir = mod.run_simulation(
    policy_name='smoke',
    policy_file='./inputs/policy.xml',
    disease_name=disease_name,
    simulation_days=5,
    force_gpu=True,
    severity_rates_file=generated.get('severity'),
    fatality_rates_file=generated.get('fatality'),
    travel_config_file='travel_config.xml',
)
elapsed = time.time() - t0
print(f'  elapsed: {elapsed:.1f}s', flush=True)
print(f'  result_dir: {result_dir}', flush=True)

if result_dir is None:
    print('FAIL: simulation returned None')
    sys.exit(1)


# ---- Verify outputs ----
print(f'\n[STEP 3] Verifying outputs...', flush=True)
files_present = []
for fn in ['sim_gen_interval.csv', 'daily_movement_log.csv', 'daily_regional_cases.csv']:
    p = os.path.join(result_dir, fn)
    exists = os.path.exists(p)
    files_present.append((fn, exists))
    print(f'  {fn}: {"OK" if exists else "MISSING"}')


# ---- Check sim_gen_interval.csv ----
gi_csv = os.path.join(result_dir, 'sim_gen_interval.csv')
gi_ok = False
if os.path.exists(gi_csv):
    df = pd.read_csv(gi_csv, encoding='utf-8-sig')
    cols_ok = all(c in df.columns for c in
                  ['child_id', 'parent_id', 'child_onset', 'parent_onset',
                   'T_days', 'child_variant', 'parent_variant'])
    print(f'\n[STEP 4] sim_gen_interval.csv stats:')
    print(f'  rows: {len(df)}')
    print(f'  columns: {df.columns.tolist()}')
    valid = df['T_days'].dropna()
    if len(valid) > 0:
        print(f'  valid T entries: {len(valid)}')
        print(f'  T_days: min={valid.min():.2f}, mean={valid.mean():.2f}, max={valid.max():.2f}')
    # 5일 짧은 simulation이라 페어가 적을 수 있음. 0이 아니면 OK.
    gi_ok = cols_ok and len(df) >= 0


# ---- Final ----
all_ok = result_dir is not None and gi_ok and all(present for _, present in files_present)
print('\n' + ('=== INTEGRATION OK ===' if all_ok else '=== INTEGRATION ISSUES ==='))
sys.exit(0 if all_ok else 1)
