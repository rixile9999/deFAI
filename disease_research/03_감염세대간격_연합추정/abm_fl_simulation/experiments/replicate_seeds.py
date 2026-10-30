# -*- coding: utf-8 -*-
"""
[E5] 비포화 다회 seed 반복 — "한국 시그니처 재현"의 통계적 안정성 검증
=======================================================================
보정 확정 조건(β=0.000618, T_OFF, 120일, 시드 5,000명)을 seed만 바꿔 반복하고,
풀링 Burr 모수(=E1이 증명한 대로 연합 추정과 동일)의 seed 간 산포를 정량화한다.
출력: results/seed_replication.json
"""
from __future__ import annotations
import sys, io, os, json, time, copy, importlib.util
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE = ROOT / 'abm_core'
RESULTS = ROOT / 'results'
os.chdir(CORE)

import numpy as np
import pandas as pd
from scipy import stats

BETA_NAME = 'K_ns_b0p000618'   # calibrate_nonsat.py가 주입한 확정 시나리오
DAYS = 120
SEEDS_INIT = 5000
RNG_SEEDS = [7, 101, 202, 303, 404]   # seed 42(기준)는 manifest에서 가져옴

spec = importlib.util.spec_from_file_location(
    'abm_flext', CORE / 'abm_simulation_rev31_timescale.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# [T_OFF] — calibrate_nonsat.py와 동일
_orig_get_activity = mod.Scheduler.get_activity
def _ga_toff(self, agent_type, step):
    a = _orig_get_activity(self, agent_type, step)
    return 'home' if a == 'transport' else a
mod.Scheduler.get_activity = _ga_toff

# 파싱 캐시 — calibrate_nonsat.py와 동일
_raw_cache = {}
_orig_load = mod.load_and_prepare_data
def _cached_load(*args, **kwargs):
    key = json.dumps([str(a) for a in args]) + json.dumps(
        {k: str(v) for k, v in kwargs.items()}, sort_keys=True)
    if key not in _raw_cache:
        _raw_cache[key] = _orig_load(*args, **kwargs)
    raw = _raw_cache[key]
    return {k: (v.copy() if hasattr(v, 'copy') and not isinstance(v, (dict, list))
                else copy.deepcopy(v)) for k, v in raw.items()}
mod.load_and_prepare_data = _cached_load

sdir = f'./inputs/scenarios/{BETA_NAME}'
os.makedirs(sdir, exist_ok=True)
gen = mod.generate_scenario_rates_for_disease(
    disease_name=BETA_NAME, base_input_dir='./inputs', scenario_output_dir=sdir)

rows = []
# 기준 seed 42는 기존 manifest에서
m42 = json.loads((RESULTS / 'abm_run_manifest_nonsat.json').read_text(encoding='utf-8'))
e1_42 = json.loads((RESULTS / 'e1_correctness_nonsat.json').read_text(encoding='utf-8'))
rows.append({'seed': 42, 'attack': m42['attack_rate'], 'n_pairs': m42['n_T_pos'],
             'c': e1_42['pooled']['c'], 'k': e1_42['pooled']['k'],
             'sigma': e1_42['pooled']['sigma'], 'source': 'baseline'})
print(f"seed 42 (기준): c={rows[0]['c']:.4f} k={rows[0]['k']:.4f} s={rows[0]['sigma']:.4f} "
      f"attack={rows[0]['attack']:.4%}")

for sd in RNG_SEEDS:
    t0 = time.time()
    rd = mod.run_simulation(
        policy_name=f'rep_s{sd}', policy_file='./inputs/policy.xml',
        disease_name=BETA_NAME, simulation_days=DAYS, force_gpu=True,
        severity_rates_file=gen.get('severity'),
        fatality_rates_file=gen.get('fatality'),
        travel_config_file='travel_config.xml',
        extra_config={'rng_seed': sd, 'initial_infections_file': None,
                      'initial_infected_count': SEEDS_INIT})
    if rd is None:
        print(f'seed {sd}: run failed'); continue
    df = pd.read_csv(os.path.join(rd, 'sim_gen_interval.csv'), encoding='utf-8-sig')
    T = df.dropna(subset=['T_days'])
    T = T[T['T_days'] > 0]['T_days'].to_numpy(dtype=np.float64)
    logs = list(Path(rd).glob('simulation_log_*.csv'))
    sl = pd.read_csv(logs[0])
    S0 = sl['S'].iloc[0] + SEEDS_INIT
    attack = 1.0 - sl.groupby('day')['S'].last().iloc[-1] / S0
    c, k, _, s = stats.burr12.fit(T, floc=0)
    rows.append({'seed': sd, 'attack': float(attack), 'n_pairs': int(len(T)),
                 'c': float(c), 'k': float(k), 'sigma': float(s), 'source': 'replicate'})
    print(f'seed {sd}: c={c:.4f} k={k:.4f} s={s:.4f} attack={attack:.4%} '
          f'pairs={len(T)} ({time.time()-t0:.0f}s)')

arr = {p: np.array([r[p] for r in rows]) for p in ('c', 'k', 'sigma', 'attack')}
KOREA = {'c': 2.2210, 'k': 1.0083, 'sigma': 3.7701}   # onset clock 정본
summary = {
    'n_runs': len(rows), 'beta': 0.000618, 'days': DAYS, 'transport': 'T_OFF',
    'mean': {p: float(arr[p].mean()) for p in arr},
    'sd': {p: float(arr[p].std(ddof=1)) for p in arr},
    'korea_onset': KOREA,
    'korea_within_2sd': {p: bool(abs(arr[p].mean() - KOREA[p]) <= 2 * arr[p].std(ddof=1))
                         for p in ('c', 'k', 'sigma')},
    'runs': rows,
}
(RESULTS / 'seed_replication.json').write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('\n=== SUMMARY ===')
for p in ('c', 'k', 'sigma'):
    print(f'{p}: {arr[p].mean():.4f} ± {arr[p].std(ddof=1):.4f}  '
          f'(한국 onset {KOREA[p]}, |편차|/sd={abs(arr[p].mean()-KOREA[p])/max(arr[p].std(ddof=1),1e-12):.1f})')
print(f'attack: {arr["attack"].mean():.4%} ± {arr["attack"].std(ddof=1):.4%}')
print('saved -> results/seed_replication.json')
