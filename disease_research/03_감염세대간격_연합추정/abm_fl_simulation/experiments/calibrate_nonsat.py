# -*- coding: utf-8 -*-
"""
[E4] 비포화 레짐 재보정 (논문 §5.6 처방 이행)
==============================================
처방: ① transport 완전혼합 인공물 제거(T_OFF — 논문 diag_transport_artifact.py와 동일 패치)
      ② β 재보정 → 성장률 r ∈ [0.02, 0.035]/일 (R_eff ≈ 1.1)
      ③ 장기(120일) 비포화 실행

방법: K_target 클론(β만 스케일) 주입 → 5,000명 무작위 시드 → 30일 프로브로
      r(일별 신규노출 로그선형 기울기)·발병률 측정 → log β 이분탐색 → 120일 본 실행.
출력: results/calibration_log.json, results/abm_run_manifest_nonsat.json
"""
from __future__ import annotations
import sys, io, os, json, time, copy, importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE = ROOT / 'abm_core'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
os.chdir(CORE)

import numpy as np
import pandas as pd

BASE_DISEASE = 'K_target'
BASE_BETA = 0.05
SEEDS = 5000
FINAL_DAYS = 120
RNG_SEED = 42
# [v2] 보정 타깃 변경: 성장률 r은 시드 대량투입·백신캠페인·격리 때문에 β에 비단조
#      (1차 시도에서 실증) → 120일 '발병률'을 직접 이분탐색 (프로브에서 β에 단조 확인됨)
ATTACK_TARGET = (0.005, 0.05)   # 120일 발병률 0.5%~5% = 비포화 (S 고갈 없음)
MIN_PAIRS = 20000               # Burr 적합 최소 표본
MAX_PROBES = 7

# ── 패치된 ABM 로드 ────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    'abm_flext', CORE / 'abm_simulation_rev31_timescale.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ── [T_OFF] transport 완전혼합 제거 (논문 진단과 동일 패치) ────────
_orig_get_activity = mod.Scheduler.get_activity
def _ga_toff(self, agent_type, step):
    a = _orig_get_activity(self, agent_type, step)
    return 'home' if a == 'transport' else a
mod.Scheduler.get_activity = _ga_toff
print('[T_OFF] step1 transport → home (완전혼합 인공물 제거)')

# ── 905MB 파싱 캐시 (프로브마다 재파싱 방지; 배열은 복사 반환) ─────
_raw_cache = {}
_orig_load = mod.load_and_prepare_data
def _cached_load(*args, **kwargs):
    key = json.dumps([str(a) for a in args]) + json.dumps(
        {k: str(v) for k, v in kwargs.items()}, sort_keys=True)
    if key not in _raw_cache:
        _raw_cache[key] = _orig_load(*args, **kwargs)
    raw = _raw_cache[key]
    out = {}
    for k, v in raw.items():
        out[k] = v.copy() if hasattr(v, 'copy') and not isinstance(v, (dict, list)) \
                 else copy.deepcopy(v)
    return out
mod.load_and_prepare_data = _cached_load

# ── β-스케일 시나리오 주입 (기존 run_* 드라이버 패턴, 멱등) ────────
DP_PATH = CORE / 'inputs' / 'disease_params.xml'
def ensure_scenario(beta):
    name = f'K_ns_b{beta:.6f}'.replace('.', 'p')
    tree = ET.parse(DP_PATH)
    root = tree.getroot()
    if any(d.get('name') == name for d in root.findall('Disease')):
        return name
    base = next(d for d in root.findall('Disease') if d.get('name') == BASE_DISEASE)
    clone = copy.deepcopy(base)
    clone.set('name', name)
    clone.find('TransmissionRate').text = f'{beta:.8f}'
    root.append(clone)
    tree.write(DP_PATH, encoding='utf-8', xml_declaration=True)
    return name

def run_once(beta, days, tag):
    name = ensure_scenario(beta)
    sdir = f'./inputs/scenarios/{name}'
    os.makedirs(sdir, exist_ok=True)
    gen = mod.generate_scenario_rates_for_disease(
        disease_name=name, base_input_dir='./inputs', scenario_output_dir=sdir)
    t0 = time.time()
    rd = mod.run_simulation(
        policy_name=f'ns_{tag}', policy_file='./inputs/policy.xml',
        disease_name=name, simulation_days=days, force_gpu=True,
        severity_rates_file=gen.get('severity'),
        fatality_rates_file=gen.get('fatality'),
        travel_config_file='travel_config.xml',
        extra_config={'rng_seed': RNG_SEED,
                      'initial_infections_file': None,
                      'initial_infected_count': SEEDS})
    return rd, time.time() - t0

def measure(result_dir, days):
    """simulation_log → 일별 New_E 합 → 성장률 r(로그선형, day 8..days-2), 발병률."""
    logs = list(Path(result_dir).glob('simulation_log_*.csv'))
    df = pd.read_csv(logs[0])
    daily = df.groupby('day').agg(newE=('New_E', 'sum'), S=('S', 'last')).reset_index()
    S0 = df['S'].iloc[0] + SEEDS
    attack = 1.0 - daily['S'].iloc[-1] / S0
    sub = daily[(daily['day'] >= 8) & (daily['day'] <= days - 2) & (daily['newE'] > 0)]
    if len(sub) < 5:
        return None, attack, daily
    y = np.log(sub['newE'].rolling(3, center=True, min_periods=1).mean())
    r = float(np.polyfit(sub['day'], y, 1)[0])
    return r, float(attack), daily

# ── 이분탐색: 120일 발병률 직접 타깃 ───────────────────────────────
def count_pairs(result_dir):
    df = pd.read_csv(os.path.join(result_dir, 'sim_gen_interval.csv'), encoding='utf-8-sig')
    pos = df.dropna(subset=['T_days'])
    return len(pos[pos['T_days'] > 0])

log = []
lo, hi = np.log(0.0002), np.log(0.0009)  # 1차 시도 프로브에서 도출한 브래킷
beta = float(np.exp((lo + hi) / 2))
chosen, chosen_rd, chosen_el = None, None, None
for i in range(1, MAX_PROBES + 1):
    print(f'\n[PROBE {i}] beta={beta:.6f} ({BASE_BETA/beta:.0f}x down), {FINAL_DAYS}d...')
    rd, el = run_once(beta, FINAL_DAYS, f'p{i}')
    if rd is None:
        print('  run failed'); sys.exit(1)
    r, attack, _ = measure(rd, FINAL_DAYS)
    npairs = count_pairs(rd)
    print(f'  attack={attack:.4%}, pairs={npairs}, r(참고)={r if r is None else round(r,4)}, {el:.0f}s')
    log.append({'probe': i, 'beta': beta, 'attack': attack, 'n_pairs': npairs,
                'r': r, 'elapsed_s': round(el, 1)})
    if ATTACK_TARGET[0] <= attack <= ATTACK_TARGET[1] and npairs >= MIN_PAIRS:
        chosen, chosen_rd, chosen_el = beta, rd, el
        print(f'  ✅ attack in {ATTACK_TARGET}, pairs>={MIN_PAIRS}')
        break
    if attack > ATTACK_TARGET[1]:
        hi = np.log(beta)          # 너무 많음 → β 내림
    else:
        lo = np.log(beta)          # 너무 적음(발병률 미달 또는 페어 부족) → β 올림
    beta = float(np.exp((lo + hi) / 2))

if chosen is None:
    print(f'\n❌ {MAX_PROBES}회 내 미수렴 — calibration_log.json 확인 후 브래킷 조정 필요')
    (RESULTS / 'calibration_log.json').write_text(
        json.dumps({'probes': log, 'converged': False}, indent=2), encoding='utf-8')
    sys.exit(1)

# ── 채택 실행이 곧 본 실행 (동일 120일) ────────────────────────────
rd, el = chosen_rd, chosen_el
r_fin, attack_fin, daily = measure(rd, FINAL_DAYS)
print(f'\n[FINAL=PROBE {len(log)}] beta={chosen:.6f}, attack={attack_fin:.4%}')

gi_csv = os.path.join(rd, 'sim_gen_interval.csv')
df = pd.read_csv(gi_csv, encoding='utf-8-sig')
pos = df.dropna(subset=['T_days'])
pos = pos[pos['T_days'] > 0]
n_district = pos['child_sigungu'].nunique()
per_d = pos.groupby('child_sigungu').size()
print(f'  pairs(T>0)={len(pos)}, districts={n_district}, per-district median={per_d.median():.0f}')

manifest = {
    'regime': 'non-saturated', 'transport': 'T_OFF',
    'disease_base': BASE_DISEASE, 'beta': chosen, 'beta_reduction_x': BASE_BETA / chosen,
    'days': FINAL_DAYS, 'seeds_initial': SEEDS, 'seed': RNG_SEED,
    'growth_rate_per_day': r_fin, 'attack_rate': attack_fin,
    'elapsed_s': round(el, 1), 'result_dir': str(rd), 'gi_csv': gi_csv,
    'n_T_pos': int(len(pos)), 'n_districts': int(n_district),
    'per_district_n': {'median': float(per_d.median()), 'min': int(per_d.min()),
                       'max': int(per_d.max())},
}
(RESULTS / 'abm_run_manifest_nonsat.json').write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
(RESULTS / 'calibration_log.json').write_text(
    json.dumps({'probes': log, 'chosen_beta': chosen, 'target_attack': ATTACK_TARGET,
                'converged': True,
                'final': {'r': r_fin, 'attack': attack_fin}}, indent=2), encoding='utf-8')
print(f'\n[OK] manifest -> abm_run_manifest_nonsat.json, calibration_log.json')
