# -*- coding: utf-8 -*-
"""
새 규모 합성데이터 생성 — 외부 팀 보고 규모(8시도·21시군구·~51k행) 재현
====================================================================
data_gen_hierarchical.py 와 동일한 방법론(지수성장+주간패턴+선택편향+진단지연).
구조: 서울(처치) 3구 + 대조 7개 시도 18구 = 8시도·21시군구.
※ 외부 팀의 '실제' 데이터가 아니라 동일 규모의 '합성' 재현임.
저장: fcausal_HIERARCHICAL/fcausal_hierarchical/data/districts21/
"""
import numpy as np, pandas as pd, os, random
from datetime import datetime, timedelta

SEED = 42
SCALE = 1.08          # 총행수를 ~51,305에 맞추기 위한 base_cases 배율
TRUE_SEVERE_RATE = 0.02
START = datetime(2024, 1, 1); DAYS = 100
OUT = "fcausal_HIERARCHICAL/fcausal_hierarchical/data/districts21"

# 8시도 · 21시군구 (처치=서울 3구, 대조=7개 시도 18구) | (시군구, base_cases, obs_ratio)
HIER = {
    "Seoul":   ("서울", True,  [("gangnam",22,20),("seocho",20,22),("jongno",14,18)]),
    "Gyeonggi":("경기", False, [("suwon",18,50),("seongnam",16,48),("yongin",15,52)]),
    "Busan":   ("부산", False, [("haeundae",14,30),("saha",12,28),("dongnae",10,32)]),
    "Incheon": ("인천", False, [("bupyeong",16,40),("namdong",15,38),("yeonsu",13,42)]),
    "Daegu":   ("대구", False, [("suseong",15,35),("dalseo",14,33),("buk",12,37)]),
    "Daejeon": ("대전", False, [("yuseong",14,45),("seo",12,43)]),
    "Gwangju": ("광주", False, [("gwangsan",13,36),("bukgu",11,34)]),
    "Ulsan":   ("울산", False, [("nam",12,31),("dong",10,33)]),
}


def gen_district(city_kr, district_kr, base_cases, obs_ratio, treated, policy_day):
    rng_local = np.random
    p_detect_mild = min(obs_ratio * (TRUE_SEVERE_RATE / (1 - TRUE_SEVERE_RATE)), 1.0)
    records, gid, recent = [], 1, []
    for t in range(DAYS):
        cur = START + timedelta(days=t)
        lam = base_cases * SCALE * np.exp(0.015 * t) * (1 + 0.15 * np.sin(2 * np.pi * t / 7))
        if policy_day and t >= policy_day:   # 처치(서울) 정책효과: 50% 억제(1주 램프업)
            ramp = min(1.0, (t - policy_day) / 7.0); lam *= 1.0 - 0.5 * ramp
        n_true = int(np.random.poisson(max(lam, 0)))
        for _ in range(n_true):
            severe = np.random.random() < TRUE_SEVERE_RATE
            if not severe and np.random.random() >= p_detect_mild:
                continue                      # 경증 누락(선택편향)
            prec = random.choice(recent) if recent and np.random.random() < 0.3 else np.nan
            rep = cur + timedelta(days=int(np.random.randint(1, 6)))
            death = 1 if (severe and np.random.random() < 0.2) else 0
            sev_rec = 1 if (severe and not death) else 0
            records.append({
                "id": gid, "시도": city_kr, "시군구": district_kr,
                "거주지시도": f"{city_kr} {district_kr}",
                "발병일": cur.strftime("%Y-%m-%d"), "진단일": rep.strftime("%Y-%m-%d"),
                "위중증 유무(유:1)": sev_rec, "사망유무(유:1)": death,
                "선행확진자_번호": prec, "성별": random.choice(["남", "여"]),
                "연령": int(np.random.randint(10, 90)),
            })
            recent.append(gid); gid += 1
        recent = recent[-300:]
    return pd.DataFrame(records)


def main():
    np.random.seed(SEED); random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    total, n_dist, n_city = 0, 0, 0
    summary = []
    for city_eng, (city_kr, treated, dists) in HIER.items():
        n_city += 1
        policy_day = 30 if treated else None
        for d_eng, base, obs in dists:
            df = gen_district(city_kr, d_eng, base, obs, treated, policy_day)
            fn = f"{OUT}/{city_eng.lower()}_{d_eng}.xlsx"
            df.to_excel(fn, index=False)
            total += len(df); n_dist += 1
            summary.append((city_kr, d_eng, len(df), "처치" if treated else "대조"))
    print(f"생성 완료: {n_city}개 시도 · {n_dist}개 시군구 · 총 {total:,}행  → {OUT}/")
    treat_d = sum(1 for s in summary if s[3] == "처치")
    print(f"   처치 {treat_d}구 / 대조 {n_dist - treat_d}구")
    print(f"   처치 행수 {sum(s[2] for s in summary if s[3]=='처치'):,} / 대조 행수 {sum(s[2] for s in summary if s[3]=='대조'):,}")


if __name__ == "__main__":
    main()
