import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_hierarchical_data():
    """
    계층형 구조를 위한 시군구 단위 데이터 생성
    
    구조:
    - 서울 (Treatment): 강남, 서초, 종로
    - 경기 (Control): 수원, 성남, 용인
    - 부산 (Control): 해운대, 사하, 동래
    """
    np.random.seed(42)
    random.seed(42)
    os.makedirs('data/districts', exist_ok=True)
    
    # 중증화율
    TRUE_SEVERE_RATE = 0.02
    
    # 계층 구조 정의
    HIERARCHY = {
        'Seoul': {
            'name_kr': '서울',
            'is_treated': True,
            'policy_day': 30,
            'policy_effect': 0.5,  # 50% 억제
            'districts': {
                'gangnam': {'name_kr': '강남구', 'base_cases': 20, 'obs_ratio': 20},
                'seocho': {'name_kr': '서초구', 'base_cases': 18, 'obs_ratio': 22},
                'jongno': {'name_kr': '종로구', 'base_cases': 12, 'obs_ratio': 18}
            }
        },
        'Gyeonggi': {
            'name_kr': '경기',
            'is_treated': False,
            'policy_day': None,
            'policy_effect': 1.0,
            'districts': {
                'suwon': {'name_kr': '수원시', 'base_cases': 15, 'obs_ratio': 50},
                'seongnam': {'name_kr': '성남시', 'base_cases': 13, 'obs_ratio': 48},
                'yongin': {'name_kr': '용인시', 'base_cases': 12, 'obs_ratio': 52}
            }
        },
        'Busan': {
            'name_kr': '부산',
            'is_treated': False,
            'policy_day': None,
            'policy_effect': 1.0,
            'districts': {
                'haeundae': {'name_kr': '해운대구', 'base_cases': 12, 'obs_ratio': 30},
                'saha': {'name_kr': '사하구', 'base_cases': 10, 'obs_ratio': 28},
                'dongnae': {'name_kr': '동래구', 'base_cases': 8, 'obs_ratio': 32}
            }
        }
    }
    
    START_DATE = datetime(2024, 1, 1)
    DAYS = 100

    print("🚀 Generating Hierarchical F-Causal Data (District-level)...")
    print(f"   Structure: Tier 0 (Central) ← Tier 1 (Province) ← Tier 2 (District)")
    print()

    for region_eng, region_info in HIERARCHY.items():
        region_kr = region_info['name_kr']
        is_treated = region_info['is_treated']
        policy_day = region_info['policy_day']
        policy_effect = region_info['policy_effect']
        
        print(f"📍 {region_kr} ({region_eng}) - {'Treatment' if is_treated else 'Control'}")
        
        for district_eng, district_info in region_info['districts'].items():
            records = []
            district_kr = district_info['name_kr']
            base_cases = district_info['base_cases']
            obs_ratio_target = district_info['obs_ratio']
            
            # Selection Bias 계산
            true_mild_rate = 1 - TRUE_SEVERE_RATE
            p_detect_mild = obs_ratio_target * (TRUE_SEVERE_RATE / true_mild_rate)
            p_detect_mild = min(p_detect_mild, 1.0)
            
            global_id = 1
            recent_ids = []
            
            for t in range(DAYS):
                curr_date = START_DATE + timedelta(days=t)
                
                # 지수적 성장 + 주간 패턴
                growth_trend = np.exp(0.015 * t)
                weekly_pattern = 1.0 + 0.15 * np.sin(2 * np.pi * t / 7)
                
                lambda_val = base_cases * growth_trend * weekly_pattern
                
                # 정책 효과 (서울만)
                if policy_day and t >= policy_day:
                    days_after = t - policy_day
                    ramp_up = min(1.0, days_after / 7.0)
                    current_effect = 1.0 - (1.0 - policy_effect) * ramp_up
                    lambda_val *= current_effect
                
                n_true = int(np.random.poisson(max(lambda_val, 0)))
                
                for _ in range(n_true):
                    is_severe_truth = np.random.random() < TRUE_SEVERE_RATE
                    
                    # Selection Bias
                    if is_severe_truth:
                        is_observed = True
                    else:
                        is_observed = np.random.random() < p_detect_mild
                    
                    if not is_observed:
                        continue
                    
                    # 네트워크 구조
                    prec_id = np.nan
                    if recent_ids and np.random.random() < 0.3:
                        prec_id = random.choice(recent_ids)
                    
                    # 진단 지연
                    delay = np.random.randint(1, 6)
                    report_date = curr_date + timedelta(days=delay)
                    
                    # 결과
                    is_death = 0
                    is_severe_rec = 0
                    if is_severe_truth:
                        if np.random.random() < 0.2:
                            is_death = 1
                        else:
                            is_severe_rec = 1
                    
                    records.append({
                        'id': global_id,
                        '시도': region_kr,
                        '시군구': district_kr,
                        '거주지시도': f"{region_kr} {district_kr}",  # 호환성
                        '발병일': curr_date.strftime('%Y-%m-%d'),
                        '진단일': report_date.strftime('%Y-%m-%d'),
                        '위중증 유무(유:1)': 1 if is_severe_rec else 0,
                        '사망유무(유:1)': 1 if is_death else 0,
                        '선행확진자_번호': prec_id,
                        '성별': random.choice(['남', '여']),
                        '연령': np.random.randint(10, 90)
                    })
                    
                    recent_ids.append(global_id)
                    global_id += 1
                
                recent_ids = recent_ids[-300:]
            
            # 시군구 파일 저장
            df = pd.DataFrame(records)
            filename = f"data/districts/{region_eng.lower()}_{district_eng}.xlsx"
            df.to_excel(filename, index=False)
            print(f"   ✅ {district_kr:8s} → {filename:40s} ({len(df):4d} cases)")
        
        print()
    
    print("=" * 70)
    print("✅ Hierarchical data generation complete!")
    print(f"   Files saved in: data/districts/")
    print(f"   Structure:")
    print(f"     - Seoul: gangnam, seocho, jongno (Treatment)")
    print(f"     - Gyeonggi: suwon, seongnam, yongin (Control)")
    print(f"     - Busan: haeundae, saha, dongnae (Control)")


if __name__ == "__main__":
    import argparse
    from core.policy_stats import add_analysis_arguments, config_from_args
    from core.synthetic_regions import RegionGeneration, generate_regions, export_regions

    parser = argparse.ArgumentParser(description="Legacy case lists or expanded synthetic daily panels")
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--output", default="data/expanded_regions")
    parser.add_argument("--n-provinces", type=int, default=32)
    parser.add_argument("--n-treated", type=int, default=None, help="Default: half the provinces")
    parser.add_argument("--districts-per-province", type=int, default=3)
    parser.add_argument("--unequal-districts", action="store_true")
    parser.add_argument("--policy-ratio", type=float, default=.7)
    parser.add_argument("--ramp-days", type=int, default=7)
    parser.add_argument("--base-cases", type=float, default=20)
    parser.add_argument("--baseline-sigma", type=float, default=.35)
    parser.add_argument("--serial-correlation", type=float, default=.8)
    parser.add_argument("--province-shock-sd", type=float, default=.15)
    parser.add_argument("--differential-trend", type=float, default=0)
    parser.add_argument("--reporting-ratio", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    add_analysis_arguments(parser)
    args = parser.parse_args()
    if args.expanded:
        if args.n_treated is None:
            args.n_treated = args.n_provinces // 2
        settings = RegionGeneration(**{name: getattr(args, name) for name in RegionGeneration.__dataclass_fields__})
        config = config_from_args(args)
        records, truth = generate_regions(settings, config)
        export_regions(args.output, settings, config, records, truth)
        print(f"Generated {settings.n_provinces} fictional provinces, {len(records)} districts: {args.output}")
    else:
        generate_hierarchical_data()
