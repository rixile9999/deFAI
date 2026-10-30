import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_f_causal_data_v2():
    # 재현성을 위한 시드 고정
    np.random.seed(42)
    random.seed(42)
    os.makedirs('data', exist_ok=True)
    
    # ---------------------------------------------------------
    # 🛑 수정된 시뮬레이션 설정: 유행 확산기 (Rising Phase)
    # ---------------------------------------------------------
    TRUE_SEVERE_RATE = 0.02  # 실제 중증화율 (2%)
    
    REGIONS = {
        'Seoul': {
            'name_kr': '서울',
            'base_cases': 50,         # 초기 확진자 수
            'obs_ratio_target': 20,   # [Bias] 1:20 (검사 부족, 과소표집)
            'policy_day': 30,         # Day 30 마스크 의무화
            'policy_effect': 0.5      # [강화] 확산을 억제할 정도로 강력한 개입 (50% 감소)
        },
        'Gyeonggi': {
            'name_kr': '경기',
            'base_cases': 40,
            'obs_ratio_target': 50,   # [Truth] 1:50 (정상 표집)
            'policy_day': None,
            'policy_effect': 1.0      # 자연 증가 방치
        },
        'Busan': {
            'name_kr': '부산',
            'base_cases': 30,
            'obs_ratio_target': 30,   # [Bias] 1:30
            'policy_day': None,
            'policy_effect': 1.0      # 자연 증가 방치
        }
    }
    
    START_DATE = datetime(2024, 1, 1)
    DAYS = 100

    print("🚀 Generating F-Causal Synthetic Data v2 (Exponential + Seasonality)...")

    for region_eng, params in REGIONS.items():
        records = []
        
        # 발견 확률(Bias) 계산
        true_mild_rate = 1 - TRUE_SEVERE_RATE
        p_detect_mild = params['obs_ratio_target'] * (TRUE_SEVERE_RATE / true_mild_rate)
        p_detect_mild = min(p_detect_mild, 1.0)
        
        global_id = 1
        recent_ids = [] 
        
        for t in range(DAYS):
            curr_date = START_DATE + timedelta(days=t)
            
            # -----------------------------------------------------
            # [Trend 수정] 지수적 성장 + 주간 패턴
            # -----------------------------------------------------
            # 1. Exponential Growth: t=0(1.0) -> t=60(약 2.5배)
            growth_trend = np.exp(0.015 * t)
            
            # 2. Weekly Seasonality: 주말에 줄어드는 7일 주기 패턴 (현실성 강화)
            weekly_pattern = 1.0 + 0.15 * np.sin(2 * np.pi * t / 7)
            
            lambda_val = params['base_cases'] * growth_trend * weekly_pattern
            
            # 정책 효과 적용 (Ramp-up Logic)
            if params['policy_day'] and t >= params['policy_day']:
                days_after = t - params['policy_day']
                # 정책 효과가 1주일에 걸쳐 서서히 나타남
                ramp_up = min(1.0, days_after / 7.0)
                current_effect = 1.0 - (1.0 - params['policy_effect']) * ramp_up
                lambda_val *= current_effect
            
            n_true = int(np.random.poisson(max(lambda_val, 0)))
            
            # 환자 생성 루프
            for _ in range(n_true):
                is_severe_truth = np.random.random() < TRUE_SEVERE_RATE
                
                # Selection Bias 적용
                if is_severe_truth:
                    is_observed = True 
                else:
                    is_observed = np.random.random() < p_detect_mild
                
                if not is_observed: continue
                
                # 네트워크 구조 (Cluster vs Sporadic)
                prec_id = np.nan
                if recent_ids and np.random.random() < 0.3:
                    prec_id = random.choice(recent_ids)
                
                # 진단 지연 (Reporting Delay)
                delay = np.random.randint(1, 6)
                report_date = curr_date + timedelta(days=delay)
                
                # 결과(Outcome) 생성
                is_death = 0
                is_severe_rec = 0
                if is_severe_truth:
                    if np.random.random() < 0.2: is_death = 1
                    else: is_severe_rec = 1
                
                records.append({
                    'id': global_id,
                    '거주지시도': params['name_kr'],
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
            
        df = pd.DataFrame(records)
        filename = f"data/{region_eng.lower()}.xlsx"
        df.to_excel(filename, index=False)
        print(f"   -> Saved {filename} ({len(df)} rows)")

if __name__ == "__main__":
    generate_f_causal_data_v2()