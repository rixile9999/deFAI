import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_f_causal_data():
    # 시드 고정 및 디렉토리 설정
    np.random.seed(42)
    random.seed(42)
    os.makedirs('data', exist_ok=True)
    
    # ---------------------------------------------------------
    # 🛑 시뮬레이션 설정 (Simulation Config)
    # ---------------------------------------------------------
    # 공통 상수 (Biological Truth): 실제 중증화율 2% (1:50)
    TRUE_SEVERE_RATE = 0.02
    
    # 지자체별 시나리오 설정
    REGIONS = {
        'Seoul': {
            'name_kr': '서울',
            'base_cases': 150,        # 일일 확산 규모
            'obs_ratio_target': 20,   # [Bias High] 관측 중증 1명당 경증 20명 (과소표집)
            'policy_day': 30,         # Day 30 마스크 의무화
            'policy_effect': 0.6      # 감염 40% 감소 효과 (Treatment)
        },
        'Gyeonggi': {
            'name_kr': '경기',
            'base_cases': 120,
            'obs_ratio_target': 50,   # [Bias None] 관측 중증 1명당 경증 50명 (Ground Truth 일치)
            'policy_day': None,       # 정책 없음
            'policy_effect': 1.0      # (Control)
        },
        'Busan': {
            'name_kr': '부산',
            'base_cases': 80,
            'obs_ratio_target': 30,   # [Bias Medium] 중간 수준
            'policy_day': None,
            'policy_effect': 1.0      # (Control)
        }
    }
    
    START_DATE = datetime(2024, 1, 1)
    DAYS = 100

    print("🚀 Generating F-Causal Synthetic Data...")

    # ---------------------------------------------------------
    # 🔄 데이터 생성 루프
    # ---------------------------------------------------------
    for region_eng, params in REGIONS.items():
        records = []
        
        # 1. Detection Probability 계산 (Selection Bias 주입)
        # 목표: Observed_Mild / Observed_Severe = Target_Ratio
        # 식: (True_Mild * P_detect) / (True_Severe * 1.0) = Target_Ratio
        # 따라서 P_detect = Target_Ratio * (True_Severe / True_Mild)
        true_mild_rate = 1 - TRUE_SEVERE_RATE
        p_detect_mild = params['obs_ratio_target'] * (TRUE_SEVERE_RATE / true_mild_rate)
        p_detect_mild = min(p_detect_mild, 1.0)
        
        print(f"[{region_eng}] Target Ratio 1:{params['obs_ratio_target']} -> Mild Detection Prob: {p_detect_mild:.1%}")
        
        global_id = 1
        recent_ids = [] # 선행확진자 연결용 풀
        
        for t in range(DAYS):
            curr_date = START_DATE + timedelta(days=t)
            
            # A. True Infection Dynamics (진실)
            # 자연 파동 (Sine wave)
            trend = 1.0 + 0.2 * np.sin(t / 10.0)
            lambda_val = params['base_cases'] * trend
            
            # 서울 Day 30 정책 효과 적용
            if params['policy_day'] and t >= params['policy_day']:
                # 효과가 3일에 걸쳐 점진적으로 나타남 (Lag)
                days_after = t - params['policy_day']
                impact_factor = max(params['policy_effect'], 1.0 - 0.1 * days_after)
                impact_factor = max(impact_factor, params['policy_effect'])
                lambda_val *= impact_factor
            
            n_true = np.random.poisson(lambda_val)
            
            # B. Observation & Reporting (관측)
            for _ in range(n_true):
                # 생물학적 상태 결정
                is_severe_truth = np.random.random() < TRUE_SEVERE_RATE
                
                # 신고/발견 여부 (Bias Injection)
                if is_severe_truth:
                    is_observed = True # 중증은 100% 발견 가정
                else:
                    is_observed = np.random.random() < p_detect_mild
                
                if not is_observed:
                    continue # 데이터셋 누락 (Missing)
                
                # C. 레코드 작성 (KDCA Format)
                # 선행확진자: 30% 확률로 연결 (Cluster), 나머지는 산발적(Sporadic)
                prec_id = np.nan
                if recent_ids and np.random.random() < 0.3:
                    prec_id = random.choice(recent_ids)
                
                # 날짜 지연 (진단일 = 발병일 + 1~5일)
                delay = np.random.randint(1, 6)
                report_date = curr_date + timedelta(days=delay)
                
                # 위중증/사망 마킹 (중증 중 20% 사망 가정)
                is_death = 0
                is_severe_rec = 0
                if is_severe_truth:
                    if np.random.random() < 0.2:
                        is_death = 1
                    else:
                        is_severe_rec = 1
                
                record = {
                    'id': global_id,
                    '거주지시도': params['name_kr'],
                    '발병일': curr_date.strftime('%Y-%m-%d'),
                    '진단일': report_date.strftime('%Y-%m-%d'),
                    '위중증 유무(유:1)': 1 if is_severe_rec else 0,
                    '사망유무(유:1)': 1 if is_death else 0,
                    '선행확진자_번호': prec_id,
                    # 기타 더미 컬럼
                    '성별': random.choice(['남', '여']),
                    '연령': np.random.randint(10, 90)
                }
                records.append(record)
                
                recent_ids.append(global_id)
                global_id += 1
            
            # ID 풀 관리 (최근 7일치만 유지)
            recent_ids = recent_ids[-300:]
            
        # 파일 저장
        df = pd.DataFrame(records)
        filename = f"data/{region_eng.lower()}.xlsx"
        df.to_excel(filename, index=False)
        print(f"   -> Saved {filename} ({len(df)} rows)")

if __name__ == "__main__":
    generate_f_causal_data()