import pandas as pd
import numpy as np
import glob

def verify_data():
    files = glob.glob("data/*.xlsx")
    print("\n📊 SYNTHETIC DATA VALIDATION REPORT")
    print("="*60)
    
    for f in files:
        df = pd.read_excel(f)
        region = df['거주지시도'].iloc[0]
        
        # -----------------------------------------------------
        # 1. Phase 2 검증: Severity Ratio (Selection Bias)
        # -----------------------------------------------------
        # 중증+사망 합계
        df['severe_total'] = df['위중증 유무(유:1)'].fillna(0) + df['사망유무(유:1)'].fillna(0)
        n_severe = df['severe_total'].sum()
        n_total = len(df)
        
        # Ratio Calculation
        ratio = (n_total - n_severe) / n_severe if n_severe > 0 else 0
        
        print(f"📍 Region: {region}")
        print(f"   [Bias Check] Observed Ratio (Mild : Severe)")
        print(f"     -> Actual: {ratio:.1f} : 1")
        
        # 기대값 판정
        if '서울' in region:
            print("     -> Target: ~20 : 1 (Pass ✅)" if 15 < ratio < 25 else "     -> Target: ~20 : 1 (Fail ❌)")
        elif '경기' in region:
            print("     -> Target: ~50 : 1 (Pass ✅)" if 40 < ratio < 60 else "     -> Target: ~50 : 1 (Fail ❌)")

        # -----------------------------------------------------
        # 2. Phase 3 검증: DiD Effect (Policy Impact)
        # -----------------------------------------------------
        df['date'] = pd.to_datetime(df['진단일'])
        start_date = df['date'].min()
        
        # Pre (Day 0-29) vs Post (Day 30-59) 비교
        policy_date = start_date + pd.Timedelta(days=30)
        end_eval_date = start_date + pd.Timedelta(days=60)
        
        pre_mask = df['date'] < policy_date
        post_mask = (df['date'] >= policy_date) & (df['date'] < end_eval_date)
        
        pre_daily_avg = df[pre_mask].groupby('date').size().mean()
        post_daily_avg = df[post_mask].groupby('date').size().mean()
        
        growth = (post_daily_avg - pre_daily_avg) / pre_daily_avg * 100
        
        print(f"   [Causal Check] Growth Rate (Pre vs Post)")
        print(f"     -> Change: {growth:+.1f}%")
        
        if '서울' in region:
            print("     -> Expect: Decrease or Low Growth due to Mask Mandate (Pass ✅)" if growth < 10 else "     -> Expect: Decrease (Fail ❌)")
        else:
            print("     -> Expect: Natural Increase (Pass ✅)" if growth > 0 else "     -> Expect: Increase (Fail ❌)")
            
        print("-" * 60)

if __name__ == "__main__":
    verify_data()