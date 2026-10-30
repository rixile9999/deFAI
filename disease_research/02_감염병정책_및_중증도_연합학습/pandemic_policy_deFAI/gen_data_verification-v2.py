import pandas as pd
import numpy as np
import glob
from datetime import timedelta

def verify_data_v2():
    files = glob.glob("data/*.xlsx")
    print("\n📊 F-CAUSAL DATA VALIDATION REPORT (V2)")
    print("="*70)
    
    stats = {}
    
    for f in files:
        df = pd.read_excel(f)
        region = df['거주지시도'].iloc[0]
        df['date'] = pd.to_datetime(df['진단일'])
        
        # 1. Bias Check (Selection Bias)
        df['severe_total'] = df['위중증 유무(유:1)'].fillna(0) + df['사망유무(유:1)'].fillna(0)
        n_severe = df['severe_total'].sum()
        # 관측된 경증:중증 비율
        ratio = (len(df) - n_severe) / n_severe if n_severe > 0 else 0
        
        print(f"📍 Region: {region}")
        print(f"   [Bias] Observed Mild:Severe = {ratio:.1f} : 1")
        
        # 2. Growth Check (Pre vs Post Policy)
        # 정책일(Day 30) 기준, 전후 20일 비교 (적응기 5일 제외)
        start_date = df['date'].min()
        policy_date = start_date + timedelta(days=30)
        
        pre_mask = (df['date'] >= start_date + timedelta(days=5)) & (df['date'] < policy_date)
        post_mask = (df['date'] >= policy_date + timedelta(days=5)) & (df['date'] < policy_date + timedelta(days=30))
        
        pre_avg = df[pre_mask].groupby('date').size().mean()
        post_avg = df[post_mask].groupby('date').size().mean()
        
        if pd.isna(pre_avg) or pre_avg == 0: pre_avg = 1
        
        growth_rate = ((post_avg - pre_avg) / pre_avg) * 100
        stats[region] = growth_rate
        
        print(f"   [Trend] Growth Rate: {growth_rate:+.1f}%")
        
        # 통제군(경기, 부산)은 반드시 증가해야 함 (Natural Outbreak)
        if '서울' not in region:
            if growth_rate > 15:
                print("     -> ✅ Control group growing naturally.")
            else:
                print("     -> ❌ Control group failed to grow.")
        print("-" * 70)

    # 3. DiD (Difference-in-Differences) Analysis
    seoul_growth = stats.get('서울', 0)
    control_growth_avg = np.mean([v for k, v in stats.items() if '서울' not in k])
    
    # Net Policy Effect = (서울 성장률) - (통제군 평균 성장률)
    did_effect = seoul_growth - control_growth_avg
    
    print(f"\n🧪 DiD Analysis Summary")
    print(f"   - Control Group Avg Growth: {control_growth_avg:+.1f}% (Counterfactual)")
    print(f"   - Seoul Growth: {seoul_growth:+.1f}% (Intervention)")
    print(f"   - Net Policy Effect (DiD): {did_effect:+.1f}%p")
    
    # 서울의 성장률이 통제군보다 30%p 이상 낮으면 효과 입증
    if did_effect < -30:
        print("\n✅ VALIDATION SUCCESS: Significant causal effect detected.")
        print("   (Seoul suppressed the outbreak relative to neighbors)")
    else:
        print("\n❌ VALIDATION FAIL: Policy effect unclear.")

if __name__ == "__main__":
    verify_data_v2()