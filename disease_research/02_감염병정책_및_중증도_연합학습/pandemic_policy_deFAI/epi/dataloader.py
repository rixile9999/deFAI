import pandas as pd
import numpy as np
from pathlib import Path


def load_data_and_preprocess(filepath, policy_date_str='2024-01-31'):
    """엑셀 데이터를 로드하고 DiD 분석용 변수(Post, Treated)를 생성합니다."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"❌ Error: Data file not found at {path.absolute()}")

    df = pd.read_excel(path)

    # 1. 날짜 및 중증 여부 전처리
    df['date'] = pd.to_datetime(df['진단일'])
    df['severe_outcome'] = df['위중증 유무(유:1)'].fillna(0) + df['사망유무(유:1)'].fillna(0)
    df['severe_outcome'] = (df['severe_outcome'] > 0).astype(int)

    # 2. 일별 집계 (Daily Aggregation)
    daily = df.groupby('date').agg({
        'id': 'count',
        'severe_outcome': 'sum',
        '거주지시도': 'first'
    }).rename(columns={'id': 'cases', 'severe_outcome': 'severe_total'})

    # 3. DiD 변수 생성
    # Post: 정책 시행일 이후
    policy_date = pd.to_datetime(policy_date_str)
    daily['post'] = (daily.index >= policy_date).astype(int)

    # Treated: 서울 지역
    region = str(daily['거주지시도'].iloc[0])
    daily['treated'] = 1 if '서울' in region else 0

    # Interaction Term (핵심: 정책 시행 후 x 치료군)
    daily['did'] = daily['post'] * daily['treated']
    daily['intercept'] = 1.0

    return daily


def prepare_xy(daily_df):
    """X(Design Matrix)와 y(Log Outcome)를 추출합니다."""
    if daily_df.empty: return np.array([]), np.array([])

    # X: [Intercept, Post, Treated, DiD]
    X = daily_df[['intercept', 'post', 'treated', 'did']].values.astype(np.float64)

    # y: Log(Cases) -> 회귀계수를 '변화율(%)'로 해석하기 위함 (log(0) 방지 epsilon 추가)
    y = np.log(daily_df['cases'].values.astype(np.float64) + 1e-5)

    return X, y