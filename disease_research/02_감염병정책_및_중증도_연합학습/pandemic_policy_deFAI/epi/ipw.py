import numpy as np
import pandas as pd


def calculate_sa_ipw(daily_df: pd.DataFrame, global_severe_ratio=0.02) -> np.ndarray:
    """
    Severity-Anchored IPW:
    관측된 중증 비율이 글로벌 기준(2%)보다 높으면(=경증 누락),
    과소표집된 것으로 간주하고 가중치를 부여합니다.
    """
    if daily_df.empty: return np.array([])

    weights = []
    # Target: 중증 1명당 경증 49명 (비율 1:49)
    target_mild_per_severe = (1.0 - global_severe_ratio) / global_severe_ratio

    for _, row in daily_df.iterrows():
        n_severe = row['severe_total']
        n_total = row['cases']
        n_mild_obs = n_total - n_severe

        # 데이터 오류/부족 시 가중치 1.0
        if n_severe <= 0 or n_mild_obs <= 0:
            weights.append(1.0)
            continue

        # 기대되는 경증 환자 수 = 중증 * 49
        expected_mild = n_severe * target_mild_per_severe

        # 가중치 = 기대값 / 관측값 (최대 20배로 제한)
        w = expected_mild / n_mild_obs
        weights.append(max(1.0, min(w, 20.0)))

    return np.array(weights)