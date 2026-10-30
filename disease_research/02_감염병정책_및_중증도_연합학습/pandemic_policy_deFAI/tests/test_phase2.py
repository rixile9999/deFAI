import sys
import os
import numpy as np
from pathlib import Path

# [Path Fix] 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.client import FCausalClient
from epi.ipw import calculate_sa_ipw


def test_phase2():
    print("\n🧪 TEST: Phase 2 (Data Loading & SA-IPW)")

    # 데이터 경로 자동 탐색 (절대 경로)
    data_path = PROJECT_ROOT / 'data' / 'seoul.xlsx'

    if not data_path.exists():
        print(f"   ⚠️ Data not found at {data_path}")
        print("   -> Please run 'data_gen-v2.py' first.")
        return

    # Client 초기화 (절대 경로 전달)
    print(f"   -> Loading data from: {data_path.name}")
    try:
        client = FCausalClient(str(data_path))
    except Exception as e:
        print(f"   ❌ Client Error: {e}")
        return

    # 로직 검증
    weights = calculate_sa_ipw(client.daily_df)
    mean_w = np.mean(weights)
    print(f"   - Mean Weight (Seoul): {mean_w:.2f}")

    # 서울은 과소표집(1:20)되었으므로 가중치가 평균 1.0보다 커야 함 (약 2.5배 기대)
    if mean_w > 1.5:
        print("   ✅ Phase 2 Passed: Weights are correctly inflated.")
    else:
        print(f"   ❌ Phase 2 Failed: Weights are too low ({mean_w:.2f}).")


if __name__ == "__main__":
    test_phase2()