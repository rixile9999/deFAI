import sys
import os
from pathlib import Path
import flwr.common as flc

# [Path Fix] 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.crypto import create_context
from core.client import FCausalClient
from core.strategy import FCausalStrategy


def test_phase3_simulation():
    print("\n🧪 TEST: Phase 3 (Secure Aggregation & DiD Simulation)")

    seoul_path = PROJECT_ROOT / 'data' / 'seoul.xlsx'
    gyeonggi_path = PROJECT_ROOT / 'data' / 'gyeonggi.xlsx'

    if not seoul_path.exists():
        print("   ⚠️ Data files missing. Run 'data_gen-v2.py' first.")
        return

    # 1. 암호화 컨텍스트 생성
    print("   -> Initializing Homomorphic Encryption Context...")
    ctx = create_context()

    # 2. 클라이언트 설정 (SMPC Mode)
    c1 = FCausalClient(str(seoul_path), context=ctx, use_smpc=True)
    c2 = FCausalClient(str(gyeonggi_path), context=ctx, use_smpc=True)

    # 3. 로컬 학습 및 암호화
    print("   -> Clients computing & encrypting local statistics...")
    res1, n1, m1 = c1.fit(None, {})
    res2, n2, m2 = c2.fit(None, {})

    # 암호화 여부 체크
    assert m1['encrypted'] is True, "Data should be encrypted!"

    # 4. 서버 집계 시뮬레이션
    class MockRes:
        def __init__(self, params, metrics):
            self.parameters = params
            self.metrics = metrics

    p1 = flc.ndarrays_to_parameters(res1)
    p2 = flc.ndarrays_to_parameters(res2)

    results = [
        (None, MockRes(p1, m1)),
        (None, MockRes(p2, m2))
    ]

    # 5. 전략 실행
    print("   -> Server aggregating encrypted vectors...")
    strategy = FCausalStrategy(context=ctx)
    strategy.aggregate_fit(1, results, [])

    print("\n   ✅ Phase 3 Simulation Completed.")


if __name__ == "__main__":
    test_phase3_simulation()