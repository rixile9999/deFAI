import flwr as fl
import numpy as np
import sys
import os
from pathlib import Path

# ---------------------------------------------------------
# [Path Fix] 프로젝트 루트를 sys.path에 추가하여 모듈 Import 보장
# ---------------------------------------------------------
# 현재 파일(core/client.py)의 상위(core)의 상위(root)를 찾음
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.serialization import pack_payload
from core.crypto import encrypt_vector
# 명시적 Import (이제 에러 없이 작동합니다)
from epi.dataloader import load_data_and_preprocess, prepare_xy
from epi.ipw import calculate_sa_ipw


class FCausalClient(fl.client.NumPyClient):
    def __init__(self, filepath, context=None, use_smpc=False):
        self.context = context
        self.use_smpc = use_smpc

        # 1. 데이터 로드 (경로 스마트 탐색)
        fpath = Path(filepath)
        if not fpath.is_absolute():
            # 상대 경로가 들어오면 프로젝트 루트 기준으로 변환
            fpath = PROJECT_ROOT / filepath

        self.filepath = fpath
        self.daily_df = load_data_and_preprocess(fpath)
        self.X, self.y = prepare_xy(self.daily_df)

    def fit(self, parameters, config):
        if self.daily_df.empty:
            print(f"⚠️ Warning: Empty dataset in {self.filepath}")
            return [], 0, {}

        # 2. [Phase 2] SA-IPW 가중치 계산
        weights = calculate_sa_ipw(self.daily_df)

        # 3. Weighted Sufficient Statistics 계산
        # X 행렬의 각 행에 가중치를 곱함 (Broadcasting)
        X_weighted = self.X * weights[:, np.newaxis]

        xtwx = self.X.T @ X_weighted  # Weighted Hessian
        xtwy = self.X.T @ (self.y * weights)  # Weighted Gradient

        n_samples = len(self.y)

        # 4. [Phase 3] 전송 (암호화 vs 평문 분기)
        if self.use_smpc and self.context:
            # 암호화 모드 (TenSEAL)
            enc_xtwx = np.frombuffer(encrypt_vector(self.context, xtwx), dtype=np.uint8)
            enc_xtwy = np.frombuffer(encrypt_vector(self.context, xtwy), dtype=np.uint8)
            return [enc_xtwx, enc_xtwy], n_samples, {"encrypted": True}
        else:
            # 평문 모드 (Phase 1 호환)
            payload = {"xtx": xtwx, "xty": xtwy}
            return pack_payload(payload), n_samples, {"encrypted": False}

    def evaluate(self, parameters, config):
        return 0.0, len(self.y), {}