import os
import sys
import pandas as pd
import numpy as np
import tenseal as ts
from pathlib import Path


# 1. 키 생성 함수
def generate_pki():
    print("🔐 Generating PKI Keys (Public/Secret)...")
    os.makedirs("keys", exist_ok=True)

    # CKKS Context 생성
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    # Secret Key 포함 (중앙정부용)
    with open("keys/secret.ctx", "wb") as f:
        f.write(context.serialize(save_secret_key=True))

    # Secret Key 제거 (지자체용 - 암호화 및 합산만 가능)
    with open("keys/public.ctx", "wb") as f:
        f.write(context.serialize(save_secret_key=False))
    print("   ✅ Keys saved in 'keys/' directory.")


# 2. 데이터 분할 함수
def split_data():
    print("✂️ Splitting data into Tier-2 Districts...")
    os.makedirs("data/districts", exist_ok=True)

    # 시뮬레이션 매핑
    mapping = {
        'seoul.xlsx': ['gangnam', 'seocho', 'jongno'],
        'gyeonggi.xlsx': ['suwon', 'seongnam', 'yongin']
    }

    for filename, districts in mapping.items():
        src = Path(f"data/{filename}")
        if not src.exists(): continue

        df = pd.read_excel(src)
        # 데이터를 랜덤하게 분할
        chunks = np.array_split(df.sample(frac=1), len(districts))

        for name, chunk in zip(districts, chunks):
            sido = filename.split('.')[0]
            path = f"data/districts/{sido}_{name}.xlsx"
            chunk.to_excel(path, index=False)
            print(f"   -> Created {path}")


if __name__ == "__main__":
    generate_pki()
    split_data()