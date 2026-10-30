import os
import shutil
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def verify_and_split_data(original_data_dir="./data", output_dir="./debug_data", num_clients=3):
    """
    전체 데이터를 검증 후 클라이언트별로 폴더에 복사합니다.
    깨진 이미지나 이상한 데이터는 자동으로 걸러냅니다.
    """
    csv_path = os.path.join(original_data_dir, "metadata.csv")
    img_src_dir = os.path.join(original_data_dir, "images")

    # 1. 초기화
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 메타데이터가 없습니다: {csv_path}")

    # 2. 데이터 로드 및 셔플
    df = pd.read_csv(csv_path)
    df['patient_id'] = df['patient_id'].astype(str)

    unique_pids = df['patient_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    pid_chunks = np.array_split(unique_pids, num_clients)

    print(f"📦 데이터 검증 및 분할 시작 (총 {len(df)}개 샘플)")
    print("=" * 60)

    # 3. 클라이언트별 분배 및 복사
    for client_id in range(num_clients):
        client_dir = os.path.join(output_dir, f"client_{client_id}")
        client_img_dir = os.path.join(client_dir, "images")
        os.makedirs(client_img_dir, exist_ok=True)

        my_pids = pid_chunks[client_id]
        temp_df = df[df['patient_id'].isin(my_pids)].copy()

        valid_rows = []
        error_count = 0

        print(f"  > [Client {client_id}] 환자 {len(my_pids)}명 할당. 데이터 검증 및 복사 중...")

        for idx, row in tqdm(temp_df.iterrows(), total=len(temp_df), leave=False):
            img_name = str(row['image_idx']).strip()
            src_path = os.path.join(img_src_dir, img_name)
            dst_path = os.path.join(client_img_dir, img_name)

            # [검증 1] 파일 존재 여부
            if not os.path.exists(src_path):
                print(f"    ⚠️ [Missing] 파일 없음: {img_name}")
                error_count += 1
                continue

            # [검증 2] 이미지 손상 여부 (OpenCV 로드 테스트)
            try:
                img = cv2.imread(src_path)
                if img is None:
                    print(f"    ⚠️ [Corrupt] 이미지 깨짐: {img_name}")
                    error_count += 1
                    continue
                if img.shape[0] < 10 or img.shape[1] < 10:
                    print(f"    ⚠️ [Too Small] 이미지 너무 작음: {img_name}")
                    error_count += 1
                    continue
            except Exception:
                error_count += 1
                continue

            # [검증 3] 라벨 값 확인
            try:
                blood = float(row['blood(mg/dL)'])
                if np.isnan(blood): raise ValueError
            except:
                print(f"    ⚠️ [Label Error] 빌리루빈 수치 오류: {img_name}")
                error_count += 1
                continue

            # 정상 데이터만 복사 및 추가
            shutil.copy2(src_path, dst_path)
            valid_rows.append(row)

        # CSV 저장
        if valid_rows:
            client_df = pd.DataFrame(valid_rows)
            client_df.to_csv(os.path.join(client_dir, "metadata.csv"), index=False)
            print(f"    ✅ 완료: {len(client_df)}장 저장됨. (제외된 불량 데이터: {error_count}개)")
        else:
            print(f"    🚨 [Critical] Client {client_id}에 유효한 데이터가 하나도 없습니다!")


def load_client_data_from_debug(client_id, debug_dir="./debug_data"):
    """분할된 debug_data 폴더에서 로드"""
    client_dir = os.path.join(debug_dir, f"client_{client_id}")
    csv_path = os.path.join(client_dir, "metadata.csv")
    img_dir = os.path.join(client_dir, "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Client {client_id} 데이터가 없습니다.")

    df = pd.read_csv(csv_path)

    df['patient_id'] = df['patient_id'].astype(str)
    ids = df['patient_id'].unique()
    if len(ids) < 2:
        raise ValueError("At least two patients are required for disjoint train/validation")
    train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=42)
    train_df = df[df['patient_id'].isin(train_ids)].copy()
    val_df = df[df['patient_id'].isin(val_ids)].copy()
    return train_df, val_df, img_dir
