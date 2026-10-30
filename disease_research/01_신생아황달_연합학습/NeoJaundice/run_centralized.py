import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# 기존 프로젝트 모듈 재사용 (개선된 로직이 포함된 파일들)
from src.dataset import JaundiceDataset, get_transforms
from src.model import MultiModalNet

# 경고 무시
warnings.filterwarnings("ignore")

# --- 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"  # 원본 데이터 폴더
CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")

BATCH_SIZE = 8  # 연합학습과 동일하게 설정 (메모리 이슈 및 배치 통계 일치)
LEARNING_RATE = 1e-4
EPOCHS = 20  # 연합학습 라운드 수와 비슷하게 설정


def filter_valid_data(csv_path, img_dir):
    """
    [데이터 검증] 연합학습의 utils.py와 동일한 기준으로
    깨진 이미지나 잘못된 데이터를 사전에 걸러냅니다.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"데이터 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    valid_indices = []

    print("🔍 데이터 무결성 검사 중 (Centralized)...")

    for idx, row in tqdm(df.iterrows(), total=len(df), leave=False):
        img_name = str(row['image_idx']).strip()
        img_path = os.path.join(img_dir, img_name)

        # 1. 파일 존재 여부
        if not os.path.exists(img_path):
            continue

        # 2. 이미지 로딩 테스트 (OpenCV)
        try:
            img = cv2.imread(img_path)
            if img is None or img.shape[0] < 10 or img.shape[1] < 10:
                continue
        except:
            continue

        valid_indices.append(idx)

    clean_df = df.iloc[valid_indices].reset_index(drop=True)
    print(f"✅ 전체 {len(df)}개 중 유효한 데이터 {len(clean_df)}개 선별 완료.")
    return clean_df


def train_centralized():
    print(f"🚀 Centralized Learning (일반 학습) 시작 (Device: {DEVICE})")

    # 1. 데이터 로드 및 분할
    # 깨진 파일 필터링
    try:
        full_df = filter_valid_data(CSV_PATH, IMG_DIR)
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return

    # 환자 ID 기준 Split (Data Leakage 방지)
    full_df['patient_id'] = full_df['patient_id'].astype(str)
    all_pids = full_df['patient_id'].unique()
    train_pids, val_pids = train_test_split(all_pids, test_size=0.2, random_state=42)

    train_df = full_df[full_df['patient_id'].isin(train_pids)].reset_index(drop=True)
    val_df = full_df[full_df['patient_id'].isin(val_pids)].reset_index(drop=True)

    print(f"📊 데이터셋 구성:")
    print(f"   - Train: {len(train_df)} samples (Patients: {len(train_pids)})")
    print(f"   - Val  : {len(val_df)} samples (Patients: {len(val_pids)})")

    # 2. DataLoader 설정
    # [중요] drop_last=True: 배치 정규화 충돌 방지
    train_loader = DataLoader(
        JaundiceDataset(train_df, IMG_DIR, transform=get_transforms('train')),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        JaundiceDataset(val_df, IMG_DIR, transform=get_transforms('val')),
        batch_size=BATCH_SIZE, drop_last=False
    )

    # 3. 모델 초기화
    model = MultiModalNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    crit_reg = nn.MSELoss()
    crit_cls = nn.BCEWithLogitsLoss()

    # 4. 학습 루프
    print("\n" + "=" * 80)
    print(f"{'Epoch':^5} | {'Train Loss':^12} | {'Val MAE':^12} | {'Val Acc':^12} | {'Status':^10}")
    print("=" * 80)

    best_mae = float('inf')

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        for (img, meta), (y_blood, y_treat) in train_loader:
            img, meta = img.to(DEVICE), meta.to(DEVICE)
            y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

            # [안전장치] NaN 입력 방지
            if torch.isnan(img).any() or torch.isnan(meta).any():
                continue

            optimizer.zero_grad()
            pred_blood, pred_treat = model(img, meta)

            # Loss 계산 (연합학습과 동일 비중)
            loss = crit_reg(pred_blood, y_blood) + 2.0 * crit_cls(pred_treat, y_treat)

            # [안전장치] NaN Loss 방지
            if torch.isnan(loss):
                continue

            loss.backward()

            # [안전장치] Gradient Clipping (가중치 폭발 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0

        # --- Validation ---
        model.eval()
        mae_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for (img, meta), (y_blood, y_treat) in val_loader:
                img, meta = img.to(DEVICE), meta.to(DEVICE)
                y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

                p_blood, p_treat = model(img, meta)

                # [복원] 평가 시 스케일링(30.0) 복원하여 실제 오차 계산
                real_pred = p_blood * 30.0
                real_target = y_blood * 30.0

                mae_sum += torch.abs(real_pred - real_target).sum().item()

                acc_pred = (torch.sigmoid(p_treat) > 0.5).float()
                correct += (acc_pred == y_treat).sum().item()
                total += len(y_blood)

        val_mae = mae_sum / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0

        # 결과 출력 및 저장
        status = ""
        if val_mae < best_mae:
            best_mae = val_mae
            status = "★ Best"
            torch.save(model.state_dict(), "centralized_best.pth")

        print(f"{epoch + 1:^5} | {avg_train_loss:^12.4f} | {val_mae:^12.4f} | {val_acc:^12.2%} | {status}")

    print("=" * 80)
    print(f"🏆 학습 종료. Best Valid MAE: {best_mae:.4f}")


if __name__ == "__main__":
    train_centralized()