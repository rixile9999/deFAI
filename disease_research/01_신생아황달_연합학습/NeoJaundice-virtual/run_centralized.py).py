import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 기존 프로젝트 모듈 재사용 (동일 조건 비교를 위해 필수)
from src.dataset import JaundiceDataset, get_transforms
from src.model import MultiModalNet
from src.utils import generate_dummy_data

# --- 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20  # 연합학습(5라운드 x 3에폭 = 15)과 비슷한 총 학습량 설정


def load_split_data():
    """
    모든 데이터를 로드하고 '환자 ID'를 기준으로 학습/검증 데이터로 분리합니다.
    """
    # 1. 데이터 확인 및 생성
    if not os.path.exists(CSV_PATH):
        print("데이터가 없어 더미 데이터를 생성합니다...")
        generate_dummy_data(DATA_DIR)

    full_df = pd.read_csv(CSV_PATH)
    print(f"✅ 전체 데이터 로드 완료: 총 {len(full_df)}개 샘플")

    # 2. 환자 ID 기준 분할 (Data Leakage 방지 중요!)
    # 이미지를 무작위로 섞으면 같은 환자의 사진이 train/val에 섞여 성능이 뻥튀기됨
    all_pids = full_df['patient_id'].unique()
    train_pids, val_pids = train_test_split(all_pids, test_size=0.2, random_state=42)

    train_df = full_df[full_df['patient_id'].isin(train_pids)].reset_index(drop=True)
    val_df = full_df[full_df['patient_id'].isin(val_pids)].reset_index(drop=True)

    print(f"📊 데이터 분할 결과:")
    print(f"   - 학습용: 환자 {len(train_pids)}명 ({len(train_df)}장)")
    print(f"   - 검증용: 환자 {len(val_pids)}명 ({len(val_df)}장)")

    return train_df, val_df


def train_centralized():
    print(f"🚀 Centralized Learning 시작 (Device: {DEVICE})")

    # 1. 데이터셋 준비
    train_df, val_df = load_split_data()

    train_ds = JaundiceDataset(train_df, IMG_DIR, transform=get_transforms('train'))
    val_ds = JaundiceDataset(val_df, IMG_DIR, transform=get_transforms('val'))

    # num_workers=0은 윈도우 환경 호환성을 위함
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 모델 초기화
    model = MultiModalNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    crit_reg = nn.MSELoss()  # 회귀 (빌리루빈 수치)
    crit_cls = nn.BCEWithLogitsLoss()  # 분류 (치료 필요 여부)

    # 3. 학습 루프
    best_acc = 0.0
    print("\n" + "=" * 60)
    print(f"{'Epoch':^5} | {'Train Loss':^10} | {'Val MAE':^10} | {'Val Acc':^10}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # --- 학습 (Train) ---
        model.train()
        train_loss = 0.0

        # tqdm으로 진행바 표시
        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}", leave=False)
        for (img, meta), (y_blood, y_treat) in loop:
            img, meta = img.to(DEVICE), meta.to(DEVICE)
            y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

            optimizer.zero_grad()
            pred_blood, pred_treat = model(img, meta)

            # Loss: MSE + 5 * BCE (연합학습과 동일 가중치)
            loss = crit_reg(pred_blood, y_blood) + 5.0 * crit_cls(pred_treat, y_treat)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = train_loss / len(train_loader)

        # --- 검증 (Validation) ---
        model.eval()
        mae_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for (img, meta), (y_blood, y_treat) in val_loader:
                img, meta = img.to(DEVICE), meta.to(DEVICE)
                y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

                p_blood, p_treat = model(img, meta)

                mae_sum += torch.abs(p_blood - y_blood).sum().item()

                acc_pred = (torch.sigmoid(p_treat) > 0.5).float()
                correct += (acc_pred == y_treat).sum().item()
                total += len(y_blood)

        val_mae = mae_sum / total
        val_acc = correct / total

        # 결과 출력
        print(f"{epoch + 1:^5} | {avg_loss:^10.4f} | {val_mae:^10.4f} | {val_acc:^10.2%}")

        # 최고 성능 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "centralized_best_model.pth")

    print("=" * 60)
    print(f"🏆 학습 완료! 최고 정확도: {best_acc * 100:.2f}%")
    print("💾 모델 저장됨: centralized_best_model.pth")


if __name__ == "__main__":
    train_centralized()