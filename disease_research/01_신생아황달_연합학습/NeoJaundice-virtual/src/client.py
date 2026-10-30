import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
import os
import warnings

warnings.filterwarnings("ignore")

from src.dataset import JaundiceDataset, get_transforms
from src.model import MultiModalNet
from src.utils import load_partition, generate_dummy_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeonatalClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader):
        self.model = MultiModalNet().to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Freeze 했으므로 requires_grad=True인 파라미터만 학습
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)
        crit_reg = nn.MSELoss()
        crit_cls = nn.BCEWithLogitsLoss()

        self.model.train()
        self.model.cnn.eval()  # CNN은 eval 모드 고정

        epoch_loss = 0.0
        epochs = 3

        for _ in range(epochs):
            for (img, meta), (y_blood, y_treat) in self.train_loader:
                img, meta = img.to(DEVICE), meta.to(DEVICE)
                y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

                optimizer.zero_grad()
                pred_blood, pred_treat = self.model(img, meta)

                # pred_blood가 Sigmoid로 0~1 사이이므로 y_blood도 0~1이어야 함 (dataset 확인)
                loss = crit_reg(pred_blood, y_blood) + 2.0 * crit_cls(pred_treat, y_treat)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        mae_sum = 0.0
        correct = 0
        total = 0
        loss_sum = 0.0
        crit_reg = nn.MSELoss()

        with torch.no_grad():
            for (img, meta), (y_blood, y_treat) in self.val_loader:
                img, meta = img.to(DEVICE), meta.to(DEVICE)
                y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

                p_blood, p_treat = self.model(img, meta)

                loss_sum += crit_reg(p_blood, y_blood).item()

                # 결과 복원: 0~1 값에 30.0을 곱해 원래 수치로 변환
                real_pred = p_blood * 30.0
                real_target = y_blood * 30.0

                mae_sum += torch.abs(real_pred - real_target).sum().item()
                acc_pred = (torch.sigmoid(p_treat) > 0.5).float()
                correct += (acc_pred == y_treat).sum().item()
                total += len(y_blood)

        if total == 0: return 0.0, 0, {}
        return loss_sum / len(self.val_loader), total, {"mae": mae_sum / total, "accuracy": correct / total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()

    DATA_DIR = "./data"
    if not os.path.exists(os.path.join(DATA_DIR, "metadata.csv")):
        generate_dummy_data(DATA_DIR)

    train_df, val_df = load_partition(args.client_id, args.num_clients, os.path.join(DATA_DIR, "metadata.csv"))

    # ★ drop_last=True: 배치가 1개 남으면 버림 (BatchNorm 에러 방지)
    train_loader = DataLoader(
        JaundiceDataset(train_df, os.path.join(DATA_DIR, "images"), transform=get_transforms('train')),
        batch_size=8, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        JaundiceDataset(val_df, os.path.join(DATA_DIR, "images"), transform=get_transforms('val')),
        batch_size=8, drop_last=False
    )

    print(f"Starting Client {args.client_id}...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=NeonatalClient(train_loader, val_loader).to_client()
    )