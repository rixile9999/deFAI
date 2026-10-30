import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
import os
import sys
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

# utils에서 load_client_data_from_debug 임포트
from src.dataset import JaundiceDataset, get_transforms
from src.model import MultiModalNet
from src.utils import load_client_data_from_debug

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeonatalClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, client_id):
        self.model = MultiModalNet().to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if len(parameters) != len(self.model.state_dict()):
            raise ValueError("Wrong number of model parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        reference = self.model.state_dict()
        state_dict = OrderedDict({k: torch.as_tensor(v, dtype=reference[k].dtype) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        crit_reg = nn.SmoothL1Loss()
        crit_cls = nn.BCEWithLogitsLoss()

        self.model.train()
        epoch_loss = 0.0
        epochs = int(config.get("local_epochs", 3))
        examples = 0

        if len(self.train_loader) == 0:
            return self.get_parameters(config={}), 0, {"loss": 0.0}

        for _ in range(epochs):
            for i, ((img, meta), (y_blood, y_treat)) in enumerate(self.train_loader):
                img, meta = img.to(DEVICE), meta.to(DEVICE)
                y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

                # [안전장치] 배치 내 NaN 존재 여부 확인
                if not torch.isfinite(img).all() or not torch.isfinite(meta).all():
                    print(f"❌ [Client {self.client_id}] Input NaN detected at batch {i}")
                    raise ValueError("Nonfinite local input; refusing to silently omit training rows")

                optimizer.zero_grad()
                pred_blood, pred_treat = self.model(img, meta)

                loss = crit_reg(pred_blood * 30.0, y_blood * 30.0) + 0.5 * crit_cls(pred_treat, y_treat)

                if not torch.isfinite(loss):
                    print(f"⚠️ [Client {self.client_id}] NaN Loss! Target: {y_blood.mean().item():.2f}")
                    raise ValueError("Nonfinite local loss")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(img)
                examples += len(img)

        avg_loss = epoch_loss / examples if examples else 0.0
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        mae_sum = 0.0;
        correct = 0;
        total = 0;
        loss_sum = 0.0
        crit_reg = nn.MSELoss()

        with torch.no_grad():
            for (img, meta), (y_blood, y_treat) in self.val_loader:
                img, meta = img.to(DEVICE), meta.to(DEVICE)
                y_blood, y_treat = y_blood.to(DEVICE), y_treat.to(DEVICE)

                p_blood, p_treat = self.model(img, meta)
                loss_sum += crit_reg(p_blood, y_blood).item() * len(y_blood)

                real_pred = p_blood * 30.0
                real_target = y_blood * 30.0

                mae_sum += torch.abs(real_pred - real_target).sum().item()
                acc_pred = (torch.sigmoid(p_treat) > 0.5).float()
                correct += (acc_pred == y_treat).sum().item()
                total += len(y_blood)

        if total == 0: return 0.0, 0, {}
        return loss_sum / total, total, {"mae": mae_sum / total, "accuracy": correct / total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()

    DEBUG_DIR = "./debug_data"

    # 1. 분할된 데이터 로드
    try:
        train_df, val_df, img_dir = load_client_data_from_debug(args.client_id, DEBUG_DIR)
        print(f"🔍 [Client {args.client_id}] Loaded: {len(train_df)} train, {len(val_df)} val from {img_dir}")

        # 2. 데이터로더 설정
        train_loader = DataLoader(
            JaundiceDataset(train_df, img_dir, transform=get_transforms('train')),
            batch_size=8, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            JaundiceDataset(val_df, img_dir, transform=get_transforms('val')),
            batch_size=8, drop_last=False
        )

        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=NeonatalClient(train_loader, val_loader, args.client_id).to_client()
        )
    except Exception as e:
        print(f"❌ [Client {args.client_id}] Critical Error: {e}")
