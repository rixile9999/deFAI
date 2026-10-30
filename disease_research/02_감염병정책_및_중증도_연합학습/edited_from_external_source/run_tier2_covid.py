import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")

import argparse
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score,
                              precision_recall_curve)
from sklearn.model_selection import train_test_split
import tenseal as ts
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────
MU           = 0.005   # FedProx proximal strength
LOCAL_LR     = 0.001   # Local learning rate
LOCAL_EPOCHS = 5       # Local training epochs
BATCH_SIZE   = 32
HIDDEN       = [64, 32]
TARGET_COL   = "위중증 유무(유:1)"   # Severe case as target
FEATURE_COLS = ["연령", "gender_num", "diag_delay"]


# ── MLP Model ─────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_COLS), HIDDEN[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN[1], 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def weights_to_array(model):
    """Flatten all model parameters to a 1D numpy array."""
    return np.concatenate([
        p.data.numpy().flatten() for p in model.parameters()
    ])

def array_to_weights(model, arr):
    """Load 1D array back into model parameters."""
    idx = 0
    for p in model.parameters():
        size = p.data.numel()
        p.data = torch.tensor(
            arr[idx:idx+size].reshape(p.data.shape),
            dtype=torch.float32
        )
        idx += size


# ── District Client ────────────────────────────────────────────
class DistrictClient(fl.client.NumPyClient):
    def __init__(self, filepath, context, use_smpc=True):
        self.context  = context
        self.use_smpc = use_smpc

        # Load data
        df = pd.read_excel(filepath)
        if df.empty:
            raise ValueError(f"Empty dataset: {filepath}")

        # District info
        self.province = df["시도"].iloc[0]
        self.district = df["시군구"].iloc[0]
        self.n_total  = len(df)

        # Feature engineering
        df["gender_num"]  = (df["성별"] == "남").astype(int)
        df["onset"]       = pd.to_datetime(df["발병일"])
        df["diag"]        = pd.to_datetime(df["진단일"])
        df["diag_delay"]  = (df["diag"] - df["onset"]).dt.days.clip(0, 30)
        df["target"]      = df[TARGET_COL].astype(int)

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)

        # Scale
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=0.2,
            stratify=y if y.sum() > 1 else None,
            random_state=42
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15,
            stratify=y_tr if y_tr.sum() > 1 else None,
            random_state=42
        )

        self.X_tr  = torch.tensor(X_tr)
        self.y_tr  = torch.tensor(y_tr)
        self.X_val = torch.tensor(X_val)
        self.y_val = torch.tensor(y_val)
        self.X_te  = torch.tensor(X_te)
        self.y_te  = torch.tensor(y_te)

        # Weighted BCE
        n_neg = (y_tr == 0).sum()
        n_pos = (y_tr == 1).sum()
        self.pos_weight = float(min(n_neg / max(n_pos, 1), 20.0))
        self.n_train    = len(X_tr)

        severe_rate = df["target"].mean() * 100

        print(f"\n{'='*60}")
        print(f"[TIER 2] District Client Initialized")
        print(f"{'='*60}")
        print(f"  Province    : {self.province}")
        print(f"  District    : {self.district}")
        print(f"  Total rows  : {self.n_total:,}")
        print(f"  Severe rate : {severe_rate:.2f}%")
        print(f"  Train/Val/Test: {len(X_tr)}/{len(X_val)}/{len(X_te)}")
        print(f"  Pos weight  : {self.pos_weight:.1f}x")
        print(f"{'='*60}")

    def get_parameters(self, config):
        """Return initial local model weights to the server."""
        model = MLP()
        initial_arr = weights_to_array(model).astype(np.float64)
        return [initial_arr]

    def fit(self, parameters, config):
        """Train local MLP with FedProx, return updated weights."""
        # Load global weights from server
        global_arr = parameters[0].astype(np.float32)
        model      = MLP()
        array_to_weights(model, global_arr)
        global_weights = [p.data.clone() for p in model.parameters()]

        # ── FedProx local training ────────────────────────────
        ds     = TensorDataset(self.X_tr, self.y_tr)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        opt    = torch.optim.Adam(model.parameters(), lr=LOCAL_LR)
        pw     = torch.tensor([self.pos_weight])

        model.train()
        for _ in range(LOCAL_EPOCHS):
            for Xb, yb in loader:
                opt.zero_grad()
                pred = model(Xb)

                # Weighted BCE loss
                w_batch = torch.where(yb == 1,
                                      pw.expand_as(yb),
                                      torch.ones_like(yb))
                loss = (w_batch * nn.functional.binary_cross_entropy(
                    pred, yb, reduction="none")).mean()

                # FedProx proximal term
                prox = sum(
                    ((p - gw) ** 2).sum()
                    for p, gw in zip(model.parameters(), global_weights)
                )
                loss = loss + (MU / 2) * prox
                loss.backward()
                opt.step()

        # ── Local threshold calibration ───────────────────────
        model.eval()
        with torch.no_grad():
            val_probs = model(self.X_val).numpy()
        prec, rec, thrs = precision_recall_curve(
            self.y_val.numpy(), val_probs
        )
        f1s     = [2*p*r/(p+r+1e-9)
                   for p, r in zip(prec[:-1], rec[:-1])]
        opt_thr = float(np.clip(thrs[np.argmax(f1s)], 0.05, 0.95))

        # Evaluate on test set with calibrated threshold
        with torch.no_grad():
            te_probs = model(self.X_te).numpy()
        te_preds = (te_probs >= opt_thr).astype(int)
        local_f1 = f1_score(self.y_te.numpy(), te_preds, zero_division=0)
        local_auc = roc_auc_score(
            self.y_te.numpy(), te_probs
        ) if len(np.unique(self.y_te.numpy())) > 1 else float("nan")

        print(f"\n  [{self.district}] Local F1={local_f1:.4f} "
              f"AUC={local_auc:.4f} thr={opt_thr:.3f}")

        # ── Serialize weights ──────────────────────────────────
        updated_arr = weights_to_array(model).astype(np.float64)

        metadata = {
            "district"   : self.district,
            "province"   : self.province,
            "n_train"    : self.n_train,
            "local_f1"   : float(local_f1),
            "local_auc"  : float(local_auc),
            "threshold"  : float(opt_thr),
            "encrypted"  : False,
        }

        if self.use_smpc and self.context:
            print(f"  [{self.district}] Encrypting weights "
                  f"({len(updated_arr)} values)...")
            enc_vec   = ts.ckks_vector(self.context, updated_arr.tolist())
            enc_bytes = enc_vec.serialize()
            enc_array = np.frombuffer(enc_bytes, dtype=np.uint8)
            metadata["encrypted"] = True
            print(f"  [{self.district}] Weights encrypted "
                  f"({len(enc_array)} bytes) -> sending to Tier 1")
            return [enc_array], self.n_train, metadata
        else:
            print(f"  [{self.district}] Sending weights (plaintext)")
            return [updated_arr], self.n_train, metadata

    def evaluate(self, parameters, config):
        return 0.0, self.n_train, {}


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tier 2 — District Office (COVID 18-district)"
    )
    parser.add_argument("--data",      required=True,
                        help="Path to district xlsx file")
    parser.add_argument("--target",    required=True,
                        help="Tier 1 address (e.g., 127.0.0.1:9001)")
    parser.add_argument("--plaintext", action="store_true",
                        help="Send without encryption (test mode)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: File not found: {args.data}")
        return

    use_smpc = not args.plaintext
    ctx      = None

    if use_smpc:
        pk_path = Path("keys/public.ctx")
        if not pk_path.exists():
            print("Error: keys/public.ctx not found.")
            print("Run setup_hierarchical.py first, or use --plaintext")
            return
        with open(pk_path, "rb") as f:
            ctx = ts.context_from(f.read())

    print("\n" + "="*70)
    print("[TIER 2] DISTRICT OFFICE — COVID 18-district")
    print("="*70)
    print(f"  Data       : {data_path.name}")
    print(f"  Target     : {args.target}")
    print(f"  Encryption : {'ENABLED' if use_smpc else 'DISABLED (plaintext)'}")
    print(f"  FedProx mu : {MU}")
    print(f"  Local epochs: {LOCAL_EPOCHS}")
    print("="*70)

    # try:
    client = DistrictClient(args.data, ctx, use_smpc)
    fl.client.start_numpy_client(
        server_address=args.target,
        client=client,
        grpc_max_message_length=1024*1024*1024
    )
    print(f"\n[TIER 2] {client.district}: Weights sent successfully!")
 #   except Exception as e:
  #      print(f"\n[TIER 2] Error: {e}")
   #     print("\nPress Enter to close...")
    #    input()


if __name__ == "__main__":
    main()
