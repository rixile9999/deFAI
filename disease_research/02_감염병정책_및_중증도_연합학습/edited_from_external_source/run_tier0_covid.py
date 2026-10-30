import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")

import argparse
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              precision_score, roc_auc_score, log_loss,
                              precision_recall_curve)
import tenseal as ts
from pathlib import Path
import glob

# ── CONFIG ─────────────────────────────────────────────────────
SERVER_LR    = 0.001   # FedAdam learning rate
BETA1        = 0.9     # FedAdam momentum
BETA2        = 0.999   # FedAdam second moment
EPS_ADAM     = 1e-8
HIDDEN       = [64, 32]
FEATURE_COLS = ["연령", "gender_num", "diag_delay"]
TARGET_COL   = "위중증 유무(유:1)"

# Path to district data for evaluation
DATA_DIR = "data/districts"   # update if needed


# ── MLP Model (same as Tier 2) ────────────────────────────────
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
    return np.concatenate([
        p.data.numpy().flatten() for p in model.parameters()
    ])

def array_to_weights(model, arr):
    idx = 0
    for p in model.parameters():
        size = p.data.numel()
        p.data = torch.tensor(
            arr[idx:idx+size].reshape(p.data.shape),
            dtype=torch.float32
        )
        idx += size

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


# ── Load all district data for evaluation ─────────────────────
def load_all_districts(data_dir):
    """Load all xlsx files from data directory for evaluation."""
    files = glob.glob(f"{data_dir}/*.xlsx")
    if not files:
        # Try current directory
        files = glob.glob("*.xlsx")
    if not files:
        print(f"  Warning: No xlsx files found in {data_dir}")
        return {}

    clients = {}
    for fpath in sorted(files):
        fname = Path(fpath).stem
        try:
            df = pd.read_excel(fpath)
            df["gender_num"] = (df["성별"] == "남").astype(int)
            df["onset"]      = pd.to_datetime(df["발병일"])
            df["diag"]       = pd.to_datetime(df["진단일"])
            df["diag_delay"] = (df["diag"] - df["onset"]).dt.days.clip(0, 30)
            df["target"]     = df[TARGET_COL].astype(int)

            X = df[FEATURE_COLS].values.astype(np.float32)
            y = df["target"].values

            sc  = StandardScaler()
            X_s = sc.fit_transform(X)

            clients[fname] = {
                "X": torch.tensor(X_s),
                "y": torch.tensor(y.astype(np.float32)),
                "province": df["시도"].iloc[0],
                "district": df["시군구"].iloc[0],
                "n": len(df),
            }
        except Exception as e:
            print(f"  Warning: Could not load {fname}: {e}")

    return clients


# ── Central Strategy ───────────────────────────────────────────
class CentralStrategy(fl.server.strategy.FedAvg):
    """
    FedAdam aggregation at central level.
    Receives city-level aggregated weights, applies
    server-side Adam optimizer to update global model.
    """
    def __init__(self, context, num_cities, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.context    = context
        self.num_cities = num_cities
        self.data_dir   = data_dir
        self.round_num  = 0
        self.history    = []

        # FedAdam state
        self.global_model  = MLP()
        self.global_arr    = weights_to_array(self.global_model)
        self.m_adam = np.zeros_like(self.global_arr)
        self.v_adam = np.zeros_like(self.global_arr)

        # Load district data for evaluation
        print("\n  Loading district data for evaluation...")
        self.clients = load_all_districts(data_dir)
        print(f"  Loaded {len(self.clients)} districts")

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        self.round_num += 1
        print(f"\n{'='*70}")
        print(f"[TIER 0] Round {server_round}: Aggregating from "
              f"{len(results)} cities")
        print(f"{'='*70}")

        city_weights = []
        city_n       = []
        city_info    = []

        for _, fit_res in results:
            params   = fl.common.parameters_to_ndarrays(fit_res.parameters)
            is_enc   = fit_res.metrics.get("encrypted", False)
            city     = fit_res.metrics.get("city", "Unknown")
            total_n  = fit_res.metrics.get("total_n", fit_res.num_examples)

            if is_enc and self.context:
                enc_bytes = params[0].tobytes()
                vec       = ts.ckks_vector_from(self.context, enc_bytes)
                weights   = np.array(vec.decrypt(), dtype=np.float32)
            else:
                weights = params[0].astype(np.float32)

            city_weights.append(weights)
            city_n.append(total_n)
            city_info.append({"city": city, "n": total_n})
            print(f"  City: {city:<12} n={total_n:>6,} samples")

        # ── FedAdam: weighted average + adaptive update ───────
        # Step 1: weighted average of city weights (inter-cluster)
        total_n  = sum(city_n)
        agg_arr  = sum(
            w * (n / total_n)
            for w, n in zip(city_weights, city_n)
        )

        # Step 2: compute pseudo-gradient (difference from global)
        pseudo_grad = agg_arr - self.global_arr

        # Step 3: FedAdam update
        t = server_round
        self.m_adam = (BETA1 * self.m_adam +
                       (1 - BETA1) * pseudo_grad)
        self.v_adam = (BETA2 * self.v_adam +
                       (1 - BETA2) * pseudo_grad ** 2)
        m_hat = self.m_adam / (1 - BETA1 ** t)
        v_hat = self.v_adam / (1 - BETA2 ** t)

        self.global_arr = (self.global_arr +
                           SERVER_LR * m_hat / (np.sqrt(v_hat) + EPS_ADAM))

        # Update model
        array_to_weights(self.global_model, self.global_arr)

        # ── Evaluate global model ─────────────────────────────
        self._evaluate_global(server_round)

        return None, {}

    def _evaluate_global(self, server_round):
        """Evaluate global model across all district test sets."""
        if not self.clients:
            return

        self.global_model.eval()
        per_client_aucs = []
        all_y    = []
        all_prob = []

        with torch.no_grad():
            for fname, c in self.clients.items():
                probs = self.global_model(c["X"]).numpy()
                y     = c["y"].numpy()

                auc = safe_auc(y, probs)
                if not np.isnan(auc):
                    per_client_aucs.append(auc)

                all_y.extend(y.tolist())
                all_prob.extend(probs.tolist())

        all_y    = np.array(all_y)
        all_prob = np.array(all_prob)

        # Global optimal threshold
        prec, rec, thrs = precision_recall_curve(all_y, all_prob)
        f1s     = [2*p*r/(p+r+1e-9)
                   for p, r in zip(prec[:-1], rec[:-1])]
        opt_thr = float(np.clip(thrs[np.argmax(f1s)], 0.05, 0.95))
        preds   = (all_prob >= opt_thr).astype(int)

        overall_auc  = safe_auc(all_y, all_prob)
        macro_auc    = float(np.nanmean(per_client_aucs))
        worst_auc    = float(np.nanmin(per_client_aucs))
        overall_f1   = f1_score(all_y, preds, zero_division=0)
        overall_rec  = recall_score(all_y, preds, zero_division=0)
        overall_prec = precision_score(all_y, preds, zero_division=0)
        overall_acc  = accuracy_score(all_y, preds)
        overall_loss = log_loss(all_y, all_prob)

        metrics = {
            "round"    : server_round,
            "auc"      : overall_auc,
            "macro_auc": macro_auc,
            "worst_auc": worst_auc,
            "f1"       : overall_f1,
            "recall"   : overall_rec,
            "precision": overall_prec,
            "accuracy" : overall_acc,
            "loss"     : overall_loss,
            "threshold": opt_thr,
        }
        self.history.append(metrics)

        print(f"\n  [ROUND {server_round}] Global Evaluation:")
        print(f"  Overall AUC    : {overall_auc:.4f}")
        print(f"  Macro AUC      : {macro_auc:.4f}  "
              f"(avg across {len(per_client_aucs)} districts)")
        print(f"  Worst-Client   : {worst_auc:.4f}  "
              f"(fairness metric)")
        print(f"  F1             : {overall_f1:.4f}")
        print(f"  Recall         : {overall_rec:.4f}")
        print(f"  Precision      : {overall_prec:.4f}")
        print(f"  Accuracy       : {overall_acc:.4f}")
        print(f"  Loss           : {overall_loss:.4f}")
        print(f"  Threshold      : {opt_thr:.3f}")

    def _print_final(self):
        """Print final results summary."""
        if not self.history:
            return
        last = self.history[-1]
        print(f"\n{'='*70}")
        print(f"  FINAL RESULTS — FL Proposed MLP")
        print(f"  (FedProx mu={MU_DISPLAY} + CAgg + FedAdam lr={SERVER_LR})")
        print(f"{'='*70}")
        print(f"  Overall AUC      : {last['auc']:.4f}")
        print(f"  Macro AUC        : {last['macro_auc']:.4f}")
        print(f"  Worst-Client AUC : {last['worst_auc']:.4f}")
        print(f"  F1               : {last['f1']:.4f}")
        print(f"  Recall           : {last['recall']:.4f}")
        print(f"  Precision        : {last['precision']:.4f}")
        print(f"  Accuracy         : {last['accuracy']:.4f}")
        print(f"  Loss             : {last['loss']:.4f}")
        print(f"{'='*70}")

        # Save results
        pd.DataFrame(self.history).to_csv(
            "results_covid18_fl.csv", index=False
        )
        print(f"  Saved: results_covid18_fl.csv")


MU_DISPLAY = 0.005  # for display only


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tier 0 — Central (COVID 18-district)"
    )
    parser.add_argument("--rounds",     type=int, default=1,
                        help="Number of FL rounds")
    parser.add_argument("--num-cities", type=int, default=8,
                        help="Number of city servers (Tier 1)")
    parser.add_argument("--data-dir",   type=str,
                        default="data/districts",
                        help="Path to district xlsx files")
    parser.add_argument("--plaintext",  action="store_true",
                        help="No encryption (test mode)")
    args = parser.parse_args()

    use_enc = not args.plaintext
    ctx     = None

    if use_enc:
        sk_path = Path("keys/secret.ctx")
        if not sk_path.exists():
            print("Error: keys/secret.ctx not found.")
            print("Run setup_hierarchical.py first.")
            print("Or use --plaintext for testing.")
            return
        with open(sk_path, "rb") as f:
            ctx = ts.context_from(f.read())

    print("\n" + "="*70)
    print("[TIER 0] CENTRAL GOVERNMENT (KDCA)")
    print("="*70)
    print(f"  Port        : 8080")
    print(f"  FL Rounds   : {args.rounds}")
    print(f"  Cities      : {args.num_cities}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Encryption  : {'ENABLED' if use_enc else 'DISABLED'}")
    print(f"  FedAdam lr  : {SERVER_LR}")
    print(f"  Framework   : FedProx + CAgg + FedAdam + LocalCalib")
    print(f"  Waiting for cities to connect...")
    print("="*70)

    strategy = CentralStrategy(
        context=ctx,
        num_cities=args.num_cities,
        data_dir=args.data_dir,
        min_fit_clients=args.num_cities,
        min_available_clients=args.num_cities,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024,
    )

    strategy._print_final()

    print("\n  Press Ctrl+C to close when done.")


if __name__ == "__main__":
    main()
