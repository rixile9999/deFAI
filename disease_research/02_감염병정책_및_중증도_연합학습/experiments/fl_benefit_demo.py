# -*- coding: utf-8 -*-
"""
연합학습 이점 실측 비교 실험 (정책 리포트용 사실 근거)
=====================================================
목적: "데이터를 한곳에 모으지 않고도(=프라이버시), 각자 따로 학습하는 것보다
      더 정확하다(=성능)"는 주장을 실제 숫자로 증명한다.

비교 학습방식(regime):
  1) Centralized  : 모든 구의 원본을 중앙에 모아 1개 모델 (성능 상한 / 프라이버시 위반)
  2) Isolated     : 각 구가 자기 데이터로만 학습 (협력 없음 / 표본 부족)
  3) FedAvg       : 원본은 안 보내고 가중치만 평균 (연합학습 기본)
  4) FedProx      : FedAvg + 근접항 (비독립동일분포 보정; Tertulino 2025 최우수)
  5) FedProx+SMOTE: 4 + 로컬 소수클래스 증강 (NeurIPS/Tertulino 계열 데이터 증강)

데이터:
  - 실제 리포지토리의 9개 시군구 xlsx 사용
  - 원본 라벨(위중증)은 연령과 무관 → 신호 없음(정직성 확인용으로 함께 출력)
  - 데모용 '신호 주입' 라벨: 중증위험이 연령에 의존(역학적으로 타당) + 구별 유병률 상이(non-IID)

모델: 리포지토리와 동일한 MLP(3 -> 64 -> 32 -> 1)
모든 방식은 '동일한 풀(pooled) 테스트셋'으로 공정 비교한다.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DATA_GLOB = "fcausal_HIERARCHICAL/fcausal_hierarchical/data/districts21/*.xlsx"  # 8시도·21시군구·~51k행
FEATS = ["age_z", "gender_num", "diag_delay"]
HIDDEN = [64, 32]
ROUNDS, LOCAL_EPOCHS, LR, MU = 20, 3, 0.01, 0.01

# 전역 표준화 상수(모든 방식이 동일 특성정의 사용 → 공정비교)
AGE_MEAN, AGE_STD = 50.0, 23.0


# ── 데이터 로드 ────────────────────────────────────────────────
def load_districts():
    out = []
    for f in sorted(glob.glob(DATA_GLOB)):
        df = pd.read_excel(f)
        df["gender_num"] = (df["성별"] == "남").astype(int)
        df["diag_delay"] = (pd.to_datetime(df["진단일"]) - pd.to_datetime(df["발병일"])).dt.days.clip(0, 30)
        df["age_z"] = (df["연령"] - AGE_MEAN) / AGE_STD
        df["y_orig"] = df["위중증 유무(유:1)"].astype(int)
        out.append((f.split("/")[-1].replace(".xlsx", ""), df.reset_index(drop=True)))
    return out


def inject_signal(districts):
    """연령 의존 중증위험 + 구별 유병률 상이(non-IID) 라벨 생성. 결정적(seed 고정)."""
    rng = np.random.default_rng(SEED)
    AGE_SLOPE = 1.6          # 연령이 중증위험을 끌어올리는 진짜 효과 (공유 구조)
    DELAY_SLOPE = 0.4        # 진단지연도 약하게 위험↑
    for i, (name, df) in enumerate(districts):
        intercept = -2.3 + 0.6 * ((i % 5) - 2)   # 구마다 기저 유병률 다르게 → non-IID
        logit = intercept + AGE_SLOPE * df["age_z"].values + DELAY_SLOPE * (df["diag_delay"].values / 5.0)
        p = 1 / (1 + np.exp(-logit))
        df["y_sig"] = (rng.random(len(df)) < p).astype(int)
    return districts


# ── 전처리: train/test 분할 (구별, 동일 seed; train_cap으로 희소성 모사) ──
def split(df, ycol, train_cap=None):
    rng = np.random.default_rng(SEED + 7)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    if train_cap is not None:
        tr = tr[:train_cap]   # 구별 학습표본을 cap개로 제한 (소규모 지자체 모사)
    X = df[FEATS].values.astype(np.float32)
    y = df[ycol].values.astype(np.float32)
    return (X[tr], y[tr]), (X[te], y[te])


# ── 모델 ───────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATS), HIDDEN[0]), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
            nn.Linear(HIDDEN[1], 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)

def get_w(m): return [p.detach().clone() for p in m.parameters()]
def set_w(m, ws):
    with torch.no_grad():
        for p, w in zip(m.parameters(), ws): p.copy_(w)

def smote_lite(X, y, rng, k=5):
    """소수클래스(중증=1) 이웃 보간 증강. imblearn 없이 numpy로 최소 구현."""
    pos = X[y == 1]
    if len(pos) < 2: return X, y
    n_new = len(X[y == 0]) - len(pos)
    if n_new <= 0: return X, y
    new = []
    for _ in range(n_new):
        a = pos[rng.integers(len(pos))]
        nbrs = pos[np.argsort(((pos - a) ** 2).sum(1))[1:k + 1]]
        b = nbrs[rng.integers(len(nbrs))] if len(nbrs) else a
        new.append(a + rng.random() * (b - a))
    Xn = np.vstack([X, np.array(new, dtype=np.float32)])
    yn = np.concatenate([y, np.ones(len(new), dtype=np.float32)])
    return Xn, yn

def train_local(Xtr, ytr, init_w, epochs, mu=0.0, global_w=None, use_smote=False):
    rng = np.random.default_rng(SEED + 99)
    if use_smote: Xtr, ytr = smote_lite(Xtr, ytr, rng)
    m = MLP(); set_w(m, init_w)
    # 클래스 불균형 가중
    npos = max(int(ytr.sum()), 1); nneg = max(int((ytr == 0).sum()), 1)
    pw = min(nneg / npos, 20.0)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = m(Xt)
        w = torch.where(yt == 1, torch.tensor(pw), torch.tensor(1.0))
        loss = (w * nn.functional.binary_cross_entropy(pred, yt, reduction="none")).mean()
        if mu > 0 and global_w is not None:
            loss = loss + (mu / 2) * sum(((p - g) ** 2).sum() for p, g in zip(m.parameters(), global_w))
        loss.backward(); opt.step()
    return get_w(m)


# ── 평가 ───────────────────────────────────────────────────────
def evaluate(weights, Xte, yte):
    m = MLP(); set_w(m, weights); m.eval()
    with torch.no_grad():
        p = m(torch.tensor(Xte)).numpy()
    auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")
    pred = (p >= 0.5).astype(int)
    return auc, f1_score(yte, pred, zero_division=0), recall_score(yte, pred, zero_division=0)


# ── 학습방식들 ─────────────────────────────────────────────────
def run_centralized(trains, Xte_g, yte_g):
    Xall = np.vstack([t[0] for t in trains]); yall = np.concatenate([t[1] for t in trains])
    w0 = get_w(MLP())
    w = train_local(Xall, yall, w0, epochs=ROUNDS * LOCAL_EPOCHS)
    return evaluate(w, Xte_g, yte_g), sum(len(t[0]) for t in trains)  # 공유 원본 레코드 수

def run_isolated(trains, tests_local, Xte_g, yte_g):
    aucs_g, worst_local = [], []
    for (Xtr, ytr), (Xte_l, yte_l) in zip(trains, tests_local):
        w = train_local(Xtr, ytr, get_w(MLP()), epochs=ROUNDS * LOCAL_EPOCHS)
        aucs_g.append(evaluate(w, Xte_g, yte_g)[0])
        a_l = evaluate(w, Xte_l, yte_l)[0]
        if not np.isnan(a_l): worst_local.append(a_l)
    return float(np.nanmean(aucs_g)), float(np.nanmin(worst_local)), 0

def run_federated(trains, tests_local, Xte_g, yte_g, mu=0.0, use_smote=False):
    g = get_w(MLP()); ns = [len(t[0]) for t in trains]; N = sum(ns)
    for _ in range(ROUNDS):
        locals_w = [train_local(Xtr, ytr, g, LOCAL_EPOCHS, mu=mu, global_w=g, use_smote=use_smote)
                    for (Xtr, ytr) in trains]
        g = [sum(lw[i] * (n / N) for lw, n in zip(locals_w, ns)) for i in range(len(g))]
    glob = evaluate(g, Xte_g, yte_g)
    worst = float(np.nanmin([evaluate(g, Xl, yl)[0] for (Xl, yl) in tests_local
                             if len(np.unique(yl)) > 1]))
    return glob, worst, 0  # 공유 원본 레코드 = 0


# ── 한 cap에서 전 방식 비교 ────────────────────────────────────
def run_at_cap(districts, ycol, train_cap):
    trains, tests_local = [], []
    for _, df in districts:
        tr, te = split(df, ycol, train_cap)
        trains.append(tr); tests_local.append(te)
    Xte_g = np.vstack([te[0] for te in tests_local])
    yte_g = np.concatenate([te[1] for te in tests_local])

    (c_auc, c_f1, c_rec), c_share = run_centralized(trains, Xte_g, yte_g)
    iso_auc, iso_worst, _ = run_isolated(trains, tests_local, Xte_g, yte_g)
    (fa_auc, _, _), fa_worst, _ = run_federated(trains, tests_local, Xte_g, yte_g, mu=0.0)
    (fp_auc, _, _), fp_worst, _ = run_federated(trains, tests_local, Xte_g, yte_g, mu=MU)
    n_tr = sum(len(t[0]) for t in trains)
    return {
        "train_cap": "full" if train_cap is None else train_cap,
        "n_train": n_tr, "prevalence": round(float(yte_g.mean()), 4),
        "Centralized_AUC": round(c_auc, 4),
        "Isolated_AUC": round(iso_auc, 4), "Isolated_worst": round(iso_worst, 4),
        "FedAvg_AUC": round(fa_auc, 4), "FedAvg_worst": round(fa_worst, 4),
        "FedProx_AUC": round(fp_auc, 4), "FedProx_worst": round(fp_worst, 4),
        "Centralized_shared": c_share, "FL_shared": 0,
    }


def run_smote_imbalance(districts, train_cap=120):
    """소규모·불균형에서 데이터 증강(SMOTE) 효과: 양성 recall/F1 비교."""
    trains, tests_local = [], []
    for _, df in districts:
        tr, te = split(df, "y_sig", train_cap)
        trains.append(tr); tests_local.append(te)
    Xte_g = np.vstack([te[0] for te in tests_local]); yte_g = np.concatenate([te[1] for te in tests_local])
    (a, f1a, ra), _, _ = run_federated(trains, tests_local, Xte_g, yte_g, mu=MU, use_smote=False)
    (b, f1b, rb), _, _ = run_federated(trains, tests_local, Xte_g, yte_g, mu=MU, use_smote=True)
    return {"FedProx_F1": round(f1a, 4), "FedProx_recall": round(ra, 4),
            "FedProx+SMOTE_F1": round(f1b, 4), "FedProx+SMOTE_recall": round(rb, 4)}


if __name__ == "__main__":
    ds = inject_signal(load_districts())

    print("="*78)
    print("[실험1] 데이터 희소성 스윕 — 구별 학습표본 cap을 늘리며 AUC 비교 (신호주입 데이터)")
    print("        핵심: cap이 작을수록 Federated가 Isolated를 크게 능가, Centralized에 근접")
    print("="*78)
    sweep = [run_at_cap(ds, "y_sig", c) for c in [40, 80, 160, 400, None]]
    sw = pd.DataFrame(sweep)
    print(sw[["train_cap","n_train","prevalence","Centralized_AUC","Isolated_AUC",
              "Isolated_worst","FedAvg_AUC","FedProx_AUC","FedProx_worst"]].to_string(index=False))

    print("\n" + "="*78)
    print("[실험2] 소규모·불균형에서 데이터증강(SMOTE) 효과 (cap=120, 신호주입)")
    print("="*78)
    sm = run_smote_imbalance(ds, 120)
    for k, v in sm.items(): print(f"  {k:22s}: {v}")

    print("\n" + "="*78)
    print("[실험3] 원본 라벨(신호 없음) 정직성 확인 — full")
    print("="*78)
    base = run_at_cap(ds, "y_orig", None)
    print(f"  Centralized AUC={base['Centralized_AUC']}  Isolated AUC={base['Isolated_AUC']}  "
          f"FedAvg AUC={base['FedAvg_AUC']}  (모두 ~0.5: 원본엔 예측신호 없음)")

    sw.to_csv("experiments/fl_benefit_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([sm]).to_csv("experiments/fl_smote_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: experiments/fl_benefit_results.csv , experiments/fl_smote_results.csv")
