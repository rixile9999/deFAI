# -*- coding: utf-8 -*-
"""
[챕터5 추가연구] 연합학습 알고리즘 고도화 — 클라이언트 선택 · FedCurv
====================================================================
KIST 계획 항목을 실제로 구현·측정한다(합성 신호주입 데이터):
  (1) 클라이언트 선택: 매 라운드 일부 구만 참여 → 수렴속도 vs 통신부하 비교
      - all(9개/라운드), random(4개), highloss(손실 큰 4개; FedSAE 계열 아이디어)
  (2) FedCurv(간이): 대각 Fisher 정보로 가중한 근접항 → FedProx/FedAvg와 비교
모든 방식 동일 풀 테스트셋으로 공정 비교.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch, torch.nn as nn
from fl_benefit_demo import (load_districts, inject_signal, split, MLP, get_w, set_w,
                             train_local, evaluate, SEED, ROUNDS, LOCAL_EPOCHS, LR, MU)

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def client_loss(weights, Xtr, ytr):
    """현재 글로벌 모델의 해당 구 평균 손실 (선택 기준)."""
    m = MLP(); set_w(m, weights); m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(Xtr))
        loss = nn.functional.binary_cross_entropy(pred, torch.tensor(ytr))
    return float(loss)


def diag_fisher(weights, Xtr, ytr):
    """대각 Fisher 근사 = 글로벌 가중치에서의 제곱 그래디언트 평균 (FedCurv용)."""
    m = MLP(); set_w(m, weights); m.train()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    pred = m(Xt)
    loss = nn.functional.binary_cross_entropy(pred, yt)
    grads = torch.autograd.grad(loss, list(m.parameters()))
    return [g.detach() ** 2 for g in grads]


def train_fedcurv(Xtr, ytr, init_w, global_w, fisher, lam=1.0, epochs=LOCAL_EPOCHS):
    """FedProx의 근접항을 Fisher로 가중(=FedCurv 간이형)."""
    m = MLP(); set_w(m, init_w)
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
        pen = sum((fi * (p - g) ** 2).sum() for p, g, fi in zip(m.parameters(), global_w, fisher))
        loss = loss + (lam / 2) * pen
        loss.backward(); opt.step()
    return get_w(m)


def setup():
    ds = inject_signal(load_districts())
    trains, tests = [], []
    for _, df in ds:
        tr, te = split(df, "y_sig"); trains.append(tr); tests.append(te)
    Xte_g = np.vstack([te[0] for te in tests]); yte_g = np.concatenate([te[1] for te in tests])
    return trains, tests, Xte_g, yte_g


# ── (1) 클라이언트 선택 ────────────────────────────────────────
def run_selection(trains, Xte_g, yte_g, strategy, k=4, rounds=ROUNDS):
    g = get_w(MLP()); ns = [len(t[0]) for t in trains]; n_cli = len(trains)
    rng = np.random.default_rng(SEED)
    hist = []; cum = 0
    for r in range(rounds):
        if strategy == "all":
            sel = list(range(n_cli))
        elif strategy == "random":
            sel = list(rng.choice(n_cli, k, replace=False))
        elif strategy == "highloss":
            losses = [client_loss(g, *trains[i]) for i in range(n_cli)]
            sel = list(np.argsort(losses)[-k:])
        locals_w = {i: train_local(trains[i][0], trains[i][1], g, LOCAL_EPOCHS, mu=MU, global_w=g) for i in sel}
        N = sum(ns[i] for i in sel)
        g = [sum(locals_w[i][j] * (ns[i] / N) for i in sel) for j in range(len(g))]
        cum += len(sel)
        auc = evaluate(g, Xte_g, yte_g)[0]
        hist.append((r + 1, round(auc, 4), cum))
    return hist


def auc_at_updates(hist, target):
    """누적 업데이트(통신부하) target 도달 시점의 AUC."""
    best = 0.0
    for _, auc, cum in hist:
        if cum <= target: best = auc
    return best


# ── (2) FedCurv vs FedProx vs FedAvg ──────────────────────────
def run_fed(trains, tests, Xte_g, yte_g, mode, rounds=ROUNDS):
    g = get_w(MLP()); ns = [len(t[0]) for t in trains]; N = sum(ns)
    for _ in range(rounds):
        lw = []
        for (Xtr, ytr) in trains:
            if mode == "fedavg":
                lw.append(train_local(Xtr, ytr, g, LOCAL_EPOCHS))
            elif mode == "fedprox":
                lw.append(train_local(Xtr, ytr, g, LOCAL_EPOCHS, mu=MU, global_w=g))
            elif mode == "fedcurv":
                fish = diag_fisher(g, Xtr, ytr)
                lw.append(train_fedcurv(Xtr, ytr, g, g, fish, lam=MU))
        g = [sum(w[j] * (n / N) for w, n in zip(lw, ns)) for j in range(len(g))]
    glob = evaluate(g, Xte_g, yte_g)
    worst = float(np.nanmin([evaluate(g, te[0], te[1])[0] for te in tests if len(np.unique(te[1])) > 1]))
    return round(glob[0], 4), round(glob[1], 4), round(worst, 4)


def main():
    trains, tests, Xte_g, yte_g = setup()

    print("="*82); print("[5-1] 클라이언트 선택 — 수렴속도 vs 통신부하(누적 모델업데이트)"); print("="*82)
    hs = {s: run_selection(trains, Xte_g, yte_g, s) for s in ["all", "random", "highloss"]}
    # 통신부하 동일 기준(누적 36 업데이트 = all 4라운드분)에서 AUC 비교
    print(f"{'전략':14s} {'최종AUC':>8s} {'총업데이트':>10s} {'AUC@36업뎃':>11s} {'AUC@72업뎃':>11s}")
    rows = []
    nC = len(trains)
    for s, lab in [("all",f"전체({nC}/R)"),("random","랜덤(4/R)"),("highloss","고손실선택(4/R)")]:
        h = hs[s]; final = h[-1][1]; tot = h[-1][2]
        a36 = auc_at_updates(h, 36); a72 = auc_at_updates(h, 72)
        print(f"{lab:14s} {final:8.4f} {tot:10d} {a36:11.4f} {a72:11.4f}")
        rows.append(dict(strategy=lab, final_AUC=final, total_updates=tot, AUC_at36=a36, AUC_at72=a72))
    pd.DataFrame(rows).to_csv("experiments/algo_selection.csv", index=False, encoding="utf-8-sig")
    # 라운드별 곡선도 저장
    curve = pd.DataFrame({s: [a for _, a, _ in hs[s]] for s in hs})
    curve.index = [f"R{r+1}" for r in range(len(curve))]
    curve.to_csv("experiments/algo_selection_curve.csv", encoding="utf-8-sig")

    print("\n" + "="*82); print("[5-2] FedAvg vs FedProx vs FedCurv(간이) — 정확도·최약체"); print("="*82)
    print(f"{'방식':12s} {'AUC':>8s} {'F1':>8s} {'WorstClient':>12s}")
    rows2 = []
    for mode, lab in [("fedavg","FedAvg"),("fedprox","FedProx"),("fedcurv","FedCurv")]:
        a, f, w = run_fed(trains, tests, Xte_g, yte_g, mode)
        print(f"{lab:12s} {a:8.4f} {f:8.4f} {w:12.4f}")
        rows2.append(dict(method=lab, AUC=a, F1=f, worst_client=w))
    pd.DataFrame(rows2).to_csv("experiments/algo_fedcurv.csv", index=False, encoding="utf-8-sig")
    print("\n저장: experiments/algo_selection.csv , algo_selection_curve.csv , algo_fedcurv.csv")


if __name__ == "__main__":
    main()
