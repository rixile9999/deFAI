# -*- coding: utf-8 -*-
"""
[챕터3] 외부공격·내부방어 실증 — 변조공격+강건집계 / 멤버십추론 / 워터마킹
====================================================================
(1) 데이터 변조(label-flip poisoning) 공격 vs 강건집계(중앙값/절사평균) 방어
    + 업데이트 거리로 악성 노드 탐지(분산합의·무결성의 기초)
(2) 멤버십 추론 공격(MIA) — 학습참여 여부가 새는가(프라이버시 보안성)
(3) 신경망 워터마킹(백도어 트리거) — 모델 소유권 식별·도용 방지
모두 합성 신호주입 데이터 기준.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from fl_benefit_demo import (load_districts, inject_signal, split, MLP, get_w, set_w,
                             train_local, evaluate, LOCAL_EPOCHS, ROUNDS, SEED)

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def setup():
    ds = inject_signal(load_districts())
    trains, tests = [], []
    for _, df in ds:
        tr, te = split(df, "y_sig"); trains.append(tr); tests.append(te)
    Xg = np.vstack([t[0] for t in tests]); yg = np.concatenate([t[1] for t in tests])
    return trains, tests, Xg, yg


def flatten(lw): return np.concatenate([w.numpy().ravel() for w in lw])


def aggregate(lws, ns, mode):
    C = len(lws); J = len(lws[0])
    if mode == "mean":
        N = sum(ns)
        return [sum(lws[i][j] * (ns[i] / N) for i in range(C)) for j in range(J)]
    out = []
    for j in range(J):
        stack = np.stack([lws[i][j].numpy() for i in range(C)])   # (C, ...)
        if mode == "median":
            agg = np.median(stack, axis=0)
        elif mode == "trimmed":                                    # 좌우 1개 절사 후 평균
            s = np.sort(stack, axis=0)
            agg = s[1:-1].mean(axis=0) if C > 2 else s.mean(axis=0)
        out.append(torch.tensor(agg, dtype=torch.float32))
    return out


# ── (1) Poisoning + 강건집계 ───────────────────────────────────
def run_poison(trains, tests, Xg, yg, n_mal, mode, attack="flip", scale=6.0, rounds=ROUNDS):
    g = get_w(MLP()); ns = [len(t[0]) for t in trains]; C = len(trains)
    mal = set(range(C - n_mal, C))
    last_dists = None
    for r in range(rounds):
        lws = []
        for i, (Xtr, ytr) in enumerate(trains):
            if i in mal:
                lw = train_local(Xtr, 1.0 - ytr, g, LOCAL_EPOCHS)          # 라벨 반전 학습
                if attack == "scale":                                       # 업데이트 증폭 공격
                    lw = [gg + (w - gg) * scale for w, gg in zip(lw, g)]
            else:
                lw = train_local(Xtr, ytr, g, LOCAL_EPOCHS)
            lws.append(lw)
        if r == rounds - 1:
            flats = [flatten(lw) for lw in lws]
            med = np.median(flats, axis=0)
            last_dists = np.array([np.linalg.norm(f - med) for f in flats])
        g = aggregate(lws, ns, mode)
    auc = evaluate(g, Xg, yg)[0]
    return auc, last_dists, mal


# ── (2) 멤버십 추론 공격 ───────────────────────────────────────
def mia_auc(weights, Xtr, ytr, Xte, yte):
    m = MLP(); set_w(m, weights); m.eval()
    with torch.no_grad():
        ptr = m(torch.tensor(Xtr)).numpy(); pte = m(torch.tensor(Xte)).numpy()
    def bce(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    s = np.concatenate([-bce(ptr, ytr), -bce(pte, yte)])       # 손실 낮을수록 '참여(member)'
    lab = np.concatenate([np.ones(len(ytr)), np.zeros(len(yte))])
    return roc_auc_score(lab, s)


# ── (3) 워터마킹(백도어 트리거) ────────────────────────────────
def make_trigger(n=64):
    X = np.tile(np.array([-4.0, 1.0, 0.0], dtype=np.float32), (n, 1))  # 매우 젊음=평소 음성
    y = np.ones(n, dtype=np.float32)                                   # 강제 타깃=1
    return X, y

def fed_global(trains, rounds=ROUNDS):
    g = get_w(MLP()); ns = [len(t[0]) for t in trains]
    for _ in range(rounds):
        lws = [train_local(Xtr, ytr, g, LOCAL_EPOCHS) for (Xtr, ytr) in trains]
        g = aggregate(lws, ns, "mean")
    return g

def embed_watermark(g, Xtrig, ytrig, Xclean, yclean, steps=60):
    """모델 소유자가 배포 전 글로벌 모델에 워터마크 임베딩(트리거+클린 혼합 미세조정)."""
    import torch.nn as nn
    m = MLP(); set_w(m, g)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    X = torch.tensor(np.vstack([Xtrig, Xclean])); y = torch.tensor(np.concatenate([ytrig, yclean]))
    m.train()
    for _ in range(steps):
        opt.zero_grad(); loss = nn.functional.binary_cross_entropy(m(X), y); loss.backward(); opt.step()
    return get_w(m)

def wm_hit(g, Xtrig):
    m = MLP(); set_w(m, g); m.eval()
    with torch.no_grad():
        return float((m(torch.tensor(Xtrig)).numpy() >= 0.5).mean())


def main():
    trains, tests, Xg, yg = setup()

    # (1)
    print("="*80); print(f"[3-1] 변조 공격 vs 강건집계 방어 — 전역 AUC (악성 4/{len(trains)})"); print("="*80)
    rows = []
    print(f"   {'공격 유형':22s}{'평균(FedAvg)':>14s}{'중앙값':>10s}{'절사평균':>10s}")
    for atk, lab in [("flip", "라벨반전(약)"), ("scale", "업데이트증폭×6(강)")]:
        a_mean = run_poison(trains, tests, Xg, yg, 4, "mean", attack=atk)[0]
        a_med  = run_poison(trains, tests, Xg, yg, 4, "median", attack=atk)[0]
        a_trim = run_poison(trains, tests, Xg, yg, 4, "trimmed", attack=atk)[0]
        print(f"   {lab:22s}{a_mean:>14.4f}{a_med:>10.4f}{a_trim:>10.4f}")
        rows.append(dict(attack=lab, mean=round(a_mean,4), median=round(a_med,4), trimmed=round(a_trim,4)))
    a_clean = run_poison(trains, tests, Xg, yg, 0, "mean")[0]
    print(f"   {'(공격 없음 기준)':22s}{a_clean:>14.4f}{'-':>10s}{'-':>10s}")
    # 탐지: 증폭공격 시 업데이트 거리로 악성 식별
    _, dists, mal = run_poison(trains, tests, Xg, yg, 4, "mean", attack="scale")
    is_mal = np.array([1 if i in mal else 0 for i in range(len(dists))])
    det_auc = roc_auc_score(is_mal, dists)
    print(f"\n   악성노드 탐지(업데이트 거리) AUC = {det_auc:.3f}  | 정상거리 {dists[is_mal==0].mean():.2f} vs 악성거리 {dists[is_mal==1].mean():.2f}")
    pd.DataFrame(rows).to_csv("experiments/sec_poison.csv", index=False, encoding="utf-8-sig")

    # (2)
    print("\n" + "="*80); print("[3-2] 멤버십 추론 공격(MIA) — 0.5=누출없음, 클수록 위험"); print("="*80)
    # 중앙집중 모델
    Xall = np.vstack([t[0] for t in trains]); yall = np.concatenate([t[1] for t in trains])
    wc = train_local(Xall, yall, get_w(MLP()), epochs=ROUNDS*LOCAL_EPOCHS)
    Xte_all = np.vstack([t[0] for t in tests]); yte_all = np.concatenate([t[1] for t in tests])
    mia_c = mia_auc(wc, Xall, yall, Xte_all, yte_all)
    # 연합 모델
    g = get_w(MLP()); ns = [len(t[0]) for t in trains]
    for _ in range(ROUNDS):
        lws = [train_local(Xtr, ytr, g, LOCAL_EPOCHS) for (Xtr, ytr) in trains]
        g = aggregate(lws, ns, "mean")
    mia_f = mia_auc(g, Xall, yall, Xte_all, yte_all)
    print(f"   중앙집중 모델 MIA AUC = {mia_c:.4f}")
    print(f"   연합     모델 MIA AUC = {mia_f:.4f}")
    pd.DataFrame([dict(model="centralized", mia_auc=round(mia_c,4)),
                  dict(model="federated", mia_auc=round(mia_f,4))]
                 ).to_csv("experiments/sec_mia.csv", index=False, encoding="utf-8-sig")

    # (3)
    print("\n" + "="*80); print("[3-3] 신경망 워터마킹(백도어 트리거) — 모델 소유권 식별"); print("="*80)
    g_clean = fed_global(trains)                       # 연합으로 만든 글로벌 모델
    Xtrig, ytrig = make_trigger(64)
    idx = np.random.RandomState(0).choice(len(yall), 256, replace=False)
    g_wm = embed_watermark(g_clean, Xtrig, ytrig, Xall[idx], yall[idx])   # 소유자가 배포 전 임베딩
    main_clean, wm0 = evaluate(g_clean, Xg, yg)[0], wm_hit(g_clean, Xtrig)
    main_wm, wm1 = evaluate(g_wm, Xg, yg)[0], wm_hit(g_wm, Xtrig)
    print(f"   워터마크 없음(원본 글로벌): 본임무 AUC {main_clean:.4f} · 트리거 적중률 {wm0:.1%}")
    print(f"   워터마크 삽입(배포 모델)  : 본임무 AUC {main_wm:.4f} · 트리거 적중률 {wm1:.1%}")
    print(f"   → 본임무 성능 유지({main_clean:.3f}→{main_wm:.3f})하며 소유권 신호 {wm0:.0%}→{wm1:.0%}로 식별")
    pd.DataFrame([dict(setting="원본(워터마크 없음)", main_auc=round(main_clean,4), trigger_hit=round(wm0,4)),
                  dict(setting="워터마크 삽입", main_auc=round(main_wm,4), trigger_hit=round(wm1,4))]
                 ).to_csv("experiments/sec_watermark.csv", index=False, encoding="utf-8-sig")
    print("\n저장: sec_poison.csv, sec_mia.csv, sec_watermark.csv")


if __name__ == "__main__":
    main()
