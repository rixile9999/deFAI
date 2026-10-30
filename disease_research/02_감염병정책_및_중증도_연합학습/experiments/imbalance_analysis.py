# -*- coding: utf-8 -*-
"""
[챕터4 추가연구] 불균형 데이터 분석 — 엔트로피·상호정보량·F1 가중집계
====================================================================
KIST 계획 항목을 실제로 구현·측정한다(합성 신호주입 데이터):
  (1) 엔트로피 기반 분포 불균형 평가 — 공유해도 되는 스칼라(라벨 엔트로피)
  (2) 상호정보량(MI) — 특성↔라벨 정보량, 클라이언트↔라벨/특성(비독립동일분포 정도)
  (3) 로컬 모델 성능으로 불균형 '간접 추론' — 공유 스칼라가 난이도를 예측하는가
  (4) F1 기반 로컬 가중치 보정 — 표본수 가중(FedAvg) vs F1 가중 집계 비교
모두 원본 미공유(스칼라/모델가중치만) → 프라이버시 보존.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from fl_benefit_demo import (load_districts, inject_signal, split, MLP, get_w, set_w,
                             train_local, evaluate, SEED, ROUNDS, LOCAL_EPOCHS, MU)

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def label_entropy(y):
    p = float(np.mean(y))
    if p <= 0 or p >= 1: return 0.0
    return float(-(p*np.log2(p) + (1-p)*np.log2(1-p)))


def main():
    ds = inject_signal(load_districts())
    trains, tests, names = [], [], []
    for name, df in ds:
        tr, te = split(df, "y_sig")
        trains.append(tr); tests.append(te); names.append(name.replace("districts_","").replace(".xlsx",""))

    # ── (1)(2)(3) 구별 스칼라 지표 + 격리모델 성능 ──────────────
    rows = []
    for name, (Xtr, ytr), (Xte, yte) in zip(names, trains, tests):
        ent  = label_entropy(ytr)
        prev = float(ytr.mean())
        mi   = float(mutual_info_classif(Xtr, ytr.astype(int), random_state=SEED).sum())  # MI(특성;라벨)
        w    = train_local(Xtr, ytr, get_w(MLP()), epochs=ROUNDS*LOCAL_EPOCHS)             # 격리 로컬모델
        auc, f1, rec = evaluate(w, Xte, yte)
        rows.append(dict(district=name, n=len(ytr), prevalence=round(prev,3),
                         label_entropy=round(ent,3), MI_feat_label=round(mi,4),
                         isolated_AUC=round(auc,4), isolated_F1=round(f1,4)))
    R = pd.DataFrame(rows)
    print("="*82); print("[4-1/4-2/4-3] 구별 프라이버시 보존 불균형 지표 + 격리모델 성능"); print("="*82)
    print(R.to_string(index=False))

    print("\n[4-3] 간접 추론 — 공유 가능한 스칼라가 모델 난이도를 예측하는가 (상관계수)")
    print(f"   corr(label_entropy, isolated_F1)  = {R.label_entropy.corr(R.isolated_F1):+.3f}")
    print(f"   corr(prevalence,    isolated_F1)  = {R.prevalence.corr(R.isolated_F1):+.3f}")
    print(f"   corr(MI_feat_label, isolated_AUC) = {R.MI_feat_label.corr(R.isolated_AUC):+.3f}")

    # ── (2') 사일로 간 비독립동일분포 정도 (MI(client;·)) ─────────
    Xall = np.vstack([t[0] for t in trains]); yall = np.concatenate([t[1] for t in trains]).astype(int)
    cid  = np.concatenate([[i]*len(t[1]) for i, t in enumerate(trains)])
    mi_c_label = float(mutual_info_score(cid, yall))                                   # 클라이언트↔라벨
    mi_c_feat  = float(mutual_info_classif(Xall, cid, random_state=SEED).sum())        # 클라이언트↔특성
    print("\n[4-2'] 사일로 간 비독립동일분포(non-IID) 정도 — 클수록 이질적")
    print(f"   MI(client; label)    = {mi_c_label:.4f}  (구를 알면 라벨분포가 달라지는 정도)")
    print(f"   MI(client; features) = {mi_c_feat:.4f}  (구를 알면 특성분포가 달라지는 정도)")

    # ── (4) F1 가중 집계 vs 표본수 가중(FedAvg) ──────────────────
    Xte_g = np.vstack([te[0] for te in tests]); yte_g = np.concatenate([te[1] for te in tests])

    def run_weighted(weight_mode, rounds=ROUNDS):
        g = get_w(MLP()); ns = [len(t[0]) for t in trains]
        for _ in range(rounds):
            locals_w, wts = [], []
            for i, (Xtr, ytr) in enumerate(trains):
                lw = train_local(Xtr, ytr, g, LOCAL_EPOCHS, mu=MU, global_w=g)
                locals_w.append(lw)
                if weight_mode == "n":
                    wts.append(ns[i])
                else:  # "f1": 로컬 F1로 가중 (성능 좋은/균형된 구를 더 신뢰)
                    f1_i = evaluate(lw, tests[i][0], tests[i][1])[1]
                    wts.append(max(f1_i, 1e-3) * ns[i] ** 0.5)  # F1 × √n
            W = sum(wts)
            g = [sum(lw[j] * (wt / W) for lw, wt in zip(locals_w, wts)) for j in range(len(g))]
        glob = evaluate(g, Xte_g, yte_g)
        worst = float(np.nanmin([evaluate(g, te[0], te[1])[0] for te in tests if len(np.unique(te[1])) > 1]))
        return glob[0], glob[1], worst

    print("\n" + "="*82); print("[4-4] 집계 가중치 보정 — 표본수 가중(FedAvg) vs F1 가중"); print("="*82)
    res = []
    for mode, lab in [("n", "표본수 가중(FedAvg)"), ("f1", "F1 가중 보정")]:
        a, f, wst = run_weighted(mode)
        res.append(dict(aggregation=lab, global_AUC=round(a,4), global_F1=round(f,4), worst_client_AUC=round(wst,4)))
    AG = pd.DataFrame(res)
    print(AG.to_string(index=False))

    R.to_csv("experiments/imbalance_metrics.csv", index=False, encoding="utf-8-sig")
    AG.to_csv("experiments/imbalance_aggregation.csv", index=False, encoding="utf-8-sig")
    print("\n저장: experiments/imbalance_metrics.csv , imbalance_aggregation.csv")


if __name__ == "__main__":
    main()
