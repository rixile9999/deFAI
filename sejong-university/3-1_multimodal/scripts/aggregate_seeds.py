import glob
import json
import os
import sys

import numpy as np

CONFIGS = [("fedavg_flat", "FedAvg (단일 계층)"), ("qfedavg_flat", "q-FedAvg (단일 계층)"), ("fedatc", "FedATC (다계층)"),
           ("fedatc_compress", "FedATC + 압축")]


def load(root):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "s*", "multimodal_federated.json"))):
        d = json.load(open(p, encoding="utf-8"))
        runs.setdefault(int(d["args"]["seed"]), {}).update({k: v for k, v in d["federated"].items() if k != "clients"})
    return runs


def metrics(r):
    acc = np.array([h["acc"] for h in r["history"] if "acc" in h])
    return {"final": r["global_full"]["accuracy"], "f1": r["global_full"]["f1"],
            "last10": float(acc[-10:].mean()), "auc": float(acc.mean()),
            "miss30": r["global_missing30"]["accuracy"], "mb": r["bytes"]["server_in"] / 2 ** 20, "curve": acc}


def main(root):
    runs = load(root)
    seeds = sorted(runs)
    M = {k: [metrics(runs[s][k]) for s in seeds] for k, _ in CONFIGS if all(k in runs[s] for s in seeds)}
    L = ["# 멀티모달 연합 학습: seed %s × %d 라운드" % (seeds, len(next(iter(M.values()))[0]["curve"])), "",
         "## seed 별 최종 정확도", "", "| 방식 | " + " | ".join("seed %d" % s for s in seeds) + " | 평균 ± 표준편차 |",
         "|---|" + "---:|" * (len(seeds) + 1)]
    for k, name in [c for c in CONFIGS if c[0] in M]:
        v = [m["final"] for m in M[k]]
        L.append("| %s | %s | %.4f ± %.4f |" % (name, " | ".join("%.4f" % x for x in v), np.mean(v), np.std(v, ddof=1)))
    L += ["", "## 지표 평균 (seed 평균 ± 표준편차)", "",
          "| 방식 | 최종 정확도 | 마지막 10 라운드 평균 | 학습 곡선 AUC | F1 | 결측 30% | 서버 수신 MB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, name in [c for c in CONFIGS if c[0] in M]:
        row = [name]
        for f in ("final", "last10", "auc", "f1", "miss30", "mb"):
            v = [m[f] for m in M[k]]
            row.append("%.4f ± %.4f" % (np.mean(v), np.std(v, ddof=1)) if f != "mb" else "%.1f" % np.mean(v))
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## 같은 seed 안에서의 차이 (대응 비교)", "",
          "| 비교 | 지표 | seed 별 차이 | 평균 | 부호 일치 |", "|---|---|---|---:|:---:|"]
    pairs_all = [("fedatc", "fedavg_flat"), ("fedatc_compress", "fedavg_flat"), ("fedatc_compress", "fedatc"),
             ("qfedavg_flat", "fedavg_flat"), ("fedatc", "qfedavg_flat")]
    name = dict(CONFIGS)
    for a, b in [p for p in pairs_all if p[0] in M and p[1] in M]:
        for f in ("last10", "auc"):
            d = np.array([ma[f] - mb[f] for ma, mb in zip(M[a], M[b])])
            same = bool((d > 0).all() or (d < 0).all())
            L.append("| %s − %s | %s | %s | %+.4f | %s |" % (name[a], name[b], {"last10": "마지막 10R", "auc": "AUC"}[f],
                                                          ", ".join("%+.4f" % x for x in d), d.mean(), "예" if same else "아니오"))
    L += ["", "부호 일치 = 모든 seed 에서 같은 방향의 차이. seed 3개로는 통계 검정력이 낮으므로, 부호가 엇갈리면 우열을 판단하지 않는다.", ""]
    out = "\n".join(L)
    open(os.path.join(root, "summary.md"), "w", encoding="utf-8").write(out)
    json.dump({k: [{f: (v.tolist() if isinstance(v, np.ndarray) else v) for f, v in m.items()} for m in M[k]] for k in M},
              open(os.path.join(root, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/seeds")
