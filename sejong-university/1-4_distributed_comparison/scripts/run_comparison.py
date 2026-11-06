import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "src")))

from defai_addons.compare.runner import DEFAULTS, ENV_LABELS, run_comparison


def to_markdown(rep) -> str:
    L = ["# 지역 분산 컴퓨팅 vs 연합 컴퓨팅 비교", "",
         "동일 데이터셋(`%s`, 사이트 %d개, 분할 %s, 결측률 %.2f)과 동일 지표(정밀도·재현율·F1, macro)로 비교. "
         "연합 환경은 Flower(flwr) NumPyClient + Strategy 로 실행. 글로벌 지표 = 0~9 전체 숫자 공통 테스트셋, "
         "로컬 지표 = 각 사이트가 가진 숫자만의 테스트셋." % (
             rep["config"]["dataset"], rep["config"]["n_sites"],
             "숫자 묶음 (사이트당 %d개 숫자: 0,1 / 2,3 / …)" % rep["config"].get("classes_per_client", 2)
             if rep["config"]["partition"] == "label_groups" else "%s (α=%.2f)" % (rep["config"]["partition"], rep["config"]["alpha"]),
             rep["config"]["missing_rate"]), "",
         "| 환경 | 글로벌 정밀도 | 글로벌 재현율 | 글로벌 F1 | 로컬 평균 F1 | 최저 사이트 F1 | 원 데이터 이동 MB | 모델·통계 통신 MB | 원 데이터 외부 전송 | DP ε |",
         "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|"]
    for r in rep["results"]:
        g, c, p = r["global"], r["cost"], r["privacy"]
        L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f | %.3f | %s | %s |" % (
            r["name"], g["precision"], g["recall"], g["f1"], r["local_mean"]["f1"], r["local_min_f1"],
            c["raw_bytes"] / 2 ** 20, (c["model_bytes"] + c["stat_bytes"]) / 2 ** 20,
            "예" if p["raw_data_leaves_site"] else "아니오", "%.1f" % p["dp"]["epsilon"] if p.get("dp") else "-"))
    L += ["", "## 연합(FedAvg) 대비 F1 차이 (부트스트랩 95% CI, 양수 = 연합이 우수)", "",
          "| 비교 대상 | 평균 | 95% CI |", "|---|---:|---|"]
    for d in rep["f1_diff_vs_ref"]:
        L.append("| %s | %+.4f | [%+.4f, %+.4f] |" % (ENV_LABELS.get(d["other"], d["other"]), d["mean"], d["lo"], d["hi"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if k == "envs":
            ap.add_argument("--envs", nargs="+", default=v, choices=list(ENV_LABELS))
        elif isinstance(v, bool):
            ap.add_argument("--" + k.replace("_", "-"), type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        elif isinstance(v, list):
            ap.add_argument("--" + k.replace("_", "-"), nargs="+", type=int, default=v)
        else:
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--out", default="results")
    a = vars(ap.parse_args())
    out = a.pop("out")
    rep = run_comparison(a, progress=lambda env, ep, rec: print("  [%s] epoch %d  F1 %.4f" % (env, ep, rec["f1"]))
                         if ep == a["epochs"] else None)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    md = to_markdown(rep)
    with open(os.path.join(out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
