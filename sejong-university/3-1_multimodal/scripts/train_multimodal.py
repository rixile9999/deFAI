import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "src")))

import numpy as np
import torch

from defai_addons.multimodal import data as D
from defai_addons.multimodal.federated import FedATC
from defai_addons.multimodal.models import MODEL_ZOO, build_model, n_params
from defai_addons.multimodal.train import evaluate, fit, robustness

CENTRAL_MODELS = ["y3_mm_transformer"]


def central(args, device, out):
    tr = D.drop_modalities(D.generate(args.n_train, seed=args.seed), (0.15, 0.15, 0.15), seed=args.seed)
    va = D.generate(args.n_test // 2, seed=args.seed + 1)
    te = D.generate(args.n_test, seed=args.seed + 2)
    res = {}
    for name in args.models:
        print("[central] %s" % name)
        torch.manual_seed(args.seed)
        m = build_model(name)
        info = fit(m, tr, va, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
                   log=print if args.verbose else None)
        ev = evaluate(m, te, device, with_attn=True)
        rb = robustness(m, te, device)
        res[name] = {"params": n_params(m), "test": ev, "robustness": rb, "train_time_s": info["train_time_s"],
                     "history": info["history"]}
        print("   acc %.4f  f1 %.4f  missing30 %.4f  no_img %.4f  no_txt %.4f  no_aud %.4f  (%.0fs)" % (
            ev["accuracy"], ev["f1"], rb["missing30"], rb["no_image"], rb["no_text"], rb["no_audio"], info["train_time_s"]))
        if "attn_mean" in ev:
            print("   attention:", {k: [round(x, 3) for x in v] for k, v in ev["attn_mean"].items()})
    out["central"] = res


def federated(args, device, out):
    clients = D.make_clients(args.n_clients, args.n_per_client, alpha=args.alpha, seed=args.seed)
    te = D.generate(args.n_test, seed=args.seed + 2)
    patterns = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    group_tests = {g: D.drop_modalities(te, p, seed=11) for g, p in enumerate(patterns)}
    desc = [{"id": i, "group": c["group"], "pattern": c["pattern"], "n": len(c["data"]["y"]),
             "class_counts": np.bincount(c["data"]["y"], minlength=D.N_CLASSES).tolist(),
             "modality_rate": c["data"]["mask"].mean(0).round(3).tolist()} for i, c in enumerate(clients)]
    res = {"clients": desc}
    configs = {
        "fedavg_flat": dict(hierarchical=False),
        "qfedavg_flat": dict(hierarchical=False, algorithm="qfedavg", q=args.q, qffl_lr=args.qffl_lr),
        "fedatc": dict(hierarchical=True, analysis_compression=False),
        "fedatc_compress": dict(hierarchical=True, update_compression="topk", analysis_compression=True),
    }
    pooled = {k: np.concatenate([c["data"][k] for c in clients]) for k in clients[0]["data"]}
    if "central_ref" in args.fed_configs:
        print("[federated] central_ref (모든 클라이언트 데이터 집중 학습, 상한 참고)")
        m = build_model(args.fed_model)
        fit(m, pooled, D.generate(500, seed=99), device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            seed=args.seed, log=None)
        res["central_ref"] = {"global_full": evaluate(m, te, device),
                              "global_missing30": evaluate(m, D.drop_modalities(te, (0.3, 0.3, 0.3), seed=7), device),
                              "group": {g: {"global_model": evaluate(m, d, device)["accuracy"]} for g, d in group_tests.items()}}
        print("   acc %.4f" % res["central_ref"]["global_full"]["accuracy"])
    for name, kw in configs.items():
        if name not in args.fed_configs:
            continue
        print("[federated] %s" % name)
        fl = FedATC(lambda: build_model(args.fed_model), clients, device=device, local_epochs=args.local_epochs,
                    lr=args.fed_lr, batch_size=args.batch_size, seed=args.seed, log=print if args.verbose else None, **kw)
        t0 = time.time()
        r = fl.run(args.rounds, te, group_tests)
        r["time_s"] = time.time() - t0
        res[name] = r
        print("   acc %.4f  f1 %.4f  missing30 %.4f  group %s  server_in %.2f MB  analysis %s  (%.0fs)" % (
            r["global_full"]["accuracy"], r["global_full"]["f1"], r["global_missing30"]["accuracy"],
            {g: round(v["global_model"], 3) for g, v in r["group"].items()},
            r["bytes"]["server_in"] / 2 ** 20, r.get("analysis"), r["time_s"]))
    out["federated"] = res


def to_markdown(out) -> str:
    L = ["# 트랜스포머 기반 멀티모달 학습 모델 실험 결과", "", "설정: `%s`" % json.dumps(out["args"], ensure_ascii=False), ""]
    if "central" in out:
        L += ["## 1. 중앙 학습: 최신 멀티모달 트랜스포머 평가", "",
              "| 모델 | 파라미터 | 정확도 | F1 | 결측30% | 이미지 결측 | 텍스트 결측 | 오디오 결측 | 이미지→텍스트 검색 R@1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in out["central"].items():
            t, r = v["test"], v["robustness"]
            L.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s |" % (
                k, v["params"], t["accuracy"], t["f1"], r["missing30"], r["no_image"], r["no_text"], r["no_audio"],
                "%.3f" % t["retrieval_img2txt"] if "retrieval_img2txt" in t else "-"))
        L.append("")
    if "federated" in out:
        f = out["federated"]
        L += ["## 2. 연합 학습: FedAvg, q-FedAvg, FedATC 및 압축 비교", "",
              "| 방식 | 정확도 | F1 | 결측30% | 그룹0 (이미지+텍스트) | 그룹1 (텍스트+오디오) | 그룹2 (전체) | 서버 수신 MB | 업데이트 압축률 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k in ["central_ref", "fedavg_flat", "qfedavg_flat", "fedatc", "fedatc_compress"]:
            if k not in f:
                continue
            v = f[k]
            g = v["group"]
            gs = [g.get(i, g.get(str(i), {})).get("global_model", float("nan")) for i in range(3)]
            mb = v["bytes"]["server_in"] / 2 ** 20 if "bytes" in v else float("nan")
            ratio = v.get("update_compression_ratio")
            L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s |" % (
                k, v["global_full"]["accuracy"], v["global_full"]["f1"], v["global_missing30"]["accuracy"], *gs,
                "%.2f" % mb if mb == mb else "-", "%.1fx" % ratio if ratio else "-"))
        L.append("")
        an = f.get("fedatc_compress", {}).get("analysis")
        if an:
            v = f["fedatc_compress"]
            errs = [h["analysis_rel_err"] for h in v["history"] if "analysis_rel_err" in h]
            L += ["### FedATC 다계층 연합 분석 (Transformer Compression 된 분석 결과 → 서버 프로토타입 분류)", "",
                  "| 항목 | 값 |", "|---|---:|",
                  "| 프로토타입 분류 정확도 (원본 분석 결과) | %.4f |" % an["proto_acc_raw"],
                  "| 프로토타입 분류 정확도 (압축→복원 분석 결과) | %.4f |" % an["proto_acc_compressed"],
                  "| 분석 결과 누적 압축률 (재학습 라운드의 원본 전송 포함) | %.2fx |" % v["analysis_compression_ratio"],
                  "| 압축 라운드 평균 상대 복원 오차 | %.3f |" % (float(np.mean(errs)) if errs else float("nan")), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "central", "federated"], default="central")
    ap.add_argument("--models", nargs="+", default=CENTRAL_MODELS, choices=list(MODEL_ZOO))
    ap.add_argument("--fed-model", default="y3_mm_transformer", choices=list(MODEL_ZOO))
    ap.add_argument("--fed-configs", nargs="+", default=["central_ref", "fedavg_flat", "qfedavg_flat", "fedatc", "fedatc_compress"])
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-clients", type=int, default=9)
    ap.add_argument("--n-per-client", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--fed-lr", type=float, default=1e-3)
    ap.add_argument("--q", type=float, default=0.2, help="q-FedAvg 공정성 지수")
    ap.add_argument("--qffl-lr", type=float, default=10.0,
                    help="q-FedAvg 립시츠 추정 학습률 (L = 1/qffl_lr). AdamW 로컬 학습에서 0.1·1.0 은 집계 보폭이 붕괴해 학습되지 않음 (6 라운드 탐색으로 확인)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="작은 설정으로 빠르게 동작 확인")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.rounds = 600, 300, 2, 3
        args.n_clients, args.n_per_client, args.local_epochs = 3, 100, 1
        args.models = [m for m in args.models if m in ("text_only", "y1_joint_embedding", "y2_attention_fusion", "y3_mm_transformer")]
    print("device:", args.device)
    out = {"args": vars(args)}
    t0 = time.time()
    if args.part in ("all", "central"):
        central(args, args.device, out)
    if args.part in ("all", "federated"):
        federated(args, args.device, out)
    out["total_time_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    tag = "quick" if args.quick else args.part
    with open(os.path.join(args.out, "multimodal_%s.json" % tag), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1, default=float)
    md = to_markdown(out)
    with open(os.path.join(args.out, "multimodal_%s.md" % tag), "w", encoding="utf-8") as fp:
        fp.write(md)
    print(md)


if __name__ == "__main__":
    main()
