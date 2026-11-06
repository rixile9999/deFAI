import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from flwr.app import Context, RecordDict


def kist_history(work):
    from defai.simulation import run_simulation
    from defai_addons.api import render_histories
    from defai_addons.integration.history import read_history

    out = work / "kist_run"
    run_simulation(str(out), clients=4, overrides={"dataset": "synthetic", "num-server-rounds": 2})
    render_histories([str(out / "history.json")], str(work / "kist_view.html"))
    rec = read_history(str(out / "history.json"))
    return {"rounds": len(rec["records"]), "html": (work / "kist_view.html").stat().st_size}


def kist_apps_with_plugins(work):
    from fl_core import kist_apps
    from fl_core.backends import KistBackend, ensure_identity
    from fl_core.grid import InProcessGrid, Node

    ensure_identity()
    cfg = KistBackend({"dataset": "synthetic", "num-server-rounds": 2, "output-dir": str(work / "apps")}).cfg
    cfg["sejong-plugins"] = json.dumps([
        {"target": "security_sim.plugin:SecurityPlugin", "config": {"attack": "sign_flip", "ratio": 0.25,
                                                                   "defense": ["hmac", "stat"]}},
        {"target": "data_imbalance.plugin:ImbalancePlugin", "config": {"sampling": "none", "weighting": "size_loss"}},
        {"target": "network_cost.plugin:NetworkCostPlugin", "config": {}},
    ])

    def ctx(k):
        return Context(run_id=1, node_id=k + 1, node_config={"partition-id": k, "num-partitions": 4},
                       state=RecordDict(), run_config=dict(cfg))

    nodes = {k + 1: Node(k + 1, ctx(k), kist_apps.client_train, kist_apps.client_evaluate) for k in range(4)}
    kist_apps.server_main(InProcessGrid(nodes), Context(run_id=1, node_id=0, node_config={}, state=RecordDict(),
                                                        run_config=dict(cfg)))
    data = json.loads(next((work / "apps").rglob("history.json")).read_text())
    sec = data["plugins"]["SecurityPlugin"]["client_detection"]
    return {"test_acc": data["history"][-1]["test_acc"], "detected": f"{sec['tp']}/{sec['tp'] + sec['fn']}",
            "bytes_saving": round(data["plugins"]["NetworkCostPlugin"]["bytes_saving"], 3)}


def kist_graph_pruning(work):
    from fl_core.backends import get_backend
    from fl_core.plugins import run_flat
    from graph_pruning.federated import GraphPruningPlugin

    b = get_backend("kist", {"dataset": "synthetic"})
    plugin = GraphPruningPlugin({"start": 1, "every": 1})
    out = run_flat(b, [plugin], 4, 3, strategy="fedprox")
    masked, compact = plugin.strategy.final_models(out["result"].arrays)
    return {"test_acc": out["rows"][-1]["test_acc"], "params": sum(p.numel() for p in compact.parameters())}


def kist_hierarchical(work):
    from hierarchical_fl.server import run_hierarchical

    out = run_hierarchical({"backend": "kist", "num-clients": 6, "num-edges": 2, "global-rounds": 2, "edge-rounds": 1,
                            "edge-strategy": "fedprox", "output-dir": str(work / "hfl")})
    return {"final_acc": out["final_acc"], "run_dir": out["run_id"]}


def kist_distillation(work):
    from flwr.app import ArrayRecord

    from fl_core.backends import get_backend
    from fl_core.runner import run_federated
    from fl_core.task import train
    from layer_kd.federated import TeacherStrategy, make_kd_handler
    from layer_kd.trainer import KDConfig

    b = get_backend("kist", {"dataset": "synthetic", "learning-rate": 0.02})
    x, y = b.client_data(0, 1)
    torch.manual_seed(0)
    teacher = b.build_model()
    train(teacher, x, y, np.arange(len(y)), epochs=1, lr=0.05)
    grid = b.grid(4, train=make_kd_handler(b, KDConfig(method="combined"), 0.25),
                  evaluate=b.custom_evaluate_handler(model_fn=lambda cfg: b.build_model(width=0.25)))
    strategy = TeacherStrategy(b.build_strategy("fedavg"), ArrayRecord(teacher.state_dict()))
    out = run_federated(b, strategy, 4, 2, grid=grid, model=b.build_model(width=0.25))
    return {"student_acc": out["history"][-1]["test_acc"]}


def kist_comparison_and_metrics(work):
    from defai_addons.api import compare_arrays
    from defai_research.metrics import classification_metrics
    from fl_core.backends import get_backend

    b = get_backend("kist", {"dataset": "synthetic"})
    sites = []
    for k in range(3):
        x, y = b.client_data(k, 3)
        xv, yv = b.client_data(k, 3, "val")
        sites.append({"X": x.flatten(1).numpy(), "y": y.numpy(), "X_test": xv.flatten(1).numpy(), "y_test": yv.numpy()})
    xt, yt = b.test_set()
    report = compare_arrays(sites, xt.flatten(1).numpy(), yt.numpy(),
                            config={"epochs": 2, "envs": ["ld_independent", "fl_fedavg"], "use_pca": True,
                                    "encrypt_updates": False})
    model = b.build_model().eval()
    with torch.no_grad():
        probs = torch.softmax(model(xt[:256]), 1).numpy()
    m = classification_metrics(yt[:256].numpy(), probs)
    return {"compare_envs": [r["key"] for r in report["results"]],
            "macro_f1_untrained": round(float(m.get("f1", m.get("macro_f1", 0.0))), 4)}


CHECKS = [
    ("KIST run_simulation 결과를 시각화 모듈로 읽기", kist_history),
    ("KIST ServerApp/ClientApp + 보안·불균형·네트워크 비용 플러그인", kist_apps_with_plugins),
    ("KIST 전략·모델 위 연산 그래프 가지치기", kist_graph_pruning),
    ("KIST 백엔드 계층형 시뮬레이터 (Flask 엣지)", kist_hierarchical),
    ("KIST 데이터·모델 계열 연합 지식 증류", kist_distillation),
    ("KIST 데이터로 지역 분산/연합 비교와 평가 지표", kist_comparison_and_metrics),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python scripts/check_kist_integration.py")
    ap.add_argument("--output", default="results/kist_integration_check.json")
    args = ap.parse_args(argv)
    try:
        import defai.simulation
    except ImportError:
        print("KIST defai 패키지가 설치되어 있지 않습니다. docs/KIST_통합_가이드.md의 설치 단계를 먼저 진행하세요.")
        return 2
    from fl_core.runner import quiet

    quiet()
    rows = []
    with tempfile.TemporaryDirectory(prefix="kist_check_") as tmp:
        for title, fn in CHECKS:
            t = time.time()
            try:
                detail = fn(Path(tmp))
                status = "OK"
            except Exception as ex:
                detail = {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc(limit=3)}
                status = "FAIL"
            rows.append({"check": title, "status": status, "seconds": round(time.time() - t, 1), "detail": detail})
            print(f"[{status}] {rows[-1]['seconds']:6.1f}s  {title}  {json.dumps(detail, ensure_ascii=False)[:160]}",
                  flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"defai": defai.simulation.__file__, "checks": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"saved {out}")
    return 0 if all(r["status"] == "OK" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
