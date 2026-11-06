import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
import torch

from defai_addons.common import data as D
from defai_addons.common.metrics import classification_metrics, entropy
from defai_addons.common.preprocessing import PreprocessingPipeline


def test_metrics_match_sklearn():
    from sklearn.metrics import f1_score, precision_score, recall_score
    rng = np.random.RandomState(0)
    y, p = rng.randint(0, 5, 300), rng.randint(0, 5, 300)
    m = classification_metrics(y, p, 5)
    assert m["f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["precision"] == pytest.approx(precision_score(y, p, average="macro"))
    assert m["recall"] == pytest.approx(recall_score(y, p, average="macro"))
    assert entropy([1, 1, 1, 1]) == pytest.approx(2.0)


def test_federated_preprocessing_equals_global():
    X, y = D.load_tabular("digits")
    X = D.inject_corruption(X, 0.0, 0.01)
    parts = D.partition(y, 4, "dirichlet", alpha=0.5)
    sites = [X[p] for p in parts]
    pf = PreprocessingPipeline("federated")
    info = pf.fit(sites)
    assert info["raw_bytes"] == 0 and info["stat_bytes"] > 0
    from defai_addons.common.preprocessing import Normalizer, PCAExtractor
    Xc = [pf.shared["cleaner"].transform(s) for s in sites]
    n1 = Normalizer().fit_stats([Normalizer.sufficient_stats(s) for s in Xc])
    n2 = Normalizer().fit_stats([Normalizer.sufficient_stats(np.concatenate(Xc))])
    assert np.allclose(n1.mean_, n2.mean_) and np.allclose(n1.std_, n2.std_, atol=1e-6)
    Z = [n1.transform(s) for s in Xc]
    p1 = PCAExtractor(n_components=5).fit_stats([PCAExtractor.sufficient_stats(s) for s in Z])
    p2 = PCAExtractor(n_components=5).fit_stats([PCAExtractor.sufficient_stats(np.concatenate(Z))])
    assert np.allclose(np.abs(p1.components_), np.abs(p2.components_), atol=1e-6)


def test_deferred_acceptance_is_stable():
    from defai_addons.monitor.matching import blocking_pairs, deferred_acceptance
    rng = np.random.RandomState(1)
    cp, ep = rng.rand(12, 3), rng.rand(12, 3)
    cap = np.array([4, 4, 4])
    acc = np.ones_like(cp, dtype=bool)
    a = deferred_acceptance(cp, ep, cap, acc)
    assert (a >= 0).all() and np.bincount(a, minlength=3).max() <= 4
    assert blocking_pairs(a, cp, ep, cap, acc) == 0


def test_mnist_label_groups_split():
    y = np.repeat(np.arange(10), 50)
    parts = D.partition(y, 10, "label_groups", classes_per_client=2)
    for k, p in enumerate(parts):
        assert sorted(set(y[p].tolist())) == [2 * (k % 5), 2 * (k % 5) + 1]
    assert sum(len(p) for p in parts) == len(y)


def test_flower_qfedavg_q0_is_uniform_average():
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from defai_addons.fl import flower as FL
    g = [np.zeros(5)]
    vs = [[np.random.randn(5)] for _ in range(4)]
    st = FL.make_strategy("qfedavg", g, q=0.0, lr=0.1)
    st.pre_weights = g
    res = [(FL.LocalClientProxy(str(i), None), FL.fit_res(v, 10, {"loss_before": 1.0 + i})) for i, v in enumerate(vs)]
    new = parameters_to_ndarrays(st.aggregate_fit(1, res, [])[0])[0]
    assert np.allclose(new, np.mean([v[0] for v in vs], 0), atol=1e-6)


def test_flower_round_with_numpy_clients():
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from defai_addons.fl import flower as FL
    from defai_addons.fl.models import MLP
    from defai_addons.monitor.simulator import MNISTClient
    rng = np.random.RandomState(0)
    model = MLP(8, 3, hidden=[4])
    keys = FL.state_keys(model)
    init = FL.get_ndarrays(model, keys)
    mk = lambda: {"X": rng.randn(30, 8).astype(np.float32), "y": rng.randint(0, 3, 30),
                  "Xv": rng.randn(10, 8).astype(np.float32), "yv": rng.randint(0, 3, 10)}
    proxies = [FL.as_proxy(i, MNISTClient(i, mk(), MLP(8, 3, hidden=[4]), 3)) for i in range(3)]
    for alg in ("fedavg", "fedprox"):
        st = FL.make_strategy(alg, init, mu=0.5)
        results, failures = FL.run_fit(st, 1, ndarrays_to_parameters(init), proxies)
        assert len(results) == 3 and not failures
        if alg == "fedprox":
            assert all("loss_before" in r.metrics for _, r in results)
        agg = parameters_to_ndarrays(st.aggregate_fit(1, results, failures)[0])
        manual = sum(FL.flatten(parameters_to_ndarrays(r.parameters)) * r.num_examples for _, r in results) / 90
        assert np.allclose(FL.flatten(agg), manual, atol=1e-5)


def test_outlier_detection_flags_attackers():
    from defai_addons.fl.algorithms import detect_update_outliers
    torch.manual_seed(0)
    base = torch.randn(200)
    ups = [base + 0.3 * torch.randn(200) for _ in range(10)] + [-3 * base, base * 10]
    od = detect_update_outliers(ups, 3.5)
    assert od["outlier"][-2:].all() and od["outlier"][:10].sum() <= 1


def test_simulator_and_views():
    from defai_addons.monitor.simulator import run_simulation
    from defai_addons.monitor.analytics import compute_views
    rec = run_simulation(dict(rounds=4, n_clients=10, n_edges=2, n_malicious=1, rematch_every=2, timeout_ms=300,
                              delay_growth_ms=50, algorithm="qfedavg", max_train=3000))
    assert rec.status == "finished" and len(rec.rounds) == 4
    v = compute_views(rec.snapshot(0))
    for k in ("trend", "outliers", "contribution", "similarity", "data", "network", "resource", "clients"):
        assert k in v
    assert len(v["network"]["matchings"]) == 2
    assert abs(sum(v["contribution"]["pareto"]["share"]) - 1) < 1e-6
    assert rec.topology["framework"] == "flwr" and len(rec.rounds[-1]["class_acc"]) == 10
    assert all(len(c["classes"]) == 2 for c in rec.clients_static)


def test_flask_api():
    pytest.importorskip("flask")
    import time
    from defai_addons.monitor.server import create_app
    app = create_app(None)
    c = app.test_client()
    assert c.get("/").status_code == 200
    rid = c.post("/api/runs", json={"config": {"rounds": 3, "n_clients": 6, "n_edges": 2, "max_train": 2000}}).get_json()["id"]
    for _ in range(300):
        s = c.get("/api/runs/%s" % rid).get_json()
        if s["status"] == "finished":
            break
        time.sleep(0.1)
    assert s["status"] == "finished" and "views" in s
    ext = c.post("/api/ext/runs", json={"config": {"rounds": 1}, "topology": {"edges": []}, "clients": []}).get_json()["id"]
    assert c.post("/api/ext/runs/%s/rounds" % ext, json={"round": 1}).get_json()["n_rounds"] == 1
    r = c.post("/api/ext/client_stats", json={"client_id": 1, "n": 10, "class_counts": [5, 5]}).get_json()
    assert r["entropy"] == pytest.approx(1.0) and r["ratio"] == pytest.approx(1.0)


def test_secure_aggregation_cancels():
    from defai_addons.compare.privacy import SecureAggregator
    sa = SecureAggregator(5, 100, seed=3)
    vs = [torch.randn(100) for _ in range(5)]
    masked = [sa.mask(i, v, list(range(5))) for i, v in enumerate(vs)]
    assert (masked[0].float() - vs[0]).abs().max() > 1.0
    assert torch.allclose(sa.aggregate(masked), torch.stack(vs).sum(0), atol=1e-4)


def test_comparison_runs():
    from defai_addons.compare.runner import run_comparison
    rep = run_comparison(dict(epochs=2, max_train=3000, envs=["ld_independent", "ld_sync", "fl_fedavg", "fl_fedavg_dp"]))
    keys = [r["key"] for r in rep["results"]]
    assert keys == ["ld_independent", "ld_sync", "fl_fedavg", "fl_fedavg_dp"]
    fl = rep["results"][2]
    assert fl["cost"]["raw_bytes"] == 0 and not fl["privacy"]["raw_data_leaves_site"]
    assert fl["privacy"]["framework"] == "flwr" and fl["secure_agg_error"] < 1e-6
    assert rep["results"][3]["privacy"]["dp"]["epsilon"] > 0
    assert rep["results"][1]["cost"]["raw_bytes"] > 0
    for r in rep["results"]:
        assert set(("precision", "recall", "f1")) <= set(r["global"])


@pytest.mark.parametrize("name", ["y1_joint_embedding", "y2_attention_fusion", "y3_mm_transformer", "y3_no_decoder"])
def test_multimodal_forward_with_missing(name):
    from defai_addons.multimodal import data as MD
    from defai_addons.multimodal.models import build_model
    from defai_addons.multimodal.train import loader
    d = MD.generate(16, seed=0)
    d["mask"][:4] = [True, False, False]
    d["mask"][4:8] = [False, False, True]
    b = next(iter(loader(d, 16, shuffle=False)))
    m = build_model(name).eval()
    out = m(b)
    assert out["logits"].shape == (16, 6) and torch.isfinite(out["logits"]).all()
    b2 = dict(b)
    b2["audio"] = b["audio"].clone()
    b2["audio"][:4] = torch.randn_like(b2["audio"][:4]) * 5
    assert torch.allclose(m(b2)["logits"][:4], out["logits"][:4], atol=1e-4)


def test_gmu_ignores_missing():
    from defai_addons.multimodal.fusion import GatedMultimodalFusion
    g = GatedMultimodalFusion(8)
    x = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    _, gate = g(x, mask)
    assert torch.all(gate[:, 2] == 0)


def test_transformer_compressor_and_fedatc_weights():
    from defai_addons.multimodal.federated import TransformerCompressor, fedatc_weights
    c = TransformerCompressor(P=32, r=4)
    v = torch.randn(5000)
    msg = c.compress(v)
    assert c.decompress(msg).shape == v.shape
    assert TransformerCompressor.message_bytes(msg) * 10 < v.numel() * 4
    w = fedatc_weights([100, 300], [1.0, 1.0], beta=0.5)
    assert w[1] > w[0] and abs(w.sum() - 1) < 1e-9


def test_fedatc_flower_round():
    from defai_addons.multimodal import data as MD
    from defai_addons.multimodal.federated import FedATC
    from defai_addons.multimodal.models import build_model
    cl = MD.make_clients(3, 32, seed=0)
    te = MD.generate(32, seed=1)
    fl = FedATC(lambda: build_model("y3_no_decoder"), cl, local_epochs=1, batch_size=16, log=None,
                update_compression="topk", analysis_compression=True, analysis_samples=16)
    r = fl.run(2, te)
    assert r["framework"] == "flwr" and r["update_compression_ratio"] > 1
    assert r["analysis"]["proto_acc_raw"] is not None and abs(sum(r["history"][-1]["group_weight"].values()) - 1) < 1e-6
