import json

import pytest
from flwr.app import Context, RecordDict

from fl_core.backends import kist_available

pytestmark = pytest.mark.skipif(not kist_available(), reason="KIST defai package is not installed")


def test_kist_flat_with_plugins():
    from data_imbalance.plugin import ImbalancePlugin
    from fl_core.backends import get_backend
    from fl_core.plugins import run_flat
    from security_sim.plugin import SecurityPlugin

    b = get_backend("kist", {"dataset": "synthetic"})
    sec = SecurityPlugin({"attack": "sign_flip", "defense": ["stat"], "ratio": 0.25})
    out = run_flat(b, [sec, ImbalancePlugin({"sampling": "none", "weighting": "size_loss"})], 4, 1)
    assert out["rows"][-1]["test_acc"] > 0.1
    assert sec.summary(None)["client_detection"]["recall"] == 1.0


def test_kist_hierarchical(tmp_path):
    from hierarchical_fl.server import run_hierarchical

    out = run_hierarchical({"backend": "kist", "num-clients": 4, "num-edges": 2, "global-rounds": 1,
                            "edge-rounds": 1, "edge-strategy": "fedprox", "output-dir": str(tmp_path)})
    assert out["final_acc"] is not None


def test_kist_server_client_apps(tmp_path):
    from fl_core import kist_apps
    from fl_core.backends import KistBackend, ensure_identity
    from fl_core.grid import InProcessGrid, Node

    ensure_identity()
    cfg = KistBackend({"dataset": "synthetic", "num-server-rounds": 1, "output-dir": str(tmp_path)}).cfg
    cfg["sejong-plugins"] = json.dumps([{"target": "network_cost.plugin:NetworkCostPlugin", "config": {}}])

    def ctx(k):
        return Context(run_id=1, node_id=k + 1, node_config={"partition-id": k, "num-partitions": 3},
                       state=RecordDict(), run_config=dict(cfg))

    nodes = {k + 1: Node(k + 1, ctx(k), kist_apps.client_train, kist_apps.client_evaluate) for k in range(3)}
    kist_apps.server_main(InProcessGrid(nodes), Context(run_id=1, node_id=0, node_config={}, state=RecordDict(),
                                                        run_config=dict(cfg)))
    data = json.loads(next(tmp_path.rglob("history.json")).read_text())
    assert data["plugins"]["NetworkCostPlugin"]["bytes_saving"] > 0


def test_kist_apps_with_deployment_node_ids(tmp_path):
    import numpy as np

    from fl_core import kist_apps
    from fl_core.backends import KistBackend, ensure_identity
    from fl_core.grid import InProcessGrid, Node

    ensure_identity()
    cfg = KistBackend({"dataset": "synthetic", "num-server-rounds": 2, "output-dir": str(tmp_path)}).cfg
    cfg["sejong-plugins"] = json.dumps([{"target": "security_sim.plugin:SecurityPlugin",
                                         "config": {"attack": "sign_flip", "ratio": 0.25, "defense": ["hmac", "stat"]}}])
    ids = [int(x) for x in np.random.default_rng(7).integers(2 ** 40, 2 ** 62, 4)]
    nodes = {nid: Node(nid, Context(run_id=1, node_id=nid, node_config={"partition-id": k, "num-partitions": 4},
                                    state=RecordDict(), run_config=dict(cfg)),
                       kist_apps.client_train, kist_apps.client_evaluate) for k, nid in enumerate(ids)}
    kist_apps.server_main(InProcessGrid(nodes), Context(run_id=1, node_id=0, node_config={}, state=RecordDict(),
                                                        run_config=dict(cfg)))
    data = json.loads(next(tmp_path.rglob("history.json")).read_text())
    det = data["plugins"]["SecurityPlugin"]["client_detection"]
    assert det["precision"] == 1.0 and det["recall"] == 1.0
    assert data["history"][-1]["test_acc"] > 0.5
