import json

import numpy as np

from hierarchical_fl.dashboard import create_app
from hierarchical_fl.matching import blocking_pairs, deferred_acceptance
from hierarchical_fl.registry import fetch, read_only
from hierarchical_fl.server import HierarchicalServer


def test_deferred_acceptance_is_stable():
    rng = np.random.default_rng(0)
    u = rng.random((12, 3))
    prio = rng.random((3, 12))
    a = deferred_acceptance(u, prio, [4, 4, 4], [0, 1, 2], list(range(12)))
    assert (a >= 0).all()
    assert blocking_pairs(a, u, prio, [4, 4, 4], [0, 1, 2], list(range(12))) == 0


def test_hierarchical_run_and_dashboard(tmp_path, small):
    server = HierarchicalServer({"backend-config": small, "num-clients": 6, "num-edges": 2, "global-rounds": 2,
                                 "edge-rounds": 1, "deadline": {"policy": "adaptive"},
                                 "output-dir": str(tmp_path)})
    out = server.run()
    assert [r["round"] for r in out["rounds"]] == [0, 1, 2]
    assert out["final_acc"] > 0.1
    db = read_only(server.run_dir / "hfl.sqlite")
    assert len(fetch(db, "SELECT * FROM clients")) == 6
    assert len(fetch(db, "SELECT * FROM edge_rounds")) >= 2
    db.close()
    client = create_app(tmp_path).test_client()
    data = client.get(f"/api/run/{out['run_id']}").get_json()
    assert data["run"]["status"] == "finished"
    assert len(data["rounds"]) == 3
    assert client.get(f"/run/{out['run_id']}").status_code == 200
    assert json.loads(client.get("/api/runs").data)
