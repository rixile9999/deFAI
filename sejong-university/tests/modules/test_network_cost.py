import numpy as np

from fl_core.backends import get_backend
from fl_core.plugins import run_flat
from network_cost.compress import compress_update
from network_cost.plugin import NetworkCostPlugin
from network_cost.topology import Topology

WEIGHTS = {"price": 1.0, "time": 1.0, "congestion": 0.5}


def path_cost(topology, path, nbytes):
    return sum(link.cost(nbytes, WEIGHTS) for link in topology.hops(path))


def test_topology_routes():
    t = Topology(8, 3, seed=0)
    for c in t.clients:
        for dst in t.edges + [t.cloud]:
            best = t.best_path(c, dst, 2 ** 20, WEIGHTS)
            static = t.static_path(c, dst)
            assert path_cost(t, best, 2 ** 20) <= path_cost(t, static, 2 ** 20) + 1e-9


def test_compression_bytes():
    u = np.random.default_rng(0).normal(size=(100, 100))
    approx, nb = compress_update(u, 0.1, 8)
    assert nb < u.size * 4 * 0.2
    assert np.count_nonzero(approx) <= 1000


def test_flat_cost_savings(small):
    base = NetworkCostPlugin({"optimizations": []})
    run_flat(get_backend("builtin", small), [base], 4, 2)
    assert abs(base.summary(None)["cost_saving"]) < 1e-9
    opt = NetworkCostPlugin({"optimizations": ["edge_compute", "cache", "dag"]})
    run_flat(get_backend("builtin", small), [opt], 4, 2)
    s1 = opt.summary(None)
    assert s1["bytes_saving"] > 0.3 and s1["cost_saving"] > 0.3
