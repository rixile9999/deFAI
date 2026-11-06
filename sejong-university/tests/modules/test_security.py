import numpy as np
from flwr.app import Array, ArrayRecord, RecordDict

from fl_core.backends import get_backend
from fl_core.plugins import run_flat
from security_sim import detectors
from security_sim.keys import KeyStore, sign, verify
from security_sim.plugin import SecurityPlugin


def test_signatures():
    keys = KeyStore("k")
    content = RecordDict({"arrays": ArrayRecord({"w": Array(np.ones(3, dtype=np.float32))})})
    sign(content, "client1", 2, keys)
    assert verify(content, "client1", 2, keys, ["cert", "hmac", "mfa"]) == []
    assert verify(content, "client1", 3, keys, ["hmac"])
    assert verify(content, "client2", 2, keys, ["cert"])


def test_detectors():
    rng = np.random.default_rng(0)
    U = rng.normal(0, 1, (8, 50))
    U[0] *= 30
    assert 0 in detectors.statistical(U)
    U2 = rng.normal(0, 1, (6, 50))
    U2[4] = U2[5] = np.ones(50) * 3
    assert {4, 5} <= set(detectors.kmeans_collusion(U2))


def test_flat_sybil_certificate(small):
    plugin = SecurityPlugin({"attack": "sybil", "defense": ["cert"], "sybils": 2})
    out = run_flat(get_backend("builtin", small), [plugin], 4, 1)
    det = plugin.summary(None)["client_detection"]
    assert det["recall"] == 1.0 and det["precision"] == 1.0
    assert out["alerts"]


def test_flat_model_poison_statistics(small):
    plugin = SecurityPlugin({"attack": "model_poison", "defense": ["stat", "hmac"], "ratio": 0.2})
    run_flat(get_backend("builtin", small), [plugin], 5, 1)
    det = plugin.summary(None)["client_detection"]
    assert det["recall"] == 1.0 and det["precision"] == 1.0
