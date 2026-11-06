import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MessageType, RecordDict

from fl_core import wire
from fl_core.backends import ensure_identity, get_backend
from fl_core.records import digest, to_arrays
from fl_core.runner import run_federated
from fl_core.task import partition


def test_partitions_cover_all_samples():
    labels = np.repeat(np.arange(10), 50)
    for scheme in ("iid", "dirichlet", "quantity"):
        parts = partition(labels, 5, scheme, alpha=0.3, sigma=1.0, seed=1)
        joined = np.sort(np.concatenate(parts))
        assert len(parts) == 5
        assert len(np.unique(joined)) == len(joined)


def test_wire_roundtrip():
    ensure_identity()
    arrays = ArrayRecord({"w": Array(np.arange(6, dtype=np.float32).reshape(2, 3))})
    msg = Message(content=RecordDict({"arrays": arrays, "config": ConfigRecord({"lr": 0.1, "plan": "{}"})}),
                  dst_node_id=3, message_type=MessageType.TRAIN)
    back = wire.unpack(wire.pack([msg, msg]))
    assert len(back) == 2
    assert digest(to_arrays(back[0].content["arrays"])) == digest(to_arrays(arrays))
    assert back[1].metadata.dst_node_id == 3


def test_flat_round_builtin(small):
    b = get_backend("builtin", small)
    out = run_federated(b, b.build_strategy("fedprox"), 4, 1)
    assert len(out["history"]) == 2
    assert out["history"][-1]["test_acc"] > 0.1
