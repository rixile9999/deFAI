import numpy as np
import torch

from fl_core.task import build_model, load_split
from layer_kd.features import stage_points
from layer_kd.trainer import METHODS, KDConfig, KDTrainer


def test_stage_points_match():
    t, s = build_model("synthetic", "cnn", 1.0), build_model("synthetic", "cnn", 0.25)
    ts, _, _ = stage_points(t, (1, 1, 28, 28))
    ss, _, _ = stage_points(s, (1, 1, 28, 28))
    assert ts == ss and len(ts) == 4


def test_all_methods_step():
    x, y = load_split("synthetic", "train")
    x, y = x[:128], y[:128]
    teachers = [build_model("synthetic", "cnn", 1.0).eval() for _ in range(2)]
    for method in METHODS:
        torch.manual_seed(0)
        student = build_model("synthetic", "cnn", 0.25)
        tr = KDTrainer(KDConfig(method=method), teachers if method == "ensemble" else teachers[:1], student,
                       (1, 1, 28, 28))
        logs, steps = tr.fit(x, y, epochs=1, batch_size=64, lr=0.02)
        tr.close()
        assert steps == 2 and np.isfinite(logs["loss"])
