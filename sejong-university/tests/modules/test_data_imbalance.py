import numpy as np
import torch

from data_imbalance.plugin import ImbalancePlugin
from data_imbalance.sampling import enn, imbalance_ratio, resample, smote, tomek_links
from data_imbalance.weighting import DynamicWeightStrategy
from fl_core.backends import get_backend
from fl_core.plugins import run_flat


def test_sampling():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(130, 1, 4, 4, generator=g)
    y = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    _, ys = smote(x, y)
    assert int((ys == 1).sum()) == 100
    assert len(tomek_links(x, y)[1]) <= 130 and len(enn(x, y)[1]) <= 130
    assert imbalance_ratio(resample(x, y, "combo")[1], 2) < imbalance_ratio(y, 2)


def test_dynamic_weights():
    s = DynamicWeightStrategy(None, "size_loss", ema=0.0)
    w = s.weights([100, 100], [1.0, 3.0], [1, 2])
    assert w[1] > w[0] and np.isclose(w.sum(), 1.0)


def test_flat_imbalance(small):
    small.update({"partitioner": "dirichlet", "dirichlet-alpha": 0.3})
    plugin = ImbalancePlugin({"sampling": "combo", "weighting": "size_loss"})
    out = run_flat(get_backend("builtin", small), [plugin], 4, 1)
    assert out["rows"][-1]["test_acc"] > 0.1
    assert plugin.summary(None)["weight_log"]
