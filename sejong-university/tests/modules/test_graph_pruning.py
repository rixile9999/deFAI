import copy

import torch

from fl_core.backends import get_backend
from fl_core.plugins import run_flat
from fl_core.task import build_model
from graph_pruning.federated import GraphPruningPlugin
from graph_pruning.graph import build_graph, init_meta, mask_channels, prune


def test_resnet_graph_and_equivalence():
    torch.manual_seed(0)
    model = build_model("synthetic", "resnet8").eval()
    shape = (1, 1, 28, 28)
    graph = build_graph(model, shape)
    assert sum(g.coupled for g in graph.prunable) == 3
    init_meta(model, graph)
    compact, keep, removed = prune(model, 1.0)
    assert removed > 0
    masked = mask_channels(copy.deepcopy(model), compact.prune_meta["orig_idx"]).eval()
    x = torch.randn(4, *shape[1:])
    with torch.no_grad():
        assert torch.allclose(masked(x), compact.eval()(x), atol=1e-4)


def test_federated_pruning(small):
    b = get_backend("builtin", small)
    plugin = GraphPruningPlugin({"start": 1, "every": 1, "percentile": 30})
    out = run_flat(b, [plugin], 4, 3)
    masked, compact = plugin.strategy.final_models(out["result"].arrays)
    x, _ = b.test_set()
    with torch.no_grad():
        assert torch.equal(masked.eval()(x[:64]).argmax(1), compact.eval()(x[:64]).argmax(1))
    assert plugin.strategy.history
