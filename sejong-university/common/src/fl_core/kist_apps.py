import json
import os

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp

from fl_core.plugins import FlatOwner, load_plugins, merge_hooks

PLUGIN_KEY = "sejong-plugins"
PLUGIN_ENV = "SEJONG_PLUGINS"


def plugin_specs(run_config):
    raw = run_config.get(PLUGIN_KEY) or os.environ.get(PLUGIN_ENV) or "[]"
    return json.loads(raw) if isinstance(raw, str) else list(raw)


def _owner(run_config, num_clients):
    from fl_core.backends import KistBackend

    backend = KistBackend({k: v for k, v in run_config.items() if k != PLUGIN_KEY})
    plugins = load_plugins(plugin_specs(run_config))
    owner = FlatOwner(backend, num_clients, plugins)
    for p in plugins:
        p.setup(owner)
    return owner


server_app = ServerApp()


@server_app.main()
def server_main(grid: Grid, context: Context) -> None:
    from defai.config import validate_config
    from defai.results import build_history, run_name, write_summary
    from defai.strategies import build_strategy
    from defai.task import build_model, get_device, load_centralized_testset, test

    cfg = context.run_config
    validate_config(cfg)
    owner = _owner(cfg, len(list(grid.get_node_ids())))
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["dataset"])
    loader = load_centralized_testset(cfg["dataset"])
    device = get_device()
    rows = []

    def evaluate(server_round, arrays):
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test(model, loader, device)
        row = {"round": server_round, "test_loss": loss, "test_acc": acc}
        for p in owner.plugins:
            p.on_round_end(owner, server_round, row)
        rows.append(row)
        return MetricRecord({"test_loss": loss, "test_acc": acc})

    strategy = build_strategy(cfg)
    for p in owner.plugins:
        strategy = p.wrap_strategy(strategy, "flat", owner)
    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
                            train_config=ConfigRecord({"lr": cfg["learning-rate"]}),
                            num_rounds=cfg["num-server-rounds"], evaluate_fn=evaluate)
    out_dir = os.path.join(cfg["output-dir"], "sejong_" + run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)
    write_summary(os.path.join(out_dir, "history.json"),
                  {"config": dict(cfg), "history": build_history(result), "plugin_rows": rows,
                   "alerts": owner.alerts,
                   "plugins": {type(p).__name__: p.summary(owner) for p in owner.plugins}})
    print(f"Saved results to {os.path.abspath(out_dir)}")


_CLIENT = {}


def _client_side(context):
    key = (context.node_id, json.dumps(plugin_specs(context.run_config)))
    if key not in _CLIENT:
        import defai.client_app

        owner = _owner(context.run_config, int(context.node_config["num-partitions"]))
        hooks = merge_hooks(owner.plugins, owner)
        train = owner.backend.custom_train_handler(**hooks) if hooks else defai.client_app.train
        middleware = [m for p in owner.plugins for m in p.client_middleware(owner)]
        _CLIENT[key] = (owner, train, defai.client_app.evaluate, middleware)
    return _CLIENT[key]


class _NodeView:
    def __init__(self, context):
        self.context = context


def _run(handler, middleware, msg, context):
    node = _NodeView(context)
    for m in middleware:
        early = m.before(msg, node)
        if early is not None:
            return early
    reply = handler(msg, context)
    for m in reversed(middleware):
        reply = m.after(msg, reply, node)
    return reply


client_app = ClientApp()


@client_app.train()
def client_train(msg: Message, context: Context):
    owner, train, _, middleware = _client_side(context)
    return _run(train, middleware, msg, context)


@client_app.evaluate()
def client_evaluate(msg: Message, context: Context):
    owner, _, evaluate, middleware = _client_side(context)
    return _run(evaluate, middleware, msg, context)
