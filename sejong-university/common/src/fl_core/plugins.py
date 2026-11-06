import importlib


class Plugin:
    name = "plugin"

    def __init__(self, config=None):
        self.config = dict(config or {})

    def configure(self, cfg):
        return cfg

    def setup(self, owner):
        pass

    def wrap_strategy(self, strategy, tier, owner):
        return strategy

    def client_middleware(self, owner):
        return []

    def train_hooks(self, owner):
        return {}

    def round_state(self, server, rnd, state):
        return state

    def edge_reply(self, session, rnd, content):
        return content

    def transport(self, server, rnd, edge_id, reply):
        return reply

    def on_round_end(self, server, rnd, row):
        pass

    def summary(self, owner):
        return {}


class Middleware:
    def before(self, msg, node):
        return None

    def after(self, msg, reply, node):
        return reply


def spec_of(plugin):
    return {"target": f"{type(plugin).__module__}:{type(plugin).__name__}", "config": plugin.config}


def load_plugin(spec):
    if isinstance(spec, Plugin):
        return spec
    module, _, attr = spec["target"].partition(":")
    cls = getattr(importlib.import_module(module), attr)
    return cls(spec.get("config"))


def load_plugins(specs):
    return [load_plugin(s) for s in specs or []]


def merge_hooks(plugins, owner):
    hooks = {}
    for p in plugins:
        for k, v in p.train_hooks(owner).items():
            if k in hooks:
                raise ValueError(f"two plugins provide the client hook {k!r}")
            hooks[k] = v
    return hooks


class FlatOwner:
    def __init__(self, backend, num_clients, plugins, event_sink=None):
        self.backend = backend
        self.num_clients = num_clients
        self.plugins = plugins
        self.edge_id = None
        self.round = 0
        self.alerts = []
        self.roles = {}
        self.event_sink = event_sink

    def alert(self, level, message, source="server"):
        item = {"round": self.round, "level": level, "source": source, "message": message}
        self.alerts.append(item)
        if self.event_sink is not None:
            self.event_sink(item)


class _RoundTracker(Middleware):
    def __init__(self, owner):
        self.owner = owner

    def before(self, msg, node):
        if msg.has_content() and "config" in msg.content:
            self.owner.round = int(msg.content["config"].get("server-round", self.owner.round))
        return None


def run_flat(backend, plugins, num_clients, rounds, strategy="fedavg", workers=1, model=None, callback=None):
    from fl_core.runner import run_federated

    plugins = load_plugins(plugins)
    owner = FlatOwner(backend, num_clients, plugins)
    for p in plugins:
        p.setup(owner)
    hooks = merge_hooks(plugins, owner)
    train = backend.custom_train_handler(**hooks) if hooks else backend.train_handler
    middleware = [_RoundTracker(owner)] + [m for p in plugins for m in p.client_middleware(owner)]
    grid = backend.grid(num_clients, workers=workers, middleware=middleware, train=train)
    strat = backend.build_strategy(strategy, 1.0, 0.0) if isinstance(strategy, str) else strategy
    for p in plugins:
        strat = p.wrap_strategy(strat, "flat", owner)
    rows = []

    def on_round(rnd, arrays, record):
        row = {"round": rnd, **dict(record)}
        for p in plugins:
            p.on_round_end(owner, rnd, row)
        rows.append(row)
        if callback is not None:
            callback(rnd, arrays, row)

    out = run_federated(backend, strat, num_clients, rounds, grid=grid, model=model, callback=on_round)
    out["rows"] = rows
    out["alerts"] = owner.alerts
    out["plugins"] = {type(p).__name__: p.summary(owner) for p in plugins}
    return out
