import inspect

from flwr.serverapp import strategy as fs
from flwr.serverapp.strategy import Strategy

FLOWER = {
    "fedavg": fs.FedAvg,
    "fedprox": fs.FedProx,
    "qfedavg": fs.QFedAvg,
    "fedmedian": fs.FedMedian,
    "fedtrimmedavg": fs.FedTrimmedAvg,
    "krum": fs.Krum,
    "multikrum": fs.MultiKrum,
    "bulyan": fs.Bulyan,
    "fedavgm": fs.FedAvgM,
    "fedadam": fs.FedAdam,
    "fedyogi": fs.FedYogi,
    "fedadagrad": fs.FedAdagrad,
}


def build_flower_strategy(name, cfg=None, fraction_train=1.0, fraction_evaluate=0.0, min_nodes=1, **extra):
    cfg = cfg or {}
    if name not in FLOWER:
        raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(FLOWER)}")
    cls = FLOWER[name]
    kwargs = dict(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate, min_train_nodes=min_nodes,
                  min_evaluate_nodes=min_nodes, min_available_nodes=min_nodes,
                  proximal_mu=cfg.get("proximal-mu", 0.01), q=cfg.get("qfedavg-q", 0.1),
                  client_learning_rate=cfg.get("learning-rate", 0.05), beta=cfg.get("trim-ratio", 0.2),
                  num_malicious_nodes=cfg.get("num-byzantine", 0))
    kwargs.update(extra)
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def relax_minimums(strategy, min_nodes=1):
    for attr in ("min_train_nodes", "min_evaluate_nodes", "min_available_nodes"):
        if hasattr(strategy, attr):
            setattr(strategy, attr, min(getattr(strategy, attr), min_nodes) if getattr(strategy, attr) else 0)
    return strategy


def innermost(strategy):
    while isinstance(strategy, StrategyWrapper):
        strategy = strategy.inner
    return strategy


class StrategyWrapper(Strategy):
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def summary(self):
        self.inner.summary()

    def configure_train(self, server_round, arrays, config, grid):
        return self.inner.configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        return self.inner.aggregate_train(server_round, replies)

    def configure_evaluate(self, server_round, arrays, config, grid):
        return self.inner.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round, replies):
        return self.inner.aggregate_evaluate(server_round, replies)
