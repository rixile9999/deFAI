import numpy as np
from flwr.app import MetricRecord

from fl_core.strategies import StrategyWrapper

MODES = ("size", "loss", "size_loss")


class DynamicWeightStrategy(StrategyWrapper):
    def __init__(self, inner, mode="size_loss", alpha=1.0, beta=1.0, loss_key="initial-loss", ema=0.5,
                 weight_key="num-examples"):
        super().__init__(inner)
        if mode not in MODES:
            raise ValueError(f"unknown weighting mode {mode!r}; choose from {', '.join(MODES)}")
        self.mode, self.alpha, self.beta = mode, alpha, beta
        self.loss_key, self.ema, self.weight_key = loss_key, ema, weight_key
        self.smoothed = {}
        self.log = []

    def weights(self, sizes, losses, ids):
        sizes = np.asarray(sizes, dtype=float)
        losses = np.asarray(losses, dtype=float)
        sm = []
        for i, l in zip(ids, losses):
            prev = self.smoothed.get(i, l)
            v = self.ema * prev + (1 - self.ema) * l
            self.smoothed[i] = v
            sm.append(v)
        sm = np.asarray(sm)
        rel = sm / max(sm.mean(), 1e-12)
        if self.mode == "size":
            w = sizes ** self.alpha
        elif self.mode == "loss":
            w = rel ** self.beta
        else:
            w = (sizes ** self.alpha) * (rel ** self.beta)
        return w / w.sum()

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        valid = [r for r in replies if not r.has_error()]
        if len(valid) >= 2:
            ids = [int(r.metadata.src_node_id) for r in valid]
            sizes = [float(r.content["metrics"][self.weight_key]) for r in valid]
            losses = [float(r.content["metrics"].get(self.loss_key, r.content["metrics"].get("train_loss", 1.0)))
                      for r in valid]
            w = self.weights(sizes, losses, ids)
            total = sum(sizes)
            for r, wi in zip(valid, w):
                m = dict(r.content["metrics"])
                m[self.weight_key] = float(max(wi * total, 1e-6))
                r.content["metrics"] = MetricRecord(m)
            self.log.append({"round": server_round, "clients": ids, "sizes": sizes, "losses": losses,
                             "weights": w.round(5).tolist(), "fedavg": (np.asarray(sizes) / total).round(5).tolist()})
        return self.inner.aggregate_train(server_round, replies)
