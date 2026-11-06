import copy

import numpy as np
import torch
from flwr.app import ArrayRecord, MetricRecord

from fl_core.plugins import Plugin
from fl_core.strategies import StrategyWrapper
from fl_core.task import test
from graph_pruning.dynamic import count_macs, count_params
from graph_pruning.graph import (build_graph, init_meta, mask_channels, path_importance, percentile_threshold, prune,
                                 prune_model, taylor_importance)
from graph_pruning.sparsity import bn_l1_loss_factory


class GraphPruningStrategy(StrategyWrapper):
    def __init__(self, inner, template, input_shape, val_data, start=2, every=2, percentile=30.0, eta=0.5,
                 tol=0.15, min_eta=0.05, target=None, importance="path", log=None, acc_tol=0.05):
        super().__init__(inner)
        self.template = copy.deepcopy(template)
        self.input_shape = tuple(input_shape)
        self.x_val, self.y_val = val_data
        self.start_round, self.every = start, every
        self.percentile, self.eta, self.tol, self.min_eta, self.target = percentile, eta, tol, min_eta, target
        self.acc_tol = acc_tol
        self.importance = importance
        self.log_fn = log
        graph = build_graph(self.template, self.input_shape)
        init_meta(self.template, graph)
        self.meta = copy.deepcopy(self.template.prune_meta)
        self.orig_idx = copy.deepcopy(self.meta["orig_idx"])
        self.tau = None
        self.tau_acc = None
        self.pending = None
        self.stopped = False
        self.history = []
        self.p0 = count_params(self.template)
        self.m0 = count_macs(self.template, self.input_shape)

    def _full(self, arrays):
        m = copy.deepcopy(self.template)
        m.load_state_dict(arrays.to_torch_state_dict())
        m.prune_meta = copy.deepcopy(self.meta)
        return m

    def _keep(self, orig_idx):
        return {k: torch.as_tensor(v, dtype=torch.long) for k, v in orig_idx.items()}

    def compact(self, arrays, orig_idx=None):
        full = self._full(arrays)
        graph = build_graph(full, self.input_shape)
        return prune_model(full, self._keep(orig_idx or self.orig_idx), graph)

    def _loss(self, model):
        return test(model, self.x_val, self.y_val)

    def _importance(self, model, graph):
        if self.importance == "taylor":
            idx = np.arange(min(512, len(self.y_val)))
            batches = [(self.x_val[idx[s:s + 128]], self.y_val[idx[s:s + 128]]) for s in range(0, len(idx), 128)]
            return taylor_importance(model, graph, batches)
        return path_importance(model, graph)

    def _record(self, row):
        self.history.append(row)
        if self.log_fn is not None:
            self.log_fn({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    def _masked(self, arrays):
        full = self._full(arrays)
        masked = mask_channels(full, self.orig_idx)
        return ArrayRecord(masked.state_dict()), masked

    def aggregate_train(self, server_round, replies):
        arrays, metrics = self.inner.aggregate_train(server_round, replies)
        if arrays is None:
            return arrays, metrics
        arrays, masked = self._masked(arrays)
        extra = {}
        if not self.stopped and server_round >= self.start_round and (server_round - self.start_round) % self.every == 0:
            arrays, masked, extra = self._step(server_round, arrays, masked)
        compact = self.compact(arrays)
        extra.update({"prune-params": count_params(compact), "prune-params-red": 1 - count_params(compact) / self.p0,
                      "prune-macs-red": 1 - count_macs(compact, self.input_shape) / self.m0})
        metrics = MetricRecord({**dict(metrics or {}), **extra})
        return arrays, metrics

    def _step(self, rnd, arrays, masked):
        L_now, A_now = self._loss(masked)
        if self.pending is not None:
            snap_arrays, snap_idx, L_before, A_before, tau = self.pending
            d = (L_now - L_before) / L_before
            ok = d <= self.tol and A_now >= A_before - self.acc_tol
            self._record({"round": rnd, "event": "check", "tau": tau, "val_loss": L_now, "val_acc": A_now,
                          "L_before": L_before, "acc_before": A_before, "rel_change": d, "accepted": int(ok)})
            self.pending = None
            if ok:
                self.tau_acc = tau
                self.tau = tau * (1 + self.eta)
                if self.target is not None and 1 - count_params(self.compact(arrays)) / self.p0 >= self.target:
                    self.stopped = True
                    return arrays, masked, {"prune-event": 2.0}
            else:
                self.orig_idx = snap_idx
                arrays = snap_arrays
                masked = self._full(arrays)
                self.eta /= 2
                if self.eta < self.min_eta:
                    self.stopped = True
                    return arrays, masked, {"prune-event": -2.0}
                self.tau = self.tau_acc * (1 + self.eta) if self.tau_acc is not None else tau / (1 + self.eta)
                return arrays, masked, {"prune-event": -1.0}
        compact = self.compact(arrays)
        graph = build_graph(compact, self.input_shape)
        imp = self._importance(compact, graph)
        if "imp0" in compact.prune_meta and "imp0" not in self.meta:
            self.meta["imp0"] = copy.deepcopy(compact.prune_meta["imp0"])
        if self.tau is None:
            self.tau = percentile_threshold(imp, self.percentile)
        cand, keep, removed = prune(compact, self.tau, lambda m, g: self._importance(m, g))
        if removed == 0:
            self.tau_acc, self.tau = self.tau, self.tau * (1 + self.eta)
            self._record({"round": rnd, "event": "no-op", "tau": self.tau_acc})
            return arrays, masked, {"prune-event": 0.0}
        self.pending = (arrays, copy.deepcopy(self.orig_idx), L_now, A_now, self.tau)
        self.orig_idx = copy.deepcopy(cand.prune_meta["orig_idx"])
        arrays, masked = self._masked(arrays)
        L_new, A_new = self._loss(masked)
        self._record({"round": rnd, "event": "prune", "tau": self.tau, "removed": removed, "val_loss_before": L_now,
                      "val_loss_after": L_new, "val_acc_after": A_new,
                      "params_red": 1 - count_params(cand) / self.p0})
        return arrays, masked, {"prune-event": 1.0, "prune-removed": float(removed)}

    def final_models(self, arrays):
        masked = self._full(arrays)
        masked = mask_channels(masked, self.orig_idx)
        compact = self.compact(ArrayRecord(masked.state_dict()))
        return masked, compact


class GraphPruningPlugin(Plugin):
    name = "graph_pruning"

    def __init__(self, config=None):
        defaults = {"start": 2, "every": 2, "percentile": 30.0, "eta": 0.5, "tol": 0.15, "acc-tol": 0.05, "target": None,
                    "importance": "path", "bn-l1": 0.0, "val-samples": 1000, "tiers": ["flat", "global"]}
        super().__init__({**defaults, **(config or {})})
        self.strategy = None

    def train_hooks(self, owner):
        if self.config["bn-l1"] > 0:
            return {"loss_factory": bn_l1_loss_factory(self.config["bn-l1"])}
        return {}

    def wrap_strategy(self, strategy, tier, owner):
        if tier not in self.config["tiers"]:
            return strategy
        backend = owner.backend
        x, y = backend.test_set()
        n = min(self.config["val-samples"], len(y) // 2)
        torch.manual_seed(int(backend.cfg["seed"]))
        c = self.config
        self.strategy = GraphPruningStrategy(strategy, backend.build_model(), backend.input_shape(), (x[:n], y[:n]),
                                             c["start"], c["every"], c["percentile"], c["eta"], c["tol"],
                                             target=c["target"], importance=c["importance"], log=print,
                                             acc_tol=c["acc-tol"])
        return self.strategy

    def summary(self, owner):
        return {"history": self.strategy.history if self.strategy else []}
