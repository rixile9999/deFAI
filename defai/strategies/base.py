"""Federated Averaging and the machinery all strategies in this package share.

Written against the ``flwr.serverapp.strategy.Strategy`` interface of Flower 1.40
(``configure_train`` -> ``aggregate_train`` -> ``configure_evaluate`` ->
``aggregate_evaluate``, driven by ``Strategy.start``), following the structure of
Flower's own ``FedAvg``. Only that abstract interface is imported from Flower; every
algorithm below is implemented here.

Aggregation is organised around *updates* (``local - global``) so that algorithms
plug in at one of three small hooks:

``select_updates``     which replies take part (Krum, ...)
``aggregate_updates``  how updates are combined into one (mean, median, ...)
``update_global``      how that combined update moves the global model (lr, momentum, Adam, ...)

Differential privacy (``defai.privacy.dp``) replaces ``aggregate_updates`` and runs
*before* ``update_global``, so server-side optimisers are post-processing of the noised
update.

McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized
Data", AISTATS 2017.
"""

import math
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from logging import INFO, WARNING

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.supercore import log

from defai.aggregation import Arrays, cast_like, weighted_mean
from defai.privacy.dp import KEY_UNCLIPPED_BIT
from defai.records import record_bytes, to_arrays, to_record

KEY_ARRAYS = "arrays"
KEY_CONFIG = "config"
KEY_METRICS = "metrics"


@dataclass
class ClientUpdate:
    """One validated train reply."""

    node_id: int
    delta: Arrays  # local - global, float64
    weight: float  # value of ``weighted_by_key`` (e.g. number of local examples)
    metrics: dict
    extras: dict[str, Arrays] = field(default_factory=dict)  # additional ArrayRecords in the reply


def ordered_node_ids(grid: Grid) -> list[int]:
    """Connected node ids in registration order.

    Node ids are random and ``Grid.get_node_ids`` is unordered, so neither is stable across
    runs. The registration order is: the simulation registers node ``i`` before node ``i+1``
    and gives it ``partition-id = i``, so a seeded sample of positions picks the same
    partitions in every run. Grids that cannot tell (no ``registered_at``) fall back to
    sorting by node id.
    """
    get_nodes = getattr(grid, "get_nodes", None)
    if get_nodes is not None:
        nodes = list(get_nodes())
        if all(getattr(n, "registered_at", "") for n in nodes):
            return [n.node_id for n in sorted(nodes, key=lambda n: (n.registered_at, n.node_id))]
    return sorted(grid.get_node_ids())


def weighted_metrics(metrics: list[dict], weights: list[float], skip: str) -> dict:
    """Weighted mean of numeric metrics (scalars or lists); non-numeric entries are ignored."""
    total = float(sum(weights))
    out: dict = {}
    for key in metrics[0]:
        if key == skip:
            continue
        try:
            columns = np.asarray([m[key] for m in metrics], dtype=np.float64)
        except (TypeError, ValueError, KeyError):
            continue
        mean = np.tensordot(np.asarray(weights, dtype=np.float64) / total, columns, axes=1)
        out[key] = mean.tolist() if mean.ndim else float(mean)
    return out


FAIRNESS_METRICS = ("eval_acc", "eval_acc_personalized")
KEY_UPLOAD_BYTES = "upload-bytes"  # optional client metric: bytes actually sent (set when updates are compressed)
MB = 2**20


def client_spread(metrics: list[dict], key: str) -> dict:
    """Unweighted min / max / std of ``key`` over clients.

    The weighted mean hides clients the model serves badly, which is the usual failure of
    federated learning on non-IID data: ``<key>_min`` is the worst client, ``<key>_std`` the
    disagreement between clients.
    """
    values = [m[key] for m in metrics if isinstance(m.get(key), (int, float)) and not isinstance(m.get(key), bool)]
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"{key}_min": float(arr.min()), f"{key}_max": float(arr.max()), f"{key}_std": float(arr.std())}


class FedAvg(Strategy):
    """Federated Averaging (McMahan et al., 2017).

    Parameters
    ----------
    fraction_train, fraction_evaluate : float
        Fraction of connected nodes sampled for training / evaluation each round.
    min_train_nodes, min_evaluate_nodes, min_available_nodes : int
        Lower bounds on the sample sizes and on the number of connected nodes.
    weighted_by_key : str
        MetricRecord entry holding each client's aggregation weight.
    server_learning_rate : float
        Scales the aggregated update (1.0 = plain FedAvg).
    lr_decay : float
        The clients' learning rate in round ``t`` is ``lr * lr_decay ** (t - 1)`` (1.0 = constant).
    dp : defai.privacy.dp FixedClipping | AdaptiveClipping | None
        Central differential privacy applied to the aggregated update.
    seed : int | None
        Seed of the client-sampling RNG.
    """

    name = "fedavg"
    supports_dp = True
    #: ArrayRecords besides ``arrays`` that every train reply must contain.
    extra_reply_records: tuple[str, ...] = ()
    #: MetricRecord entries (positive finite numbers) that every train reply must contain.
    required_metrics: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        server_learning_rate: float = 1.0,
        lr_decay: float = 1.0,
        dp=None,
        seed: int | None = None,
    ) -> None:
        if dp is not None and not self.supports_dp:
            raise ValueError(f"{type(self).__name__} cannot be combined with differential privacy")
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes if fraction_train > 0 else 0
        self.min_evaluate_nodes = min_evaluate_nodes if fraction_evaluate > 0 else 0
        self.min_available_nodes = min_available_nodes
        self.weighted_by_key = weighted_by_key
        self.server_learning_rate = server_learning_rate
        if not 0.0 < lr_decay <= 1.0:
            raise ValueError("lr_decay must be in (0, 1]")
        self.lr_decay = lr_decay
        self.dp = dp
        self._rng = random.Random(seed)
        self._rank: dict[int, int] = {}     # node id -> position in registration order
        self._down_bytes = 0                # model payload sent to the clients of the current round
        self._comm_total = 0                # cumulative training traffic (down + up) of the run
        self._global: Arrays | None = None  # model sent in the current round
        self._num_sampled = 0               # clients asked to train in the current round
        self._population = 0                # clients connected in the current round
        self._budget_warned = False
        self._round_metrics: dict[str, float] = {}  # strategies may report extra numbers for the round here

    # ------------------------------------------------------------------ logging
    def summary(self) -> None:
        log(INFO, "\t├──> %s settings:", type(self).__name__)
        log(INFO, "\t│\t├── Sampling: train %.2f | evaluate %.2f", self.fraction_train, self.fraction_evaluate)
        log(INFO, "\t│\t├── Minimum nodes: train %d | evaluate %d | available %d",
            self.min_train_nodes, self.min_evaluate_nodes, self.min_available_nodes)
        log(INFO, "\t│\t├── Server learning rate: %s | client lr decay per round: %s", self.server_learning_rate, self.lr_decay)
        log(INFO, "\t│\t└── Differential privacy: %s", self.dp.describe() if self.dp else "off")

    # ----------------------------------------------------------------- sampling
    def _sample(self, grid: Grid, fraction: float, min_nodes: int) -> tuple[list[int], int]:
        while len(all_nodes := ordered_node_ids(grid)) < max(self.min_available_nodes, min_nodes):
            log(INFO, "Waiting for nodes to connect: %d connected (minimum required: %d).",
                len(all_nodes), max(self.min_available_nodes, min_nodes))
            time.sleep(1)
        self._rank = {node_id: i for i, node_id in enumerate(all_nodes)}
        size = min(max(int(len(all_nodes) * fraction), min_nodes), len(all_nodes))
        return self._rng.sample(all_nodes, size), len(all_nodes)

    @staticmethod
    def _messages(content: RecordDict, node_ids: list[int], message_type: str) -> list[Message]:
        return [Message(content=content, message_type=message_type, dst_node_id=n) for n in node_ids]

    # -------------------------------------------------------------------- train
    def _extend_train_config(self, server_round: int, config: ConfigRecord) -> None:
        """Hook: add algorithm-specific hyper-parameters the clients need."""

    def _extra_train_records(self, server_round: int) -> dict[str, ArrayRecord]:
        """Hook: extra ArrayRecords sent next to the model (e.g. SCAFFOLD's control variate)."""
        return {}

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.fraction_train == 0.0:
            return []
        if self.dp is not None and self.dp.budget_exhausted():
            if not self._budget_warned:
                log(WARNING, "Privacy budget reached: epsilon is %.4f and another round would exceed "
                    "max_epsilon=%g; training stops here.", self.dp.epsilon, self.dp.max_epsilon)
                self._budget_warned = True
            return []
        node_ids, self._population = self._sample(grid, self.fraction_train, self.min_train_nodes)
        self._num_sampled = len(node_ids)
        if self.dp is not None:
            self.dp.begin_round(self._num_sampled)
        log(INFO, "configure_train: Sampled %d nodes (out of %d)", len(node_ids), self._population)

        self._global = to_arrays(arrays)
        config = ConfigRecord(dict(config))  # never mutate the caller's (shared) config
        config["server-round"] = server_round
        if self.lr_decay != 1.0 and "lr" in config:
            config["lr"] = float(config["lr"]) * self.lr_decay ** (server_round - 1)
        self._extend_train_config(server_round, config)
        if self.dp is not None:
            config.update(self.dp.client_config())
        content = RecordDict(
            {KEY_ARRAYS: arrays, KEY_CONFIG: config, **self._extra_train_records(server_round)}
        )
        self._down_bytes = len(node_ids) * sum(record_bytes(r) for r in content.array_records.values())
        self._comm_total += self._down_bytes
        return self._messages(content, node_ids, MessageType.TRAIN)

    def _check_reply(self, content: RecordDict, train: bool) -> str | None:
        """Return a reason if the reply is unusable, else None."""
        metrics = content.get(KEY_METRICS)
        if not isinstance(metrics, MetricRecord):
            return f"missing MetricRecord '{KEY_METRICS}'"
        weight = metrics.get(self.weighted_by_key)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight < math.inf:
            return f"'{self.weighted_by_key}' must be a positive finite number, got {weight!r}"
        if not train:
            return None
        for key in self.required_metrics:
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
                return f"'{key}' must be a positive finite number, got {value!r}"
        if self.dp is not None and self.dp.adaptive and self.dp.side == "client":
            bit = metrics.get(KEY_UNCLIPPED_BIT)
            if isinstance(bit, bool) or bit not in (0, 1):
                return f"'{KEY_UNCLIPPED_BIT}' must be 0 or 1, got {bit!r}"
        for key in (KEY_ARRAYS, *self.extra_reply_records):
            if not isinstance(content.get(key), ArrayRecord):
                return f"missing ArrayRecord '{key}'"
        if self._global is None:
            return "aggregate_train called before configure_train"
        record = content[KEY_ARRAYS]
        if list(record.keys()) != list(self._global.keys()):
            return "parameter names differ from the global model"
        for k, v in record.items():
            if v.numpy().shape != self._global[k].shape:
                return f"shape mismatch for '{k}'"
        # A single NaN/inf would survive any averaging, clipping or noise and ruin the global model for good
        for key in (KEY_ARRAYS, *self.extra_reply_records):
            for k, v in content[key].items():
                if not np.isfinite(v.numpy()).all():
                    return f"non-finite values in '{key}/{k}'"
        return None

    def _valid_replies(self, replies: Iterable[Message], train: bool) -> list[Message]:
        valid, failures = [], 0
        for msg in replies:
            if msg.has_error():
                failures += 1
                log(INFO, "\t> Received error in reply from node %d: %s", msg.metadata.src_node_id, msg.error.reason)
            elif (reason := self._check_reply(msg.content, train)) is not None:
                failures += 1
                log(WARNING, "\t> Dropping malformed reply from node %d: %s", msg.metadata.src_node_id, reason)
            else:
                valid.append(msg)
        log(INFO, "%s: Received %d results and %d failures",
            "aggregate_train" if train else "aggregate_evaluate", len(valid), failures)
        return valid

    def _to_update(self, msg: Message) -> ClientUpdate:
        content = msg.content
        local = to_arrays(content[KEY_ARRAYS])
        delta = {k: local[k].astype(np.float64) - self._global[k].astype(np.float64) for k in local}
        metrics = dict(content[KEY_METRICS])
        return ClientUpdate(
            node_id=msg.metadata.src_node_id,
            delta=delta,
            weight=float(metrics[self.weighted_by_key]),
            metrics=metrics,
            extras={k: to_arrays(content[k]) for k in self.extra_reply_records},
        )

    # Hooks -----------------------------------------------------------------
    def select_updates(self, updates: list[ClientUpdate]) -> list[ClientUpdate]:
        return updates

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        return weighted_mean([u.delta for u in updates], [u.weight for u in updates])

    def update_global(self, server_round: int, delta: Arrays) -> Arrays:
        """New global model = global + server_learning_rate * delta."""
        return self._step(self.server_learning_rate, delta)

    def _step(self, scale: float, delta: Arrays) -> Arrays:
        """global + scale * delta, in the dtype of the global model."""
        return {
            k: cast_like(g.astype(np.float64) + scale * delta[k], g.dtype) for k, g in self._global.items()
        }

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid = self._valid_replies(replies, train=True)
        if not valid:
            return None, None
        # Replies arrive in arbitrary order; aggregate in a fixed one so float sums are reproducible
        valid.sort(key=lambda m: self._rank.get(m.metadata.src_node_id, len(self._rank)))
        updates = self.select_updates([self._to_update(m) for m in valid])
        self._round_metrics = {}

        dp_metrics: dict = {}
        if self.dp is not None:
            bits = [u.metrics.get(KEY_UNCLIPPED_BIT) for u in updates]
            delta, dp_metrics = self.dp.aggregate([u.delta for u in updates], self._num_sampled, bits)
            log(INFO, "\tDP: clip=%.4g noise_std=%.3g epsilon=%.4f (delta=%g)",
                dp_metrics["dp-clip-norm"], dp_metrics["dp-noise-std"], dp_metrics["dp-epsilon"], self.dp.delta)
        else:
            delta = self.aggregate_updates(updates)

        new_global = self.update_global(server_round, delta)
        metrics = weighted_metrics([u.metrics for u in updates], [u.weight for u in updates], self.weighted_by_key)
        if self.dp is not None:
            metrics.pop(KEY_UNCLIPPED_BIT, None)  # their exact mean is the clipped count the DP noise protects
        metrics.pop(KEY_UPLOAD_BYTES, None)
        metrics.update(dp_metrics)
        metrics.update(self._round_metrics)
        metrics.update(self._communication(valid))
        return to_record(new_global), MetricRecord(metrics)

    def _communication(self, replies: list[Message]) -> dict[str, float]:
        """Training traffic of this round in MiB, and the running total of the run.

        Uploads are what the clients report in ``upload-bytes`` (compressed size) or, if they do not,
        the payload of the ArrayRecords they sent. Evaluation messages are not counted.
        """
        up = 0
        for msg in replies:
            reported = msg.content[KEY_METRICS].get(KEY_UPLOAD_BYTES)
            up += int(reported) if reported is not None else sum(record_bytes(r) for r in msg.content.array_records.values())
        self._comm_total += up
        return {"comm-mb-down": self._down_bytes / MB, "comm-mb-up": up / MB, "comm-mb-total": self._comm_total / MB}

    # ----------------------------------------------------------------- evaluate
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.fraction_evaluate == 0.0:
            return []
        node_ids, population = self._sample(grid, self.fraction_evaluate, self.min_evaluate_nodes)
        log(INFO, "configure_evaluate: Sampled %d nodes (out of %d)", len(node_ids), population)
        config = ConfigRecord(dict(config))
        config["server-round"] = server_round
        content = RecordDict({KEY_ARRAYS: arrays, KEY_CONFIG: config})
        return self._messages(content, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        valid = self._valid_replies(replies, train=False)
        if not valid:
            return None
        metrics = [dict(m.content[KEY_METRICS]) for m in valid]
        weights = [float(m[self.weighted_by_key]) for m in metrics]
        aggregated = weighted_metrics(metrics, weights, self.weighted_by_key)
        for key in FAIRNESS_METRICS:
            aggregated.update(client_spread(metrics, key))
        return MetricRecord(aggregated)
