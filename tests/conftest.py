import numpy as np
import pytest
from flwr.app import Message, MessageType, MetricRecord, RecordDict
from flwr.supercore.task_identity import TaskIdentity

from defai.records import to_arrays, to_record


@pytest.fixture(autouse=True)
def task_identity():
    """Flower stamps every Message with the identity of the running task."""
    TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = 1, 0, 1


class FakeGrid:
    """In-process stand-in for Flower's Grid: delivers messages straight to ``client_fn``."""

    def __init__(self, num_nodes, client_fn):
        self.nodes = list(range(1, num_nodes + 1))
        self.client_fn = client_fn
        self.sent = []  # every message that was sent, in order

    def get_node_ids(self):
        return self.nodes

    def send_and_receive(self, messages, timeout=None):
        messages = list(messages)
        self.sent.extend(messages)
        return [self.client_fn(m) for m in messages]


class QuadraticClients:
    """Client ``i`` minimises f_i(w) = 0.5 * ||w - t_i||^2 with ``steps`` full-batch GD steps.

    The federated optimum is mean(t_i). Implements the client halves of FedProx and
    SCAFFOLD in plain numpy so the *server* logic can be tested without torch/Flower runtime.
    """

    def __init__(self, targets, lr=0.1, steps=5, num_examples=None, bad=()):
        self.targets, self.lr, self.steps, self.bad = targets, lr, steps, set(bad)
        self.num_examples = num_examples or {n: 10 for n in targets}
        self.control = {}  # SCAFFOLD client state: node -> c_i

    def __call__(self, msg):
        node = msg.metadata.dst_node_id
        content = msg.content
        w = to_arrays(content["arrays"])["w"].astype(np.float64)
        t = self.targets[node]
        if msg.metadata.message_type == MessageType.EVALUATE:
            metrics = {"eval_loss": float(0.5 * np.sum((w - t) ** 2)), "num-examples": self.num_examples[node]}
            return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)

        config = content["config"]
        mu = config.get("proximal-mu", 0.0)
        scaffold = "c" in content
        if scaffold:
            c = to_arrays(content["c"])["w"].astype(np.float64)
            c_i = self.control.get(node, np.zeros_like(c))
        steps = self.steps[node] if isinstance(self.steps, dict) else self.steps
        y = w.copy()
        for _ in range(steps):
            grad = (y - t) + mu * (y - w)
            if scaffold:
                grad = grad - c_i + c
            y = y - self.lr * grad
        reply = {"arrays": to_record({"w": y.astype(np.float32)})}
        if scaffold:
            c_new = c_i - c + (w - y) / (steps * self.lr)
            self.control[node] = c_new
            reply["c-delta"] = to_record({"w": (c_new - c_i).astype(np.float32)})
        if node in self.bad:
            reply["arrays"] = to_record({"w": np.full_like(y, 1e3, dtype=np.float32)})
        metrics = {"train_loss": float(0.5 * np.sum((w - t) ** 2)), "num-examples": self.num_examples[node], "local-steps": steps}
        if config.get("report-initial-loss", False):  # loss of the received model, before local training
            metrics["initial-loss"] = max(float(0.5 * np.sum((w - t) ** 2)), 1e-12)
        reply["metrics"] = MetricRecord(metrics)
        return Message(content=RecordDict(reply), reply_to=msg)


def run_strategy(strategy, clients, num_nodes, rounds, dim=4, start=None):
    """Drive ``Strategy.start`` against fake clients; returns (final w, Result, grid)."""
    from flwr.app import ConfigRecord

    grid = FakeGrid(num_nodes, clients)
    start = np.zeros(dim, dtype=np.float32) if start is None else start
    result = strategy.start(
        grid=grid,
        initial_arrays=to_record({"w": start}),
        num_rounds=rounds,
        train_config=ConfigRecord({"lr": clients.lr}),
    )
    return to_arrays(result.arrays)["w"].astype(np.float64), result, grid


class RegistrationOrderGrid(FakeGrid):
    """Like the simulation: random node ids, registered one after the other (``get_nodes`` order is not id order)."""

    def __init__(self, num_nodes, client_fn, seed):
        super().__init__(num_nodes, client_fn)
        rng = np.random.default_rng(seed)
        self.nodes = [int(i) for i in rng.integers(1, 2**62, size=num_nodes)]  # position i == partition i
        self.targets = dict(enumerate(self.nodes))

    def get_node_ids(self):
        return set(self.nodes)  # unordered, as in the SuperLink state

    def get_nodes(self):
        from flwr.proto.node_pb2 import NodeInfo

        shuffled = list(self.nodes)
        np.random.default_rng(0).shuffle(shuffled)  # the server must not rely on delivery order either
        position = {n: i for i, n in enumerate(self.nodes)}
        return [NodeInfo(node_id=n, registered_at=f"2026-01-01T00:00:00.{position[n]:06d}+00:00") for n in shuffled]


class InProcessGrid(FakeGrid):
    """Routes messages to the real ClientApp functions, one persistent Context per node.

    Like the simulation engine: node ``i`` gets ``partition-id = i`` and keeps its ``state`` between
    rounds. This runs the whole stack (ServerApp, strategy, ClientApp, model, data) without Flower's runtime.
    """

    def __init__(self, num_nodes, run_config):
        from flwr.app import Context, RecordDict

        super().__init__(num_nodes, self._dispatch)
        self.contexts = {
            node: Context(
                run_id=1,
                node_id=node,
                node_config={"partition-id": i, "num-partitions": num_nodes},
                state=RecordDict(),
                run_config=run_config,
            )
            for i, node in enumerate(self.nodes)
        }

    def _dispatch(self, msg):
        from defai import client_app

        context = self.contexts[msg.metadata.dst_node_id]
        if msg.metadata.message_type == MessageType.EVALUATE:
            return client_app.evaluate(msg, context)
        return client_app.train(msg, context)
