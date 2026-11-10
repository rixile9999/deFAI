import torch
from flwr.app import ConfigRecord, Context, RecordDict

from fl_core import client_app
from fl_core.config import BUILTIN_DEFAULTS, merge
from fl_core.grid import InProcessGrid, Node
from fl_core.strategies import build_flower_strategy, relax_minimums
from fl_core.task import DATASETS, load_split


def ensure_identity(run_id=1, node_id=0, task_id=1):
    from flwr.supercore.task_identity import TaskIdentity

    if TaskIdentity._run_id is None:
        TaskIdentity.run_id = run_id
    if TaskIdentity._node_id is None:
        TaskIdentity.node_id = node_id
    if TaskIdentity._task_id is None:
        TaskIdentity.task_id = task_id


class Backend:
    name = "base"

    def __init__(self, overrides=None):
        self.cfg = self.default_config()
        self.cfg = merge(self.cfg, overrides or {})
        self.validate()
        self.train_handler = None
        self.evaluate_handler = None

    def default_config(self):
        raise NotImplementedError

    def validate(self):
        pass

    def context(self, partition_id, num_partitions, run_config=None):
        return Context(run_id=1, node_id=partition_id + 1,
                       node_config={"partition-id": partition_id, "num-partitions": num_partitions},
                       state=RecordDict(), run_config=dict(run_config or self.cfg))

    def nodes(self, num_clients, train=None, evaluate=None):
        return {k + 1: Node(k + 1, self.context(k, num_clients), train or self.train_handler,
                            evaluate or self.evaluate_handler) for k in range(num_clients)}

    def grid(self, num_clients, workers=1, middleware=(), train=None, evaluate=None):
        ensure_identity()
        return InProcessGrid(self.nodes(num_clients, train, evaluate), workers, middleware)

    def train_config(self):
        return ConfigRecord({"lr": float(self.cfg["learning-rate"])})

    def data_fn(self, cfg, partition_id, num_partitions, split="train"):
        return self.client_data(partition_id, num_partitions, split)

    def model_fn(self, cfg):
        return self.build_model()

    def custom_train_handler(self, **hooks):
        return client_app.make_train_handler(data_fn=self.data_fn, model_fn=hooks.pop("model_fn", self.model_fn),
                                             **hooks)

    def custom_evaluate_handler(self, model_fn=None):
        return client_app.make_evaluate_handler(data_fn=self.data_fn, model_fn=model_fn or self.model_fn)

    @property
    def lr(self):
        return float(self.cfg["learning-rate"])


class BuiltinBackend(Backend):
    name = "builtin"

    def __init__(self, overrides=None):
        super().__init__(overrides)
        self.train_handler = client_app.train_handler
        self.evaluate_handler = client_app.evaluate_handler

    def default_config(self):
        return dict(BUILTIN_DEFAULTS)

    def validate(self):
        if self.cfg["dataset"] not in DATASETS:
            raise ValueError(f"unknown dataset {self.cfg['dataset']!r}")

    def build_model(self, width=None, arch=None):
        from fl_core.task import build_model

        return build_model(self.cfg["dataset"], arch or self.cfg["model"],
                           self.cfg["model-width"] if width is None else width)

    def test_set(self):
        return load_split(self.cfg["dataset"], "test")

    def client_data(self, partition_id, num_partitions, split="train"):
        return client_app.local_data(self.cfg, partition_id, num_partitions, split)

    def input_shape(self):
        spec = DATASETS[self.cfg["dataset"]]
        return (1, spec["channels"], spec["size"], spec["size"])

    def build_strategy(self, name="fedavg", fraction_train=1.0, fraction_evaluate=0.0, min_nodes=1, **extra):
        return build_flower_strategy(name, self.cfg, fraction_train, fraction_evaluate, min_nodes, **extra)


def _concat(loader):
    xs, ys = [], []
    for batch in loader:
        xs.append(batch["x"] if not isinstance(batch["x"], list) else torch.stack(batch["x"]))
        ys.append(torch.as_tensor(batch["y"]))
    return torch.cat(xs).float(), torch.cat(ys).long()


class KistBackend(Backend):
    name = "kist"

    def __init__(self, overrides=None):
        import defai.client_app
        import defai.task

        self._kist_client = defai.client_app
        self._kist_task = defai.task
        super().__init__(overrides)
        self.train_handler = defai.client_app.train
        self.evaluate_handler = defai.client_app.evaluate
        self._test = None
        self._clients = {}

    def default_config(self):
        from defai.simulation import default_config

        cfg = dict(default_config())
        cfg.update({"dataset": "synthetic", "fraction-train": 1.0, "fraction-evaluate": 0.0, "seed": 0})
        return cfg

    def validate(self):
        from defai.config import validate_config

        validate_config(self.cfg)

    def build_model(self, width=None, arch=None):
        model = self._kist_task.build_model(self.cfg["dataset"])
        if width is None or width == 1.0:
            return model
        from fl_core.task import Net

        spec = self._kist_task.DATASETS[self.cfg["dataset"]]
        return Net(spec["channels"], 10, spec["img_size"], width)

    def test_set(self):
        if self._test is None:
            self._test = _concat(self._kist_task.load_centralized_testset(self.cfg["dataset"]))
        return self._test

    def client_data(self, partition_id, num_partitions, split="train"):
        key = (partition_id, num_partitions, split)
        if key not in self._clients:
            tr, va = self._kist_task.load_client_data(partition_id, num_partitions, self.cfg)
            self._clients[key] = _concat(tr if split == "train" else va)
        return self._clients[key]

    def input_shape(self):
        spec = self._kist_task.DATASETS[self.cfg["dataset"]]
        return (1, spec["channels"], spec["img_size"], spec["img_size"])

    def build_strategy(self, name="fedavg", fraction_train=1.0, fraction_evaluate=0.0, min_nodes=1, **extra):
        from defai.strategies import build_strategy

        cfg = dict(self.cfg)
        cfg.update({"strategy": name, "fraction-train": fraction_train, "fraction-evaluate": fraction_evaluate})
        return relax_minimums(build_strategy(cfg), min_nodes)


BACKENDS = {"builtin": BuiltinBackend, "kist": KistBackend}


def get_backend(name="builtin", overrides=None):
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[name](overrides)


def kist_available():
    try:
        import defai.client_app
        import defai.strategies
    except Exception:
        return False
    return defai.client_app is not None and defai.strategies is not None
