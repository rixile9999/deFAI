import copy

import torch
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from fl_core.client_app import node_info, seed_for
from fl_core.plugins import Plugin
from fl_core.strategies import StrategyWrapper
from layer_kd.trainer import KDConfig, KDTrainer


class TeacherStrategy(StrategyWrapper):
    def __init__(self, inner, teacher_arrays):
        super().__init__(inner)
        self.teacher = teacher_arrays

    def configure_train(self, server_round, arrays, config, grid):
        messages = list(self.inner.configure_train(server_round, arrays, config, grid))
        for m in messages:
            m.content["teacher"] = self.teacher
        return messages


def make_kd_handler(backend, kd_config, student_width=0.5, teacher_width=1.0):
    state = {}

    def handler(msg, context):
        cfg = context.run_config
        config = msg.content["config"]
        pid, num = node_info(context)
        rnd = int(config.get("server-round", 0))
        x, y = backend.data_fn(cfg, pid, num)
        student = backend.build_model(width=student_width)
        student.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        teacher = None
        if "teacher" in msg.content:
            teacher = backend.build_model(width=teacher_width)
            teacher.load_state_dict(msg.content["teacher"].to_torch_state_dict())
        torch.manual_seed(seed_for(cfg["seed"], pid, rnd))
        trainer = KDTrainer(copy.deepcopy(kd_config), teacher, student, backend.input_shape())
        if pid in state and state[pid]["keys"] == list(trainer.aux.state_dict()):
            trainer.aux.load_state_dict(state[pid]["aux"])
            trainer.disc.load_state_dict(state[pid]["disc"])
        logs, steps = trainer.fit(x, y, epochs=float(cfg["local-epochs"]), batch_size=int(cfg["batch-size"]),
                                  lr=float(config.get("lr", cfg["learning-rate"])), seed=seed_for(cfg["seed"], pid, rnd, 2))
        state[pid] = {"keys": list(trainer.aux.state_dict()), "aux": trainer.aux.state_dict(),
                      "disc": trainer.disc.state_dict()}
        trainer.close()
        metrics = {"train_loss": float(logs.get("ce", 0.0)), "kd_loss": float(logs.get("loss", 0.0)),
                   "num-examples": int(len(y)), "local-steps": int(steps), "initial-loss": float(logs.get("ce", 1.0))}
        content = RecordDict({"arrays": ArrayRecord(student.state_dict()), "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    return handler


class KDPlugin(Plugin):
    name = "layer_kd"

    def __init__(self, config=None, teacher_arrays=None):
        defaults = {"method": "combined", "student-width": 0.5, "teacher-width": 1.0}
        super().__init__({**defaults, **(config or {})})
        self.teacher_arrays = teacher_arrays

    def kd_config(self):
        keys = set(KDConfig.__dataclass_fields__) - {"parts"}
        return KDConfig(**{k: v for k, v in self.config.items() if k in keys})

    def wrap_strategy(self, strategy, tier, owner):
        if tier in ("flat", "edge") and self.teacher_arrays is not None:
            return TeacherStrategy(strategy, self.teacher_arrays)
        return strategy
