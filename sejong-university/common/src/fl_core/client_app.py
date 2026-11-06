import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_core.task import build_model, client_indices, load_split, test, train


def seed_for(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def model_for(cfg):
    return build_model(cfg["dataset"], cfg.get("model", "cnn"), cfg.get("model-width", 1.0))


def local_data(cfg, partition_id, num_partitions, split="train"):
    x, y = load_split(cfg["dataset"], "train")
    tr, va = client_indices(cfg, partition_id, num_partitions)
    idx = tr if split == "train" else va
    return x[idx], y[idx]


def node_info(context):
    return int(context.node_config["partition-id"]), int(context.node_config["num-partitions"])


def make_train_handler(data_fn=None, model_fn=None, data_transform=None, loss_factory=None, param_lr=None,
                       report_fn=None, epochs_key="local-epochs"):
    data_fn = data_fn or local_data
    model_fn = model_fn or model_for

    def handler(msg: Message, context: Context) -> Message:
        cfg = context.run_config
        config = msg.content["config"]
        pid, num = node_info(context)
        rnd = int(config.get("server-round", 0))
        x, y = data_fn(cfg, pid, num)
        if data_transform is not None:
            x, y = data_transform(x, y, cfg, pid, config)
        model = model_fn(cfg)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        torch.manual_seed(seed_for(cfg["seed"], pid, rnd))
        initial_loss, _ = test(model, x, y)
        extra = loss_factory(model, msg, context) if loss_factory is not None else None
        loss, steps = train(model, x, y, np.arange(len(y)), epochs=float(cfg[epochs_key]),
                            batch_size=int(cfg["batch-size"]), lr=float(config.get("lr", cfg["learning-rate"])),
                            momentum=float(cfg.get("momentum", 0.9)),
                            proximal_mu=float(config.get("proximal-mu", 0.0)),
                            seed=seed_for(cfg["seed"], pid, rnd, 1), extra_loss=extra,
                            param_lr=param_lr(msg) if param_lr is not None else None)
        metrics = {"train_loss": float(loss), "initial-loss": float(max(initial_loss, 1e-12)),
                   "num-examples": int(len(y)), "local-steps": int(steps)}
        if report_fn is not None:
            metrics.update(report_fn(x, y, cfg, pid, config))
        content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    return handler


def make_evaluate_handler(data_fn=None, model_fn=None):
    data_fn = data_fn or local_data
    model_fn = model_fn or model_for

    def handler(msg: Message, context: Context) -> Message:
        cfg = context.run_config
        pid, num = node_info(context)
        x, y = data_fn(cfg, pid, num, "val")
        model = model_fn(cfg)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        loss, acc = test(model, x, y)
        metrics = {"eval_loss": float(loss), "eval_acc": float(acc), "num-examples": int(max(len(y), 1))}
        return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)

    return handler


train_handler = make_train_handler()
evaluate_handler = make_evaluate_handler()

app = ClientApp()


@app.train()
def _train(msg: Message, context: Context):
    return train_handler(msg, context)


@app.evaluate()
def _evaluate(msg: Message, context: Context):
    return evaluate_handler(msg, context)
