"""Sequential in-process execution of real Flower strategies and ClientApp handlers.

This backend trains the actual image model but measures no network/transport cost.
Use the Flower Simulation/Deployment Engine for distributed execution.
"""
import os
from pathlib import Path
import time
import threading
import tomllib

_RUN_LOCK = threading.Lock()


def default_config():
    from defai.research.resources import source_path
    with source_path("pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["tool"]["flwr"]["app"]["config"]


def run_simulation(output, *, overrides=None, clients=4, device="cpu", threads=1):
    """Train/evaluate with persistent client states and all normal run-config axes."""
    import torch
    from flwr.app import ArrayRecord, ConfigRecord, Context, MessageType, MetricRecord, RecordDict
    from flwr.supercore.task_identity import TaskIdentity
    from defai import client_app
    from defai.config import validate_config
    from defai.results import build_history, final_arrays, privacy_summary, write_summary
    from defai.strategies import build_strategy
    from defai.task import build_model, load_centralized_testset, test

    cfg = default_config()
    overrides = dict(overrides or {})
    unknown = set(overrides) - cfg.keys()
    if unknown:
        raise ValueError(f"Unknown run-config keys: {', '.join(sorted(unknown))}")
    cfg.update(overrides)
    validate_config(cfg)
    if isinstance(clients, bool) or not isinstance(clients, int) or clients < 2:
        raise ValueError("clients must be an integer >= 2")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be an integer >= 1")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a fresh run directory")
    contexts = {n: Context(run_id=1, node_id=n,
                          node_config={"partition-id": n-1, "num-partitions": clients},
                          state=RecordDict(), run_config=dict(cfg)) for n in range(1, clients+1)}

    class LocalGrid:
        def get_node_ids(self):
            return contexts.keys()

        def send_and_receive(self, messages, timeout=None):
            replies = []
            for message in messages:
                handler = client_app.train if message.metadata.message_type == MessageType.TRAIN else client_app.evaluate
                replies.append(handler(message, contexts[message.metadata.dst_node_id]))
            return replies

    if not _RUN_LOCK.acquire(blocking=False):
        raise ValueError("In-process simulations are sequential; use separate processes for concurrent runs")
    old_device = os.environ.get("DEFAI_DEVICE")
    old_threads = torch.get_num_threads()
    identity = TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id
    os.environ["DEFAI_DEVICE"] = device
    start = time.perf_counter()
    try:
        torch.set_num_threads(threads)
        TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = 1, 0, 1
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
            torch.manual_seed(cfg["seed"])
            model = build_model(cfg["dataset"])
            testloader = load_centralized_testset(cfg["dataset"])

            def evaluate(server_round, arrays):
                model.load_state_dict(arrays.to_torch_state_dict())
                loss, accuracy = test(model, testloader, torch.device(device))
                return MetricRecord({"test_loss": loss, "test_acc": accuracy})

            strategy = build_strategy(cfg)
            initial = ArrayRecord(model.state_dict())
            result = strategy.start(grid=LocalGrid(), initial_arrays=initial,
                                    train_config=ConfigRecord({"lr": cfg["learning-rate"]}),
                                    num_rounds=cfg["num-server-rounds"], evaluate_fn=evaluate)
            summary = {"config": cfg, "history": build_history(result), "backend": "in_process",
                       "clients": clients, "device": device, "threads": threads,
                       "elapsed_seconds": time.perf_counter()-start,
                       "research_only": True, "network_transport_measured": False}
            if strategy.dp is not None:
                summary["privacy"] = privacy_summary(strategy.dp)
            output.mkdir(parents=True, exist_ok=False)
            write_summary(output / "history.json", summary)
            if cfg["save-model"]:
                torch.save(final_arrays(result, initial).to_torch_state_dict(), output / "final_model.pt")
            return summary
    finally:
        torch.set_num_threads(old_threads)
        TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = identity
        if old_device is None:
            os.environ.pop("DEFAI_DEVICE", None)
        else:
            os.environ["DEFAI_DEVICE"] = old_device
        _RUN_LOCK.release()
