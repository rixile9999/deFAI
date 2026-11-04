"""Export completed common-model runs without coupling consumers to deFAI/Flower."""

import hashlib
import json
import math
from pathlib import Path
import pickle
import tempfile

import numpy as np

from defai import __version__

FORMAT = "federated-result-v1"
MAX_HISTORY_BYTES = 4 * 1024 * 1024


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _history(path):
    if path.stat().st_size > MAX_HISTORY_BYTES:
        raise ValueError("history.json exceeds the 4 MiB interchange limit")
    data = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_invalid_constant
    )
    if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
        raise ValueError("Expected a completed run's config and history")
    rows = data.get("history")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10001:
        raise ValueError("Run history must have 1 to 10001 rows")
    previous = -1
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each history row must be an object")
        number = row.get("round")
        if type(number) is not int or number <= previous:
            raise ValueError(
                "History rounds must be nonnegative and strictly increasing"
            )
        previous = number
        for key, value in row.items():
            if key == "round" or value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("History cells must be finite numeric scalars or null")
        for key in ("test_acc", "test_loss"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Missing or invalid {key}")
        if not 0 <= row["test_acc"] <= 1 or row["test_loss"] < 0:
            raise ValueError("Accuracy/loss outside their declared ranges")
    return data


def export_result(run_directory, output):
    """Create a fresh result.json/weights.npz/model.ts bundle atomically.

    Input is a trusted local deFAI common-model run (history.json plus numeric
    final_model.pt), never a downloaded executable model. This exports inference
    weights, not optimizer/client state or a complete training-resume protocol.
    """
    run, output = Path(run_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    history_path, model_path = run / "history.json", run / "final_model.pt"
    result = _history(history_path)
    history_hash, source_model_hash = file_hash(history_path), file_hash(model_path)
    dataset = result["config"].get("dataset")
    strategy = result["config"].get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Run config is missing strategy")
    import torch
    from defai.task import DATASETS, build_model

    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError("Export supports the common image-model datasets only")
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model = build_model(dataset).cpu().eval()
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError(
                "Model tensor keys do not match the declared dataset model"
            )
        for key, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Model tensor shape/dtype/value mismatch: {key}")
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("Cannot restore the run's numeric model state") from exc
    arrays = {key: value.detach().cpu().numpy() for key, value in state.items()}
    spec = DATASETS[dataset]
    shape = [spec["channels"], spec["img_size"], spec["img_size"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".defai-export-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        np.savez(stage / "weights.npz", **arrays)
        with torch.no_grad():
            traced = torch.jit.trace(model, torch.zeros(1, *shape))
        # File objects work with non-ASCII Windows paths as well.
        with (stage / "model.ts").open("wb") as stream:
            torch.jit.save(traced, stream)
        manifest = {
            "format": FORMAT,
            "origin": {"engine": "deFAI", "version": __version__},
            "source_history_sha256": history_hash,
            "source_model_sha256": source_model_hash,
            "result": result,
            "metric_specs": {
                "test_acc": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "higher_is_better": True,
                },
                "test_loss": {"unit": "cross_entropy", "higher_is_better": False},
            },
            "model": {
                "framework": "pytorch",
                "inference_format": "torchscript",
                "input_schema": {
                    "shape": [None, *shape],
                    "dtype": "float32",
                    "preprocessing": "already_normalized_tensor",
                    "normalization": spec["norm"],
                },
                "output_schema": {
                    "shape": [None, 10],
                    "dtype": "float32",
                    "meaning": "logits",
                },
                "runtime": {"torch": str(torch.__version__)},
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": value.dtype.str}
                    for key, value in arrays.items()
                },
            },
            "artifacts": {
                "weights": {
                    "path": "weights.npz",
                    "sha256": file_hash(stage / "weights.npz"),
                },
                "inference": {
                    "path": "model.ts",
                    "sha256": file_hash(stage / "model.ts"),
                },
            },
        }
        encoded = (
            json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        )
        if len(encoded.encode("utf-8")) > MAX_HISTORY_BYTES:
            raise ValueError("Result manifest exceeds the 4 MiB interchange limit")
        (stage / "result.json").write_text(encoded, encoding="utf-8")
        if (
            file_hash(history_path) != history_hash
            or file_hash(model_path) != source_model_hash
        ):
            raise ValueError("Source run changed during export; export a completed run")
        if output.exists():
            raise FileExistsError(f"Output exists: {output}")
        stage.rename(output)
    return {
        "output": str(output),
        "format": FORMAT,
        "strategy": strategy,
        "dataset": dataset,
        "inference_format": "torchscript",
    }
