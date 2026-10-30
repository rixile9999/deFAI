"""Installation-facing APIs and research-workspace contracts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from defai.aggregation import weighted_mean
from defai.cli import main
from defai.research import create_workspace, list_features, run_feature
from defai.research.resources import catalog, source_path


def test_catalog_covers_every_research_python_file_and_only_known_resources():
    root = Path(__file__).resolve().parents[1]
    actual = {p.relative_to(root).as_posix() for p in (root / "disease_research").rglob("*.py")
              if "__pycache__" not in p.parts and "build" not in p.parts}
    assert {m["path"] for m in list_features()} == actual
    assert len({m["id"] for m in list_features()}) == len(actual)
    for relative in catalog()["resources"]:
        assert hashlib.sha256(source_path(relative).read_bytes()).hexdigest() == catalog()["resource_hashes"][relative]


def test_workspace_rejects_overwrite_and_can_run_actual_extracted_script(tmp_path):
    target = create_workspace(tmp_path / "workspace with spaces")
    assert (target / "pyproject.toml").is_file()
    with pytest.raises(FileExistsError):
        create_workspace(target)
    feature = next(f for f in list_features("neonatal") if f["path"].endswith("/prepare_data.py"))
    assert run_feature(feature["id"], target, ["--help"], timeout=60) == 0
    receipt = json.loads(next((target / "run_receipts").glob("*.json")).read_text())
    assert receipt["returncode"] == 0 and receipt["device"] == "cpu"
    assert receipt["elapsed_seconds"] >= 0


def test_workspace_rejects_unknown_library_and_uninitialized_execution(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        run_feature("../../malicious.py", tmp_path)
    library = next(f for f in list_features() if f["role"] == "library")
    with pytest.raises(ValueError, match="not a standalone"):
        run_feature(library["id"], tmp_path)
    script = next(f for f in list_features() if f["role"] == "script")
    with pytest.raises(ValueError, match="Not an initialized"):
        run_feature(script["id"], tmp_path)


def test_lightweight_discovery_does_not_import_training_or_open_gpu():
    code = "import sys; from defai.research import list_features; assert list_features(); assert 'torch' not in sys.modules; assert 'flwr' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_cli_rejects_unknown_options_and_features(capsys):
    assert main(["research", "show", "not_a_feature"]) == 2
    assert "Unknown feature" in capsys.readouterr().err
    assert main(["run", "--output", "unused", "--set", "not_an_assignment"]) == 2


@pytest.mark.parametrize("weights", ([np.inf], [np.nan], [-1], [0], [[1]]))
def test_aggregation_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_mean([{"w": np.array([1., 2.])}], weights)


@pytest.mark.parametrize("updates", ([], [{"w": np.array([np.nan])}],
                                     [{"w": np.ones(2)}, {"w": np.ones(3)}],
                                     [{"w": np.ones(2)}, {"other": np.ones(2)}]))
def test_aggregation_rejects_invalid_client_arrays(updates):
    with pytest.raises(ValueError):
        weighted_mean(updates, [1] * len(updates))


def test_streaming_mean_matches_independent_stacked_reference():
    rng = np.random.default_rng(4)
    updates = [{"w": rng.normal(size=(5, 7)).astype(np.float32), "count": np.array([i], dtype=np.int64)} for i in range(13)]
    weights = rng.uniform(.1, 10, 13)
    result = weighted_mean(updates, weights)
    for key in result:
        expected = np.average(np.stack([u[key] for u in updates]).astype(np.float64), axis=0, weights=weights)
        if key == "count":
            expected = np.rint(expected)
        np.testing.assert_allclose(result[key], expected, atol=1e-6)
        assert result[key].dtype == updates[0][key].dtype


@pytest.mark.parametrize("values", ([], [0, 1], [-1, 1], [np.nan], [np.inf], [[1, 2]]))
def test_generation_rejects_invalid_intervals(values):
    from defai.research.generation import DistrictNode
    with pytest.raises(ValueError, match="positive 1D"):
        DistrictNode(11001, values)


def test_generation_fit_matches_central_likelihood():
    from scipy.stats import burr12
    from defai.research.generation import build_hierarchy, federated_fit, pooled_reference
    intervals = burr12.rvs(1.7, 1.5, scale=5.6, size=4000, random_state=np.random.default_rng(0))
    _, provinces = build_hierarchy({11001+i: a for i, a in enumerate(np.array_split(intervals, 4))})
    _, nll, _, result = federated_fit(provinces)
    _, reference = pooled_reference(intervals)
    assert result.success and abs(nll-reference) / len(intervals) < 1e-6


@pytest.mark.parametrize("key,value", (("seed", -1), ("batch-size", 1.5), ("local-epochs", True),
                                      ("dp-clip-norm", np.inf), ("learning-rate", "fast"),
                                      ("server-learning-rate", np.nan)))
def test_config_rejects_nonfinite_wrong_type_and_fractional_counts(key, value):
    from defai.config import validate_config
    with pytest.raises(ValueError):
        validate_config({key: value})


def test_hba1c_import_preserves_thread_settings():
    import torch
    previous = torch.get_num_threads()
    from defai.research.hba1c import initial_parameters, local_fit
    assert torch.get_num_threads() == previous
    x, y = torch.ones(2, 3, 20), torch.zeros(2, 3)
    p = [v[None].expand(2, *v.shape).clone() for v in initial_parameters(9, hidden=4, device="cpu")]
    with pytest.raises(ValueError, match="observed sample"):
        local_fit(x, y, torch.zeros_like(y), p, 1, .1)


def test_atomic_outputs_never_overwrite_or_publish_nan(tmp_path):
    from defai.io import write_json
    path = tmp_path / "nested/out.json"
    write_json(path, {"value": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        write_json(path, {"value": 2}, overwrite=False)
    with pytest.raises(ValueError):
        write_json(path, {"value": np.nan})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(path.parent.glob(".defai-*"))
