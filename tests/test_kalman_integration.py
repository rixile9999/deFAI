import numpy as np
import pytest

from conftest import QuadraticClients, run_strategy
from defai.config import validate_config
from defai.kalman import KalmanConfig
from defai.privacy.dp import FixedClipping
from defai.simulation import default_config
from defai.strategies import FedAdamKalman, build_strategy


def test_factory_routes_all_parameters_and_server_learning_rate():
    cfg = {**default_config(), "strategy": "fedadam-kalman", "fedopt-learning-rate": .007,
           "kf-process-adaptation": .2, "kf-rho": .8, "kf-blend": .4, "kf-max-step-norm": .02}
    validate_config(cfg)
    s = build_strategy(cfg)
    assert isinstance(s, FedAdamKalman)
    assert (s.kalman.rho, s.kalman.process_adaptation, s.kalman_blend, s.max_step_norm, s.server_learning_rate) == (.8, .2, .4, .02, .007)


@pytest.mark.parametrize("cfg", [{"kf-blend": True}, {"kf-rho": 3}, {"kf-gain-min": .8, "kf-gain-max": .1},
                                 {"kf-process-adaptation": np.inf}, {"kf-max-step-norm": -1}])
def test_invalid_run_config_fails_before_training(cfg):
    with pytest.raises(ValueError, match="kf"):
        validate_config({**default_config(), "strategy": "fedadam-kalman", **cfg})


def test_full_strategy_loop_reports_filter_diagnostics_and_converges():
    targets = {i: np.array([1., -2.]) for i in range(1, 5)}
    s = FedAdamKalman(server_learning_rate=.08, kalman=KalmanConfig(process_noise=.01), seed=2)
    w, result, _ = run_strategy(s, QuadraticClients(targets), 4, rounds=60, dim=2)
    assert np.linalg.norm(w - [1., -2.]) < .15
    assert result.train_metrics_clientapp[1]["kf-gain-min"] > 0
    assert s.optimizer_state_bytes() > 0


def test_privacy_pipeline_exposes_only_the_noised_aggregate_to_filter():
    s = FedAdamKalman(dp=FixedClipping(.5, 1., seed=7), seed=2)
    released = []
    original = s.dp.aggregate
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        released.append(result[0])
        return result
    s.dp.aggregate = record
    seen = []
    update = s.update_global
    def spy(rnd, delta):
        seen.append(delta)
        return update(rnd, delta)
    s.update_global = spy
    _, result, _ = run_strategy(s, QuadraticClients({i: np.ones(2) for i in range(1, 5)}), 4, rounds=2, dim=2)
    for a, b in zip(seen, released):
        np.testing.assert_array_equal(a["w"], b["w"])
    assert "dp-epsilon" in result.train_metrics_clientapp[2]


def test_real_client_training_cli_saves_kalman_history(tmp_path):
    import json
    from defai.cli import main
    output = tmp_path / "kalman run with spaces"
    assert main(["run", "--strategy", "fedadam-kalman", "--clients", "2", "--rounds", "2",
                 "--set", "kf-process-adaptation=0.2", "--output", str(output)]) == 0
    history = json.loads((output / "history.json").read_text())
    assert history["config"]["strategy"] == "fedadam-kalman"
    assert history["research_only"] is True
    assert "kf-gain-min" in json.dumps(history)


def test_benchmark_cli_creates_strict_json_and_rejects_overwrite(tmp_path):
    import json
    from defai.cli import main
    output = tmp_path / "paired.json"
    args = ["kalman-benchmark", "--rounds", "4", "--output", str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert len(result["results"]) == 12
    assert main(args) == 2


def test_real_training_snapshot_contains_final_model_and_valid_optimizer(tmp_path):
    import json
    from defai.cli import main
    output, snapshot = tmp_path/"run", tmp_path/"optimizer.npz"
    assert main(["run", "--strategy", "fedadam-kalman", "--clients", "2", "--rounds", "2",
                 "--optimizer-checkpoint", str(snapshot), "--output", str(output)]) == 0
    resumed = build_strategy({**default_config(), "strategy": "fedadam-kalman"})
    model = resumed.load_checkpoint(snapshot)
    assert resumed._last_round == 2 and model
    assert json.loads((output/"history.json").read_text())["optimizer_snapshot"] == str(snapshot.resolve())
    assert all(np.isfinite(a).all() for a in model.values())


def test_unsupported_snapshot_request_fails_before_training_creates_output(tmp_path):
    from defai.cli import main
    output = tmp_path/"invalid"
    assert main(["run", "--strategy", "fedavg", "--optimizer-checkpoint", str(tmp_path/"bad.npz"),
                 "--output", str(output)]) == 2
    assert not output.exists()
