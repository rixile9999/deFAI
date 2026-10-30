import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(tmp_path, name, accs, seed=0, privacy=None, columns=None, **config):
    cfg = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "seed": seed,
           "output-dir": str(tmp_path), "attack": "none", "num-malicious": 0, "dp": "none", **config}
    history = [{"round": i, "test_acc": a} for i, a in enumerate(accs)]
    for key, values in (columns or {}).items():  # extra per-round metrics, one value (or None) per round
        for row, value in zip(history, values):
            row[key] = value
    path = tmp_path / name / "history.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"config": cfg, "history": history, **({"privacy": privacy} if privacy else {})}))
    return str(path)


def test_compare_groups_seeds_and_reports_mean_std(tmp_path, capsys):
    compare = load_script("compare")
    paths = [
        write_run(tmp_path, "a0", [0.1, 0.5, 0.9], seed=0),
        write_run(tmp_path, "a1", [0.1, 0.7, 0.95], seed=1),
        write_run(tmp_path, "b", [0.1, 0.2, 0.3], strategy="scaffold"),
    ]
    rows = {r["run"]: r for r in compare.summarise(paths, target=0.9)}
    avg = rows["fedavg synthetic iid"]
    assert avg["runs"] == 2 and avg["final"] == "0.925 ± 0.035" and avg["best"] == "0.925 ± 0.035"
    assert avg["rounds_to_target"] == "2.0 (2/2)"
    assert rows["scaffold synthetic iid"]["rounds_to_target"] == "-"
    compare.main([*paths, "--target", "0.9", "--csv", str(tmp_path / "t.csv")])
    out = capsys.readouterr().out
    assert "fedavg synthetic iid" in out and "rounds to 0.9" in out
    assert (tmp_path / "t.csv").read_text().startswith("run,runs,final")


def test_compare_adds_columns_for_any_history_key(tmp_path, capsys):
    compare = load_script("compare")
    paths = [
        write_run(tmp_path, "a0", [0.1, 0.5], seed=0, columns={"eval_acc_min": [0.0, 0.2], "comm-mb-total": [0.0, 1.5]}),
        # the last round has no client-side value (null): the last recorded one is used
        write_run(tmp_path, "a1", [0.1, 0.7, 0.8], seed=1, columns={"eval_acc_min": [0.1, 0.4, None], "comm-mb-total": [0.0, 2.0, 2.5]}),
        write_run(tmp_path, "b", [0.1, 0.2], strategy="scaffold"),  # does not record the keys at all
    ]
    rows = {r["run"]: r for r in compare.summarise(paths, 0.9, extra=["eval_acc_min", "comm-mb-total"])}
    assert rows["fedavg synthetic iid"]["eval_acc_min"] == "0.300 ± 0.141"
    assert rows["fedavg synthetic iid"]["comm-mb-total"] == "2.000 ± 0.707"  # 1.5 and 2.5
    assert rows["scaffold synthetic iid"]["eval_acc_min"] == "-"
    compare.main([*paths, "--extra", "comm-mb-total", "--csv", str(tmp_path / "t.csv")])
    assert "comm-mb-total" in capsys.readouterr().out.splitlines()[0]
    assert (tmp_path / "t.csv").read_text().splitlines()[0].endswith(",comm-mb-total")


def test_compare_tail_averages_the_last_rounds(tmp_path, capsys):
    compare = load_script("compare")
    paths = [
        write_run(tmp_path, "a0", [0.1, 0.9, 0.5, 0.7], seed=0),   # last three: 0.9 0.5 0.7 -> 0.7
        write_run(tmp_path, "a1", [0.1, 0.3, 0.5, 0.9], seed=1),   # last three: 0.3 0.5 0.9 -> 0.5667
    ]
    row = compare.summarise(paths, 0.9, tail=3)[0]
    assert row["tail"] == "0.633 ± 0.094" and row["final"] == "0.800 ± 0.141"
    assert compare.summarise(paths, 0.9)[0]["tail"] == "-"  # off by default
    assert compare.summarise(paths, 0.9, tail=10)[0]["tail"] == "0.500 ± 0.071"  # shorter runs: all four rounds (0.55 and 0.45)
    compare.main([*paths, "--tail", "2"])
    header = capsys.readouterr().out.splitlines()[0].split()
    assert header[:5] == ["run", "runs", "final", "acc", "last-2"] and "mean" in header
    compare.main(paths)
    assert "last-" not in capsys.readouterr().out


def test_compare_can_print_a_markdown_table(tmp_path, capsys):
    compare = load_script("compare")
    path = write_run(tmp_path, "a", [0.1, 0.8], columns={"eval_acc_min": [0.0, 0.5]})
    compare.main([path, "--extra", "eval_acc_min", "--markdown"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and all(line.startswith("| ") and line.endswith(" |") for line in lines)
    assert lines[0].split("|")[1].strip() == "run" and lines[0].split("|")[-2].strip() == "eval_acc_min"
    assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}  # separator row
    assert lines[2].split("|")[1].strip() == "fedavg synthetic iid" and lines[2].split("|")[-2].strip() == "0.500"


def test_compare_tells_apart_groups_that_differ_only_in_other_settings(tmp_path):
    compare = load_script("compare")
    paths = [
        write_run(tmp_path, "r5", [0.1, 0.5], **{"num-server-rounds": 5, "learning-rate": 0.01}),
        write_run(tmp_path, "r10", [0.1, 0.5], **{"num-server-rounds": 10, "learning-rate": 0.01}),
    ]
    runs = [r["run"] for r in compare.summarise(paths, 0.9)]
    assert runs == ["fedavg synthetic iid num-server-rounds=10", "fedavg synthetic iid num-server-rounds=5"]
    # a setting shared by every group never clutters the label
    assert all("learning-rate" not in r for r in runs)


def test_compare_shows_attack_dp_and_epsilon(tmp_path):
    compare = load_script("compare")
    path = write_run(tmp_path, "c", [0.1, 0.8], attack="sign-flip", **{"num-malicious": 2, "dp": "fixed", "dp-side": "server", "dp-noise-multiplier": 1.0},
                     privacy={"epsilon": 12.5})
    row = compare.summarise([path], 0.9)[0]
    assert "sign-flipx2" in row["run"] and "dp-fixed-server(z=1.0)" in row["run"] and row["epsilon"] == "12.5"
    assert row["final"] == "0.800"  # a single run shows no ± term


def test_plot_script_accepts_any_history_key(tmp_path):
    pytest.importorskip("matplotlib")
    plot = load_script("plot")
    path = write_run(tmp_path, "d", [0.1, 0.5])
    data = json.loads(Path(path).read_text())
    data["history"][1]["dp-epsilon"] = 3.0
    Path(path).write_text(json.dumps(data))
    out = tmp_path / "eps.png"
    import sys
    old = sys.argv
    sys.argv = ["plot.py", path, "-m", "dp-epsilon", "-o", str(out)]
    try:
        plot.main()
    finally:
        sys.argv = old
    assert out.exists() and out.stat().st_size > 0


def test_plot_group_draws_the_mean_with_a_one_std_band_per_configuration(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    plot = load_script("plot")
    plt.close("all")
    paths = [
        write_run(tmp_path, "a0", [0.1, 0.5, 0.9], seed=0),
        write_run(tmp_path, "a1", [0.3, 0.7, 0.5], seed=1),
        write_run(tmp_path, "b", [0.1, 0.2, 0.3], strategy="scaffold"),
    ]
    out = tmp_path / "g.png"
    figure = plot.main([*paths, "--group", "--title", "two seeds", "-o", str(out)])
    ax = figure.axes[0]
    lines = ax.get_lines()
    assert [line.get_label() for line in lines] == ["fedavg synthetic iid (n=2)", "scaffold synthetic iid (n=1)"]
    assert list(lines[0].get_ydata()) == pytest.approx([0.2, 0.6, 0.7]) and list(lines[0].get_xdata()) == [0, 1, 2]
    assert ax.get_title() == "two seeds" and out.exists() and out.stat().st_size > 0
    std = [0.1414214, 0.1414214, 0.2828427]  # sample standard deviation of the two seeds in each round
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert band.min() == pytest.approx(min(m - d for m, d in zip([0.2, 0.6, 0.7], std)), abs=1e-6)
    assert band.max() == pytest.approx(max(m + d for m, d in zip([0.2, 0.6, 0.7], std)), abs=1e-6)
    flat = ax.collections[1].get_paths()[0].vertices[:, 1]  # a single run: no spread
    assert sorted(set(flat.round(6))) == [0.1, 0.2, 0.3]
    plt.close("all")


def test_plot_group_skips_rounds_without_a_value_for_the_metric(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    plot = load_script("plot")
    plt.close("all")
    paths = [write_run(tmp_path, "a0", [0.1, 0.5, 0.9], seed=0, columns={"dp-epsilon": [None, 1.0, 3.0]}),
             write_run(tmp_path, "a1", [0.1, 0.5, 0.9], seed=1, columns={"dp-epsilon": [None, 3.0, None]})]
    figure = plot.main([*paths, "--group", "-m", "dp-epsilon", "-o", str(tmp_path / "e.png")])
    line = figure.axes[0].get_lines()[0]
    assert list(line.get_xdata()) == [1, 2] and list(line.get_ydata()) == pytest.approx([2.0, 3.0])
    plt.close("all")


def test_scripts_handle_null_epsilon_from_unbounded_runs(tmp_path):
    compare, plot = load_script("compare"), load_script("plot")
    path = write_run(tmp_path, "inf", [0.1, 0.5], dp="fixed", **{"dp-side": "server", "dp-noise-multiplier": 0.0},
                     privacy={"epsilon": None, "epsilon_unbounded": True})
    data = json.loads(Path(path).read_text())
    data["history"][1]["dp-epsilon"] = None
    Path(path).write_text(json.dumps(data))
    assert compare.summarise([path], 0.9)[0]["epsilon"] == "inf"
    pytest.importorskip("matplotlib")
    import sys

    old, sys.argv = sys.argv, ["plot.py", path, "-m", "dp-epsilon", "-o", str(tmp_path / "e.png")]
    try:
        plot.main()  # all values null -> an empty curve, not a crash
    finally:
        sys.argv = old


def test_partition_stats_prints_a_table_and_the_skew(capsys):
    stats = load_script("partition_stats")
    stats.main(["--dataset", "synthetic", "--partitioner", "dirichlet", "--alpha", "0.1", "--num-partitions", "5"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "dataset=synthetic partitioner=dirichlet alpha=0.1 clients=5 seed=42"
    rows = [line.split() for line in out[2:7]]
    assert [r[0] for r in rows] == ["0", "1", "2", "3", "4"] and sum(int(r[1]) for r in rows) == 3000
    assert out[-1].startswith("mean max-class share:") and float(out[-1].split(":")[1].split()[0]) > 0.4  # alpha 0.1 is skewed
    stats.main(["--partitioner", "iid", "--num-partitions", "5"])
    assert "partitioner=iid clients=5" in capsys.readouterr().out  # no alpha shown for IID


def test_partition_stats_can_save_a_heat_map(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    stats = load_script("partition_stats")
    png = tmp_path / "p.png"
    stats.main(["--num-partitions", "4", "--plot", str(png)])
    assert png.exists() and png.stat().st_size > 1000


def test_run_grid_parses_run_config_fragments():
    grid = load_script("run_grid")
    parsed = grid.parse_assignments("strategy='fedavg' lr=0.5 rounds=3 on=true off=false path='a b'")
    assert parsed == {"strategy": "fedavg", "lr": 0.5, "rounds": 3, "on": True, "off": False, "path": "a b"}
    assert grid.parse_assignments("") == {}
    assert grid.format_run_config({"strategy": "fedavg", "lr": 1e-05, "on": True, "n": 3}) == "strategy='fedavg' lr=1e-05 on=true n=3"
    with pytest.raises(ValueError, match="key=value"):
        grid.parse_assignments("strategy")
    with pytest.raises(ValueError, match="key=value"):
        grid.parse_assignments("=3")
    # formatting and parsing are inverse, also for strings with backslashes and quotes
    tricky = {"a": "back\\slash", "b": "it's", "c": "tab\there", "d": "plain", "e": -2, "f": False}
    assert grid.parse_assignments(grid.format_run_config(tricky)) == tricky


@pytest.fixture
def fake_flwr(monkeypatch):
    """Replace `flwr run` by a function that writes the history.json a successful ServerApp would write."""
    import tomllib

    class Fake:
        def __init__(self):
            self.calls, self.stops = [], []

        def install(self, grid, writes=True, error=None):
            def run(command, stdout=None, stderr=None, env=None, timeout=None, check=False, capture_output=False):
                if command[1] == "stop":
                    self.stops.append(command[2])
                    return None
                run_config = command[command.index("--run-config") + 1]
                cfg = tomllib.loads(run_config.replace(" ", "\n"))  # independent of run_grid's own parser
                self.calls.append({"command": command, "env": env, "timeout": timeout, "cfg": cfg})
                stdout.write("Successfully started run 777\nfake flwr output\n")
                stdout.flush()
                if writes:
                    full = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "attack": "none",
                            "num-malicious": 0, "dp": "none", **cfg}  # what the server records
                    out = Path(cfg["output-dir"]) / f"run{len(self.calls)}"
                    out.mkdir(parents=True)
                    (out / "history.json").write_text(json.dumps({"config": full, "history": [{"round": 1, "test_acc": 0.5}]}))
                if error:
                    raise error
                return None

            monkeypatch.setattr(grid.subprocess, "run", run)
            return self

    return Fake().install


GRID_ARGS = ["--seeds", "1", "2", "--num-supernodes", "4", "--base", "dataset='synthetic' num-server-rounds=3",
             "--vary", "strategy='fedavg'", "strategy='scaffold' qfedavg-q=0.5"]


def test_run_grid_runs_every_variant_and_seed_and_resumes(tmp_path, fake_flwr, capsys):
    grid = load_script("run_grid")
    fake = fake_flwr(grid)
    calls = fake.calls
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 0
    assert [(c["cfg"]["seed"], c["cfg"]["strategy"]) for c in calls] == [(1, "fedavg"), (1, "scaffold"), (2, "fedavg"), (2, "scaffold")]
    second = calls[1]
    assert second["command"][:3] == ["flwr", "run", grid.PROJECT_DIR] and "--stream" in second["command"]
    assert second["command"][second["command"].index("--federation-config") + 1] == "num-supernodes=4"
    assert second["cfg"] == {"dataset": "synthetic", "num-server-rounds": 3, "strategy": "scaffold", "qfedavg-q": 0.5,
                             "seed": 1, "output-dir": str(tmp_path)}
    assert second["env"]["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] == "1" and second["timeout"] == 1800
    logs = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert len(logs) == 4 and all(name.startswith("seed") and name.endswith(".log") for name in logs)
    out = capsys.readouterr().out
    assert "4 run, 0 skipped, 0 failed" in out and "fedavg synthetic" in out  # summarised with compare.py

    # the same grid again does nothing; one more seed only runs the new seed
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 0
    assert len(calls) == 4 and "0 run, 4 skipped" in capsys.readouterr().out
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS[:1], "1", "2", "3", *GRID_ARGS[3:]]) == 0
    assert [c["cfg"]["seed"] for c in calls[4:]] == [3, 3]
    out = capsys.readouterr().out
    assert "2 run, 4 skipped" in out and out.count("-> skip (done)") == 4
    # the summary covers this invocation's runs: seeds 1 and 2 (skipped) and 3
    assert [line.split() for line in out.splitlines() if line.startswith("fedavg synthetic iid")][0][3] == "3"


def test_run_grid_no_resume_repeats_everything_and_summarises_only_the_new_runs(tmp_path, fake_flwr, capsys):
    grid = load_script("run_grid")
    calls = fake_flwr(grid).calls
    grid.main(["--output-dir", str(tmp_path), *GRID_ARGS])
    capsys.readouterr()
    assert grid.main(["--output-dir", str(tmp_path), "--no-resume", *GRID_ARGS]) == 0
    out = capsys.readouterr().out
    assert len(calls) == 8 and "4 run, 0 skipped" in out
    assert len(list(tmp_path.glob("*/history.json"))) == 8  # the old results stay on disk ...
    assert [line.split() for line in out.splitlines() if line.startswith("fedavg synthetic iid")][0][3] == "2"  # ... but are not mixed in
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 0  # and the manifest now points at the new ones
    assert len(calls) == 8


def test_run_grid_runs_again_when_anything_that_matters_changes(tmp_path, fake_flwr, monkeypatch):
    grid = load_script("run_grid")
    calls = fake_flwr(grid).calls
    args = ["--output-dir", str(tmp_path), "--seeds", "1"]
    grid.main(args)
    grid.main(args)
    assert len(calls) == 1
    grid.main([*args, "--num-supernodes", "20"])  # not part of the run config, but of the federation
    grid.main([*args, "--base", "learning-rate=0.5"])
    assert len(calls) == 3
    defaults = grid.default_config()
    monkeypatch.setattr(grid, "default_config", lambda: {**defaults, "local-epochs": defaults["local-epochs"] + 1})
    grid.main(args)  # a pyproject default that was never named in the grid
    assert len(calls) == 4


def test_run_grid_install_deps_leaves_the_environment_alone(tmp_path, fake_flwr, monkeypatch):
    grid = load_script("run_grid")
    monkeypatch.delenv("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", raising=False)
    calls = fake_flwr(grid).calls
    grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--install-deps", "--timeout", "7"])
    assert "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION" not in calls[0]["env"] and calls[0]["timeout"] == 7


def test_run_grid_reports_runs_that_wrote_no_results(tmp_path, fake_flwr, capsys):
    # `flwr run` exits with 0 even when the ServerApp fails, so only a new history.json counts as success
    grid = load_script("run_grid")
    calls = fake_flwr(grid, writes=False).calls
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 1
    out = capsys.readouterr().out
    assert len(calls) == 4 and "0 run, 0 skipped, 4 failed (runs [1, 2, 3, 4])" in out and "FAILED" in out
    assert "final acc" not in out  # nothing to summarise
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 1 and len(calls) == 8  # failures are retried, not skipped


def test_run_grid_stops_a_run_that_times_out_and_goes_on(tmp_path, fake_flwr, capsys):
    # the CLI only streams logs; killing it would leave the run going in the background SuperLink
    grid = load_script("run_grid")
    fake = fake_flwr(grid, error=grid.subprocess.TimeoutExpired("flwr", 1), writes=False)
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "2", "--timeout", "1"]) == 1
    out = capsys.readouterr().out
    assert "timed out after 1s" in out and "stopped run 777" in out and "(runs [1, 2])" in out
    assert fake.stops == ["777", "777"] and len(fake.calls) == 2


def test_run_grid_stops_the_run_on_ctrl_c_and_keeps_finished_results(tmp_path, fake_flwr):
    grid = load_script("run_grid")
    fake = fake_flwr(grid)
    grid.main(["--output-dir", str(tmp_path), "--seeds", "1"])
    fake_flwr(grid, error=KeyboardInterrupt(), writes=False)
    with pytest.raises(KeyboardInterrupt):
        grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "2"])
    assert fake.stops == ["777"] and [c["cfg"]["seed"] for c in fake.calls] == [1, 2]  # seed 1 was skipped the second time
    assert len(list(tmp_path.glob("*/history.json"))) == 1 and len(grid.load_manifest(str(tmp_path))) == 1


def test_run_grid_records_the_newest_results_when_several_appear(tmp_path, monkeypatch):
    # e.g. another process writes into the same directory while the job runs
    import os

    grid = load_script("run_grid")
    cfg = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "attack": "none", "num-malicious": 0, "dp": "none"}

    def run(command, **kwargs):
        for name, mtime in (("mine", 20), ("other", 10), ("later", 30)):
            (tmp_path / name).mkdir()
            (tmp_path / name / "history.json").write_text(json.dumps({"config": cfg, "history": [{"round": 1, "test_acc": 0.5}]}))
            os.utime(tmp_path / name / "history.json", ns=(mtime * 10**9, mtime * 10**9))

    monkeypatch.setattr(grid.subprocess, "run", run)
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1"]) == 0
    assert [entry["history"] for entry in grid.load_manifest(str(tmp_path)).values()] == ["later/history.json"]


def test_run_grid_counts_results_written_by_a_run_that_was_stopped_at_the_timeout(tmp_path, fake_flwr, capsys):
    grid = load_script("run_grid")
    fake = fake_flwr(grid, error=grid.subprocess.TimeoutExpired("flwr", 1), writes=True)
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--timeout", "1"]) == 0
    assert fake.stops == ["777"] and "1 run, 0 skipped, 0 failed" in capsys.readouterr().out


def test_run_grid_does_not_crash_when_the_log_has_no_run_id(tmp_path, monkeypatch, capsys):
    grid = load_script("run_grid")

    def hang(command, **kwargs):
        if command[1] == "stop":
            raise AssertionError("no run id, nothing to stop")
        kwargs["stdout"].write("the CLI died before the run started\n")
        raise grid.subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(grid.subprocess, "run", hang)
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--timeout", "1"]) == 1
    assert "timed out" in capsys.readouterr().out


def test_run_grid_reports_a_missing_flwr_executable(tmp_path, monkeypatch, capsys):
    grid = load_script("run_grid")

    def missing(command, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'flwr'")

    monkeypatch.setattr(grid.subprocess, "run", missing)
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1"]) == 2
    assert "cannot start `flwr`" in capsys.readouterr().err


def test_run_grid_ignores_unreadable_manifests_and_results(tmp_path, fake_flwr):
    grid = load_script("run_grid")
    calls = fake_flwr(grid).calls
    (tmp_path / "grid.json").write_text("{ half written")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "history.json").write_text("{ half written")
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1"]) == 0 and len(calls) == 1
    # the manifest says the run is done, but its results were deleted: run it again
    next(tmp_path.glob("run1/history.json")).unlink()
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1"]) == 0 and len(calls) == 2


def test_run_grid_dry_run_only_prints_the_commands(tmp_path, fake_flwr, capsys):
    grid = load_script("run_grid")
    calls = fake_flwr(grid).calls
    out_dir = tmp_path / "new"
    assert grid.main(["--output-dir", str(out_dir), "--dry-run", *GRID_ARGS]) == 0
    out = capsys.readouterr().out
    assert calls == [] and out.count("flwr run") == 4 and "seed=2" in out
    assert "0 run, 0 skipped, 0 failed" in out and not out_dir.exists()  # a dry run leaves no trace


def test_run_grid_refuses_settings_it_controls_itself(tmp_path, fake_flwr):
    grid = load_script("run_grid")
    fake_flwr(grid)
    for args in (["--base", "seed=3"], ["--vary", "strategy='fedavg'", "seed=3 strategy='fedavg'"], ["--base", "output-dir='x'"]):
        with pytest.raises(SystemExit):
            grid.main(["--output-dir", str(tmp_path), *args])


def test_run_grid_finds_results_in_directories_with_glob_characters(tmp_path, fake_flwr):
    grid = load_script("run_grid")
    calls = fake_flwr(grid).calls
    out_dir = tmp_path / "grid[alpha=0.1]"
    assert grid.main(["--output-dir", str(out_dir), "--seeds", "1"]) == 0 and len(calls) == 1
    assert grid.main(["--output-dir", str(out_dir), "--seeds", "1"]) == 0 and len(calls) == 1


def test_run_grid_counts_a_rewritten_history_as_success(tmp_path, monkeypatch, capsys):
    # two runs that get the same results directory name: the second overwrites the first one's history.json
    import os

    grid = load_script("run_grid")
    path = tmp_path / "same_name" / "history.json"
    path.parent.mkdir()
    cfg = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "attack": "none", "num-malicious": 0, "dp": "none"}

    def run(command, stdout, **kwargs):
        path.write_text(json.dumps({"config": cfg, "history": [{"round": 1, "test_acc": 0.5}]}))
        stamp = (1 + len(calls)) * 10**9  # NTFS timestamps have 100 ns precision.
        os.utime(path, ns=(stamp, stamp))
        calls.append(command)

    calls = []
    monkeypatch.setattr(grid.subprocess, "run", run)
    path.write_text("{}")
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--no-resume"]) == 0
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--no-resume"]) == 0
    assert len(calls) == 2 and "1 run, 0 skipped, 0 failed" in capsys.readouterr().out
