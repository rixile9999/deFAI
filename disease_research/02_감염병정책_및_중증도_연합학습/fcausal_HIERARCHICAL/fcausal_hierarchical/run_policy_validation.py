"""Reproduce pooled/three-tier/CKKS agreement and inference simulations.

Runs all 9- and 21-district workbooks without changing them. --network also
starts real Flower servers/clients on localhost (no visible terminal windows).
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import flwr as fl
import numpy as np
import pandas as pd
import tenseal as ts

from core.policy_stats import (AnalysisConfig, analyze, daily_counts, district_identity,
                               district_vector, province_summary, district_treatment, read_district_file)
from core.policy_transport import encode, metadata
from run_tier0 import CentralStrategy
from run_tier1 import ProvinceStrategy, ProvinceClient

ROOT = Path(__file__).resolve().parent


def make_contexts():
    secret = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    secret.global_scale = 2**40
    public = ts.context_from(secret.serialize(save_secret_key=False))
    return secret, public


def fit_response(arrays, info):
    return None, SimpleNamespace(parameters=fl.common.ndarrays_to_parameters(arrays), metrics=info)


def load_records(data_dir, config):
    records = []
    for path in sorted([*data_dir.glob("*.xlsx"), *data_dir.glob("*.csv")]):
        frame = read_district_file(path)
        province, district = district_identity(frame)
        records.append({"path": path, "province": province, "district": district,
                        "treated": district_treatment(frame, province), "counts": daily_counts(frame, config)})
    if not records:
        raise ValueError(f"No district data files: {data_dir}")
    return records


def reference_analysis(records, config):
    """Independent pooled calculation: use district arrays, not moment decoding."""
    pre = config.calendar < pd.Timestamp(config.policy)
    groups = {arm: np.stack([np.log1p(r["counts"]) for r in records if r["treated"] == arm])
              for arm in (0, 1)}
    effect = ((groups[1][:, ~pre].mean()-groups[1][:, pre].mean()) -
              (groups[0][:, ~pre].mean()-groups[0][:, pre].mean()))
    return {"effect": float(effect), "cell_moments": {
        f"{arm}_{period}": {"mean": float(y[:, mask].mean()),
                            "variance": float(y[:, mask].var(ddof=1)), "n": int(y[:, mask].size)}
        for arm, y in groups.items() for period, mask in [("pre", pre), ("post", ~pre)]}}


def run_strategies(records, config, contexts, encrypted):
    secret, public = contexts
    ctx, central_ctx = (public, secret) if encrypted else (None, None)
    key_id = "validation_shared_key" if encrypted else "plaintext"
    provinces = sorted({r["province"] for r in records})
    relays = []
    for region in provinces:
        selected = [r for r in records if r["province"] == region]
        treated = selected[0]["treated"]
        strategy = ProvinceStrategy(ctx, region, config, treated, key_id,
                                    min_fit_clients=len(selected), min_available_clients=len(selected))
        responses = []
        for r in selected:
            info = {**metadata(config, encrypted, key_id), "province": region,
                    "district": r["district"], "treated": r["treated"]}
            responses.append(fit_response(encode(district_vector(r["counts"], config), ctx), info))
        strategy.aggregate_fit(1, responses, [])
        relay = ProvinceClient(strategy.aggregated_stats, region, len(selected), treated,
                               config, encrypted, key_id)
        arrays, _, info = relay.fit([], {})
        relays.append(fit_response(arrays, info))
    central = CentralStrategy(central_ctx, config, key_id,
                              min_fit_clients=len(provinces), min_available_clients=len(provinces))
    central.aggregate_fit(1, relays, [])
    return central.result


def legacy_effect(data_dir):
    """Reproduce the old mean/std summation for comparison, without claiming validity."""
    provinces = {}
    for path in sorted(data_dir.glob("*.xlsx")):
        frame = pd.read_excel(path)
        region, _ = district_identity(frame)
        daily = frame.groupby(pd.to_datetime(frame["진단일"])).size()
        pre = daily[daily.index < "2024-01-31"].to_numpy()
        post = daily[daily.index >= "2024-01-31"].to_numpy()
        vector = np.array([len(pre), np.log(pre.mean()+1e-5), np.log(pre).std(),
                           len(post), np.log(post.mean()+1e-5), np.log(post).std()])
        provinces[region] = provinces.get(region, np.zeros(6)) + vector
    treated = provinces["서울"]
    controls = [v for name, v in provinces.items() if name != "서울"]
    weights = np.array([v[0]+v[3] for v in controls]); weights = weights/weights.sum()
    return float(treated[4]-treated[1] - sum((v[4]-v[1])*w for v, w in zip(controls, weights)))


def validate_datasets(contexts, output):
    comparisons, file_hashes = [], {}
    for name, column in itertools.product(["districts", "districts21"], ["발병일", "진단일"]):
        config = AnalysisConfig(date_column=column)
        folder = ROOT / "data" / name
        records = load_records(folder, config)
        reference = reference_analysis(records, config)
        plain = run_strategies(records, config, contexts, False)
        encrypted = run_strategies(records, config, contexts, True)
        # Compare pooled means/variances in every group-period cell as well as effect.
        max_moment_error = 0.0
        for arm in (0, 1):
            vector = sum(district_vector(r["counts"], config) for r in records if r["treated"] == arm)
            summary = province_summary(vector, "pooled", arm, config)
            for period in ("pre", "post"):
                ref = reference["cell_moments"][f"{arm}_{period}"]
                max_moment_error = max(max_moment_error, abs(ref["mean"]-summary[period]["mean"]),
                                       abs(ref["variance"]-summary[period]["variance"]))
        errors = {"plaintext_effect_abs_error": abs(plain["effect"]-reference["effect"]),
                  "ckks_effect_abs_error": abs(encrypted["effect"]-reference["effect"]),
                  "pooled_moments_max_abs_error": max_moment_error}
        assert errors["plaintext_effect_abs_error"] < 1e-12
        assert errors["ckks_effect_abs_error"] < 1e-6
        assert max_moment_error < 1e-10
        for r in records:
            file_hashes[str(r["path"].relative_to(ROOT))] = sha256(r["path"].read_bytes()).hexdigest()
        row = {"dataset": name, "n_districts": len(records), "config": asdict(config),
               "reference": reference, "plaintext": plain, "ckks": encrypted,
               "legacy_invalid_effect": legacy_effect(folder), **errors}
        comparisons.append(row)
        if name == "districts" and column == "발병일":
            latest = {**encrypted, "config": asdict(config), "encrypted": True}
            (output / "latest_policy_analysis.json").write_text(
                json.dumps(latest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return comparisons, file_hashes


def simulate(repetitions):
    """Known additive effect on log1p outcome, province shocks and serial dependence.

    This is a statistical DGP, not a patient/epidemic model. Random province
    assignment is simulated; existing workbooks are NOT randomized evidence.
    """
    cells = []
    for effect, rho, size_mode in itertools.product([0.0, -0.4], [0.0, 0.8], ["equal", "unequal"]):
        rng = np.random.default_rng(20261016)
        estimates, covered, rejected = [], [], []
        for _ in range(repetitions):
            rows = []
            treatment = rng.permutation(np.repeat([0, 1], 15))
            for p, arm in enumerate(treatment):
                size = 3 if size_mode == "equal" else int(rng.integers(1, 7))
                innovations = rng.normal(0, 0.12*np.sqrt(1-rho*rho), 100)
                shock = np.empty(100); shock[0] = rng.normal(0, 0.12)
                for day in range(1, 100):
                    shock[day] = rho*shock[day-1]+innovations[day]
                # Common calendar shocks cancel because all districts share the same calendar.
                y = 3 + .003*np.arange(100) + shock + rng.normal(0, .06, (size, 100))
                y[:, 30:] += arm*effect
                rows.append({"region": str(p), "treated": int(arm), "n_districts": size,
                             "change": float(y[:, 30:].mean()-y[:, :30].mean()),
                             "pre_placebo_change": float(y[:, 15:30].mean()-y[:, :15].mean())})
            result = analyze(rows)
            low, high = result["inference"]["ci95"]
            estimates.append(result["effect"])
            covered.append(low <= effect <= high)
            rejected.append(result["inference"]["p_value"] < .05)
        estimates = np.asarray(estimates)
        cells.append({"true_effect_log1p": effect, "serial_correlation": rho,
                      "district_sizes": size_mode, "n_provinces": 30, "replications": repetitions,
                      "bias": float(estimates.mean()-effect), "rmse": float(np.sqrt(np.mean((estimates-effect)**2))),
                      "coverage95": float(np.mean(covered)), "rejection_rate05": float(np.mean(rejected)),
                      "rejection_mc_se": float(np.sqrt(np.mean(rejected)*(1-np.mean(rejected))/repetitions))})
    # Identification counterexamples: an untreated differential trend and reporting change.
    examples = []
    for scenario in ["parallel", "differential_pretrend", "reporting_change"]:
        rows = []
        for p in range(30):
            arm = int(p < 15)
            y = 3+.003*np.arange(100)
            if scenario == "differential_pretrend":
                y = y+arm*.004*np.arange(100)
            if scenario == "reporting_change":
                y[30:] -= arm*.3
            rows.append({"region": str(p), "treated": arm, "n_districts": 3,
                         "change": float(y[30:].mean()-y[:30].mean()),
                         "pre_placebo_change": float(y[15:30].mean()-y[:15].mean())})
        result = analyze(rows)
        examples.append({"scenario": scenario, "true_policy_effect": 0,
                         "estimated_contrast": result["effect"], "pre_placebo": result["pre_placebo_effect"]})
    return cells, examples


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_port(port, processes, timeout=60):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if any(p.poll() not in (None, 0) for p in processes):
            raise RuntimeError("A network node failed; inspect results/network logs")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                return
        except OSError:
            time.sleep(.1)
    raise TimeoutError(f"Port {port} did not become ready")


def validate_network(contexts, output, data_dir=None, config=None):
    config = config or AnalysisConfig()
    records = load_records(data_dir or ROOT / "data" / "districts", config)
    regions = sorted({r["province"] for r in records})
    reference = reference_analysis(records, config)["effect"]
    all_results = []
    with tempfile.TemporaryDirectory(prefix="policy_keys_") as temp:
        secret, public = contexts
        Path(temp, "secret.ctx").write_bytes(secret.serialize(save_secret_key=True))
        Path(temp, "public.ctx").write_bytes(public.serialize(save_secret_key=False))
        for encrypted in [False, True]:
            mode = "ckks" if encrypted else "plaintext"
            folder = output / "network" / mode
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / "analysis.json"
            if result_path.exists():
                result_path.unlink()  # only our own previous test output
            processes, logs = [], []
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
            def launch(name, args):
                handle = (folder / f"{name}.log").open("w", encoding="utf-8")
                logs.append(handle)
                process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                processes.append(process)
                return process
            common = ["--key-dir", temp] if encrypted else ["--plaintext"]
            common += ["--start", config.start, "--policy", config.policy, "--end", config.end,
                       "--date-column", config.date_column]
            ports = [free_port() for _ in range(len(regions)+1)]
            try:
                launch("central", ["run_tier0.py", "--address", f"127.0.0.1:{ports[0]}",
                                   "--min-provinces", str(len(regions)), "--output", str(result_path), *common])
                wait_port(ports[0], processes)
                for i, region in enumerate(regions):
                    selected = [r for r in records if r["province"] == region]
                    launch(f"province_{i}", ["run_tier1.py", "--port", str(ports[i+1]), "--region", region,
                                             "--target", f"127.0.0.1:{ports[0]}", "--min-districts", str(len(selected)),
                                             *(["--treated"] if selected[0]["treated"] else []), *common])
                    wait_port(ports[i+1], processes)
                    for j, r in enumerate(selected):
                        launch(f"district_{i}_{j}", ["run_tier2.py", "--data", str(r["path"]),
                                                     "--target", f"127.0.0.1:{ports[i+1]}",
                                                     "--treated", str(r["treated"]), *common])
                deadline = time.monotonic()+180
                for process in processes:
                    code = process.wait(timeout=max(1, deadline-time.monotonic()))
                    if code:
                        raise RuntimeError(f"Network node exited {code}; see {folder}")
                actual = json.loads(result_path.read_text(encoding="utf-8"))
                error = abs(actual["effect"]-reference)
                assert error < (1e-6 if encrypted else 1e-12)
                treated_regions = {r["province"] for r in records if r["treated"]}
                expected_status = "withheld" if min(len(treated_regions), len(regions)-len(treated_regions)) < 2 else "exploratory"
                assert actual["inference"]["status"] == expected_status
                all_results.append({"mode": mode, "processes": len(processes), "n_districts": len(records),
                                    "n_provinces": len(regions), "effect": actual["effect"],
                                    "absolute_error_vs_pooled": error, "all_exit_codes_zero": True,
                                    "n_treated_provinces": len(treated_regions),
                                    "inference_status": actual["inference"]["status"]})
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait(timeout=10)
                for handle in logs:
                    handle.close()
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("replications must be at least 2")
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    contexts = make_contexts()
    comparisons, hashes = validate_datasets(contexts, output)
    cells, examples = simulate(args.replications)
    report = {"stage": "research_only", "comparisons": comparisons, "simulations": cells,
              "identification_counterexamples": examples, "source_workbook_sha256": hashes,
              "network": validate_network(contexts, output) if args.network else [],
              "public_context_has_secret_key": contexts[1].has_secret_key()}
    for path, expected in hashes.items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == expected
    (output / "policy_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(cells).to_csv(output / "policy_simulation.csv", index=False, encoding="utf-8-sig")
    print(f"Validation complete: {output / 'policy_validation.json'}", flush=True)


if __name__ == "__main__":
    main()
