"""Run a grid of experiments (variants x seeds) with `flwr run`, resumably, and summarise them.

usage: python scripts/run_grid.py --output-dir results/noniid --seeds 1 2 3 --num-supernodes 10 \\
           --base "dataset='fashion-mnist-mirror' partitioner='dirichlet' dirichlet-alpha=0.1 num-server-rounds=20" \\
           --vary "strategy='fedavg'" "strategy='scaffold'" "strategy='fednova'"

Every ``--vary`` entry is a run-config fragment that is appended to ``--base``; ``seed`` and ``output-dir``
are added per run. The grid keeps a manifest, ``<output-dir>/grid.json``, that maps each finished run to its
results. A run is identified by its complete run config (the pyproject defaults plus the fragments and the seed)
and ``--num-supernodes``, so an interrupted grid continues where it stopped, while changing any of those
(or ``--no-resume``) runs the job again. The final summary, made with scripts/compare.py, covers the runs of this
invocation only.

``flwr run`` exits with 0 even when the ServerApp fails (for example on an invalid config), so a run counts as
successful only if it wrote a new history.json. ``flwr run --stream`` hands the run to a background SuperLink: a run
that exceeds ``--timeout`` (or a Ctrl-C) is stopped with ``flwr stop``. Logs go to ``<output-dir>/logs/``.

Environment variables (FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1, which makes flwr reuse the current
environment instead of building a new one for every run; DEFAI_DATA_DIR) are read by that SuperLink, which the
first `flwr run` starts and later ones reuse. Stop it (``pkill -f flower-superlink``) to change them.
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = re.compile(r"""(?:[^\s'"]|'[^']*'|"[^"]*")+""")  # whitespace-separated, quotes may contain spaces
RUN_ID = re.compile(r"Successfully started run (\d+)")
MANIFEST = "grid.json"


def parse_assignments(fragment):
    """"a=1 b='x' c=true" -> {"a": 1, "b": "x", "c": True} (values use TOML syntax, like --run-config)."""
    out = {}
    for token in TOKEN.findall(fragment):
        key, sep, value = token.partition("=")
        if not sep or not key:
            raise ValueError(f"expected key=value, got {token!r}")
        out[key] = tomllib.loads(f"v = {value}")["v"]
    return out


def format_value(value):
    if isinstance(value, str) and "'" not in value and "\n" not in value:
        return f"'{value}'"  # TOML literal string: no escapes, so backslashes survive
    return json.dumps(value)  # numbers, true/false, and strings that need escaping (a valid TOML basic string)


def format_run_config(assignments):
    """Inverse of parse_assignments."""
    return " ".join(f"{k}={format_value(v)}" for k, v in assignments.items())


def default_config(project_dir=PROJECT_DIR):
    with open(os.path.join(project_dir, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["tool"]["flwr"]["app"]["config"]


def job_key(assignments, num_supernodes, defaults):
    """Identity of a job: everything that decides what the run does, as canonical JSON."""
    complete = {**defaults, **{k: v for k, v in assignments.items() if k != "output-dir"}}
    return json.dumps({"config": complete, "num-supernodes": num_supernodes}, sort_keys=True)


def histories(output_dir):
    """{path: modification time} of every history.json below output_dir."""
    paths = glob.glob(os.path.join(glob.escape(output_dir), "*", "history.json"))
    return {path: os.stat(path).st_mtime_ns for path in paths}


def read_config(path):
    """The config a finished run was made with, or None if the file is unreadable or half written."""
    try:
        with open(path) as f:
            return json.load(f)["config"]
    except (OSError, ValueError, KeyError):
        return None


def load_manifest(output_dir):
    try:
        with open(os.path.join(output_dir, MANIFEST)) as f:
            return json.load(f)["jobs"]
    except (OSError, ValueError, KeyError):
        return {}


def save_manifest(output_dir, jobs):
    path = os.path.join(output_dir, MANIFEST)
    with open(path + ".tmp", "w") as f:
        json.dump({"jobs": jobs}, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def finished_history(output_dir, jobs, key):
    """Path of the finished run for ``key``, or None."""
    entry = jobs.get(key)
    if entry:
        path = os.path.join(output_dir, entry["history"])
        if read_config(path) is not None:
            return path
    return None


def build_command(num_supernodes, assignments, project_dir=PROJECT_DIR):
    return ["flwr", "run", project_dir, "--stream", "--federation-config", f"num-supernodes={num_supernodes}",
            "--run-config", format_run_config(assignments)]


def stop_run(log_path, env):
    """Ask the SuperLink to stop the run whose id the log mentions (killing the CLI does not stop it)."""
    try:
        with open(log_path, errors="replace") as f:
            found = RUN_ID.search(f.read())
        if found:
            subprocess.run(["flwr", "stop", found.group(1)], env=env, capture_output=True, timeout=60, check=False)
            print(f"    stopped run {found.group(1)}")
    except (OSError, subprocess.SubprocessError):
        print(f"    could not stop the run; check `flwr ls` (see {log_path})")


def run_job(command, log_path, env, timeout):
    """Run one `flwr run`, and stop it if it takes too long or is interrupted."""
    try:
        with open(log_path, "w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"    timed out after {timeout:g}s (see {log_path})")
        stop_run(log_path, env)
    except KeyboardInterrupt:
        stop_run(log_path, env)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base", default="", help="run-config fragment shared by every run")
    p.add_argument("--vary", nargs="+", default=[""], help="run-config fragments, one run per fragment and seed")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--num-supernodes", type=int, default=10)
    p.add_argument("--timeout", type=float, default=1800, help="seconds allowed per run")
    p.add_argument("--no-resume", action="store_true", help="run everything again even if results exist")
    p.add_argument("--install-deps", action="store_true", help="let flwr build a fresh environment for each run")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    p.add_argument("--target", type=float, default=0.9, help="accuracy for the rounds-to-target column")
    args = p.parse_args(argv)

    base = parse_assignments(args.base)
    variants = [parse_assignments(v) for v in args.vary]
    for fragment in (base, *variants):
        reserved = sorted({"seed", "output-dir"} & set(fragment))
        if reserved:
            p.error(f"{', '.join(reserved)} is set by run_grid itself (use --seeds / --output-dir)")
    output_dir = os.path.abspath(args.output_dir)
    env = dict(os.environ)
    if not args.install_deps:
        env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    defaults = default_config()
    manifest = load_manifest(output_dir)

    jobs = [(seed, vary) for seed in args.seeds for vary in variants]  # seed-major: partial grids stay comparable
    failed, skipped, done, results = [], 0, 0, []
    for number, (seed, vary) in enumerate(jobs, 1):
        assignments = {**base, **vary, "seed": seed}
        key = job_key(assignments, args.num_supernodes, defaults)
        label = f"[{number}/{len(jobs)}] {format_run_config(assignments)}"
        existing = None if args.no_resume else finished_history(output_dir, manifest, key)
        if existing:
            skipped += 1
            results.append(existing)
            print(f"{label} -> skip (done)")
            continue
        command = build_command(args.num_supernodes, {**assignments, "output-dir": output_dir})
        print(label)
        if args.dry_run:
            print("   ", " ".join(shlex.quote(c) for c in command))
            continue
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        log_path = os.path.join(output_dir, "logs", f"seed{seed}_{hashlib.sha1(key.encode()).hexdigest()[:8]}.log")
        before = histories(output_dir)
        try:
            run_job(command, log_path, env, args.timeout)
        except FileNotFoundError as error:
            print(f"cannot start `{command[0]}`: {error}", file=sys.stderr)
            return 2
        after = histories(output_dir)
        changed = [path for path, mtime in after.items() if before.get(path) != mtime]  # new, or rewritten
        if changed:
            path = max(changed, key=after.get)
            manifest[key] = {"history": Path(os.path.relpath(path, output_dir)).as_posix()}
            save_manifest(output_dir, manifest)
            results.append(path)
            done += 1
        else:
            failed.append(number)
            print(f"    FAILED: no results were written (see {log_path})")

    print(f"\n{done} run, {skipped} skipped, {len(failed)} failed" + (f" (runs {failed})" if failed else ""))
    if not args.dry_run and results:
        compare.main([*sorted(set(results)), "--target", str(args.target)])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
