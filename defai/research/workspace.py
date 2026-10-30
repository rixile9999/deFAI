"""Writable workspaces and isolated subprocesses prevent study import collisions."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from defai import __version__
from defai.research.resources import catalog, source_path


def list_features(domain=None):
    return [dict(m) for m in catalog()["modules"] if domain is None or m["domain"] == domain]


def create_workspace(destination):
    """Create a fresh workspace atomically; never merge into or overwrite user data."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".defai-init-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "workspace"
        stage.mkdir()
        for relative in catalog()["resources"]:
            source = source_path(relative)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != catalog()["resource_hashes"][relative]:
                raise ValueError(f"Resource integrity check failed: {relative}")
        target = stage / "defai/research/catalog.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(catalog(), ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / ".defai-workspace.json").write_text(json.dumps({"schema": 1, "version": __version__}), encoding="utf-8")
        stage.rename(destination)
    return destination


def run_feature(feature_id, workspace, arguments=(), *, device="cpu", timeout=None):
    """Run one registered script with its own imports and writable project directory."""
    features = {m["id"]: m for m in list_features()}
    if feature_id not in features:
        raise ValueError(f"Unknown feature: {feature_id}; use 'defai research list'")
    feature = features[feature_id]
    if feature["role"] != "script":
        raise ValueError(f"{feature_id} is a {feature['role']}, not a standalone script; see 'defai research show'")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    root = Path(workspace).resolve()
    marker = root / ".defai-workspace.json"
    if not marker.is_file():
        raise ValueError(f"Not an initialized workspace: {root}; run 'defai init NEW_DIRECTORY'")
    if json.loads(marker.read_text(encoding="utf-8"))["version"] != __version__:
        raise ValueError("Workspace/package version mismatch; initialize a fresh workspace")
    script = (root / feature["path"]).resolve()
    if not script.is_relative_to(root) or not script.is_file():
        raise ValueError("Research script missing or outside the workspace")
    args = list(arguments)
    if args[:1] == ["--"]:
        args.pop(0)
    if "--help" in args and not feature["arguments"]:
        raise ValueError("This inherited script has no argument parser; use 'defai research show' instead")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", DEFAI_DEVICE=device)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.perf_counter()
    receipt = {"feature": feature_id, "arguments": args, "device": device,
               "research_only": True, "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    try:
        result = subprocess.run([sys.executable, str(script), *args], cwd=script.parent,
                                env=env, timeout=timeout, check=False)
        receipt["returncode"] = result.returncode
        return result.returncode
    except subprocess.TimeoutExpired:
        receipt["returncode"] = 124
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - start
        log = root / "run_receipts"
        log.mkdir(exist_ok=True)
        (log / f"{time.time_ns()}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
