"""Access the same explicit resources in editable installs and installed wheels."""
from functools import cache, lru_cache
import importlib.util
import json
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def catalog():
    return json.loads(Path(__file__).with_name("catalog.json").read_text(encoding="utf-8"))


def source_path(relative):
    if relative not in catalog()["resources"]:
        raise ValueError(f"Unknown package resource: {relative}")
    package = Path(__file__).resolve().parents[1]
    root = package / "_bundle"
    if relative.startswith("defai/") or not root.is_dir():
        root = package.parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Package resource missing: {relative}; reinstall defai")
    return path


@cache
def load_api(name):
    """Load isolated numerical modules under unique names; never change sys.path."""
    path = source_path(catalog()["api_modules"][name])
    module_name = "defai.research._impl_" + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        sys.modules.pop(module_name, None)
        raise ModuleNotFoundError(f"{name} needs optional dependencies. Install defai[research]: {exc}") from exc
    return module
