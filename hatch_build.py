"""Ship only the explicit research/resource inventory in an installed wheel."""
import json
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name not in {"wheel", "sdist"}:
            return
        root = Path(self.root)
        inventory = json.loads((root / "defai/research/catalog.json").read_text(encoding="utf-8"))
        for relative in inventory["resources"]:
            source = root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Required package resource missing: {relative}")
            if self.target_name == "wheel" and relative.startswith("defai/"):
                continue  # Canonical modules are already selected by packages=["defai"].
            destination = "defai/_bundle/" + relative if self.target_name == "wheel" else relative
            build_data["force_include"][str(source)] = destination
