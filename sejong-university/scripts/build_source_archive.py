from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
TOP = "Federated_Learning"
DIRECTORIES = ("1_simulation", "2_analysis_speed", "3_heterogeneous_data", "common", "examples", "tests",
               "scripts", "docs")
ROOT_FILES = ("README.md", "LICENSE", "pyproject.toml", ".gitignore")
SUFFIXES = {".py", ".md", ".toml", ".json", ".ipynb", ".yml", ".yaml", ".html", ".js", ".css", ".csv", ".txt",
            ".code-workspace"}


def payloads():
    selected = [ROOT / name for name in ROOT_FILES]
    for directory in DIRECTORIES:
        path = ROOT / directory
        if path.exists():
            selected.extend(p for p in path.rglob("*")
                            if p.is_file() and p.suffix in SUFFIXES
                            and not any(part in {"__pycache__", ".ipynb_checkpoints"}
                                        for part in p.relative_to(ROOT).parts))
    files = {}
    for path in sorted(set(selected)):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Source path escapes release root: {path}")
        content = path.read_bytes()
        if path.suffix == ".ipynb":
            notebook = json.loads(content)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            content = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        files[path.relative_to(ROOT).as_posix()] = content
    return files


def build(output: Path):
    files = payloads()
    manifest = {
        "project": "sejong-defai", "version": VERSION, "license": "MIT",
        "scope": "Source, tests, configuration and documentation only; no datasets, trained weights, predictions or experiment outputs",
        "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in files.items()},
    }
    files["SOURCE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(f"{TOP}/{name}", content)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC failed: {bad}")
        for name, content in files.items():
            if archive.read(f"{TOP}/{name}") != content:
                raise ValueError(f"Archive content changed: {name}")
    receipt = {"archive": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "bytes": output.stat().st_size, "file_count": len(files),
               "crc_and_content_verified": True, "source_manifest": manifest}
    output.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "source_manifest"}, ensure_ascii=False))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / f"{TOP}.zip")
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
