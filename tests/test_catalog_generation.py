"""Resource inventory generation must have the same bytes on Windows and Linux."""
import hashlib
import importlib.util
import json
from pathlib import Path


def test_catalog_uses_case_sensitive_posix_order_and_lf(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    spec = importlib.util.spec_from_file_location("build_catalog", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    paths = [
        "01_neonatal/fl_pipeline/federated_burr.py",
        "01_neonatal/fl_pipeline/paillier.py",
        "01_neonatal/core/policy_stats.py",
        "01_neonatal/fl_comparison/learning.py",
        "01_neonatal/fl_comparison/cohort.py",
        "01_neonatal/GAN_research/prepare_data.py",
        "01_neonatal/NeoJaundice/src/model.py",
        "01_neonatal/NeoJaundice/src/federated.py",
        "01_neonatal/GAN_research/gan_models.py",
        "01_neonatal/GAN_research/federated_algorithms.py",
    ]
    for relative in paths:
        path = tmp_path / "disease_research" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'"""Fixture."""\n')
    (tmp_path / "defai/research").mkdir(parents=True)
    (tmp_path / "LICENSE").write_bytes(b"License fixture\n")
    (tmp_path / "README.md").write_bytes(b"Readme fixture\n")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    builder.build()
    output = tmp_path / "defai/research/catalog.json"
    first = output.read_bytes()
    catalog = json.loads(first)
    assert b"\r" not in first
    assert catalog["resources"] == sorted(catalog["resources"])
    assert catalog["resources"][:2] == ["LICENSE", "README.md"]
    assert [m["path"] for m in catalog["modules"]] == sorted(m["path"] for m in catalog["modules"])
    assert catalog["resource_hashes"]["LICENSE"] == hashlib.sha256(b"License fixture\n").hexdigest()
    builder.build()
    assert output.read_bytes() == first
