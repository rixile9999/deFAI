"""Downloads are explicit, checksum-checked, confined and non-overwriting."""
import hashlib
import io

import pytest

from defai import data


class Response(io.BytesIO):
    url = "https://example.test/public.xpt"


def entry(payload=b"public source", path="data/public/test.xpt"):
    return {"fixture": [{"dataset": "fixture", "source_url": "https://example.test/public.xpt",
                         "relative_path": path, "sha256": hashlib.sha256(payload).hexdigest(),
                         "size_bytes": str(len(payload))}]}


def test_fetch_verifies_and_reuses_exact_source(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "datasets", entry)
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        return Response(b"public source")

    monkeypatch.setattr(data, "urlopen", fetch)
    assert data.fetch_dataset("fixture", tmp_path)[0]["cached"] is False
    assert data.fetch_dataset("fixture", tmp_path)[0]["cached"] is True
    assert len(calls) == 1


def test_mismatch_removes_partial_file_and_retains_existing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "datasets", entry)
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: Response(b"wrong"))
    with pytest.raises(ValueError, match="recorded source"):
        data.fetch_dataset("fixture", tmp_path)
    target = tmp_path / "data/public/test.xpt"
    assert not target.exists() and not list(target.parent.glob(".defai-*"))
    target.write_bytes(b"existing local file")
    with pytest.raises(ValueError, match="left intact"):
        data.fetch_dataset("fixture", tmp_path)
    assert target.read_bytes() == b"existing local file"


def test_destination_traversal_is_rejected_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "datasets", lambda: entry(path="../escape.xpt"))
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: pytest.fail("No download should happen"))
    with pytest.raises(ValueError, match="outside"):
        data.fetch_dataset("fixture", tmp_path)


def test_archive_checksum_is_required_before_extracting(tmp_path):
    archive = tmp_path / "not_official.zip"
    archive.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="official"):
        data.prepare_neojaundice(archive, tmp_path / "data")
    assert not (tmp_path / "data").exists()
