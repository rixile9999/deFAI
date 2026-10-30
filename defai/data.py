"""Explicit public-source downloads, checked against the acquisition inventory."""
import csv
import hashlib
import os
from pathlib import Path
import tempfile
import shutil
import zipfile
from urllib.parse import urlsplit
from urllib.request import urlopen

from defai.research.resources import catalog, source_path


def datasets():
    result = {"neojaundice": [{"dataset": "NeoJaundice v1 (CC0)", "relative_path": "NeoJaundice.zip",
                               "source_url": "https://ndownloader.figshare.com/files/39672982",
                               "size_bytes": "461702440", "md5": "7330ef5106eeff4099b072f4d98d96b6"}]}
    for path in catalog()["resources"]:
        if "/references/public_inventory_" in path and path.endswith(".csv"):
            with source_path(path).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            result[Path(path).stem] = rows
    return result


def fetch_dataset(name, destination, *, timeout=30):
    """Fetch only the chosen public inventory; existing mismatched files are retained."""
    registry = datasets()
    if name not in registry:
        raise ValueError("Unknown dataset; use 'defai data list'")
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for row in registry[name]:
        url = row["source_url"]
        if urlsplit(url).scheme != "https":
            raise ValueError(f"Inventory entry has no downloadable HTTPS source: {url}")
        relative = Path(row["relative_path"].replace("\\", "/"))
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError("Inventory contains a destination outside the selected directory")
        algorithm = "sha256" if row.get("sha256") else "md5"
        expected = row.get(algorithm)
        if not expected:
            raise ValueError("A recorded checksum is required before downloading")
        if target.exists():
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, algorithm).hexdigest()
            if digest != expected:
                raise ValueError(f"Existing data checksum mismatch; file left intact: {target}")
            results.append({"path": str(target), "cached": True})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name_on_disk = tempfile.mkstemp(prefix=".defai-download-", dir=target.parent)
        temporary = Path(name_on_disk)
        try:
            hasher = hashlib.new(algorithm)
            size = 0
            with os.fdopen(fd, "wb") as stream, urlopen(url, timeout=timeout) as response:
                if urlsplit(response.url).scheme != "https":
                    raise ValueError("Download redirected to an insecure source")
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    hasher.update(block)
                    size += len(block)
            if hasher.hexdigest() != expected or row.get("size_bytes") and size != int(row["size_bytes"]):
                raise ValueError(f"Downloaded bytes do not match the recorded source: {url}")
            os.link(temporary, target)
            results.append({"path": str(target), "bytes": size, "cached": False})
        finally:
            temporary.unlink(missing_ok=True)
    return results


def prepare_neojaundice(archive, destination):
    """Extract the verified official archive into the study's images/metadata layout."""
    archive, destination = Path(archive), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Data destination already exists: {destination}")
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    if checksum != datasets()["neojaundice"][0]["md5"]:
        raise ValueError("Archive does not match the official NeoJaundice v1 checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".defai-extract-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        (stage / "images").mkdir(parents=True)
        with zipfile.ZipFile(archive) as zipped:
            raw = zipped.read("NeoJaundice/chd_jaundice_published_2.csv")
            rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
            if not rows or "image_idx" not in rows[0]:
                raise ValueError("Archive metadata has an unexpected schema")
            for row in rows:
                name = row["image_idx"]
                if Path(name).name != name or "/" in name or "\\" in name or not name.lower().endswith(".jpg"):
                    raise ValueError("Invalid image filename in archive metadata")
                with zipped.open("NeoJaundice/images/" + name) as source, (stage / "images" / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
            (stage / "metadata.csv").write_bytes(raw)
        stage.rename(destination)
    return {"directory": str(destination), "images": len(rows), "stage": "research_only"}
