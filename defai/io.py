"""Atomic strict-JSON output; callers explicitly choose whether to overwrite."""
import json
import os
from pathlib import Path
import tempfile


def write_json(path, value, *, overwrite=True):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".defai-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard link publishes the complete file and fails if a destination exists.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
