"""MNIST and Fashion-MNIST from plain-file mirrors, for environments without Hugging Face access.

The files are the original IDX archives. Every download is checked against a pinned SHA-256 sum before
it is kept; files are cached under ``$DEFAI_DATA_DIR`` (default ``~/.cache/defai``), so the clients of a
simulation read them from disk. Nothing here depends on Flower or torch.
"""

import gzip
import hashlib
import http.client
import os
import struct
import tempfile
import time
import urllib.error
import urllib.request
import warnings

import numpy as np

MIRRORS = {
    "mnist-mirror": {
        "url": "https://ossci-datasets.s3.amazonaws.com/mnist/",  # the mirror torchvision downloads MNIST from
        "sha256": {
            "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
            "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
            "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
            "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        },
    },
    "fashion-mnist-mirror": {
        "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",  # the mirror Keras downloads it from
        "sha256": {
            "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
            "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
            "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
            "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
        },
    },
}
SPLIT_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
}
IMG_SIZE = 28
NUM_CLASSES = 10


def cache_dir() -> str:
    return os.environ.get("DEFAI_DATA_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "defai")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, urlopen, retries: int, sleep) -> bytes:
    """Body of ``url``; connection problems and server errors are retried with a growing pause."""
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == retries - 1:  # a missing file will not appear on the second try
                raise
        except (OSError, http.client.HTTPException):  # reset connection, timeout, truncated body
            if attempt == retries - 1:
                raise
        sleep(2**attempt)


def fetch(name: str, filename: str, directory: str | None = None, urlopen=urllib.request.urlopen, retries: int = 3, sleep=time.sleep) -> bytes:
    """Contents of one archive: from the cache if it is intact, otherwise downloaded, verified and cached."""
    mirror = MIRRORS[name]
    expected = mirror["sha256"][filename]
    path = os.path.join(directory or cache_dir(), name, filename)
    try:
        with open(path, "rb") as f:
            data = f.read()
        if _sha256(data) == expected:
            return data
    except OSError:
        pass  # not cached yet, or not readable: download it
    data = _download(mirror["url"] + filename, urlopen, retries, sleep)
    if _sha256(data) != expected:
        raise ValueError(f"{mirror['url'] + filename}: checksum mismatch, refusing to use the download")
    tmp = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))  # write-then-rename: parallel clients never see half a file
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; other users of a shared cache need to read it
        os.replace(tmp, path)
    except OSError as error:  # the data is verified and in memory: failing to cache it is not fatal
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
        warnings.warn(f"could not cache {filename} in {os.path.dirname(path)}: {error}", stacklevel=2)
    return data


def read_idx(data: bytes) -> np.ndarray:
    """Decode a gzip-compressed IDX file of unsigned bytes into an array of the stored shape."""
    raw = gzip.decompress(data)
    zero, dtype, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype != 0x08:
        raise ValueError("not an IDX file of unsigned bytes")
    shape = struct.unpack(f">{ndim}I", raw[4 : 4 + 4 * ndim])
    values = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if values.size != int(np.prod(shape)):
        raise ValueError("IDX file is truncated")
    return values.reshape(shape)


def load_split(name: str, split: str, directory: str | None = None, urlopen=urllib.request.urlopen):
    """(images, labels) of a split as uint8 arrays of shape (n, 28, 28) and (n,)."""
    images_file, labels_file, expected = SPLIT_FILES[split]
    images = read_idx(fetch(name, images_file, directory, urlopen))
    labels = read_idx(fetch(name, labels_file, directory, urlopen))
    if images.shape != (expected, IMG_SIZE, IMG_SIZE) or labels.shape != (expected,) or labels.max() >= NUM_CLASSES:
        raise ValueError(f"{name} {split}: unexpected shapes {images.shape} / {labels.shape}")
    return images, labels
