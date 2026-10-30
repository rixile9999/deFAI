"""Lossy compression of client updates, simulated: encode, count the bytes, decode.

The client sends the *decoded* (lossy) update as a dense array, so the server needs no
changes, and reports what the encoded message would have cost in the ``upload-bytes``
metric. Nothing here depends on Flower.

* ``topk``      keep the ``ratio`` largest-magnitude entries over the whole update (Aji & Heafield,
                2017; Lin et al., 2018): each kept entry costs a float32 value and a uint32 index.
* ``quantize``  per tensor, ``bits``-bit uniform quantisation with stochastic rounding, which is
                unbiased (QSGD, Alistarh et al., 2017): ``bits`` per entry plus a float32 scale.
* error feedback (Seide et al., 2014; Stich et al., 2018): the part of the update lost by
  compression is remembered and added to the next update, so nothing is lost for good.
"""

import math

import numpy as np

from defai.aggregation import Arrays, l2_norm

METHODS = ("none", "topk", "quantize")
FLOAT32_BYTES = 4


def dense_bytes(update: Arrays) -> int:
    """Cost of sending the update uncompressed, as float32."""
    return FLOAT32_BYTES * sum(a.size for a in update.values())


def topk(update: Arrays, ratio: float) -> tuple[Arrays, int]:
    """Zero all but the ``ceil(ratio * n)`` largest-magnitude entries; 8 bytes per kept entry."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    sizes = [a.size for a in update.values()]
    flat = np.concatenate([a.ravel().astype(np.float64) for a in update.values()])
    keep = min(flat.size, max(1, math.ceil(ratio * flat.size)))
    mask = np.zeros(flat.size, dtype=bool)
    mask[np.argpartition(-np.abs(flat), keep - 1)[:keep]] = True
    flat = np.where(mask, flat, 0.0)
    out, start = {}, 0
    for (name, a), n in zip(update.items(), sizes):
        out[name] = flat[start : start + n].reshape(a.shape).astype(a.dtype)
        start += n
    return out, 2 * FLOAT32_BYTES * keep


def quantize(update: Arrays, bits: int, rng: np.random.Generator) -> tuple[Arrays, int]:
    """Unbiased ``bits``-bit quantisation per tensor with stochastic rounding."""
    if not 2 <= bits <= 16:
        raise ValueError("bits must be in [2, 16]")
    levels = 2 ** (bits - 1) - 1
    out, nbytes = {}, 0
    for name, a in update.items():
        x = a.astype(np.float64)
        scale = float(np.max(np.abs(x))) if x.size else 0.0
        nbytes += math.ceil(x.size * bits / 8) + FLOAT32_BYTES  # the codes and the scale
        if scale == 0.0:
            out[name] = a.copy()
            continue
        y = x / scale * levels  # in [-levels, levels]
        low = np.floor(y)
        codes = low + (rng.random(x.shape) < (y - low))  # round up with probability = fractional part
        out[name] = (codes / levels * scale).astype(a.dtype)
    return out, nbytes


def compress(update: Arrays, method: str, *, ratio: float = 0.1, bits: int = 8, rng=None) -> tuple[Arrays, int]:
    """Return (decoded update, bytes the encoded message would take)."""
    if method == "none":
        return update, dense_bytes(update)
    if method == "topk":
        return topk(update, ratio)
    if method == "quantize":
        return quantize(update, bits, rng if rng is not None else np.random.default_rng())
    raise ValueError(f"unknown compression: {method!r} (choose from {', '.join(METHODS)})")


def compress_with_feedback(
    update: Arrays, residual: Arrays | None, method: str, **kwargs
) -> tuple[Arrays, int, Arrays]:
    """Compress ``update + residual``; the new residual is whatever the compression lost."""
    corrected = update if residual is None else {k: update[k] + residual[k] for k in update}
    decoded, nbytes = compress(corrected, method, **kwargs)
    return decoded, nbytes, {k: corrected[k] - decoded[k] for k in update}


def relative_error(original: Arrays, decoded: Arrays) -> float:
    """||decoded - original|| / ||original|| (0 for a lossless transfer)."""
    norm = l2_norm(original)
    return 0.0 if norm == 0 else l2_norm({k: decoded[k] - original[k] for k in original}) / norm
