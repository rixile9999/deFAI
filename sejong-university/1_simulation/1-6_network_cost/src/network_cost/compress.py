import numpy as np


def compress_update(update, ratio=0.1, bits=8):
    flat = update.ravel()
    k = max(1, int(np.ceil(ratio * flat.size)))
    idx = np.argpartition(np.abs(flat), -k)[-k:] if k < flat.size else np.arange(flat.size)
    vals = flat[idx]
    lo, hi = float(vals.min()), float(vals.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    q = np.round((vals - lo) / scale).astype(np.uint32)
    index_bytes = 4 if flat.size > 65535 else 2
    nbytes = k * (bits / 8 + index_bytes) + 8
    if k >= flat.size:
        nbytes = flat.size * bits / 8 + 8
    out = np.zeros_like(flat)
    out[idx] = lo + q * scale
    return out.reshape(update.shape), int(np.ceil(nbytes))


def compress_arrays(local, reference, ratio=0.1, bits=8, residual=None):
    decoded, total, new_residual = {}, 0, {}
    for k, v in local.items():
        if not np.issubdtype(v.dtype, np.floating):
            decoded[k] = v
            total += v.nbytes
            continue
        d = v.astype(np.float64) - reference[k].astype(np.float64)
        if residual is not None and k in residual:
            d = d + residual[k]
        approx, nb = compress_update(d, ratio, bits)
        new_residual[k] = d - approx
        decoded[k] = (reference[k].astype(np.float64) + approx).astype(v.dtype)
        total += nb
    return decoded, total, new_residual
