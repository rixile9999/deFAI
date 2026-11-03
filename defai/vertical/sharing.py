"""Two-helper additive sum prototype over a fixed prime field.

Helpers must not collude; final recipient learns the sum. All endpoints run in one
process here. This is not general MPC or the complete VFGNN protocol.
"""
import secrets
import numpy as np

MODULUS = 2**61 - 1
SCALE = 10**6


def split_shares(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or np.max(np.abs(value), initial=0) * SCALE >= MODULUS // 4:
        raise ValueError('Value outside fixed-point secret-sharing range')
    encoded = np.array([int(round(float(x)*SCALE)) % MODULUS for x in value.ravel()], dtype=object).reshape(value.shape)
    first = np.array([secrets.randbelow(MODULUS) for _ in value.ravel()], dtype=object).reshape(value.shape)
    return first, (encoded-first) % MODULUS


def helper_sum(shares):
    shares = list(shares)
    if not shares or any(x.shape != shares[0].shape for x in shares):
        raise ValueError('Matching share shapes required')
    return sum(shares) % MODULUS


def reconstruct(first, second):
    if first.shape != second.shape:
        raise ValueError('Share shapes differ')
    encoded = (first+second) % MODULUS
    signed = np.where(encoded > MODULUS//2, encoded-MODULUS, encoded)
    return signed.astype(float)/SCALE


def shared_sum(values):
    values = [np.asarray(v, dtype=float) for v in values]
    if len(values) < 2 or any(v.shape != values[0].shape for v in values):
        raise ValueError('At least two matching local embeddings required')
    # Conservative bound checked locally before any modular sum can wrap.
    bound = sum(float(np.max(np.abs(v), initial=0)) for v in values)
    if bound * SCALE >= MODULUS//4:
        raise OverflowError('Aggregate can exceed the signed field range')
    shares = [split_shares(v) for v in values]
    return reconstruct(helper_sum(s[0] for s in shares), helper_sum(s[1] for s in shares))
