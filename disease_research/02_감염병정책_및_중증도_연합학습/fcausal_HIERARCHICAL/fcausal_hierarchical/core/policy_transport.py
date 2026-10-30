"""Version/configuration checks shared by the three Flower tiers."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from core.policy_stats import SCHEMA, VECTOR_SIZE


def normalize_region(region):
    return {"Seoul": "서울", "Gyeonggi": "경기", "Busan": "부산"}.get(region, region)


def load_context(key_dir, secret=False):
    import tenseal as ts
    key_dir = Path(key_dir)
    public = (key_dir / "public.ctx").read_bytes()
    context = ts.context_from((key_dir / "secret.ctx").read_bytes() if secret else public)
    if not secret and context.has_secret_key():
        raise ValueError("Tier 1/2 must use a public-only context")
    if secret:
        probe = ts.ckks_vector(ts.context_from(public), [1.25]).serialize()
        if abs(ts.ckks_vector_from(context, probe).decrypt()[0] - 1.25) > 1e-4:
            raise ValueError("Public and secret key files do not match")
    return context, sha256(public).hexdigest()


def metadata(config, encrypted, key_id):
    return {"schema": SCHEMA, "analysis_config": config.signature,
            "encrypted": bool(encrypted), "key_id": key_id}


def validate_metadata(metrics, config, encrypted, key_id):
    for field, value in metadata(config, encrypted, key_id).items():
        if metrics.get(field) != value:
            raise ValueError(f"Protocol mismatch: {field}")


def encode(vector, context):
    if context is None:
        return [np.asarray(vector, dtype=np.float64)]
    import tenseal as ts
    return [np.frombuffer(ts.ckks_vector(context, vector.tolist()).serialize(), dtype=np.uint8)]


def decode(arrays, context):
    if len(arrays) != 1:
        raise ValueError("Expected exactly one statistic vector")
    if context is None:
        vector = np.asarray(arrays[0], dtype=np.float64)
        if vector.shape != (VECTOR_SIZE,):
            raise ValueError("Invalid plaintext statistic vector")
        return vector
    import tenseal as ts
    vector = ts.ckks_vector_from(context, arrays[0].tobytes())
    if vector.size() != VECTOR_SIZE:
        raise ValueError("Invalid encrypted statistic vector")
    return vector
