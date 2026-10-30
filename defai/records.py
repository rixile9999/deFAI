"""Conversions between Flower ArrayRecords and plain ``dict[str, ndarray]``."""

import numpy as np
from flwr.app import Array, ArrayRecord

from defai.aggregation import Arrays

KEY_C = "c"                  # SCAFFOLD server control variate (train message)
KEY_C_DELTA = "c-delta"      # SCAFFOLD client control-variate change (train reply)
STATE_SCAFFOLD_C = "scaffold-c"        # ClientApp state: this client's control variate
STATE_MOON_PREVIOUS = "moon-previous"  # ClientApp state: this client's previous local model
STATE_EF_RESIDUAL = "ef-residual"      # ClientApp state: update mass lost to compression (error feedback)


def to_arrays(record: ArrayRecord) -> Arrays:
    """ArrayRecord -> dict of ndarrays (key order preserved)."""
    return {k: v.numpy() for k, v in record.items()}


def to_record(arrays: Arrays) -> ArrayRecord:
    """dict of ndarrays -> ArrayRecord (key order preserved)."""
    return ArrayRecord({k: Array(np.asarray(v)) for k, v in arrays.items()})


def record_bytes(record: ArrayRecord) -> int:
    """Size of an ArrayRecord's payload, counted like Flower does (``len(array.data)``)."""
    return sum(len(array.data) for array in record.values())
