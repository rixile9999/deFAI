"""Malicious-client behaviours for robustness experiments.

Clients with ``partition-id < num-malicious`` misbehave according to ``attack``:

* ``label-flip``  data poisoning: training labels y -> (num_classes - 1 - y)
* ``sign-flip``   model poisoning: sends ``global - scale * (local - global)``
* ``gaussian``    model poisoning: sends ``global + N(0, scale^2)`` instead of training

Only the update a malicious client *sends* is altered; its evaluation data and the
``num-examples`` it reports stay honest.
"""

import numpy as np

from defai.aggregation import Arrays

ATTACKS = ("none", "label-flip", "sign-flip", "gaussian")


def is_malicious(partition_id: int, num_malicious: int) -> bool:
    return partition_id < num_malicious


def flip_labels(y, num_classes: int):
    """Map class c to num_classes - 1 - c (works on tensors and ndarrays)."""
    return num_classes - 1 - y


def poison_update(
    attack: str,
    global_arrays: Arrays,
    local_arrays: Arrays,
    scale: float,
    rng: np.random.Generator,
) -> Arrays:
    """Model-poisoning attacks; data-poisoning attacks leave the update untouched."""
    if attack == "sign-flip":
        return {
            k: (global_arrays[k] - scale * (local_arrays[k] - global_arrays[k])).astype(local_arrays[k].dtype)
            for k in local_arrays
        }
    if attack == "gaussian":
        return {
            k: (global_arrays[k] + rng.normal(0.0, scale, global_arrays[k].shape)).astype(local_arrays[k].dtype)
            for k in local_arrays
        }
    return local_arrays
