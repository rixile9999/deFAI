"""Fail-closed weighted aggregation with integer buffers kept as integers."""
from collections import OrderedDict

import torch


def average_states(states, weights):
    if not states or len(states) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("Expected nonempty positive client weights")
    total = sum(weights)
    reference = states[0]
    result = OrderedDict()
    for state in states:
        if list(state) != list(reference):
            raise ValueError("Client model keys differ")
    for name, value in reference.items():
        if any(state[name].shape != value.shape or state[name].dtype != value.dtype for state in states):
            raise ValueError("Client model shape/dtype differs")
        if value.is_floating_point():
            if not all(torch.isfinite(state[name]).all() for state in states):
                raise ValueError("Nonfinite client update")
            result[name] = sum(state[name].float()*(weight/total) for state, weight in zip(states, weights)).to(value.dtype)
        else:
            if not all(torch.equal(state[name], value) for state in states):
                raise ValueError("Nonfloating buffers disagree; do not average integer counters")
            result[name] = value.clone()
    return result
