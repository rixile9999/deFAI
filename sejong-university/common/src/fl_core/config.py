import copy
import json

BUILTIN_DEFAULTS = {
    "dataset": "synthetic",
    "model": "cnn",
    "model-width": 1.0,
    "partitioner": "iid",
    "dirichlet-alpha": 0.5,
    "quantity-sigma": 1.0,
    "max-samples": 0,
    "seed": 0,
    "local-epochs": 1.0,
    "batch-size": 32,
    "learning-rate": 0.05,
    "momentum": 0.9,
    "proximal-mu": 0.01,
    "qfedavg-q": 0.1,
    "trim-ratio": 0.2,
    "num-byzantine": 1,
}


def merge(base, overrides):
    out = copy.deepcopy(dict(base))
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_sets(items):
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_value(value)
    return out
