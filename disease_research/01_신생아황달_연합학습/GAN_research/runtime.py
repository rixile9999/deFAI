"""Explicit, portable device selection; CPU is the default."""
import os
import torch

DEVICE = torch.device(os.environ.get("DEFAI_DEVICE", "cpu"))


def require_device():
    if DEVICE.type not in {"cpu", "cuda"}:
        raise ValueError("DEFAI_DEVICE must be cpu or cuda")
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu")


def device_name():
    return torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "cpu"
