from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: Sequence[int] = (128, 64), dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_vec(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def set_vec(model: nn.Module, vec: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_bytes(model: nn.Module, bytes_per_param: int = 4) -> int:
    return n_params(model) * bytes_per_param
