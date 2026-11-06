from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MODALITIES


class JointProjection(nn.Module):
    def __init__(self, d: int = 64, d_joint: int = 32, modalities=MODALITIES):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d)) for m in modalities})
        self.reduce = nn.ModuleDict({m: nn.Linear(d, d_joint) for m in modalities})

    def tokens(self, m: str, x: torch.Tensor) -> torch.Tensor:
        return self.proj[m](x)

    def embed(self, m: str, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.reduce[m](pooled), dim=-1)


def contrastive_alignment(emb: Dict[str, torch.Tensor], mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    names = list(emb.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = mask[:, MODALITIES.index(names[i])] & mask[:, MODALITIES.index(names[j])]
            if both.sum() < 2:
                continue
            a, b = emb[names[i]][both], emb[names[j]][both]
            logits = a @ b.t() / temperature
            t = torch.arange(len(a), device=a.device)
            losses.append(0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.t(), t)))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=mask.device)


def joint_consistency(feats: torch.Tensor, fused: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = (feats - fused.unsqueeze(1)).norm(dim=-1)
    return (dist * mask.float()).sum(1).div(mask.float().sum(1).clamp(min=1)).mean()


@torch.no_grad()
def cross_modal_retrieval(emb_a: torch.Tensor, emb_b: torch.Tensor, y: torch.Tensor) -> float:
    sim = emb_a @ emb_b.t()
    sim.fill_diagonal_(-1e9)
    return float((y[sim.argmax(1)] == y).float().mean())
