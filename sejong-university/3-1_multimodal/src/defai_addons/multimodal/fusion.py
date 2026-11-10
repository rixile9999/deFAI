from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask, NEG), dim=dim) * mask.float()


class DynamicCoAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = ~mask
        s, _ = self.self_attn(z, z, z, key_padding_mask=pad)
        fused0 = (z * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze(-1).clamp(min=1)
        c, _ = self.cross_attn(z, fused0, fused0)
        alpha = masked_softmax(self.score(torch.cat([z, s, c], -1)).squeeze(-1), mask)
        return (alpha.unsqueeze(-1) * z).sum(1), alpha


class HierarchicalFeatureFusion(nn.Module):
    def __init__(self, d: int, n_mod: int = 3):
        super().__init__()
        self.level1 = nn.Sequential(nn.Linear(n_mod * d, d), nn.ReLU(), nn.LayerNorm(d))
        self.level2 = nn.Sequential(nn.Linear((n_mod + 1) * d, d), nn.ReLU(), nn.LayerNorm(d))

    def forward(self, low: torch.Tensor, high: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        l1 = self.level1((low * m).flatten(1))
        return self.level2(torch.cat([(high * m).flatten(1), l1], -1))


class CoAttentionFusion(nn.Module):
    def __init__(self, d: int, n_mod: int = 3):
        super().__init__()
        self.pairs = [(i, j) for i in range(n_mod) for j in range(i + 1, n_mod)]
        self.W = nn.ParameterList([nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d)) for _ in self.pairs])
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.LayerNorm(d))
        self.score = nn.Linear(d, 1)
        self.scale = d ** -0.5

    def forward(self, H: List[torch.Tensor], pads: List[torch.Tensor], mask: torch.Tensor):
        feats, scores, valid, mats = [], [], [], {}
        for k, (i, j) in enumerate(self.pairs):
            A = (H[i] @ self.W[k] @ H[j].transpose(1, 2)) * self.scale
            A = A.masked_fill(pads[j].unsqueeze(1), NEG).masked_fill(pads[i].unsqueeze(2), NEG)
            hi = torch.softmax(A, 2) @ H[j]
            hj = torch.softmax(A, 1).transpose(1, 2) @ H[i]
            keep_i, keep_j = (~pads[i]).unsqueeze(-1).float(), (~pads[j]).unsqueeze(-1).float()
            p = self.mlp(torch.cat([(hi * keep_i).sum(1) / keep_i.sum(1).clamp(min=1),
                                    (hj * keep_j).sum(1) / keep_j.sum(1).clamp(min=1)], -1))
            feats.append(p)
            scores.append(self.score(p).squeeze(-1))
            valid.append(mask[:, i] & mask[:, j])
            mats[(i, j)] = A
        P, S, Vd = torch.stack(feats, 1), torch.stack(scores, 1), torch.stack(valid, 1)
        Vd_eff = Vd | (~Vd.any(1, keepdim=True))
        beta = masked_softmax(S, Vd_eff)
        return (beta.unsqueeze(-1) * P).sum(1), beta, mats


class ModalityGAT(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dk = n_heads, d // n_heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.dk) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.dk) * 0.1)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, d = x.shape
        Wh = self.W(x).view(B, M, self.h, self.dk)
        e = F.leaky_relu((Wh * self.a_src).sum(-1).unsqueeze(2) + (Wh * self.a_dst).sum(-1).unsqueeze(1), 0.2)
        adj = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1).expand_as(e)
        alpha = masked_softmax(e, adj, dim=2)
        h = torch.einsum("bijh,bjhk->bihk", alpha, Wh).reshape(B, M, d)
        return self.norm(x + self.out(F.elu(h))), alpha.mean(-1)


class GatedMultimodalFusion(nn.Module):
    def __init__(self, d: int, n_mod: int = 3):
        super().__init__()
        self.h = nn.ModuleList([nn.Linear(d, d) for _ in range(n_mod)])
        self.gate = nn.Linear(n_mod * d, n_mod * d)
        self.n_mod = n_mod

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, d = x.shape
        hs = torch.stack([torch.tanh(self.h[m](x[:, m])) for m in range(M)], 1)
        g = self.gate((x * mask.unsqueeze(-1)).flatten(1)).view(B, M, d)
        g = masked_softmax(g.transpose(1, 2), mask.unsqueeze(1).expand(B, d, M)).transpose(1, 2)
        return (g * hs).sum(1), g.mean(-1)


class CrossModalAligner(nn.Module):
    def __init__(self, d: int, temperature: float = 0.1, iters: int = 5):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.t, self.iters = temperature, iters

    def attention(self, a: torch.Tensor, b: torch.Tensor, sinkhorn: bool = True) -> torch.Tensor:
        logits = F.normalize(self.q(a), dim=-1) @ F.normalize(self.k(b), dim=-1).t() / self.t
        if not sinkhorn:
            return torch.softmax(logits, 1)
        log_p = logits
        for _ in range(self.iters):
            log_p = log_p - torch.logsumexp(log_p, 1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, 0, keepdim=True)
        return log_p.exp()

    def loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        P = self.attention(a, b)
        return -torch.log(P.diagonal() + 1e-8).mean()

    @torch.no_grad()
    def match(self, a: torch.Tensor, b: torch.Tensor, sinkhorn: bool = True) -> torch.Tensor:
        return self.attention(a, b, sinkhorn).argmax(1)
