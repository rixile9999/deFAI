from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .encoders import SinusoidalPE


class ModalityEmbeddingIntegration(nn.Module):
    def __init__(self, d: int, n_mod: int = 3, max_len: int = 64, learnable_weights: bool = True):
        super().__init__()
        self.n_mod = n_mod
        self.type_emb = nn.Embedding(n_mod + 1, d)
        base = SinusoidalPE(d, max_len).pe
        self.pos = nn.ParameterList([nn.Parameter(base.clone()) for _ in range(n_mod)])
        self.learnable_weights = learnable_weights
        self.logit = nn.Parameter(torch.zeros(n_mod))
        self.ctx = nn.Linear(d, 1)

    def forward(self, tokens: List[torch.Tensor], pads: List[torch.Tensor], pooled: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        B = mask.size(0)
        if self.learnable_weights:
            logits = self.logit.unsqueeze(0) + self.ctx(pooled).squeeze(-1)
        else:
            logits = torch.zeros(B, self.n_mod, device=mask.device)
        alpha = torch.softmax(logits.masked_fill(~mask, -1e4), 1) * mask.float()
        fuse = (alpha.unsqueeze(-1) * pooled).sum(1, keepdim=True) + self.type_emb.weight[self.n_mod]
        seq, pad, lens = [fuse], [torch.zeros(B, 1, dtype=torch.bool, device=mask.device)], []
        for m, (t, p) in enumerate(zip(tokens, pads)):
            scale = (alpha[:, m] * self.n_mod).view(B, 1, 1) if self.learnable_weights else 1.0
            seq.append(t * scale + self.pos[m][: t.size(1)].unsqueeze(0) + self.type_emb.weight[m])
            pad.append(p | (~mask[:, m]).unsqueeze(1))
            lens.append(t.size(1))
        return torch.cat(seq, 1), torch.cat(pad, 1), lens, alpha


class MultimodalTransformerEncoder(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, n_heads: int = 4, ff: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, n_heads, ff * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, seq: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.norm(self.enc(seq, src_key_padding_mask=pad))


class CrossModalDecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, ff: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, tgt_pad, src, src_pad):
        h = self.n1(tgt)
        tgt = tgt + self.drop(self.self_attn(h, h, h, key_padding_mask=tgt_pad)[0])
        h = self.n2(tgt)
        a, w = self.cross_attn(h, src, src, key_padding_mask=src_pad)
        tgt = tgt + self.drop(a)
        return tgt + self.drop(self.ff(self.n3(tgt))), w


class CrossModalTransformerDecoder(nn.Module):
    def __init__(self, d: int, n_layers: int = 1, n_heads: int = 4, ff: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([CrossModalDecoderLayer(d, n_heads, ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, H: torch.Tensor, pad: torch.Tensor, lens: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        bounds, s = [], 1
        for L in lens:
            bounds.append((s, s + L))
            s += L
        outs, attn = [], []
        for t, (a, b) in enumerate(bounds):
            src_idx = [0] + [i for k, (c, e) in enumerate(bounds) if k != t for i in range(c, e)]
            src, src_pad = H[:, src_idx], pad[:, src_idx]
            tgt, tgt_pad = H[:, a:b], pad[:, a:b].clone()
            tgt_pad[:, 0] = False
            w = None
            for layer in self.layers:
                tgt, w = layer(tgt, tgt_pad, src, src_pad)
            outs.append(self.norm(tgt))
            attn.append(w)
        return outs, attn
