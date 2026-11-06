import math

import torch
import torch.nn as nn

from .data import AUDIO_LEN, PAD, SEQ_LEN, VOCAB


class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class ImageEncoder(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, d, 3, padding=1), nn.BatchNorm2d(d), nn.ReLU())
        self.low = nn.Linear(32, d)

    def forward(self, x):
        h1 = self.block1(x)
        h = self.block3(self.block2(h1))
        tokens = h.flatten(2).transpose(1, 2)
        pad = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=x.device)
        return tokens, pad, self.low(h1.mean((2, 3)))


class TextEncoder(nn.Module):
    def __init__(self, d: int = 64, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pe = SinusoidalPE(d, SEQ_LEN + 4)
        layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        pad = x == PAD
        pad[:, 0] = False
        e = self.emb(x)
        low = (e * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        return self.enc(self.pe(e), src_key_padding_mask=pad), pad, low


class AudioEncoder(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, 9, stride=4, padding=4), nn.BatchNorm1d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 9, stride=4, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.rnn = nn.GRU(64, d // 2, batch_first=True, bidirectional=True)
        self.low = nn.Linear(32, d)

    def forward(self, x):
        h1 = self.conv1(x)
        h = self.conv2(h1).transpose(1, 2)
        tokens, _ = self.rnn(h)
        pad = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=x.device)
        return tokens, pad, self.low(h1.mean(2))


def build_encoders(d: int = 64) -> nn.ModuleDict:
    return nn.ModuleDict({"image": ImageEncoder(d), "text": TextEncoder(d), "audio": AudioEncoder(d)})


def masked_mean(tokens: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    keep = (~pad).unsqueeze(-1).float()
    return (tokens * keep).sum(1) / keep.sum(1).clamp(min=1.0)


AUDIO_TOKENS = AUDIO_LEN // 32
