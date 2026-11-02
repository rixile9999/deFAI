"""Small, trainable feature-party encoders. No pretrained weights are downloaded."""
import torch
from torch import nn


def mlp(features, embedding=16):
    return nn.Sequential(nn.Linear(features, 32), nn.ReLU(), nn.Linear(32, embedding))


class ImageEncoder(nn.Module):
    def __init__(self, channels=1, embedding=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, embedding))

    def forward(self, x):
        return self.net(x)


class RecurrentEncoder(nn.Module):
    def __init__(self, features, embedding=16, *, cell='lstm', sequence=False):
        super().__init__()
        if cell not in {'lstm', 'gru'}:
            raise ValueError('cell must be lstm or gru')
        self.rnn = (nn.LSTM if cell == 'lstm' else nn.GRU)(features, embedding, batch_first=True)
        self.sequence = sequence

    def forward(self, x):
        out, _ = self.rnn(x)
        return out if self.sequence else out[:, -1]


class CausalConv(nn.Module):
    def __init__(self, channels, hidden, dilation):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(channels, hidden, 3, dilation=dilation)

    def forward(self, x):
        return torch.relu(self.conv(nn.functional.pad(x, (self.left, 0))))


class TemporalEncoder(nn.Module):
    def __init__(self, features, embedding=16, *, sequence=False):
        super().__init__()
        self.net = nn.Sequential(CausalConv(features, embedding, 1), CausalConv(embedding, embedding, 2))
        self.sequence = sequence

    def forward(self, x):
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        return out if self.sequence else out[:, -1]


class TransformerEncoder(nn.Module):
    def __init__(self, features, embedding=16, *, tokens=False, max_length=512):
        super().__init__()
        if embedding % 2:
            raise ValueError('Embedding dimension must be divisible by 2 attention heads')
        self.project = nn.Linear(features, embedding)
        self.position = nn.Parameter(torch.zeros(1, max_length, embedding))
        nn.init.normal_(self.position, std=.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding, 2, embedding * 2,
                                            dropout=0., batch_first=True), 1, enable_nested_tensor=False)
        self.tokens = tokens

    def forward(self, x):
        if x.shape[1] > self.position.shape[1]:
            raise ValueError('Sequence exceeds configured maximum length')
        out = self.encoder(self.project(x) + self.position[:, :x.shape[1]])
        return out if self.tokens else out.mean(1)


class FrozenEncoder(nn.Module):
    """Wrap a user-supplied local language encoder; never download a model.

    Encoder must return a tensor [batch,features] or [batch,tokens,features].
    The small demo encoder is NOT a large language model.
    """
    def __init__(self, encoder, *, pool=True):
        super().__init__()
        self.encoder, self.pool = encoder, pool
        encoder.requires_grad_(False)
        encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            result = self.encoder(x)
        if not isinstance(result, torch.Tensor) or result.ndim not in (2, 3):
            raise ValueError('Adapt the local encoder to return a tensor')
        return result.mean(1) if result.ndim == 3 and self.pool else result


class GraphEncoder(nn.Module):
    """Local full-graph GCN; does not implement VFGNN's MPC/security protocol."""
    requires_full_graph = True

    def __init__(self, features, adjacency, embedding=16):
        super().__init__()
        a = torch.as_tensor(adjacency, dtype=torch.float32)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not torch.isfinite(a).all() or (a < 0).any():
            raise ValueError('Expected a finite nonnegative square adjacency matrix')
        a = a + torch.eye(len(a))
        degree = a.sum(1).rsqrt()
        self.register_buffer('adjacency', degree[:, None] * a * degree[None, :])
        self.first, self.second = nn.Linear(features, embedding), nn.Linear(embedding, embedding)

    def forward(self, x):
        if len(x) != len(self.adjacency):
            raise ValueError('All locally aligned graph nodes are required')
        return self.adjacency @ self.second(torch.relu(self.adjacency @ self.first(x)))
