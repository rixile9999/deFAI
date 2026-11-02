"""Detached embedding/gradient boundary with local bottom-model ownership.

In-process research runtime, not network isolation or encrypted neural computation.
"""
from dataclasses import dataclass, field
import math
import torch
from torch import nn


@dataclass
class Traffic:
    forward_bytes: int = 0
    backward_bytes: int = 0
    exchanges: int = 0
    events: list = field(default_factory=list)

    def record(self, party, direction, tensor):
        size = tensor.numel() * tensor.element_size()
        if direction == 'embedding':
            self.forward_bytes += size
        else:
            self.backward_bytes += size
        self.events.append({'party': party, 'direction': direction, 'shape': list(tensor.shape), 'bytes': size})


class FeatureParty:
    def __init__(self, name, model, features, *, lr=.01):
        if not name or not math.isfinite(lr) or lr <= 0:
            raise ValueError('A name and positive finite learning rate are required')
        if not isinstance(features, torch.Tensor) or features.ndim < 2 or not len(features) or not torch.isfinite(features).all():
            raise ValueError('Expected finite local tensor [samples,...]')
        self.name, self.model, self.features = name, model, features
        parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(parameters, lr=lr) if parameters else None
        self._pending = None

    def embedding(self, indices, token):
        if self._pending is not None:
            raise RuntimeError('Outstanding batch must be completed or aborted')
        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=True)
        if getattr(self.model, 'requires_full_graph', False):
            local = self.model(self.features)[indices]
        else:
            local = self.model(self.features[indices])
        if not torch.isfinite(local).all():
            raise FloatingPointError('Non-finite local embedding')
        self._pending = (token, local)
        return local.detach().clone().requires_grad_(True)

    def accept_gradient(self, gradient, token):
        if self._pending is None or self._pending[0] != token:
            raise RuntimeError('Stale or replayed gradient')
        local = self._pending[1]
        if gradient.shape != local.shape or not torch.isfinite(gradient).all():
            raise ValueError('Invalid boundary gradient')
        if local.requires_grad:
            local.backward(gradient.detach())

    def commit(self):
        if self.optimizer:
            self.optimizer.step()
        self._pending = None

    def abort(self):
        self._pending = None
        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=True)


class SplitTrainer:
    def __init__(self, parties, top, labels, *, task='classification', lr=.01, defense=None):
        self.parties = list(parties)
        if len(self.parties) < 2 or len({p.name for p in self.parties}) != len(self.parties):
            raise ValueError('At least two uniquely named feature parties are required')
        n = len(self.parties[0].features)
        if any(len(p.features) != n for p in self.parties) or len(labels) != n or not torch.isfinite(labels).all():
            raise ValueError('Align all sample rows before training')
        if task not in {'classification', 'regression'} or not math.isfinite(lr) or lr <= 0:
            raise ValueError('Invalid task or learning rate')
        self.top, self.labels, self.task, self.defense = top, labels, task, defense
        self.optimizer = torch.optim.SGD(top.parameters(), lr=lr)
        self.traffic, self.steps = Traffic(), 0

    def _indices(self, indices):
        indices = torch.as_tensor(indices)
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64) or not len(indices):
            raise ValueError('Batch indices must be a nonempty integer vector')
        if len(indices.unique()) != len(indices) or (indices < 0).any() or (indices >= len(self.labels)).any():
            raise ValueError('Duplicate or out-of-range batch index')
        return indices.long()

    def loss(self, prediction, target):
        if self.task == 'classification':
            if target.dtype != torch.long or prediction.shape[:-1] != target.shape:
                raise ValueError('Classification needs integer labels matching prediction rows')
            return nn.functional.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), target.reshape(-1))
        if prediction.shape != target.shape:
            raise ValueError('Regression prediction and label shapes must match')
        return nn.functional.mse_loss(prediction, target)

    def step(self, indices):
        indices = self._indices(indices)
        token = self.steps
        self.optimizer.zero_grad(set_to_none=True)
        try:
            sent = [p.embedding(indices, token) for p in self.parties]
            for p, value in zip(self.parties, sent):
                self.traffic.record(p.name, 'embedding', value)
            prediction = self.top(torch.cat(sent, dim=-1))
            loss = self.loss(prediction, self.labels[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            loss.backward()
            for party, leaf in zip(self.parties, sent):
                gradient = leaf.grad.detach().clone()
                if self.defense is not None:
                    gradient = self.defense(gradient)
                self.traffic.record(party.name, 'gradient', gradient)
                party.accept_gradient(gradient, token)
            parameters = [p for party in self.parties for p in party.model.parameters()] + list(self.top.parameters())
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Non-finite parameter gradient; no optimizer step applied')
            for party in self.parties:
                party.commit()
            self.optimizer.step()
        except Exception:
            for party in self.parties:
                party.abort()
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.steps += 1
        self.traffic.exchanges += 1
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, indices):
        indices = self._indices(indices)
        modules = [p.model for p in self.parties] + [self.top]
        modes = [m.training for m in modules]
        try:
            for model in modules:
                model.eval()
            values = [(p.model(p.features)[indices] if getattr(p.model, 'requires_full_graph', False)
                       else p.model(p.features[indices])) for p in self.parties]
            return self.top(torch.cat(values, dim=-1))
        finally:
            for model, mode in zip(modules, modes):
                model.train(mode)
