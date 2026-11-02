"""FedBCD-p/s for additive linear/logistic predictors (paper equations 2, 5, 7).

Passive parties reuse the cached loss derivative for Q local updates of the SAME
batch, including local L2/proximal terms. The label owner recomputes its derivative
against cached remote contributions. No labels are sent to passive-party methods.
Plain local protocol simulation; this is not arbitrary neural FedBCD.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class LinearParty:
    name: str
    x: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim != 2 or not len(self.x) or not np.isfinite(self.x).all():
            raise ValueError('Expected finite local matrix')
        self.weight = np.zeros(self.x.shape[1])

    def contribution(self, batch):
        return self.x[batch] @ self.weight

    def local_updates(self, batch, derivative, *, q, lr, l2, proximal):
        start = self.weight.copy()
        data_gradient = self.x[batch].T @ derivative / len(batch)
        for _ in range(q):
            self.weight -= lr * (data_gradient + l2 * self.weight + proximal * (self.weight - start))


class FedBCD:
    def __init__(self, parties, labels, *, label_owner, task='logistic', lr=.1, q=1,
                 mode='parallel', l2=0., proximal=0.):
        self.parties = list(parties)
        names = [p.name for p in self.parties]
        self.labels = np.asarray(labels, dtype=float)
        if len(names) < 2 or len(set(names)) != len(names) or label_owner not in names:
            raise ValueError('Need distinct parties and a label owner')
        if self.labels.ndim != 1 or not np.isfinite(self.labels).all() or any(len(p.x) != len(labels) for p in self.parties):
            raise ValueError('Labels and feature rows must be aligned')
        if task not in {'logistic', 'regression'} or mode not in {'parallel', 'sequential'}:
            raise ValueError('Unknown task or FedBCD mode')
        if task == 'logistic' and not np.isin(self.labels, [0, 1]).all():
            raise ValueError('Binary labels required')
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError('q must be a positive integer')
        if any(not math.isfinite(v) or v < 0 for v in (lr, l2, proximal)) or lr == 0:
            raise ValueError('Invalid optimizer setting')
        self.owner = names.index(label_owner)
        self.task, self.lr, self.q, self.mode, self.l2, self.proximal = task, lr, q, mode, l2, proximal
        self.rounds = self.local_steps = self.bytes = 0
        self.batch_history = []

    def _derivative(self, score, batch):
        if self.task == 'regression':
            return score - self.labels[batch]
        return np.exp(-np.logaddexp(0, -score)) - self.labels[batch]

    def step(self, batch):
        batch = np.asarray(batch)
        if batch.ndim != 1 or batch.dtype.kind not in 'iu' or not len(batch) or len(set(batch.tolist())) != len(batch):
            raise ValueError('Nonempty distinct integer batch indices required')
        if batch.min() < 0 or batch.max() >= len(self.labels):
            raise ValueError('Batch index outside aligned sample range')
        cached = [p.contribution(batch) for p in self.parties]
        old = [p.weight.copy() for p in self.parties]
        residual = self._derivative(sum(cached), batch)
        for i, party in enumerate(self.parties):
            if i == self.owner:
                others = sum(v for j, v in enumerate(cached) if j != i)
                start = party.weight.copy()
                for _ in range(self.q):
                    derivative = self._derivative(others + party.contribution(batch), batch)
                    gradient = party.x[batch].T @ derivative / len(batch)
                    party.weight -= self.lr * (gradient + self.l2 * party.weight + self.proximal * (party.weight - start))
            else:
                party.local_updates(batch, residual, q=self.q, lr=self.lr, l2=self.l2, proximal=self.proximal)
            if self.mode == 'sequential':
                cached[i] = party.contribution(batch)
                residual = self._derivative(sum(cached), batch)
        if any(not np.isfinite(p.weight).all() for p in self.parties):
            for party, weight in zip(self.parties, old):
                party.weight[:] = weight
            raise FloatingPointError('Non-finite BCD update reverted')
        # Per passive party: contribution -> owner; derivative <- owner.
        # Sequential also refreshes both after each passive block.
        exchanges = len(self.parties) - 1
        if self.mode == 'sequential':
            exchanges *= 2
        self.bytes += 2 * exchanges * len(batch) * 8
        self.rounds += 1
        self.local_steps += self.q
        self.batch_history.append({'indices': batch.tolist(), 'reused_steps': self.q})
        return self.loss()

    def predict(self):
        score = sum(p.contribution(np.arange(len(self.labels))) for p in self.parties)
        return np.exp(-np.logaddexp(0, -score)) if self.task == 'logistic' else score

    def loss(self):
        score = sum(p.contribution(np.arange(len(self.labels))) for p in self.parties)
        data = np.mean(np.logaddexp(0, score) - self.labels * score) if self.task == 'logistic' else np.mean((score - self.labels)**2) / 2
        return float(data + self.l2 * sum(p.weight @ p.weight for p in self.parties) / 2)
