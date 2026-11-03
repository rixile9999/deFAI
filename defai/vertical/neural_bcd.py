"""Neural FedBCD-style block updates using cached cut gradients/embeddings.

Passive blocks reuse a cut derivative and recompute their LOCAL Jacobian Q times.
The label-owner block updates its bottom and top using cached remote embeddings.
This realizes a cut-gradient reuse variant; no general neural convergence claim.
"""
import copy
import secrets
import torch
from .split import SplitTrainer


class NeuralFedBCD(SplitTrainer):
    def __init__(self, parties, top, labels, *, label_owner, q=3, mode='parallel', **kwargs):
        super().__init__(parties, top, labels, **kwargs)
        names = [p.name for p in self.parties]
        if label_owner not in names or mode not in {'parallel', 'sequential'}:
            raise ValueError('Need a participating label owner and valid block mode')
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError('q must be a positive integer')
        self.owner, self.q, self.mode = names.index(label_owner), q, mode
        self.local_steps = 0

    def _leaf_gradients(self, cached, indices):
        leaves = [x.detach().clone().requires_grad_(True) for x in cached]
        loss = self.loss(self.top(torch.cat(leaves, -1)), self.labels[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite block loss')
        return torch.autograd.grad(loss, leaves)

    def _local_update(self, party, indices, gradient):
        token = secrets.token_hex(16)
        party.embedding(indices, token)
        party.accept_gradient(gradient, token)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in party.model.parameters()):
            raise FloatingPointError('Non-finite local block gradient')
        party.commit()

    def step(self, indices):
        if self.q == 1 and self.mode == 'parallel':
            loss = super().step(indices)
            self.local_steps += 1
            return loss
        indices = self._indices(indices)
        modules = [p.model for p in self.parties]+[self.top]
        optimizers = [p.optimizer for p in self.parties]+[self.optimizer]
        saved = [copy.deepcopy(m.state_dict()) for m in modules]
        optimizer_states = [copy.deepcopy(o.state_dict()) if o else None for o in optimizers]
        traffic_state = copy.deepcopy(self.traffic)
        rng_state = torch.random.get_rng_state()
        try:
            cached = []
            for party in self.parties:
                value = party.embedding(indices, secrets.token_hex(16))
                cached.append(value.detach())
                party.abort()
                self.traffic.record(party.name, 'embedding', value)
            gradients = self._leaf_gradients(cached, indices)
            for i, party in enumerate(self.parties):
                if i == self.owner:
                    continue
                if self.mode == 'sequential':
                    gradients = self._leaf_gradients(cached, indices)
                gradient = gradients[i].detach()
                if self.defense is not None:
                    gradient = self.defense(gradient)
                self.traffic.record(party.name, 'gradient', gradient)
                for _ in range(self.q):
                    self._local_update(party, indices, gradient)
                if self.mode == 'sequential':
                    value = party.embedding(indices, secrets.token_hex(16))
                    cached[i] = value.detach()
                    party.abort()
                    self.traffic.record(party.name, 'embedding', value)
            owner = self.parties[self.owner]
            for _ in range(self.q):
                self.optimizer.zero_grad(set_to_none=True)
                token = secrets.token_hex(16)
                local = owner.embedding(indices, token)
                block = [local if i == self.owner else value for i, value in enumerate(cached)]
                loss = self.loss(self.top(torch.cat(block, -1)), self.labels[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite label-owner loss')
                loss.backward()
                owner.accept_gradient(local.grad, token)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for m in modules for p in m.parameters()):
                    raise FloatingPointError('Non-finite block gradient')
                owner.commit()
                self.optimizer.step()
        except Exception:
            for model, state in zip(modules, saved):
                model.load_state_dict(state)
            for optimizer, state in zip(optimizers, optimizer_states):
                if optimizer:
                    optimizer.load_state_dict(state)
            for party in self.parties:
                party.abort()
            self.traffic = traffic_state
            torch.random.set_rng_state(rng_state)
            raise
        self.steps += 1
        self.local_steps += self.q
        self.traffic.exchanges += 1
        return float(loss.detach())
