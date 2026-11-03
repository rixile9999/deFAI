"""Unlabeled vertical AE/VAE with private local reconstruction targets."""
import torch
from torch import nn


class VerticalAutoencoder:
    def __init__(self, features, *, latent=4, variational=False, lr=.01, beta=.01):
        self.features = [torch.as_tensor(x, dtype=torch.float32) for x in features]
        if len(self.features) < 2 or latent < 1 or lr <= 0 or beta < 0:
            raise ValueError('Need >=2 parties, positive latent/lr and nonnegative beta')
        if any(x.ndim != 2 or not len(x) or not torch.isfinite(x).all() or len(x) != len(self.features[0]) for x in self.features):
            raise ValueError('Expected aligned finite local feature matrices')
        self.variational, self.beta = variational, beta
        self.encoders = [nn.Linear(x.shape[1], latent*(2 if variational else 1)) for x in self.features]
        self.decoders = [nn.Linear(latent*len(features), x.shape[1]) for x in self.features]
        self.optimizers = [torch.optim.Adam(list(e.parameters())+list(d.parameters()), lr=lr)
                           for e, d in zip(self.encoders, self.decoders)]
        self.last_local_losses = []

    def step(self, indices, *, generator=None):
        if not len(indices):
            raise ValueError('Empty batch')
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        local, kl = [], []
        for encoder, x in zip(self.encoders, self.features):
            encoded = encoder(x[indices])
            if self.variational:
                mean, logvar = encoded.chunk(2, -1)
                logvar = logvar.clamp(-20, 10)
                eps = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
                local.append(mean + eps*(.5*logvar).exp())
                kl.append(.5*(mean.square()+logvar.exp()-1-logvar).sum(-1).mean())
            else:
                local.append(encoded)
        sent = [z.detach().clone().requires_grad_(True) for z in local]
        joint = torch.cat(sent, -1)
        returned = torch.zeros_like(joint)
        losses = []
        for decoder, x in zip(self.decoders, self.features):
            received = joint.detach().clone().requires_grad_(True)
            loss = nn.functional.mse_loss(decoder(received), x[indices])
            loss.backward()
            returned += received.grad
            losses.append(float(loss.detach()))
        joint.backward(returned)
        for i, z in enumerate(local):
            z.backward(sent[i].grad, retain_graph=self.variational)
            if self.variational:
                (self.beta*kl[i]).backward()
        parameters = [p for m in self.encoders+self.decoders for p in m.parameters()]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Non-finite reconstruction gradient')
        for optimizer in self.optimizers:
            optimizer.step()
        self.last_local_losses = losses
        return sum(losses) + self.beta*sum(float(k.detach()) for k in kl)

    @torch.no_grad()
    def reconstruct(self, features=None):
        features = self.features if features is None else features
        local = [e(x) for e, x in zip(self.encoders, features)]
        if self.variational:
            local = [z.chunk(2, -1)[0] for z in local]
        joint = torch.cat(local, -1)
        return [d(joint) for d in self.decoders]
