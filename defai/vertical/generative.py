"""Small feature-partitioned GAN and diffusion research baselines.

Generated blocks are local to feature owners. Shared latent/time values and cut
embeddings link their outputs. No privacy, sample-quality or scaling claims.
"""
import torch
from torch import nn


def _validate_features(features):
    features = [torch.as_tensor(x, dtype=torch.float32) for x in features]
    if len(features) < 2 or any(x.ndim != 2 or not x.size(1) or not len(x) or not torch.isfinite(x).all()
                                or len(x) != len(features[0]) for x in features):
        raise ValueError('At least two aligned finite feature matrices are required')
    return features


def _boundary(values):
    return [x.detach().clone().requires_grad_(True) for x in values]


def _return_gradients(values, sent):
    torch.autograd.backward(values, [x.grad for x in sent])


class VerticalGAN:
    def __init__(self, features, *, latent=4, embedding=4, lr=.001):
        self.features = _validate_features(features)
        self.latent = latent
        self.generators = [nn.Sequential(nn.Linear(latent, 16), nn.Tanh(), nn.Linear(16, x.size(1))) for x in self.features]
        self.encoders = [nn.Sequential(nn.Linear(x.size(1), embedding), nn.LeakyReLU(.2)) for x in self.features]
        self.top = nn.Linear(embedding*len(features), 1)
        self.g_parameters = [p for g in self.generators for p in g.parameters()]
        self.d_parameters = [p for e in self.encoders+[self.top] for p in e.parameters()]
        self.g_optimizer = torch.optim.Adam(self.g_parameters, lr=lr)
        self.d_optimizer = torch.optim.Adam(self.d_parameters, lr=lr)

    def _score(self, chunks):
        values = [e(x) for e, x in zip(self.encoders, chunks)]
        sent = _boundary(values)
        return self.top(torch.cat(sent, -1)), values, sent

    def step(self, indices, *, generator=None):
        n = len(indices)
        if n == 0:
            raise ValueError('Empty batch')
        z = torch.randn((n, self.latent), generator=generator)
        fake = [g(z) for g in self.generators]
        self.d_optimizer.zero_grad(set_to_none=True)
        losses = []
        for chunks, target in (([x[indices] for x in self.features], 1.), ([x.detach() for x in fake], 0.)):
            score, local, sent = self._score(chunks)
            loss = nn.functional.binary_cross_entropy_with_logits(score, torch.full_like(score, target))
            loss.backward()
            _return_gradients(local, sent)
            losses.append(float(loss.detach()))
        self.d_optimizer.step()
        self.g_optimizer.zero_grad(set_to_none=True)
        for p in self.d_parameters:
            p.requires_grad_(False)
        try:
            score, local, sent = self._score(fake)
            loss = nn.functional.binary_cross_entropy_with_logits(score, torch.ones_like(score))
            loss.backward()
            _return_gradients(local, sent)
            self.g_optimizer.step()
        finally:
            for p in self.d_parameters:
                p.requires_grad_(True)
        return {'discriminator_loss': sum(losses), 'generator_loss': float(loss.detach())}

    @torch.no_grad()
    def sample(self, count, *, generator=None):
        z = torch.randn((count, self.latent), generator=generator)
        return [g(z) for g in self.generators]


class VerticalDiffusion:
    def __init__(self, features, *, steps=20, embedding=8, lr=.001):
        self.features = _validate_features(features)
        if not isinstance(steps, int) or steps < 2:
            raise ValueError('At least two diffusion timesteps required')
        self.beta = torch.linspace(.0001, .02, steps)
        self.alpha = 1-self.beta
        self.cumulative = self.alpha.cumprod(0)
        self.encoders = [nn.Sequential(nn.Linear(x.size(1)+1, embedding), nn.SiLU()) for x in self.features]
        self.decoders = [nn.Linear(embedding*len(features), x.size(1)) for x in self.features]
        self.parameters = [p for m in self.encoders+self.decoders for p in m.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

    def _local_embeddings(self, chunks, t):
        time = t.float().unsqueeze(-1) / len(self.beta)
        return [e(torch.cat((x, time), -1)) for e, x in zip(self.encoders, chunks)]

    def step(self, indices, *, generator=None):
        n = len(indices)
        if not n:
            raise ValueError('Empty batch')
        t = torch.randint(len(self.beta), (n,), generator=generator)
        noises = [torch.randn(x[indices].shape, generator=generator) for x in self.features]
        alpha = self.cumulative[t].unsqueeze(-1)
        noisy = [alpha.sqrt()*x[indices] + (1-alpha).sqrt()*noise for x, noise in zip(self.features, noises)]
        self.optimizer.zero_grad(set_to_none=True)
        local = self._local_embeddings(noisy, t)
        sent = _boundary(local)
        joint = torch.cat(sent, -1)
        returned = torch.zeros_like(joint)
        total = 0.
        for decoder, noise in zip(self.decoders, noises):
            received = joint.detach().clone().requires_grad_(True)
            loss = nn.functional.mse_loss(decoder(received), noise)
            loss.backward()
            returned += received.grad
            total += float(loss.detach())
        joint.backward(returned)
        _return_gradients(local, sent)
        self.optimizer.step()
        return total

    @torch.no_grad()
    def sample(self, count, *, generator=None):
        chunks = [torch.randn((count, x.size(1)), generator=generator) for x in self.features]
        for step in reversed(range(len(self.beta))):
            t = torch.full((count,), step)
            joint = torch.cat(self._local_embeddings(chunks, t), -1)
            predicted = [d(joint) for d in self.decoders]
            variance = (self.beta[step]*(1-self.cumulative[step-1])/(1-self.cumulative[step])) if step else 0.
            chunks = [(x-self.beta[step]*noise/(1-self.cumulative[step]).sqrt())/self.alpha[step].sqrt()
                      + (variance.sqrt()*torch.randn(x.shape, generator=generator) if step else 0.)
                      for x, noise in zip(chunks, predicted)]
        return chunks
