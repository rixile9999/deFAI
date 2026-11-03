"""Differentiable sum boundary backed by the two-helper sharing simulation.

Straight-through gradients differentiate the unquantized sum; forward arithmetic
uses 1e-6 fixed point. This is not a nonlinear secure-inference engine.
"""
import torch
from torch import nn
from .sharing import shared_sum


class _SharedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *values):
        if any(v.shape != values[0].shape or v.device != values[0].device or v.dtype != values[0].dtype for v in values):
            raise ValueError('Sum fusion requires equal embedding shapes, devices and dtypes')
        ctx.count = len(values)
        result = shared_sum([v.detach().cpu().numpy() for v in values])
        return torch.as_tensor(result, device=values[0].device, dtype=values[0].dtype)

    @staticmethod
    def backward(ctx, gradient):
        return tuple(gradient for _ in range(ctx.count))


class SharedSumHead(nn.Module):
    def __init__(self, head, parties, embedding):
        super().__init__()
        if parties < 2 or embedding < 1:
            raise ValueError('Need >=2 parties with positive equal embedding size')
        self.head, self.parties, self.embedding = head, parties, embedding

    def forward(self, concatenated):
        if concatenated.shape[-1] != self.parties*self.embedding:
            raise ValueError('Embedding schema differs from the sum head')
        values = concatenated.split(self.embedding, dim=-1)
        return self.head(_SharedSum.apply(*values))
