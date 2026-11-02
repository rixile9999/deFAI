"""Boundary perturbations and leakage diagnostics, NOT a privacy guarantee.

No epsilon/delta claim is made without a matching sensitivity/accountant analysis.
Marvell uses a fitted binary-class Gaussian covariance approximation.
"""
import math
import numpy as np
import torch


def clip_and_noise(gradient, *, bound=1., noise_std=0., generator=None):
    if any(not math.isfinite(v) for v in (bound, noise_std)) or bound <= 0 or noise_std < 0:
        raise ValueError('Positive bound and nonnegative finite noise required')
    flat = gradient.flatten(1)
    clipped = flat * (bound / flat.norm(dim=1, keepdim=True).clamp_min(bound))
    noise = torch.randn(flat.shape, dtype=flat.dtype, device=flat.device, generator=generator) * noise_std
    return (clipped + noise).reshape_as(gradient)


def max_norm(gradient, *, generator=None):
    """Random norm equalization: E[||g+noise||²]=max_i ||g_i||² for nonzero rows.

    Direction remains observable; zero rows remain zero. Not a label-DP mechanism.
    """
    flat = gradient.flatten(1)
    squared = flat.square().sum(1, keepdim=True)
    variance = (squared.max() / squared.clamp_min(torch.finfo(flat.dtype).eps) - 1).clamp_min(0)
    scale = 1 + torch.randn((len(flat), 1), device=flat.device, dtype=flat.dtype, generator=generator) * variance.sqrt()
    return (flat * scale).reshape_as(gradient)


def binary_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary labels and finite scores')
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        raise ValueError('Both label classes are required for AUC')
    from scipy.stats import rankdata
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def marvell(gradient, labels, *, power=1., generator=None):
    """Fit structured Gaussian noise under a mean-squared power constraint.

    Active label holder only. Returns perturbed gradient and optimizer diagnostics;
    no raw labels or diagnostics need be transmitted to feature parties.
    """
    from scipy.optimize import minimize
    flat = gradient.detach().flatten(1)
    y = torch.as_tensor(labels, device=flat.device)
    if y.shape != (len(flat),) or not torch.isin(y, torch.tensor([0, 1], device=y.device)).all():
        raise ValueError('Marvell requires one binary label per gradient row')
    if not math.isfinite(power) or power <= 0 or flat.shape[1] < 2:
        raise ValueError('Positive power and at least two gradient dimensions required')
    groups = [flat[y == i] for i in (0, 1)]
    if any(len(g) < 2 for g in groups):
        raise ValueError('At least two rows of each class are needed to estimate covariance')
    delta = groups[1].mean(0) - groups[0].mean(0)
    norm = delta.norm()
    direction = delta / norm.clamp_min(torch.finfo(flat.dtype).eps)
    if norm == 0:
        direction = torch.zeros_like(delta)
        direction[0] = 1
    d = flat.shape[1]
    variances = []
    for group in groups:
        centered = group - group.mean(0)
        along = (centered @ direction).square().mean()
        across = (centered.square().sum(1).mean() - along).clamp_min(0) / (d - 1)
        variances.extend([float(along), float(across)])
    base = np.maximum(variances, 1e-12)
    p = float((y == 1).float().mean())
    weights = np.array([1-p, (1-p)*(d-1), p, p*(d-1)])
    distance = float(norm.square())

    def objective(v):
        a, b, c, e = base + v
        return .5 * (a/c + c/a - 2 + (d-1)*(b/e + e/b - 2) + distance*(1/a + 1/c))

    result = minimize(objective, np.full(4, power / d), method='SLSQP', bounds=[(0, power/min(weights))]*4,
                      constraints=[{'type': 'ineq', 'fun': lambda v: power - weights @ v},
                                   {'type': 'ineq', 'fun': lambda v: np.array([v[0]-v[1], v[2]-v[3]])}],
                      options={'maxiter': 300, 'ftol': 1e-10})
    if not result.success or weights @ result.x > power + 1e-6:
        raise RuntimeError('Marvell covariance optimization failed: ' + result.message)
    noise = torch.empty_like(flat)
    for i in (0, 1):
        mask = y == i
        a, b = result.x[2*i:2*i+2]
        isotropic = torch.randn(groups[i].shape, dtype=flat.dtype, device=flat.device, generator=generator) * math.sqrt(max(0, b))
        axis = torch.randn((len(groups[i]), 1), dtype=flat.dtype, device=flat.device, generator=generator) * math.sqrt(max(0, a-b))
        noise[mask] = isotropic + axis * direction
    return (flat + noise).reshape_as(gradient), {'expected_noise_power': float(weights @ result.x),
                                               'gaussian_symmetric_kl': float(result.fun), 'formal_dp': False}
