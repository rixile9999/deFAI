import numpy as np
import torch
import pytest
from defai.vertical.protection import binary_auc, clip_and_noise, max_norm, marvell


def test_clipping_is_per_sample_and_noise_reproducible_with_explicit_generator():
    g = torch.tensor([[3., 4.], [0., 0.]])
    clipped = clip_and_noise(g, bound=2)
    torch.testing.assert_close(clipped.norm(dim=1), torch.tensor([2., 0.]))
    a = clip_and_noise(g, noise_std=.1, generator=torch.Generator().manual_seed(1))
    b = clip_and_noise(g, noise_std=.1, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_max_norm_equalizes_expected_squared_norm_not_directions():
    g = torch.tensor([[1., 0.], [0., 2.]])
    generator = torch.Generator().manual_seed(4)
    samples = torch.stack([max_norm(g, generator=generator) for _ in range(5000)])
    torch.testing.assert_close(samples.square().sum(-1).mean(0), torch.tensor([4., 4.]), rtol=.08, atol=.1)


def test_attack_auc_handles_ties_and_perfect_separation():
    assert binary_auc([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    assert binary_auc([0, 0, 1, 1], [1, 1, 1, 1]) == .5
    with pytest.raises(ValueError):
        binary_auc([1, 1], [1, 2])


def test_marvell_respects_covariance_power_budget_and_is_finite():
    generator = torch.Generator().manual_seed(7)
    g = torch.randn(40, 4, generator=generator) * .2
    g[20:] += 1
    y = torch.tensor([0]*20+[1]*20)
    protected, report = marvell(g, y, power=1., generator=generator)
    assert protected.shape == g.shape and torch.isfinite(protected).all()
    assert report['expected_noise_power'] <= 1.000001
    assert report['formal_dp'] is False
    assert not np.array_equal(protected.numpy(), g.numpy())
