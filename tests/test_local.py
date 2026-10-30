"""Local-training hooks in defai.task, checked against hand-computed SGD."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from defai.task import MoonReference, Net, moon_contrastive_loss, scaffold_control_update, train

DEVICE = torch.device("cpu")


def batches(n_batches=3, batch_size=4, dim=3, classes=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        {"x": torch.randn(batch_size, dim, generator=g), "y": torch.randint(0, classes, (batch_size,), generator=g)}
        for _ in range(n_batches)
    ]


def linear(seed=0, dim=3, classes=2):
    torch.manual_seed(seed)
    return nn.Linear(dim, classes)


def manual_sgd(model, data, lr, extra_grad=None, flip=None, prox=None):
    """Reference implementation of the same local loop, written with explicit autograd."""
    model = copy.deepcopy(model)
    anchor = [p.detach().clone() for p in model.parameters()]
    for b in data:
        y = b["y"] if flip is None else flip - 1 - b["y"]
        loss = F.cross_entropy(model(b["x"]), y)
        if prox:
            loss = loss + 0.5 * prox * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
        grads = torch.autograd.grad(loss, list(model.parameters()))
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.parameters(), grads)):
                p -= lr * (g + (extra_grad[i] if extra_grad is not None else 0))
    return model


def params(model):
    return [p.detach().clone() for p in model.parameters()]


def assert_same(a, b, atol=1e-6):
    for x, y in zip(params(a), params(b)):
        torch.testing.assert_close(x, y, atol=atol, rtol=0)


def test_plain_training_matches_manual_sgd_and_reports_steps():
    data, model = batches(), linear()
    expected = manual_sgd(model, data, 0.1)
    loss, steps = train(model, data, epochs=1, lr=0.1, device=DEVICE)
    assert steps == 3 and np.isfinite(loss)
    assert_same(model, expected)


def test_epochs_multiply_steps():
    _, steps = train(linear(), batches(), epochs=4, lr=0.1, device=DEVICE)
    assert steps == 12


def test_grad_correction_is_added_to_every_gradient():
    data, model = batches(), linear()
    correction = [torch.full_like(p, 0.5) for p in model.parameters()]
    expected = manual_sgd(model, data, 0.1, extra_grad=correction)
    train(model, data, epochs=1, lr=0.1, device=DEVICE, grad_correction=correction)
    assert_same(model, expected)


def test_fedprox_term_matches_manual_gradient():
    data, model = batches(), linear()
    expected = manual_sgd(model, data, 0.1, prox=2.0)
    train(model, data, epochs=1, lr=0.1, device=DEVICE, proximal_mu=2.0)
    assert_same(model, expected)
    # and it really pulls towards the start: a big mu keeps the model closer to it
    start = params(linear())
    far, near = linear(), linear()
    train(far, data, 3, 0.1, DEVICE)
    train(near, data, 3, 0.1, DEVICE, proximal_mu=5.0)  # lr * mu < 2 keeps the prox term stable
    dist = lambda m: sum(((p - s) ** 2).sum() for p, s in zip(params(m), start)).item()  # noqa: E731
    assert dist(near) < dist(far)


def test_reported_loss_is_the_task_loss_without_regularisers():
    data = batches(1)
    plain_loss, _ = train(linear(), data, 1, 0.1, DEVICE)
    prox_loss, _ = train(linear(), data, 1, 0.1, DEVICE, proximal_mu=100.0)
    assert prox_loss == pytest.approx(plain_loss)  # first step: prox term is 0 and loss is pre-update


def test_label_flip_trains_on_flipped_labels():
    data, model = batches(), linear()
    expected = manual_sgd(model, data, 0.1, flip=2)
    train(model, data, 1, 0.1, DEVICE, flip_num_classes=2)
    assert_same(model, expected)
    other = linear()
    train(other, data, 1, 0.1, DEVICE)
    assert any(not torch.allclose(a, b) for a, b in zip(params(model), params(other)))


# ------------------------------------------------------------------ SCAFFOLD maths
def test_control_variate_equals_the_local_gradient_when_it_is_constant():
    """Option II: with a constant gradient g, c_i^+ = g regardless of c and c_i."""
    rng = np.random.default_rng(0)
    g, c, c_i = rng.normal(size=5), rng.normal(size=5), rng.normal(size=5)
    x, lr, steps = rng.normal(size=5), 0.1, 7
    y = x - lr * steps * (g - c_i + c)  # corrected SGD with a constant gradient
    out = scaffold_control_update({"w": c_i}, {"w": c}, {"w": x}, {"w": y}, steps, lr)
    np.testing.assert_allclose(out["w"], g)


def test_control_variate_is_unchanged_without_steps():
    out = scaffold_control_update({"w": np.ones(2)}, {"w": np.zeros(2)}, {"w": np.zeros(2)}, {"w": np.ones(2)}, 0, 0.1)
    np.testing.assert_array_equal(out["w"], np.ones(2))


def test_scaffold_corrected_training_recovers_the_control_variate_from_a_real_run():
    """Full-batch SGD on a linear model: after training, c_i^+ is the average corrected gradient."""
    data, model = batches(1), linear()
    start = [p.detach().clone().numpy() for p in model.parameters()]
    names = [n for n, _ in model.named_parameters()]
    c = [torch.randn_like(p) for p in model.parameters()]
    c_i = [torch.randn_like(p) for p in model.parameters()]
    lr = 0.1
    train(model, data, 1, lr, DEVICE, grad_correction=[a - b for a, b in zip(c, c_i)])
    y = {n: p.detach().numpy() for n, p in model.named_parameters()}
    new = scaffold_control_update(
        {n: t.numpy() for n, t in zip(names, c_i)}, {n: t.numpy() for n, t in zip(names, c)},
        dict(zip(names, start)), y, steps=1, lr=lr,
    )
    # one step: (x - y)/lr = g - c_i + c  ->  c_i^+ = g, the plain gradient at the start point
    reference = linear()
    loss = F.cross_entropy(reference(data[0]["x"]), data[0]["y"])
    grads = torch.autograd.grad(loss, list(reference.parameters()))
    for n, g in zip(names, grads):
        np.testing.assert_allclose(new[n], g.numpy(), atol=1e-5)


# ------------------------------------------------------------------------------- MOON
def test_moon_loss_matches_the_formula():
    z, zg, zp = torch.randn(5, 8), torch.randn(5, 8), torch.randn(5, 8)
    tau = 0.5
    pos = F.cosine_similarity(z, zg, dim=1) / tau
    neg = F.cosine_similarity(z, zp, dim=1) / tau
    expected = (-torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg)))).mean()
    torch.testing.assert_close(moon_contrastive_loss(z, zg, zp, tau), expected)


def test_moon_loss_prefers_the_global_over_the_previous_representation():
    z = torch.randn(6, 8)
    close = moon_contrastive_loss(z, z.clone(), -z, 0.5)    # matches global, opposes previous
    far = moon_contrastive_loss(z, -z, z.clone(), 0.5)      # the other way round
    equal = moon_contrastive_loss(z, z.clone(), z.clone(), 0.5)
    assert close < equal < far
    assert equal.item() == pytest.approx(np.log(2.0))


def test_moon_with_previous_equal_to_global_is_plain_training():
    """First participation: z_glob == z_prev, the contrastive gradient cancels exactly."""
    torch.manual_seed(0)
    data = [{"x": torch.randn(4, 1, 28, 28), "y": torch.randint(0, 10, (4,))} for _ in range(2)]
    torch.manual_seed(1)
    base = Net()
    plain, moon_model = copy.deepcopy(base), copy.deepcopy(base)
    frozen = copy.deepcopy(base).eval()
    train(plain, data, 1, 0.05, DEVICE)
    ref = MoonReference(frozen, frozen, mu=5.0, temperature=0.5)
    train(moon_model, data, 1, 0.05, DEVICE, moon=ref)
    assert_same(plain, moon_model, atol=1e-5)


def test_moon_with_a_different_previous_model_changes_training():
    torch.manual_seed(0)
    data = [{"x": torch.randn(4, 1, 28, 28), "y": torch.randint(0, 10, (4,))} for _ in range(2)]
    torch.manual_seed(1)
    base = Net()
    previous = copy.deepcopy(base)
    with torch.no_grad():
        for p in previous.parameters():
            p.add_(0.05 * torch.randn_like(p))
    plain, moon_model = copy.deepcopy(base), copy.deepcopy(base)
    train(plain, data, 1, 0.05, DEVICE)
    ref = MoonReference(copy.deepcopy(base).eval(), previous.eval(), mu=5.0, temperature=0.5)
    train(moon_model, data, 1, 0.05, DEVICE, moon=ref)
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(params(plain), params(moon_model)))


def test_net_head_of_representation_is_forward():
    net = Net().eval()
    x = torch.randn(3, 1, 28, 28)
    torch.testing.assert_close(net.head(net.represent(x)), net(x))


def test_cifar10_model_has_the_right_shapes_for_training_and_moon():
    from defai.task import build_model

    net = build_model("cifar10").eval()
    x = torch.randn(2, 3, 32, 32)
    assert net(x).shape == (2, 10)
    assert net.represent(x).shape == (2, 512)
    net.train()
    loss = F.cross_entropy(net(x), torch.tensor([1, 2]))
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())
