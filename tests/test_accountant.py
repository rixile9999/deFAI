import math

import numpy as np
import pytest

from defai.privacy.accountant import DEFAULT_ORDERS, RdpAccountant


def gaussian_delta(eps, z):
    """Exact delta(eps) of the Gaussian mechanism with sensitivity 1, std z (Balle & Wang, 2018)."""
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    return phi(-eps * z + 1 / (2 * z)) - math.exp(eps) * phi(-eps * z - 1 / (2 * z))


def exact_gaussian_epsilon(z, delta):
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if gaussian_delta(mid, z) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def test_no_releases_means_zero_epsilon():
    assert RdpAccountant().get_epsilon(1e-5) == 0.0


def test_matches_closed_form_minimisation():
    z, rounds, delta = 1.3, 7, 1e-5
    acc = RdpAccountant()
    for _ in range(rounds):
        acc.compose(z)
    orders = np.array(DEFAULT_ORDERS)
    rdp = rounds * orders / (2 * z**2)
    expected = np.min(rdp + np.log((orders - 1) / orders) - (math.log(delta) + np.log(orders)) / (orders - 1))
    assert acc.get_epsilon(delta) == pytest.approx(expected)


def test_count_argument_equals_repeated_compose():
    a, b = RdpAccountant(), RdpAccountant()
    a.compose(2.0, count=5)
    for _ in range(5):
        b.compose(2.0)
    assert a.get_epsilon(1e-6) == pytest.approx(b.get_epsilon(1e-6))
    assert a.num_releases == b.num_releases == 5


def test_epsilon_grows_with_rounds_and_shrinks_with_noise():
    acc = RdpAccountant()
    eps = []
    for _ in range(5):
        acc.compose(1.0)
        eps.append(acc.get_epsilon(1e-5))
    assert eps == sorted(eps) and len(set(eps)) == 5
    noisy, quiet = RdpAccountant(), RdpAccountant()
    noisy.compose(4.0, 5)
    quiet.compose(1.0, 5)
    assert noisy.get_epsilon(1e-5) < quiet.get_epsilon(1e-5)


@pytest.mark.parametrize("z", [0.8, 1.0, 2.0, 5.0])
def test_is_a_valid_but_not_wasteful_upper_bound_for_one_release(z):
    """RDP-derived epsilon must be >= the exact single-release epsilon (soundness)."""
    delta = 1e-5
    acc = RdpAccountant()
    acc.compose(z)
    exact = exact_gaussian_epsilon(z, delta)
    got = acc.get_epsilon(delta)
    assert got >= exact - 1e-6
    assert got <= 1.35 * exact  # tight enough to be useful


def test_zero_noise_is_infinite():
    acc = RdpAccountant()
    acc.compose(0.0)
    assert math.isinf(acc.get_epsilon(1e-5))


def test_would_exceed_does_not_mutate_and_matches_compose():
    acc = RdpAccountant()
    acc.compose(1.0, 3)
    before = acc.get_epsilon(1e-5)
    limit = before + 0.01
    assert acc.would_exceed(1.0, 1e-5, limit)
    assert acc.get_epsilon(1e-5) == before
    assert not acc.would_exceed(1.0, 1e-5, 1e9)
    acc.compose(1.0)
    assert acc.get_epsilon(1e-5) > limit


def test_agrees_with_google_dp_accounting():
    pytest.importorskip("dp_accounting")
    from dp_accounting import dp_event
    from dp_accounting.rdp import RdpAccountant as Google

    for z, rounds in [(0.9, 1), (1.0, 10), (2.5, 50), (7.0, 200)]:
        google = Google()
        google.compose(dp_event.GaussianDpEvent(z), rounds)
        ours = RdpAccountant()
        ours.compose(z, rounds)
        assert ours.get_epsilon(1e-5) == pytest.approx(google.get_epsilon(1e-5), rel=1e-9), (z, rounds)
