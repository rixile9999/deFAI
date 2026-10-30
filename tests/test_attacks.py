import numpy as np
import pytest

from defai.attacks import flip_labels, is_malicious, poison_update


def test_malicious_clients_are_the_first_partitions():
    assert [is_malicious(i, 2) for i in range(4)] == [True, True, False, False]
    assert not is_malicious(0, 0)


def test_flip_labels_is_an_involution():
    y = np.arange(10)
    np.testing.assert_array_equal(flip_labels(y, 10), y[::-1])
    np.testing.assert_array_equal(flip_labels(flip_labels(y, 10), 10), y)


def arrays(v):
    return {"w": np.array(v, dtype=np.float32)}


def test_sign_flip_reverses_and_scales_the_update():
    out = poison_update("sign-flip", arrays([1.0, 1.0]), arrays([2.0, 0.0]), 3.0, np.random.default_rng(0))
    np.testing.assert_allclose(out["w"], [-2.0, 4.0])  # g - 3 * (l - g)
    assert out["w"].dtype == np.float32


def test_gaussian_replaces_update_with_noise_around_global():
    g = {"w": np.zeros(100_000, dtype=np.float32)}
    out = poison_update("gaussian", g, {"w": np.ones(100_000, dtype=np.float32)}, 0.5, np.random.default_rng(0))
    assert out["w"].std() == pytest.approx(0.5, rel=0.02) and abs(out["w"].mean()) < 0.01


@pytest.mark.parametrize("attack", ["none", "label-flip"])
def test_data_poisoning_and_none_leave_update_unchanged(attack):
    local = arrays([1.0, 2.0])
    assert poison_update(attack, arrays([0.0, 0.0]), local, 1.0, np.random.default_rng(0)) is local
