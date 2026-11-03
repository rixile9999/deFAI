import numpy as np
import pytest
from defai.vertical.sharing import split_shares, reconstruct, shared_sum


def test_random_shares_reconstruct_signed_values_and_sum():
    a = np.array([[1.25, -2.5], [0., .000001]])
    x, y = split_shares(a)
    z, _ = split_shares(a)
    assert not np.array_equal(x, z)
    np.testing.assert_allclose(reconstruct(x, y), a, atol=1e-6)
    np.testing.assert_allclose(shared_sum([a, -a, a*2]), a*2, atol=3e-6)


def test_mismatched_shares_and_overflow_rejected():
    with pytest.raises(ValueError):
        shared_sum([np.ones(2), np.ones(3)])
    with pytest.raises(OverflowError):
        shared_sum([np.ones(1)*1e12]*3)
