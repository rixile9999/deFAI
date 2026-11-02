import numpy as np
import pytest

from defai.kalman_data import prepare_public_data


@pytest.mark.parametrize("name,shape", [("digits", (1797, 64)), ("breast-cancer", (569, 30))])
def test_public_split_is_disjoint_complete_and_train_normalized(name, shape):
    d = prepare_public_data(name, seed=2, alpha=.1)
    assert d.x.shape == shape
    train, validation, test = map(set, (d.train, d.validation, d.test))
    assert not train & validation and not train & test and not validation & test
    assert train | validation | test == set(range(shape[0]))
    np.testing.assert_allclose(d.x[d.train].mean(axis=0), 0, atol=1e-10)
    for split, nodes in d.clients.items():
        flat = np.concatenate(nodes)
        assert all(len(node) > 0 for node in nodes)
        assert len(flat) == len(set(flat))
        assert set(flat) == set(getattr(d, split))
    assert not d.x.flags.writeable and not d.y.flags.writeable
    assert d.metadata["source"]["license"] == "CC BY 4.0"


def test_splits_are_reproducible_and_identical_across_heterogeneity_conditions():
    a = prepare_public_data("digits", seed=3, alpha=.1)
    b = prepare_public_data("digits", seed=3, alpha=10.)
    c = prepare_public_data("digits", seed=3, alpha=.1)
    assert a.metadata["split_sha256"] == b.metadata["split_sha256"] == c.metadata["split_sha256"]
    for x, y in zip(a.clients["train"], c.clients["train"]): np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kwargs", [{"name": "unknown"}, {"seed": True}, {"alpha": np.nan}, {"clients": 1}])
def test_invalid_public_data_settings(kwargs):
    options = dict(name="digits", seed=0, alpha=.5, clients=10)
    options.update(kwargs)
    with pytest.raises(ValueError): prepare_public_data(**options)
