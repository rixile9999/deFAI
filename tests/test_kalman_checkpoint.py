from copy import deepcopy

import numpy as np
import pytest

from defai.privacy.dp import FixedClipping
from defai.strategies import FedAdamKalman


def make(blend=1.):
    s = FedAdamKalman(kalman_blend=blend, server_learning_rate=.1)
    s._global = {"w": np.zeros(4)}
    return s


@pytest.mark.parametrize("blend", [0., .3, 1.])
def test_exact_optimizer_continuation_with_separately_restored_global_model(blend):
    full = make(blend)
    for rnd in range(1, 4):
        full._global = full.update_global(rnd, {"w": np.full(4, rnd * .1)})
    state = full.state_dict()
    resumed = make(blend)
    resumed._global = deepcopy(full._global)
    resumed.load_state_dict(state)
    # Neither exported nor loaded state aliases the running optimizer.
    state["v"]["w"][:] = 999
    for rnd in range(4, 10):
        d = {"w": np.full(4, (-1)**rnd * .1)}
        full._global = full.update_global(rnd, d)
        resumed._global = resumed.update_global(rnd, d)
        np.testing.assert_array_equal(full._global["w"], resumed._global["w"])


@pytest.mark.parametrize("field,value", [("version", 2), ("last_round", True), ("last_round", -1),
                                        ("schema", {}), ("config", {}), ("layers", {})])
def test_invalid_metadata_does_not_mutate_live_state(field, value):
    s = make()
    s.update_global(1, {"w": np.ones(4)})
    before = s.state_dict()
    bad = s.state_dict()
    bad[field] = value
    with pytest.raises(ValueError): s.load_state_dict(bad)
    assert s._last_round == 1
    np.testing.assert_array_equal(s._m["w"], before["m"]["w"])


@pytest.mark.parametrize("field", ["mean", "variance", "process_noise", "observation_noise", "gain", "innovation_mse"])
def test_nonfinite_filter_state_is_rejected(field):
    s = make()
    s.update_global(1, {"w": np.ones(4)})
    bad = s.state_dict()
    bad["layers"]["w"][field] = np.full(4, np.nan) if field == "mean" else np.nan
    with pytest.raises(ValueError): s.load_state_dict(bad)


def test_privacy_resume_cannot_silently_reset_accounting():
    s = make()
    state = s.state_dict()
    s.dp = FixedClipping(1., 1.)
    with pytest.raises(ValueError, match="accountant"): s.state_dict()
    with pytest.raises(ValueError, match="DP"): s.load_state_dict(state)


def test_initial_state_and_corrupt_moments():
    s = make()
    s.load_state_dict(s.state_dict())
    s.update_global(1, {"w": np.ones(4)})
    for group, value in (("v", -1.), ("m", 7.), ("v", np.inf)):
        bad = s.state_dict()
        bad[group]["w"][:] = value
        with pytest.raises(ValueError): s.load_state_dict(bad)
