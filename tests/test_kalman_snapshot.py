import json

import numpy as np
import pytest

from defai.strategies import FedAdamKalman


@pytest.mark.parametrize("blend", [0., .3, 1.])
def test_numeric_snapshot_exactly_restores_model_and_optimizer(tmp_path, blend):
    original = FedAdamKalman(kalman_blend=blend)
    original._global = {"array": np.zeros(3), "bias": np.array(0.)}
    latest = original.update_global(1, {"array": np.ones(3), "bias": np.array(.2)})
    path = original.save_checkpoint(tmp_path/"snapshot with spaces.npz", model=latest)
    restored = FedAdamKalman(kalman_blend=blend)
    model = restored.load_checkpoint(path)
    for k in model: np.testing.assert_array_equal(model[k], latest[k])
    original._global = latest
    delta = {"array": np.full(3, -.1), "bias": np.array(.3)}
    a, b = original.update_global(2, delta), restored.update_global(2, delta)
    for k in a: np.testing.assert_array_equal(a[k], b[k])
    assert restored._last_round == 2


def test_snapshot_never_silently_overwrites_and_cleans_temporary_files(tmp_path):
    s = FedAdamKalman(); s._global = {"w": np.zeros(2)}
    path = s.save_checkpoint(tmp_path/"state.npz", model=s._global)
    before = path.read_bytes()
    with pytest.raises(FileExistsError): s.save_checkpoint(path, model=s._global)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".kalman-*"))
    s.save_checkpoint(path, model=s._global, overwrite=True)


@pytest.mark.parametrize("mutation", ["payload", "version", "config"])
def test_corrupt_snapshot_cannot_mutate_running_optimizer(tmp_path, mutation):
    s = FedAdamKalman(); s._global = {"w": np.zeros(2)}
    latest = s.update_global(1, {"w": np.ones(2)})
    path = s.save_checkpoint(tmp_path/"state.npz", model=latest)
    with np.load(path, allow_pickle=False) as archive: content = dict(archive)
    metadata = json.loads(str(content["metadata"]))
    if mutation == "payload": content["array_0"][0] += 1
    if mutation == "version": metadata["version"] = True
    if mutation == "config": metadata["optimizer"]["config"]["tau"] = .2
    content["metadata"] = np.array(json.dumps(metadata))
    np.savez(path, **content)
    before = s.state_dict(); global_before = s._global
    with pytest.raises(ValueError): s.load_checkpoint(path)
    assert s._global is global_before and s._last_round == before["last_round"]
    np.testing.assert_array_equal(s._m["w"], before["m"]["w"])


def test_object_arrays_are_not_unpickled(tmp_path):
    path = tmp_path/"objects.npz"
    np.savez(path, metadata=np.array({"arbitrary": "object"}, dtype=object))
    with pytest.raises(ValueError): FedAdamKalman().load_checkpoint(path)
