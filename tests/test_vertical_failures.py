import copy
import numpy as np
import pytest
import torch
from torch import nn
from defai.vertical.alignment import align_parties, align_time_series
from defai.vertical.split import FeatureParty, SplitTrainer
from defai.vertical.protection import clip_and_noise
from defai.vertical.paillier import keygen
from defai.vertical.checkpoint import save_checkpoint, load_checkpoint


def trainer(top=None):
    return SplitTrainer([FeatureParty(str(i), nn.Linear(2, 3), torch.ones(4, 2)) for i in range(2)],
                        nn.Linear(6, 2) if top is None else top, torch.zeros(4, dtype=torch.long))


def test_nonfinite_top_aborts_pending_batches_before_optimizer_updates():
    m = trainer()
    with torch.no_grad():
        m.top.weight.fill_(float('nan'))
    before = [copy.deepcopy(p.model.state_dict()) for p in m.parties]
    with pytest.raises(FloatingPointError):
        m.step([0, 1])
    assert all(p._pending is None for p in m.parties)
    assert m.steps == 0
    for party, state in zip(m.parties, before):
        for name, parameter in party.model.state_dict().items():
            torch.testing.assert_close(parameter, state[name])


def test_embedding_clipping_and_gradient_defense_both_run_at_boundary():
    m = trainer()
    for p in m.parties:
        p.embedding_transform = lambda value: clip_and_noise(value, bound=.1)
    m.defense = lambda value: clip_and_noise(value, bound=.2)
    assert m.step([0, 1]) > 0


@pytest.mark.parametrize('grid', [[1, 0], [0, 0], [0, np.nan], []])
def test_bad_shared_time_grids_rejected(grid):
    with pytest.raises(ValueError):
        align_time_series({'a': ([0, 1], [[1], [2]]), 'b': ([0, 1], [[3], [4]])}, grid)


def test_partial_missing_identifiers_are_intersected_but_empty_intersection_rejected():
    result = align_parties({'a': (['a', 'b', 'c'], [[1], [2], [3]]), 'b': (['b', 'c', 'd'], [[2], [3], [4]])})
    assert result.size == 2
    assert result.labels is None


def test_default_cryptographic_key_size_is_exercised():
    public, private = keygen()
    assert public.n.bit_length() == 2048
    assert private.decrypt(public.encrypt(-.125)) == -.125


def test_checkpoint_nonfinite_weights_do_not_modify_trainer(tmp_path):
    a, b = trainer(), trainer()
    path = tmp_path/'nan.pt'
    save_checkpoint(a, path)
    state = torch.load(path, weights_only=True)
    state['bottoms'][0]['weight'][0, 0] = float('nan')
    torch.save(state, path)
    original = b.parties[0].model.weight.detach().clone()
    with pytest.raises(ValueError, match='non-finite'):
        load_checkpoint(b, path)
    assert torch.equal(original, b.parties[0].model.weight)
