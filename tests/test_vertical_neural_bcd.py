import copy
import torch
from torch import nn
import pytest
from defai.vertical.split import FeatureParty, SplitTrainer
from defai.vertical.neural_bcd import NeuralFedBCD


def make(q, mode='parallel'):
    torch.manual_seed(3)
    parties = [FeatureParty(str(i), nn.Sequential(nn.Linear(2, 4), nn.Tanh()), torch.randn(12, 2), lr=.05) for i in range(2)]
    return NeuralFedBCD(parties, nn.Linear(8, 2), torch.zeros(12, dtype=torch.long),
                        label_owner='1', q=q, mode=mode, lr=.05)


def test_neural_q_one_is_identical_to_splitnn():
    a = make(1)
    b = SplitTrainer([FeatureParty(p.name, copy.deepcopy(p.model), p.features.clone(), lr=.05) for p in a.parties],
                     copy.deepcopy(a.top), a.labels.clone(), lr=.05)
    assert a.step([0, 1, 2]) == pytest.approx(b.step([0, 1, 2]), abs=1e-8)
    for x, y in zip(a.top.parameters(), b.top.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['parallel', 'sequential'])
def test_neural_multiple_local_steps_reuse_batch_and_learn(mode):
    m = make(3, mode)
    before = float(m.loss(m.predict([0, 1, 2]), m.labels[:3]))
    for _ in range(4):
        m.step([0, 1, 2])
    after = float(m.loss(m.predict([0, 1, 2]), m.labels[:3]))
    assert after < before
    assert m.steps == 4 and m.local_steps == 12 and m.traffic.exchanges == 4
    assert all(p._pending is None for p in m.parties)


def test_neural_failure_reverts_all_blocks_and_traffic():
    m = make(3)
    before = copy.deepcopy(m.parties[0].model.state_dict())
    m.defense = lambda gradient: gradient*float('nan')
    with pytest.raises(ValueError):
        m.step([0, 1, 2])
    for name, parameter in m.parties[0].model.state_dict().items():
        torch.testing.assert_close(parameter, before[name])
    assert m.steps == m.traffic.exchanges == m.local_steps == 0
