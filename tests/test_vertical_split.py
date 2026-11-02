import copy
import torch
from torch import nn
import pytest
from defai.vertical.split import FeatureParty, SplitTrainer


def make_trainer():
    torch.manual_seed(12)
    xs = [torch.randn(12, 3), torch.randn(12, 2)]
    bottoms = [nn.Sequential(nn.Linear(3, 4), nn.Tanh()), nn.Linear(2, 3)]
    top = nn.Linear(7, 2)
    y = (xs[0][:, 0] + xs[1][:, 0] > 0).long()
    return SplitTrainer([FeatureParty(str(i), m, x, lr=.1) for i, (m, x) in enumerate(zip(bottoms, xs))], top, y, lr=.1)


def test_split_matches_centralized_sgd_parameter_for_parameter():
    trainer = make_trainer()
    models = [copy.deepcopy(p.model) for p in trainer.parties]
    top = copy.deepcopy(trainer.top)
    parameters = [p for m in models + [top] for p in m.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=.1)
    indices = torch.tensor([1, 3, 7, 8])
    for _ in range(3):
        optimizer.zero_grad()
        logits = top(torch.cat([m(p.features[indices]) for m, p in zip(models, trainer.parties)], -1))
        central_loss = nn.functional.cross_entropy(logits, trainer.labels[indices])
        central_loss.backward()
        optimizer.step()
        assert trainer.step(indices) == pytest.approx(float(central_loss), abs=1e-7)
        for expected, actual in zip(parameters, [p for party in trainer.parties for p in party.model.parameters()] + list(trainer.top.parameters())):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    assert trainer.traffic.exchanges == 3
    assert trainer.traffic.forward_bytes == trainer.traffic.backward_bytes == 3 * 4 * 7 * 4


def test_boundary_is_detached_and_rejects_gradient_replay():
    party = make_trainer().parties[0]
    leaf = party.embedding(torch.tensor([0, 1]), 5)
    assert leaf.is_leaf and leaf.grad_fn is None
    with pytest.raises(RuntimeError, match='Stale'):
        party.accept_gradient(torch.ones_like(leaf), 4)
    party.accept_gradient(torch.ones_like(leaf), 5)
    party.commit()
    with pytest.raises(RuntimeError, match='Stale'):
        party.accept_gradient(torch.ones_like(leaf), 5)


@pytest.mark.parametrize('indices', [[0, 0], [-1], [12], [], [1.5]])
def test_invalid_batches_rejected(indices):
    with pytest.raises(ValueError):
        make_trainer().step(indices)


def test_inference_restores_training_state_and_uses_all_parties():
    trainer = make_trainer()
    trainer.parties[0].model.eval()
    output = trainer.predict([0, 1])
    assert output.shape == (2, 2)
    assert not trainer.parties[0].model.training and trainer.top.training
