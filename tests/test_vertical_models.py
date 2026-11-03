import torch
from torch import nn
import pytest
from defai.vertical.models import (mlp, ImageEncoder, RecurrentEncoder, TemporalEncoder,
                                   TransformerEncoder, FrozenEncoder, GraphEncoder)
from defai.vertical.split import FeatureParty, SplitTrainer


@pytest.mark.parametrize('kind', ['mlp', 'cnn', 'lstm', 'gru', 'tcn', 'transformer', 'tokens', 'frozen', 'gnn'])
def test_encoder_can_train_across_the_split_boundary(kind):
    torch.manual_seed(2)
    if kind == 'mlp':
        x, factory = torch.randn(6, 3), lambda: mlp(3, 4)
    elif kind == 'cnn':
        x, factory = torch.randn(6, 1, 8, 8), lambda: ImageEncoder(1, 4)
    elif kind in {'lstm', 'gru'}:
        x, factory = torch.randn(6, 5, 3), lambda: RecurrentEncoder(3, 4, cell=kind)
    elif kind == 'tcn':
        x, factory = torch.randn(6, 5, 3), lambda: TemporalEncoder(3, 4)
    elif kind in {'transformer', 'tokens'}:
        x, factory = torch.randn(6, 5, 3), lambda: TransformerEncoder(3, 4, tokens=kind == 'tokens')
    elif kind == 'frozen':
        x, factory = torch.randn(6, 3), lambda: FrozenEncoder(nn.Linear(3, 4))
    else:
        x, factory = torch.randn(6, 3), lambda: GraphEncoder(3, torch.eye(6), 4)
    parties = [FeatureParty(str(i), factory(), x.clone()) for i in range(2)]
    y = torch.zeros((6, 5) if kind == 'tokens' else (6,), dtype=torch.long)
    trainer = SplitTrainer(parties, nn.Linear(8, 2), y)
    before = trainer.top.weight.detach().clone()
    assert trainer.step([0, 1, 2]) > 0
    assert not torch.equal(before, trainer.top.weight)
    assert trainer.predict([0, 1]).shape == ((2, 5, 2) if kind == 'tokens' else (2, 2))


@pytest.mark.parametrize('factory', [lambda: TemporalEncoder(2, 4, sequence=True),
                                     lambda: RecurrentEncoder(2, 4, sequence=True)])
def test_causal_sequence_outputs_do_not_see_future_inputs(factory):
    torch.manual_seed(1)
    model = factory().eval()
    x = torch.randn(3, 12, 2)
    changed = x.clone()
    changed[:, 7:] += 100
    torch.testing.assert_close(model(x)[:, :7], model(changed)[:, :7], rtol=0, atol=0)


def test_frozen_encoder_is_not_updated_and_remains_in_eval_mode():
    model = FrozenEncoder(nn.Sequential(nn.Linear(3, 4), nn.Dropout(.9)))
    model.train()
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.parameters())
