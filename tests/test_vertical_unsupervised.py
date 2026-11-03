import torch
import pytest
from defai.vertical.unsupervised import VerticalAutoencoder


@pytest.mark.parametrize('variational', [False, True])
def test_unlabeled_reconstruction_learns_and_keeps_targets_local(variational):
    torch.manual_seed(4)
    x = torch.randn(30, 2)
    features = [x, x[:, :1]*2]
    model = VerticalAutoencoder(features, latent=3, variational=variational, lr=.03)
    generator = torch.Generator().manual_seed(1)
    losses = [model.step(list(range(30)), generator=generator) for _ in range(60)]
    assert sum(losses[-5:]) < sum(losses[:5])*.4
    assert not hasattr(model, 'labels')
    for prediction, target in zip(model.reconstruct(), features):
        assert prediction.shape == target.shape
    assert len(model.last_local_losses) == 2
