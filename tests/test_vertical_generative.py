import torch
import pytest
from defai.vertical.generative import VerticalGAN, VerticalDiffusion


@pytest.mark.parametrize('factory', [VerticalGAN, VerticalDiffusion])
def test_joint_generation_trains_local_models_and_samples_every_block(factory):
    torch.manual_seed(5)
    features = [torch.randn(12, 2), torch.randn(12, 3)]
    model = factory(features)
    parameters = list(model.generators[0].parameters()) if factory is VerticalGAN else model.parameters
    before = parameters[0].detach().clone()
    result = model.step(list(range(8)), generator=torch.Generator().manual_seed(9))
    assert not torch.equal(parameters[0], before)
    if isinstance(result, dict):
        assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    else:
        assert torch.isfinite(torch.tensor(result))
    generated = model.sample(4, generator=torch.Generator().manual_seed(1))
    assert [x.shape for x in generated] == [(4, 2), (4, 3)]
    assert all(torch.isfinite(x).all() for x in generated)
