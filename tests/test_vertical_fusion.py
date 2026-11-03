import torch
from torch import nn
from defai.vertical.fusion import SharedSumHead
from defai.vertical.models import GraphEncoder
from defai.vertical.split import FeatureParty, SplitTrainer


def test_shared_sum_forward_and_straight_through_gradient():
    a, b = torch.randn(4, 3, requires_grad=True), torch.randn(4, 3, requires_grad=True)
    layer = SharedSumHead(nn.Identity(), 2, 3)
    output = layer(torch.cat([a, b], -1))
    torch.testing.assert_close(output, a+b, atol=2e-6, rtol=0)
    output.sum().backward()
    torch.testing.assert_close(a.grad, torch.ones_like(a))
    torch.testing.assert_close(b.grad, torch.ones_like(b))


def test_local_graph_encoders_train_through_shared_sum_fusion():
    torch.manual_seed(2)
    parties = [FeatureParty(str(i), GraphEncoder(2, torch.eye(6), 4), torch.randn(6, 2)) for i in range(2)]
    head = SharedSumHead(nn.Linear(4, 2), 2, 4)
    trainer = SplitTrainer(parties, head, torch.zeros(6, dtype=torch.long))
    before = parties[0].model.first.weight.detach().clone()
    assert trainer.step([0, 1, 2]) > 0
    assert not torch.equal(before, parties[0].model.first.weight)
