from collections import OrderedDict
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federated_algorithms import (bn_keys, train_mode, proximal_penalty,
                                  scaffold_correct, scaffold_control, personalized_state)
from run_federated_study import MultiModalNet, average_states


def test_proximal_gradient_is_mu_times_distance_not_parameter_mean():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[3., 4.]]))
    anchor = {"weight": torch.tensor([[1., 1.]])}
    loss = .1*proximal_penalty(model, anchor)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(.65))
    torch.testing.assert_close(model.weight.grad, torch.tensor([[.2, .3]]))


def test_scaffold_option_two_and_gradient_sign_on_known_steps():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.)
    model.weight.grad = torch.tensor([[.5]])
    old_client = {"weight": torch.tensor([[.2]])}
    old_server = {"weight": torch.tensor([[.1]])}
    scaffold_correct(model, old_server, old_client)
    torch.testing.assert_close(model.weight.grad, torch.tensor([[.4]]))
    # Two constant corrected steps: 2 -> 1.96 -> 1.92.
    new = scaffold_control(old_client, old_server, {"weight": torch.tensor([[2.]])},
                           {"weight": torch.tensor([[1.92]])}, 2, .1)
    torch.testing.assert_close(new["weight"], torch.tensor([[.5]]))
    with pytest.raises(ValueError):
        scaffold_control(old_client, old_server, old_client, old_client, 0, .1)


def test_scaffold_global_control_is_mean_for_full_participation():
    client = [{"w": torch.tensor([v])} for v in [1., 3., 8.]]
    result = average_states(client, [1, 1, 1])
    torch.testing.assert_close(result["w"], torch.tensor([4.]))
    weighted = average_states(client, [174, 173, 172])
    torch.testing.assert_close(weighted["w"], torch.tensor([(174+173*3+172*8)/519]))


def test_fedbn_reactivates_statistics_and_retains_all_local_bn_keys():
    torch.set_num_threads(2)
    model = MultiModalNet(pretrained=False)
    keys = bn_keys(model)
    assert "cnn.bn1.weight" in keys and "cnn.bn1.running_var" in keys
    assert "cnn.bn1.num_batches_tracked" in keys
    assert "meta_mlp.1.weight" not in keys  # LayerNorm is shared.
    train_mode(model, True)
    before = model.cnn.bn1.running_mean.clone()
    model(torch.randn(2, 3, 64, 64)+3, torch.ones(2, 4))
    assert model.cnn.bn1.num_batches_tracked == 1
    assert not torch.equal(before, model.cnn.bn1.running_mean)
    local = {k: v.clone()+1 for k, v in model.state_dict().items() if k in keys}
    state = personalized_state(model.state_dict(), local)
    torch.testing.assert_close(state["cnn.bn1.weight"], local["cnn.bn1.weight"])
    torch.testing.assert_close(state["head_reg.weight"], model.head_reg.weight)
    train_mode(model, False)
    assert not model.cnn.bn1.training


def test_weighted_scaffold_converges_to_weighted_quadratic_solution():
    model = torch.nn.Linear(1, 1, bias=False)
    shared = {"weight": torch.zeros(1, 1)}
    controls = [{"weight": torch.zeros(1, 1)} for _ in range(2)]
    server = {"weight": torch.zeros(1, 1)}
    weights, targets, lr = [1, 3], [-.3, .6], .1
    for _ in range(60):
        states = []
        for i, target in enumerate(targets):
            model.load_state_dict(shared)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            for step in range(2):
                optimizer.zero_grad()
                ((model.weight-target).square().sum()/2).backward()
                scaffold_correct(model, server, controls[i]); optimizer.step()
            local = {"weight": model.weight.detach().clone()}
            controls[i] = scaffold_control(controls[i], server, shared, local, 2, lr)
            states.append(local)
        shared = average_states(states, weights)
        server = average_states(controls, weights)
    torch.testing.assert_close(shared["weight"], torch.tensor([[.375]]), atol=1e-5, rtol=1e-5)
