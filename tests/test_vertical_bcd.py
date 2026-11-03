import numpy as np
import pytest
from defai.vertical.bcd import LinearParty, FedBCD


def problem(q=1, mode='parallel', task='regression'):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 5))
    y = x @ np.array([1., -.5, .2, .7, -.3])
    if task == 'logistic':
        y = (y > 0).astype(float)
    model = FedBCD([LinearParty('a', x[:, :2]), LinearParty('b', x[:, 2:])], y,
                   label_owner='b', task=task, lr=.05, q=q, mode=mode)
    return model, x, y


def test_q_one_parallel_matches_centralized_gradient_steps():
    m, x, y = problem()
    w = np.zeros(5)
    batch = np.arange(15)
    for _ in range(5):
        w -= .05 * x[batch].T @ (x[batch] @ w - y[batch]) / len(batch)
        m.step(batch)
        np.testing.assert_allclose(np.concatenate([p.weight for p in m.parties]), w, atol=1e-15)


@pytest.mark.parametrize('mode', ['parallel', 'sequential'])
@pytest.mark.parametrize('task', ['regression', 'logistic'])
def test_bcd_reuses_one_batch_and_decreases_loss(mode, task):
    model, x, y = problem(q=3, mode=mode, task=task)
    initial = model.loss()
    for _ in range(15):
        model.step(np.arange(len(y)))
    assert model.loss() < initial * .8
    assert model.local_steps == 45 and model.rounds == 15
    assert all(r['indices'] == list(range(40)) and r['reused_steps'] == 3 for r in model.batch_history)


def test_q_three_has_one_third_exchanges_for_equal_local_step_budget():
    a, _, _ = problem(q=1)
    b, _, _ = problem(q=3)
    for _ in range(6):
        a.step(np.arange(40))
    for _ in range(2):
        b.step(np.arange(40))
    assert a.local_steps == b.local_steps == 6
    assert a.bytes == b.bytes * 3
