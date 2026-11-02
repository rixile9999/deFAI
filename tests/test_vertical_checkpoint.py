import torch
import pytest
from defai.vertical.split import FeatureParty, SplitTrainer
from defai.vertical.checkpoint import save_checkpoint, load_checkpoint


def trainer():
    return SplitTrainer([FeatureParty(str(i), torch.nn.Linear(2, 3), torch.ones(6, 2)) for i in range(2)],
                        torch.nn.Linear(6, 2), torch.zeros(6, dtype=torch.long))


def test_checkpoint_resumes_identically_and_contains_no_raw_data(tmp_path):
    a, b = trainer(), trainer()
    a.step([0, 1, 2])
    path = tmp_path/'model.pt'
    save_checkpoint(a, path)
    saved = torch.load(path, weights_only=True)
    assert not {'labels', 'features', 'ids'} & set(saved)
    load_checkpoint(b, path)
    assert a.steps == b.steps == 1
    assert a.step([2, 3, 4]) == pytest.approx(b.step([2, 3, 4]), abs=1e-8)
    torch.testing.assert_close(a.predict([0, 1]), b.predict([0, 1]))
    with pytest.raises(FileExistsError):
        save_checkpoint(a, path)


def test_wrong_shape_does_not_partially_restore_live_parameters(tmp_path):
    a, b = trainer(), trainer()
    path = tmp_path/'bad.pt'
    save_checkpoint(a, path)
    state = torch.load(path, weights_only=True)
    state['top']['weight'] = torch.zeros(8, 8)
    torch.save(state, path)
    before = b.parties[0].model.weight.detach().clone()
    with pytest.raises(RuntimeError):
        load_checkpoint(b, path)
    assert torch.equal(before, b.parties[0].model.weight)


def test_graph_topology_is_not_serialized_as_model_parameters(tmp_path):
    from defai.vertical.models import GraphEncoder
    m = SplitTrainer([FeatureParty(str(i), GraphEncoder(2, torch.eye(6), 3), torch.ones(6, 2)) for i in range(2)],
                     torch.nn.Linear(6, 2), torch.zeros(6, dtype=torch.long))
    path = tmp_path/'graph.pt'
    save_checkpoint(m, path)
    state = torch.load(path, weights_only=True)
    assert all('adjacency' not in local for local in state['bottoms'])


def test_invalid_optimizer_rate_is_rejected_before_restoring(tmp_path):
    a, b = trainer(), trainer()
    path = tmp_path/'optimizer.pt'
    save_checkpoint(a, path)
    state = torch.load(path, weights_only=True)
    state['optimizer']['param_groups'][0]['lr'] = float('nan')
    torch.save(state, path)
    with pytest.raises(ValueError, match='learning rate'):
        load_checkpoint(b, path)
