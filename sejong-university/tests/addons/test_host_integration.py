import json
from pathlib import Path
import numpy as np
import pytest

from defai_addons.api import compare_arrays, create_multimodal_model, predict_multimodal, save_multimodal_checkpoint, load_multimodal_checkpoint
from defai_addons.integration.history import read_history, render_histories


def host_file(tmp_path):
    path = tmp_path/'history.json'
    path.write_text(json.dumps({'config': {'strategy': 'fedavg'}, 'history': [
        {'round': 0, 'test_acc': 0.1, 'test_loss': 2.3},
        {'round': 1, 'test_acc': 0.6, 'test_loss': 1.1, 'eval_acc': 0.5,
         'comm-mb-up': 2.0, 'comm-mb-down': 2.0, 'comm-mb-total': 4.0}],
         'network_transport_measured': False}), encoding='utf-8')
    return path


def test_history_missing_metrics_and_read_only(tmp_path):
    source = host_file(tmp_path)
    original = source.read_bytes()
    report = read_history(source)
    assert report['records'][0]['cumulative_total_mib'] is None
    assert report['records'][1]['global_f1'] is None
    assert report['records'][1]['client_weighted_accuracy'] == 0.5
    render_histories([source], tmp_path/'view.html')
    assert '기록 없음' in (tmp_path/'view.html').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        render_histories([source], source)
    assert source.read_bytes() == original


@pytest.mark.parametrize('row', [{'round': 1, 'test_acc': 60}, {'round': -1}, {'round': 0, 'comm-mb-up': -1}])
def test_history_rejects_invalid_values(tmp_path, row):
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps({'config': {}, 'history': [row]}))
    with pytest.raises(ValueError):
        read_history(path)


def test_external_sites_use_real_comparison_engine():
    rng = np.random.RandomState(19)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    sites = [{'X': X[:20], 'y': y[:20], 'X_test': X[40:50], 'y_test': y[40:50]},
             {'X': X[20:40], 'y': y[20:40], 'X_test': X[50:], 'y_test': y[50:]}]
    before = X.copy()
    report = compare_arrays(sites, X[40:], y[40:], {'epochs': 1, 'batch_size': 8,
        'use_pca': False, 'hidden': [8], 'envs': ['ld_independent', 'fl_fedavg']})
    assert len(report['results']) == 2
    assert report['n_test'] == 20
    assert all(0 <= r['global']['f1'] <= 1 for r in report['results'])
    assert report['results'][1]['privacy']['raw_data_leaves_site'] is False
    np.testing.assert_array_equal(X, before)
    with pytest.raises(ValueError):
        compare_arrays([], X, y)


def test_checkpoint_prediction_roundtrip(tmp_path):
    import torch
    from defai_addons.multimodal.data import generate
    torch.set_num_threads(2)
    config = {'d': 16, 'd_joint': 8, 'enc_layers': 1, 'dec_layers': 1}
    model = create_multimodal_model(**config)
    batch = generate(3, seed=71)
    expected = predict_multimodal(model, batch)
    assert model.training
    checkpoint = tmp_path/'model.pt'
    save_multimodal_checkpoint(model, checkpoint, config)
    restored = predict_multimodal(load_multimodal_checkpoint(checkpoint), batch)
    torch.testing.assert_close(expected['probabilities'], restored['probabilities'])
    assert restored['representation'].shape == (3, 48)
    batch['mask'][:] = False
    with pytest.raises(ValueError):
        predict_multimodal(model, batch)
