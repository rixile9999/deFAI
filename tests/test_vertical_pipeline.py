import json
import numpy as np
import torch
import pytest
from defai.vertical.alignment import AlignedData
from defai.vertical.pipeline import read_manifest, run_tabular
from defai.vertical.boost import HistogramParty


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('method', ['split', 'fedbcd-p', 'fedbcd-s', 'neural-bcd-p', 'neural-bcd-s', 'secureboost'])
def test_heldout_pipeline_runs_every_core_method(method):
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(40, 2)), rng.normal(size=(40, 3))
    y = (a[:, 0] + b[:, 0] > 0).astype(int)
    data = AlignedData({'a': a, 'b': b}, y, 'a', 'plain')
    report = run_tabular(data, method=method, epochs=4, batch_size=16, key_bits=256, insecure_test_key=True)
    assert report['train_rows'] == 30 and report['test_rows'] == 10
    assert np.isfinite(report['log_loss']) and 0 <= report['accuracy'] <= 1
    assert all(v['fitted_on'] == 'train_only' for v in report['normalization'].values())
    assert not {'labels', 'features', 'ids'} & report.keys()


def test_manifest_aligns_reordered_csv_and_rejects_label_as_feature(tmp_path):
    (tmp_path/'a.csv').write_text('id,x,y\nb,2,1\na,1,0\n', encoding='utf-8')
    (tmp_path/'b.csv').write_text('id,z\na,10\nb,20\n', encoding='utf-8')
    cfg = {'parties': [{'name': 'a', 'path': 'a.csv', 'id_column': 'id', 'features': ['x']},
                       {'name': 'b', 'path': 'b.csv', 'id_column': 'id', 'features': ['z']}],
           'label_owner': 'a', 'label_column': 'y'}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    data, receipts = read_manifest(path)
    np.testing.assert_array_equal(data.features['b'].ravel(), [10, 20])
    assert len(receipts) == 2
    cfg['parties'][0]['features'] = ['y']
    path.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='exclude'):
        read_manifest(path)


def test_tree_inference_preserves_training_thresholds():
    party = HistogramParty('a', [[0], [2], [10]])
    view = party.with_features([[1], [1.5], [100]])
    assert view._lookup == party._lookup
    np.testing.assert_array_equal(view.route(0, [0, 1, 2]), [0])


def test_test_labels_do_not_change_training_history():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(32, 2)), (rng.random(32) > .5).astype(int)
    changed = y.copy()
    test = np.random.default_rng(42).permutation(32)[:8]
    changed[test] = 1 - changed[test]
    a = AlignedData({'a': x, 'b': x}, y, 'a', 'plain')
    b = AlignedData({'a': x, 'b': x}, changed, 'a', 'plain')
    first = run_tabular(a, method='fedbcd-p', epochs=4)
    second = run_tabular(b, method='fedbcd-p', epochs=4)
    assert first['loss_history'] == second['loss_history']
