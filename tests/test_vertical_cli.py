import json
import subprocess
import sys
import torch
import pytest
from defai.cli import main
from defai.vertical.demo import run_demo


def test_discovery_does_not_import_neural_runtime():
    code = "import sys; from defai.cli import main; main(['vertical','list']); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)


def test_cli_initializes_aligns_trains_and_preserves_outputs(tmp_path):
    assert main(['vertical', 'init', str(tmp_path/'data')]) == 0
    output = tmp_path/'result.json'
    assert main(['vertical', 'run', str(tmp_path/'data/manifest.json'), '--method', 'fedbcd-p',
                 '--epochs', '3', '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['train_rows'] == 48 and report['test_rows'] == 16
    assert main(['vertical', 'run', str(tmp_path/'data/manifest.json'), '--output', str(output)]) == 2
    assert main(['vertical', 'init', str(tmp_path/'data')]) == 2


def test_all_families_demo_records_limits_and_restores_global_state():
    before_rng = torch.random.get_rng_state()
    before_threads = torch.get_num_threads()
    report = run_demo(insecure_test_key=True)
    assert len(report['families']) == 10
    assert len(report['families']['rnn']['variants']) == 2
    assert len(report['families']['transformer']['variants']) == 2
    assert not report['formal_privacy_guarantee']
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert torch.get_num_threads() == before_threads


def test_neural_demo_uses_distinct_local_feature_views(monkeypatch):
    import defai.vertical.split as split
    original, observed = split.FeatureParty, []

    def capture(name, model, features, **kwargs):
        observed.append(features.clone())
        return original(name, model, features, **kwargs)

    monkeypatch.setattr(split, 'FeatureParty', capture)
    run_demo(family='dnn')
    assert len(observed) == 2 and not torch.equal(observed[0], observed[1])


@pytest.mark.parametrize('family', ['../invalid', 'not-a-model'])
def test_unknown_family_is_rejected_without_creating_result(tmp_path, family):
    output = tmp_path/'result.json'
    assert main(['vertical', 'demo', '--family', family, '--output', str(output)]) == 2
    assert not output.exists()
