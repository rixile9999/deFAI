from pathlib import Path
import importlib
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'common' / 'src'))
checks = []
owners = {
    'defai_addons.monitor.server': '1_simulation/1-2_fl_visualization',
    'defai_addons.monitor.simulator': '1_simulation/1-2_fl_visualization',
    'defai_addons.monitor.matching': '1_simulation/1-2_fl_visualization',
    'defai_addons.compare.runner': '1_simulation/1-4_distributed_comparison',
    'defai_addons.compare.transport': '1_simulation/1-4_distributed_comparison',
    'defai_addons.multimodal.models': '3_heterogeneous_data/3-1_multimodal',
    'defai_addons.multimodal.baselines.models': '3_heterogeneous_data/3-1_multimodal',
    'defai_addons.common.metrics': 'common',
    'defai_addons.fl.flower': 'common',
}
for name, owner in owners.items():
    module = importlib.import_module(name)
    relative = Path(module.__file__).resolve().relative_to(ROOT)
    assert relative.as_posix().startswith(owner + '/'), (name, relative)
    checks.append({'module': name, 'path': relative.as_posix(), 'status': 'PASS'})

script = runpy.run_path(str(ROOT / '3_heterogeneous_data/3-1_multimodal/scripts/train_multimodal.py'))
assert script['CENTRAL_MODELS'] == ['y3_mm_transformer']
from defai_addons.multimodal.models import MODEL_ZOO, build_model
assert 'y1_joint_embedding' not in MODEL_ZOO and 'y2_attention_fusion' not in MODEL_ZOO
for old in ['y1_joint_embedding', 'y2_attention_fusion']:
    assert '.baselines.' in type(build_model(old)).__module__
checks.append({'check': 'latest default and isolated historical baselines', 'status': 'PASS'})

from defai_addons.monitor.server import create_app
client = create_app().test_client()
for route in ['/', '/api/defaults', '/api/runs', '/static/dashboard.js']:
    response = client.get(route)
    assert response.status_code == 200, route
    checks.append({'route': route, 'status': 'PASS'})
data = client.get('/api/defaults').get_json()
assert 'comparison' in data and 'simulation' in data

for workspace in ROOT.glob('*/*/workspace.code-workspace'):
    settings = json.loads(workspace.read_text(encoding='utf-8'))
    for folder in settings['folders']:
        assert (workspace.parent / folder['path']).is_dir()
checks.append({'check': 'VS Code workspace folder references', 'status': 'PASS'})
print(json.dumps({'checks': checks, 'status': 'PASS'}, ensure_ascii=False, indent=2))
