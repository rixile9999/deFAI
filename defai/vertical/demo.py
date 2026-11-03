"""Offline synthetic examples; never interpret these as task benchmarks."""
import csv
import json
from pathlib import Path
import numpy as np

FAMILIES = {
    'dnn': 'MLP feature parties and a label-owner classifier',
    'cnn': 'Local image views with pooled CNN embeddings',
    'rnn': 'LSTM and GRU over aligned local sequences',
    'tcn': 'Causal temporal convolutions',
    'transformer': 'Pooled and token-level encoder boundaries',
    'llm': 'Frozen LOCAL encoder adapter; demo is NOT a pretrained LLM',
    'gnn': 'Local graph encoders; complete VFGNN MPC is NOT implemented',
    'autoencoder': 'Label-free AE and VAE local reconstruction losses',
    'generative': 'Feature-partitioned GAN and diffusion prototypes',
    'gbdt': 'Encrypted histogram binary tree boosting',
}


def init_example(directory, *, rows=64, seed=42):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    if not isinstance(rows, int) or rows < 8:
        raise ValueError('At least 8 example rows required')
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(rows, 2)), rng.normal(size=(rows, 3))
    y = (a[:, 0]+b[:, 0] > 0).astype(int)
    for name, x in [('hospital', a), ('insurer', b)]:
        with (directory/f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            owner = name == 'hospital'
            writer.writerow(['id', *[f'x{i}' for i in range(x.shape[1])], *(['label'] if owner else [])])
            # Reverse passive rows to make accidental row-wise joining fail visibly.
            for i in (range(rows) if owner else reversed(range(rows))):
                writer.writerow([f'synthetic-{i}', *x[i].tolist(), *([int(y[i])] if owner else [])])
    manifest = {'parties': [{'name': n, 'path': f'{n}.csv', 'id_column': 'id',
                              'features': [f'x{i}' for i in range(x.shape[1])]}
                             for n, x in [('hospital', a), ('insurer', b)]],
                'label_owner': 'hospital', 'label_column': 'label'}
    path = directory/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return path


def run_demo(*, family='all', seed=42, insecure_test_key=False):
    import torch
    from torch import nn
    from .split import FeatureParty, SplitTrainer
    from .models import (mlp, ImageEncoder, RecurrentEncoder, TemporalEncoder, TransformerEncoder, FrozenEncoder, GraphEncoder)
    if family != 'all' and family not in FAMILIES:
        raise ValueError('Unknown model family')
    selected = list(FAMILIES) if family == 'all' else [family]
    previous = torch.get_num_threads()
    results = {}
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for name in selected:
                variants = []
                if name == 'autoencoder':
                    from .unsupervised import VerticalAutoencoder
                    x = [torch.randn(12, 2), torch.randn(12, 3)]
                    for variational in (False, True):
                        m = VerticalAutoencoder(x, variational=variational)
                        losses = [m.step(list(range(12))) for _ in range(3)]
                        variants.append({'variant': 'vae' if variational else 'ae', 'losses': losses})
                elif name == 'generative':
                    from .generative import VerticalGAN, VerticalDiffusion
                    for cls in (VerticalGAN, VerticalDiffusion):
                        m = cls([torch.randn(12, 2), torch.randn(12, 3)])
                        result = m.step(list(range(12)))
                        variants.append({'variant': cls.__name__, 'loss': result,
                                         'sample_shapes': [list(v.shape) for v in m.sample(3)]})
                elif name == 'gbdt':
                    from .boost import HistogramParty, SecureBoostClassifier
                    x = np.random.default_rng(seed).normal(size=(16, 4))
                    local = [HistogramParty('a', x[:, :2], bins=4), HistogramParty('b', x[:, 2:], bins=4)]
                    y = (x[:, 0]+x[:, 2] > 0).astype(int)
                    m = SecureBoostClassifier(trees=2, depth=2, key_bits=256 if insecure_test_key else 2048,
                                              insecure_test_key=insecure_test_key).fit(local, y)
                    variants.append({'variant': 'encrypted_histogram', 'training_accuracy': float((m.predict_proba(local).argmax(1) == y).mean()),
                                     'splits': len(m.audit), 'insecure_test_key': insecure_test_key})
                else:
                    kinds = ['lstm', 'gru'] if name == 'rnn' else ['pooled', 'tokens'] if name == 'transformer' else [name]
                    for kind in kinds:
                        if name == 'dnn':
                            x, factory = torch.randn(8, 3), lambda: mlp(3, 4)
                        elif name == 'cnn':
                            x, factory = torch.randn(8, 1, 8, 8), lambda: ImageEncoder(1, 4)
                        elif name == 'rnn':
                            x, factory = torch.randn(8, 5, 3), lambda kind=kind: RecurrentEncoder(3, 4, cell=kind)
                        elif name == 'tcn':
                            x, factory = torch.randn(8, 5, 3), lambda: TemporalEncoder(3, 4)
                        elif name == 'transformer':
                            x, factory = torch.randn(8, 5, 3), lambda kind=kind: TransformerEncoder(3, 4, tokens=kind == 'tokens')
                        elif name == 'llm':
                            x, factory = torch.randn(8, 5, 3), lambda: FrozenEncoder(TransformerEncoder(3, 4))
                        else:
                            x, factory = torch.randn(8, 3), lambda: GraphEncoder(3, torch.eye(8), 4)
                        labels = torch.zeros((8, 5) if kind == 'tokens' else (8,), dtype=torch.long)
                        local_views = [x.clone(), torch.randn_like(x)]
                        m = SplitTrainer([FeatureParty(str(i), factory(), view) for i, view in enumerate(local_views)], nn.Linear(8, 2), labels)
                        losses = [m.step(list(range(8))) for _ in range(3)]
                        variants.append({'variant': kind, 'losses': losses,
                                         'embedding_bytes': m.traffic.forward_bytes, 'gradient_bytes': m.traffic.backward_bytes})
                results[name] = {'description': FAMILIES[name], 'variants': variants}
    finally:
        torch.set_num_threads(previous)
    return {'status': 'research_local_simulation', 'data': 'synthetic_only', 'seed': seed, 'families': results,
            'real_network_tested': False, 'clinical_validation': False, 'formal_privacy_guarantee': False}
