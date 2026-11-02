"""Manifest-driven tabular experiments with training-only local normalization."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .alignment import align_parties


def read_manifest(path, *, alignment='plain'):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if set(cfg) != {'parties', 'label_owner', 'label_column'}:
        raise ValueError('Manifest needs exactly parties, label_owner and label_column')
    records, labels, receipts = {}, None, []
    for item in cfg['parties']:
        if set(item) != {'name', 'path', 'id_column', 'features'}:
            raise ValueError('Each party needs name, path, id_column and features')
        name = item['name']
        if name in records:
            raise ValueError('Duplicate party name')
        filename = (path.parent / item['path']).resolve()
        columns = item['features']
        if not columns or len(set(columns)) != len(columns) or item['id_column'] in columns or cfg['label_column'] in columns:
            raise ValueError('Feature columns must be distinct and exclude identity/label columns')
        with filename.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('Missing or duplicate CSV column names')
            rows = list(reader)
        required = [item['id_column'], *columns] + ([cfg['label_column']] if name == cfg['label_owner'] else [])
        if not rows or not set(required) <= set(reader.fieldnames):
            raise ValueError('Missing required CSV columns or rows')
        try:
            records[name] = ([r[item['id_column']] for r in rows], [[float(r[c]) for c in columns] for r in rows])
            if name == cfg['label_owner']:
                labels = [float(r[cfg['label_column']]) for r in rows]
        except (ValueError, TypeError) as exc:
            raise ValueError('CSV features/labels must be complete numeric values') from exc
        receipts.append({'party': name, 'sha256': hashlib.sha256(filename.read_bytes()).hexdigest(),
                         'local_rows': len(rows), 'feature_count': len(columns)})
    return align_parties(records, labels=labels, label_owner=cfg['label_owner'], mode=alignment), receipts


def run_tabular(data, *, method='split', epochs=30, batch_size=32, lr=.05, seed=42,
                q=3, test_fraction=.25, key_bits=2048, insecure_test_key=False):
    if method not in {'split', 'fedbcd-p', 'fedbcd-s', 'neural-bcd-p', 'neural-bcd-s', 'secureboost'}:
        raise ValueError('Unknown vertical method')
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1 or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive integers')
    if not 0 < test_fraction < 1 or data.size < 8 or data.labels is None or data.labels.ndim != 1 or not np.isin(data.labels, [0, 1]).all():
        raise ValueError('Need >=8 aligned rows, binary labels and a held-out fraction in (0,1)')
    rng = np.random.default_rng(seed)
    order = rng.permutation(data.size)
    ntest = max(1, int(data.size*test_fraction))
    train, test = order[ntest:], order[:ntest]
    arrays, normalization = {}, {}
    for name, value in data.features.items():
        if value.ndim != 2:
            raise ValueError('Tabular command accepts matrices; use the model APIs for other modalities')
        mean, std = value[train].mean(0), value[train].std(0)
        std = np.where(std < 1e-12, 1., std)
        arrays[name] = (value-mean)/std
        normalization[name] = {'fitted_on': 'train_only', 'feature_count': value.shape[1]}
    history, traffic = [], None
    if method in {'split', 'neural-bcd-p', 'neural-bcd-s'}:
        import torch
        from torch import nn
        from .models import mlp
        from .split import FeatureParty, SplitTrainer
        # Reproducible initialization without replacing the caller's CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            parties = [FeatureParty(name, mlp(x.shape[1], 8), torch.tensor(x, dtype=torch.float32), lr=lr) for name, x in arrays.items()]
            top, labels = nn.Linear(len(parties)*8, 2), torch.tensor(data.labels, dtype=torch.long)
            if method == 'split':
                trainer = SplitTrainer(parties, top, labels, lr=lr)
            else:
                from .neural_bcd import NeuralFedBCD
                trainer = NeuralFedBCD(parties, top, labels, label_owner=data.label_owner, q=q,
                                       mode='parallel' if method == 'neural-bcd-p' else 'sequential', lr=lr)
            for _ in range(epochs):
                batch_order = rng.permutation(train)
                losses = [trainer.step(batch_order[i:i+batch_size]) for i in range(0, len(train), batch_size)]
                history.append(float(np.mean(losses)))
            probabilities = trainer.predict(test).softmax(-1)[:, 1].numpy()
            traffic = {'embedding_bytes': trainer.traffic.forward_bytes, 'gradient_bytes': trainer.traffic.backward_bytes,
                       'batch_exchanges': trainer.traffic.exchanges, 'measured_network': False}
            if method != 'split':
                traffic['local_steps'] = trainer.local_steps
    elif method.startswith('fedbcd'):
        from .bcd import LinearParty, FedBCD
        local = [LinearParty(name, x) for name, x in arrays.items()]
        trainer = FedBCD(local, data.labels, label_owner=data.label_owner, lr=lr, q=q,
                         mode='parallel' if method == 'fedbcd-p' else 'sequential')
        for _ in range(epochs):
            batch_order = rng.permutation(train)
            for i in range(0, len(train), batch_size):
                trainer.step(batch_order[i:i+batch_size])
            history.append(trainer.loss(train))
        probabilities = trainer.predict()[test]
        traffic = {'payload_bytes': trainer.bytes, 'batch_exchanges': trainer.rounds, 'local_steps': trainer.local_steps,
                   'measured_network': False}
    else:
        from .boost import HistogramParty, SecureBoostClassifier
        local = [HistogramParty(name, x[train]) for name, x in arrays.items()]
        trainer = SecureBoostClassifier(trees=epochs, lr=lr, key_bits=key_bits, insecure_test_key=insecure_test_key).fit(local, data.labels[train])
        inferred = [party.with_features(arrays[party.name][test]) for party in local]
        probabilities = trainer.predict_proba(inferred)[:, 1]
    labels = data.labels[test]
    p = np.clip(probabilities, 1e-12, 1-1e-12)
    return {'schema': 1, 'status': 'research_local_simulation', 'method': method, 'alignment': data.alignment,
            'seed': seed, 'train_rows': len(train), 'test_rows': len(test), 'normalization': normalization,
            'accuracy': float(((p >= .5) == labels).mean()),
            'log_loss': float(-(labels*np.log(p)+(1-labels)*np.log1p(-p)).mean()),
            'loss_history': history, 'communication': traffic, 'clinical_validation': False,
            'neural_messages_encrypted': False, 'insecure_test_key': insecure_test_key if method == 'secureboost' else False}
