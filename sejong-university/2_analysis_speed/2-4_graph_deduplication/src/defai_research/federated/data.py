from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from .graph import GraphDeduplicator

SPLIT_SEED = 20261023


def prepare_data(dataset, seed, alpha=.3, duplicate_rate=.3, clients=12, split_seed=SPLIT_SEED):
    if dataset not in {'digits', 'breast_cancer'}:
        raise ValueError('Expected digits or breast_cancer')
    if split_seed != SPLIT_SEED:
        raise ValueError('All Transformer stages must use fixed split seed 20261023')
    if clients < 1 or alpha <= 0 or not 0 <= duplicate_rate <= 1:
        raise ValueError('Invalid client partition parameters')
    source = load_digits() if dataset == 'digits' else load_breast_cancer()
    x, y = source.data.astype(np.float32), source.target.astype(np.int64)
    ids = np.arange(len(y))
    train, other = train_test_split(ids, train_size=.6, stratify=y, random_state=split_seed)
    val, test = train_test_split(other, test_size=.5, stratify=y[other], random_state=split_seed + 1)
    split = {'dataset': dataset, 'split_seed': split_seed,
             'train_ids': train.tolist(), 'validation_ids': val.tolist(), 'test_ids': test.tolist()}
    fixed_split_sha256 = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise AssertionError('Source split leakage')
    if dataset == 'digits':
        x /= 16
        preprocessing = {'kind': 'fixed_public_scale', 'divisor': 16}
    else:
        mean, scale = x[train].mean(0), np.maximum(x[train].std(0), 1e-6)
        x = np.clip((x - mean) / scale, -5, 5).astype(np.float32)
        preprocessing = {'kind': 'train_only_zscore_then_clip', 'mean': mean.tolist(),
                         'scale': scale.tolist(), 'clip': [-5, 5], 'fit_source_ids': train.tolist()}
    hashes = GraphDeduplicator._fingerprints(x, y)
    content_sets = [set(hashes[int(i)] for i in subset) for subset in (train, val, test)]
    if (content_sets[0] & content_sets[1] or content_sets[0] & content_sets[2]
            or content_sets[1] & content_sets[2]):
        raise AssertionError('Equal feature/label records cross fixed source splits')
    if len(content_sets[0]) != len(train):
        raise AssertionError('Equal-content training source IDs require a grouped source protocol')

    rng = np.random.default_rng(np.random.SeedSequence([seed, 211]))
    shuffled, floor = rng.permutation(train), 3
    if clients * floor > len(train):
        raise ValueError('Not enough training records for the declared client floor')
    base = [shuffled[cid * floor:(cid + 1) * floor].tolist() for cid in range(clients)]
    rest = shuffled[clients * floor:]
    for label in range(len(np.unique(y))):
        selected = rng.permutation(rest[y[rest] == label])
        allocation = rng.multinomial(len(selected), rng.dirichlet(np.full(clients, alpha)))
        offset = 0
        for cid, n in enumerate(allocation):
            base[cid].extend(selected[offset:offset + n].tolist())
            offset += n
    base = [np.asarray(sorted(a), np.int64) for a in base]
    migrated = [[] for _ in range(clients)]
    drift_rng = np.random.default_rng(np.random.SeedSequence([seed, 321]))
    for cid, a in enumerate(base):
        shuffled = drift_rng.permutation(a)
        count = max(1, int(.2 * len(a)))
        migrated[cid].extend(shuffled[count:].tolist())
        migrated[(cid + 4) % clients].extend(shuffled[:count].tolist())
    migrated = [np.asarray(sorted(a), np.int64) for a in migrated]

    def duplicate(parts, direction, tag):
        rows = [a.tolist() for a in parts]
        generator = np.random.default_rng(np.random.SeedSequence([seed, tag]))
        for cid, a in enumerate(parts):
            count = min(len(a) - 1, int(round(duplicate_rate * len(a))))
            if count:
                rows[(cid + direction) % clients].extend(generator.choice(a, count, replace=False).tolist())
        return [np.asarray(a, np.int64) for a in rows]

    phases = {'initial': duplicate(base, 1, 431), 'drift': duplicate(migrated, 5, 432)}
    for parts in phases.values():
        if set(np.concatenate(parts).tolist()) != set(train.tolist()):
            raise AssertionError('Client phases must contain precisely the global training sources')
    counts = [np.bincount(y[a], minlength=len(np.unique(y))) / len(a) for a in base]
    entropy = [-float(np.sum(p[p > 0] * np.log2(p[p > 0]))) for p in counts]
    audit = {**split, 'seed': seed, 'alpha': alpha, 'duplicate_rate': duplicate_rate,
             'fixed_split_sha256': fixed_split_sha256,
             'initial_unique_source_ids': [a.tolist() for a in base],
             'drift_unique_source_ids': [a.tolist() for a in migrated],
             'phase_source_ids': {key: [a.tolist() for a in value] for key, value in phases.items()},
             'preprocessing': preprocessing, 'minimum_preallocation_records_per_client': floor,
             'mean_client_label_entropy_bits': float(np.mean(entropy)),
             'min_client_unique_records': min(map(len, base)), 'max_client_unique_records': max(map(len, base)),
             'train_validation_test_source_disjoint': True,
             'train_validation_test_exact_content_disjoint': True,
             'partition_seed_does_not_change_source_split': True}
    scenario_digest = hashlib.sha256(x.tobytes() + y.tobytes())
    scenario_digest.update(json.dumps(audit, sort_keys=True, separators=(',', ':')).encode())
    return {'x': x, 'y': y, 'train': train, 'val': val, 'test': test, 'phases': phases,
            'classes': len(np.unique(y)), 'audit': audit, 'fixed_split_sha256': fixed_split_sha256,
            'scenario_sha256': scenario_digest.hexdigest()}
