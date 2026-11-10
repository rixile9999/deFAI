from __future__ import annotations
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import flwr
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from threadpoolctl import threadpool_limits
from ..federated import learning as base
from ..aggregation import map_reduce
from ..federated.cache import GraphSAGECachedDeduplicator
from ..federated.snapshots import ImmutableTrainingSnapshot
from ..federated.data import prepare_data, SPLIT_SEED
from .runner import _source_hashes as source_hashes

DATASETS = ('digits', 'breast_cancer')
SEEDS = (14001, 14002, 14003)
ROLES = {0:'client_fp32_delta', 1:'group_delta_from_int8_received_clients'}
GRAPH = dict(refresh_interval=4, regions=4, candidate_k=2, candidate_threshold=.2,
             epochs=100, fanout=3, hidden_dim=12, embedding_dim=8, mode='graphsage')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receive_int8(vector):
    reference, outgoing = base.quantize(vector)
    codes, scales = parameters_to_ndarrays(ndarrays_to_parameters(outgoing))
    if codes.dtype != np.int8 or scales.dtype != np.float32:
        raise AssertionError('Unexpected INT8 wire dtype')
    received = np.empty(len(codes), np.float32)
    for block, start in enumerate(range(0, len(codes), 256)):
        received[start:start + 256] = codes[start:start + 256].astype(np.float32) * scales[block]
    if received.tobytes() != reference.tobytes():
        raise AssertionError('Received code/scale decoding differs from quantizer reconstruction')
    return received


class BlockReservoir:
    def __init__(self, max_blocks_per_role=12000, block_size=256):
        if isinstance(max_blocks_per_role, bool) or not isinstance(max_blocks_per_role, int):
            raise ValueError('max_blocks_per_role must be an integer')
        if max_blocks_per_role < len(DATASETS) * len(SEEDS):
            raise ValueError('Each dataset/seed/role stratum needs at least one block')
        self.capacity = max_blocks_per_role // (len(DATASETS) * len(SEEDS))
        self.block_size = block_size
        self.rows, self.seen, self.rng = {}, Counter(), {}
        self.update_count = 0

    def add(self, vector, *, role, dataset, seed, round_id, endpoint):
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Only finite, nonempty real model updates may be collected')
        key = (dataset, seed, role)
        if key not in self.rows:
            self.rows[key] = []
            self.rng[key] = np.random.default_rng(np.random.SeedSequence([20261023, *key]))
        uid = self.update_count
        self.update_count += 1
        for index, start in enumerate(range(0, len(values), self.block_size)):
            valid = min(self.block_size, len(values) - start)
            block = np.zeros(self.block_size, np.float32)
            block[:valid] = values[start:start + valid]
            row = (block, valid, role, dataset, seed, round_id, endpoint, uid, index)
            self.seen[key] += 1
            if len(self.rows[key]) < self.capacity:
                self.rows[key].append(row)
            else:
                candidate = int(self.rng[key].integers(self.seen[key]))
                if candidate < self.capacity:
                    self.rows[key][candidate] = row

    def arrays(self):
        rows = [row for key in sorted(self.rows) for row in self.rows[key]]
        names = ('blocks', 'valid_lengths', 'role', 'dataset', 'seed', 'round', 'endpoint', 'update_id', 'block_index')
        result = {name: np.stack([row[i] for row in rows]).astype(np.float32 if i == 0 else np.int64)
                  for i, name in enumerate(names)}
        result['codec_holdout'] = result['seed'] == 14003
        if not result['codec_holdout'].any() or result['codec_holdout'].all():
            raise AssertionError('Codec train and held-out update partitions are both required')
        if set(result['update_id'][result['codec_holdout']]) & set(result['update_id'][~result['codec_holdout']]):
            raise AssertionError('A whole model update crosses codec train/holdout partitions')
        return result


def _collect_calibration(output, max_blocks_per_role=12000, *, rounds=24, drift_round=13):
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2:
        raise ValueError('rounds must be an integer >=2')
    if not isinstance(drift_round, int) or isinstance(drift_round, bool) or not 1 <= drift_round <= rounds:
        raise ValueError('drift_round must be an integer within calibration rounds')
    protocol = dict(schema=1, datasets=list(DATASETS), seeds=list(SEEDS), rounds=rounds, drift_round=drift_round,
                    source_sha256=source_hashes(), split_seed=SPLIT_SEED, graph=GRAPH,
                    roles=ROLES, block_size=256, max_blocks_per_role=max_blocks_per_role,
                    codec_training_seeds=[14001, 14002], codec_heldout_seed=14003,
                    learning=dict(clients=12, groups=4, alpha=.3, duplicate_rate=.3,
                                  local_epochs=2, batch_size=32, learning_rate=.2, l2=.0001,
                                  correction_cap=1.5, aggregation_weight='retained record count'),
                    actual_loaded_versions=dict(python=sys.version, numpy=np.__version__, flwr=flwr.__version__),
                    trajectory='INT8 client receive -> weighted group Reduce -> INT8 group receive -> central Flower FedAvg',
                    holdout_rule='whole updates from seed14003; no rows from source validation/test used in any calibration fit',
                    predictions_computed=False, test_evaluated=False)
    protocol = json.loads(json.dumps(protocol))
    output = Path(output)
    if (output / 'manifest.json').exists():
        existing = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if existing['protocol'] != protocol or existing['npz_sha256'] != sha(output / 'calibration_blocks.npz'):
            raise ValueError('Existing completed calibration differs; use another output directory')
        return existing
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Frozen calibration protocol differs; use another output directory')
    base.json_write(protocol_path, protocol)
    reservoir, histories, audits, splits = BlockReservoir(max_blocks_per_role), [], [], {}
    overall_cpu, overall_wall = time.process_time(), time.perf_counter()
    client_calls = group_calls = 0
    for dataset_id, dataset in enumerate(DATASETS):
        for seed in SEEDS:
            run_cpu, run_wall = time.process_time(), time.perf_counter()
            data = prepare_data(dataset, seed)
            if dataset in splits and splits[dataset]['fixed_split_sha256'] != data['fixed_split_sha256']:
                raise AssertionError('Source split changed across calibration seeds')
            splits[dataset] = {key: data['audit'][key] for key in
                               ('fixed_split_sha256', 'train_ids', 'validation_ids', 'test_ids', 'preprocessing')}
            out = output / 'runs' / f'{dataset}_s{seed}'
            base.json_write(out / 'source_audit.json', data['audit'])
            policy = GraphSAGECachedDeduplicator(seed=seed, dynamic=True, refresh_policy='event_only',
                                               audit_dir=out / 'graph', **GRAPH)
            strategy = FedAvg(accept_failures=False, inplace=False, fit_metrics_aggregation_fn=lambda _: {})
            arrays = base.make_model(seed, data['x'].shape[1], data['classes'])
            snapshots = {}
            local_history = []
            for rnd in range(1, rounds + 1):
                phase = 'initial' if rnd < drift_round else 'drift'
                parts = data['phases'][phase]
                if phase not in snapshots:
                    snapshots[phase] = ImmutableTrainingSnapshot.from_arrays(
                        [data['x'][ids] for ids in parts], [data['y'][ids] for ids in parts])
                snapshot = snapshots[phase]
                masks, graph_meta = policy.apply(snapshot, rnd, force_refresh=rnd == drift_round)
                record_audit = base.source_audit(parts, masks, list(range(12)))
                if record_audit['lost_unique_records']:
                    raise AssertionError('Calibration deduplication lost unique training sources')
                old, fits, models, fitted = [a.copy() for a in arrays], [], [], []
                for cid in range(12):
                    kept = int(masks[cid].sum())
                    if not kept:
                        continue
                    ref_steps, actual_steps = 2 * math.ceil(len(parts[cid]) / 32), 2 * math.ceil(kept / 32)
                    factor = min(1.5, ref_steps / actual_steps)
                    cfg = dict(round=rnd, local_epochs=2, batch_size=32, l2=.0001,
                               learning_rate=.2 / math.sqrt(1 + (rnd - 1) / 20) * factor)
                    client = base.BenchmarkClient(cid, snapshot.x[cid][masks[cid]], snapshot.y[cid][masks[cid]], seed).to_client()
                    fit = client.fit(FitIns(ndarrays_to_parameters(old), cfg))
                    if fit.num_examples != kept:
                        raise AssertionError('Actual Flower fit weight differs from retained records')
                    fits.append(fit)
                    models.append(parameters_to_ndarrays(fit.parameters))
                    fitted.append(cid)
                    client_calls += 1
                weights = [fit.num_examples for fit in fits]
                groups = [(cid // 3 + int(phase == 'drift' and cid % 3 == 0)) % 4 for cid in fitted]
                deltas = [base.pack(model).astype(np.float64) - base.pack(old).astype(np.float64) for model in models]
                for delta, cid in zip(deltas, fitted):
                    reservoir.add(delta, role=0, dataset=dataset_id, seed=seed, round_id=rnd, endpoint=cid)
                decoded_clients = [receive_int8(delta) for delta in deltas]
                _, group_updates, counts, reduce_gap = map_reduce(decoded_clients, weights, groups)
                for group, delta in sorted(group_updates.items()):
                    reservoir.add(delta, role=1, dataset=dataset_id, seed=seed, round_id=rnd, endpoint=group)
                decoded_groups = {group: receive_int8(delta) for group, delta in group_updates.items()}
                group_results = []
                for group in sorted(decoded_groups):
                    received_model = base.reshape_like(base.pack(old).astype(np.float64) + decoded_groups[group], old)
                    group_results.append((None, replace(fits[0], parameters=ndarrays_to_parameters(received_model),
                                                        num_examples=counts[group], metrics={})))
                aggregate, _ = strategy.aggregate_fit(rnd, group_results, failures=[])
                arrays = parameters_to_ndarrays(aggregate)
                if not all(np.isfinite(a).all() for a in arrays):
                    raise AssertionError('Nonfinite actual received global model')
                reference_update = sum(counts[g] * decoded_groups[g].astype(np.float64) for g in decoded_groups) / sum(weights)
                reference_model = base.reshape_like(base.pack(old).astype(np.float64) + reference_update, old)
                received_gap = float(np.max(np.abs(base.pack(arrays) - base.pack(reference_model))))
                if reduce_gap > 1e-10 or received_gap > 2e-6:
                    raise AssertionError('Actual received group/Flower aggregation mismatch')
                group_calls += len(group_results)
                local_history.append(dict(dataset=dataset, seed=seed, round=rnd, phase=phase,
                                          client_fits=len(fits), group_models=len(group_results),
                                          group_reduce_abs_error=reduce_gap, central_flower_abs_error=received_gap,
                                          graph_fits=graph_meta['actual_graphsage_fits_this_round'], **record_audit))
            base.csv_write(out / 'rounds.csv', local_history)
            audit = dict(dataset=dataset, seed=seed, fixed_split_sha256=data['fixed_split_sha256'],
                         scenario_sha256=data['scenario_sha256'], client_fits=sum(r['client_fits'] for r in local_history),
                         group_updates=sum(r['group_models'] for r in local_history), graph_fits=len(policy.history),
                         process_cpu_seconds=time.process_time() - run_cpu, elapsed_seconds=time.perf_counter() - run_wall,
                         no_validation_or_test_prediction=True,
                         max_group_reduce_abs_error=max(r['group_reduce_abs_error'] for r in local_history),
                         max_central_flower_abs_error=max(r['central_flower_abs_error'] for r in local_history))
            audits.append(audit)
            histories.extend(local_history)
            base.json_write(out / 'completed.json', audit)
            print(f'Calibration {dataset} seed{seed}: {audit["client_fits"]} real Flower fits; {audit["group_updates"]} groups', flush=True)
    corpus = reservoir.arrays()
    if source_hashes() != protocol['source_sha256']:
        raise RuntimeError('A frozen source changed during calibration')
    corpus_path = output / 'calibration_blocks.npz'
    np.savez_compressed(corpus_path, **corpus)
    with np.load(corpus_path, allow_pickle=False) as restored:
        if any(not np.array_equal(restored[key], value) for key, value in corpus.items()):
            raise AssertionError('Saved corpus does not reconstruct the collected blocks exactly')
    strata = [dict(dataset=DATASETS[key[0]], seed=key[1], role=ROLES[key[2]],
                   seen_blocks=reservoir.seen[key], retained_blocks=len(reservoir.rows[key])) for key in sorted(reservoir.rows)]
    manifest = dict(status='completed', protocol=protocol, fixed_source_splits=splits,
                    source_unchanged=True, source_sha256_at_end=source_hashes(), source_ids_disjoint=True,
                    no_validation_or_test_predictions=True, no_validation_or_test_records_in_updates=True,
                    npz_path=str(corpus_path), npz_sha256=sha(corpus_path), npz_file_bytes=corpus_path.stat().st_size,
                    retained_blocks=len(corpus['blocks']), raw_array_bytes=sum(a.nbytes for a in corpus.values()),
                    unique_seen_updates=reservoir.update_count, flower_client_fit_calls=client_calls,
                    flower_central_aggregate_calls=len(histories), received_group_models=group_calls,
                    calibration_codec_train_blocks=int((~corpus['codec_holdout']).sum()),
                    calibration_codec_heldout_blocks=int(corpus['codec_holdout'].sum()),
                    codec_holdout_seed=14003, roles=ROLES, datasets=dict(enumerate(DATASETS)),
                    process_cpu_seconds=time.process_time() - overall_cpu,
                    elapsed_seconds=time.perf_counter() - overall_wall,
                    cpu_scope='complete serial collection including graph audit and final corpus writes; no torch fitting',
                    strata=strata, runs=audits)
    base.json_write(output / 'manifest.json', manifest)
    return manifest


def collect_calibration(output, max_blocks_per_role=12000, *, rounds=24, drift_round=13):
    with threadpool_limits(limits=1):
        return _collect_calibration(output, max_blocks_per_role, rounds=rounds, drift_round=drift_round)
