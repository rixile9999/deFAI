from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import time

import flwr
from flwr.common import Code, FitIns, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
import numpy as np
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits

from ..federated import learning as base
from ..aggregation import map_reduce
from ..federated.cache import GraphSAGECachedDeduplicator
from ..federated.data import prepare_data, SPLIT_SEED
from ..federated.snapshots import ImmutableTrainingSnapshot
from .lossless import CompactLosslessCodec
from .legacy import LegacyLosslessCodec
from .transformer import TransformerCompactCodec

GRAPH_PARAMETERS = dict(refresh_interval=4, regions=4, candidate_k=2,
                        candidate_threshold=.2, epochs=100, fanout=3,
                        hidden_dim=12, embedding_dim=8, mode='graphsage')


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _source_hashes():
    from ..source_files import source_files
    return {name: _sha(path.read_bytes()) for name, path in source_files().items()}


def _fit_result(arrays, n):
    return FitRes(status=Status(code=Code.OK, message='received packet update'),
                  parameters=ndarrays_to_parameters(arrays), num_examples=int(n), metrics={})


def _corrections(parts, masks, epochs, batch_size):
    result = {}
    for cid in range(len(parts)):
        raw_n, kept_n = len(parts[cid]), int(masks[cid].sum())
        reference, actual = epochs * math.ceil(raw_n / batch_size), epochs * math.ceil(kept_n / batch_size)
        result[cid] = dict(cid=cid, raw_n=raw_n, kept_n=kept_n,
                           reference_steps=reference, actual_steps=actual,
                           power=1., cap=1.5, lr_factor=min(1.5, reference / actual) if actual else 1.,
                           aggregation_weight=kept_n)
    return result


def _descriptor(round_id, version, c):
    payload = struct.pack('<I6Q3dQ', 0, c['cid'], round_id, c['raw_n'], c['kept_n'],
                          c['reference_steps'], c['actual_steps'], c['power'], c['cap'],
                          c['lr_factor'], c['aggregation_weight']) + version
    if len(payload) != 116:
        raise AssertionError('Invalid correction descriptor length')
    return dict(kind='quality_policy', payload=payload, bootstrap=False)


class WireSession:
    def __init__(self, codec, artifact=None):
        self.codec_name = codec
        lossless_type = LegacyLosslessCodec if codec == 'legacy' else CompactLosslessCodec
        self.down_codec, self.down_receiver = lossless_type(), lossless_type()
        self.up_codec = lossless_type() if codec in {'compact', 'legacy'} else None
        self.up_receiver = lossless_type() if codec in {'compact', 'legacy'} else None
        self.transformer = TransformerCompactCodec(artifact, error_feedback=False) if codec == 'transformer' else None
        self.receivers = {}
        self.deployment = []
        self.residuals, self.ledger, self.samples = {}, [], []
        self.sample_arrays, self.sampled_channels = {}, set()
        if self.transformer is not None:
            packets = self.transformer.deployment_packets()
            self.receivers[('central', 0)] = TransformerCompactCodec(self.transformer.artifact)
            for packet in packets:
                payload = packet['payload']
                if packet['destination'][0] == 'group':
                    self.receivers[tuple(packet['destination'])] = TransformerCompactCodec.receiver_from_deployment(payload)
                self.deployment.append({k: v for k, v in packet.items() if k != 'payload'} |
                                       dict(payload_bytes=len(payload), payload_sha256=_sha(payload)))
            if len(packets) != 16:
                raise AssertionError('Codec deployment must cover four groups and twelve clients')
        self.deployment_bytes = sum(p['payload_bytes'] for p in self.deployment)

    def begin(self, rnd, model):
        self.round, self.shapes = rnd, [list(a.shape) for a in model]
        self.frames, self.counts = [], Counter()

    def _record(self, transfer, kind, token, n, channel, direction, extra=None):
        header = json.dumps(dict(round=self.round, kind=kind, token=int(token), n=int(n),
                                 shapes=self.shapes,
                                 codec_id=(2 if self.transformer else 1) if direction == 'up' else 1),
                            sort_keys=True, separators=(',', ':')).encode()
        evidence = transfer.evidence()
        self.counts['model_payload_bytes'] += transfer.frame_bytes
        self.counts['model_protocol_header_bytes'] += len(header)
        self.counts[direction + '_bytes'] += transfer.frame_bytes + len(header)
        self.frames.append({**evidence, **(extra or {}), 'kind': kind,
                            'channel': list(channel), 'direction': direction,
                            'json_header_bytes': len(header), 'json_header_sha256': _sha(header),
                            'aggregation_weight': int(n)})

    def down(self, arrays, *, kind, token, channel):
        transfer = self.down_codec.transmit(arrays, channel=channel, round_id=self.round)
        received = self.down_receiver.decode_frame(transfer.frame, expected_channel=channel,
                                                   expected_round=self.round).arrays
        if len(received) != len(arrays) or any(a.dtype != b.dtype or a.shape != b.shape
                                              or a.tobytes() != b.tobytes()
                                              for a, b in zip(arrays, received)):
            raise AssertionError('Byte-exact FP32 downlink changed the model')
        self._record(transfer, kind, token, 0, channel, 'down', dict(lossless_byte_equal=True))
        return received

    def up(self, raw, *, kind, token, n, channel):
        raw = np.asarray(raw, np.float32)
        corrected = (raw + self.residuals.get(channel, np.zeros_like(raw))).astype(np.float32)
        if self.transformer:
            transfer = self.transformer.transmit([corrected], channel=channel, round_id=self.round)
            receiver = ('group', channel[3]) if channel[0] == 'edge' else ('central', 0)
            received, _ = self.receivers[receiver].decode_frame(transfer.frame, channel=channel, round_id=self.round)
            decoded = received[0]
            if not np.array_equal(decoded, transfer.decoded_arrays[0]):
                raise AssertionError('Separate decoder differs from sender-side verification')
            self.counts['transformer_frames'] += 1
            self.counts['transformer_frame_bytes'] += transfer.frame_bytes
        else:
            _, arrays = base.quantize(corrected)
            transfer = self.up_codec.transmit(arrays, channel=channel, round_id=self.round)
            received = self.up_receiver.decode_frame(transfer.frame, expected_channel=channel,
                                                     expected_round=self.round).arrays
            codes, scales = received
            if codes.dtype != np.int8 or scales.dtype != np.float32 or len(scales) != math.ceil(len(codes) / 256):
                raise AssertionError('Invalid received block-INT8 payload')
            decoded = (codes.astype(np.float32) * np.repeat(scales, 256)[:len(codes)]).astype(np.float32)
            self.counts['int8_frames'] += 1
        if decoded.shape != raw.shape or decoded.dtype != np.float32 or not np.isfinite(decoded).all():
            raise FloatingPointError('Nonfinite or malformed independently received update')
        self.residuals[channel] = (corrected - decoded).astype(np.float32)
        error = decoded.astype(np.float64) - raw.astype(np.float64)
        self.counts['reconstruction_error_squared'] += float(np.dot(error, error))
        self.counts['raw_update_energy'] += float(np.dot(raw.astype(np.float64), raw.astype(np.float64)))
        self._record(transfer, kind, token, n, channel, 'up',
                     dict(received_update_used_by_flower=True, separate_receiver_object=True,
                          input_sha256=_sha(corrected.tobytes()), decoded_sha256=_sha(decoded.tobytes())))
        if channel not in self.sampled_channels:
            index = len(self.samples)
            self.samples.append({**transfer.evidence(include_frame=True), 'channel': list(channel),
                                 'round': self.round, 'array_prefix': f's{index}'})
            for name, values in [('raw', raw), ('corrected', corrected), ('received', decoded)]:
                self.sample_arrays[f's{index}_{name}'] = values.copy()
            self.sampled_channels.add(channel)
        return decoded

    def finish(self, control_bytes):
        deployment = self.deployment_bytes if self.round == 1 else 0
        model = self.counts['up_bytes'] + self.counts['down_bytes']
        result = dict(model_wire_bytes=model, model_payload_bytes=self.counts['model_payload_bytes'],
                      model_protocol_header_bytes=self.counts['model_protocol_header_bytes'],
                      up_bytes=self.counts['up_bytes'], down_bytes=self.counts['down_bytes'],
                      graph_control_both_hops_bytes=2 * control_bytes,
                      codec_deployment_bytes=deployment, total_bytes=model + 2 * control_bytes + deployment,
                      transformer_frames=self.counts['transformer_frames'],
                      transformer_frame_bytes=self.counts['transformer_frame_bytes'],
                      int8_frames=self.counts['int8_frames'],
                      reconstruction_error_squared=self.counts['reconstruction_error_squared'],
                      raw_update_energy=self.counts['raw_update_energy'])
        self.ledger.append(dict(round=self.round, frames=self.frames, **result))
        return result


def run_benchmark(*, output, dataset='digits', seed=19001, rounds=80, drift_round=41,
                  codec='transformer', artifact=None, dynamic=True, allow_test=True,
                  evaluate_validation=True, graph_epochs=100, local_epochs=2,
                  batch_size=32, learning_rate=.2, alpha=.3, duplicate_rate=.3,
                  block_observer=None):
    if codec not in {'transformer', 'compact', 'legacy'}:
        raise ValueError('codec must be transformer, compact or legacy (reference control)')
    if codec == 'transformer' and artifact is None:
        raise ValueError('An explicitly trained codec artifact is required')
    if not isinstance(dynamic, bool) or not isinstance(allow_test, bool) or not isinstance(evaluate_validation, bool):
        raise TypeError('dynamic, allow_test and evaluate_validation must be bool')
    if not evaluate_validation and allow_test:
        raise ValueError('Calibration cannot inspect test predictions')
    for name, value in [('seed', seed), ('rounds', rounds), ('drift_round', drift_round),
                        ('graph_epochs', graph_epochs), ('local_epochs', local_epochs), ('batch_size', batch_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
            raise ValueError(name + ' must be a valid integer')
    if not 1 <= drift_round <= rounds + 1:
        raise ValueError('drift_round must be within the run, or rounds+1 for no observed drift')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve existing results; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    hashes = _source_hashes()
    protocol = dict(schema=1, dataset=dataset, seed=seed, rounds=rounds, drift_round=drift_round,
                    clients=12, groups=4, codec=codec, dynamic=dynamic, graph_epochs=graph_epochs,
                    local_epochs=local_epochs, batch_size=batch_size, learning_rate=learning_rate,
                    alpha=alpha, duplicate_rate=duplicate_rate, l2=.0001, correction_cap=1.5,
                    split_seed=SPLIT_SEED, source_sha256=hashes, python=platform.python_version(),
                    numpy=np.__version__, flower=flwr.__version__,
                    codec_artifact_sha256=_sha(Path(artifact).read_bytes()) if artifact else None,
                    validation_evaluated=evaluate_validation, test_evaluated=allow_test,
                    byte_scope='Application frames, context headers, both hierarchy control hops and codec deployment',
                    physical_network_measured=False, prior_public_dataset_exploration=True)
    base.json_write(out / 'protocol.json', protocol)
    data = prepare_data(dataset, seed, alpha, duplicate_rate, clients=12, split_seed=SPLIT_SEED)
    base.json_write(out / 'source_audit.json', data['audit'])
    import torch
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    wire = WireSession(codec, artifact)
    policy = GraphSAGECachedDeduplicator(seed=seed, dynamic=dynamic, refresh_policy='event_only',
                audit_dir=out / 'graph', **{**GRAPH_PARAMETERS, 'epochs': graph_epochs})
    strategy = FedAvg(accept_failures=False, inplace=False, fit_metrics_aggregation_fn=lambda _: {})
    arrays = base.make_model(seed, data['x'].shape[1], data['classes'])
    snapshots, descriptors, masks_saved = {}, {}, {}
    history, controls, validation = [], [], []
    best, best_round, best_loss = None, 0, float('inf')
    active = list(range(12))
    with threadpool_limits(limits=1):
        for rnd in range(1, rounds + 1):
            phase = 'drift' if rnd >= drift_round else 'initial'
            parts = data['phases'][phase]
            if phase not in snapshots:
                snapshots[phase] = ImmutableTrainingSnapshot.from_arrays(
                    [data['x'][ids] for ids in parts], [data['y'][ids] for ids in parts])
            snapshot = snapshots[phase]
            masks, meta = policy.apply(snapshot, rnd, active=active, force_refresh=dynamic and rnd == drift_round)
            packets, corrections = list(policy.last_packets), _corrections(parts, masks, local_epochs, batch_size)
            quality_bytes = 0
            for cid in active:
                packet = _descriptor(rnd, snapshot.ordered_versions[cid], corrections[cid])
                logical = packet['payload'][:12] + packet['payload'][20:]
                if descriptors.get(cid) != logical:
                    packets.append(packet)
                    quality_bytes += len(packet['payload'])
                    descriptors[cid] = logical
                masks_saved[f'r{rnd:02d}_c{cid:02d}'] = masks[cid]
            meta = dict(meta)
            meta['quality_control_bytes'] = quality_bytes
            meta['metadata_bytes'] += quality_bytes
            if sum(len(p['payload']) for p in packets) != meta['metadata_bytes']:
                raise AssertionError('Control packet lengths differ from accounting')
            controls.append(dict(round=rnd, packets=[{k: v for k, v in p.items() if k != 'payload'} |
                                                     dict(hex=p['payload'].hex()) for p in packets]))
            audit = base.source_audit(parts, masks, active)
            if audit['lost_unique_records']:
                raise AssertionError('Deduplication lost a unique training source')
            old = [a.copy() for a in arrays]
            wire.begin(rnd, old)
            fitted = [cid for cid in active if masks[cid].any()]
            group_of = {cid: (cid // 3 + int(phase == 'drift' and cid % 3 == 0)) % 4 for cid in fitted}
            received_groups = {g: wire.down(old, kind='group_down', token=g,
                                channel=('cloud', 'down', 'central', g)) for g in sorted(set(group_of.values()))}
            group_results, raw_deltas, received_deltas, weights, groups = {}, [], [], [], []
            steps = examples = 0
            for cid in fitted:
                group = group_of[cid]
                received_old = wire.down(received_groups[group], kind='client_down', token=cid,
                                         channel=('edge', 'down', group, cid))
                client = base.BenchmarkClient(cid, snapshot.x[cid][masks[cid]], snapshot.y[cid][masks[cid]], seed).to_client()
                fit = client.fit(FitIns(ndarrays_to_parameters(received_old),
                    dict(round=rnd, local_epochs=local_epochs, batch_size=batch_size, l2=.0001,
                         learning_rate=learning_rate / math.sqrt(1 + (rnd - 1) / 20) * corrections[cid]['lr_factor'])))
                local = parameters_to_ndarrays(fit.parameters)
                raw = (base.pack(local).astype(np.float64) - base.pack(received_old).astype(np.float64)).astype(np.float32)
                if block_observer is not None:
                    block_observer(raw.copy(), role=0, round_id=rnd, endpoint=cid)
                n = corrections[cid]['kept_n']
                if fit.num_examples != n:
                    raise AssertionError('Flower fit count differs from retained sample count')
                decoded = wire.up(raw, kind='client_up', token=cid, n=n,
                                  channel=('edge', 'up', cid, group))
                received_model = base.reshape_like(base.pack(received_old).astype(np.float64) + decoded, old)
                group_results.setdefault(group, []).append((None, _fit_result(received_model, n)))
                raw_deltas.append(raw); received_deltas.append(decoded); weights.append(n); groups.append(group)
                steps += int(fit.metrics['optimizer_steps']); examples += int(fit.metrics['examples_processed'])
            exact_raw, _, _, map_gap = map_reduce(raw_deltas, weights, groups)
            _, expected_groups, expected_counts, received_gap = map_reduce(received_deltas, weights, groups)
            central_results, received_cloud, group_gap = [], {}, 0.
            scale = max(1., max(float(np.max(np.abs(a))) for a in old))
            for group, fits in sorted(group_results.items()):
                aggregated, _ = strategy.aggregate_fit(rnd, fits, failures=[])
                group_model = parameters_to_ndarrays(aggregated)
                delta = (base.pack(group_model).astype(np.float64) - base.pack(old).astype(np.float64)).astype(np.float32)
                if block_observer is not None:
                    block_observer(delta.copy(), role=1, round_id=rnd, endpoint=group)
                scale = max(scale, max(float(np.max(np.abs(a))) for a in group_model), float(np.max(np.abs(delta))))
                group_gap = max(group_gap, float(np.max(np.abs(delta - expected_groups[group]))))
                n = expected_counts[group]
                decoded = wire.up(delta, kind='group_up', token=group, n=n,
                                  channel=('cloud', 'up', group, 'central'))
                received_cloud[group] = decoded
                model = base.reshape_like(base.pack(old).astype(np.float64) + decoded, old)
                central_results.append((None, _fit_result(model, n)))
            aggregated, _ = strategy.aggregate_fit(rnd, central_results, failures=[])
            arrays = parameters_to_ndarrays(aggregated)
            independent = sum(expected_counts[g] * a.astype(np.float64) for g, a in received_cloud.items()) / sum(weights)
            central_gap = float(np.max(np.abs(base.pack(arrays).astype(np.float64) - base.pack(old).astype(np.float64) - independent)))
            scale = max(scale, max(float(np.max(np.abs(a))) for a in arrays), float(np.max(np.abs(independent))))
            if group_gap > 2e-6 * scale or central_gap > 2e-6 * scale or not all(np.isfinite(a).all() for a in arrays):
                raise FloatingPointError('Actual decoded Flower aggregation failed numerical verification')
            row = {**audit, **meta, 'round': rnd, 'phase': phase,
                   'examples_processed': examples, 'optimizer_steps': steps,
                   'max_mapreduce_abs': map_gap, 'received_mapreduce_abs': received_gap,
                   'group_flower_abs': group_gap, 'central_flower_abs': central_gap,
                   **wire.finish(int(meta['metadata_bytes']))}
            if evaluate_validation:
                probabilities = base.predict(arrays, data['x'][data['val']])
                loss = float(log_loss(data['y'][data['val']], probabilities, labels=np.arange(data['classes'])))
                validation.append(probabilities.copy())
                row.update(val_log_loss=loss, val_macro_f1=float(f1_score(data['y'][data['val']], probabilities.argmax(1), average='macro', zero_division=0)))
                if loss < best_loss:
                    best_loss, best_round, best = loss, rnd, [a.copy() for a in arrays]
            history.append(row)
    if best is None:
        best, best_round = arrays, rounds
    keys = ('total_bytes', 'model_wire_bytes', 'model_payload_bytes', 'model_protocol_header_bytes',
            'up_bytes', 'down_bytes', 'graph_control_both_hops_bytes', 'codec_deployment_bytes',
            'transformer_frames', 'transformer_frame_bytes', 'int8_frames',
            'reconstruction_error_squared', 'raw_update_energy', 'examples_processed', 'optimizer_steps', 'lost_unique_records')
    drift_recall = [r['dedup_recall'] for r in history
                    if r['phase'] == 'drift' and r['dedup_recall'] is not None]
    result = dict(status='completed', dataset=dataset, seed=seed, rounds=rounds, codec=codec, dynamic=dynamic,
                  selected_round=best_round, selected_val_log_loss=best_loss if evaluate_validation else None,
                  selected_val_macro_f1=history[best_round-1]['val_macro_f1'] if evaluate_validation else None,
                  host_process_cpu_seconds=time.process_time()-started_cpu, host_elapsed_seconds=time.perf_counter()-started_wall,
                  graphsage_fits=len(policy.history), graphsage_parameter_count=policy.history[0]['learned_parameter_count'],
                  post_drift_dedup_recall=float(np.mean(drift_recall)) if drift_recall else None,
                  post_drift_dedup_recall_defined_rounds=len(drift_recall),
                  fixed_split_sha256=data['fixed_split_sha256'], scenario_sha256=data['scenario_sha256'],
                  received_updates_used_by_flower=True, separate_receiver_objects=True, common_error_feedback=True,
                  test_evaluated=allow_test, **{f'sum_{key}': sum(r[key] for r in history) for key in keys})
    if allow_test:
        probabilities, final_probabilities = base.predict(best, data['x'][data['test']]), base.predict(arrays, data['x'][data['test']])
        result.update(test_accuracy=float(np.mean(probabilities.argmax(1)==data['y'][data['test']])),
                      test_macro_f1=float(f1_score(data['y'][data['test']], probabilities.argmax(1), average='macro', zero_division=0)),
                      final_test_macro_f1=float(f1_score(data['y'][data['test']], final_probabilities.argmax(1), average='macro', zero_division=0)))
        np.savez_compressed(out/'test_predictions.npz', source_ids=data['test'], y=data['y'][data['test']], probabilities=probabilities, final_probabilities=final_probabilities)
    if evaluate_validation:
        np.savez_compressed(out/'validation_predictions.npz', source_ids=data['val'], y=data['y'][data['val']], probabilities=np.stack(validation))
    for name, model in [('selected_model.npz', best), ('final_model.npz', arrays)]:
        np.savez_compressed(out/name, **{f'array_{i}': a for i, a in enumerate(model)})
    np.savez_compressed(out/'retained_masks.npz', **masks_saved)
    np.savez_compressed(out/'codec_samples.npz', **wire.sample_arrays)
    for name, value in [('codec_samples.json', wire.samples), ('codec_deployment_audit.json', wire.deployment),
                        ('model_wire_audit.json', wire.ledger), ('wire_audit.json', controls), ('graph_history.json', policy.history)]:
        base.json_write(out/name, value)
    if hashes != _source_hashes():
        raise RuntimeError('Source changed during execution; retain run for inspection')
    base.csv_write(out/'rounds.csv', history)
    base.json_write(out/'final.json', result)
    return result
