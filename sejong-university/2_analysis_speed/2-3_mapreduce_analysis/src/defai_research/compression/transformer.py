from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import torch
from .model import (
    ARTIFACT_HEADER, TransformerCodecConfig, _read_artifact, _normalize,
    _endpoint_pair,
)
from .lossless import (
    HEADER, MODES, XOR_MODES, SHUFFLE_MODES, CacheEntry,
    _channel, _uint, _crc, byte_shuffle4, byte_unshuffle4,
)
from ._wire import _inflate, _xor

MAGIC = b'TFC1'
VERSION = 1
CODEC_ID = 2
BODY_HEADER = struct.Struct('<8sIBBBB')
DEPLOY_HEADER = struct.Struct('<4sBBHHII32s')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CompactTransformerTransfer:
    frame: bytes
    decoded_arrays: list
    audit: dict

    @property
    def frame_bytes(self):
        return len(self.frame)

    @property
    def mode(self):
        return self.audit['mode']

    def evidence(self, include_frame=False):
        result = dict(self.audit)
        if include_frame:
            result['frame_hex'] = self.frame.hex()
        return result


class TransformerCompactCodec:
    def __init__(self, artifact=None, *, error_feedback=False, residual_topk=8,
                 residual_bits=8, latent_enabled=True, block_gain=True,
                 compression_level=6, max_raw_bytes=128 * 1024 * 1024):
        self.residual_topk = _uint(residual_topk, 'residual_topk', 32)
        self.residual_bits = _uint(residual_bits, 'residual_bits', 16)
        if self.residual_bits not in (8, 16):
            raise ValueError('residual_bits must be 8 or 16')
        if not isinstance(latent_enabled, bool) or not isinstance(block_gain, bool):
            raise TypeError('latent_enabled and block_gain must be bool')
        self.latent_enabled = latent_enabled
        self.block_gain = block_gain and latent_enabled
        self.error_feedback = bool(error_feedback)
        self.compression_level = _uint(compression_level, 'compression_level', 9)
        self.max_raw_bytes = _uint(max_raw_bytes, 'max_raw_bytes')
        if self.compression_level < 1 or self.max_raw_bytes < BODY_HEADER.size:
            raise ValueError('Invalid codec limits')
        self.artifact_path = (str(Path(artifact).resolve())
                              if isinstance(artifact, (str, Path)) else None)
        self.artifact = (artifact if isinstance(artifact, bytes) else
                         Path(artifact).read_bytes()) if self.latent_enabled else b''
        if self.latent_enabled:
            if len(self.artifact) < ARTIFACT_HEADER.size:
                raise ValueError('Complete artifact required')
            magic, metadata_bytes, weight_bytes = ARTIFACT_HEADER.unpack_from(self.artifact)
            if magic != b'TCA1' or len(self.artifact) != ARTIFACT_HEADER.size + metadata_bytes + weight_bytes:
                raise ValueError('Invalid source artifact lengths')
            end = ARTIFACT_HEADER.size + metadata_bytes
            self.document = json.loads(self.artifact[ARTIFACT_HEADER.size:end])
            if _sha(self.artifact[end:]) != self.document['weights_sha256']:
                raise ValueError('Source weight archive checksum mismatch')
            self.config = TransformerCodecConfig(**self.document['config'])
        else:
            self.document = dict(parameter_count=0)
            self.config = TransformerCodecConfig()
        self.config.validate()
        self.source_artifact_sha256 = _sha(self.artifact) if self.latent_enabled else None
        self.model_id = self.source_artifact_sha256
        self._model_tag = bytes.fromhex(self.model_id)[:8] if self.model_id else bytes(8)
        self._models, self._endpoint_artifacts = {}, {}
        if self.latent_enabled:
            model, _ = _read_artifact(self.artifact)
            self._models[('central', 0)] = model
            self._endpoint_artifacts[('central', 0)] = self.artifact
        self._tx, self._rx, self._residuals = {}, {}, {}
        self._totals = dict(frames=0, frame_bytes=0, deployment_packets=0,
                            deployment_payload_bytes=0, deployment_network_bytes=0)

    @classmethod
    def from_artifact(cls, artifact, **kwargs):
        return cls(artifact, **kwargs)

    @staticmethod
    def _parse_deployment(payload):
        if not isinstance(payload, bytes) or len(payload) < DEPLOY_HEADER.size:
            raise ValueError('Complete immutable deployment packet required')
        magic, version, link, sender, receiver, raw_size, packed_size, digest = DEPLOY_HEADER.unpack_from(payload)
        if (magic != b'TCD2' or version != VERSION or link not in (2, 3)
                or not ARTIFACT_HEADER.size <= raw_size <= 16 * 1024 * 1024
                or len(payload) != DEPLOY_HEADER.size + packed_size
                or link == 2 and sender != 0):
            raise ValueError('Invalid compressed model deployment schema')
        restored = _inflate(payload[DEPLOY_HEADER.size:], raw_size)
        if len(restored) != raw_size or _sha(restored) != digest.hex():
            raise ValueError('Deployment artifact SHA or length mismatch')
        source = ('central', 0) if link == 2 else ('group', sender)
        endpoint = ('group', receiver) if link == 2 else ('client', receiver)
        return restored, source, endpoint

    def receive_deployment(self, payload):
        if not self.latent_enabled:
            raise ValueError('Mean control has no learned-model deployment')
        restored, source, endpoint = self._parse_deployment(payload)
        if _sha(restored) != self.model_id:
            raise ValueError('Received model differs from this codec identity')
        if endpoint in self._endpoint_artifacts:
            raise ValueError('Deployment endpoint already received its weights')
        model, document = _read_artifact(restored)
        if document['config'] != self.document['config']:
            raise ValueError('Received architecture differs from codec configuration')
        self._models[endpoint] = model
        self._endpoint_artifacts[endpoint] = restored
        return dict(source=list(source), destination=list(endpoint), model_id=self.model_id,
                    artifact_bytes=len(restored), payload_bytes=len(payload))

    @classmethod
    def receiver_from_deployment(cls, payload, **kwargs):
        restored, _, endpoint = cls._parse_deployment(payload)
        if kwargs.get('latent_enabled') is False:
            raise ValueError('A mean receiver must not receive neural weights')
        result = cls(restored, **kwargs)
        model = result._models.pop(('central', 0))
        result._endpoint_artifacts.clear()
        result._models[endpoint] = model
        result._endpoint_artifacts[endpoint] = restored
        return result

    def deployment_packets(self, *, client_ids=range(12), group_ids=range(4), client_groups=None):
        if not self.latent_enabled:
            return []
        groups = [_uint(x, 'group_id', 65535) for x in group_ids]
        clients = [_uint(x, 'client_id', 65535) for x in client_ids]
        if len(set(groups)) != len(groups) or len(set(clients)) != len(clients):
            raise ValueError('Duplicate deployment endpoint')
        if clients and not groups:
            raise ValueError('Clients require a receiving relay group')
        assignments = (dict(client_groups) if client_groups is not None else
                       {cid: groups[min(i * len(groups) // max(len(clients), 1), len(groups) - 1)]
                        for i, cid in enumerate(clients)})
        if any(cid not in assignments or assignments[cid] not in groups for cid in clients):
            raise ValueError('Client relay assignment is absent or not deployed')
        routes = [('cloud', ('central', 0), ('group', gid)) for gid in groups]
        routes += [('edge', ('group', int(assignments[cid])), ('client', cid)) for cid in clients]
        result = []
        for hop, source, receiver in routes:
            if receiver in self._models:
                continue
            if source not in self._endpoint_artifacts:
                raise RuntimeError('Relay must first receive the exact artifact it forwards')
            source_artifact = self._endpoint_artifacts[source]
            packed = zlib.compress(source_artifact, self.compression_level)
            link = 2 if hop == 'cloud' else 3
            payload = DEPLOY_HEADER.pack(b'TCD2', VERSION, link, source[1], receiver[1],
                                         len(source_artifact), len(packed), bytes.fromhex(self.model_id)) + packed
            magic, ver, actual_link, sender_id, receiver_id, raw_size, packed_size, digest = DEPLOY_HEADER.unpack_from(payload)
            restored = _inflate(payload[DEPLOY_HEADER.size:], raw_size)
            if (magic != b'TCD2' or ver != VERSION or actual_link != link
                    or sender_id != source[1] or receiver_id != receiver[1]
                    or len(payload) != DEPLOY_HEADER.size + packed_size
                    or digest.hex() != self.model_id or _sha(restored) != self.model_id
                    or restored != source_artifact):
                raise AssertionError('Deployment did not reconstruct exact immutable weights')
            model, _ = _read_artifact(restored)
            self._models[receiver] = model
            self._endpoint_artifacts[receiver] = restored
            result.append(dict(kind='transformer_model_deployment', hop=hop,
                               source=list(source), destination=list(receiver), endpoint=list(receiver),
                               payload=payload, hop_count=1, model_id=self.model_id,
                               artifact_bytes=len(restored), packed_artifact_bytes=len(packed),
                               deployment_header_bytes=DEPLOY_HEADER.size,
                               artifact_compression='zlib_lossless'))
            self._totals['deployment_packets'] += 1
            self._totals['deployment_payload_bytes'] += len(payload)
            self._totals['deployment_network_bytes'] += len(payload)
        return result

    def _prediction(self, endpoint, codes, latent_scales, gains, count):
        if not self.latent_enabled:
            return np.zeros((count, 256), np.float32)
        if endpoint not in self._models:
            raise RuntimeError('Deploy actual decoder weights before sending or receiving')
        latent = codes.astype(np.float32) * latent_scales[:, None, None]
        with torch.inference_mode():
            base = self._models[endpoint].decode(torch.from_numpy(latent)).numpy().copy()
        if self.block_gain:
            base *= gains.astype(np.float32)[:, None]
        return base

    def _decode_body(self, raw, receiver):
        if len(raw) < BODY_HEADER.size:
            raise ValueError('Truncated Transformer body')
        tag, values, tokens, dim, topk, flags = BODY_HEADER.unpack_from(raw)
        expected_flags = ((self.residual_bits == 16) | (int(self.block_gain) << 1)
                          | (int(self.latent_enabled) << 2) | 8)
        expected_tokens = self.config.latent_tokens if self.latent_enabled else 0
        expected_dim = self.config.latent_dim if self.latent_enabled else 0
        if (tag != self._model_tag or tokens != expected_tokens or dim != expected_dim
                or topk != self.residual_topk or flags != expected_flags or not values):
            raise ValueError('Received body differs from deployed codec schema')
        count = (values + 255) // 256
        per_block = 8 + (4 + tokens * dim + (2 if self.block_gain else 0) if self.latent_enabled else 0)
        per_block += 4 + topk * (1 + self.residual_bits // 8) if topk else 0
        if len(raw) != BODY_HEADER.size + count * per_block:
            raise ValueError('Transformer block count disagrees with exact body length')
        cursor = BODY_HEADER.size
        def take(dtype, number):
            nonlocal cursor
            array = np.frombuffer(raw, dtype=dtype, count=number, offset=cursor).copy()
            cursor += array.nbytes
            return array
        means, scales = take('<f4', count), take('<f4', count)
        latent_scales = codes = gains = None
        if self.latent_enabled:
            latent_scales = take('<f4', count)
            codes = take('i1', count * tokens * dim).reshape(count, tokens, dim)
            if self.block_gain:
                gains = take('<f2', count)
        residual_scales = indices = residual_values = None
        if topk:
            residual_scales = take('<f4', count)
            delta_indices = take('u1', count * topk).reshape(count, topk)
            indices = np.cumsum(delta_indices.astype(np.int64), axis=1)
            residual_values = take('i1' if self.residual_bits == 8 else '<f2', count * topk).reshape(count, topk)
            if (np.any(indices > 255) or topk > 1 and np.any(delta_indices[:, 1:] == 0)
                    or not np.isfinite(residual_scales).all() or np.any(residual_scales <= 0)
                    or not np.isfinite(residual_values).all()):
                raise ValueError('Invalid sparse residual values or canonical indices')
        if (cursor != len(raw) or not np.isfinite(means).all() or not np.isfinite(scales).all()
                or np.any(scales < 0) or self.latent_enabled
                and (not np.isfinite(latent_scales).all() or np.any(latent_scales <= 0))
                or self.block_gain and (not np.isfinite(gains).all() or np.any((gains < 0) | (gains > 2)))):
            raise ValueError('Invalid normalization, latent scale or gain')
        normalized = self._prediction(receiver, codes, latent_scales, gains, count)
        if topk:
            normalized[np.arange(count)[:, None], indices] += residual_values.astype(np.float32) * residual_scales[:, None]
        reconstructed = (normalized * scales[:, None] + means[:, None]).reshape(-1)[:values]
        if not np.isfinite(reconstructed).all():
            raise ValueError('Nonfinite reconstructed Transformer update')
        return reconstructed.astype(np.float32), dict(values=values, blocks=count,
                latent_tokens=tokens, latent_dim=dim, residual_topk=topk,
                residual_bits=self.residual_bits, block_gain=self.block_gain,
                latent_enabled=self.latent_enabled)

    def decode_frame(self, frame, *, channel=None, round_id=None,
                     expected_channel=None, expected_round=None):
        channel = channel if channel is not None else expected_channel
        round_id = round_id if round_id is not None else expected_round
        key, rnd = _channel(channel), _uint(round_id, 'round_id')
        _, receiver = _endpoint_pair(channel)
        if self.latent_enabled and receiver not in self._models:
            raise RuntimeError('Explicit decoder deployment required before receiving')
        if not isinstance(frame, bytes) or len(frame) < HEADER.size:
            raise ValueError('Complete immutable compact Transformer frame required')
        magic, version, codec, mode, link, sender, destination, actual_round, seq, raw_size, packed_size, crc, base_seq, base_crc = HEADER.unpack_from(frame)
        if magic != MAGIC or version != VERSION or codec != CODEC_ID or mode not in MODES:
            raise ValueError('Unknown compact Transformer frame schema')
        if (link, sender, destination) != key or actual_round != rnd:
            raise ValueError('Receiver channel or round mismatch')
        if len(frame) != HEADER.size + packed_size or not BODY_HEADER.size <= raw_size <= self.max_raw_bytes:
            raise ValueError('Invalid frame sizes')
        previous = self._rx.get(key)
        if seq != (previous.sequence + 1 if previous else 1) or previous is not None and rnd <= previous.round_id:
            raise ValueError('Stale or out-of-order frame')
        if mode in XOR_MODES:
            if (previous is None or base_seq != previous.sequence or base_crc != previous.crc
                    or raw_size != len(previous.raw)):
                raise ValueError('Missing or mismatched synchronized XOR history')
        elif base_seq or base_crc:
            raise ValueError('Independent frame declares a history dependency')
        payload = frame[HEADER.size:]
        raw = payload if mode == 0 else _inflate(payload, raw_size)
        if mode in SHUFFLE_MODES:
            raw = byte_unshuffle4(raw)
        if mode in XOR_MODES:
            raw = _xor(raw, previous.raw)
        if len(raw) != raw_size or _crc(raw) != crc:
            raise ValueError('Restored compact body fails CRC or size verification')
        decoded, metadata = self._decode_body(raw, receiver)
        self._rx[key] = CacheEntry(seq, rnd, raw, crc)
        return [decoded], dict(metadata, model_id=self.model_id, sequence=seq,
                               round=rnd, channel=list(channel), transport_mode=MODES[mode],
                               raw_crc32=crc, raw_body_sha256=_sha(raw))

    def transmit(self, arrays, *, channel, round_id):
        arrays = list(arrays)
        if len(arrays) != 1:
            raise ValueError('Compact update schema requires exactly one flat FP32 array')
        source = np.asarray(arrays[0])
        if source.dtype != np.dtype(np.float32) or source.ndim != 1 or not source.size or not np.isfinite(source).all():
            raise ValueError('Compact update must be one nonempty finite flat FP32 array')
        key, rnd = _channel(channel), _uint(round_id, 'round_id')
        sender, receiver = _endpoint_pair(channel)
        if self.latent_enabled and (sender not in self._models or receiver not in self._models):
            raise RuntimeError('Call deployment_packets; no hidden bootstrap is allowed')
        previous = self._tx.get(key)
        if previous is not None and rnd <= previous.round_id:
            raise ValueError('Directed update channel requires increasing rounds')
        old_rx = self._rx.get(key)
        prior = self._residuals.get(key)
        residual_reset = prior is not None and prior.shape != source.shape
        compensated = source.copy()
        if self.error_feedback and prior is not None and not residual_reset:
            compensated += prior
        count = (len(source) + 255) // 256
        blocks = np.zeros((count, 256), np.float32)
        blocks.reshape(-1)[:len(source)] = compensated
        lengths = np.full(count, 256, np.int64)
        lengths[-1] = len(source) - 256 * (count - 1)
        normalized, means, scales, valid = _normalize(blocks, lengths)
        codes = latent_scales = gains = None
        parts = [means.astype('<f4').tobytes(), scales.astype('<f4').tobytes()]
        if self.latent_enabled:
            with torch.inference_mode():
                latent = self._models[sender].encode(torch.from_numpy(normalized)).numpy().copy()
            latent_scales = np.maximum(np.max(np.abs(latent), axis=(1, 2)), 1e-12).astype(np.float32) / 127
            codes = np.clip(np.rint(latent / latent_scales[:, None, None]), -127, 127).astype(np.int8)
            gains = np.ones(count, dtype='<f2') if self.block_gain else None
            prediction = self._prediction(sender, codes, latent_scales, gains, count)
            if self.block_gain:
                numerator = np.where(valid, normalized * prediction, 0).sum(1, dtype=np.float64)
                denominator = np.where(valid, prediction * prediction, 0).sum(1, dtype=np.float64)
                gains = np.clip(numerator / np.maximum(denominator, 1e-12), 0, 2).astype('<f2')
                prediction *= gains.astype(np.float32)[:, None]
            parts += [latent_scales.astype('<f4').tobytes(), codes.tobytes()]
            if self.block_gain:
                parts.append(gains.tobytes())
        else:
            prediction = np.zeros((count, 256), np.float32)
        if self.residual_topk:
            difference = np.where(valid, normalized - prediction, 0)
            indices = np.argsort(-np.abs(difference), axis=1, kind='stable')[:, :self.residual_topk]
            indices.sort(axis=1)
            selected = np.take_along_axis(difference, indices, axis=1)
            maximum = np.maximum(np.max(np.abs(selected), axis=1), 1e-12).astype(np.float32)
            if self.residual_bits == 8:
                residual_scales = maximum / 127
                residual_values = np.clip(np.rint(selected / residual_scales[:, None]), -127, 127).astype(np.int8)
            else:
                residual_scales = maximum
                residual_values = (selected / residual_scales[:, None]).astype('<f2')
            delta_indices = np.diff(indices, axis=1, prepend=np.zeros((count, 1), np.int64)).astype(np.uint8)
            parts += [residual_scales.astype('<f4').tobytes(), delta_indices.tobytes(), residual_values.tobytes()]
        tokens = self.config.latent_tokens if self.latent_enabled else 0
        dim = self.config.latent_dim if self.latent_enabled else 0
        flags = ((self.residual_bits == 16) | (int(self.block_gain) << 1)
                 | (int(self.latent_enabled) << 2) | 8)
        raw = BODY_HEADER.pack(self._model_tag, _uint(len(source), 'values'), tokens, dim,
                               self.residual_topk, flags) + b''.join(parts)
        if len(raw) > self.max_raw_bytes:
            raise ValueError('Coded update body exceeds max_raw_bytes')
        local, _ = self._decode_body(raw, sender)
        candidates = [(0, raw), (1, zlib.compress(raw, self.compression_level)),
                      (3, zlib.compress(byte_shuffle4(raw), self.compression_level))]
        if previous is not None and len(raw) == len(previous.raw):
            difference = _xor(raw, previous.raw)
            candidates += [(2, zlib.compress(difference, self.compression_level)),
                           (4, zlib.compress(byte_shuffle4(difference), self.compression_level))]
        mode, payload = min(candidates, key=lambda item: (len(item[1]), item[0]))
        seq = _uint(previous.sequence + 1 if previous else 1, 'sequence')
        base_seq = previous.sequence if mode in XOR_MODES else 0
        base_crc = previous.crc if mode in XOR_MODES else 0
        crc = _crc(raw)
        frame = HEADER.pack(MAGIC, VERSION, CODEC_ID, mode, *key, rnd, seq,
                            len(raw), len(payload), crc, base_seq, base_crc) + payload
        try:
            decoded, received_header = self.decode_frame(frame, channel=channel, round_id=rnd)
            if (not np.array_equal(local, decoded[0]) or received_header['raw_body_sha256'] != _sha(raw)
                    or self._rx[key].raw != raw):
                raise AssertionError('Sender reconstruction differs from independent packet receiver')
        except Exception:
            if old_rx is None:
                self._rx.pop(key, None)
            else:
                self._rx[key] = old_rx
            raise
        self._tx[key] = CacheEntry(seq, rnd, raw, crc)
        if self.error_feedback:
            self._residuals[key] = (compensated - local).astype(np.float32)
        self._totals['frames'] += 1
        self._totals['frame_bytes'] += len(frame)
        audit = dict(mode='transformer' if self.latent_enabled else 'mean_residual',
            codec='compact_transformer' if self.latent_enabled else 'compact_mean_residual',
            codec_id=CODEC_ID, transport_mode=MODES[mode], model_id=self.model_id,
            source_artifact_sha256=self.source_artifact_sha256 if self.latent_enabled else None,
            round=rnd, sequence=seq, channel=list(channel), sender=list(sender), receiver=list(receiver),
            frame_bytes=len(frame), header_bytes=HEADER.size, payload_bytes=len(payload),
            raw_container_bytes=len(raw), frame_sha256=_sha(frame), raw_body_sha256=_sha(raw),
            raw_crc32=crc, base_sequence=base_seq, base_crc32=base_crc,
            candidate_frame_bytes={MODES[m]: HEADER.size + len(p) for m, p in candidates},
            blocks=count, latent_values=count * tokens * dim, latent_tokens=tokens, latent_dim=dim,
            latent_enabled=self.latent_enabled, residual_topk=self.residual_topk,
            residual_bits=self.residual_bits, block_gain=self.block_gain,
            block_gain_min=float(np.min(gains)) if gains is not None else None,
            block_gain_max=float(np.max(gains)) if gains is not None else None,
            error_feedback=self.error_feedback, error_feedback_reset_on_shape_change=residual_reset,
            raw_FP32_update_bytes=source.nbytes, parameter_count=self.document['parameter_count'] if self.latent_enabled else 0,
            raw_input_norm=float(np.linalg.norm(source)),
            relative_l2_error_vs_compensated=float(np.linalg.norm(compensated-local) / max(np.linalg.norm(compensated), 1e-12)),
            relative_l2_error_vs_current_update=float(np.linalg.norm(source-local) / max(np.linalg.norm(source), 1e-12)),
            maximum_abs_error_vs_compensated=float(np.max(np.abs(compensated-local))),
            receiver_uses_only_packet_and_deployed_decoder=True,
            sender_local_decoder_matches_independent_receiver=True,
            bootstrap_bytes_in_frame=0,
            scope='40-byte application frame; actual lossy Transformer or mean residual body; explicit deployment separate; no raw update fallback')
        return CompactTransformerTransfer(frame, decoded, audit)

    def accounting(self):
        return dict(self._totals, model_id=self.model_id,
                    artifact_bytes=len(self.artifact) if self.latent_enabled else 0,
                    deployed_endpoints=len(self._models),
                    error_feedback_channels=len(self._residuals),
                    tx_channels=len(self._tx), rx_channels=len(self._rx),
                    parameter_count=self.document['parameter_count'] if self.latent_enabled else 0)

