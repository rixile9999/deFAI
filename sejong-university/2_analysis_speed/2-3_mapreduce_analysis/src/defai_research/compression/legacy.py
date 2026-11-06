from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import struct
from typing import Any
import zlib
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from ._wire import _inflate, _xor

MAGIC = b'GWC1'
VERSION = 1
MODES = {0:'raw',1:'zlib',2:'xor_zlib'}
MODE_IDS = {value:key for key,value in MODES.items()}
HEADER = struct.Struct('<4sBBHQQQQQ32s32s')
COUNT = struct.Struct('<I')
LENGTH = struct.Struct('<Q')
ZERO_SHA256 = bytes(32)

def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _channel_bytes(channel: str | tuple | list) -> bytes:
    if isinstance(channel, str):
        value: Any = channel
    elif isinstance(channel, (tuple, list)) and channel and all(
            isinstance(part, (str, int)) and not isinstance(part, bool) for part in channel):
        value = list(channel)
    else:
        raise TypeError('Channel must be a nonempty string or a tuple/list of string/integer link identifiers')
    if value == '':
        raise ValueError('Channel must be nonempty')
    encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > 65535:
        raise ValueError('Channel header exceeds uint16 length')
    return encoded


def _pack_tensors(tensors: list[bytes]) -> bytes:
    return COUNT.pack(len(tensors)) + b''.join(LENGTH.pack(len(tensor)) + tensor for tensor in tensors)


def _unpack_tensors(raw: bytes) -> list[bytes]:
    if len(raw) < COUNT.size:
        raise ValueError('Truncated tensor container')
    count, = COUNT.unpack_from(raw)
    if count > (len(raw) - COUNT.size) // LENGTH.size:
        raise ValueError('Impossible tensor count')
    offset = COUNT.size
    tensors = []
    for _ in range(count):
        if offset + LENGTH.size > len(raw):
            raise ValueError('Truncated tensor length')
        size, = LENGTH.unpack_from(raw, offset)
        offset += LENGTH.size
        if size > len(raw) - offset:
            raise ValueError('Truncated NPY tensor')
        tensors.append(raw[offset:offset + size])
        offset += size
    if offset != len(raw):
        raise ValueError('Unexpected trailing container bytes')
    return tensors


@dataclass(frozen=True)
class _CacheEntry:
    sequence: int
    round_id: int
    raw: bytes
    digest: bytes


@dataclass(frozen=True)
class EncodedFrame:
    frame: bytes
    channel_bytes: bytes
    round_id: int
    sequence: int
    mode: str
    raw_bytes: int
    flower_tensor_bytes: int
    tensor_count: int
    payload_bytes: int
    header_bytes: int
    base_sequence: int
    raw_sha256: str
    base_sha256: str | None
    candidate_frame_bytes: dict[str, int]
    _raw: bytes = dataclass_field(repr=False)

    @property
    def frame_bytes(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class DecodedFrame:
    channel_bytes: bytes
    round_id: int
    sequence: int
    frame_sha256: str
    raw_sha256: str
    arrays: list[np.ndarray] = dataclass_field(repr=False)
    tensors: list[bytes] = dataclass_field(repr=False)
    _raw: bytes = dataclass_field(repr=False)


@dataclass(frozen=True)
class WireTransfer:
    encoded: EncodedFrame
    decoded_arrays: list[np.ndarray] = dataclass_field(repr=False)

    @property
    def frame(self) -> bytes:
        return self.encoded.frame

    @property
    def frame_bytes(self) -> int:
        return self.encoded.frame_bytes

    @property
    def mode(self) -> str:
        return self.encoded.mode

    @property
    def raw_bytes(self) -> int:
        return self.encoded.raw_bytes

    @property
    def payload_bytes(self) -> int:
        return self.encoded.payload_bytes

    @property
    def header_bytes(self) -> int:
        return self.encoded.header_bytes

    @property
    def raw_sha256(self) -> str:
        return self.encoded.raw_sha256

    @property
    def flower_tensor_bytes(self) -> int:
        return self.encoded.flower_tensor_bytes

    def evidence(self, include_frame: bool = False) -> dict[str, Any]:
        encoded = self.encoded
        item = {
            'channel': json.loads(encoded.channel_bytes),
            'round': encoded.round_id, 'sequence': encoded.sequence,
            'mode': encoded.mode, 'flower_tensor_count': encoded.tensor_count,
            'flower_tensor_bytes': encoded.flower_tensor_bytes,
            'tensor_container_overhead_bytes': encoded.raw_bytes - encoded.flower_tensor_bytes,
            'raw_container_bytes': encoded.raw_bytes,
            'header_bytes': encoded.header_bytes, 'payload_bytes': encoded.payload_bytes,
            'frame_bytes': encoded.frame_bytes, 'base_sequence': encoded.base_sequence,
            'raw_sha256': encoded.raw_sha256, 'base_sha256': encoded.base_sha256,
            'frame_sha256': _sha(encoded.frame).hex(),
            'candidate_frame_bytes': dict(encoded.candidate_frame_bytes),
            'decoded_byte_identical': True,
            'scope': 'actual application frame buffers; NPY bytes exact; physical network excluded',
        }
        if include_frame:
            item['frame_hex'] = encoded.frame.hex()
        return item


class LegacyLosslessCodec:
    def __init__(self, *, compression_level: int = 6, max_raw_bytes: int = 128 * 1024 * 1024):
        if compression_level not in range(1, 10):
            raise ValueError('compression_level must be between 1 and 9')
        if max_raw_bytes < 4:
            raise ValueError('max_raw_bytes must permit the tensor-count header')
        self.compression_level = int(compression_level)
        self.max_raw_bytes = int(max_raw_bytes)
        self._tx: dict[bytes, _CacheEntry] = {}
        self._rx: dict[bytes, _CacheEntry] = {}
        self._totals = defaultdict(int)

    def encode_arrays(self, arrays, *, channel, round_id: int) -> EncodedFrame:
        key = _channel_bytes(channel)
        if not isinstance(round_id, int) or isinstance(round_id, bool) or round_id < 0:
            raise ValueError('round_id must be a nonnegative integer')
        previous = self._tx.get(key)
        if previous is not None and round_id <= previous.round_id:
            raise ValueError('A directed channel requires strictly increasing round numbers')
        parameters = ndarrays_to_parameters(list(arrays))
        if parameters.tensor_type != 'numpy.ndarray':
            raise ValueError('This codec accepts Flower NumPy serialization only')
        tensors = list(parameters.tensors)
        raw = _pack_tensors(tensors)
        if len(raw) > self.max_raw_bytes:
            raise ValueError('Serialized tensors exceed max_raw_bytes')
        sequence = 1 if previous is None else previous.sequence + 1
        candidates = [('raw', raw), ('zlib', zlib.compress(raw, self.compression_level))]
        if previous is not None and len(previous.raw) == len(raw):
            candidates.append(('xor_zlib', zlib.compress(_xor(raw, previous.raw), self.compression_level)))
        header_size = HEADER.size + len(key)
        mode, payload = min(candidates, key=lambda item: header_size + len(item[1]))
        base_sequence = previous.sequence if mode == 'xor_zlib' else 0
        base_digest = previous.digest if mode == 'xor_zlib' else ZERO_SHA256
        raw_digest = _sha(raw)
        frame = HEADER.pack(MAGIC, VERSION, MODE_IDS[mode], len(key), round_id, sequence,
                            len(raw), len(payload), base_sequence, raw_digest, base_digest) + key + payload
        return EncodedFrame(
            frame, key, round_id, sequence, mode, len(raw), sum(map(len, tensors)), len(tensors),
            len(payload), header_size, base_sequence, raw_digest.hex(),
            base_digest.hex() if mode == 'xor_zlib' else None,
            {name: header_size + len(value) for name, value in candidates}, raw,
        )

    def decode_frame(self, frame: bytes, *, expected_channel, expected_round: int) -> DecodedFrame:
        key = _channel_bytes(expected_channel)
        if not isinstance(frame, bytes) or len(frame) < HEADER.size:
            raise ValueError('A complete immutable frame is required')
        magic, version, mode_id, key_length, rnd, sequence, raw_size, packed_size, base_seq, raw_hash, base_hash = HEADER.unpack_from(frame)
        if magic != MAGIC or version != VERSION or mode_id not in MODES:
            raise ValueError('Unsupported frame magic, version or codec mode')
        header_size = HEADER.size + key_length
        if header_size + packed_size != len(frame):
            raise ValueError('Frame length differs from its declared header/payload lengths')
        if frame[HEADER.size:header_size] != key or rnd != expected_round:
            raise ValueError('Receiver channel or round does not match the frame')
        if raw_size < COUNT.size or raw_size > self.max_raw_bytes:
            raise ValueError('Decoded size is outside configured bounds')
        previous = self._rx.get(key)
        expected_sequence = 1 if previous is None else previous.sequence + 1
        if sequence != expected_sequence or (previous is not None and rnd <= previous.round_id):
            raise ValueError('Receiver sequence or round is stale, replayed or out of order')
        mode = MODES[mode_id]
        payload = frame[header_size:]
        if mode == 'raw':
            raw = payload
        else:
            raw = _inflate(payload, raw_size)
        if mode == 'xor_zlib':
            if (previous is None or previous.sequence != base_seq
                    or previous.digest != base_hash or len(previous.raw) != raw_size):
                raise ValueError('Previous-frame receiver cache identity mismatch')
            raw = _xor(raw, previous.raw)
        elif base_seq != 0 or base_hash != ZERO_SHA256:
            raise ValueError('Independent frames must not declare an XOR base')
        if len(raw) != raw_size or _sha(raw) != raw_hash:
            raise ValueError('Restored tensor container length or SHA256 mismatch')
        tensors = _unpack_tensors(raw)
        arrays = parameters_to_ndarrays(Parameters(tensors=tensors, tensor_type='numpy.ndarray'))
        if list(ndarrays_to_parameters(arrays).tensors) != tensors:
            raise ValueError('Flower NPY re-serialization differs from the received buffers')
        result = DecodedFrame(key, rnd, sequence, _sha(frame).hex(), raw_hash.hex(), arrays, tensors, raw)
        self._rx[key] = _CacheEntry(sequence, rnd, raw, raw_hash)
        return result

    def confirm_delivery(self, encoded: EncodedFrame, decoded: DecodedFrame) -> None:
        previous = self._tx.get(encoded.channel_bytes)
        next_sequence = 1 if previous is None else previous.sequence + 1
        if (encoded.sequence != next_sequence
                or (previous is not None and encoded.round_id <= previous.round_id)
                or decoded.channel_bytes != encoded.channel_bytes
                or decoded.round_id != encoded.round_id or decoded.sequence != encoded.sequence
                or decoded.frame_sha256 != _sha(encoded.frame).hex()
                or decoded.raw_sha256 != encoded.raw_sha256 or decoded._raw != encoded._raw):
            raise ValueError('Delivery confirmation does not match the pending frame')
        self._tx[encoded.channel_bytes] = _CacheEntry(encoded.sequence, encoded.round_id,
                                                     encoded._raw, bytes.fromhex(encoded.raw_sha256))

    def transmit(self, arrays, *, channel, round_id: int) -> WireTransfer:
        key = _channel_bytes(channel)
        old_tx, old_rx = self._tx.get(key), self._rx.get(key)
        try:
            encoded = self.encode_arrays(arrays, channel=channel, round_id=round_id)
            decoded = self.decode_frame(encoded.frame, expected_channel=channel, expected_round=round_id)
            self.confirm_delivery(encoded, decoded)
        except Exception:
            for cache, old in ((self._tx, old_tx), (self._rx, old_rx)):
                if old is None:
                    cache.pop(key, None)
                else:
                    cache[key] = old
            raise
        transfer = WireTransfer(encoded, decoded.arrays)
        self._totals['frames'] += 1
        self._totals[f'{encoded.mode}_frames'] += 1
        for name in ('frame_bytes', 'header_bytes', 'payload_bytes', 'raw_bytes', 'flower_tensor_bytes'):
            self._totals[name] += getattr(encoded, name)
        return transfer

    def accounting(self) -> dict[str, Any]:
        return {**dict(self._totals), 'tx_channels': len(self._tx), 'rx_channels': len(self._rx),
                'tx_cached_raw_bytes': sum(len(x.raw) for x in self._tx.values()),
                'rx_cached_raw_bytes': sum(len(x.raw) for x in self._rx.values()),
                'scope': 'NPY tensor container plus exact codec/channel header; separate protocol metadata excluded'}
