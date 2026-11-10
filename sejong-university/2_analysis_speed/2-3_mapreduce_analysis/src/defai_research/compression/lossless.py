from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import struct
import zlib

from ._wire import _inflate, _xor
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

MAGIC = b'CLC1'
VERSION = 1
CODEC_ID = 1
HEADER = struct.Struct('<4sBBBBHHIIIIIII')
COUNT = struct.Struct('<I')
TENSOR = struct.Struct('<BBBBI')
MODES = {0: 'raw', 1: 'zlib', 2: 'xor_zlib', 3: 'shuffle_zlib', 4: 'xor_shuffle_zlib'}
MODE_IDS = {value: key for key, value in MODES.items()}
XOR_MODES = {2, 4}
SHUFFLE_MODES = {3, 4}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _crc(value):
    return zlib.crc32(value) & 0xffffffff


def _uint(value, name, maximum=0xffffffff):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(name + ' must be an integer')
    if value < 0 or value > maximum:
        raise ValueError(name + ' is outside the declared wire range')
    return int(value)


def _channel(channel):
    if not isinstance(channel, (tuple, list)) or len(channel) != 4:
        raise ValueError('Channel must contain hop, direction, sender and receiver')
    hop, direction, sender, receiver = channel
    if (hop, direction) == ('edge', 'up'):
        link = 0
    elif (hop, direction) == ('cloud', 'up') and receiver == 'central':
        link, receiver = 1, 0
    elif (hop, direction) == ('cloud', 'down') and sender == 'central':
        link, sender = 2, 0
    elif (hop, direction) == ('edge', 'down'):
        link = 3
    else:
        raise ValueError('Unsupported directed hierarchy channel')
    return link, _uint(sender, 'sender', 65535), _uint(receiver, 'receiver', 65535)


def byte_shuffle4(raw):
    whole = len(raw) // 4 * 4
    return (np.frombuffer(raw[:whole], np.uint8).reshape(-1, 4).T.copy().tobytes()
            + raw[whole:])


def byte_unshuffle4(raw):
    whole = len(raw) // 4 * 4
    return (np.frombuffer(raw[:whole], np.uint8).reshape(4, -1).T.copy().tobytes()
            + raw[whole:])


def _padding(size):
    return bytes((-size) % 4)


def _serialize(arrays):
    arrays = list(arrays)
    for value in arrays:
        array = np.asarray(value)
        if array.dtype.kind not in 'biufc' or array.dtype.fields is not None:
            raise TypeError('Only primitive numeric Flower arrays are supported')
    parameters = ndarrays_to_parameters(arrays)
    if parameters.tensor_type != 'numpy.ndarray':
        raise TypeError('Expected Flower NumPy tensor buffers')
    canonical = parameters_to_ndarrays(parameters)
    chunks = [COUNT.pack(_uint(len(canonical), 'tensor_count'))]
    for array in canonical:
        dtype = array.dtype.str.encode('ascii')
        if len(dtype) > 63 or array.ndim > 32:
            raise ValueError('Tensor descriptor exceeds compact schema')
        order = 1 if array.flags.f_contiguous and not array.flags.c_contiguous else 0
        content = array.tobytes(order='F' if order else 'C')
        descriptor = (TENSOR.pack(len(dtype), array.ndim, order, 0, _uint(len(content), 'tensor_bytes'))
                      + dtype + b''.join(COUNT.pack(_uint(d, 'tensor_dimension')) for d in array.shape))
        chunks.extend([descriptor, _padding(len(descriptor)), content, _padding(len(content))])
    return b''.join(chunks), list(parameters.tensors)


def _deserialize(raw):
    if len(raw) < COUNT.size:
        raise ValueError('Truncated compact tensor count')
    count, = COUNT.unpack_from(raw)
    if count > (len(raw) - COUNT.size) // TENSOR.size:
        raise ValueError('Impossible compact tensor count')
    arrays = []
    offset = COUNT.size
    for _ in range(count):
        start = offset
        if offset + TENSOR.size > len(raw):
            raise ValueError('Truncated compact tensor descriptor')
        dtype_size, ndim, order, reserved, nbytes = TENSOR.unpack_from(raw, offset)
        offset += TENSOR.size
        if not 1 <= dtype_size <= 63 or ndim > 32 or order not in (0, 1) or reserved:
            raise ValueError('Invalid compact descriptor flags')
        if offset + dtype_size + ndim * COUNT.size > len(raw):
            raise ValueError('Truncated compact tensor shape')
        try:
            dtype = np.dtype(raw[offset:offset + dtype_size].decode('ascii'))
        except (TypeError, ValueError, UnicodeError) as error:
            raise ValueError('Invalid compact dtype') from error
        offset += dtype_size
        shape = tuple(COUNT.unpack_from(raw, offset + index * COUNT.size)[0] for index in range(ndim))
        offset += ndim * COUNT.size
        if (dtype.kind not in 'biufc' or dtype.fields is not None
                or math.prod(shape) * dtype.itemsize != nbytes):
            raise ValueError('Compact dtype/shape disagrees with tensor size')
        pad = (- (offset - start)) % 4
        if offset + pad + nbytes > len(raw) or raw[offset:offset + pad] != bytes(pad):
            raise ValueError('Invalid compact descriptor padding or tensor extent')
        offset += pad
        content = raw[offset:offset + nbytes]
        offset += nbytes
        data_pad = (-nbytes) % 4
        if offset + data_pad > len(raw) or raw[offset:offset + data_pad] != bytes(data_pad):
            raise ValueError('Invalid compact tensor padding')
        offset += data_pad
        array = np.frombuffer(content, dtype=dtype).reshape(shape, order='F' if order else 'C')
        arrays.append(array.copy(order='F' if order else 'C'))
    if offset != len(raw):
        raise ValueError('Unexpected trailing compact tensor bytes')
    canonical, tensors = _serialize(arrays)
    if canonical != raw:
        raise ValueError('Received descriptor is not canonical')
    return arrays, tensors


@dataclass(frozen=True)
class CacheEntry:
    sequence: int
    round_id: int
    raw: bytes = field(repr=False)
    crc: int


@dataclass(frozen=True)
class EncodedFrame:
    frame: bytes
    key: tuple
    channel: tuple
    round_id: int
    sequence: int
    mode: str
    raw_bytes: int
    payload_bytes: int
    raw_crc32: int
    base_sequence: int
    base_crc32: int
    flower_tensor_bytes: int
    tensor_count: int
    candidate_frame_bytes: dict
    _raw: bytes = field(repr=False)
    _tensors: list = field(repr=False)

    @property
    def frame_bytes(self):
        return len(self.frame)

    @property
    def header_bytes(self):
        return HEADER.size


@dataclass(frozen=True)
class DecodedFrame:
    key: tuple
    round_id: int
    sequence: int
    frame_sha256: str
    raw_crc32: int
    arrays: list = field(repr=False)
    tensors: list = field(repr=False)
    _raw: bytes = field(repr=False)


@dataclass(frozen=True)
class CompactTransfer:
    encoded: EncodedFrame
    decoded_arrays: list = field(repr=False)

    @property
    def frame(self):
        return self.encoded.frame

    @property
    def frame_bytes(self):
        return self.encoded.frame_bytes

    @property
    def mode(self):
        return self.encoded.mode

    def evidence(self, include_frame=False):
        value = dict(mode=self.mode, codec='compact_lossless', codec_id=CODEC_ID,
                     channel=list(self.encoded.channel), round=self.encoded.round_id,
                     sequence=self.encoded.sequence, frame_bytes=self.frame_bytes,
                     header_bytes=HEADER.size, payload_bytes=self.encoded.payload_bytes,
                     raw_container_bytes=self.encoded.raw_bytes,
                     flower_tensor_bytes=self.encoded.flower_tensor_bytes,
                     tensor_count=self.encoded.tensor_count,
                     base_sequence=self.encoded.base_sequence,
                     raw_crc32=self.encoded.raw_crc32, base_crc32=self.encoded.base_crc32,
                     frame_sha256=_sha(self.frame), raw_sha256=_sha(self.encoded._raw),
                     candidate_frame_bytes=self.encoded.candidate_frame_bytes,
                     flower_npy_byte_identical=True, decoded_byte_identical=True,
                     scope='40-byte matched frame plus compact numeric descriptors and exact data; CRC detects errors, no authentication or WAN claim')
        if include_frame:
            value['frame_hex'] = self.frame.hex()
        return value


class CompactLosslessCodec:
    def __init__(self, *, compression_level=6, max_raw_bytes=128 * 1024 * 1024):
        self.compression_level = _uint(compression_level, 'compression_level', 9)
        self.max_raw_bytes = _uint(max_raw_bytes, 'max_raw_bytes')
        if self.compression_level < 1 or self.max_raw_bytes < COUNT.size:
            raise ValueError('Invalid compact lossless codec limits')
        self._tx, self._rx = {}, {}
        self._totals = Counter()

    def encode_arrays(self, arrays, *, channel, round_id):
        key = _channel(channel)
        rnd = _uint(round_id, 'round_id')
        previous = self._tx.get(key)
        if previous is not None and rnd <= previous.round_id:
            raise ValueError('Directed channel requires increasing rounds')
        raw, tensors = _serialize(arrays)
        if len(raw) > self.max_raw_bytes:
            raise ValueError('Raw compact tensors exceed max_raw_bytes')
        sequence = _uint(previous.sequence + 1 if previous else 1, 'sequence')
        candidates = [(0, raw), (1, zlib.compress(raw, self.compression_level)),
                      (3, zlib.compress(byte_shuffle4(raw), self.compression_level))]
        if previous is not None and len(previous.raw) == len(raw):
            difference = _xor(raw, previous.raw)
            candidates += [(2, zlib.compress(difference, self.compression_level)),
                           (4, zlib.compress(byte_shuffle4(difference), self.compression_level))]
        mode, payload = min(candidates, key=lambda item: (len(item[1]), item[0]))
        crc = _crc(raw)
        base_seq = previous.sequence if mode in XOR_MODES else 0
        base_crc = previous.crc if mode in XOR_MODES else 0
        frame = HEADER.pack(MAGIC, VERSION, CODEC_ID, mode, *key, rnd, sequence,
                            len(raw), len(payload), crc, base_seq, base_crc) + payload
        return EncodedFrame(frame, key, tuple(channel), rnd, sequence, MODES[mode],
                            len(raw), len(payload), crc, base_seq, base_crc,
                            sum(map(len, tensors)), len(tensors),
                            {MODES[k]: HEADER.size + len(v) for k, v in candidates}, raw, tensors)

    def decode_frame(self, frame, *, expected_channel, expected_round):
        key = _channel(expected_channel)
        expected_round = _uint(expected_round, 'expected_round')
        if not isinstance(frame, bytes) or len(frame) < HEADER.size:
            raise ValueError('Complete immutable compact frame required')
        magic, version, codec_id, mode, link, sender, receiver, rnd, seq, raw_len, packed_len, crc, base_seq, base_crc = HEADER.unpack_from(frame)
        if magic != MAGIC or version != VERSION or codec_id != CODEC_ID or mode not in MODES:
            raise ValueError('Unsupported compact wire schema')
        if (link, sender, receiver) != key or rnd != expected_round:
            raise ValueError('Compact receiver channel or round mismatch')
        if len(frame) != HEADER.size + packed_len or not COUNT.size <= raw_len <= self.max_raw_bytes:
            raise ValueError('Compact frame lengths exceed declared boundaries')
        previous = self._rx.get(key)
        if seq != (previous.sequence + 1 if previous else 1) or previous is not None and rnd <= previous.round_id:
            raise ValueError('Compact frame sequence is stale or out of order')
        if mode in XOR_MODES:
            if (previous is None or previous.sequence != base_seq or previous.crc != base_crc
                    or len(previous.raw) != raw_len):
                raise ValueError('Compact XOR base cache mismatch')
        elif base_seq or base_crc:
            raise ValueError('Independent compact frame declares an XOR base')
        payload = frame[HEADER.size:]
        raw = payload if mode == 0 else _inflate(payload, raw_len)
        if mode in SHUFFLE_MODES:
            raw = byte_unshuffle4(raw)
        if mode in XOR_MODES:
            raw = _xor(raw, previous.raw)
        if len(raw) != raw_len or _crc(raw) != crc:
            raise ValueError('Compact restored bytes fail CRC or length check')
        arrays, tensors = _deserialize(raw)
        self._rx[key] = CacheEntry(seq, rnd, raw, crc)
        return DecodedFrame(key, rnd, seq, _sha(frame), crc, arrays, tensors, raw)

    def confirm_delivery(self, encoded, decoded):
        previous = self._tx.get(encoded.key)
        if (encoded.sequence != (previous.sequence + 1 if previous else 1)
                or previous is not None and encoded.round_id <= previous.round_id
                or decoded.key != encoded.key or decoded.round_id != encoded.round_id
                or decoded.sequence != encoded.sequence or decoded.frame_sha256 != _sha(encoded.frame)
                or decoded._raw != encoded._raw or decoded.tensors != encoded._tensors):
            raise ValueError('Compact sender confirmation differs from actual received tensors')
        self._tx[encoded.key] = CacheEntry(encoded.sequence, encoded.round_id,
                                          encoded._raw, encoded.raw_crc32)

    def transmit(self, arrays, *, channel, round_id):
        key = _channel(channel)
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
        self._totals['frames'] += 1
        self._totals['frame_bytes'] += encoded.frame_bytes
        self._totals['payload_bytes'] += encoded.payload_bytes
        self._totals['header_bytes'] += HEADER.size
        self._totals[encoded.mode + '_frames'] += 1
        return CompactTransfer(encoded, decoded.arrays)

    def accounting(self):
        return dict(self._totals, tx_channels=len(self._tx), rx_channels=len(self._rx))


def self_test():
    rng = np.random.default_rng(1771)
    arrays = [rng.normal(size=(64, 32)).astype(np.float32), np.zeros(32, np.float32),
              np.asfortranarray(rng.normal(size=(9, 7)).astype('>f8')),
              np.asarray([0x7fc01234, 0x80000000, 0x7f800000], np.uint32).view(np.float32),
              np.empty((0, 3), np.float32), np.array(2.5, np.float64)]
    channel = ('edge', 'down', 0, 1)
    codec = CompactLosslessCodec()
    first = codec.transmit(arrays, channel=channel, round_id=1)
    second = codec.transmit(arrays, channel=channel, round_id=2)
    assert HEADER.size == 40 and first.mode not in {'xor_zlib', 'xor_shuffle_zlib'}
    assert second.mode in {'xor_zlib', 'xor_shuffle_zlib'}
    assert ndarrays_to_parameters(second.decoded_arrays).tensors == ndarrays_to_parameters(arrays).tensors
    assert second.frame_bytes == min(second.encoded.candidate_frame_bytes.values())
    quantized = [rng.integers(-127, 128, 2410, np.int8), np.full(10, .005, np.float32)]
    codec.transmit(quantized, channel=('edge', 'up', 1, 0), round_id=1)
    resized = codec.transmit([np.zeros(5, np.float32)], channel=channel, round_id=3)
    assert 'xor_zlib' not in resized.encoded.candidate_frame_bytes
    for size in (0, 1, 2, 4, 5, 1025):
        raw = rng.bytes(size)
        assert byte_unshuffle4(byte_shuffle4(raw)) == raw
    sender, receiver = CompactLosslessCodec(), CompactLosslessCodec()
    pending = sender.encode_arrays(arrays, channel=channel, round_id=1)
    assert not sender._tx
    received = receiver.decode_frame(pending.frame, expected_channel=channel, expected_round=1)
    sender.confirm_delivery(pending, received)
    guards = []
    def reject(name, fn):
        try:
            fn()
        except (ValueError, TypeError):
            guards.append(name)
        else:
            raise AssertionError('Missing guard: ' + name)
    reject('duplicate_confirmation', lambda: sender.confirm_delivery(pending, received))
    reject('missing_previous', lambda: CompactLosslessCodec().decode_frame(second.frame, expected_channel=channel, expected_round=2))
    reject('stale_round', lambda: codec.transmit(arrays, channel=channel, round_id=2))
    reject('wrong_channel', lambda: CompactLosslessCodec().decode_frame(first.frame, expected_channel=('edge', 'down', 0, 2), expected_round=1))
    reject('truncated', lambda: CompactLosslessCodec().decode_frame(first.frame[:-1], expected_channel=channel, expected_round=1))
    damaged = bytearray(first.frame); damaged[-1] ^= 1
    reject('corruption', lambda: CompactLosslessCodec().decode_frame(bytes(damaged), expected_channel=channel, expected_round=1))
    reject('bool_round', lambda: codec.transmit(arrays, channel=('cloud', 'down', 'central', 0), round_id=True))
    reject('endpoint_overflow', lambda: codec.transmit(arrays, channel=('edge', 'up', 65536, 0), round_id=1))
    reject('object_array', lambda: codec.transmit([np.array([object()])], channel=('cloud', 'up', 0, 'central'), round_id=1))
    reject('raw_size_limit', lambda: CompactLosslessCodec(max_raw_bytes=32).transmit(arrays, channel=channel, round_id=1))
    class RejectOnce(CompactLosslessCodec):
        reject_next = True
        def confirm_delivery(self, encoded, decoded):
            if self.reject_next:
                self.reject_next = False
                raise ValueError('injected delivery mismatch')
            return super().confirm_delivery(encoded, decoded)
    retry = RejectOnce()
    reject('rollback_injection', lambda: retry.transmit(arrays, channel=channel, round_id=1))
    assert not retry._tx and not retry._rx
    retry.transmit(arrays, channel=channel, round_id=1)
    return dict(status='passed', header_bytes=HEADER.size, flower_npy_byte_exact=True,
                endian_fortran_nan_signed_zero_scalar_empty_preserved=True,
                shuffle_roundtrip=True, separate_receiver_acknowledgment=True,
                guard_cases=guards, first_frame=first.evidence(), repeated_frame=second.evidence())


