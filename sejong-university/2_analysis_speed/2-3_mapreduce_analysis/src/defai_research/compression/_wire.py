import zlib
import numpy as np

def _xor(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError('XOR operands must have identical byte lengths')
    return np.bitwise_xor(np.frombuffer(left, dtype=np.uint8),
                          np.frombuffer(right, dtype=np.uint8)).tobytes()


def _inflate(payload: bytes, expected_length: int) -> bytes:
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, expected_length + 1)
    except zlib.error as exc:
        raise ValueError('Invalid zlib payload') from exc
    if (len(raw) != expected_length or not decoder.eof
            or decoder.unconsumed_tail or decoder.unused_data):
        raise ValueError('Decompressed length or zlib stream boundary mismatch')
    return raw
