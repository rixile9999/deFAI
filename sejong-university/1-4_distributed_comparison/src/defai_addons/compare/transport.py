import os

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class UpdateCipher:
    def __init__(self, key=None):
        self._aead = AESGCM(key if key is not None else AESGCM.generate_key(bit_length=256))

    @staticmethod
    def _context(client_id, server_round):
        return ("defai-update-v1:%s:%d" % (client_id, server_round)).encode("utf-8")

    def encrypt(self, update, client_id, server_round):
        vec = np.asarray(update, dtype="<f8")
        if vec.ndim != 1 or not np.isfinite(vec).all():
            raise ValueError("Expected a finite one-dimensional update")
        nonce = os.urandom(12)
        payload = nonce + self._aead.encrypt(nonce, vec.tobytes(), self._context(client_id, server_round))
        return np.frombuffer(payload, dtype=np.uint8).copy()

    def decrypt(self, payload, client_id, server_round, expected_size):
        if payload.dtype != np.uint8 or payload.ndim != 1 or payload.size != 28 + expected_size * 8:
            raise ValueError("Invalid encrypted update size or type")
        packet = payload.tobytes()
        plain = self._aead.decrypt(packet[:12], packet[12:], self._context(client_id, server_round))
        return np.frombuffer(plain, dtype="<f8").copy()
