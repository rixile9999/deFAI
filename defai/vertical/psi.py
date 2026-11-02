"""Commutative-DH intersection prototype, semi-honest LOCAL simulation only.

No malicious-security proof, transport authentication or side-channel protection.
Set sizes, intersection membership and a shared ordering are revealed. Endpoint
secrets and raw IDs stay in endpoint objects, but this process hosts every endpoint.
"""
import hashlib
import secrets
import numpy as np
from .alignment import checked_ids

# RFC 3526 group 14, 2048-bit safe prime; quadratic-residue subgroup of order q.
P = int('''FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1
29024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD
3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44
C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B
1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163
FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966
D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C
180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718
3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF'''.replace('\n', ''), 16)
Q = (P - 1) // 2


class DHParty:
    def __init__(self, ids, session):
        self._ids = checked_ids(ids)
        self._secret = secrets.randbelow(Q - 2) + 2
        self._session = session

    def encode(self):
        points = []
        for value in self._ids:
            digest = hashlib.shake_256(b'defai-psi-v1\0' + self._session + value.encode('utf-8')).digest(256)
            point = pow(int.from_bytes(digest, 'big') % P, 2, P)
            if point in (0, 1):
                raise ValueError("Degenerate hash-to-group point")
            points.append(point)
        return points

    def blind(self, points):
        if any(not isinstance(x, int) or x <= 1 or x >= P - 1 or pow(x, Q, P) != 1 for x in points):
            raise ValueError("Invalid DH subgroup element")
        return [pow(x, self._secret, P) for x in points]


def intersection_rows(ids_by_party):
    if len(ids_by_party) < 2:
        raise ValueError("At least two PSI parties are required")
    session = secrets.token_bytes(32)
    parties = {k: DHParty(v, session) for k, v in ids_by_party.items()}
    tokens = {}
    for name, owner in parties.items():
        points = owner.encode()
        for endpoint in parties.values():
            points = endpoint.blind(points)
        tokens[name] = points
    common = sorted(set.intersection(*(set(v) for v in tokens.values())))
    rows = {}
    for name, values in tokens.items():
        positions = {value: index for index, value in enumerate(values)}
        rows[name] = np.array([positions[x] for x in common], dtype=int)
    return rows
