"""Randomized signed fixed-point Paillier for the vertical tree prototype.

Cryptographic randomness and >=2048-bit modulus by default. This Python prototype
is not audited or constant-time. Small keys require explicit insecure test mode.
"""
from dataclasses import dataclass
from math import gcd, lcm, isfinite
import secrets


@dataclass(frozen=True)
class Cipher:
    value: int
    n: int

    def __add__(self, other):
        if not isinstance(other, Cipher) or self.n != other.n:
            raise ValueError('Ciphertexts use different public keys')
        return Cipher(self.value * other.value % (self.n * self.n), self.n)


@dataclass(frozen=True)
class PublicKey:
    n: int
    scale: int = 10**8

    def encrypt(self, value):
        if not isfinite(value):
            raise ValueError('Cannot encrypt non-finite values')
        message = round(float(value) * self.scale)
        if abs(message) >= self.n // 3:
            raise OverflowError('Fixed-point plaintext outside safe signed range')
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if gcd(r, self.n) == 1:
                break
        return Cipher(((1 + self.n * (message % self.n)) * pow(r, self.n, self.n**2)) % self.n**2, self.n)

    def sum(self, values):
        result = Cipher(1, self.n)
        for value in values:
            result = result + value
        return result


@dataclass(frozen=True)
class PrivateKey:
    public: PublicKey
    lam: int
    mu: int

    def decrypt(self, value):
        n = self.public.n
        if not isinstance(value, Cipher) or value.n != n or not 0 < value.value < n*n or gcd(value.value, n) != 1:
            raise ValueError('Malformed ciphertext or wrong key')
        encoded = ((pow(value.value, self.lam, n*n) - 1) // n * self.mu) % n
        if n // 3 <= encoded <= 2*n // 3:
            raise OverflowError('Aggregate exceeded signed fixed-point range')
        return (encoded - n if encoded > n // 2 else encoded) / self.public.scale


def keygen(bits=2048, *, insecure_test_key=False):
    from Crypto.Util.number import getPrime
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 128 or bits % 2:
        raise ValueError('Even modulus size >=128 bits required')
    if bits < 2048 and not insecure_test_key:
        raise ValueError('Small keys require explicit insecure_test_key=True')
    while True:
        p, q = getPrime(bits // 2), getPrime(bits // 2)
        n = p * q
        lam = lcm(p-1, q-1)
        if p != q and gcd(lam, n) == 1 and n.bit_length() == bits:
            pub = PublicKey(n)
            return pub, PrivateKey(pub, lam, pow(lam, -1, n))
