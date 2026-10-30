# -*- coding: utf-8 -*-
"""
최소 Paillier 가산 동형암호 (의존성 0) — blind aggregation 데모용
=================================================================
he_key_bug_demo.py(2026-06 검증됨)의 구현을 부호 있는 고정소수점으로 확장.
- 공개키(n)만으로 암호화·동형덧셈 가능 → Tier 1(시도)은 내용을 못 본 채 합산
- 비밀키(lam, mu)는 Tier 0(중앙)만 보유 → 합산 결과만 복호화
연합 Burr 추정의 전송량은 노드당 라운드마다 숫자 4개(ll, grad3)뿐이라 실시간 처리 가능.
"""
import random
from math import gcd

SCALE = 10 ** 10  # 고정소수점 스케일 (n~2^192 대비 여유 막대; 양자화가 L-BFGS 경로에 영향 없도록 1e-10)


def _is_prime(n, rng, k=25):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(k):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _gen_prime(bits, rng):
    while True:
        cand = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_prime(cand, rng):
            return cand


def keygen(bits=96, seed=42):
    """1회 실행(중앙 setup). 반환: (public_n, secret=(lam, mu, n))."""
    rng = random.Random(seed)
    while True:
        p, q = _gen_prime(bits, rng), _gen_prime(bits, rng)
        if p == q:
            continue
        n = p * q
        lam = (p - 1) * (q - 1) // gcd(p - 1, q - 1)
        if gcd(lam, n) != 1:
            continue
        mu = pow(lam, -1, n)
        return n, (lam, mu, n)


def encrypt(pub_n, value_float, rng):
    """부호 있는 실수 → 암호문 (공개키만 필요)."""
    n = pub_n
    m = round(value_float * SCALE) % n          # 음수는 mod n 표현
    nsq = n * n
    while True:
        r = rng.randrange(1, n)
        if gcd(r, n) == 1:
            break
    return (pow(n + 1, m, nsq) * pow(r, n, nsq)) % nsq


def he_add(c1, c2, pub_n):
    """암호문 곱 == 평문 합 (복호화 불필요 → blind)."""
    return (c1 * c2) % (pub_n * pub_n)


def decrypt(secret, ciph):
    """비밀키 보유자(Tier 0)만 호출 가능. 부호 복원 포함."""
    lam, mu, n = secret
    x = pow(ciph, lam, n * n)
    m = ((x - 1) // n * mu) % n
    if m > n // 2:                              # 부호 복원
        m -= n
    return m / SCALE


if __name__ == '__main__':
    pub, sec = keygen(bits=64, seed=7)
    rng = random.Random(1)
    vals = [3.14159, -2.71828, 1e-6, -12345.678]
    cs = [encrypt(pub, v, rng) for v in vals]
    acc = cs[0]
    for c in cs[1:]:
        acc = he_add(acc, c, pub)
    got = decrypt(sec, acc)
    want = sum(vals)
    assert abs(got - want) < 1e-6, (got, want)
    print(f"[selftest] Paillier signed sum OK: {got:.6f} == {want:.6f}")
