# -*- coding: utf-8 -*-
"""
동형암호 '키 분리 결함' 재현 & 수정 데모
=========================================
2세대 코드(pandemic_policy_deFAI)의 run_server.py / run_client.py 는
각자 따로 create_context() 를 호출한다 → 서버와 클라이언트의 비밀키가 서로 다르다.
CKKS/Paillier 같은 동형암호는 '같은 키'로 암호화/합산/복호화해야만 한다.
키가 다르면 합산·복호화 결과가 난수(=Giant numbers)가 된다.

TenSEAL(CKKS)은 이 Python 3.13 환경에 설치 불가하므로,
의존성 0짜리 실제 가산 동형암호인 Paillier 로 '동일한 실패 양상'을 재현한다.
(Paillier: 곱셈 c1*c2 == 평문 덧셈 m1+m2 / 공개키 암호화 / 비밀키 복호화)
"""
import sys
import random
from math import gcd

# Windows 한국어 콘솔(cp949)에서도 한글/이모지 출력이 깨지지 않도록 UTF-8 고정
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCALE = 10**6  # 실수를 정수로 인코딩하기 위한 스케일 (mean=3.12 -> 3,120,000)


# ----------------------------------------------------------------------
# 최소 Paillier 구현 (keygen / encrypt / homomorphic add / decrypt)
# ----------------------------------------------------------------------
def _is_prime(n, rng, k=20):
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


def create_context(seed, bits=48):
    """CKKS 의 ts.context(...) 에 대응. seed 가 다르면 '다른 키'가 나온다."""
    rng = random.Random(seed)
    while True:
        p = _gen_prime(bits, rng)
        q = _gen_prime(bits, rng)
        if p == q:
            continue
        n = p * q
        lam = (p - 1) * (q - 1) // gcd(p - 1, q - 1)  # lcm
        if gcd(lam, n) != 1:
            continue
        mu = pow(lam, -1, n)  # g = n+1 이므로 L(g^lam mod n^2) = lam
        return {"pub": n, "sec": (lam, mu, n), "rng": rng}


def encrypt(ctx_pub_n, m, rng):
    n = ctx_pub_n
    nsq = n * n
    while True:
        r = rng.randrange(1, n)
        if gcd(r, n) == 1:
            break
    return (pow(n + 1, m, nsq) * pow(r, n, nsq)) % nsq


def he_add(c1, c2, pub_n):
    """암호문 곱 == 평문 덧셈 (Homomorphic Addition)"""
    return (c1 * c2) % (pub_n * pub_n)


def decrypt(sec, c):
    lam, mu, n = sec
    x = pow(c, lam, n * n)
    L = (x - 1) // n
    return (L * mu) % n


# 벡터 단위 헬퍼 (시군구가 보내는 통계 벡터를 통째로 처리)
def enc_vec(pub_n, vec, rng):
    return [encrypt(pub_n, round(x * SCALE), rng) for x in vec]


def add_vec(cs1, cs2, pub_n):
    return [he_add(a, b, pub_n) for a, b in zip(cs1, cs2)]


def dec_vec(sec, cs):
    return [decrypt(sec, c) / SCALE for c in cs]


# ----------------------------------------------------------------------
# 시나리오: 시군구 2곳이 통계 벡터를 보내고, 상위 노드가 합산해야 한다.
# 정답(평문 합) = districtA + districtB
# 벡터 = [n_pre, mean_pre, std_pre, n_post, mean_post, std_post]
# ----------------------------------------------------------------------
districtA = [58.0, 3.1234, 0.3456, 150.0, 3.4567, 0.3210]
districtB = [42.0, 2.9876, 0.2890, 110.0, 3.2011, 0.2754]
truth = [a + b for a, b in zip(districtA, districtB)]


def show(title, vec):
    print(f"   {title}")
    print("      [" + ", ".join(f"{v:14.4f}" for v in vec) + "]")


def scenario_bug():
    print("=" * 72)
    print(" [BUG] run_server.py / run_client.py 처럼 '각자' create_context() 호출")
    print("=" * 72)
    # 원본 코드 그대로: 서버도, 클라 A도, 클라 B도 따로 컨텍스트 생성 → 키가 전부 다름
    ctx_server = create_context(seed=1)
    ctx_A = create_context(seed=2)
    ctx_B = create_context(seed=3)
    print("   서버 키 n != 클라A 키 n != 클라B 키 n  (서로 무관한 키 3개)\n")

    cA = enc_vec(ctx_A["pub"], districtA, ctx_A["rng"])
    cB = enc_vec(ctx_B["pub"], districtB, ctx_B["rng"])

    # 서버는 자기 키(modulus)로 합산하고 자기 비밀키로 복호화한다 (원본 strategy.py 그대로)
    summed = add_vec(cA, cB, ctx_server["pub"])
    out = dec_vec(ctx_server["sec"], summed)

    show("정답 (평문 합):", truth)
    show("복호화 결과 :", out)
    print("\n   → 천문학적 난수(Giant numbers). 암호화 경로가 사실상 동작하지 않음. ❌\n")


def scenario_fix():
    print("=" * 72)
    print(" [FIX] setup 에서 키 1개 생성 → 공개키는 클라에, 비밀키는 서버에 (계층형 방식)")
    print("=" * 72)
    kp = create_context(seed=42)            # setup_hierarchical.py 한 번만 실행
    public_n = kp["pub"]                    # keys/public.ctx  (암호화/합산만 가능)
    secret = kp["sec"]                      # keys/secret.ctx  (서버만 복호화 가능)
    rngA = random.Random(101)
    rngB = random.Random(102)
    print("   모든 클라가 '같은' public_n 으로 암호화. 서버만 secret 보유.\n")

    cA = enc_vec(public_n, districtA, rngA)
    cB = enc_vec(public_n, districtB, rngB)
    summed = add_vec(cA, cB, public_n)      # 상위 노드: 복호화 없이 합산만 (blind)
    out = dec_vec(secret, summed)           # 최상위: 비밀키로 복호화

    show("정답 (평문 합):", truth)
    show("복호화 결과 :", out)
    ok = all(abs(a - b) < 1e-3 for a, b in zip(truth, out))
    print(f"\n   → 정답과 일치: {ok}  {'✅' if ok else '❌'}\n")


if __name__ == "__main__":
    scenario_bug()
    scenario_fix()
