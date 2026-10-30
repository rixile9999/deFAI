# -*- coding: utf-8 -*-
"""
연합 세대간격(GI) Burr XII 최대우도 추정 (Federated Fisher/L-BFGS)
================================================================
핵심 성질: 로그우도 ℓ(θ) = Σ_지역 ℓ_d(θ), ∇ℓ(θ) = Σ_지역 ∇ℓ_d(θ)  (덧셈만!)
→ 각 시군구는 원본 감염쌍을 절대 보내지 않고, 라운드마다 (ℓ_d, ∇ℓ_d) 4개 숫자만 전송 (n은 최초 1회).
→ 합산만 필요하므로 Tier 1(시도)은 동형암호 blind aggregation 가능 (paillier.py 참조).

Burr XII (scipy.stats.burr12과 동일 모수화):
  f(x; c, k, σ) = (c·k/σ)·(x/σ)^(c-1)·[1+(x/σ)^c]^(-(k+1)),  x>0
η = (log c, log k, log σ) 공간에서 최적화 (양수 제약 자동 처리).
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy import stats
import pandas as pd


# ── 로그우도·그래디언트 (η-공간, 수치 안정) ─────────────────────────
def burr_ll_grad(T, eta):
    """표본 T(>0 배열)에서 (ℓ, ∇_η ℓ) 반환. η=(log c, log k, log σ)."""
    c, k, s = np.exp(eta)
    n = len(T)
    u = np.log(T) - np.log(s)          # ln(x/σ)
    cu = c * u
    log1pz = np.logaddexp(0.0, cu)     # ln(1+z), z=(x/σ)^c — 오버플로 안전
    w = expit(cu)                      # z/(1+z)

    ll = n * (np.log(c) + np.log(k) - np.log(s)) + (c - 1) * u.sum() - (k + 1) * log1pz.sum()
    g_logc = n + c * u.sum() - (k + 1) * c * (u * w).sum()
    g_logk = n - k * log1pz.sum()
    g_logs = -c * n + (k + 1) * c * w.sum()
    return ll, np.array([g_logc, g_logk, g_logs])


# ── 연합 토폴로지 노드 ──────────────────────────────────────────────
class DistrictNode:
    """Tier 2 시군구: 원본 감염쌍 T를 보유. 외부로는 (n, ℓ, ∇ℓ)만 나감."""
    def __init__(self, sigungu_id, T):
        self.sigungu_id = int(sigungu_id)
        self.T = np.asarray(T, dtype=np.float64)
        if self.T.ndim != 1 or self.T.size == 0 or not np.isfinite(self.T).all() or (self.T <= 0).any():
            raise ValueError("Generation intervals must be a nonempty finite positive 1D array in days")
        self.n = len(self.T)

    def stats(self, eta):
        return burr_ll_grad(self.T, eta)


class ProvinceNode:
    """Tier 1 시도: 산하 시군구 기여분을 '합산만' (내용 열람 불필요 = blind 가능)."""
    def __init__(self, sido_code, districts):
        self.sido_code = sido_code
        self.districts = districts

    def stats(self, eta):
        ll, g = 0.0, np.zeros(3)
        for d in self.districts:
            ll_d, g_d = d.stats(eta)
            ll += ll_d
            g += g_d
        return ll, g


def federated_fit(nodes, eta0=None, T_hint_median=5.0):
    """중앙(Tier 0): 라운드마다 각 노드에 η를 배포하고 (ℓ, ∇ℓ) 합산을 받아
    L-BFGS-B로 갱신. 반환: (c, k, σ), 최종 NLL, 라운드 수."""
    nodes = list(nodes)
    if not nodes:
        raise ValueError("At least one nonempty federation node is required")
    if not np.isfinite(T_hint_median) or T_hint_median <= 0:
        raise ValueError("T_hint_median must be finite and positive")
    if eta0 is not None and (np.shape(eta0) != (3,) or not np.isfinite(eta0).all()):
        raise ValueError("eta0 must have three finite log-parameters")
    if eta0 is None:
        eta0 = np.log([1.5, 1.5, T_hint_median])
    rounds = {'count': 0}

    def neg_ll_and_grad(eta):
        rounds['count'] += 1
        ll, g = 0.0, np.zeros(3)
        for nd in nodes:                       # ← 1 연합 라운드 (노드별 4개 숫자 수신)
            ll_d, g_d = nd.stats(eta)
            ll += ll_d
            g += g_d
        return -ll, -g

    bounds = [(np.log(0.05), np.log(60.0))] * 3
    res = minimize(neg_ll_and_grad, eta0, jac=True, method='L-BFGS-B',
                   bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    # 경계 도달 = 사실상 적합 실패(Burr가 Weibull 극한 등으로 퇴화) — 풀링 무제약 적합과 달라질 수 있음
    for i, (lo, hi) in enumerate(bounds):
        if res.x[i] <= lo + 1e-9 or res.x[i] >= hi - 1e-9:
            import warnings as _w
            _w.warn(f"federated_fit: eta[{i}]가 경계에 도달 — MLE가 제약됨(퇴화 가능), 추정치 신뢰 불가")
    c, k, s = np.exp(res.x)
    return (c, k, s), res.fun, rounds['count'], res


def pooled_reference(T_all):
    """중앙집중(비교기준): 전체 표본을 한곳에 모아 scipy MLE."""
    c, k, loc, s = stats.burr12.fit(T_all, floc=0)
    nll = -stats.burr12.logpdf(T_all, c, k, loc=0, scale=s).sum()
    return (c, k, s), nll


# ── 데이터 로더 ─────────────────────────────────────────────────────
def load_district_pairs(csv_path, min_n=1, strict=False):
    """패치된 sim_gen_interval.csv → {sigungu_id: T배열(T>0)}."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if not {'T_days', 'child_sigungu'} <= set(df.columns):
        raise ValueError('Interval CSV requires T_days and child_sigungu columns')
    if strict:
        values = df[['T_days', 'child_sigungu']].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all() or (values[:, 0] <= 0).any():
            raise ValueError('Every interval must be finite and positive; missing rows are not silently omitted')
        if (values[:, 1] < 0).any() or not np.equal(values[:, 1], np.floor(values[:, 1])).all():
            raise ValueError('District identifiers must be nonnegative integers')
    df = df.dropna(subset=['T_days'])
    df = df[df['T_days'] > 0]
    out = {}
    for sgg, grp in df.groupby('child_sigungu'):
        if len(grp) >= min_n:
            out[int(sgg)] = grp['T_days'].to_numpy(dtype=np.float64)
    return out


def build_hierarchy(district_T):
    """{sigungu: T} → (DistrictNode 리스트, 시도별 ProvinceNode 리스트).
    시도 = 5자리 시군구 코드의 앞 2자리."""
    districts = [DistrictNode(sgg, T) for sgg, T in sorted(district_T.items())]
    by_sido = {}
    for d in districts:
        by_sido.setdefault(d.sigungu_id // 1000, []).append(d)
    provinces = [ProvinceNode(code, ds) for code, ds in sorted(by_sido.items())]
    return districts, provinces


# ── 자가 검증 ───────────────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # 1) 그래디언트 수치 대조
    T = stats.burr12.rvs(1.7, 1.5, scale=5.6, size=20000, random_state=rng)
    eta = np.log([1.4, 1.2, 4.0])
    _, g_ana = burr_ll_grad(T, eta)
    g_num = np.zeros(3)
    h = 1e-6
    for i in range(3):
        ep, em = eta.copy(), eta.copy()
        ep[i] += h; em[i] -= h
        g_num[i] = (burr_ll_grad(T, ep)[0] - burr_ll_grad(T, em)[0]) / (2 * h)
    rel = np.abs(g_ana - g_num) / np.maximum(np.abs(g_num), 1)
    assert rel.max() < 1e-5, f"gradient mismatch: {rel}"
    print(f"[selftest] gradient OK (max rel err {rel.max():.2e})")

    # 2) 연합(3분할) == 풀링 동일성
    parts = np.array_split(T, 3)
    nodes = [DistrictNode(i, p) for i, p in enumerate(parts)]
    (cf, kf, sf), nll_f, nrounds, _ = federated_fit(nodes, T_hint_median=float(np.median(T)))
    (cp_, kp, sp), nll_p = pooled_reference(T)
    print(f"[selftest] federated ({cf:.4f},{kf:.4f},{sf:.4f}) NLL={nll_f:.2f} rounds={nrounds}")
    print(f"[selftest] pooled    ({cp_:.4f},{kp:.4f},{sp:.4f}) NLL={nll_p:.2f}")
    assert abs(nll_f - nll_p) / len(T) < 1e-6, "NLL mismatch"
    print("[selftest] ALL OK")
