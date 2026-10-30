# -*- coding: utf-8 -*-
"""
[챕터1 + 챕터2 HE오버헤드] 통신 페이로드·압축·동형암호 오버헤드 실측
====================================================================
(1) 전송 페이로드 바이트 — 중앙집중(원본) vs 연합(가중치) vs 연합(충분통계, DiD)
(2) 압축 효과 — gzip / 델타부호화 / float16 양자화
(3) 동형암호(Paillier) 오버헤드 — 암호문 팽창률·암호화/합산/복호화 지연
    (TenSEAL/CKKS 미설치 → 실제 가산 동형암호 Paillier로 대표 측정)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, gzip, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from fl_benefit_demo import MLP, get_w, load_districts, ROUNDS
import he_key_bug_demo as he

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def fmt(b):
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"


def main():
    # 데이터 규모
    ds = load_districts()
    N_total = sum(len(df) for _, df in ds)
    C = len(ds)                      # 클라이언트(구) 수 = 9
    R = ROUNDS                       # FL 라운드 = 20

    # 모델 파라미터 수
    ws = get_w(MLP())
    n_params = int(sum(w.numel() for w in ws))
    w_bytes = n_params * 4           # float32

    # ── (1) 전송 페이로드 ───────────────────────────────────────
    rec_bytes = (3 + 1) * 4          # 특성3 + 라벨1, float32 (모델 전송분)
    centralized = N_total * rec_bytes                 # 원본 일괄 업로드 1회
    fl_mlp = w_bytes * C * R * 2                       # 가중치 업/다운 × 구 × 라운드
    did_stats_bytes = (4 * 4 + 4) * 8                  # XᵀWX(4x4)+XᵀWy(4), float64
    fl_did = did_stats_bytes * C * 1                   # 충분통계 1-shot 업로드

    print("="*78); print("[1-1] 전송 페이로드 비교 (전체 데이터)"); print("="*78)
    print(f"   데이터: {N_total:,} 레코드, 구 {C}개, MLP 파라미터 {n_params:,}개({fmt(w_bytes)}/회)")
    rows = [
        ["중앙집중(원본 일괄)", fmt(centralized), N_total, "원본 환자레코드"],
        ["연합-MLP(가중치, 20R)", fmt(fl_mlp), 0, f"{fmt(w_bytes)}×{C}×{R}×2"],
        ["연합-충분통계(DiD,1R)", fmt(fl_did), 0, f"{did_stats_bytes}B×{C}"],
    ]
    print(f"   {'방식':24s}{'총 전송량':>12s}{'원본전송':>10s}  설명")
    for r in rows: print(f"   {r[0]:24s}{r[1]:>12s}{r[2]:>10d}  {r[3]}")
    print(f"\n   ▶ 연합-MLP는 중앙집중 대비 약 {fl_mlp/centralized:.0f}배 더 보냄(반복통신) → 압축 필요")
    print(f"   ▶ 연합-충분통계는 중앙집중의 {fl_did/centralized*100:.2f}% (초경량, 1-shot)")

    # ── (2) 압축 ────────────────────────────────────────────────
    flat = np.concatenate([w.numpy().ravel() for w in ws]).astype(np.float32)
    raw = flat.tobytes()
    gz = gzip.compress(raw, 6)
    f16 = flat.astype(np.float16).tobytes()
    # 델타: 1 스텝 학습 후 변화량(연속 라운드 차분 모사) → 작은 값 → 더 잘 압축
    import torch, torch.nn as nn
    from fl_benefit_demo import set_w
    # 델타부호화 모사: 가중치에 소량 학습을 가해 연속 라운드 변화량(작은 값)을 만들고 차분
    m = MLP(); set_w(m, ws)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    xb = torch.randn(64, 3); yb = (torch.rand(64) > 0.7).float()
    for _ in range(3):
        opt.zero_grad(); loss = nn.functional.binary_cross_entropy(m(xb), yb); loss.backward(); opt.step()
    new = np.concatenate([p.detach().numpy().ravel() for p in m.parameters()]).astype(np.float32)
    delta = (new - flat).astype(np.float32)
    gz_delta = gzip.compress(delta.tobytes(), 6)

    print("\n" + "="*78); print("[1-2] 가중치 페이로드 압축 효과 (1회 전송분)"); print("="*78)
    comp = [
        ["원본 float32", len(raw), 1.0],
        ["gzip", len(gz), len(raw)/len(gz)],
        ["float16 양자화", len(f16), len(raw)/len(f16)],
        ["델타 + gzip", len(gz_delta), len(raw)/len(gz_delta)],
    ]
    print(f"   {'방식':16s}{'크기':>10s}{'압축비':>8s}")
    for c in comp: print(f"   {c[0]:16s}{fmt(c[1]):>10s}{c[2]:>7.2f}x")

    # ── (3) 동형암호(Paillier) 오버헤드 ─────────────────────────
    L = 16                                  # 충분통계급 벡터 길이
    vec = np.random.RandomState(0).rand(L)
    ctx = he.create_context(seed=42, bits=512)   # n≈1024bit(데모; 실서비스는 2048+)
    rng = random.Random(1); REPS = 5
    t = time.perf_counter()
    for _ in range(REPS): cs = he.enc_vec(ctx["pub"], vec, rng)
    t_enc = (time.perf_counter()-t)/REPS
    t = time.perf_counter()
    for _ in range(REPS): _ = he.add_vec(cs, cs, ctx["pub"])
    t_add = (time.perf_counter()-t)/REPS
    t = time.perf_counter()
    for _ in range(REPS): _ = he.dec_vec(ctx["sec"], cs)
    t_dec = (time.perf_counter()-t)/REPS
    plain_bytes = vec.astype(np.float64).nbytes
    cipher_bytes = sum((int(c).bit_length()+7)//8 for c in cs)

    print("\n" + "="*78); print("[2-HE] 동형암호 오버헤드 (Paillier, 벡터길이 16)"); print("="*78)
    print(f"   평문 {plain_bytes}B → 암호문 {cipher_bytes}B  (팽창률 {cipher_bytes/plain_bytes:.1f}배)")
    print(f"   암호화 {t_enc*1000:.2f} ms · 동형합산 {t_add*1000:.2f} ms · 복호화 {t_dec*1000:.2f} ms (벡터당)")

    # 저장
    pd.DataFrame([
        dict(method="중앙집중(원본)", total_bytes=int(centralized), raw_records=N_total),
        dict(method="연합-MLP(20R)", total_bytes=int(fl_mlp), raw_records=0),
        dict(method="연합-충분통계(1R)", total_bytes=int(fl_did), raw_records=0),
    ]).to_csv("experiments/comm_payload.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(scheme=c[0], bytes=c[1], ratio=round(c[2],2)) for c in comp]
                 ).to_csv("experiments/comm_compression.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(plain_B=plain_bytes, cipher_B=cipher_bytes,
                       expansion=round(cipher_bytes/plain_bytes,1),
                       enc_ms=round(t_enc*1000,2), add_ms=round(t_add*1000,2),
                       dec_ms=round(t_dec*1000,2))]
                 ).to_csv("experiments/he_overhead.csv", index=False, encoding="utf-8-sig")
    print("\n저장: comm_payload.csv, comm_compression.csv, he_overhead.csv")


if __name__ == "__main__":
    main()
