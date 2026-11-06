import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def steps(out, backend, quick):
    r = "3" if quick else "6"
    common = ["--backend", backend]
    hfl = ["--set", f"backend={backend}", "--set", f"output-dir={out / 'hfl'}", "--set", f"global-rounds={r}",
           "--set", "num-clients=9", "--set", "num-edges=3"]
    return [
        ("계층형 연합학습 시뮬레이터 (in-process Flask)", ["hierarchical_fl", "run"] + hfl),
        ("계층형 연합학습 시뮬레이터 (Flask 엣지 프로세스, HTTP)",
         ["hierarchical_fl", "run", "--spawn-edges"] + hfl + ["--set", "run-id=hfl_http_demo"]),
        ("보안 공격·방어 시뮬레이션 (Y1~Y3 공격/방어 매트릭스)",
         ["security_sim", "matrix"] + common + ["--rounds", r, "--clients", "9", "--output-dir",
                                                str(out / "security")]),
        ("보안: 모델 역추적 공격과 출력 방어",
         ["security_sim", "inversion"] + common + ["--queries", "1000", "--steps", "100", "--output-dir",
                                                   str(out / "security")]),
        ("연합학습 시나리오 테스트 (S1~S5, 적응력)",
         ["scenario_test"] + common + ["--rounds", r, "--clients", "9", "--output-dir", str(out / "scenarios")]),
        ("연산 그래프 기반 가지치기 (중앙)",
         ["graph_pruning", "central", "--set", "model=cnn_bn", "--bn-l1", "1e-4", "--iters", "4", "--output-dir",
          str(out / "graph_pruning")] + common),
        ("연산 그래프 기반 가지치기 (연합)",
         ["graph_pruning", "federated", "--rounds", "5", "--clients", "6", "--output-dir",
          str(out / "graph_pruning")] + common),
        ("중간층 매칭 기반 지식 증류 (중앙)",
         ["layer_kd", "central", "--teacher-epochs", "2", "--epochs", "2", "--output-dir", str(out / "layer_kd")]
         + common),
        ("중간층 매칭 기반 지식 증류 (연합)",
         ["layer_kd", "federated", "--methods", "student,logit,combined", "--rounds", r, "--clients", "6",
          "--teacher-epochs", "2", "--output-dir", str(out / "layer_kd")] + common),
        ("데이터 불균형 해결",
         ["data_imbalance", "--rounds", r, "--clients", "8", "--local-study", "--output-dir",
          str(out / "data_imbalance")] + common),
        ("다계층 네트워크 비용 최적화",
         ["network_cost", "--rounds", r, "--clients", "8", "--output-dir", str(out / "network_cost")] + common),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python scripts/run_demo.py")
    ap.add_argument("--backend", default="builtin", choices=("builtin", "kist"))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--output-dir", default=str(ROOT / "results" / "demo"))
    args = ap.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    report = []
    for k, (title, cmd) in enumerate(steps(out, args.backend, not args.full), 1):
        if args.only and str(k) not in args.only.split(","):
            continue
        print(f"\n[{k}] {title}\n    python -m {' '.join(cmd)}", flush=True)
        t = time.time()
        log = out / f"step{k:02d}.log"
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run([sys.executable, "-m"] + cmd, cwd=str(ROOT), env=env, stdout=f,
                                  stderr=subprocess.STDOUT)
        tail = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and "WARNING" not in line][-6:]
        for line in tail:
            print("    " + line[:160])
        report.append((k, title, proc.returncode, time.time() - t))
    print("\n" + "=" * 72)
    for k, title, code, sec in report:
        print(f"[{k:2d}] {'OK  ' if code == 0 else 'FAIL'} {sec:7.1f}s  {title}")
    print(f"\nresults: {out}")
    print(f"dashboard: python -m hierarchical_fl dashboard --results {out}")
    return 0 if all(code == 0 for _, _, code, _ in report) else 1


if __name__ == "__main__":
    sys.exit(main())
