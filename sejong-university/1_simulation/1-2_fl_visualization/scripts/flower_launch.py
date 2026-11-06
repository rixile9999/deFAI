import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--algorithm", default="fedavg")
    ap.add_argument("--address", default="127.0.0.1:8080")
    ap.add_argument("--dashboard", default=None)
    a = ap.parse_args()
    py = sys.executable
    srv = [py, os.path.join(HERE, "flower_server.py"), "--address", a.address, "--rounds", str(a.rounds),
           "--n-clients", str(a.n_clients), "--algorithm", a.algorithm]
    if a.dashboard:
        srv += ["--dashboard", a.dashboard]
    server = subprocess.Popen(srv)
    time.sleep(5)
    clients = [subprocess.Popen([py, os.path.join(HERE, "flower_client.py"), "--cid", str(k),
                                 "--n-clients", str(a.n_clients), "--server", a.address])
               for k in range(a.n_clients)]
    rc = server.wait()
    for c in clients:
        try:
            c.wait(timeout=30)
        except subprocess.TimeoutExpired:
            c.kill()
    sys.exit(rc)


if __name__ == "__main__":
    main()
