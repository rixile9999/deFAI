import subprocess
import time
import sys
import os


def main():
    # 데이터 폴더가 없으면 더미 데이터 생성을 위해 client 하나를 먼저 잠시 실행하거나
    # utils를 직접 호출 (여기선 client 실행 시 생성되도록 함)

    # 1. 서버 시작
    print("Launching Server...")
    server = subprocess.Popen([sys.executable, "src/server.py"])
    time.sleep(3)  # 서버 부팅 대기

    # 2. 클라이언트 시작
    num_clients = 3
    clients = []
    for i in range(num_clients):
        print(f"Launching Client {i}...")
        c = subprocess.Popen([
            sys.executable, "src/client.py",
            "--client_id", str(i),
            "--num_clients", str(num_clients)
        ])
        clients.append(c)

    # 3. 종료 대기
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping Federated Learning...")
        server.terminate()
        for c in clients:
            c.terminate()


if __name__ == "__main__":
    main()