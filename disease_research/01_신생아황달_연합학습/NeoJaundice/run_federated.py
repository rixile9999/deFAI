import subprocess
import time
import sys
import os
# 분할 함수 임포트
from src.utils import verify_and_split_data


def main():
    print("=" * 60)
    print("🛠️  [1단계] 데이터 검증 및 물리적 분할 시작...")
    print("    -> ./debug_data 폴더에 실제 파일이 복사됩니다.")

    try:
        # 이 함수가 실행되면서 깨진 파일은 경고를 띄우고 걸러냅니다.
        verify_and_split_data(original_data_dir="./data", output_dir="./debug_data", num_clients=3)
    except Exception as e:
        print(f"❌ 데이터 준비 중 치명적 오류: {e}")
        return

    print("=" * 60)
    print("🚀  [2단계] 연합학습 시스템 시작")

    # 서버 시작
    server = subprocess.Popen([sys.executable, "src/server.py"])
    time.sleep(5)

    # 클라이언트 시작
    clients = []
    for i in range(3):
        c = subprocess.Popen([
            sys.executable, "src/client.py",
            "--client_id", str(i),
            "--num_clients", "3"
        ])
        clients.append(c)

    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        for c in clients:
            c.terminate()


if __name__ == "__main__":
    main()