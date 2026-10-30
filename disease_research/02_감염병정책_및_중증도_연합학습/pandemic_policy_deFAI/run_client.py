import argparse
import flwr as fl
from pathlib import Path
from core.client import FCausalClient
from core.crypto import load_context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="데이터 파일 경로 (예: data/seoul.xlsx)")
    args = parser.parse_args()

    # [키 결함 수정] 서버와 '같은 키'를 써야 한다. 클라이언트는 public 컨텍스트만 로드
    # (암호화/합산만 가능, 복호화 불가). setup_keys.py 를 먼저 실행해야 함.
    key_path = Path("keys/public.ctx")
    if not key_path.exists():
        print("❌ keys/public.ctx 가 없습니다. 먼저 'python setup_keys.py' 를 실행하세요.")
        return
    ctx = load_context(str(key_path))

    print(f"🏥 Starting Client with data: {args.data}")

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FCausalClient(args.data, context=ctx, use_smpc=True),
        grpc_max_message_length=1024*1024*1024
    )


# python run_client.py --data data/seoul.xlsx
if __name__ == "__main__":
    main()