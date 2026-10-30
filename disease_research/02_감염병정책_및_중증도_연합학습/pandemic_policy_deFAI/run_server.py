import flwr as fl
from pathlib import Path
from core.strategy import FCausalStrategy
from core.crypto import load_context


def main():
    # [키 결함 수정] 클라이언트와 '같은 키'를 공유해야 동형 합산·복호화가 동작한다.
    # setup_keys.py 로 생성한 secret 컨텍스트를 로드 (서버만 복호화 가능)
    key_path = Path("keys/secret.ctx")
    if not key_path.exists():
        print("❌ keys/secret.ctx 가 없습니다. 먼저 'python setup_keys.py' 를 실행하세요.")
        return
    ctx = load_context(str(key_path))

    # 2개 이상의 클라이언트가 접속해야 학습 시작
    strategy = FCausalStrategy(
        context=ctx,
        min_fit_clients=2,
        min_available_clients=2
    )

    print("🚀 Starting F-Causal Server on port 8080...")

    # 1GB 이상의 메시지 허용 (암호문 전송용)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024
    )


if __name__ == "__main__":
    main()