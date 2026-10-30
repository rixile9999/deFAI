"""
[키 결함 수정] 공유 암호화 키 생성 스크립트
=============================================
기존 버그: run_server.py 와 run_client.py 가 '각자' create_context() 를 호출 →
          서버/클라이언트의 비밀키가 서로 달라 동형 합산·복호화가 깨짐(Giant numbers).

수정: 키를 '한 번만' 생성해서
      - keys/secret.ctx : 서버 전용 (복호화 가능)
      - keys/public.ctx : 클라이언트 공유 (암호화/합산만 가능, 복호화 불가)
계층형(setup_hierarchical.py)과 동일한 PKI 분리 방식.

실행:  python setup_keys.py   (run_server / run_client 보다 먼저)
"""
from pathlib import Path
from core.crypto import create_context, save_context


def main():
    Path("keys").mkdir(exist_ok=True)

    ctx = create_context()  # CKKS 컨텍스트를 '한 번만' 생성
    save_context(ctx, "keys/secret.ctx", save_secret_key=True)   # 서버 전용
    save_context(ctx, "keys/public.ctx", save_secret_key=False)  # 클라이언트 공유

    print("🔐 공유 키 생성 완료")
    print("   keys/secret.ctx  → run_server.py 가 사용 (복호화 가능)")
    print("   keys/public.ctx  → run_client.py 가 사용 (암호화/합산만)")


if __name__ == "__main__":
    main()
