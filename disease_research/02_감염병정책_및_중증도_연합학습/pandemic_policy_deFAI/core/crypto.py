import tenseal as ts
import numpy as np

def create_context():
    """CKKS 암호화 컨텍스트 생성 (서버-클라이언트 공유용)"""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def encrypt_vector(context, vector: np.ndarray) -> bytes:
    """Numpy Vector -> Encrypted Bytes"""
    # CKKS는 1D 실수 벡터 연산에 최적화됨
    return ts.ckks_vector(context, vector.flatten()).serialize()

def decrypt_vector(context, enc_bytes: bytes, original_shape) -> np.ndarray:
    """Encrypted Bytes -> Numpy Matrix"""
    vec = ts.ckks_vector_from(context, enc_bytes)
    return np.array(vec.decrypt()).reshape(original_shape)

def save_context(context, path, save_secret_key: bool):
    """컨텍스트를 파일로 직렬화. 서버는 secret 포함, 클라이언트는 public(secret 제거)."""
    with open(path, "wb") as f:
        f.write(context.serialize(save_secret_key=save_secret_key))

def load_context(path):
    """직렬화된 컨텍스트를 로드 (서버/클라이언트가 '같은 키'를 공유하기 위함)"""
    with open(path, "rb") as f:
        return ts.context_from(f.read())