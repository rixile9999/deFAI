import pickle
import numpy as np
from typing import Any, List
from flwr.common import NDArrays

def pack_payload(payload: Any) -> NDArrays:
    """임의의 Python 객체(행렬 딕셔너리 등)를 직렬화하여 uint8 NDArray 리스트로 변환"""
    serialized = pickle.dumps(payload)
    # Flower의 타입 체크를 통과하기 위해 uint8 배열로 래핑
    return [np.frombuffer(serialized, dtype=np.uint8)]

def unpack_payload(ndarrays: NDArrays) -> Any:
    """수신된 NDArray를 다시 Python 객체로 복원"""
    if not ndarrays:
        return None
    bytes_data = ndarrays[0].tobytes()
    return pickle.loads(bytes_data)