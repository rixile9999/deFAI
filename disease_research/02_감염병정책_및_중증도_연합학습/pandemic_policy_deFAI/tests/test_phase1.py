import numpy as np
from core.serialization import pack_payload, unpack_payload

def test_serialization_logic():
    original = {"xtx": np.eye(3), "xty": np.ones(3)}
    packed = pack_payload(original)
    restored = unpack_payload(packed)
    
    assert np.allclose(original['xtx'], restored['xtx'])
    print("✅ Phase 1: Serialization Test Passed")

if __name__ == "__main__":
    test_serialization_logic()