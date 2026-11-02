import pytest
from defai.vertical.paillier import keygen, Cipher


def test_randomized_signed_encryption_and_homomorphic_sum():
    pub, private = keygen(256, insecure_test_key=True)
    a, b = pub.encrypt(-1.25), pub.encrypt(-1.25)
    assert a != b
    assert private.decrypt(a) == -1.25
    assert private.decrypt(pub.sum([a, pub.encrypt(2.5)])) == 1.25
    assert not hasattr(pub, 'decrypt')


def test_small_keys_and_cross_key_ciphertexts_rejected():
    with pytest.raises(ValueError, match='Small keys'):
        keygen(256)
    a, private = keygen(128, insecure_test_key=True)
    b, _ = keygen(128, insecure_test_key=True)
    with pytest.raises(ValueError, match='different'):
        a.encrypt(1) + b.encrypt(1)
    with pytest.raises(ValueError):
        private.decrypt(Cipher(0, a.n))
    with pytest.raises(OverflowError):
        a.encrypt(float(a.n))
