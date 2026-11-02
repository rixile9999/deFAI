import numpy as np
import pytest
from defai.vertical.boost import HistogramParty, SecureBoostClassifier


def test_pdf_six_customer_example_selects_remote_threshold_and_leaf_weights():
    parties = [HistogramParty('bank', np.zeros((6, 1))), HistogramParty('telecom', [[0], [0], [1], [2], [9], [15]])]
    model = SecureBoostClassifier(trees=1, depth=1, lr=1., min_leaf=1, key_bits=256, insecure_test_key=True)
    model.fit(parties, [0, 0, 0, 0, 1, 1])
    root = model.forest[0]
    assert root.owner == 'telecom'
    assert root.left.weight == pytest.approx(-1)
    assert root.right.weight == pytest.approx(2/3)
    assert model.audit[0]['gain'] == pytest.approx(1.1333333333)
    assert not hasattr(root, 'threshold') and not hasattr(root, 'feature')


def test_encrypted_histograms_match_same_plaintext_boosting_settings():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(40, 4))
    y = (x[:, 0] - x[:, 2] > 0).astype(int)
    parties = [HistogramParty('a', x[:, :2], bins=5), HistogramParty('b', x[:, 2:], bins=5)]
    encrypted = SecureBoostClassifier(trees=3, depth=2, key_bits=256, insecure_test_key=True).fit(parties, y)
    plain = SecureBoostClassifier(trees=3, depth=2, encrypted=False).fit(parties, y)
    np.testing.assert_allclose(encrypted.predict_proba(parties), plain.predict_proba(parties), atol=1e-7)
    assert (encrypted.predict_proba(parties).argmax(1) == y).mean() > .8


def test_inference_needs_remote_split_owner_and_nonfinite_data_rejected():
    with pytest.raises(ValueError):
        HistogramParty('a', [[np.nan]])
    parties = [HistogramParty('a', [[0], [0], [0], [0]]), HistogramParty('b', [[0], [1], [9], [10]])]
    model = SecureBoostClassifier(trees=1, min_leaf=1, encrypted=False).fit(parties, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        model.predict_proba([parties[0]])
