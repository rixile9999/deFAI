import numpy as np
import pytest
import torch
import torch.nn.functional as F
from cryptography.exceptions import InvalidTag

from defai_addons.compare.transport import UpdateCipher
from defai_addons.compare.privacy import GaussianDP
from defai_addons.compare.runner import DEFAULTS
from defai_addons.compare.envs import FederatedEnv, SiteData
from defai_addons.multimodal.generation import AutoregressiveSequenceDecoder
from defai_addons.multimodal.models import MultimodalTransformer
from defai_addons.multimodal import data as MD
from defai_addons.multimodal.train import loader


def test_encrypted_updates_authenticate_identity_round_and_content():
    cipher = UpdateCipher()
    vec = np.arange(12, dtype=np.float64)
    a = cipher.encrypt(vec, '2', 3)
    b = cipher.encrypt(vec, '2', 3)
    assert not np.array_equal(a, b)
    assert a.nbytes == vec.nbytes + 28
    assert np.array_equal(cipher.decrypt(a, '2', 3, 12), vec)
    changed = a.copy()
    changed[-1] ^= 1
    for packet, cid, rnd in [(changed, '2', 3), (a, '1', 3), (a, '2', 4)]:
        with pytest.raises(InvalidTag):
            cipher.decrypt(packet, cid, rnd, 12)
    with pytest.raises(InvalidTag):
        UpdateCipher().decrypt(a, '2', 3, 12)


def test_weighted_dp_noise_uses_replace_one_sensitivity():
    a = GaussianDP(0.7, clip=2.0, seed=42)
    b = GaussianDP(0.7, clip=2.0, seed=42)
    uniform = a.noise(100, 4)
    weighted = b.noise(100, 4, max_weight=0.75)
    assert torch.allclose(weighted, 3 * uniform)
    assert b.describe()['sensitivity'] == 3.0
    assert b.epsilon() == a.epsilon()
    assert b.describe()['scope'] == 'model_updates_conditional_on_fixed_preprocessing'


def test_encrypted_flower_rounds_match_plain_and_count_payload():
    rng = np.random.RandomState(9)
    sites = [SiteData(rng.randn(n, 8).astype('float32'), rng.randint(0, 3, n),
                      rng.randn(10, 8).astype('float32'), rng.randint(0, 3, 10)) for n in (30, 60, 90)]
    cfg = dict(DEFAULTS, epochs=2, hidden=[8], use_pca=False)
    plain = FederatedEnv('plain', sites, sites[0].Xte, sites[0].yte, 3,
                         cfg=dict(cfg, encrypt_updates=False))
    encrypted = FederatedEnv('encrypted', sites, sites[0].Xte, sites[0].yte, 3,
                             cfg=dict(cfg, encrypt_updates=True))
    plain.fit()
    encrypted.fit()
    assert all(torch.equal(a, b) for a, b in zip(plain.model.parameters(), encrypted.model.parameters()))
    assert encrypted.cost['model_bytes'] - plain.cost['model_bytes'] == 28 * 3 * 2
    assert encrypted.cost['encryption_bytes'] == 28 * 3 * 2
    assert encrypted.cost['raw_bytes'] == 0
    assert encrypted.secure_agg_error < 1e-6


def test_causal_generation_cannot_read_future_or_padded_memory():
    torch.manual_seed(2)
    dec = AutoregressiveSequenceDecoder(16, 12, dropout=0.0).eval()
    memory = torch.randn(2, 7, 16)
    pad = torch.zeros(2, 7, dtype=torch.bool)
    pad[:, -2:] = True
    ids = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8]])
    first = dec(ids, memory, pad)
    changed = ids.clone()
    changed[:, 2:] = 9
    assert torch.allclose(first[:, :2], dec(changed, memory, pad)[:, :2], atol=1e-6)
    noisy = memory.clone()
    noisy[:, -2:] = 1000 * torch.randn_like(noisy[:, -2:])
    assert torch.allclose(first, dec(ids, noisy, pad), atol=1e-6)
    visible = memory.clone()
    visible[:, :5] += torch.randn_like(visible[:, :5]) * 4
    assert not torch.allclose(first, dec(ids, visible, pad), atol=1e-4)


def test_generation_stops_at_eos_and_rejects_invalid_length():
    dec = AutoregressiveSequenceDecoder(16, 12, max_len=8, dropout=0.0).eval()
    with torch.no_grad():
        dec.output.weight.zero_()
        dec.output.bias.zero_()
        dec.output.bias[2] = 100
    memory = torch.randn(2, 4, 16)
    pad = torch.zeros(2, 4, dtype=torch.bool)
    assert torch.equal(dec.generate(memory, pad, 1, 2, 8), torch.full((2, 1), 2))
    with pytest.raises(ValueError):
        dec.generate(memory, pad, 1, 2, 9)


def test_multimodal_sequence_head_trains_and_preserves_default_model():
    torch.manual_seed(10)
    default = MultimodalTransformer(d=16, dropout=0.0)
    torch.manual_seed(10)
    model = MultimodalTransformer(d=16, dropout=0.0, sequence_vocab=12, sequence_max_len=8)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in default.state_dict().items())
    batch = next(iter(loader(MD.generate(4, seed=3), 4, shuffle=False)))
    model.eval()
    out = model(batch, torch.tensor([[1, 3, 4]] * 4))
    assert out['representation'].shape == (4, 48)
    assert torch.allclose(out['probabilities'].sum(-1), torch.ones(4))
    assert out['sequence_logits'].shape == (4, 3, 12)
    assert torch.allclose(default.eval()(batch)['logits'], out['logits'])
    loss = F.cross_entropy(out['sequence_logits'].reshape(-1, 12), torch.tensor([3, 4, 2] * 4))
    loss.backward()
    assert model.sequence_decoder.output.weight.grad.abs().sum() > 0
    assert model.integration.type_emb.weight.grad.abs().sum() > 0
    model.train()
    generated = model.generate(batch, 1, 2, max_new_tokens=4)
    assert model.training and generated.size(0) == 4 and 1 <= generated.size(1) <= 4


def test_authenticated_strategy_rejects_missing_duplicate_and_stale_updates():
    from flwr.common import parameters_to_ndarrays
    from defai_addons.fl import flower as FL
    from defai_addons.compare.envs import SecureAggFedAvg
    ciphers = {'0': UpdateCipher(), '1': UpdateCipher()}
    init = [np.zeros(3, dtype=np.float32)]
    strategy = SecureAggFedAvg(init, ciphers=ciphers)
    results = [(FL.LocalClientProxy(cid, None),
                FL.fit_res([c.encrypt(np.ones(3), cid, 1)], 10, {})) for cid, c in ciphers.items()]
    with pytest.raises(ValueError):
        strategy.aggregate_fit(1, results[:1], [])
    with pytest.raises(ValueError):
        strategy.aggregate_fit(1, [results[0], results[0]], [])
    with pytest.raises(InvalidTag):
        strategy.aggregate_fit(2, results, [])
    assert np.allclose(parameters_to_ndarrays(strategy.aggregate_fit(1, results, [])[0])[0], 2)
