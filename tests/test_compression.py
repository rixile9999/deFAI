import math

import numpy as np
import pytest

from defai.aggregation import l2_norm
from defai.compression import (
    compress,
    compress_with_feedback,
    dense_bytes,
    quantize,
    relative_error,
    topk,
)


def update(seed=0, shapes=((6, 4), (10,), (3, 3))):
    rng = np.random.default_rng(seed)
    return {f"p{i}": rng.normal(size=s).astype(np.float32) for i, s in enumerate(shapes)}


# --------------------------------------------------------------------------------------- top-k
def test_topk_keeps_exactly_the_largest_entries_across_all_tensors():
    u = update()
    out, nbytes = topk(u, ratio=0.25)
    n = sum(a.size for a in u.values())  # 24 + 10 + 9 = 43
    keep = math.ceil(0.25 * n)
    flat_in = np.concatenate([a.ravel() for a in u.values()])
    flat_out = np.concatenate([a.ravel() for a in out.values()])
    assert np.count_nonzero(flat_out) == keep
    kept = np.flatnonzero(flat_out)
    assert set(kept) == set(np.argsort(-np.abs(flat_in))[:keep])           # they are the largest ones
    np.testing.assert_array_equal(flat_out[kept], flat_in[kept])           # kept values are exact
    assert nbytes == 8 * keep and nbytes < dense_bytes(u)                  # float32 value + uint32 index each
    assert all(out[k].shape == u[k].shape and out[k].dtype == u[k].dtype for k in u)


def test_topk_with_ratio_one_is_the_identity_and_tiny_ratios_keep_one_entry():
    u = update()
    out, nbytes = topk(u, 1.0)
    assert all(np.array_equal(out[k], u[k]) for k in u) and nbytes == 8 * sum(a.size for a in u.values())
    out, nbytes = topk(u, 1e-9)
    assert sum(np.count_nonzero(a) for a in out.values()) == 1 and nbytes == 8
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            topk(u, bad)


def test_topk_never_increases_the_norm_and_its_error_shrinks_as_the_ratio_grows():
    u = update(seed=3)
    errors = [relative_error(u, topk(u, r)[0]) for r in (0.1, 0.3, 0.6, 1.0)]
    assert errors == sorted(errors, reverse=True) and errors[-1] == 0.0
    assert l2_norm(topk(u, 0.3)[0]) <= l2_norm(u) + 1e-6


# ----------------------------------------------------------------------------------- quantisation
def test_quantization_is_unbiased_because_it_rounds_stochastically():
    u = {"w": np.linspace(-1, 1, 7, dtype=np.float32) + 0.013}  # values that do not sit on the grid
    rng = np.random.default_rng(0)
    mean = np.mean([quantize(u, 3, rng)[0]["w"] for _ in range(4000)], axis=0)
    np.testing.assert_allclose(mean, u["w"], atol=0.02)
    assert not np.allclose(quantize(u, 3, np.random.default_rng(1))[0]["w"], u["w"])  # but a single draw is lossy


def test_quantization_error_is_at_most_one_level_and_bytes_follow_the_bit_width():
    u = update(seed=5)
    for bits in (2, 4, 8, 16):
        out, nbytes = quantize(u, bits, np.random.default_rng(0))
        levels = 2 ** (bits - 1) - 1
        for k, a in u.items():
            assert np.max(np.abs(out[k] - a)) <= np.max(np.abs(a)) / levels * (1 + 1e-4)
        expected = sum(math.ceil(a.size * bits / 8) + 4 for a in u.values())
        assert nbytes == expected
    errors = [relative_error(u, quantize(u, b, np.random.default_rng(0))[0]) for b in (2, 4, 8, 16)]
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-3


def test_quantization_keeps_zero_tensors_and_rejects_silly_bit_widths():
    u = {"w": np.zeros(5, np.float32), "b": np.ones(3, np.float32)}
    out, _ = quantize(u, 8, np.random.default_rng(0))
    np.testing.assert_array_equal(out["w"], 0)
    for bad in (1, 17):
        with pytest.raises(ValueError):
            quantize(u, bad, np.random.default_rng(0))


# ---------------------------------------------------------------------------------- dispatch + EF
def test_compress_dispatches_and_none_costs_the_dense_size():
    u = update()
    out, nbytes = compress(u, "none")
    assert out is u and nbytes == dense_bytes(u)
    assert compress(u, "topk", ratio=0.5)[1] == topk(u, 0.5)[1]
    assert compress(u, "quantize", bits=4, rng=np.random.default_rng(0))[1] == quantize(u, 4, np.random.default_rng(0))[1]
    with pytest.raises(ValueError, match="unknown compression"):
        compress(u, "gzip")


def test_error_feedback_never_loses_anything_for_good():
    """sum of transmitted updates = sum of true updates - the final residual (telescoping)."""
    rng = np.random.default_rng(0)
    true = [{"w": rng.normal(size=50).astype(np.float64)} for _ in range(30)]
    residual, sent = None, np.zeros(50)
    for u in true:
        decoded, _, residual = compress_with_feedback(u, residual, "topk", ratio=0.1)
        sent += decoded["w"]
    np.testing.assert_allclose(sent, sum(u["w"] for u in true) - residual["w"], atol=1e-9)
    # without feedback, the mass dropped by top-k every round is simply gone
    plain = sum(compress(u, "topk", ratio=0.1)[0]["w"] for u in true)
    gap_with = np.linalg.norm(sum(u["w"] for u in true) - sent)
    gap_without = np.linalg.norm(sum(u["w"] for u in true) - plain)
    assert gap_with < 0.4 * gap_without


def test_relative_error_is_zero_for_identical_updates_and_defined_for_zero():
    u = update()
    assert relative_error(u, u) == 0.0
    assert relative_error({"w": np.zeros(3)}, {"w": np.ones(3)}) == 0.0  # nothing to measure against
