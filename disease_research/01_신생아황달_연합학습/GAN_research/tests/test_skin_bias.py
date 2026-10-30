import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skin_bias import normalize, denormalize, attenuate, suppress_brightness
from run_skin_bias_study import patient_groups
from summarize_skin_bias import error_metrics


def test_gain_retains_yellow_relative_channel_signal():
    rgb = torch.rand(4, 3, 8, 8)*.8+.1
    changed = denormalize(attenuate(normalize(rgb), .55))
    torch.testing.assert_close(changed[:, 0]/changed[:, 2], rgb[:, 0]/rgb[:, 2])
    torch.testing.assert_close(changed, rgb*.55)
    with pytest.raises(ValueError):
        attenuate(normalize(rgb), 1.2)


def test_chromaticity_is_gain_invariant_and_finite_at_black():
    rgb = torch.rand(4, 3, 8, 8)
    rgb[0, :, 0, 0] = 0
    image = normalize(rgb)
    reference = suppress_brightness(image, "chromaticity")
    for gain in [.55, .75, .9, 1.]:
        changed = suppress_brightness(attenuate(image, gain), "chromaticity")
        # Float32 denormalization roundoff at zero is tolerated separately.
        torch.testing.assert_close(changed[:, :, 1:, 1:], reference[:, :, 1:, 1:], atol=2e-5, rtol=2e-5)
        assert torch.isfinite(changed).all()


def test_group_cutoffs_never_fit_on_test_images():
    data = {"images": np.full((6, 2, 2, 3), 10, dtype=np.uint8),
            "patient": np.array(["A", "A", "B", "C", "D", "E"]),
            "split": np.array(["train", "train", "train", "train", "val", "test"])}
    data["images"][2] = 80; data["images"][3] = 150
    _, first = patient_groups(data)
    data["images"][-1] = 255
    groups, second = patient_groups(data)
    np.testing.assert_array_equal(first, second)
    assert len(groups) == 5


def test_bootstrap_gap_is_recomputed_not_mean_patient_difference():
    groups = np.array([0, 0, 1, 1, 2, 2])
    errors = np.array([1., 1., 2., 2., 4., 4.])
    np.testing.assert_allclose(error_metrics(errors, groups), [7/3, 4., 3.])
    rows = np.tile(errors, (3, 1))
    np.testing.assert_allclose(error_metrics(rows, groups), np.tile([7/3, 4., 3.], (3, 1)))
