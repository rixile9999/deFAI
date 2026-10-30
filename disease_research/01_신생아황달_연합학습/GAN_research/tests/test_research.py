from collections import OrderedDict
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "NeoJaundice"))
from gan_models import Generator, Discriminator, augment
from src.federated import average_states
from src.model import MultiModalNet
from run_federated_study import extras, patients_table


def test_patient_and_image_split_disjointness():
    import pandas as pd
    frame = pd.read_csv(ROOT / "data/split_manifest.csv", dtype={"patient_id": str})
    assert frame.groupby("patient_id")["split"].nunique().max() == 1
    assert frame.groupby("patient_id")["client"].nunique().max() == 1
    assert frame.sha256.is_unique
    assert (frame["blood(mg/dL)"] > 0).all()
    assert np.isfinite(frame[["gestational_age", "age(day)", "weight"]]).all().all()
    assert set(frame.split) == {"train", "val", "test"}


def test_known_weighted_average_and_integer_buffer():
    a = OrderedDict(w=torch.tensor([1., 4.]), counter=torch.tensor(0))
    b = OrderedDict(w=torch.tensor([5., 8.]), counter=torch.tensor(0))
    result = average_states([a, b], [1, 3])
    torch.testing.assert_close(result["w"], torch.tensor([4., 7.]))
    assert result["counter"].dtype == torch.int64
    b["counter"] = torch.tensor(2)
    with pytest.raises(ValueError, match="Nonfloating"):
        average_states([a, b], [1, 3])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_client_update_rejected(bad):
    with pytest.raises(ValueError, match="Nonfinite"):
        average_states([{"w": torch.tensor([bad])}], [1])


def test_singleton_batch_and_common_bn_statistics():
    torch.set_num_threads(2)
    model = MultiModalNet(pretrained=False).train()
    assert all(not m.training for m in model.cnn.modules() if isinstance(m, torch.nn.BatchNorm2d))
    prediction, logits = model(torch.randn(1, 3, 64, 64), torch.ones(1, 4))
    (prediction.sum()+logits.sum()).backward()
    assert prediction.shape == (1, 1) and torch.isfinite(prediction).all()
    assert all(int(m.num_batches_tracked) == 0 for m in model.cnn.modules() if isinstance(m, torch.nn.BatchNorm2d))


def test_gan_output_and_differentiable_augmentation():
    torch.set_num_threads(2)
    generator, discriminator = Generator(), Discriminator()
    image = generator(torch.randn(2, 96), torch.randn(2, 6))
    score, target = discriminator(augment(image), torch.randn(2, 6))
    (score.mean()+target.mean()).backward()
    assert image.shape == (2, 3, 64, 64)
    assert float(image.detach().abs().max()) <= 1
    assert generator.input.weight.grad is not None


def test_synthetic_test_or_other_client_provenance_rejected():
    data = {"split": np.array(["train", "test"]), "client": np.array([0, 1])}
    with pytest.raises(ValueError, match="provenance"):
        extras(data, {"source_index": np.array([1]), "client": np.array([1])})
    with pytest.raises(ValueError, match="provenance"):
        extras(data, {"source_index": np.array([0]), "client": np.array([2])})


def test_patient_predictions_are_not_three_independent_patients():
    data = {"patient": np.array(["A", "A", "B"]), "client": np.array([0, 0, 1]),
            "blood": np.array([10., 10., 20.]), "treatment": np.array([0, 0, 1])}
    table = patients_table(data, np.arange(3), [8., 12., 20.], [.2, .4, .9])
    assert len(table) == 2
    assert table.loc[table.patient == "A", "prediction"].iloc[0] == 10


def test_generated_data_only_from_local_training_patients():
    path = ROOT / "data/gan_synthetic.npz"
    if not path.exists():
        pytest.skip("Generator run still in progress")
    data = np.load(ROOT / "data/real_data.npz"); synthetic = np.load(path)
    indices = synthetic["source_index"]
    assert (data["split"][indices] == "train").all()
    np.testing.assert_array_equal(data["client"][indices], synthetic["client"])
    assert len(indices) == int((data["split"] == "train").sum())


def test_original_loader_now_splits_patients_not_photographs(tmp_path):
    import pandas as pd
    from src.utils import load_client_data_from_debug
    folder = tmp_path / "client_0"; folder.mkdir()
    pd.DataFrame({"patient_id": ["A"]*3+["B"]*3+["C"]*3}).to_csv(folder / "metadata.csv", index=False)
    training, validation, _ = load_client_data_from_debug(0, str(tmp_path))
    assert set(training.patient_id).isdisjoint(validation.patient_id)


def test_paired_generator_retains_bounded_source_structure():
    from train_paired_gan import PairedGenerator
    generator = PairedGenerator()
    source = torch.rand(2, 3, 64, 64)*2-1
    generated = generator(source, torch.randn(2, 16), torch.randn(2, 6))
    assert float((generated-source).detach().abs().max()) <= .15001
    assert generated.shape == source.shape
    generated.mean().backward()
    assert generator.style.weight.grad is not None
