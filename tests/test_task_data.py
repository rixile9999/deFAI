"""Data layer of defai.task: the synthetic dataset, partitioning and the evaluation loop."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from defai import task

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
NUM_PARTITIONS = 10


def config(**overrides):
    base = tomllib.loads(PYPROJECT.read_text())["tool"]["flwr"]["app"]["config"]
    return {**base, "dataset": "synthetic", "batch-size": 32, **overrides}


def labels_of(partition_id, cfg):
    part = task._partition_source(NUM_PARTITIONS, cfg).load_partition(partition_id)
    return np.asarray(part["label"])


# ------------------------------------------------------------------------------ synthetic dataset
def test_synthetic_dataset_is_balanced_deterministic_and_image_shaped():
    train, test = task._synthetic_split("train"), task._synthetic_split("test")
    assert (len(train), len(test)) == (3000, 1000)
    assert np.bincount(train["label"], minlength=10).tolist() == [300] * 10
    assert np.bincount(test["label"], minlength=10).tolist() == [100] * 10
    assert train[0]["image"].size == (28, 28) and train[0]["image"].mode == "L"
    task._SYNTHETIC.clear()  # regenerate from scratch: same seeds -> identical data
    again = task._synthetic_split("train")
    assert list(again["label"]) == list(train["label"])
    assert np.array_equal(np.asarray(again[5]["image"]), np.asarray(train[5]["image"]))


def test_classes_have_distinct_prototypes_so_the_task_is_learnable():
    train = task._synthetic_split("train")
    means = np.stack([
        np.mean([np.asarray(train[i]["image"], dtype=float) for i in np.flatnonzero(np.asarray(train["label"]) == c)[:40]], axis=0)
        for c in range(10)
    ])
    distances = [np.linalg.norm(means[a] - means[b]) for a in range(10) for b in range(a + 1, 10)]
    assert min(distances) > 5.0  # prototypes differ far more than the per-pixel noise averages out to


# ------------------------------------------------------------------------------------ partitioning
def test_iid_partitions_are_equal_sized_disjoint_and_cover_the_train_set():
    cfg = config(partitioner="iid")
    sizes = [len(labels_of(i, cfg)) for i in range(NUM_PARTITIONS)]
    assert sum(sizes) == 3000 and max(sizes) - min(sizes) <= 1
    # every partition sees all classes roughly uniformly
    assert all(len(np.unique(labels_of(i, cfg))) == 10 for i in range(NUM_PARTITIONS))


def label_skew(cfg):
    """Mean over clients of the share of the client's most frequent class (0.1 = uniform, 1 = one class)."""
    return float(np.mean([np.bincount(l, minlength=10).max() / len(l) for l in (labels_of(i, cfg) for i in range(NUM_PARTITIONS))]))


def test_smaller_dirichlet_alpha_means_more_label_skew():
    skew = {a: label_skew(config(partitioner="dirichlet", **{"dirichlet-alpha": a})) for a in (0.05, 0.5, 100.0)}
    assert skew[0.05] > skew[0.5] > skew[100.0]
    assert skew[0.05] > 0.5 and skew[100.0] < 0.2
    assert label_skew(config(partitioner="iid")) < 0.2  # IID is the uniform limit


def test_partitioning_is_reproducible_for_a_seed_and_differs_across_seeds():
    cfg = config(partitioner="dirichlet", **{"dirichlet-alpha": 0.3})
    first = [labels_of(i, cfg).tolist() for i in range(3)]
    task._sources.clear()
    assert [labels_of(i, cfg).tolist() for i in range(3)] == first
    other = config(partitioner="dirichlet", seed=cfg["seed"] + 1, **{"dirichlet-alpha": 0.3})
    assert [labels_of(i, other).tolist() for i in range(3)] != first


def test_partition_sources_are_cached_per_configuration():
    task._sources.clear()
    a = task._partition_source(NUM_PARTITIONS, config(partitioner="dirichlet"))
    assert task._partition_source(NUM_PARTITIONS, config(partitioner="dirichlet")) is a
    assert task._partition_source(NUM_PARTITIONS, config(partitioner="dirichlet", **{"dirichlet-alpha": 0.1})) is not a
    assert task._partition_source(NUM_PARTITIONS + 1, config(partitioner="dirichlet")) is not a


def test_an_unknown_partitioner_is_an_error():
    with pytest.raises(ValueError, match="unknown partitioner"):
        task._partition_source(NUM_PARTITIONS, config(partitioner="pathological"))


# ------------------------------------------------------------------------------------------ loaders
def test_client_loaders_hold_out_twenty_percent_and_yield_model_ready_batches():
    cfg = config(partitioner="iid")
    train, val = task.load_client_data(0, NUM_PARTITIONS, cfg)
    n_train, n_val = len(train.dataset), len(val.dataset)
    assert n_train + n_val == len(labels_of(0, cfg)) and abs(n_val / (n_train + n_val) - 0.2) < 0.01
    batch = next(iter(train))
    assert batch["x"].shape == (32, 1, 28, 28) and batch["x"].dtype == torch.float32
    assert batch["y"].shape == (32,) and batch["y"].dtype == torch.int64
    assert float(batch["x"].min()) >= -1.0 and float(batch["x"].max()) <= 1.0  # Normalize((0.5,), (0.5,))


def test_centralized_testset_is_the_full_test_split():
    loader = task.load_centralized_testset("synthetic", batch_size=256)
    assert len(loader.dataset) == 1000 and next(iter(loader))["x"].shape[0] == 256


# ----------------------------------------------------------------------------------- evaluation loop
class Lookup(nn.Module):
    """Predicts class ``round(x_mean)`` with confidence ``margin``: logits = margin * one_hot(prediction)."""

    def __init__(self, margin=3.0, classes=2):
        super().__init__()
        self.margin, self.classes = margin, classes
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        pred = x.flatten(1).mean(1).round().long().clamp(0, self.classes - 1)
        return self.margin * torch.nn.functional.one_hot(pred, self.classes).float() + self.dummy


def test_test_reports_mean_loss_and_accuracy_over_the_whole_loader():
    batches = [
        {"x": torch.tensor([[0.0], [1.0], [1.0]]), "y": torch.tensor([0, 1, 0])},  # last one is wrong
        {"x": torch.tensor([[0.0]]), "y": torch.tensor([0])},
    ]
    loss, acc = task.test(Lookup(), batches, torch.device("cpu"))
    assert acc == pytest.approx(3 / 4)
    right, wrong = np.log1p(np.exp(-3.0)), np.log1p(np.exp(3.0))  # CE of a 2-class margin-3 prediction
    assert loss == pytest.approx((3 * right + wrong) / 4, rel=1e-5)


def test_test_leaves_the_model_in_eval_mode_and_does_not_track_gradients():
    model = Lookup()
    batches = [{"x": torch.zeros(2, 1), "y": torch.zeros(2, dtype=torch.long)}]
    task.test(model, batches, torch.device("cpu"))
    assert not model.training and model.dummy.grad is None


def test_get_device_prefers_cuda_only_when_available():
    assert task.get_device().type == ("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------- partition statistics
def test_partition_label_counts_match_the_partitions_and_the_dataset():
    cfg = config(partitioner="dirichlet", **{"dirichlet-alpha": 0.2})
    counts = task.partition_label_counts(NUM_PARTITIONS, cfg)
    assert counts.shape == (NUM_PARTITIONS, task.NUM_CLASSES)
    for i in range(NUM_PARTITIONS):  # cross-checked against an independent read of each partition
        assert counts[i].tolist() == np.bincount(labels_of(i, cfg), minlength=10).tolist()
    assert counts.sum(axis=0).tolist() == [300] * 10  # every example of the train set belongs to exactly one client


def test_label_skew_is_one_over_classes_when_uniform_and_one_when_each_client_has_one_class():
    assert task.label_skew(np.full((4, 10), 7)) == pytest.approx(0.1)
    assert task.label_skew(np.eye(10, dtype=int) * 5) == pytest.approx(1.0)
    assert task.label_skew([[3, 1], [0, 4]]) == pytest.approx((0.75 + 1.0) / 2)
    assert task.label_skew(task.partition_label_counts(NUM_PARTITIONS, config(partitioner="dirichlet"))) == pytest.approx(
        label_skew(config(partitioner="dirichlet"))
    )


# ----------------------------------------------------------------------- mirrored real datasets
@pytest.fixture
def fake_mirror(monkeypatch):
    """defai.mirror.load_split replaced by small random 'images', so the tests need no download."""
    sizes = {"train": 400, "test": 100}
    calls = []

    def load_split(name, split):
        calls.append((name, split))
        rng = np.random.default_rng({"train": 1, "test": 2}[split])
        labels = (np.arange(sizes[split]) % 10).astype(np.uint8)
        images = rng.integers(0, 256, (sizes[split], 28, 28), dtype=np.uint8)
        images[:, 0, 0] = labels  # the label is readable from the first pixel
        images[:, 0, 1] = np.arange(sizes[split]) % 256  # and the position in the file from the next two
        images[:, 0, 2] = np.arange(sizes[split]) // 256
        return images, labels

    task._MIRRORED.clear()
    monkeypatch.setattr(task.mirror, "load_split", load_split)
    yield calls
    task._MIRRORED.clear()


@pytest.mark.parametrize("dataset", ["mnist-mirror", "fashion-mnist-mirror"])
def test_mirrored_datasets_are_partitioned_and_yield_model_ready_batches(fake_mirror, dataset):
    cfg = config(dataset=dataset, partitioner="iid")
    sizes = [len(labels_of(i, cfg)) for i in range(NUM_PARTITIONS)]
    assert sum(sizes) == 400 and fake_mirror == [(dataset, "train")]
    assert task._mirror_split(dataset, "train") is task._mirror_split(dataset, "train") and len(fake_mirror) == 1  # cached
    trainloader, valloader = task.load_client_data(0, NUM_PARTITIONS, cfg)
    assert (len(trainloader.dataset), len(valloader.dataset)) == (32, 8)
    batch = next(iter(trainloader))
    assert batch["x"].shape == (32, 1, 28, 28) and batch["x"].dtype == torch.float32 and batch["y"].dtype == torch.int64
    mean, std = task.DATASETS[dataset]["norm"][0][0], task.DATASETS[dataset]["norm"][1][0]
    # pixel p becomes (p / 255 - mean) / std
    assert float(batch["x"].min()) >= -mean / std - 1e-6 and float(batch["x"].max()) <= (1 - mean) / std + 1e-6
    labels = batch["y"].numpy()
    first_pixel = np.rint((batch["x"][:, 0, 0, 0].numpy() * std + mean) * 255)
    assert np.array_equal(first_pixel, labels)  # images and labels stay paired through partitioning and transform


def test_mirrored_datasets_work_with_the_dirichlet_partitioner_and_the_central_test_set(fake_mirror):
    cfg = config(dataset="mnist-mirror", partitioner="dirichlet", **{"dirichlet-alpha": 0.1})
    assert label_skew(cfg) > 0.4
    test = task.load_centralized_testset("mnist-mirror")
    assert len(test.dataset) == 100 and next(iter(test))["x"].shape == (100, 1, 28, 28)
    assert ("mnist-mirror", "test") in fake_mirror


def test_every_dataset_is_either_from_the_hub_a_mirror_or_synthetic():
    for name, spec in task.DATASETS.items():
        assert bool(spec["hf_name"]) + bool(spec.get("mirror")) + (name == "synthetic") == 1, name
        assert (spec.get("mirror") is True) == (name in task.mirror.MIRRORS), name


def test_the_training_split_is_shuffled_so_iid_partitions_are_not_contiguous_slices(fake_mirror):
    cfg = config(dataset="mnist-mirror", partitioner="iid")
    part = task._partition_source(NUM_PARTITIONS, cfg).load_partition(0)
    positions = np.array([image[0][1] + 256 * image[0][2] for image in part["image"]])
    assert len(positions) == 40 and not np.array_equal(np.sort(positions), np.arange(40))
    assert positions.max() > 100  # drawn from the whole file, not from its first tenth
    task._MIRRORED.clear()  # the shuffle uses a fixed seed: rebuilding gives the same order
    again = task._partition_source(NUM_PARTITIONS, cfg).load_partition(0)
    assert [image[0][1] for image in again["image"]] == [image[0][1] for image in part["image"]]
    # the test split keeps the order of the file
    assert [image[0][1] for image in task._mirror_split("mnist-mirror", "test")[:5]["image"]] == [0, 1, 2, 3, 4]
