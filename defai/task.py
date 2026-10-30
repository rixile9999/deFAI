"""Model, data loading, and local train/test logic shared by ServerApp and ClientApp."""

from dataclasses import dataclass
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Array2D, ClassLabel, Dataset, Features, load_dataset
from datasets import Image as HFImage
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from defai import mirror
from defai.attacks import flip_labels

DATASETS = {
    "mnist": {
        "hf_name": "ylecun/mnist",
        "image_key": "image",
        "channels": 1,
        "img_size": 28,
        "norm": ((0.1307,), (0.3081,)),
    },
    "cifar10": {
        "hf_name": "uoft-cs/cifar10",
        "image_key": "img",
        "channels": 3,
        "img_size": 32,
        "norm": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    },
    # The same data as `mnist` and, for Fashion-MNIST, a harder 28x28 task, fetched from file mirrors (defai/mirror.py)
    # instead of the Hugging Face hub
    "mnist-mirror": {
        "hf_name": None,
        "image_key": "image",
        "channels": 1,
        "img_size": 28,
        "norm": ((0.1307,), (0.3081,)),
        "mirror": True,
    },
    "fashion-mnist-mirror": {
        "hf_name": None,
        "image_key": "image",
        "channels": 1,
        "img_size": 28,
        "norm": ((0.2860,), (0.3530,)),
        "mirror": True,
    },
    # Generated locally (no download): lets tests and smoke runs work offline.
    "synthetic": {
        "hf_name": None,
        "image_key": "image",
        "channels": 1,
        "img_size": 28,
        "norm": ((0.5,), (0.5,)),
    },
}
NUM_CLASSES = 10


class Net(nn.Module):
    """Small CNN used in the FedAvg paper (McMahan et al., 2017)."""

    def __init__(self, in_channels=1, num_classes=10, img_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = 64 * (img_size // 4) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(feat, 512), nn.ReLU(), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def represent(self, x):
        """Penultimate-layer representation (input of the linear head); used by MOON."""
        return self.classifier[:3](self.features(x))

    def head(self, z):
        return self.classifier[3](z)


def build_model(dataset):
    spec = DATASETS[dataset]
    return Net(in_channels=spec["channels"], img_size=spec["img_size"])


def get_device():
    device = os.environ.get("DEFAI_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("DEFAI_DEVICE must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(device)


def _make_transform(dataset):
    spec = DATASETS[dataset]
    tf = Compose([ToTensor(), Normalize(*spec["norm"])])
    key = spec["image_key"]

    def apply(batch):
        images = batch.pop(key)
        if spec.get("mirror"):  # stored as plain arrays rather than encoded images
            images = [np.asarray(img, dtype=np.uint8) for img in images]
        batch["x"] = [tf(img) for img in images]
        batch["y"] = batch.pop("label")
        return batch

    return apply


_SYNTHETIC = {}
_MIRRORED = {}
_sources = {}  # partition sources cached per ClientApp process, keyed by their configuration


def _synthetic_split(split):
    """Deterministic MNIST-like images: one random 7x7 binary pattern per class plus noise."""
    if split not in _SYNTHETIC:
        n = {"train": 3000, "test": 1000}[split]
        patterns = np.kron(np.random.default_rng(0).random((NUM_CLASSES, 7, 7)) > 0.5, np.ones((4, 4)))
        rng = np.random.default_rng({"train": 1, "test": 2}[split])
        labels = rng.permutation(np.arange(n) % NUM_CLASSES)
        pixels = np.clip(0.8 * patterns[labels] + rng.normal(0, 0.35, (n, 28, 28)), 0, 1)
        _SYNTHETIC[split] = Dataset.from_dict(
            {
                "image": [Image.fromarray((p * 255).astype(np.uint8), "L") for p in pixels],
                "label": labels.tolist(),
            },
            features=Features({"image": HFImage(), "label": ClassLabel(num_classes=NUM_CLASSES)}),
        )
    return _SYNTHETIC[split]


def _mirror_split(dataset, split):
    """MNIST-like data from defai.mirror as a Hugging Face Dataset (so the Flower partitioners can split it)."""
    if (dataset, split) not in _MIRRORED:
        images, labels = mirror.load_split(dataset, split)
        data = Dataset.from_dict(
            {"image": images, "label": labels.tolist()},
            features=Features({"image": Array2D((mirror.IMG_SIZE, mirror.IMG_SIZE), "uint8"), "label": ClassLabel(num_classes=NUM_CLASSES)}),
        )
        if split == "train":
            # FederatedDataset shuffles the training split (seed 42) before partitioning; the files are not in
            # random order (MNIST is stored in blocks of writers), so an IID split would otherwise be contiguous slices
            data = data.shuffle(seed=42)
        _MIRRORED[(dataset, split)] = data
    return _MIRRORED[(dataset, split)]


def _local_split(dataset, split):
    """The split of a dataset that does not come from the Hugging Face hub, or None if it does."""
    if dataset == "synthetic":
        return _synthetic_split(split)
    if DATASETS[dataset].get("mirror"):
        return _mirror_split(dataset, split)
    return None


def _partition_source(num_partitions, cfg):
    """Object with ``load_partition(partition_id)`` for the configured dataset/partitioner."""
    key = (cfg["dataset"], cfg["partitioner"], cfg["dirichlet-alpha"], cfg["seed"], num_partitions)
    if key not in _sources:
        if cfg["partitioner"] == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif cfg["partitioner"] == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=cfg["dirichlet-alpha"],
                seed=cfg["seed"],
            )
        else:
            raise ValueError(f"unknown partitioner: {cfg['partitioner']}")
        local = _local_split(cfg["dataset"], "train")
        if local is not None:
            partitioner.dataset = local
            _sources[key] = partitioner
        else:
            _sources[key] = FederatedDataset(
                dataset=DATASETS[cfg["dataset"]]["hf_name"], partitioners={"train": partitioner}
            )
    return _sources[key]


def partition_label_counts(num_partitions, cfg):
    """(num_partitions, NUM_CLASSES) array: how many examples of each class every client holds."""
    source = _partition_source(num_partitions, cfg)
    counts = np.zeros((num_partitions, NUM_CLASSES), dtype=np.int64)
    for i in range(num_partitions):
        counts[i] = np.bincount(np.asarray(source.load_partition(i)["label"]), minlength=NUM_CLASSES)
    return counts


def label_skew(counts):
    """Mean over clients of the share of the client's most frequent class (1 / classes = uniform, 1 = one class)."""
    counts = np.asarray(counts, dtype=np.float64)
    return float(np.mean(counts.max(axis=1) / counts.sum(axis=1)))


def load_client_data(partition_id, num_partitions, cfg):
    """Return (trainloader, valloader) for this client's partition."""
    partition = _partition_source(num_partitions, cfg).load_partition(partition_id)
    # Each client holds out 20% of its local data for federated evaluation
    split = partition.train_test_split(test_size=0.2, seed=cfg["seed"])
    split = split.with_transform(_make_transform(cfg["dataset"]))
    trainloader = DataLoader(split["train"], batch_size=cfg["batch-size"], shuffle=True)
    valloader = DataLoader(split["test"], batch_size=cfg["batch-size"])
    return trainloader, valloader


def load_centralized_testset(dataset, batch_size=256):
    """Full test split, used by the ServerApp for centralized evaluation."""
    test = _local_split(dataset, "test")
    if test is None:
        test = load_dataset(DATASETS[dataset]["hf_name"], split="test")
    return DataLoader(test.with_transform(_make_transform(dataset)), batch_size=batch_size)


@dataclass
class MoonReference:
    """What MOON's contrastive loss compares the training model against."""

    global_model: nn.Module  # model received from the server (frozen)
    previous_model: nn.Module  # this client's model from its previous round (frozen)
    mu: float
    temperature: float


def moon_contrastive_loss(z, z_global, z_previous, temperature):
    """-log( e^{sim(z,z_g)/t} / (e^{sim(z,z_g)/t} + e^{sim(z,z_p)/t}) ), averaged over the batch."""
    logits = torch.stack(
        [F.cosine_similarity(z, z_global, dim=1), F.cosine_similarity(z, z_previous, dim=1)], dim=1
    ) / temperature
    positive_is_first = torch.zeros(len(z), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, positive_is_first)


def train(
    model,
    trainloader,
    epochs,
    lr,
    device,
    proximal_mu=0.0,
    *,
    grad_correction=None,
    moon=None,
    flip_num_classes=None,
):
    """Local SGD with optional federated-algorithm hooks.

    proximal_mu       FedProx: add mu/2 * ||w - w_global||^2 to the loss.
    grad_correction   SCAFFOLD: tensors (one per ``model.parameters()``) added to every
                      gradient, i.e. ``c - c_i`` (server minus client control variate).
    moon              MOON: a ``MoonReference``; adds ``mu *`` the contrastive loss.
    flip_num_classes  label-flip attack: train on ``flip_num_classes - 1 - y``.

    Returns (mean task loss, number of optimiser steps); the task loss excludes the
    FedProx/MOON regularisers so it is comparable across algorithms.
    """
    model.to(device)
    model.train()
    global_params = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if grad_correction is not None:
        grad_correction = [c.to(device) for c in grad_correction]

    total_loss, steps = 0.0, 0
    for _ in range(epochs):
        for batch in trainloader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            if flip_num_classes is not None:
                y = flip_labels(y, flip_num_classes)
            optimizer.zero_grad()
            if moon is None:
                task_loss = F.cross_entropy(model(x), y)
                loss = task_loss
            else:
                z = model.represent(x)
                task_loss = F.cross_entropy(model.head(z), y)
                with torch.no_grad():
                    z_global = moon.global_model.represent(x)
                    z_previous = moon.previous_model.represent(x)
                loss = task_loss + moon.mu * moon_contrastive_loss(z, z_global, z_previous, moon.temperature)
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * proximal_mu * prox
            loss.backward()
            if grad_correction is not None:
                for p, c in zip(model.parameters(), grad_correction):
                    if p.grad is not None:
                        p.grad.add_(c)
            optimizer.step()
            total_loss += task_loss.item()
            steps += 1
    return total_loss / max(steps, 1), steps


def scaffold_control_update(c_local, c_global, x_global, y_local, steps, lr):
    """SCAFFOLD option II: ``c_i^+ = c_i - c + (x - y) / (K * lr)`` (dicts of ndarrays)."""
    if steps == 0 or lr == 0:
        return {k: v.copy() for k, v in c_local.items()}
    return {
        k: c_local[k] - c_global[k] + (x_global[k] - y_local[k]) / (steps * lr) for k in c_local
    }


@torch.no_grad()
def test(model, loader, device):
    model.to(device)
    model.eval()
    correct, loss, total = 0, 0.0, 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss / total, correct / total
