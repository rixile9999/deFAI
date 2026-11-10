import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASETS = {
    "synthetic": {"channels": 1, "size": 28, "classes": 10},
    "mnist": {"channels": 1, "size": 28, "classes": 10},
    "fashion-mnist": {"channels": 1, "size": 28, "classes": 10},
    "cifar10": {"channels": 3, "size": 32, "classes": 10},
}
MODELS = ("cnn", "cnn_bn", "resnet8")

_CACHE = {}
_LOCK = threading.RLock()


def data_dir():
    return Path(os.environ.get("FL_DATA_DIR", Path.cwd() / "data"))


def _synthetic(split):
    n = {"train": 6000, "test": 2000}[split]
    patterns = np.kron(np.random.default_rng(0).random((10, 7, 7)) > 0.5, np.ones((4, 4)))
    rng = np.random.default_rng({"train": 1, "test": 2}[split])
    labels = rng.permutation(np.arange(n) % 10)
    shift = rng.integers(-4, 5, (n, 2))
    pixels = np.zeros((n, 28, 28))
    for i in range(n):
        pixels[i] = np.roll(patterns[labels[i]], tuple(shift[i]), axis=(0, 1))
    pixels = np.clip(0.55 * pixels + rng.normal(0, 0.6, (n, 28, 28)), 0, 1)
    x = torch.tensor((pixels - 0.5) / 0.5, dtype=torch.float32).unsqueeze(1)
    return x, torch.tensor(labels, dtype=torch.long)


def _torchvision(name, split):
    import torchvision

    cls = {"mnist": torchvision.datasets.MNIST, "fashion-mnist": torchvision.datasets.FashionMNIST,
           "cifar10": torchvision.datasets.CIFAR10}[name]
    ds = cls(str(data_dir()), train=split == "train", download=True)
    x = torch.as_tensor(np.asarray(ds.data)).float() / 255.0
    y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    else:
        x = x.permute(0, 3, 1, 2)
    return x, y


def load_split(name, split):
    key = (name, split)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if name == "synthetic":
            _CACHE[key] = _synthetic(split)
            return _CACHE[key]
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name!r}")
        cache = data_dir() / "cache" / f"{name}_{split}.pt"
        if cache.exists():
            _CACHE[key] = torch.load(cache)
            return _CACHE[key]
        x, y = _torchvision(name, split)
        xt = _torchvision(name, "train")[0] if split != "train" else x
        mean = xt.mean(dim=(0, 2, 3), keepdim=True)
        std = xt.std(dim=(0, 2, 3), keepdim=True)
        x = (x - mean) / std
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save((x, y), cache)
        _CACHE[key] = (x, y)
        return _CACHE[key]


class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, img_size=28, width=1.0, bn=False):
        super().__init__()
        c1, c2, h = max(4, int(32 * width)), max(4, int(64 * width)), max(16, int(512 * width))
        norm = (lambda c: nn.BatchNorm2d(c)) if bn else (lambda c: nn.Identity())
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 5, padding=2), norm(c1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 5, padding=2), norm(c2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(c2 * (img_size // 4) ** 2, h), nn.ReLU(), nn.Linear(h, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet8(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, width=1.0):
        super().__init__()
        w = [max(4, int(c * width)) for c in (16, 32, 64)]
        self.conv1 = nn.Conv2d(in_channels, w[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.layer1 = BasicBlock(w[0], w[0])
        self.layer2 = BasicBlock(w[0], w[1], 2)
        self.layer3 = BasicBlock(w[1], w[2], 2)
        self.fc = nn.Linear(w[2], num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def build_model(dataset="synthetic", arch="cnn", width=1.0):
    spec = DATASETS[dataset]
    if arch == "cnn":
        return Net(spec["channels"], spec["classes"], spec["size"], width)
    if arch == "cnn_bn":
        return Net(spec["channels"], spec["classes"], spec["size"], width, bn=True)
    if arch == "resnet8":
        return ResNet8(spec["channels"], spec["classes"], width)
    raise ValueError(f"unknown model {arch!r}")


def partition(labels, num_clients, scheme="iid", alpha=0.5, sigma=1.0, seed=0, min_size=10):
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.default_rng(seed)
    if scheme == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(n), num_clients)]
    if scheme == "quantity":
        props = rng.lognormal(0, sigma, num_clients)
        props = props / props.sum()
        counts = np.maximum(min_size, (props * n).astype(int))
        counts = (counts * n / counts.sum()).astype(int)
        cuts = np.cumsum(counts)[:-1]
        return [np.sort(p) for p in np.split(rng.permutation(n), cuts)]
    if scheme == "dirichlet":
        classes = np.unique(labels)
        for _ in range(100):
            parts = [[] for _ in range(num_clients)]
            for c in classes:
                idx = rng.permutation(np.nonzero(labels == c)[0])
                p = rng.dirichlet(np.full(num_clients, alpha))
                cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx, cuts)):
                    parts[k].extend(chunk.tolist())
            if min(len(p) for p in parts) >= min_size:
                break
        return [np.sort(np.asarray(p, dtype=int)) for p in parts]
    raise ValueError(f"unknown partition scheme {scheme!r}")


_PARTS = {}


def client_indices(cfg, partition_id, num_partitions):
    key = (cfg["dataset"], cfg["partitioner"], cfg["dirichlet-alpha"], cfg["quantity-sigma"], cfg["seed"],
           num_partitions, cfg.get("max-samples", 0))
    with _LOCK:
        if key not in _PARTS:
            _, y = load_split(cfg["dataset"], "train")
            labels = y.numpy()
            pool = np.arange(len(labels))
            if cfg.get("max-samples", 0):
                pool = np.sort(np.random.default_rng(cfg["seed"]).permutation(pool)[:cfg["max-samples"]])
            parts = partition(labels[pool], num_partitions, cfg["partitioner"], cfg["dirichlet-alpha"],
                              cfg["quantity-sigma"], cfg["seed"])
            out = []
            for k, p in enumerate(parts):
                p = pool[p]
                p = np.random.default_rng([cfg["seed"], k]).permutation(p)
                cut = max(1, int(round(0.8 * len(p))))
                out.append((np.sort(p[:cut]), np.sort(p[cut:])))
            _PARTS[key] = out
        return _PARTS[key][partition_id]


def label_counts(cfg, num_partitions, num_classes=10):
    _, y = load_split(cfg["dataset"], "train")
    out = np.zeros((num_partitions, num_classes), dtype=int)
    for k in range(num_partitions):
        tr, _ = client_indices(cfg, k, num_partitions)
        out[k] = np.bincount(y[tr].numpy(), minlength=num_classes)
    return out


def batches(x, y, idx=None, batch_size=64, shuffle=False, seed=0):
    idx = np.arange(len(y)) if idx is None else np.asarray(idx)
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s:s + batch_size], dtype=torch.long)
        yield x[b], y[b]


def train(model, x, y, idx, epochs=1.0, batch_size=32, lr=0.05, momentum=0.9, proximal_mu=0.0, seed=0,
          extra_loss=None, weight_decay=0.0, param_lr=None):
    params = list(model.parameters())
    groups = param_lr(model, lr) if param_lr is not None else [{"params": params, "lr": lr}]
    opt = torch.optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    anchor = [p.detach().clone() for p in params] if proximal_mu > 0 else None
    steps_total = max(1, int(round(float(epochs) * len(idx) / batch_size)))
    model.train()
    step, total, epoch = 0, 0.0, 0
    while step < steps_total:
        for xb, yb in batches(x, y, idx, batch_size, shuffle=True, seed=seed * 1009 + epoch):
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            task_loss = loss.item()
            if anchor is not None:
                loss = loss + 0.5 * proximal_mu * sum(((p - a) ** 2).sum() for p, a in zip(params, anchor))
            if extra_loss is not None:
                loss = loss + extra_loss(model, xb, yb, out)
            loss.backward()
            opt.step()
            total += task_loss
            step += 1
            if step >= steps_total:
                break
        epoch += 1
    return total / max(step, 1), step


@torch.no_grad()
def test(model, x, y, idx=None, batch_size=512):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    for xb, yb in batches(x, y, idx, batch_size):
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
        n += len(yb)
    return loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict(model, x, y, idx=None, batch_size=512):
    model.eval()
    labels, preds = [], []
    for xb, yb in batches(x, y, idx, batch_size):
        preds.append(model(xb).argmax(1))
        labels.append(yb)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))
