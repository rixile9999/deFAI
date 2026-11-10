from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("image", "text", "audio")
N_CLASSES = 6
VOCAB = 64
SEQ_LEN = 16
AUDIO_LEN = 800
PAD = 0


def _draw_shape(a: int, rng: np.random.RandomState, noise: float) -> np.ndarray:
    img = np.zeros((28, 28), np.float32)
    s = rng.randint(6, 10)
    cx, cy = rng.randint(s + 1, 27 - s, size=2)
    yy, xx = np.mgrid[:28, :28]
    if a == 0:
        img[(abs(xx - cx) <= s) & (abs(yy - cy) <= s)] = 1.0
        img[(abs(xx - cx) <= s - 2) & (abs(yy - cy) <= s - 2)] = 0.0
    elif a == 1:
        img[(abs(xx - cx) <= 1) & (abs(yy - cy) <= s)] = 1.0
        img[(abs(yy - cy) <= 1) & (abs(xx - cx) <= s)] = 1.0
    else:
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img[(r <= s) & (r >= s - 2)] = 1.0
    img += rng.randn(28, 28).astype(np.float32) * noise
    return img


def _audio(b: int, rng: np.random.RandomState, noise: float) -> np.ndarray:
    t = np.arange(AUDIO_LEN) / 800.0
    f0 = rng.uniform(30, 60) if b == 0 else rng.uniform(110, 160)
    am = rng.uniform(2, 5) if b == 0 else rng.uniform(8, 12)
    x = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)) * (0.6 + 0.4 * np.sin(2 * np.pi * am * t))
    x += 0.3 * np.sin(2 * np.pi * rng.uniform(5, 200) * t)
    return (x + rng.randn(AUDIO_LEN) * noise).astype(np.float32)


def _text(c: int, rng: np.random.RandomState, kw_rate: float, mislead: float) -> np.ndarray:
    seq = rng.randint(12, VOCAB, SEQ_LEN)
    n_kw = rng.binomial(SEQ_LEN, kw_rate)
    pos = rng.choice(SEQ_LEN, n_kw, replace=False)
    for p in pos:
        cc = c if rng.rand() > mislead else rng.randint(N_CLASSES)
        seq[p] = 1 + 2 * cc + rng.randint(2)
    length = rng.randint(10, SEQ_LEN + 1)
    seq[length:] = PAD
    return seq.astype(np.int64)


def generate(n: int, seed: int = 0, img_noise: float = 1.2, audio_noise: float = 2.0,
             kw_rate: float = 0.08, mislead: float = 0.55, labels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    y = labels if labels is not None else rng.randint(0, N_CLASSES, n)
    img = np.stack([_draw_shape(c % 3, rng, img_noise) for c in y])[:, None]
    aud = np.stack([_audio(c // 3, rng, audio_noise) for c in y])[:, None]
    txt = np.stack([_text(c, rng, kw_rate, mislead) for c in y])
    return {"image": img, "text": txt, "audio": aud, "y": y.astype(np.int64),
            "mask": np.ones((len(y), 3), dtype=bool)}


def drop_modalities(d: Dict[str, np.ndarray], p_missing: Sequence[float], seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(d["y"])
    m = rng.rand(n, 3) >= np.asarray(p_missing)[None, :]
    none = ~m.any(1)
    m[none, rng.randint(0, 3, none.sum())] = True
    out = dict(d)
    out["mask"] = d["mask"] & m
    return out


def subset(d: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in d.items()}


class MMDataset(Dataset):
    def __init__(self, d: Dict[str, np.ndarray]):
        self.d = d

    def __len__(self):
        return len(self.d["y"])

    def __getitem__(self, i):
        return {k: torch.as_tensor(v[i]) for k, v in self.d.items()}


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def make_clients(n_clients: int, n_per_client: int = 400, alpha: float = 0.5, seed: int = 0,
                 patterns: Optional[List[Sequence[float]]] = None) -> List[Dict]:
    rng = np.random.RandomState(seed)
    patterns = patterns or [(0.0, 0.0, 0.9), (0.9, 0.0, 0.0), (0.1, 0.1, 0.1)]
    names = ["image+text", "text+audio", "all"]
    clients = []
    for k in range(n_clients):
        g = k % len(patterns)
        p = rng.dirichlet(np.full(N_CLASSES, alpha))
        y = rng.choice(N_CLASSES, n_per_client, p=p)
        d = generate(n_per_client, seed=seed * 1000 + k, labels=y,
                     img_noise=rng.uniform(0.9, 1.5), audio_noise=rng.uniform(1.5, 2.5),
                     mislead=rng.uniform(0.45, 0.6))
        d = drop_modalities(d, patterns[g], seed=seed * 1000 + k)
        clients.append({"data": d, "pattern": names[g] if g < len(names) else "p%d" % g, "group": g})
    return clients
