from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.metrics import classification_metrics

ALGORITHMS = ("fedavg", "fedprox", "qfedavg")


def local_train(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 1, lr: float = 0.05,
                batch_size: int = 32, mu: float = 0.0, global_vec: Optional[torch.Tensor] = None,
                seed: int = 0, device: str = "cpu") -> Dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    anchors = None
    if mu > 0 and global_vec is not None:
        anchors, i = [], 0
        for p in model.parameters():
            n = p.numel()
            anchors.append(global_vec[i:i + n].view_as(p).to(device))
            i += n
    model.train()
    losses, steps = [], 0
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=g).to(device)
        for s in range(0, len(yt), batch_size):
            b = perm[s:s + batch_size]
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            if anchors is not None:
                prox = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchors))
                loss = loss + 0.5 * mu * prox
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            steps += 1
    return {"train_loss": float(np.mean(losses)) if losses else 0.0, "steps": steps}


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, n_classes: int,
             device: str = "cpu", return_pred: bool = False):
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    logits = model(Xt)
    loss = float(F.cross_entropy(logits, yt).item())
    pred = logits.argmax(1).cpu().numpy()
    m = classification_metrics(np.asarray(y), pred, n_classes)
    m["loss"] = loss
    return (m, pred) if return_pred else m


def detect_update_outliers(updates: List[torch.Tensor], z_thresh: float = 3.5,
                           reference: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
    if not updates:
        return {"norm": np.zeros(0), "cos": np.zeros(0), "z": np.zeros(0), "outlier": np.zeros(0, bool)}
    U = torch.stack(updates)
    norms = U.norm(dim=1).cpu().numpy()
    ref = reference if reference is not None and float(reference.norm()) > 0 else U.median(dim=0).values
    cos = F.cosine_similarity(U, ref.unsqueeze(0).expand_as(U), dim=1).cpu().numpy()

    def rz(v):
        m = np.median(v)
        mad = np.median(np.abs(v - m)) + 1e-12
        return 0.6745 * (v - m) / mad

    z_norm = rz(np.log(norms + 1e-12))
    z_cos = -rz(cos)
    z = np.sqrt(np.maximum(z_norm, 0) ** 2 + np.maximum(z_cos, 0) ** 2)
    out = z > z_thresh if len(updates) >= 4 else np.zeros(len(updates), bool)
    return {"norm": norms, "cos": cos, "z": z, "outlier": out}


def apply_attack(update: torch.Tensor, kind: str, rng: np.random.RandomState) -> torch.Tensor:
    if kind == "signflip":
        return -3.0 * update
    if kind == "noise":
        return update + torch.randn_like(update) * (update.norm() / np.sqrt(update.numel()) * 10.0)
    if kind == "scale":
        return 10.0 * update
    return update


__all__ = ["ALGORITHMS", "local_train", "evaluate", "detect_update_outliers", "apply_attack"]
