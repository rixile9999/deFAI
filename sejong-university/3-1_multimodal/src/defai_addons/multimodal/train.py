import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..common.metrics import classification_metrics
from .data import MODALITIES, N_CLASSES, MMDataset, drop_modalities, to_device
from .joint import cross_modal_retrieval


def loader(d: Dict[str, np.ndarray], batch_size: int = 128, shuffle: bool = True, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(MMDataset(d), batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def train_epoch(model: nn.Module, dl: DataLoader, opt: torch.optim.Optimizer, device: str,
                modality_dropout: float = 0.0, prox: Optional[Dict[str, torch.Tensor]] = None, mu: float = 0.0,
                clip: float = 5.0) -> Dict[str, float]:
    model.train()
    tot, tot_ce, n = 0.0, 0.0, 0
    for batch in dl:
        batch = to_device(batch, device)
        if modality_dropout > 0:
            drop = torch.rand(batch["mask"].shape, device=device) < modality_dropout
            m = batch["mask"] & ~drop
            none = ~m.any(1)
            m[none] = batch["mask"][none]
            batch["mask"] = m
        out = model(batch)
        ce = F.cross_entropy(out["logits"], batch["y"])
        loss = ce + out["aux"]
        if prox is not None and mu > 0:
            loss = loss + 0.5 * mu * sum(((p - prox[k]) ** 2).sum() for k, p in model.named_parameters())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        b = len(batch["y"])
        tot += float(loss) * b
        tot_ce += float(ce) * b
        n += b
    return {"loss": tot / max(n, 1), "ce": tot_ce / max(n, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, d: Dict[str, np.ndarray], device: str, batch_size: int = 256,
             with_attn: bool = False) -> Dict[str, object]:
    model.eval()
    ys, preds, embs, attn = [], [], {m: [] for m in MODALITIES}, {}
    for batch in loader(d, batch_size, shuffle=False):
        batch = to_device(batch, device)
        out = model(batch)
        ys.append(batch["y"].cpu())
        preds.append(out["logits"].argmax(1).cpu())
        for m in MODALITIES:
            embs[m].append(out["emb"][m].cpu())
        if with_attn:
            for k, v in out["attn"].items():
                if isinstance(v, torch.Tensor):
                    attn.setdefault(k, []).append(v.cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    res = classification_metrics(y, p, N_CLASSES)
    E = {m: torch.cat(v) for m, v in embs.items()}
    yt = torch.as_tensor(y)
    full = torch.as_tensor(d["mask"]).all(1)
    if full.sum() > 10:
        res["retrieval_img2txt"] = cross_modal_retrieval(E["image"][full], E["text"][full], yt[full])
        res["retrieval_aud2txt"] = cross_modal_retrieval(E["audio"][full], E["text"][full], yt[full])
    if with_attn:
        res["attn_mean"] = {k: torch.cat(v).mean(0).tolist() for k, v in attn.items() if torch.cat(v).dim() == 2}
    return res


def robustness(model: nn.Module, test: Dict[str, np.ndarray], device: str) -> Dict[str, float]:
    out = {}
    for name, p in [("full", (0, 0, 0)), ("missing30", (0.3, 0.3, 0.3)), ("no_image", (1, 0, 0)),
                    ("no_text", (0, 1, 0)), ("no_audio", (0, 0, 1))]:
        out[name] = evaluate(model, drop_modalities(test, p, seed=7), device)["accuracy"]
    return out


def fit(model: nn.Module, train: Dict[str, np.ndarray], val: Dict[str, np.ndarray], device: str = "cpu",
        epochs: int = 15, lr: float = 2e-3, batch_size: int = 128, modality_dropout: float = 0.2,
        weight_decay: float = 1e-4, seed: int = 0, log: Optional[Callable[[str], None]] = print) -> Dict[str, object]:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(train["y"]) / batch_size)))
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        dl = loader(train, batch_size, seed=seed + ep)
        model.train()
        tr = _epoch_with_sched(model, dl, opt, sched, device, modality_dropout)
        ev = evaluate(model, val, device)
        hist.append({"epoch": ep + 1, **tr, "val_acc": ev["accuracy"], "val_f1": ev["f1"]})
        if log:
            log("  epoch %2d  loss %.4f  val_acc %.4f  val_f1 %.4f" % (ep + 1, tr["loss"], ev["accuracy"], ev["f1"]))
    return {"history": hist, "train_time_s": time.time() - t0}


def _epoch_with_sched(model, dl, opt, sched, device, modality_dropout):
    tot, n = 0.0, 0
    for batch in dl:
        batch = to_device(batch, device)
        if modality_dropout > 0:
            drop = torch.rand(batch["mask"].shape, device=device) < modality_dropout
            m = batch["mask"] & ~drop
            none = ~m.any(1)
            m[none] = batch["mask"][none]
            batch["mask"] = m
        out = model(batch)
        loss = F.cross_entropy(out["logits"], batch["y"]) + out["aux"]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        tot += float(loss) * len(batch["y"])
        n += len(batch["y"])
    return {"loss": tot / max(n, 1)}
