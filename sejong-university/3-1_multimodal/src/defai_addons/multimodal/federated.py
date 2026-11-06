import copy
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.client import NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedProx

from ..common.metrics import entropy
from ..fl import flower as FL
from .data import MODALITIES, N_CLASSES, drop_modalities
from .encoders import SinusoidalPE
from .train import evaluate, loader, train_epoch


class TransformerCompressor(nn.Module):
    def __init__(self, P: int = 32, r: int = 4, d: int = 64, window: int = 16, n_heads: int = 4):
        super().__init__()
        self.P, self.r, self.W = P, r, window
        self.inp = nn.Linear(P, d)
        self.pe = SinusoidalPE(d, window)
        self.enc = nn.TransformerEncoderLayer(d, n_heads, 2 * d, 0.0, batch_first=True, norm_first=True)
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        self.dec = nn.TransformerEncoderLayer(d, n_heads, 2 * d, 0.0, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, P)
        self.norm = nn.LayerNorm(d)

    def _windows(self, vec: torch.Tensor):
        n = vec.numel()
        per = self.P * self.W
        pad = (-n) % per
        v = torch.cat([vec, vec.new_zeros(pad)]).view(-1, self.W, self.P)
        scale = v.pow(2).mean((1, 2), keepdim=True).sqrt() + 1e-12
        return v / scale, scale, n

    def encode_windows(self, x):
        h = self.norm(self.enc(self.pe(self.inp(x))))
        return self.down(h)

    def decode_windows(self, z):
        h = self.dec(self.pe(self.up(z)))
        return self.out(h)

    @torch.no_grad()
    def compress(self, vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, scale, n = self._windows(vec)
        return {"code": self.encode_windows(x).half(), "scale": scale.half(), "n": n}

    @torch.no_grad()
    def decompress(self, msg: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.decode_windows(msg["code"].float()) * msg["scale"].float()
        return x.reshape(-1)[: msg["n"]]

    @staticmethod
    def message_bytes(msg) -> int:
        return msg["code"].numel() * 2 + msg["scale"].numel() * 2

    def fit(self, samples: List[torch.Tensor], steps: int = 300, lr: float = 2e-3, batch: int = 64, seed: int = 0) -> float:
        g = torch.Generator().manual_seed(seed)
        X = torch.cat([self._windows(v)[0] for v in samples])
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        loss = torch.zeros(())
        for _ in range(steps):
            idx = torch.randint(0, len(X), (min(batch, len(X)),), generator=g)
            x = X[idx]
            loss = F.mse_loss(self.decode_windows(self.encode_windows(x)), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.eval()
        return float(loss)


def label_entropy(y: np.ndarray) -> float:
    return entropy(np.bincount(y, minlength=N_CLASSES))


def fedatc_weights(n: List[float], H: List[float], beta: float = 0.5) -> np.ndarray:
    n, H = np.asarray(n, float), np.asarray(H, float)
    w = beta * n / n.sum() + (1 - beta) * (H + 1e-6) / (H + 1e-6).sum()
    return w / w.sum()


def topk_compress(v: np.ndarray, ratio: float):
    k = max(1, int(v.size * ratio))
    idx = np.argpartition(np.abs(v), -k)[-k:]
    out = np.zeros_like(v)
    out[idx] = v[idx].astype(np.float16).astype(v.dtype)
    return out, k * (2 + 4)


@torch.no_grad()
def embed_analysis(model: nn.Module, d: Dict[str, np.ndarray], device: str) -> torch.Tensor:
    model.eval()
    out = []
    for b in loader(d, 256, shuffle=False):
        b = {k: v.to(device) for k, v in b.items()}
        e = model(b)["emb"]
        m = b["mask"].float()
        out.append(torch.cat([e[k] * m[:, j:j + 1] for j, k in enumerate(MODALITIES)], 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def local_loss(model: nn.Module, d: Dict[str, np.ndarray], device: str) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader(d, 256, shuffle=False):
        b = {k: v.to(device) for k, v in b.items()}
        tot += float(F.cross_entropy(model(b)["logits"], b["y"], reduction="sum"))
        n += len(b["y"])
    return tot / max(n, 1)


class MultimodalClient(NumPyClient):
    def __init__(self, cid: int, data: Dict[str, np.ndarray], model: nn.Module, device: str, cfg: Dict):
        self.cid, self.data, self.model, self.device, self.cfg = cid, data, model, device, cfg
        self.keys = FL.state_keys(model)
        self.H = label_entropy(data["y"])

    def get_parameters(self, config):
        return FL.get_ndarrays(self.model, self.keys)

    def fit(self, parameters, config):
        c, rnd = self.cfg, int(config.get("server_round", 0))
        FL.set_ndarrays(self.model, self.keys, parameters)
        metrics = {"n": len(self.data["y"]), "entropy": self.H}
        if config.get("need_loss"):
            metrics["loss_before"] = local_loss(self.model, self.data, self.device)
        if config.get("analysis"):
            rng = np.random.RandomState(c["seed"] + rnd * 1000 + self.cid)
            idx = rng.choice(len(self.data["y"]), min(c["analysis_samples"], len(self.data["y"])), replace=False)
            E = embed_analysis(self.model, {k: v[idx] for k, v in self.data.items()}, self.device)
            metrics.update({"emb": FL.array_to_bytes(E.numpy()), "emb_y": FL.array_to_bytes(self.data["y"][idx])})
        mu = float(config.get("proximal_mu", 0.0))
        prox = {k: p.detach().clone() for k, p in self.model.named_parameters()} if mu > 0 else None
        opt = torch.optim.AdamW(self.model.parameters(), lr=c["lr"], weight_decay=1e-4)
        for e in range(c["local_epochs"]):
            train_epoch(self.model, loader(self.data, c["batch_size"], seed=c["seed"] + rnd * 100 + self.cid * 10 + e),
                        opt, self.device, c["modality_dropout"], prox, mu)
        return FL.get_ndarrays(self.model, self.keys), len(self.data["y"]), metrics


def _fedatc_weight_fn(beta: float):
    return lambda results: fedatc_weights([r.metrics["n"] for _, r in results],
                                          [r.metrics["entropy"] for _, r in results], beta)


class FedATC:
    def __init__(self, model_fn: Callable[[], nn.Module], clients: List[Dict], device: str = "cpu",
                 local_epochs: int = 1, lr: float = 1e-3, batch_size: int = 64, beta: float = 0.5,
                 hierarchical: bool = True, algorithm: str = "fedavg", q: float = 0.2, qffl_lr: float = 10.0,
                 update_compression: Optional[str] = None, topk_ratio: float = 0.1,
                 analysis_compression: bool = False, comp_r: int = 12, analysis_samples: int = 64,
                 refit_every: int = 5, modality_dropout: float = 0.2, mu: float = 0.0,
                 seed: int = 0, log: Optional[Callable[[str], None]] = print):
        torch.manual_seed(seed)
        self.model = model_fn().to(device)
        self.keys = FL.state_keys(self.model)
        self.global_nd = FL.get_ndarrays(self.model, self.keys)
        pnames = set(k for k, _ in self.model.named_parameters())
        self.is_param = np.concatenate([np.full(a.size, k in pnames) for k, a in zip(self.keys, self.global_nd)])
        self.clients, self.device, self.log = clients, device, log
        self.hierarchical, self.update_compression, self.topk_ratio = hierarchical, update_compression, topk_ratio
        self.refit_every, self.seed = refit_every, seed
        self.ccfg = dict(lr=lr, local_epochs=local_epochs, batch_size=batch_size, modality_dropout=modality_dropout,
                         seed=seed, analysis_samples=analysis_samples)
        self.groups = sorted(set(c["group"] for c in clients)) if hierarchical else [0]
        self.members = {g: [i for i, c in enumerate(clients) if (c["group"] if hierarchical else 0) == g] for g in self.groups}

        self.proxies = [FL.as_proxy(i, MultimodalClient(i, c["data"], copy.deepcopy(self.model), device, self.ccfg))
                        for i, c in enumerate(clients)]
        self.analysis = analysis_compression
        self.algorithm = algorithm
        fit_cfg = lambda rnd: {"server_round": rnd, "analysis": bool(self.analysis), "need_loss": algorithm == "qfedavg"}
        init = ndarrays_to_parameters(self.global_nd)
        common = dict(FL._all(0), initial_parameters=init, on_fit_config_fn=fit_cfg)
        if hierarchical:
            self.group_strategies = {g: FL.WeightedFedAvg(_fedatc_weight_fn(beta), **common) for g in self.groups}
            self.server_strategy = FL.WeightedFedAvg(_fedatc_weight_fn(beta), **FL._all(0), initial_parameters=init)
        else:
            self.group_strategies = {0: FL.make_strategy(algorithm, self.global_nd, mu=mu, q=q, lr=qffl_lr,
                                                         on_fit_config=fit_cfg)}
            self.server_strategy = None
        self.group_proxies = {g: FL.LocalClientProxy("group-%d" % g, None) for g in self.groups}

        self.residual = {g: np.zeros(int(self.is_param.sum())) for g in self.groups}
        self.group_nd = {g: self.global_nd for g in self.groups}
        self.compressor: Optional[TransformerCompressor] = None
        self.comp_r = comp_r
        self.server_results: Dict[int, Dict[str, torch.Tensor]] = {}
        self.history: List[Dict] = []
        self.bytes = {"client_group": 0, "group_server_raw": 0, "server_in": 0, "analysis_raw": 0, "analysis_sent": 0}

    def round(self, rnd: int) -> Dict:
        t0 = time.time()
        server_round = rnd + 1
        P = self.is_param
        raw = FL.nbytes(self.global_nd)
        gparams = ndarrays_to_parameters(self.global_nd)
        gflat = FL.flatten(self.global_nd)
        info: Dict = {"round": server_round}
        server_results, group_analysis = [], {}
        for g in self.groups:
            mem = self.members[g]
            results, failures = FL.run_fit(self.group_strategies[g], server_round, gparams, [self.proxies[i] for i in mem])
            self.bytes["client_group"] += len(results) * raw * 2
            agg, _ = self.group_strategies[g].aggregate_fit(server_round, results, failures)
            g_nd = parameters_to_ndarrays(agg)
            self.group_nd[g] = g_nd
            if self.analysis:
                E = np.concatenate([FL.bytes_to_array(r.metrics["emb"]) for _, r in results])
                y = np.concatenate([FL.bytes_to_array(r.metrics["emb_y"]) for _, r in results])
                group_analysis[g] = (torch.as_tensor(E), torch.as_tensor(y))
                self.bytes["analysis_raw"] += E.size * 4 + y.size
            if not self.hierarchical:
                self.bytes["server_in"] += len(results) * raw
                self.global_nd = g_nd
                continue
            self.bytes["group_server_raw"] += raw
            u = FL.flatten(g_nd) - gflat
            if self.update_compression == "topk":
                v = u[P] + self.residual[g]
                dec, nb = topk_compress(v, self.topk_ratio)
                self.residual[g] = v - dec
                u = u.copy()
                u[P] = dec
                sent = nb + int((~P).sum()) * 4
            else:
                sent = raw
            self.bytes["server_in"] += sent
            ys = np.concatenate([self.clients[i]["data"]["y"] for i in mem])
            server_results.append((self.group_proxies[g], FL.fit_res(
                FL.unflatten(gflat + u, self.global_nd), len(ys), {"n": len(ys), "entropy": label_entropy(ys)})))
        if self.hierarchical:
            agg, _ = self.server_strategy.aggregate_fit(server_round, server_results, [])
            self.global_nd = parameters_to_ndarrays(agg)
            info["group_weight"] = {int(cid.split("-")[1]): w for cid, w in self.server_strategy.last_weights.items()}
        else:
            info["group_weight"] = {0: 1.0}
        if self.analysis:
            info.update(self._analysis_round(rnd, group_analysis))
        info["time_s"] = time.time() - t0
        return info

    def _analysis_round(self, rnd: int, group_results: Dict[int, tuple]) -> Dict:
        info, errs = {}, []
        refit = self.compressor is None or rnd % self.refit_every == 0
        if refit:
            if self.compressor is None:
                d_an = next(iter(group_results.values()))[0].shape[1]
                self.compressor = TransformerCompressor(P=d_an, r=self.comp_r, window=16)
            info["compressor_mse"] = self.compressor.fit([E.reshape(-1) for E, _ in group_results.values()],
                                                         steps=400, seed=self.seed + rnd)
        for g, (E, y) in group_results.items():
            if refit:
                self.server_results[g] = {"emb": E, "y": y}
                self.bytes["analysis_sent"] += E.numel() * 4 + y.numel()
            else:
                msg = self.compressor.compress(E.reshape(-1))
                R = self.compressor.decompress(msg).view_as(E)
                errs.append(float((R - E).norm() / (E.norm() + 1e-12)))
                self.server_results[g] = {"emb": R, "y": y, "raw": E}
                self.bytes["analysis_sent"] += TransformerCompressor.message_bytes(msg) + y.numel()
        if errs:
            info["analysis_rel_err"] = float(np.mean(errs))
        return info

    def _load(self, nd):
        FL.set_ndarrays(self.model, self.keys, nd)

    @torch.no_grad()
    def prototype_accuracy(self, test: Dict[str, np.ndarray], use_raw: bool = False) -> Optional[float]:
        if not self.server_results:
            return None
        E = torch.cat([r["raw"] if use_raw and "raw" in r else r["emb"] for r in self.server_results.values()])
        y = torch.cat([r["y"] for r in self.server_results.values()])
        k = E.shape[1] // 3
        self._load(self.global_nd)
        T = embed_analysis(self.model, test, self.device)
        score = torch.zeros(len(T), N_CLASSES)
        for j in range(3):
            Ej, Tj = E[:, j * k:(j + 1) * k], T[:, j * k:(j + 1) * k]
            present = Ej.norm(dim=1) > 1e-6
            protos = torch.stack([F.normalize(Ej[present & (y == c)].mean(0), dim=0) if bool((present & (y == c)).any())
                                  else torch.zeros(k) for c in range(N_CLASSES)])
            score += Tj @ protos.t()
        return float((score.argmax(1).numpy() == test["y"]).mean())

    def evaluate_all(self, test: Dict[str, np.ndarray], group_tests: Optional[Dict[int, Dict]] = None) -> Dict:
        self._load(self.global_nd)
        res = {"global_full": evaluate(self.model, test, self.device),
               "global_missing30": evaluate(self.model, drop_modalities(test, (0.3, 0.3, 0.3), seed=7), self.device)}
        if group_tests:
            res["group"] = {}
            for g, d in group_tests.items():
                self._load(self.global_nd)
                a_glob = evaluate(self.model, d, self.device)["accuracy"]
                self._load(self.group_nd.get(g, self.global_nd))
                a_loc = evaluate(self.model, d, self.device)["accuracy"]
                res["group"][int(g)] = {"global_model": a_glob, "group_model": a_loc}
            self._load(self.global_nd)
        if self.analysis:
            res["analysis"] = {"proto_acc_compressed": self.prototype_accuracy(test),
                               "proto_acc_raw": self.prototype_accuracy(test, use_raw=True)}
        return res

    def run(self, rounds: int, test: Dict[str, np.ndarray], group_tests: Optional[Dict[int, Dict]] = None,
            eval_every: int = 1) -> Dict:
        for r in range(rounds):
            info = self.round(r)
            if (r + 1) % eval_every == 0 or r + 1 == rounds:
                self._load(self.global_nd)
                ev = evaluate(self.model, test, self.device)
                info.update({"acc": ev["accuracy"], "f1": ev["f1"]})
                if self.log:
                    self.log("  round %2d  acc %.4f  f1 %.4f  group_w %s%s%s" % (
                        r + 1, ev["accuracy"], ev["f1"], {k: round(v, 3) for k, v in info["group_weight"].items()},
                        "  comp_mse %.4f" % info["compressor_mse"] if "compressor_mse" in info else "",
                        "  analysis_rel_err %.3f" % info["analysis_rel_err"] if "analysis_rel_err" in info else ""))
            self.history.append(info)
        final = self.evaluate_all(test, group_tests)
        b = self.bytes
        final["bytes"] = dict(b)
        final["update_compression_ratio"] = (b["group_server_raw"] / max(b["server_in"], 1)) if self.hierarchical else None
        final["analysis_compression_ratio"] = (b["analysis_raw"] / max(b["analysis_sent"], 1)) if self.analysis else None
        final["history"] = self.history
        final["framework"] = "flwr"
        return final
