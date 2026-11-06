import copy
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from flwr.client import NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from ..common.metrics import classification_metrics
from ..common.preprocessing import PreprocessingPipeline
from ..fl import flower as FL
from ..fl.algorithms import evaluate, local_train
from ..fl.models import MLP, model_bytes
from .privacy import GaussianDP, SecureAggregator
from .transport import UpdateCipher

ProgressFn = Optional[Callable[[str, int, Dict[str, float]], None]]


class SiteData:
    def __init__(self, X: np.ndarray, y: np.ndarray, Xte: np.ndarray, yte: np.ndarray):
        self.X, self.y, self.Xte, self.yte = X, y, Xte, yte


def _comm_ms(nbytes: float, mbps: float, latency_ms: float = 10.0) -> float:
    return latency_ms + nbytes * 8 / (mbps * 1e6) * 1e3


class BaseEnv:
    kind = "base"

    def __init__(self, name: str, sites: List[SiteData], Xte: np.ndarray, yte: np.ndarray, n_classes: int,
                 cfg: Dict):
        self.name, self.sites, self.Xte, self.yte = name, sites, Xte, yte
        self.n_classes, self.cfg = n_classes, cfg
        self.cost = {"raw_bytes": 0, "stat_bytes": 0, "model_bytes": 0, "comm_ms": 0.0, "compute_s": 0.0}
        self.history: List[Dict[str, float]] = []
        self.privacy: Dict[str, object] = {}

    def _model(self, in_dim: int) -> torch.nn.Module:
        torch.manual_seed(self.cfg["seed"])
        return MLP(in_dim, self.n_classes, hidden=self.cfg["hidden"])

    def _log(self, epoch: int, m: Dict[str, float], progress: ProgressFn):
        rec = {"epoch": epoch, **m, "raw_mb": self.cost["raw_bytes"] / 2 ** 20,
               "comm_mb": (self.cost["model_bytes"] + self.cost["stat_bytes"]) / 2 ** 20}
        self.history.append(rec)
        if progress:
            progress(self.name, epoch, rec)

    def fit(self, progress: ProgressFn = None):
        raise NotImplementedError

    def predict_global(self) -> np.ndarray:
        raise NotImplementedError

    def predict_site(self, s: int) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self) -> Dict[str, object]:
        pg = self.predict_global()
        local = []
        for s, site in enumerate(self.sites):
            m = classification_metrics(site.yte, self.predict_site(s), self.n_classes)
            m["site"] = s
            local.append(m)
        g = classification_metrics(self.yte, pg, self.n_classes)
        return {"global": g, "local": local, "pred": pg}


class LocalDistributedEnv(BaseEnv):
    kind = "local_distributed"

    def __init__(self, *a, strategy: str = "independent", **kw):
        super().__init__(*a, **kw)
        assert strategy in ("independent", "sync", "central")
        self.strategy = strategy
        self.privacy = {"raw_data_leaves_site": strategy != "independent", "secure_agg": False, "dp": None}

    def fit(self, progress: ProgressFn = None):
        c = self.cfg
        t0 = time.time()
        if self.strategy == "independent":
            self.pipe = PreprocessingPipeline("local", use_pca=c["use_pca"])
            self.pipe.fit([s.X for s in self.sites])
            self.data = [self.pipe.transform(s.X, site=k, drop_rows=True, y=s.y) for k, s in enumerate(self.sites)]
            self.models = [self._model(self.pipe.out_dim(k)) for k in range(len(self.sites))]
            for ep in range(c["epochs"]):
                for k, (m, (X, y)) in enumerate(zip(self.models, self.data)):
                    local_train(m, X, y, 1, c["lr"], c["batch_size"], seed=c["seed"] * 100 + ep * 10 + k)
                self._log(ep + 1, self._quick_eval(), progress)
            share = len(self.yte) * self.n_classes * 4
            self.cost["model_bytes"] += len(self.sites) * share
            self.cost["comm_ms"] += _comm_ms(share, c["bandwidth_mbps"])
        else:
            self.pipe = PreprocessingPipeline("global", use_pca=c["use_pca"])
            info = self.pipe.fit([s.X for s in self.sites])
            X = np.concatenate([s.X for s in self.sites])
            y = np.concatenate([s.y for s in self.sites])
            origin = np.concatenate([np.full(len(s.y), k) for k, s in enumerate(self.sites)])
            X, y = self.pipe.transform(X, drop_rows=True, y=y)
            origin = origin[self.pipe.shared["cleaner"].row_mask(np.concatenate([s.X for s in self.sites]))]
            row_bytes = self.sites[0].X.shape[1] * 4
            rng = np.random.RandomState(c["seed"])
            if self.strategy == "central":
                moved = int((origin != np.bincount(origin).argmax()).sum())
                shards = [np.arange(len(y))]
            else:
                dest = rng.randint(0, len(self.sites), len(y))
                moved = int((dest != origin).sum())
                shards = [np.where(dest == k)[0] for k in range(len(self.sites))]
            self.cost["raw_bytes"] += moved * row_bytes
            self.cost["comm_ms"] += _comm_ms(moved * row_bytes, c["bandwidth_mbps"])
            self.model = self._model(X.shape[1])
            mb = model_bytes(self.model)
            if self.strategy == "central":
                for ep in range(c["epochs"]):
                    local_train(self.model, X, y, 1, c["lr"], c["batch_size"], seed=c["seed"] * 100 + ep)
                    self._log(ep + 1, self._quick_eval(), progress)
            else:
                opt = torch.optim.SGD(self.model.parameters(), lr=c["lr"])
                Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
                bs = max(1, c["batch_size"] // len(shards))
                n_steps = int(np.ceil(max(len(s) for s in shards) / bs))
                for ep in range(c["epochs"]):
                    perms = [rng.permutation(s) for s in shards]
                    for st in range(n_steps):
                        opt.zero_grad()
                        grads = None
                        for p in perms:
                            b = p[st * bs:(st + 1) * bs]
                            if len(b) == 0:
                                continue
                            self.model.zero_grad()
                            F.cross_entropy(self.model(Xt[b]), yt[b]).backward()
                            g = [q.grad.clone() for q in self.model.parameters()]
                            grads = g if grads is None else [a + b_ for a, b_ in zip(grads, g)]
                        for q, g in zip(self.model.parameters(), grads):
                            q.grad = g / len(perms)
                        opt.step()
                        n = len(shards)
                        step_bytes = int(2 * (n - 1) / n * mb) * n
                        self.cost["model_bytes"] += step_bytes
                        self.cost["comm_ms"] += _comm_ms(step_bytes / n, c["bandwidth_mbps"], 2.0)
                    self._log(ep + 1, self._quick_eval(), progress)
        self.cost["compute_s"] = time.time() - t0

    def _quick_eval(self) -> Dict[str, float]:
        return classification_metrics(self.yte, self.predict_global(), self.n_classes)

    @torch.no_grad()
    def _proba(self, model, X) -> np.ndarray:
        model.eval()
        return torch.softmax(model(torch.as_tensor(X)), 1).numpy()

    def predict_global(self) -> np.ndarray:
        if self.strategy == "independent":
            w = np.asarray([len(d[1]) for d in self.data], dtype=np.float64)
            P = sum(wk * self._proba(m, self.pipe.transform(self.Xte, site=k))
                    for k, (m, wk) in enumerate(zip(self.models, w / w.sum())))
            return P.argmax(1)
        return self._proba(self.model, self.pipe.transform(self.Xte)).argmax(1)

    def predict_site(self, s: int) -> np.ndarray:
        Xs = self.sites[s].Xte
        if self.strategy == "independent":
            return self._proba(self.models[s], self.pipe.transform(Xs, site=s)).argmax(1)
        return self._proba(self.model, self.pipe.transform(Xs)).argmax(1)


class SiteClient(NumPyClient):
    def __init__(self, k: int, X: np.ndarray, y: np.ndarray, model: torch.nn.Module, sa: Optional[SecureAggregator],
                 dp: Optional[GaussianDP], cfg: Dict, cipher=None):
        self.k, self.X, self.y, self.model, self.sa, self.dp, self.cfg = k, X, y, model, sa, dp, cfg
        self.keys = FL.state_keys(model)
        self.cipher = cipher
        self.last_plain = None

    def get_parameters(self, config):
        return FL.get_ndarrays(self.model, self.keys)

    def fit(self, parameters, config):
        c = self.cfg
        FL.set_ndarrays(self.model, self.keys, parameters)
        g = torch.as_tensor(FL.flatten(parameters), dtype=torch.float32)
        local_train(self.model, self.X, self.y, c["local_epochs"], c["lr"], c["batch_size"],
                    mu=float(config.get("proximal_mu", 0.0)), global_vec=g, seed=int(config["seed"]) * 10 + self.k)
        upd = torch.as_tensor(FL.flatten(FL.get_ndarrays(self.model, self.keys)), dtype=torch.float32) - g
        if self.dp is not None:
            upd = self.dp.clip_update(upd)
        contrib = upd * float(len(self.y) / float(config["total_examples"]))
        self.last_plain = contrib.double()
        out = self.sa.mask(self.k, contrib, list(range(self.sa.n))) if self.sa is not None else contrib.double()
        payload = out.numpy()
        if self.cipher is not None:
            payload = self.cipher.encrypt(payload, str(self.k), int(config["server_round"]))
        return [payload], len(self.y), {}


class SecureAggFedAvg(FedAvg):
    def __init__(self, initial: List[np.ndarray], dp: Optional[GaussianDP] = None, proximal_mu: float = 0.0, ciphers=None, max_weight=None, **kw):
        super().__init__(initial_parameters=ndarrays_to_parameters(initial), **kw)
        self.current, self.dp, self.proximal_mu = initial, dp, proximal_mu
        self.last_sum: Optional[np.ndarray] = None
        self.ciphers, self.max_weight = ciphers, max_weight

    def configure_fit(self, server_round, parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        if self.proximal_mu > 0:
            for _, fi in ins:
                fi.config["proximal_mu"] = self.proximal_mu
        return ins

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if failures:
            raise ValueError("Secure comparison requires all configured clients")
        results = sorted(results, key=lambda item: item[0].cid)
        if self.ciphers is not None:
            ids = [proxy.cid for proxy, _ in results]
            if len(ids) != len(set(ids)) or set(ids) != set(self.ciphers):
                raise ValueError("Encrypted aggregation requires each client exactly once")
        dim = sum(a.size for a in self.current)
        updates = []
        for proxy, result in results:
            arrays = parameters_to_ndarrays(result.parameters)
            if len(arrays) != 1:
                raise ValueError("Expected one update payload")
            vec = arrays[0]
            if self.ciphers is not None:
                vec = self.ciphers[proxy.cid].decrypt(vec, proxy.cid, server_round, dim)
            if vec.shape != (dim,) or not np.isfinite(vec).all():
                raise ValueError("Invalid update vector")
            updates.append(vec)
        total = sum(updates)
        self.last_sum = total
        if self.dp is not None:
            total = total + self.dp.noise(total.size, len(results), max_weight=self.max_weight).double().numpy()
        new = FL.unflatten(FL.flatten(self.current) + total, self.current)
        self.current = new
        return ndarrays_to_parameters(new), {}


class FederatedEnv(BaseEnv):
    kind = "federated"

    def __init__(self, *a, algorithm: str = "fedavg", dp_noise: float = 0.0, secure_agg: bool = True, **kw):
        super().__init__(*a, **kw)
        self.algorithm, self.secure_agg = algorithm, secure_agg
        self.dp = GaussianDP(dp_noise, clip=self.cfg["dp_clip"], seed=self.cfg["seed"]) if dp_noise > 0 else None
        self.privacy = {"raw_data_leaves_site": False, "secure_agg": secure_agg, "dp": None, "framework": "flwr",
                        "update_encryption": "AES-256-GCM" if self.cfg.get("encrypt_updates", True) else None,
                        "execution": "in_process_simulation", "secure_agg_scope": "pairwise_mask_simulation",
                        "encryption_scope": "client_update_payload_only",
                        "preprocessing_dp": False}
        self.cost["encryption_bytes"] = 0

    def fit(self, progress: ProgressFn = None):
        c = self.cfg
        t0 = time.time()
        self.pipe = PreprocessingPipeline("federated", use_pca=c["use_pca"])
        info = self.pipe.fit([s.X for s in self.sites])
        self.cost["stat_bytes"] += info["stat_bytes"]
        self.data = [self.pipe.transform(s.X, drop_rows=True, y=s.y) for s in self.sites]
        self.model = self._model(self.pipe.out_dim())
        keys = FL.state_keys(self.model)
        init = FL.get_ndarrays(self.model, keys)
        dim = int(sum(a.size for a in init))
        mb = FL.nbytes(init)
        n_sites = len(self.sites)
        N = sum(len(d[1]) for d in self.data)
        sa = SecureAggregator(n_sites, dim, seed=c["seed"]) if self.secure_agg else None
        ciphers = {str(k): UpdateCipher() for k in range(n_sites)} if c.get("encrypt_updates", True) else None
        max_weight = max(len(y) / N for _, y in self.data)
        clients = [SiteClient(k, X, y, copy.deepcopy(self.model), sa, self.dp, c,
                              ciphers[str(k)] if ciphers is not None else None)
                   for k, (X, y) in enumerate(self.data)]
        proxies = [FL.as_proxy(k, cl) for k, cl in enumerate(clients)]
        strategy = SecureAggFedAvg(init, dp=self.dp, proximal_mu=c["mu"] if self.algorithm == "fedprox" else 0.0,
                                   ciphers=ciphers, max_weight=max_weight,
                                   on_fit_config_fn=lambda rnd: {"seed": c["seed"] * 100 + rnd, "total_examples": N,
                                                                 "server_round": rnd},
                                   **FL._all(0))
        params = ndarrays_to_parameters(init)
        self.secure_agg_error = 0.0
        for ep in range(c["epochs"]):
            results, failures = FL.run_fit(strategy, ep + 1, params, proxies)
            if failures or len(results) != n_sites:
                raise RuntimeError("Comparison round requires all clients; no partial masked aggregation")
            params, _ = strategy.aggregate_fit(ep + 1, results, failures)
            if sa is not None:
                plain = sum(cl.last_plain for cl in clients).numpy()
                self.secure_agg_error = max(self.secure_agg_error, float(np.abs(strategy.last_sum - plain).max()))
            FL.set_ndarrays(self.model, keys, parameters_to_ndarrays(params))
            up_bytes = sum(FL.nbytes(parameters_to_ndarrays(r.parameters)) for _, r in results)
            overhead = 28 * n_sites if ciphers is not None else 0
            self.cost["encryption_bytes"] += overhead
            self.cost["model_bytes"] += mb * n_sites + up_bytes + (sa.overhead_bytes(n_sites) if sa else 0)
            self.cost["comm_ms"] += _comm_ms(mb, c["bandwidth_mbps"]) + _comm_ms(up_bytes / n_sites, c["bandwidth_mbps"])
            self._log(ep + 1, classification_metrics(self.yte, self.predict_global(), self.n_classes), progress)
        if self.dp is not None:
            self.privacy["dp"] = self.dp.describe()
        self.cost["compute_s"] = time.time() - t0

    @torch.no_grad()
    def _pred(self, X) -> np.ndarray:
        self.model.eval()
        return self.model(torch.as_tensor(self.pipe.transform(X))).argmax(1).numpy()

    def predict_global(self) -> np.ndarray:
        return self._pred(self.Xte)

    def predict_site(self, s: int) -> np.ndarray:
        return self._pred(self.sites[s].Xte)
