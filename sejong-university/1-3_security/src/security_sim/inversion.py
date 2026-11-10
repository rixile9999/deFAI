import base64
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request

from fl_core.task import test, train

OUTPUT_DEFENSES = ("none", "top1", "round", "noise", "limit")


def encode(x):
    return {"shape": list(x.shape), "data": base64.b64encode(x.numpy().astype(np.float32).tobytes()).decode()}


def decode(obj):
    return torch.from_numpy(np.frombuffer(base64.b64decode(obj["data"]), dtype=np.float32).reshape(obj["shape"]).copy())


def protect(probs, defense, cfg, rng):
    if defense == "top1":
        return {"label": probs.argmax(1).tolist()}
    if defense == "round":
        return {"probs": np.round(probs, cfg.get("decimals", 1)).tolist()}
    if defense == "noise":
        eps = cfg.get("epsilon", 10.0)
        noisy = probs + rng.laplace(0.0, 2.0 / eps, probs.shape)
        noisy = np.clip(noisy, 1e-6, None)
        return {"probs": (noisy / noisy.sum(1, keepdims=True)).tolist()}
    return {"probs": probs.tolist()}


def create_api(model, defense="none", cfg=None, seed=0):
    cfg = dict(cfg or {})
    app = Flask("prediction_api")
    state = {"queries": 0, "per_client": {}, "rng": np.random.default_rng(seed), "log": []}
    app.config["state"] = state
    model = copy.deepcopy(model).eval()

    @app.post("/api/v1/predict")
    def predict():
        body = request.get_json(force=True)
        x = decode(body["inputs"])
        client = body.get("client", "anonymous")
        state["queries"] += len(x)
        state["per_client"][client] = state["per_client"].get(client, 0) + len(x)
        state["log"].append({"client": client, "n": len(x), "ts": time.time()})
        trusted = client in cfg.get("trusted", ("evaluator",))
        if defense == "limit" and not trusted and state["per_client"][client] > cfg.get("budget", 1000):
            return jsonify(error="query budget exceeded"), 429
        with torch.no_grad():
            probs = F.softmax(model(x), 1).numpy()
        return jsonify(protect(probs, defense, cfg, state["rng"]))

    @app.get("/api/v1/audit")
    def audit():
        return jsonify(queries=state["queries"], requests=len(state["log"]))

    return app


class ApiClient:
    def __init__(self, app=None, url=None, name="attacker"):
        self.client = app.test_client() if app is not None else None
        self.url = url
        self.name = name

    def query(self, x):
        body = {"inputs": encode(x), "client": self.name}
        if self.client is not None:
            r = self.client.post("/api/v1/predict", json=body)
            status, data = r.status_code, r.get_json()
        else:
            import requests

            r = requests.post(self.url + "/api/v1/predict", json=body, timeout=60)
            status, data = r.status_code, r.json()
        if status != 200:
            return None
        if "label" in data:
            return F.one_hot(torch.tensor(data["label"]), 10).float()
        return torch.tensor(data["probs"], dtype=torch.float32)


def extract(api, surrogate, x_query, batch=100, epochs=30, lr=0.01, seed=0):
    soft, used = [], []
    for s in range(0, len(x_query), batch):
        out = api.query(x_query[s:s + batch])
        if out is None:
            break
        soft.append(out)
        used.append(x_query[s:s + batch])
    if not soft:
        return surrogate, 0
    xq, yq = torch.cat(used), torch.cat(soft)
    opt = torch.optim.Adam(surrogate.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    surrogate.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xq), generator=g)
        for s in range(0, len(xq), 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            loss = F.kl_div(F.log_softmax(surrogate(xq[idx]), 1), yq[idx].clamp_min(1e-8), reduction="batchmean")
            loss.backward()
            opt.step()
    return surrogate.eval(), len(xq)


def invert(model, target, shape, steps=200, lr=0.1, tv=0.02, l2=0.005):
    x = torch.zeros((1,) + tuple(shape[1:]), requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    model.eval()
    for _ in range(steps):
        opt.zero_grad()
        logp = F.log_softmax(model(x), 1)[0, target]
        tv_loss = (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        loss = -logp + tv * tv_loss + l2 * x.pow(2).mean()
        loss.backward()
        opt.step()
    return x.detach()


def similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return float(F.cosine_similarity(a - a.mean(), b - b.mean(), dim=0))


def run_inversion(backend, victim=None, defenses=OUTPUT_DEFENSES, queries=2000, train_epochs=3, seed=0,
                  defense_cfg=None, invert_steps=200, log=print):
    torch.manual_seed(seed)
    x_test, y_test = backend.test_set()
    n = len(y_test)
    perm = np.random.default_rng(seed).permutation(n)
    public, holdout = perm[: n // 2], perm[n // 2:]
    x_priv, y_priv = backend.client_data(0, 1)
    if victim is None:
        victim = backend.build_model()
        train(victim, x_priv, y_priv, np.arange(len(y_priv)), epochs=train_epochs, lr=0.05, seed=seed)
    _, victim_acc = test(victim, x_test, y_test, holdout)
    class_means = torch.stack([x_priv[y_priv == c].mean(0) for c in range(10)])
    rng = np.random.default_rng(seed + 1)
    q_idx = public[:queries // 2]
    noise = torch.randn((queries - len(q_idx),) + tuple(x_test.shape[1:])) * float(x_test.std())
    x_query = torch.cat([x_test[torch.as_tensor(q_idx)], noise])
    x_query = x_query[torch.as_tensor(rng.permutation(len(x_query)))]
    rows = []
    for d in defenses:
        cfg = dict({"budget": queries // 4, "decimals": 1, "epsilon": 10.0}, **(defense_cfg or {}))
        app = create_api(victim, d, cfg, seed)
        api = ApiClient(app)
        with torch.no_grad():
            out = ApiClient(app, name="evaluator").query(x_test[torch.as_tensor(holdout[:1000])])
        utility = float((out.argmax(1).numpy() == y_test[torch.as_tensor(holdout[:1000])].numpy()).mean()) \
            if out is not None else float("nan")
        torch.manual_seed(seed + 3)
        surrogate, used = extract(api, backend.build_model(), x_query, seed=seed)
        x_h, y_h = x_test[torch.as_tensor(holdout)], y_test[torch.as_tensor(holdout)]
        with torch.no_grad():
            agree = float((surrogate(x_h).argmax(1) == victim(x_h).argmax(1)).float().mean())
            sur_acc = float((surrogate(x_h).argmax(1) == y_h).float().mean())
        sims, hits = [], 0
        for c in range(10):
            rec = invert(surrogate, c, x_test.shape, steps=invert_steps)
            sims.append(similarity(rec[0], class_means[c]))
            with torch.no_grad():
                hits += int(victim(rec).argmax(1).item() == c)
        row = {"defense": d, "api_utility_acc": utility, "queries_answered": used, "extraction_fidelity": agree,
               "surrogate_acc": sur_acc, "inversion_similarity": float(np.mean(sims)),
               "inversion_success": hits / 10, "victim_acc": victim_acc}
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        rows.append(row)
    return rows
