from typing import Any, Dict, List, Optional

import numpy as np


def ema(x: List[float], alpha: float = 0.3) -> List[float]:
    out, m = [], None
    for v in x:
        if v is None:
            out.append(m)
            continue
        m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def trend_view(rounds: List[Dict], window: int = 5) -> Dict[str, Any]:
    r = [x["round"] for x in rounds]
    acc = [x["global"]["accuracy"] for x in rounds]
    f1 = [x["global"]["f1"] for x in rounds]
    loss = [x["global"]["loss"] for x in rounds]
    acc_ema = ema(acc)
    alerts = []
    for k in range(1, len(rounds)):
        if acc[k] < acc_ema[k - 1] - 0.05:
            alerts.append({"round": r[k], "type": "drop", "value": acc[k],
                           "msg": "정확도 급락 (%.3f → %.3f)" % (acc_ema[k - 1], acc[k])})
        if loss[k] > 1.5 * min(loss[:k]) and loss[k] > loss[k - 1]:
            alerts.append({"round": r[k], "type": "diverge", "value": loss[k], "msg": "손실 증가/발산 의심"})
    slope = None
    if len(acc) >= window:
        y = np.asarray(acc_ema[-window:])
        slope = float(np.polyfit(np.arange(window), y, 1)[0])
        if abs(slope) < 1e-3 and acc[-1] < 0.999:
            alerts.append({"round": r[-1], "type": "plateau", "value": acc[-1],
                           "msg": "정체 (최근 %d 라운드 기울기 %.4f)" % (window, slope)})
    return {"round": r, "accuracy": acc, "f1": f1, "loss": loss, "acc_ema": acc_ema,
            "precision": [x["global"]["precision"] for x in rounds],
            "recall": [x["global"]["recall"] for x in rounds],
            "fair_std": [x["fairness"]["std"] for x in rounds],
            "fair_worst": [x["fairness"]["worst10"] for x in rounds],
            "slope": slope, "alerts": alerts}


def outlier_view(rounds: List[Dict]) -> Dict[str, Any]:
    pts = []
    for x in rounds:
        for c in x["clients"]:
            if c.get("z") is not None:
                pts.append({"round": x["round"], "client": c["id"], "z": c["z"], "norm": c.get("update_norm"),
                            "cos": c.get("cos"), "outlier": bool(c.get("outlier")), "edge": c.get("edge")})
    return {"points": pts}


def contribution_view(rounds: List[Dict], n_clients: int) -> Dict[str, Any]:
    W = np.zeros((len(rounds), n_clients))
    for k, x in enumerate(rounds):
        for c in x["clients"]:
            W[k, c["id"]] = c.get("weight") or 0.0
    total = W.sum(0)
    order = np.argsort(-total)
    share = total[order] / max(total.sum(), 1e-12)
    return {"stream": {"round": [x["round"] for x in rounds], "weights": W.tolist()},
            "pareto": {"client": order.tolist(), "value": total[order].tolist(), "share": share.tolist(),
                       "cum": np.cumsum(share).tolist()}}


def _to_tree(Z: np.ndarray, n: int) -> Dict[str, Any]:
    nodes: Dict[int, Dict[str, Any]] = {i: {"name": str(i), "client": i, "height": 0.0} for i in range(n)}
    for k, (a, b, h, _) in enumerate(Z):
        nodes[n + k] = {"name": "", "height": float(h), "children": [nodes[int(a)], nodes[int(b)]]}
    return nodes[n + len(Z) - 1] if len(Z) else nodes[0]


def similarity_view(sketch: Dict[Any, List[float]], clients: List[Dict], n_clients: int,
                    social_thresh: float = 0.5) -> Dict[str, Any]:
    ids = sorted(int(k) for k in sketch.keys())
    if len(ids) < 2:
        return {"ids": ids, "tree": None, "matrix": [], "order": ids, "links": []}
    V = np.asarray([sketch[k] if k in sketch else sketch[str(k)] for k in ids], dtype=np.float64)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = V @ V.T
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    Dm = np.clip(1 - S, 0, 2)
    np.fill_diagonal(Dm, 0)
    Z = linkage(squareform(Dm, checks=False), method="average")
    tree = _to_tree(Z, len(ids))

    def relabel(nd):
        if "client" in nd:
            nd["client"] = ids[nd["client"]]
            nd["name"] = str(nd["client"])
        for ch in nd.get("children", []):
            relabel(ch)
    relabel(tree)
    order = [ids[i] for i in leaves_list(Z)]
    links = [{"source": ids[a], "target": ids[b], "value": float(S[a, b])}
             for a in range(len(ids)) for b in range(a + 1, len(ids)) if S[a, b] >= social_thresh]
    return {"ids": ids, "tree": tree, "matrix": S.tolist(), "order": order, "links": links}


def data_view(clients: List[Dict]) -> Dict[str, Any]:
    M = np.asarray([c["class_counts"] for c in clients], dtype=np.float64)
    row = M / np.maximum(M.sum(1, keepdims=True), 1)
    return {"ratio": row.tolist(), "counts": M.astype(int).tolist(),
            "data_ratio": [c["ratio"] for c in clients], "entropy": [c["entropy"] for c in clients],
            "norm_entropy": [c["norm_entropy"] for c in clients], "n": [c["n"] for c in clients]}


def network_view(rounds: List[Dict], topology: Dict, n_clients: int) -> Dict[str, Any]:
    n_edges = len(topology.get("edges", []))
    ce = np.zeros((n_clients, n_edges))
    es = np.zeros(n_edges)
    for x in rounds:
        for c in x["clients"]:
            if c.get("edge", -1) >= 0:
                ce[c["id"], c["edge"]] += (c.get("bytes_up") or 0) + (c.get("bytes_down") or 0)
        for e in x["edges"]:
            es[e["id"]] += e.get("bytes_up", 0) + topology.get("model_bytes", 0)
    nodes = [{"name": "client %d" % i, "kind": "client", "id": i} for i in range(n_clients)] + \
            [{"name": "edge %d" % j, "kind": "edge", "id": j} for j in range(n_edges)] + \
            [{"name": "server", "kind": "server"}]
    links = [{"source": i, "target": n_clients + j, "value": float(ce[i, j])}
             for i in range(n_clients) for j in range(n_edges) if ce[i, j] > 0]
    links += [{"source": n_clients + j, "target": n_clients + n_edges, "value": float(es[j])}
              for j in range(n_edges) if es[j] > 0]

    last = rounds[-1] if rounds else None
    tree = None
    if last:
        tree = {"name": "server", "kind": "server", "status": "ok", "children": []}
        for e in last["edges"]:
            ch = [{"name": "c%d" % c["id"], "kind": "client", "id": c["id"], "status": c["status"],
                   "outlier": bool(c.get("outlier")), "latency": c.get("latency_ms"), "n": c["n"],
                   "quality": c.get("quality")}
                  for c in last["clients"] if c["edge"] == e["id"]]
            tree["children"].append({"name": "edge %d" % e["id"], "kind": "edge", "id": e["id"],
                                     "status": e["status"], "latency": e["latency_ms"], "children": ch,
                                     "accepted": e["accepted"], "assigned": e["assigned"]})

    matchings = [{"round": x["round"], **x["matching"]} for x in rounds if x.get("matching")]
    ranks = {"round": [x["round"] for x in rounds], "rank": [x.get("rank") for x in rounds],
             "quality": [[c.get("quality") for c in x["clients"]] for x in rounds]}
    status = [{"round": x["round"], "edges": [{"id": e["id"], "status": e["status"], "latency": e["latency_ms"],
                                                 "timeouts": e["timeouts"], "outliers": e["outliers"]}
                                                for e in x["edges"]],
               "timeouts": sum(1 for c in x["clients"] if c["status"] == "timeout")} for x in rounds]
    return {"sankey": {"nodes": nodes, "links": links}, "tree": tree, "matchings": matchings,
            "ranks": ranks, "status": status}


def resource_view(rounds: List[Dict]) -> Dict[str, Any]:
    keys = ["cpu_min", "cpu_mean", "cpu_max", "mem_min", "mem_mean", "mem_max", "sys_cpu_mean", "sys_mem_mean"]
    out: Dict[str, Any] = {"round": [x["round"] for x in rounds]}
    for k in keys:
        out[k] = [x.get("resource", {}).get(k) for x in rounds]
    out["wall_s"] = [x.get("wall_s") for x in rounds]
    out["sim_time_ms"] = [x.get("sim_time_ms") for x in rounds]
    out["bytes"] = [x.get("bytes_total") for x in rounds]
    return out


def client_series_view(rounds: List[Dict], n_clients: int, logs: List[Dict]) -> Dict[str, Any]:
    fields = ["val_acc", "global_acc", "train_loss", "latency_ms", "quality", "weight", "z"]
    series = {f: [[None] * len(rounds) for _ in range(n_clients)] for f in fields}
    status = [[None] * len(rounds) for _ in range(n_clients)]
    for k, x in enumerate(rounds):
        for c in x["clients"]:
            for f in fields:
                series[f][c["id"]][k] = c.get(f)
            status[c["id"]][k] = c["status"]
    counts = [{"timeout": 0, "outlier": 0, "malicious_update": 0} for _ in range(n_clients)]
    for l in logs:
        if l.get("client") is not None and l["event"] in counts[0]:
            counts[l["client"]][l["event"]] += 1
    return {"series": series, "status": status, "counts": counts}


def compute_views(snap: Dict[str, Any]) -> Dict[str, Any]:
    rounds, clients = snap["rounds"], snap["clients"]
    n = len(clients)
    if not rounds:
        return {"data": data_view(clients) if clients else None}
    return {
        "trend": trend_view(rounds),
        "outliers": outlier_view(rounds),
        "contribution": contribution_view(rounds, n),
        "similarity": similarity_view(snap.get("sketch", {}), clients, n),
        "data": data_view(clients),
        "network": network_view(rounds, snap["topology"], n),
        "resource": resource_view(rounds),
        "clients": client_series_view(rounds, n, snap.get("logs", [])),
    }


def compare_runs(snaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    from ..common.metrics import convergence_round
    out = []
    for s in snaps:
        rs = s["rounds"]
        if not rs:
            continue
        acc = [x["global"]["accuracy"] for x in rs]
        out.append({
            "id": s["meta"]["id"], "name": s["meta"]["name"], "status": s["status"],
            "algorithm": s["config"].get("algorithm"), "seed": s["config"].get("seed"),
            "class_acc": rs[-1].get("class_acc"),
            "round": [x["round"] for x in rs], "accuracy": acc, "f1": [x["global"]["f1"] for x in rs],
            "loss": [x["global"]["loss"] for x in rs],
            "fair_std": [x["fairness"]["std"] for x in rs], "fair_worst": [x["fairness"]["worst10"] for x in rs],
            "final": {"accuracy": acc[-1], "f1": rs[-1]["global"]["f1"],
                      "precision": rs[-1]["global"]["precision"], "recall": rs[-1]["global"]["recall"],
                      "fair_std": rs[-1]["fairness"]["std"], "fair_worst10": rs[-1]["fairness"]["worst10"],
                      "convergence_round": convergence_round(acc),
                      "total_mb": sum(x["bytes_total"] for x in rs) / 2 ** 20,
                      "sim_time_s": sum(x["sim_time_ms"] for x in rs) / 1e3,
                      "outliers": sum(1 for x in rs for c in x["clients"] if c.get("outlier"))},
            "client_acc": [c.get("global_acc") for c in rs[-1]["clients"]],
        })
    return {"runs": out}
