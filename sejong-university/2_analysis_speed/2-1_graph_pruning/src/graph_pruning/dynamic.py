import copy
import time

import numpy as np
import torch
import torch.nn as nn

from graph_pruning.graph import (build_graph, init_meta, kept_channels, l1_keep, path_importance, percentile_threshold,
                                 prune, prune_model, taylor_importance)


def count_macs(model, input_shape):
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            total[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    model.train(was)
    for h in handles:
        h.remove()
    return int(total[0])


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def latency_ms(model, input_shape, reps=30, warmup=5, threads=1):
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = copy.deepcopy(model).eval()
    x = torch.randn(input_shape)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t)
    torch.set_num_threads(old)
    return float(np.median(times) * 1e3)


def importance_fn(kind, batches=None):
    if kind == "path":
        return path_importance
    if kind == "taylor":
        return lambda model, graph: taylor_importance(model, graph, batches)
    raise ValueError(f"unknown importance {kind!r}")


def prune_dynamic(model, recover, evaluate, input_shape, percentile=30.0, eta=0.5, tol=0.1, max_iters=8,
                  min_eta=0.05, target=None, recovery_steps=100, importance="path", batches=None, log=print):
    t0 = time.perf_counter()
    model = copy.deepcopy(model)
    graph = build_graph(model, input_shape)
    init_meta(model, graph)
    imp = importance_fn(importance, batches)
    p0, m0 = count_params(model), count_macs(model, input_shape)
    ref = copy.deepcopy(model)
    recover(ref, recovery_steps, 0)
    ref_chunks = 1
    L_ref, A_ref = evaluate(ref)
    tau = percentile_threshold(imp(model, graph), percentile)
    history = [{"iter": 0, "tau": 0.0, "eta": eta, "removed": 0, "channels": kept_channels(graph), "params": p0,
                "macs": m0, "params_red": 0.0, "macs_red": 0.0, "val_loss": L_ref, "val_acc": A_ref, "L_ref": L_ref,
                "rel_change": 0.0, "accepted": 1, "note": "reference"}]
    log(history[-1])
    current, tau_acc, stop, chunks = model, None, "max_iters", 0
    for k in range(1, max_iters + 1):
        cand, keep, removed = prune(current, tau, imp)
        p, m = count_params(cand), count_macs(cand, input_shape)
        row = {"iter": k, "tau": tau, "eta": eta, "removed": removed, "channels": kept_channels(build_graph(cand)),
               "params": p, "macs": m, "params_red": 1 - p / p0, "macs_red": 1 - m / m0}
        if removed == 0:
            row.update(accepted=1, note="no-op", L_ref=L_ref)
            history.append(row)
            log(row)
            tau_acc, tau = tau, tau * (1 + eta)
            continue
        while ref_chunks < chunks + 1:
            recover(ref, recovery_steps, ref_chunks)
            ref_chunks += 1
            L_ref, _ = evaluate(ref)
        Lp, Ap = evaluate(cand)
        recover(cand, recovery_steps, chunks)
        L, A = evaluate(cand)
        d = (L - L_ref) / L_ref
        ok = d <= tol
        row.update(val_loss_pre=Lp, val_acc_pre=Ap, val_loss=L, val_acc=A, L_ref=L_ref, rel_change=d,
                   accepted=int(ok), note="accept" if ok else "reject")
        history.append(row)
        log({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()})
        if ok:
            current, tau_acc = cand, tau
            chunks += 1
            if target is not None and 1 - p / p0 >= target:
                stop = "target"
                break
            tau = tau * (1 + eta)
        else:
            eta /= 2
            if eta < min_eta:
                stop = "eta"
                break
            tau = tau_acc * (1 + eta) if tau_acc is not None else tau / (1 + eta)
    return {"model": current, "history": history, "tau": tau_acc, "stop": stop, "params0": p0, "macs0": m0,
            "L_ref": L_ref, "acc_ref": A_ref, "seconds": time.perf_counter() - t0}


def uniform_l1(model, input_shape, ratio):
    graph = build_graph(model, input_shape)
    init_meta(model, graph)
    return prune_model(model, l1_keep(model, graph, ratio, include_coupled=True), graph)


def uniform_l1_matched(model, input_shape, target_params, iters=20):
    lo, hi = 0.0, 0.95
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        cand = uniform_l1(copy.deepcopy(model), input_shape, mid)
        p = count_params(cand)
        best = cand
        if p > target_params:
            lo = mid
        else:
            hi = mid
    return uniform_l1(copy.deepcopy(model), input_shape, hi) if best is not None else model
