import os

import numpy as np
import torch
from flwr.app import ConfigRecord, Message, MessageType, MetricRecord, RecordDict

from fl_core.plugins import Middleware
from fl_core.records import delta, flatten, float_keys, to_arrays, to_record, unflatten
from fl_core.strategies import StrategyWrapper
from security_sim.keys import node_principal, reported_pid, sign


def pid_of(node):
    return int(node.context.node_config["partition-id"])


def round_of(msg):
    return int(msg.content["config"].get("server-round", 0))


def data_noise(std, seed):
    def transform(x, y, cfg, pid, config):
        g = torch.Generator().manual_seed(seed * 7919 + pid)
        return x + std * torch.randn(x.shape, generator=g), y

    return transform


def label_flip(num_classes=10):
    def transform(x, y, cfg, pid, config):
        return x, (num_classes - 1) - y

    return transform


def only_for(pids, transform):
    def wrapped(x, y, cfg, pid, config):
        return transform(x, y, cfg, pid, config) if pid in pids else (x, y)

    return wrapped


def data_report(x, y, cfg, pid, config):
    counts = np.bincount(np.asarray(y), minlength=10).astype(float)
    p = counts[counts > 0] / max(counts.sum(), 1)
    return {"data-mean": float(x.mean()), "data-std": float(x.std()),
            "label-entropy": float(-(p * np.log2(p)).sum())}


def replace_arrays(reply, arrays):
    reply.content["arrays"] = to_record(arrays)
    return reply


class ModelPoison(Middleware):
    def __init__(self, pids, mode="gaussian", scale=4.0, std=0.5, seed=0):
        self.pids, self.mode, self.scale, self.std, self.seed = set(pids), mode, scale, std, seed

    def after(self, msg, reply, node):
        pid = pid_of(node)
        if reply.has_error() or msg.metadata.message_type != MessageType.TRAIN or pid not in self.pids:
            return reply
        g = to_arrays(msg.content["arrays"])
        local = to_arrays(reply.content["arrays"])
        rng = np.random.default_rng([self.seed, pid, round_of(msg)])
        out = {}
        for k, v in local.items():
            if not np.issubdtype(v.dtype, np.floating):
                out[k] = v
            elif self.mode == "sign_flip":
                out[k] = (g[k] - self.scale * (v - g[k])).astype(v.dtype)
            else:
                out[k] = (v + rng.normal(0.0, self.std, v.shape)).astype(v.dtype)
        return replace_arrays(reply, out)


class Signer(Middleware):
    def __init__(self, keys):
        self.keys = keys

    def after(self, msg, reply, node):
        if reply.has_error() or msg.metadata.message_type != MessageType.TRAIN:
            return reply
        sign(reply.content, node_principal(node.context.node_id), round_of(msg), self.keys, pid=pid_of(node))
        return reply


class RelayTamper(Middleware):
    def __init__(self, pids, std=0.05, seed=0, events=None):
        self.pids, self.std, self.seed, self.events = set(pids), std, seed, events

    def after(self, msg, reply, node):
        pid = pid_of(node)
        if reply.has_error() or msg.metadata.message_type != MessageType.TRAIN or pid not in self.pids:
            return reply
        local = to_arrays(reply.content["arrays"])
        rng = np.random.default_rng([self.seed, pid, round_of(msg), 3])
        tampered = {k: (v + rng.normal(0, self.std, v.shape)).astype(v.dtype)
                    if np.issubdtype(v.dtype, np.floating) else v for k, v in local.items()}
        return replace_arrays(reply, tampered)


def crafted_direction(keys_like, rnd, seed):
    rng = np.random.default_rng([seed, rnd, 17])
    d = rng.standard_normal(sum(int(np.prod(v.shape)) for v in keys_like.values()))
    return d / np.linalg.norm(d)


class ClientAttackStrategy(StrategyWrapper):
    def __init__(self, inner, attack, colluders, keys, scale=3.0, jitter=0.75, sybils=0, seed=0, owner=None):
        super().__init__(inner)
        self.attack, self.colluders, self.keys = attack, set(colluders), keys
        self.scale, self.jitter, self.sybils, self.seed, self.owner = scale, jitter, sybils, seed, owner
        self.global_arrays = None

    def configure_train(self, server_round, arrays, config, grid):
        self.global_arrays = to_arrays(arrays)
        return self.inner.configure_train(server_round, arrays, config, grid)

    def _collude(self, server_round, replies):
        mine = [r for r in replies if not r.has_error() and reported_pid(r) in self.colluders]
        if not mine:
            return replies
        keys = float_keys(self.global_arrays)
        g = {k: self.global_arrays[k] for k in keys}
        U = np.stack([flatten(delta(to_arrays(r.content["arrays"]), self.global_arrays), keys) for r in mine])
        crafted = -self.scale * U.mean(0)
        rng = np.random.default_rng([self.seed, server_round, 29])
        for r in mine:
            u = crafted
            if self.attack == "adaptive_collusion":
                noise = rng.standard_normal(len(crafted))
                u = crafted + self.jitter * np.linalg.norm(crafted) * noise / np.linalg.norm(noise)
            arrays = to_arrays(r.content["arrays"])
            arrays.update(unflatten(flatten(g, keys) + u, g))
            replace_arrays(r, arrays)
            sign(r.content, node_principal(r.metadata.src_node_id), server_round, self.keys, pid=reported_pid(r))
        return replies

    def _sybil(self, server_round, replies):
        real = [r for r in replies if not r.has_error()]
        if not real:
            return replies
        keys = float_keys(self.global_arrays)
        U = np.stack([flatten(delta(to_arrays(r.content["arrays"]), self.global_arrays), keys) for r in real])
        n_ex = int(np.median([r.content["metrics"]["num-examples"] for r in real]))
        template = to_arrays(real[0].content["arrays"])
        fake_keys = type(self.keys)(os.urandom(16).hex())
        out = list(replies)
        for k in range(self.sybils):
            node_id = 100000 + k
            arrays = dict(template)
            arrays.update(unflatten(flatten({kk: self.global_arrays[kk] for kk in keys}, keys)
                                    - self.scale * U.mean(0), {kk: self.global_arrays[kk] for kk in keys}))
            instruction = Message(content=RecordDict(), dst_node_id=node_id, message_type=MessageType.TRAIN)
            content = RecordDict({"arrays": to_record(arrays),
                                  "metrics": MetricRecord({"num-examples": n_ex, "train_loss": 0.1,
                                                           "initial-loss": 0.1, "local-steps": 1})})
            sign(content, node_principal(node_id), server_round, fake_keys, pid=node_id - 1)
            content["auth"] = ConfigRecord(dict(content["auth"]))
            out.append(Message(content=content, reply_to=instruction))
        return out

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        if self.attack in ("insider_collusion", "adaptive_collusion"):
            replies = self._collude(server_round, replies)
        if self.attack == "sybil":
            replies = self._sybil(server_round, replies)
        return self.inner.aggregate_train(server_round, replies)


def poison_edge_model(arrays, global_arrays, rnd, scale, seed):
    keys = float_keys(arrays)
    g = {k: global_arrays[k] for k in keys}
    u = flatten(delta(arrays, global_arrays), keys)
    d = crafted_direction(g, rnd, seed)
    crafted = -scale * np.linalg.norm(u) * d - u
    out = dict(arrays)
    out.update(unflatten(flatten(arrays, keys) + crafted, g))
    return out
