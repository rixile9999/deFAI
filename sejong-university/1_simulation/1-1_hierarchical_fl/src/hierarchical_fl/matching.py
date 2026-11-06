import numpy as np

from hierarchical_fl.network import on_time_prob


class QualityEMA:
    def __init__(self, n_clients, beta=0.7, init=0.5):
        self.beta = beta
        self.q = np.full(n_clients, init, dtype=float)
        self.seen = np.zeros(n_clients, dtype=int)

    def update(self, cid, q_obs):
        self.q[cid] = self.beta * self.q[cid] + (1.0 - self.beta) * float(np.clip(q_obs, 0.0, 1.0))
        self.seen[cid] += 1


def deferred_acceptance(utility, priority, capacity, available, participating):
    n, m = utility.shape
    assign = -np.ones(n, dtype=int)
    prefs = {j: [i for i in np.argsort(-utility[j]) if i in available and utility[j, i] > 0] for j in participating}
    nxt = {j: 0 for j in participating}
    held = {i: [] for i in range(m)}
    free = list(participating)
    while free:
        j = free.pop(0)
        if nxt[j] >= len(prefs[j]):
            continue
        i = prefs[j][nxt[j]]
        nxt[j] += 1
        if priority[i, j] < 0:
            free.append(j)
            continue
        held[i].append(j)
        if len(held[i]) > capacity[i]:
            held[i].sort(key=lambda c: -priority[i, c])
            free.append(held[i].pop())
    for i, js in held.items():
        for j in js:
            assign[j] = i
    return assign


def blocking_pairs(assign, utility, priority, capacity, available, participating):
    count = 0
    load = {i: [j for j in participating if assign[j] == i] for i in available}
    for j in participating:
        cur = assign[j]
        cur_u = utility[j, cur] if cur >= 0 else 0.0
        for i in available:
            if i == cur or utility[j, i] <= max(cur_u, 0.0) or priority[i, j] < 0:
                continue
            if len(load[i]) < capacity[i] or min(priority[i, c] for c in load[i]) < priority[i, j]:
                count += 1
    return count


class Matcher:
    def __init__(self, mode, net, n_samples, capacity, cfg, seed=0):
        self.mode = mode
        self.net = net
        self.n_samples = np.asarray(n_samples, dtype=float)
        self.capacity = list(capacity)
        self.rng = np.random.default_rng(seed + 11)
        self.reward = np.asarray(cfg.get("edge-reward") or [1.0] * net.m, dtype=float)
        self.cost_data = float(cfg.get("cost-data", 0.05))
        self.cost_comm = float(cfg.get("cost-comm", 0.1))
        self.min_quality = float(cfg.get("min-quality", 0.0))
        self.trust_rel = float(cfg.get("trust-rel", 0.0))
        self.static = None

    def utilities(self, q_hat, expected_ms, comm_ms, deadline_ms):
        p_on = on_time_prob(expected_ms, deadline_ms)
        expected_reward = self.reward[None, :] * q_hat[:, None] * p_on
        cost = (self.cost_data * (self.n_samples / max(self.n_samples.mean(), 1e-9))[:, None]
                + self.cost_comm * comm_ms / max(float(np.median(comm_ms)), 1e-9))
        utility = expected_reward - cost
        priority = (q_hat[:, None] * p_on).T.copy()
        priority[:, q_hat < self.min_quality] = -1.0
        if self.trust_rel > 0:
            priority[:, q_hat < self.trust_rel * float(np.median(q_hat))] = -1.0
        return utility, priority, p_on

    def assign(self, q_hat, expected_ms, comm_ms, deadline_ms, available, participating):
        u, prio, p_on = self.utilities(q_hat, expected_ms, comm_ms, deadline_ms)
        n = len(q_hat)
        available = list(available)
        if self.mode == "matching":
            a = deferred_acceptance(u, prio, self.capacity, available, participating)
        elif self.mode == "static":
            if self.static is None:
                base = self.net.latency()
                load = [0] * self.net.m
                self.static = -np.ones(n, dtype=int)
                for j in np.argsort(base.min(1)):
                    for i in np.argsort(base[j]):
                        if load[i] < self.capacity[i]:
                            self.static[j] = i
                            load[i] += 1
                            break
            a = -np.ones(n, dtype=int)
            for j in participating:
                if self.static[j] in available:
                    a[j] = self.static[j]
        elif self.mode == "random":
            a = -np.ones(n, dtype=int)
            slots = [i for i in available for _ in range(self.capacity[i])]
            self.rng.shuffle(slots)
            for k, j in enumerate(self.rng.permutation(list(participating))):
                if k < len(slots):
                    a[j] = slots[k]
        else:
            raise ValueError(f"unknown matching mode {self.mode!r}")
        bp = blocking_pairs(a, u, prio, self.capacity, available, participating)
        return a, {"utility": u, "priority": prio, "p_on_time": p_on, "blocking_pairs": int(bp)}


def priority_rank(priority):
    out = []
    for i in range(priority.shape[0]):
        order = [int(j) for j in np.argsort(-priority[i]) if priority[i, j] >= 0]
        out.append(order)
    return out
