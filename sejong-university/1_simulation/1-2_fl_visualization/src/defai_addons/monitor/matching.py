from typing import Dict, List, Optional

import numpy as np


class QualityEstimator:
    def __init__(self, n_clients: int, beta: float = 0.7, init: float = 0.5):
        self.beta = beta
        self.Q = np.full(n_clients, init, dtype=np.float64)
        self.n_obs = np.zeros(n_clients, dtype=np.int64)

    def update(self, i: int, q: float) -> float:
        self.Q[i] = self.beta * self.Q[i] + (1 - self.beta) * float(q)
        self.n_obs[i] += 1
        return float(self.Q[i])


def client_profit(R: np.ndarray, n: np.ndarray, cost: np.ndarray, dist: np.ndarray,
                  capacity: np.ndarray, lam: float = 0.02) -> np.ndarray:
    expected_load = (capacity - 1).clip(min=0) * n.mean()
    share = n[:, None] / (n[:, None] + expected_load[None, :])
    return R[None, :] * share - (cost * n)[:, None] - lam * dist


def edge_score(Q: np.ndarray, n: np.ndarray, dist: np.ndarray, gamma: float = 0.01) -> np.ndarray:
    return Q[:, None] * np.log1p(n)[:, None] - gamma * dist


def deferred_acceptance(client_pref: np.ndarray, edge_pref: np.ndarray, capacity: np.ndarray,
                        acceptable: Optional[np.ndarray] = None) -> np.ndarray:
    nc, ne = client_pref.shape
    if acceptable is None:
        acceptable = np.ones_like(client_pref, dtype=bool)
    order = [[j for j in np.argsort(-client_pref[i]) if acceptable[i, j]] for i in range(nc)]
    nxt = np.zeros(nc, dtype=np.int64)
    held: List[List[int]] = [[] for _ in range(ne)]
    free = list(range(nc))
    while free:
        i = free.pop()
        if nxt[i] >= len(order[i]):
            continue
        j = order[i][nxt[i]]
        nxt[i] += 1
        held[j].append(i)
        if len(held[j]) > capacity[j]:
            held[j].sort(key=lambda c: -edge_pref[c, j])
            rejected = held[j].pop()
            free.append(rejected)
    assign = -np.ones(nc, dtype=np.int64)
    for j, cs in enumerate(held):
        for i in cs:
            assign[i] = j
    return assign


def blocking_pairs(assign: np.ndarray, client_pref: np.ndarray, edge_pref: np.ndarray,
                   capacity: np.ndarray, acceptable: np.ndarray) -> int:
    nc, ne = client_pref.shape
    cnt = 0
    for i in range(nc):
        for j in range(ne):
            if not acceptable[i, j] or assign[i] == j:
                continue
            if assign[i] >= 0 and client_pref[i, j] <= client_pref[i, assign[i]]:
                continue
            members = np.where(assign == j)[0]
            if len(members) < capacity[j] or any(edge_pref[i, j] > edge_pref[m, j] for m in members):
                cnt += 1
    return cnt


class GameMatcher:
    def __init__(self, n_clients: int, n_edges: int, positions_c: np.ndarray, positions_e: np.ndarray,
                 n_samples: np.ndarray, capacity: np.ndarray, rewards: np.ndarray, costs: np.ndarray,
                 beta: float = 0.7, lam: float = 0.02, gamma: float = 0.01):
        self.nc, self.ne = n_clients, n_edges
        self.dist = np.linalg.norm(positions_c[:, None, :] - positions_e[None, :, :], axis=-1)
        self.n = n_samples.astype(np.float64)
        self.capacity = capacity
        self.R, self.cost = rewards, costs
        self.lam, self.gamma = lam, gamma
        self.quality = QualityEstimator(n_clients, beta=beta)

    def match(self) -> Dict[str, np.ndarray]:
        profit = client_profit(self.R, self.n, self.cost, self.dist, self.capacity, self.lam)
        score = edge_score(self.quality.Q, self.n, self.dist, self.gamma)
        acceptable = profit > 0
        assign = deferred_acceptance(profit, score, self.capacity, acceptable)
        load = np.bincount(assign[assign >= 0], minlength=self.ne)
        for i in np.where(assign < 0)[0]:
            for j in np.argsort(self.dist[i]):
                if load[j] < self.capacity[j]:
                    assign[i] = j
                    load[j] += 1
                    break
        rank = np.empty(self.nc, dtype=np.int64)
        rank[np.argsort(-self.quality.Q)] = np.arange(1, self.nc + 1)
        edge_rank = np.zeros(self.nc, dtype=np.int64)
        for j in range(self.ne):
            m = np.where(assign == j)[0]
            if m.size:
                edge_rank[m[np.argsort(-score[m, j])]] = np.arange(1, m.size + 1)
        return {"assign": assign, "profit": profit, "score": score, "rank": rank,
                "edge_rank": edge_rank, "Q": self.quality.Q.copy(),
                "blocking": blocking_pairs(assign, profit, score, self.capacity, acceptable)}
