import heapq
from collections import OrderedDict

import numpy as np


class Link:
    __slots__ = ("src", "dst", "bw_mbps", "latency_ms", "price_per_mb")

    def __init__(self, src, dst, bw_mbps, latency_ms, price_per_mb):
        self.src, self.dst = src, dst
        self.bw_mbps, self.latency_ms, self.price_per_mb = bw_mbps, latency_ms, price_per_mb

    @property
    def key(self):
        return self.src, self.dst

    def cost(self, nbytes, weights, load=0.0):
        mb = nbytes / 2 ** 20
        transfer_ms = nbytes * 8 / (self.bw_mbps * 1e6) * 1e3
        congestion = (load + nbytes) / (self.bw_mbps * 1e6 / 8)
        return (weights["price"] * self.price_per_mb * mb + weights["time"] * (self.latency_ms + transfer_ms) / 1e3
                + weights["congestion"] * congestion)


class Topology:
    def __init__(self, num_clients, num_edges=1, aps=4, routers=3, seed=0):
        rng = np.random.default_rng(seed + 31)
        self.clients = [f"c{j}" for j in range(num_clients)]
        self.aps = [f"ap{k}" for k in range(aps)]
        self.routers = [f"r{k}" for k in range(routers)]
        self.edges = [f"e{i}" for i in range(num_edges)]
        self.cloud = "cloud"
        self.links = {}
        self.out = {}

        def add(a, b, bw, lat, price):
            link = Link(a, b, bw, lat, price)
            self.links[(a, b)] = link
            self.out.setdefault(a, []).append(link)

        for j, c in enumerate(self.clients):
            first = j % aps
            for k in sorted({first, int(rng.integers(aps))}):
                add(c, self.aps[k], rng.uniform(10, 50), rng.uniform(5, 20), 0.0)
        for k, ap in enumerate(self.aps):
            for r in sorted({k % routers, int(rng.integers(routers))}):
                add(ap, self.routers[r], rng.uniform(50, 200), rng.uniform(5, 15), rng.uniform(0.001, 0.004))
        for r_i, r in enumerate(self.routers):
            near = r_i % num_edges
            for e in range(num_edges):
                far = 1.0 if e == near else 2.5
                add(r, self.edges[e], rng.uniform(100, 400) / far, rng.uniform(5, 30) * far,
                    rng.uniform(0.002, 0.01) * far)
            add(r, self.cloud, rng.uniform(50, 150), rng.uniform(30, 80), rng.uniform(0.02, 0.05))
        for e in self.edges:
            add(e, self.cloud, rng.uniform(200, 1000), rng.uniform(10, 40), rng.uniform(0.005, 0.02))
        self.order = self.clients + self.aps + self.routers + self.edges + [self.cloud]
        self.rank = {n: i for i, n in enumerate(self.order)}
        assert all(self.rank[l.src] < self.rank[l.dst] for l in self.links.values())

    def static_path(self, src, dst):
        path, node = [src], src
        while node != dst:
            options = [l for l in self.out.get(node, []) if self._reaches(l.dst, dst)]
            if not options:
                raise ValueError(f"{dst} unreachable from {src}")
            node = options[0].dst
            path.append(node)
        return path

    def _reaches(self, a, b, memo=None):
        memo = {} if memo is None else memo
        if a == b:
            return True
        if a in memo:
            return memo[a]
        memo[a] = any(self._reaches(l.dst, b, memo) for l in self.out.get(a, []))
        return memo[a]

    def best_path(self, src, dst, nbytes, weights, loads=None):
        loads = loads or {}
        dist = {src: 0.0}
        prev = {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, float("inf")):
                continue
            for l in self.out.get(u, []):
                nd = d + l.cost(nbytes, weights, loads.get(l.key, 0.0))
                if nd < dist.get(l.dst, float("inf")):
                    dist[l.dst], prev[l.dst] = nd, u
                    heapq.heappush(heap, (nd, l.dst))
        if dst not in dist:
            raise ValueError(f"{dst} unreachable from {src}")
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        return path[::-1]

    def hops(self, path):
        return [self.links[(a, b)] for a, b in zip(path, path[1:])]


class LRUCache:
    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.items = OrderedDict()
        self.size = 0
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.items:
            self.items.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        return False

    def put(self, key, nbytes):
        if key in self.items:
            self.items.move_to_end(key)
            return
        self.items[key] = nbytes
        self.size += nbytes
        while self.size > self.capacity and self.items:
            _, b = self.items.popitem(last=False)
            self.size -= b
