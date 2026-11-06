import math

import numpy as np


class NetworkModel:
    def __init__(self, n_clients, n_edges, seed=0, base_ms=15.0, ms_per_unit=180.0, ms_per_sample=0.35,
                 jitter_sigma=0.25, bw_decay=3.0):
        rng = np.random.default_rng(seed + 7)
        self.n, self.m = n_clients, n_edges
        g = int(math.ceil(math.sqrt(n_edges)))
        centers = [((i % g + 0.5) / g, (i // g + 0.5) / g) for i in range(n_edges)]
        self.edge_pos = np.asarray(centers) + rng.normal(0, 0.03, (n_edges, 2))
        self.client_pos = rng.uniform(0, 1, (n_clients, 2))
        self.speed = rng.lognormal(0.0, 0.45, n_clients)
        self.bw_mbps = rng.uniform(8.0, 60.0, n_clients)
        self.base_ms, self.ms_per_unit, self.ms_per_sample = base_ms, ms_per_unit, ms_per_sample
        self.jitter_sigma, self.bw_decay = jitter_sigma, bw_decay
        self.dist = np.linalg.norm(self.client_pos[:, None, :] - self.edge_pos[None, :, :], axis=2)
        self.edge_global_ms = 20.0 + 40.0 * np.linalg.norm(self.edge_pos - 0.5, axis=1)

    def latency(self):
        return self.base_ms + self.ms_per_unit * self.dist

    def comm_ms(self, model_bytes, rng=None, delay_factor=1.0, spikes=None):
        bw = self.bw_mbps[:, None] / (1.0 + self.bw_decay * self.dist)
        transfer = 2 * model_bytes * 8 / (bw * 1e6) * 1e3
        lat = self.latency()
        if rng is not None:
            lat = lat * rng.lognormal(0.0, self.jitter_sigma, (self.n, self.m))
        comm = (2 * lat + transfer) * delay_factor
        if spikes is not None:
            comm = comm * spikes[:, None]
        return comm

    def compute_ms(self, steps):
        return np.asarray(steps, dtype=float) * self.ms_per_sample * 32 * self.speed

    def describe(self):
        return {"client_pos": self.client_pos.round(4).tolist(), "edge_pos": self.edge_pos.round(4).tolist(),
                "speed": self.speed.round(4).tolist(), "bw_mbps": self.bw_mbps.round(2).tolist()}


def on_time_prob(expected_ms, deadline_ms, sigma=0.35):
    expected_ms = np.asarray(expected_ms, dtype=float)
    if deadline_ms is None or deadline_ms <= 0 or not np.isfinite(deadline_ms):
        return np.ones_like(expected_ms)
    z = (np.log(deadline_ms) - np.log(np.maximum(expected_ms, 1e-6))) / sigma
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
