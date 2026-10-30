"""Renyi-DP accountant for the Gaussian mechanism.

Each federated round releases ``mean(clipped updates) + N(0, (z * C / m)^2)``. With
sensitivity ``C / m`` (one client added/removed, fixed denominator ``m``) this is a
Gaussian mechanism with noise multiplier ``z`` whose RDP at order ``alpha`` is
``alpha / (2 z^2)`` (Mironov, 2017). RDP composes additively over rounds and is
converted to (epsilon, delta)-DP with the tight conversion of Balle et al. (2020).

No privacy amplification by client subsampling is credited: Flower samples a fixed
number of clients per round (not Poisson), for which the amplified bounds do not
apply directly. The reported epsilon is therefore a conservative upper bound.
This mirrors Flower's ``sampling_method="no-amplification"`` accounting.
"""

import math

import numpy as np

# Same default grid as TensorFlow Privacy / Google's dp-accounting.
DEFAULT_ORDERS = tuple([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))


class RdpAccountant:
    """Track cumulative privacy loss of repeated Gaussian releases."""

    def __init__(self, orders=DEFAULT_ORDERS):
        self.orders = np.asarray(orders, dtype=np.float64)
        if self.orders.size == 0 or np.any(self.orders <= 1.0):
            raise ValueError("RDP orders must be non-empty and greater than 1")
        self._rdp = np.zeros_like(self.orders)
        self.num_releases = 0

    def _gaussian_rdp(self, noise_multiplier):
        if noise_multiplier <= 0:
            return np.full_like(self.orders, np.inf)
        return self.orders / (2.0 * noise_multiplier**2)

    def compose(self, noise_multiplier, count=1):
        """Account for ``count`` releases with the given noise multiplier."""
        self._rdp = self._rdp + count * self._gaussian_rdp(noise_multiplier)
        self.num_releases += count

    @staticmethod
    def _to_epsilon(orders, rdp, delta):
        eps = (
            rdp
            + np.log((orders - 1.0) / orders)
            - (math.log(delta) + np.log(orders)) / (orders - 1.0)
        )
        return max(0.0, float(np.min(eps)))

    def get_epsilon(self, delta):
        """Cumulative epsilon at ``delta``."""
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.num_releases == 0:
            return 0.0
        return self._to_epsilon(self.orders, self._rdp, delta)

    def would_exceed(self, noise_multiplier, delta, max_epsilon):
        """Whether one more release would push epsilon above ``max_epsilon``."""
        projected = self._rdp + self._gaussian_rdp(noise_multiplier)
        return self._to_epsilon(self.orders, projected, delta) > max_epsilon
