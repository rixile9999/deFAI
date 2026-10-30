"""Server-side optimisers: FedAvgM and the adaptive FedOpt family.

Both treat the aggregated client update ``delta = mean(local - global)`` as a
pseudo-gradient and apply an optimiser step on the server.

* FedAvgM:  Hsu et al., "Measuring the Effects of Non-Identical Data Distribution for
            Federated Visual Classification", 2019 (arXiv:1909.06335).
* FedAdagrad / FedAdam / FedYogi: Reddi et al., "Adaptive Federated Optimization",
            ICLR 2021 (arXiv:2003.00295), Algorithm 2.

Following Reddi et al. (and unlike Flower's FedOpt) the second moment starts at
``tau^2`` and no Adam bias correction is applied.
"""

from logging import INFO

import numpy as np
from flwr.supercore import log

from defai.aggregation import Arrays
from defai.strategies.base import FedAvg


class FedAvgM(FedAvg):
    """FedAvg with server momentum, using the ``torch.optim.SGD`` convention.

    With pseudo-gradient ``g = -delta``:  ``v <- momentum * v + g`` (``v <- g`` in the
    first round) and ``x <- x - server_learning_rate * v``.
    """

    name = "fedavgm"

    def __init__(self, *, momentum: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self._velocity: Arrays | None = None

    def update_global(self, server_round: int, delta: Arrays) -> Arrays:
        grad = {k: -d for k, d in delta.items()}
        if self._velocity is None:
            self._velocity = grad
        else:
            self._velocity = {k: self.momentum * self._velocity[k] + grad[k] for k in grad}
        return self._step(-self.server_learning_rate, self._velocity)

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> Server momentum: %s", self.momentum)


class _FedOpt(FedAvg):
    """Adaptive server optimiser; ``server_learning_rate`` is the paper's eta."""

    def __init__(self, *, beta_1: float = 0.9, beta_2: float = 0.99, tau: float = 1e-3, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= beta_1 < 1.0 or not 0.0 <= beta_2 < 1.0:
            raise ValueError("beta_1 and beta_2 must be in [0, 1)")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.beta_1, self.beta_2, self.tau = beta_1, beta_2, tau
        self._m: Arrays | None = None
        self._v: Arrays | None = None

    def _second_moment(self, v: np.ndarray, delta_sq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update_global(self, server_round: int, delta: Arrays) -> Arrays:
        if self._m is None:
            self._m = {k: np.zeros_like(d) for k, d in delta.items()}
            self._v = {k: np.full_like(d, self.tau**2) for k, d in delta.items()}
        self._m = {k: self.beta_1 * self._m[k] + (1 - self.beta_1) * d for k, d in delta.items()}
        self._v = {k: self._second_moment(self._v[k], np.square(d)) for k, d in delta.items()}
        step = {k: self._m[k] / (np.sqrt(self._v[k]) + self.tau) for k in delta}
        return self._step(self.server_learning_rate, step)

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> %s: beta_1=%s beta_2=%s tau=%s", type(self).__name__, self.beta_1, self.beta_2, self.tau)


class FedAdagrad(_FedOpt):
    """v <- v + delta^2"""

    name = "fedadagrad"

    def _second_moment(self, v, delta_sq):
        return v + delta_sq


class FedAdam(_FedOpt):
    """v <- beta_2 * v + (1 - beta_2) * delta^2"""

    name = "fedadam"

    def _second_moment(self, v, delta_sq):
        return self.beta_2 * v + (1 - self.beta_2) * delta_sq


class FedYogi(_FedOpt):
    """v <- v - (1 - beta_2) * delta^2 * sign(v - delta^2)"""

    name = "fedyogi"

    def _second_moment(self, v, delta_sq):
        return v - (1 - self.beta_2) * delta_sq * np.sign(v - delta_sq)
