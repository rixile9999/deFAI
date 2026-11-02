"""Layer-scalar uncertainty and vector means for aggregated update observations.

Only NumPy is required. These are optimizer heuristics, not calibrated guarantees
about a true gradient, generalization, or differential privacy.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class KalmanConfig:
    rho: float = 1.0
    process_noise: float = 1e-4
    observation_noise: float = 1e-2
    initial_variance: float = 0.0
    variance_floor: float = 1e-12

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 0 <= self.rho <= 1:
            raise ValueError("rho must be in [0, 1]")
        for name in ("process_noise", "initial_variance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("observation_noise", "variance_floor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LayerState:
    mean: np.ndarray
    variance: float
    process_noise: float
    observation_noise: float
    gain: float = 0.0
    innovation_mse: float = 0.0


def layer_step(observation: np.ndarray, previous: LayerState | None, config: KalmanConfig) -> LayerState:
    """Return a proposed next state without mutating observations or prior state."""
    raise NotImplementedError("Kalman recurrence is the next implementation milestone")
