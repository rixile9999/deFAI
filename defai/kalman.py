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
    if not isinstance(observation, np.ndarray) or observation.dtype.kind not in "fiu" or not observation.size:
        raise ValueError("observation must be a nonempty real numeric array")
    d = observation.astype(np.float64)
    if not np.isfinite(d).all():
        raise ValueError("observation must be finite")
    if previous is None:
        previous = LayerState(np.zeros_like(d), config.initial_variance,
                              config.process_noise, config.observation_noise)
    if previous.mean.shape != d.shape:
        raise ValueError("observation shape changed")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            predicted = config.rho * previous.mean
            prior = config.rho**2 * previous.variance + previous.process_noise
            gain = prior / (prior + previous.observation_noise)
            mean = (1 - gain) * predicted + gain * d
            # Joseph form remains valid when gain constraints are added.
            variance = max((1 - gain)**2 * prior + gain**2 * previous.observation_noise,
                           config.variance_floor)
            innovation_mse = float(np.mean(np.square(d - predicted)))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("non-finite Kalman arithmetic") from exc
    if not np.isfinite(mean).all() or not all(math.isfinite(v) for v in (prior, gain, variance, innovation_mse)):
        raise ValueError("non-finite Kalman state")
    return LayerState(mean, variance, previous.process_noise, previous.observation_noise, gain, innovation_mse)
