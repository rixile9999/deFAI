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
    gain_min: float = 0.0
    gain_max: float = 1.0
    process_adaptation: float = 0.0
    observation_adaptation: float = 0.0
    noise_ceiling: float = 1e6

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
        if not 0 <= self.gain_min <= self.gain_max <= 1:
            raise ValueError("gain_min/gain_max must satisfy 0 <= min <= max <= 1")
        for name in ("process_adaptation", "observation_adaptation"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.noise_ceiling < self.variance_floor:
            raise ValueError("noise_ceiling must be >= variance_floor")
        if max(self.process_noise, self.observation_noise) > self.noise_ceiling:
            raise ValueError("noise_ceiling must cover configured process_noise and observation_noise")


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
    if (previous.mean.dtype.kind != "f" or not np.isfinite(previous.mean).all() or
            any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                for v in (previous.variance, previous.process_noise, previous.observation_noise)) or
            previous.variance < 0 or not 0 <= previous.process_noise <= config.noise_ceiling or
            not 0 < previous.observation_noise <= config.noise_ceiling):
        raise ValueError("invalid previous Kalman state")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            predicted = config.rho * previous.mean
            prior = config.rho**2 * previous.variance + previous.process_noise
            # Normalize the two variances before summing: finite P and R can
            # otherwise overflow P+R and incorrectly produce a zero gain.
            scale = max(prior, previous.observation_noise)
            gain = (prior / scale) / (prior / scale + previous.observation_noise / scale)
            gain = float(np.clip(gain, config.gain_min, config.gain_max))
            mean = (1 - gain) * predicted + gain * d
            # Joseph form remains valid when gain constraints are added.
            variance = max((1 - gain)**2 * prior + gain**2 * previous.observation_noise,
                           config.variance_floor)
            innovation_mse = float(np.mean(np.square(d - predicted)))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("non-finite Kalman arithmetic") from exc
    if not np.isfinite(mean).all() or not all(math.isfinite(v) for v in (prior, gain, variance, innovation_mse)):
        raise ValueError("non-finite Kalman state")
    # Innovation matching is a bounded proxy, not an identification of the true
    # process/observation noise. New values apply only to the NEXT round.
    q_target = float(np.clip(innovation_mse - config.rho**2 * previous.variance - previous.observation_noise,
                             config.variance_floor, config.noise_ceiling))
    r_target = float(np.clip(innovation_mse - prior, config.variance_floor, config.noise_ceiling))
    q = (1 - config.process_adaptation) * previous.process_noise + config.process_adaptation * q_target
    r = (1 - config.observation_adaptation) * previous.observation_noise + config.observation_adaptation * r_target
    return LayerState(mean, variance, q, r, gain, innovation_mse)
