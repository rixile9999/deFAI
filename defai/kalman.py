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

    @classmethod
    def from_ema(cls, beta_1=.9, *, observation_noise=.01, process_adaptation=0., gain_floor=False):
        """Calibrate a rho=1 steady-state filter to an EMA's observation weight.

        K=1-beta, P=K*R and Q=K**2*R/(1-K). Fixed noise then keeps
        the EMA recurrence. A gain floor prevents adaptive over-smoothing.
        """
        if isinstance(beta_1, bool) or not isinstance(beta_1, Real) or not math.isfinite(beta_1) or not 0 <= beta_1 < 1:
            raise ValueError("beta_1 must be finite and in [0, 1)")
        if not isinstance(gain_floor, bool): raise ValueError("gain_floor must be boolean")
        gain = 1-beta_1
        if beta_1 == 0:
            return cls(process_noise=0., observation_noise=observation_noise, gain_min=1., gain_max=1.,
                       process_adaptation=process_adaptation)
        initial = gain*observation_noise
        if initial <= 0: raise ValueError("calibrated variance must remain positive")
        return cls(process_noise=gain**2*observation_noise/beta_1, observation_noise=observation_noise,
                   initial_variance=initial, variance_floor=min(1e-12, initial), process_adaptation=process_adaptation,
                   gain_min=gain if gain_floor else 0.)

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
            try:
                numeric = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite real number") from exc
            if not math.isfinite(numeric): raise ValueError(f"{name} must be a finite real number")
            object.__setattr__(self, name, numeric)
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


def stable_l2(arrays):
    """Euclidean norm without overflowing a representable result."""
    norm = 0.0
    for a in arrays:
        peak = max(abs(float(np.min(a))), abs(float(np.max(a))))
        if not peak:
            continue
        if peak <= math.sqrt(np.finfo(np.float64).max / a.size):
            layer_norm = math.sqrt(float(np.sum(np.square(a, dtype=np.float64))))
        else:
            layer_norm = peak * math.sqrt(float(np.sum(np.square(a / peak))))
        norm = math.hypot(norm, layer_norm)
    return norm


def layer_step(observation: np.ndarray, previous: LayerState | None, config: KalmanConfig) -> LayerState:
    """Return a proposed next state without mutating observations or prior state."""
    if not isinstance(observation, np.ndarray) or observation.dtype.kind not in "fiu" or not observation.size:
        raise ValueError("observation must be a nonempty real numeric array")
    d = observation.astype(np.float64, copy=False)
    if not np.isfinite(d).all():
        raise ValueError("observation must be finite")
    if previous is None:
        previous = LayerState(np.zeros_like(d), config.initial_variance,
                              config.process_noise, config.observation_noise)
    if not isinstance(previous, LayerState) or not isinstance(previous.mean, np.ndarray):
        raise ValueError("previous must contain a NumPy layer mean")
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
            gain = float(min(config.gain_max, max(config.gain_min, gain)))
            mean = np.asarray((1 - gain) * predicted + gain * d)
            # Joseph form remains valid when gain constraints are added.
            variance = max((1 - gain)**2 * prior + gain**2 * previous.observation_noise,
                           config.variance_floor)
            # Divide the fresh residual before squaring so a representable
            # mean square does not overflow an intermediate square or sum.
            residual = np.asarray((d - predicted) / math.sqrt(d.size))
            np.square(residual, out=residual)
            innovation_mse = float(np.sum(residual))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("non-finite Kalman arithmetic") from exc
    if not np.isfinite(mean).all() or not all(math.isfinite(v) for v in (prior, gain, variance, innovation_mse)):
        raise ValueError("non-finite Kalman state")
    # Innovation matching is a bounded proxy, not an identification of the true
    # process/observation noise. New values apply only to the NEXT round.
    q, r = previous.process_noise, previous.observation_noise
    if config.process_adaptation:
        q_target = min(config.noise_ceiling, max(config.variance_floor,
                       innovation_mse - config.rho**2 * previous.variance - previous.observation_noise))
        q = (1 - config.process_adaptation) * q + config.process_adaptation * q_target
    if config.observation_adaptation:
        r_target = min(config.noise_ceiling, max(config.variance_floor, innovation_mse - prior))
        r = (1 - config.observation_adaptation) * r + config.observation_adaptation * r_target
    return LayerState(mean, variance, q, r, gain, innovation_mse)
