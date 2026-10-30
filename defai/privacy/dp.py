"""Central differential privacy for federated averaging: clipping + Gaussian noise.

Each round the server releases the *fixed-denominator* mean of clipped client updates
plus Gaussian noise::

    release = (1 / m) * sum_i clip(update_i, C)  +  N(0, (z * C / m)^2 * I)

``m`` is the number of clients sampled for the round (a public constant), ``C`` the
clipping norm and ``z`` the noise multiplier, so the privacy cost per round is that
of a Gaussian mechanism with noise multiplier ``z`` (see ``accountant.py``).

The design follows Flower's ``DifferentialPrivacy{Server,Client}Side{Fixed,Adaptive}Clipping``
strategy wrappers (McMahan et al. 2018, Andrew et al. 2019) with two deliberate
differences:

* The noise is added to the mean *update* before any server-side optimiser
  (momentum, FedAdam, ...) runs, so the optimiser is mere post-processing and the
  guarantee holds for every mean-based strategy, not only plain FedAvg.
* For adaptive clipping the noise of a round is calibrated to the clipping norm
  that was used to clip that round, and the norm is updated afterwards; clients
  report whether they were *unclipped* (b_i = 1[||u_i|| <= C]) as in Andrew et al.

``seed`` exists for reproducible experiments only; leave it ``None`` (OS entropy) for
anything that needs real privacy.
"""

import numpy as np

from defai.aggregation import Arrays, l2_norm
from defai.privacy.accountant import RdpAccountant

KEY_CLIP_NORM = "dp-clip-norm"        # ConfigRecord key: clipping norm for client-side clipping
KEY_ADAPTIVE = "dp-adaptive"          # ConfigRecord key: clients must report the unclipped bit
KEY_UNCLIPPED_BIT = "dp-unclipped-bit"  # MetricRecord key: 1 if the client's update was not clipped


def clip_to_norm(update: Arrays, clip_norm: float, tol: float = 0.0) -> tuple[Arrays, bool]:
    """Scale ``update`` so its global L2 norm is at most ``clip_norm`` (FlatClip).

    Returns the clipped update and whether it counts as clipped, i.e. its norm exceeded
    ``clip_norm * (1 + tol)``. The scaling itself is exact regardless of ``tol``; the
    tolerance only keeps float32 round-off in an already-clipped update (norm a hair
    above ``clip_norm``) from being reported as clipped.
    """
    norm = l2_norm(update)
    if norm <= clip_norm:
        return update, False
    scale = clip_norm / norm
    return {k: v * scale for k, v in update.items()}, norm > clip_norm * (1.0 + tol)


def adaptive_noise_multiplier(noise_multiplier: float, bit_noise_std: float) -> float:
    """Noise multiplier for the model update when clipped-count bits are also released.

    The bit query (centred to sensitivity 1/2) has noise multiplier ``2 * bit_noise_std``;
    the two Gaussian queries compose to ``noise_multiplier`` overall when
    ``z_update^-2 + (2 * bit_noise_std)^-2 = noise_multiplier^-2``.
    """
    if noise_multiplier == 0:
        return 0.0
    if noise_multiplier >= 2 * bit_noise_std:
        raise ValueError(
            f"clipped-count noise std ({bit_noise_std:g}) is too small for noise multiplier "
            f"{noise_multiplier:g}: need noise_multiplier < 2 * clipped_count_stddev "
            f"(run-config: raise 'dp-clipped-count-stddev' or lower 'dp-noise-multiplier')"
        )
    return float((noise_multiplier**-2 - (2 * bit_noise_std) ** -2) ** -0.5)


class _GaussianDP:
    """Shared machinery of fixed and adaptive clipping."""

    adaptive = False

    def __init__(
        self,
        noise_multiplier: float,
        clip_norm: float,
        *,
        side: str = "server",
        delta: float = 1e-5,
        max_epsilon: float | None = None,
        seed: int | None = None,
    ):
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if side not in ("server", "client"):
            raise ValueError("side must be 'server' or 'client'")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.noise_multiplier = float(noise_multiplier)
        self.clip_norm = float(clip_norm)
        self.side = side
        self.delta = delta
        self.max_epsilon = max_epsilon
        self.accountant = RdpAccountant()
        self._rng = np.random.default_rng(seed)

    # --- strategy-facing API -------------------------------------------------
    @property
    def epsilon(self) -> float:
        return self.accountant.get_epsilon(self.delta)

    def client_config(self) -> dict:
        """Entries for the train ConfigRecord so clients can clip their own updates."""
        if self.side != "client":
            return {}
        return {KEY_CLIP_NORM: self.clip_norm, KEY_ADAPTIVE: self.adaptive}

    def begin_round(self, num_sampled: int) -> None:
        """Check the settings for a round with ``num_sampled`` clients, before any client trains."""
        self._update_noise_multiplier(num_sampled)  # raises for infeasible settings

    def budget_exhausted(self) -> bool:
        """True if one more round would exceed ``max_epsilon``."""
        return self.max_epsilon is not None and self.accountant.would_exceed(
            self.noise_multiplier, self.delta, self.max_epsilon
        )

    def aggregate(
        self,
        updates: list[Arrays],
        num_sampled: int,
        unclipped_bits: list[int | None] | None = None,
    ) -> tuple[Arrays, dict[str, float]]:
        """Clip, average (fixed denominator ``num_sampled``) and noise the updates."""
        if not updates:
            raise ValueError("no updates to aggregate")
        if len(updates) > num_sampled:
            raise ValueError("more updates than sampled clients would void the sensitivity bound")

        self._validate_bits(unclipped_bits)  # before anything is released or accounted
        clip_norm = self.clip_norm  # the norm used for *this* round
        total = {k: np.zeros(v.shape, dtype=np.float64) for k, v in updates[0].items()}
        num_clipped = 0
        for u in updates:
            # Always enforce the bound on the server; a no-op for compliant clients.
            clipped, was_clipped = clip_to_norm(u, clip_norm, tol=1e-6)
            num_clipped += was_clipped
            for k in total:
                total[k] += clipped[k]

        z = self._update_noise_multiplier(num_sampled)
        std = z * clip_norm / num_sampled
        released = {k: v / num_sampled for k, v in total.items()}
        if std > 0:
            released = {k: v + self._rng.normal(0.0, std, v.shape) for k, v in released.items()}

        self.accountant.compose(self.noise_multiplier)
        # Only quantities covered by the accounting are published. The exact number of clipped
        # clients is not (the adaptive mode releases a noisy version of it, below).
        metrics = {
            "dp-clip-norm": clip_norm,
            "dp-noise-std": std,
            "dp-epsilon": self.epsilon,
        }
        noisy_fraction = self._update_clip_norm(num_sampled, len(updates), num_clipped, unclipped_bits)
        if noisy_fraction is not None:
            metrics["dp-noisy-unclipped-fraction"] = noisy_fraction
        return released, metrics

    # --- hooks ---------------------------------------------------------------
    def _update_noise_multiplier(self, num_sampled: int) -> float:
        return self.noise_multiplier

    def _validate_bits(self, unclipped_bits) -> None:
        """Hook: reject malformed client-reported bits (only adaptive clipping uses them)."""

    def _update_clip_norm(self, num_sampled, num_replies, num_clipped, unclipped_bits) -> float | None:
        """Fixed clipping keeps C constant (and releases no clipping statistic)."""
        return None

    def describe(self) -> str:
        return (
            f"{type(self).__name__}(side={self.side}, z={self.noise_multiplier}, "
            f"C={self.clip_norm}, delta={self.delta})"
        )


class FixedClipping(_GaussianDP):
    """Constant clipping norm (McMahan et al., 2018)."""


class AdaptiveClipping(_GaussianDP):
    """Quantile-tracking clipping norm (Andrew et al., 2019).

    After each round ``C <- C * exp(-clip_lr * (b - target_quantile))`` where ``b`` is the
    noisy fraction of clients whose update was *not* clipped, so ``C`` converges to the
    ``target_quantile`` of the update-norm distribution.
    """

    adaptive = True

    def __init__(
        self,
        noise_multiplier: float,
        clip_norm: float,
        *,
        target_quantile: float = 0.5,
        clip_lr: float = 0.2,
        clipped_count_stddev: float | None = None,
        **kwargs,
    ):
        super().__init__(noise_multiplier, clip_norm, **kwargs)
        if not 0.0 <= target_quantile <= 1.0:
            raise ValueError("target_quantile must be in [0, 1]")
        if clip_lr <= 0:
            raise ValueError("clip_lr must be positive")
        if clipped_count_stddev is not None and clipped_count_stddev < 0:
            raise ValueError("clipped_count_stddev must be non-negative")
        self.target_quantile = target_quantile
        self.clip_lr = clip_lr
        self.clipped_count_stddev = clipped_count_stddev

    def _bit_std(self, num_sampled: int) -> float:
        if self.clipped_count_stddev is not None:
            return self.clipped_count_stddev
        if self.noise_multiplier <= 0:
            return 0.0  # the update is not noised, so neither is the count
        # Andrew et al. recommend m / 20. That is infeasible for small federations (it needs
        # noise_multiplier < m / 10), so never go below the noise multiplier itself.
        return max(num_sampled / 20, self.noise_multiplier)

    def _update_noise_multiplier(self, num_sampled: int) -> float:
        return adaptive_noise_multiplier(self.noise_multiplier, self._bit_std(num_sampled))

    def _validate_bits(self, unclipped_bits) -> None:
        if self.side != "client":
            return
        if unclipped_bits is None or any(b is None for b in unclipped_bits):
            raise ValueError(f"client-side adaptive clipping needs every reply to carry '{KEY_UNCLIPPED_BIT}'")
        if any(b not in (0, 1) for b in unclipped_bits):
            # The bit query's sensitivity bound (and the clip-norm update) assume each bit is 0 or 1
            raise ValueError(f"'{KEY_UNCLIPPED_BIT}' must be 0 or 1, got {sorted(set(unclipped_bits))}")

    def _update_clip_norm(self, num_sampled, num_replies, num_clipped, unclipped_bits) -> float:
        if self.side == "client":
            num_unclipped = float(sum(int(b) for b in unclipped_bits))
        else:
            num_unclipped = float(num_replies - num_clipped)
        noisy = num_unclipped + self._rng.normal(0.0, self._bit_std(num_sampled))
        unclipped_fraction = noisy / num_replies
        self.clip_norm *= float(np.exp(-self.clip_lr * (unclipped_fraction - self.target_quantile)))
        return float(unclipped_fraction)
