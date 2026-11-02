"""FedAdam research extension replacing its first moment with a Kalman mean."""
import math
from dataclasses import asdict

import numpy as np

from defai.kalman import KalmanConfig, layer_step
from defai.strategies.fedopt import FedAdam


class FedAdamKalman(FedAdam):
    """Layer-wise Kalman first moment; raw FedAdam second moment is retained.

    ``kalman_blend=0`` is the unmodified EMA control. For intermediate blends,
    keep an independent EMA control and combine its mean with the Kalman mean.
    This does not feed a smoothed observation through another EMA.
    """
    name = "fedadam-kalman"

    def __init__(self, *, kalman=None, kalman_blend=1.0, max_step_norm=0.0, **kwargs):
        super().__init__(**kwargs)
        if isinstance(kalman_blend, bool) or not isinstance(kalman_blend, (int, float)) or not math.isfinite(kalman_blend) or not 0 <= kalman_blend <= 1:
            raise ValueError("kalman_blend must be finite and in [0, 1]")
        if kalman is not None and not isinstance(kalman, KalmanConfig):
            raise ValueError("kalman must be a KalmanConfig")
        if not math.isfinite(self.tau) or not math.isfinite(self.server_learning_rate) or self.server_learning_rate <= 0:
            raise ValueError("tau and server_learning_rate must be finite and positive")
        self.kalman = kalman or KalmanConfig()
        self.kalman_blend = kalman_blend
        if isinstance(max_step_norm, bool) or not isinstance(max_step_norm, (int, float)) or not math.isfinite(max_step_norm) or max_step_norm < 0:
            raise ValueError("max_step_norm must be finite and nonnegative (0 disables)")
        self.max_step_norm = max_step_norm
        self._layers = {}
        self._ema = None
        self._last_round = 0

    def update_global(self, server_round, delta):
        if isinstance(server_round, bool) or not isinstance(server_round, int) or server_round <= self._last_round:
            raise ValueError("server_round must increase strictly from a positive integer")
        if self._global is None or not delta or delta.keys() != self._global.keys():
            raise ValueError("delta must match the configured global model")
        for k, d in delta.items():
            if not isinstance(d, np.ndarray) or d.dtype.kind != "f" or not d.size or d.shape != self._global[k].shape:
                raise ValueError(f"invalid floating update shape/dtype: {k}")
            if not np.isfinite(d).all() or not np.isfinite(self._global[k]).all():
                raise ValueError(f"non-finite model or update: {k}")
            if self._m is not None and (k not in self._m or self._m[k].shape != d.shape):
                raise ValueError("optimizer model schema changed; create a new strategy")
        if self._m is not None and self._m.keys() != delta.keys():
            raise ValueError("optimizer model schema changed; create a new strategy")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                ema = None
                if self.kalman_blend < 1:
                    previous = self._ema or {k: np.zeros_like(d) for k, d in delta.items()}
                    ema = {k: self.beta_1 * previous[k] + (1 - self.beta_1) * d for k, d in delta.items()}
                layers = ({k: layer_step(d, self._layers.get(k), self.kalman) for k, d in delta.items()}
                          if self.kalman_blend > 0 else {})
                if self.kalman_blend == 0:
                    moment = ema
                elif self.kalman_blend == 1:
                    moment = {k: s.mean for k, s in layers.items()}
                else:
                    moment = {k: (1 - self.kalman_blend) * ema[k] + self.kalman_blend * layers[k].mean for k in delta}
                old_v = self._v or {k: np.full_like(d, self.tau**2) for k, d in delta.items()}
                variance = {k: self._second_moment(old_v[k], np.square(d)) for k, d in delta.items()}
                step = {k: moment[k] / (np.sqrt(variance[k]) + self.tau) for k in delta}
                norm = float(np.sqrt(sum(float(np.sum(np.square(a))) for a in step.values())))
                scale = self.server_learning_rate
                if self.max_step_norm and norm:
                    scale = min(scale, self.max_step_norm / norm)
                result = self._step(scale, step)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("non-finite FedAdam Kalman arithmetic") from exc
        if any(not np.isfinite(a).all() for group in (moment, variance, result) for a in group.values()):
            raise ValueError("non-finite FedAdam Kalman result")
        # Commit only after every layer and output dtype has passed validation.
        self._m, self._v, self._ema, self._layers = moment, variance, ema, layers
        self._last_round = server_round
        self._round_metrics.update({"kf-effective-lr": scale, "kf-step-norm": scale * norm})
        if layers:
            self._round_metrics.update({
                "kf-gain-min": min(s.gain for s in layers.values()),
                "kf-gain-max": max(s.gain for s in layers.values()),
                "kf-variance-mean": float(np.mean([s.variance for s in layers.values()])),
                "kf-innovation-mse": float(np.mean([s.innovation_mse for s in layers.values()])),
                "kf-process-noise-mean": float(np.mean([s.process_noise for s in layers.values()])),
                "kf-observation-noise-mean": float(np.mean([s.observation_noise for s in layers.values()])),
            })
        return result

    def optimizer_state_bytes(self):
        """Numeric payload bytes, excluding model, Python containers and DP state."""
        arrays = [a for group in (self._m, self._v, self._ema) if group for a in group.values()]
        arrays += [s.mean for s in self._layers.values()]
        unique = {id(a): a for a in arrays}
        return sum(a.nbytes for a in unique.values()) + 6 * 8 * len(self._layers)

    def _optimizer_config(self):
        return {"kalman": asdict(self.kalman), "blend": self.kalman_blend,
                "max_step_norm": self.max_step_norm, "beta_1": self.beta_1,
                "beta_2": self.beta_2, "tau": self.tau, "learning_rate": self.server_learning_rate}

    def state_dict(self):
        """Copy optimizer-only state. Restore the global model separately first.

        This is not a full distributed-training checkpoint: client/sampler and
        privacy-accountant states are not included. DP checkpointing is rejected.
        """
        from copy import deepcopy
        if self.dp is not None:
            raise ValueError("DP resume requires the separate privacy accountant; optimizer-only checkpoint is unsupported")
        if self._global is None:
            raise ValueError("configure the global model before exporting optimizer state")
        return deepcopy({"version": 1, "config": self._optimizer_config(),
                         "schema": {k: (v.shape, v.dtype.str) for k, v in self._global.items()},
                         "last_round": self._last_round, "m": self._m, "v": self._v,
                         "ema": self._ema, "layers": {k: asdict(v) for k, v in self._layers.items()}})

    def summary(self):
        from logging import INFO
        from flwr.supercore import log
        super().summary()
        log(INFO, "\t└──> Kalman first moment: blend=%s config=%s max_step_norm=%s (research)",
            self.kalman_blend, self.kalman, self.max_step_norm)
