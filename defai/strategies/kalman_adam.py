"""FedAdam research extension replacing its first moment with a Kalman mean."""
import math
from dataclasses import asdict

import numpy as np

from defai.kalman import KalmanConfig, LayerState, layer_step, stable_l2
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
                norm = stable_l2(step.values())
                if not math.isfinite(norm):
                    raise ValueError("non-finite combined server update norm")
                scale = self.server_learning_rate
                if self.max_step_norm and norm:
                    scale = min(scale, self.max_step_norm / norm)
                result = self._step(scale, step)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("non-finite FedAdam Kalman arithmetic") from exc
        if any(not np.isfinite(a).all() for group in (moment, variance, result) for a in group.values()):
            raise ValueError("non-finite FedAdam Kalman result")
        metrics = {"kf-effective-lr": scale, "kf-step-norm": scale * norm}
        if layers:
            def mean_attribute(name):
                return sum(getattr(s, name) / len(layers) for s in layers.values())
            metrics.update({
                "kf-gain-min": min(s.gain for s in layers.values()),
                "kf-gain-max": max(s.gain for s in layers.values()),
                "kf-variance-mean": mean_attribute("variance"),
                "kf-innovation-mse": mean_attribute("innovation_mse"),
                "kf-process-noise-mean": mean_attribute("process_noise"),
                "kf-observation-noise-mean": mean_attribute("observation_noise"),
            })
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("non-finite FedAdam Kalman diagnostics")
        # Diagnostics are part of a successful round, not a post-commit failure.
        self._m, self._v, self._ema, self._layers = moment, variance, ema, layers
        self._last_round = server_round
        self._round_metrics.update(metrics)
        return result

    def optimizer_state_bytes(self):
        """Numeric payload bytes, excluding model, Python containers and DP state."""
        arrays = [a for group in (self._m, self._v, self._ema) if group for a in group.values()]
        arrays += [s.mean for s in self._layers.values()]
        unique = {id(a): a for a in arrays}
        return sum(a.nbytes for a in unique.values()) + 5 * 8 * len(self._layers)

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

    def load_state_dict(self, state):
        """Validate the entire optimizer checkpoint before changing live state."""
        from copy import deepcopy
        if self.dp is not None or self._global is None:
            raise ValueError("optimizer resume needs a configured global model and no DP mechanism")
        state = deepcopy(state)
        schema = {k: (v.shape, v.dtype.str) for k, v in self._global.items()}
        if not isinstance(state, dict) or state.get("version") != 1 or state.get("config") != self._optimizer_config() or state.get("schema") != schema:
            raise ValueError("incompatible optimizer checkpoint config/schema/version")
        rnd = state.get("last_round")
        if isinstance(rnd, bool) or not isinstance(rnd, int) or rnd < 0:
            raise ValueError("invalid checkpoint round")

        def group(name, required):
            value = state.get(name)
            if not required:
                if value is not None: raise ValueError(f"unexpected checkpoint {name}")
                return None
            if not isinstance(value, dict) or value.keys() != schema.keys():
                raise ValueError(f"invalid checkpoint {name} keys")
            for k, a in value.items():
                if not isinstance(a, np.ndarray) or a.dtype.kind != "f" or a.shape != schema[k][0] or not np.isfinite(a).all():
                    raise ValueError(f"invalid checkpoint {name} array")
            return value

        moment, variance = group("m", rnd > 0), group("v", rnd > 0)
        ema = group("ema", rnd > 0 and self.kalman_blend < 1)
        if variance is not None and any(np.any(v < 0) for v in variance.values()):
            raise ValueError("negative checkpoint second moment")
        raw_layers = state.get("layers")
        expected = schema.keys() if rnd and self.kalman_blend > 0 else {}.keys()
        if not isinstance(raw_layers, dict) or raw_layers.keys() != expected:
            raise ValueError("invalid checkpoint layers")
        layers = {}
        for k, data in raw_layers.items():
            try:
                layer = LayerState(**data)
                scalars = (layer.variance, layer.process_noise, layer.observation_noise, layer.gain, layer.innovation_mse)
                if not isinstance(layer.mean, np.ndarray) or layer.mean.dtype.kind != "f" or layer.mean.shape != schema[k][0] or not np.isfinite(layer.mean).all():
                    raise ValueError("invalid layer mean")
                if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scalars):
                    raise ValueError("invalid layer scalars")
                if not (layer.variance >= self.kalman.variance_floor and
                        0 <= layer.process_noise <= self.kalman.noise_ceiling and
                        0 < layer.observation_noise <= self.kalman.noise_ceiling and
                        self.kalman.gain_min <= layer.gain <= self.kalman.gain_max and layer.innovation_mse >= 0):
                    raise ValueError("invalid layer bounds")
                layers[k] = layer
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError("invalid checkpoint layer") from exc
        if rnd:
            expected_m = (ema if self.kalman_blend == 0 else
                          {k: s.mean for k, s in layers.items()} if self.kalman_blend == 1 else
                          {k: (1-self.kalman_blend)*ema[k] + self.kalman_blend*s.mean for k, s in layers.items()})
            if any(not np.array_equal(moment[k], expected_m[k]) for k in moment):
                raise ValueError("checkpoint first moment disagrees with filter/EMA state")
        self._m, self._v, self._ema, self._layers, self._last_round = moment, variance, ema, layers, rnd

    def summary(self):
        from logging import INFO
        from flwr.supercore import log
        super().summary()
        log(INFO, "\t└──> Kalman first moment: blend=%s config=%s max_step_norm=%s (research)",
            self.kalman_blend, self.kalman, self.max_step_norm)
