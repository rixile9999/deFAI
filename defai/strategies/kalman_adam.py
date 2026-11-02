"""FedAdam research extension replacing its first moment with a Kalman mean."""
import math

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

    def __init__(self, *, kalman=None, kalman_blend=1.0, **kwargs):
        super().__init__(**kwargs)
        if isinstance(kalman_blend, bool) or not isinstance(kalman_blend, (int, float)) or not math.isfinite(kalman_blend) or not 0 <= kalman_blend <= 1:
            raise ValueError("kalman_blend must be finite and in [0, 1]")
        if kalman is not None and not isinstance(kalman, KalmanConfig):
            raise ValueError("kalman must be a KalmanConfig")
        if not math.isfinite(self.tau) or not math.isfinite(self.server_learning_rate) or self.server_learning_rate <= 0:
            raise ValueError("tau and server_learning_rate must be finite and positive")
        self.kalman = kalman or KalmanConfig()
        self.kalman_blend = kalman_blend
        self._layers = {}
        self._ema = None
        self._last_round = 0

    def update_global(self, server_round, delta):
        raise NotImplementedError("First-moment integration is the next milestone")
