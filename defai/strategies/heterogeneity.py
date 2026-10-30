"""Algorithms for non-IID data: FedProx, SCAFFOLD and MOON.

The server halves live here; the matching local-training halves are in
``defai.task.train`` / ``defai.client_app`` and are switched on by what the server
puts into the train message (``proximal-mu``, ``moon-mu``, the ``c`` control variate).

* FedProx:  Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
* SCAFFOLD: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
            Federated Learning", ICML 2020 (option II control variates).
* MOON:     Li et al., "Model-Contrastive Federated Learning", CVPR 2021.
* FedNova:  Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous
            Federated Optimization", NeurIPS 2020.
* q-FedAvg: Li et al., "Fair Resource Allocation in Federated Learning", ICLR 2020.
"""

from logging import INFO

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord
from flwr.supercore import log

from defai.aggregation import Arrays, weighted_mean
from defai.records import KEY_C, KEY_C_DELTA, to_record
from defai.strategies.base import ClientUpdate, FedAvg


class FedProx(FedAvg):
    """FedAvg on the server; clients minimise ``loss + mu/2 * ||w - w_global||^2``."""

    name = "fedprox"

    def __init__(self, *, proximal_mu: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")
        self.proximal_mu = proximal_mu

    def _extend_train_config(self, server_round: int, config: ConfigRecord) -> None:
        config["proximal-mu"] = self.proximal_mu

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> Proximal mu: %s", self.proximal_mu)


class Moon(FedAvg):
    """FedAvg on the server; clients add the model-contrastive loss

    ``mu * -log( e^{sim(z, z_glob)/tau} / (e^{sim(z, z_glob)/tau} + e^{sim(z, z_prev)/tau}) )``

    where ``z`` is the representation of the model being trained, ``z_glob`` that of
    the received global model and ``z_prev`` that of the client's previous local model.
    """

    name = "moon"

    def __init__(self, *, mu: float = 1.0, temperature: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if mu < 0 or temperature <= 0:
            raise ValueError("mu must be non-negative and temperature positive")
        self.mu = mu
        self.temperature = temperature

    def _extend_train_config(self, server_round: int, config: ConfigRecord) -> None:
        config["moon-mu"] = self.mu
        config["moon-temperature"] = self.temperature

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> MOON: mu=%s temperature=%s", self.mu, self.temperature)


class Scaffold(FedAvg):
    """SCAFFOLD: control variates correct client drift.

    Server state: global control variate ``c`` (zeros initially). Each sampled client
    ``i`` runs corrected SGD ``y <- y - lr * (g - c_i + c)``, then returns its model and
    ``dc_i = c_i^+ - c_i`` where ``c_i^+ = c_i - c + (x - y) / (K * lr)``. The server does::

        x <- x + server_lr * mean_i(y_i - x)
        c <- c + (|S| / N) * mean_i(dc_i)

    Plain (uniform) averaging is used, as in the paper. Not compatible with DP: the
    control-variate updates are not clipped, so the sensitivity bound would not hold.
    """

    name = "scaffold"
    supports_dp = False
    extra_reply_records = (KEY_C_DELTA,)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._c: Arrays | None = None

    def _extra_train_records(self, server_round: int) -> dict[str, ArrayRecord]:
        if self._c is None:
            self._c = {k: np.zeros(v.shape, dtype=np.float64) for k, v in self._global.items()}
        return {KEY_C: to_record({k: v.astype(self._global[k].dtype) for k, v in self._c.items()})}

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        uniform = [1.0] * len(updates)
        mean_dc = weighted_mean([u.extras[KEY_C_DELTA] for u in updates], uniform)
        share = len(updates) / self._population  # |S| / N
        for k, dc in mean_dc.items():
            self._c[k] = self._c[k] + share * dc.astype(np.float64)
        log(INFO, "\tSCAFFOLD: |c| = %.4g", float(np.sqrt(sum(np.sum(c**2) for c in self._c.values()))))
        return weighted_mean([u.delta for u in updates], uniform)


class FedNova(FedAvg):
    """Normalised averaging: remove the bias that unequal numbers of local steps introduce.

    Clients with more data take more local steps, and plain averaging then converges to a
    minimiser of a *reweighted* objective, not of the one the data weights define (the paper's
    "objective inconsistency"). FedNova divides every update by its client's step count
    ``tau_i`` and rescales the average by the effective number of steps::

        delta = tau_eff * sum_i p_i * delta_i / tau_i,    tau_eff = sum_i p_i * tau_i

    with ``p_i`` the data-size weights. For vanilla local SGD (no momentum) the paper's
    normalising factor ``||a_i||_1`` is exactly ``tau_i``. With equal step counts this is FedAvg.
    Clients report ``tau_i`` as ``local-steps``. Not compatible with DP: the per-client rescaling
    depends on the data size, so clipping does not bound its influence.
    """

    name = "fednova"
    supports_dp = False
    required_metrics = ("local-steps",)

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        weights = np.asarray([u.weight for u in updates], dtype=np.float64)
        steps = np.asarray([u.metrics["local-steps"] for u in updates], dtype=np.float64)
        tau_eff = float(weights @ steps / weights.sum())
        rescaled = [{k: v * (tau_eff / tau) for k, v in u.delta.items()} for u, tau in zip(updates, steps)]
        log(INFO, "\tFedNova: local steps %s, effective %.2f", steps.astype(int).tolist(), tau_eff)
        return weighted_mean(rescaled, weights.tolist())


class QFedAvg(FedAvg):
    """q-FedAvg: weight clients by their loss to make the model fairer (Li et al., 2020).

    Minimises ``sum_k F_k^(q+1) / (q+1)`` instead of the average loss, so clients the global model
    serves badly count more. With ``F_k`` the loss of the received model on client ``k``'s data, ``L``
    an estimate of the Lipschitz constant (``1 / client learning rate``) and ``delta_k = w_k - w``::

        delta = L * sum_k F_k^q * delta_k / sum_k h_k,
        h_k   = q * F_k^(q-1) * ||L * delta_k||^2 + L * F_k^q

    ``q = 0`` is the uniform average (FedAvg with equal weights); a larger ``q`` trades mean accuracy for
    a better worst client. The update is already scaled, so ``server_learning_rate`` should stay 1.
    The ``||L * delta_k||^2`` term in ``h_k`` shrinks the step, more the larger ``L * ||delta_k||^2`` is;
    ``qfedavg-step-scale`` in the history is the factor ``sum_k L * F_k^q / sum_k h_k`` (1 = no shrinking,
    which is the case for ``q = 0``). ``lipschitz`` replaces the default ``L = 1 / lr``, the one knob that
    moves the step size (the paper treats ``L`` as a hyperparameter).
    Clients report ``F_k`` as ``initial-loss`` (measured before training, on request). Not compatible
    with DP: the weights depend on private losses.
    """

    name = "qfedavg"
    supports_dp = False
    required_metrics = ("initial-loss",)

    def __init__(self, *, q: float = 0.1, lipschitz: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        if q < 0:
            raise ValueError("q must be non-negative")
        if lipschitz < 0:
            raise ValueError("lipschitz must be non-negative (0 = 1 / client learning rate)")
        self.q = q
        self.lipschitz = lipschitz
        self._client_lr = 0.0

    def _extend_train_config(self, server_round: int, config: ConfigRecord) -> None:
        config["report-initial-loss"] = True
        self._client_lr = float(config["lr"])  # already decayed for this round

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        lipschitz = self.lipschitz or 1.0 / self._client_lr
        losses = np.asarray([u.metrics["initial-loss"] for u in updates], dtype=np.float64) + 1e-10
        numerator = {k: np.zeros_like(v) for k, v in updates[0].delta.items()}
        denominator = 0.0
        for u, loss in zip(updates, losses):
            for k, v in u.delta.items():
                numerator[k] += lipschitz * loss**self.q * v
            sq_norm = sum(float(np.sum(np.square(v))) for v in u.delta.values())
            denominator += self.q * loss ** (self.q - 1) * lipschitz**2 * sq_norm + lipschitz * loss**self.q
        self._round_metrics["qfedavg-step-scale"] = float(np.sum(lipschitz * losses**self.q) / denominator)
        return {k: v / denominator for k, v in numerator.items()}

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> q-FedAvg: q=%s, L=%s", self.q, self.lipschitz or "1/lr")
