import math
from typing import Dict, List, Optional

import torch


class SecureAggregator:
    def __init__(self, n_clients: int, dim: int, seed: int = 0, scale: float = 1e3):
        self.n, self.dim, self.scale = n_clients, dim, scale
        g = torch.Generator().manual_seed(seed)
        self.pair_seed = torch.randint(0, 2 ** 31 - 1, (n_clients, n_clients), generator=g)

    def _prg(self, i: int, j: int) -> torch.Tensor:
        a, b = min(i, j), max(i, j)
        g = torch.Generator().manual_seed(int(self.pair_seed[a, b]))
        return torch.randn(self.dim, generator=g, dtype=torch.float64) * self.scale

    def mask(self, i: int, vec: torch.Tensor, active: List[int]) -> torch.Tensor:
        m = torch.zeros(self.dim, dtype=torch.float64)
        for j in active:
            if j == i:
                continue
            m += self._prg(i, j) if i < j else -self._prg(i, j)
        return vec.double() + m

    def aggregate(self, masked: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(masked).sum(0).float()

    def overhead_bytes(self, n_active: int) -> int:
        return 32 * n_active * (n_active - 1)


class GaussianDP:
    def __init__(self, noise_multiplier: float, clip: float = 1.0, delta: float = 1e-5, seed: int = 0):
        self.sigma, self.clip, self.delta = noise_multiplier, clip, delta
        self.g = torch.Generator().manual_seed(seed)
        self.steps = 0

    def clip_update(self, u: torch.Tensor) -> torch.Tensor:
        n = float(u.norm())
        return u * min(1.0, self.clip / (n + 1e-12))

    def noise(self, dim: int, n_clients: int, max_weight: Optional[float] = None) -> torch.Tensor:
        weight = 1.0 / max(n_clients, 1) if max_weight is None else float(max_weight)
        if not 0 < weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        self.last_sensitivity = 2.0 * self.clip * weight
        self.steps += 1
        return torch.randn(dim, generator=self.g) * (self.sigma * self.last_sensitivity)

    def epsilon(self, steps: Optional[int] = None) -> float:
        T = self.steps if steps is None else steps
        if self.sigma <= 0 or T == 0:
            return float("inf") if T else 0.0
        best = float("inf")
        for a in [1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256]:
            eps = T * a / (2 * self.sigma ** 2) + math.log(1 / self.delta) / (a - 1)
            best = min(best, eps)
        return best

    def describe(self) -> Dict[str, object]:
        return {"noise_multiplier": self.sigma, "clip": self.clip, "delta": self.delta,
                "epsilon": self.epsilon(), "adjacency": "replace_one_client_fixed_public_weights",
                "sensitivity": getattr(self, "last_sensitivity", None),
                "scope": "model_updates_conditional_on_fixed_preprocessing",
                "noise_source": "seeded_simulation_not_production"}
