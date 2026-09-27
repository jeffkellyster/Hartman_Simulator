"""Measurement noise: an optional, seeded, additive term on the reported response.

The function itself is deterministic; noise only ever enters here, added to
the response the student sees, in that response's units (percent
non-uniformity in process mode, function units otherwise).

- Homoscedastic: every measurement gets N(0, sd^2).
- Heteroscedastic: the SD grows along one factor,
      sd(x) = sd * (1 + hetero * u_k)
  where u_k is that factor's position in its range (0 at the low end, 1 at the
  high end). The default factor is RF power, so high-power runs are noisier, a
  pattern students can find by replicating.

The draw for evaluation number k comes from its own generator,
`np.random.default_rng([seed, k])`. So a run reproduces exactly from its seed,
a value does not depend on how evaluations were batched, and measuring the
same point twice gives two different readings, as a real replicate would.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

DIM = 6
DEFAULT_HETERO_FACTOR = 4  # RF power in the process mapping


@dataclass(frozen=True)
class NoiseModel:
    sd: float = 0.0  # base standard deviation, in the reported response's units
    hetero: float = 0.0  # 0 = homoscedastic; 1 doubles the SD at the factor's high end
    hetero_factor: int = DEFAULT_HETERO_FACTOR  # which factor (0-5, in the order the student sees)

    def __post_init__(self):
        if not np.isfinite(self.sd) or self.sd < 0:
            raise ValueError(f"noise sd must be finite and >= 0, got {self.sd!r}")
        if not np.isfinite(self.hetero) or self.hetero < 0:
            raise ValueError(f"hetero must be finite and >= 0, got {self.hetero!r}")
        if isinstance(self.hetero_factor, bool) or self.hetero_factor not in range(DIM):
            raise ValueError(f"hetero_factor must be a factor index 0-{DIM - 1}, got {self.hetero_factor!r}")

    @property
    def active(self) -> bool:
        return self.sd > 0

    def sd_at(self, U) -> np.ndarray:
        """Noise SD at each point of an (n, 6) array in unit-cube coordinates."""
        U = np.atleast_2d(np.asarray(U, dtype=float))
        return self.sd * (1.0 + self.hetero * U[:, self.hetero_factor])

    def draw(self, seed: int, first: int, U) -> np.ndarray:
        """Noise for evaluations first, first+1, ... at the points U (unit-cube coordinates)."""
        sd = self.sd_at(U)
        if not self.active:
            return np.zeros(len(sd))
        z = np.array([np.random.default_rng([seed, first + i]).standard_normal() for i in range(len(sd))])
        return z * sd

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping | None) -> NoiseModel:
        if values is None:
            return cls()
        unknown = set(values) - {"sd", "hetero", "hetero_factor"}
        if unknown:
            raise ValueError(f"unknown noise settings: {sorted(unknown)}")
        return cls(
            sd=float(values.get("sd", 0.0)),
            hetero=float(values.get("hetero", 0.0)),
            hetero_factor=int(values.get("hetero_factor", DEFAULT_HETERO_FACTOR)),
        )
