"""Blind scenarios: move the optimum so it can't be looked up.

Hartmann 6D is published together with its optimum, so a student who
recognizes it could read x* straight off the SFU page. A scenario, set by a
seed, reorders the six coordinates and reverses some of them before the
function sees them:

    z[perm[j]] = u[j]            (or 1 - u[j] when flip[j])

Here u is the point as the student gives it (unit cube, in the order the
student sees the factors), and z is what the Hartmann function receives. This
maps the unit cube onto itself one to one, so every value is still there,
including the minimum and the flat second basin; they are just somewhere else.

It is a disguise, not security. The browser app and the Python package both
contain this code, and the 720 x 64 possible scenarios can be tried by brute
force. Blind mode works on the honor system, with an instructor toggle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

DIM = 6
_STREAM = 0x5CE  # keeps scenario draws apart from noise draws, which use [seed, k]


@dataclass(frozen=True)
class Scenario:
    perm: tuple[int, ...] = tuple(range(DIM))  # perm[j]: the Hartmann coordinate factor j drives
    flip: tuple[bool, ...] = (False,) * DIM  # flip[j]: factor j runs reversed
    seed: int | None = None  # the seed it came from, if any (for display and replay)

    def __post_init__(self):
        if sorted(self.perm) != list(range(DIM)):
            raise ValueError(f"perm must be a permutation of 0-{DIM - 1}, got {self.perm!r}")
        if len(self.flip) != DIM:
            raise ValueError(f"flip needs {DIM} entries, got {len(self.flip)}")
        object.__setattr__(self, "perm", tuple(int(p) for p in self.perm))
        object.__setattr__(self, "flip", tuple(bool(f) for f in self.flip))

    @classmethod
    def from_seed(cls, seed: int) -> Scenario:
        """A disguise drawn from a seed; never the identity, so the optimum always moves."""
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError(f"scenario seed must be a non-negative whole number, got {seed!r}")
        rng = np.random.default_rng([int(seed), _STREAM, _STREAM])
        while True:
            perm = tuple(int(p) for p in rng.permutation(DIM))
            flip = tuple(bool(f) for f in rng.random(DIM) < 0.5)
            scenario = cls(perm, flip, int(seed))
            if not scenario.is_identity:
                return scenario

    @property
    def is_identity(self) -> bool:
        return self.perm == tuple(range(DIM)) and not any(self.flip)

    def to_hartmann(self, u) -> np.ndarray:
        """Hartmann coordinates z for student coordinates u (unit cube); keeps the input's shape."""
        U = np.asarray(u, dtype=float)
        V = np.atleast_2d(U)
        Z = np.empty_like(V)
        Z[:, list(self.perm)] = np.where(self.flip, 1.0 - V, V)
        return Z.reshape(U.shape)

    def from_hartmann(self, z) -> np.ndarray:
        """Student coordinates u for Hartmann coordinates z; the inverse of to_hartmann."""
        Z = np.asarray(z, dtype=float)
        V = np.atleast_2d(Z)[:, list(self.perm)]
        return np.where(self.flip, 1.0 - V, V).reshape(Z.shape)

    def to_dict(self) -> dict:
        return {"perm": list(self.perm), "flip": list(self.flip), "seed": self.seed}

    @classmethod
    def from_dict(cls, values: Mapping | None) -> Scenario:
        if values is None:
            return IDENTITY
        return cls(tuple(values["perm"]), tuple(values["flip"]), values.get("seed"))


IDENTITY = Scenario()
