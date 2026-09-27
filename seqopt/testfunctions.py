"""Other test functions as oracles, so the kernel is exercised on more than Hartmann.

`FunctionOracle` turns any vectorized function on a box into a
`seqopt.oracle.Oracle` with the same conventions as the Hartmann oracle: a
budget, seeded additive Gaussian noise whose draw for evaluation k comes from
`default_rng([seed, k])`, and a noiseless `true_response` for scoring.

- Branin-Hoo (2D): three global minima of 0.397887, on [-5, 10] x [0, 15].
- Rosenbrock (any d >= 2): a curved valley, minimum 0 at (1, ..., 1), on [-2.048, 2.048]^d.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .oracle import Budget


class FunctionOracle:
    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        bounds,
        name: str = "function",
        optimum: float | None = None,
        minimizers=None,
        budget: int | None = None,
        noise_sd: float = 0.0,
        seed: int = 0,
    ):
        self._fn = fn
        self._bounds = np.asarray(bounds, dtype=float)
        if self._bounds.ndim != 2 or self._bounds.shape[1] != 2 or np.any(self._bounds[:, 0] >= self._bounds[:, 1]):
            raise ValueError("bounds must be a (d, 2) array of [low, high] with low < high")
        if noise_sd < 0:
            raise ValueError("noise_sd must be >= 0")
        self.dim = len(self._bounds)
        self.name = name
        self.optimum = optimum
        self.minimizers = None if minimizers is None else np.atleast_2d(minimizers)
        self.budget = Budget(budget)
        self.noise_sd = float(noise_sd)
        self.seed = int(seed)

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds.copy()

    def _points(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f"expected points with {self.dim} coordinates, got shape {X.shape}")
        low, high = self._bounds[:, 0], self._bounds[:, 1]
        slack = 1e-9 * (high - low)
        if np.any(X < low - slack) or np.any(X > high + slack):
            raise ValueError(f"points must lie inside the bounds {self._bounds.tolist()}")
        return np.clip(X, low, high)

    def true_response(self, X):
        single = np.ndim(X) == 1
        y = np.asarray(self._fn(self._points(X)), dtype=float)
        return float(y[0]) if single else y

    def evaluate(self, X):
        single = np.ndim(X) == 1
        P = self._points(X)
        first = self.budget.charge(len(P))
        y = np.asarray(self._fn(P), dtype=float)
        if self.noise_sd > 0:
            y = y + self.noise_sd * np.array(
                [np.random.default_rng([self.seed, first + i]).standard_normal() for i in range(len(P))]
            )
        return float(y[0]) if single else y


def _branin(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    return (x2 - b * x1**2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10


def _rosenbrock(X: np.ndarray) -> np.ndarray:
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def branin(budget: int | None = None, noise_sd: float = 0.0, seed: int = 0) -> FunctionOracle:
    return FunctionOracle(
        _branin, [[-5.0, 10.0], [0.0, 15.0]], "Branin-Hoo", optimum=0.397887,
        minimizers=[[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]],
        budget=budget, noise_sd=noise_sd, seed=seed,
    )


def rosenbrock(d: int = 2, budget: int | None = None, noise_sd: float = 0.0, seed: int = 0) -> FunctionOracle:
    if d < 2:
        raise ValueError("Rosenbrock needs at least 2 dimensions")
    return FunctionOracle(
        _rosenbrock, [[-2.048, 2.048]] * d, f"Rosenbrock {d}D", optimum=0.0, minimizers=[[1.0] * d],
        budget=budget, noise_sd=noise_sd, seed=seed,
    )
