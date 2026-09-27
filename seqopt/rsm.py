"""Quadratic response surface (RSM) by least squares, for comparison with the GP.

The full second-order model in d inputs has 1 + d + d(d-1)/2 + d terms
(28 in 6D): intercept, main effects, two-factor interactions and pure
quadratics. It is fitted in coded units (-1 to +1) so the coefficients read
like JMP's. With fewer runs than terms the least-squares solution is not
unique; the minimum-norm one is used and `underdetermined` is set, so the
caller can say so.

`predict` has the same shape as the GP's: mean and SD. Here the SD is the
standard error of the fitted mean, s * sqrt(x' (X'X)^-1 x), with s the
residual SD; it is NaN when there are no residual degrees of freedom.
"""

from __future__ import annotations

import itertools

import numpy as np


def term_names(factors: list[str]) -> list[str]:
    d = len(factors)
    names = ["Intercept"] + list(factors)
    names += [f"{factors[i]}*{factors[j]}" for i, j in itertools.combinations(range(d), 2)]
    names += [f"{f}^2" for f in factors]
    return names


def model_matrix(coded: np.ndarray) -> np.ndarray:
    coded = np.atleast_2d(coded)
    n, d = coded.shape
    cols = [np.ones(n)] + [coded[:, i] for i in range(d)]
    cols += [coded[:, i] * coded[:, j] for i, j in itertools.combinations(range(d), 2)]
    cols += [coded[:, i] ** 2 for i in range(d)]
    return np.column_stack(cols)


def n_terms(d: int) -> int:
    return 1 + d + d * (d - 1) // 2 + d


class QuadraticRSM:
    """Full quadratic model on the unit cube. Fit with `fit`, then `predict` mean and SD."""

    def __init__(self):
        self.coef: np.ndarray | None = None

    def fit(self, X, y) -> QuadraticRSM:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y) or len(y) == 0:
            raise ValueError(f"{len(X)} points but {len(y)} responses")
        self._X, self._y = X, y
        M = model_matrix(2.0 * X - 1.0)
        n, p = M.shape
        self.coef, _, rank, _ = np.linalg.lstsq(M, y, rcond=None)
        self.rank = int(rank)
        self.underdetermined = rank < p
        resid = y - M @ self.coef
        dof = n - rank
        self.residual_sd = float(np.sqrt(resid @ resid / dof)) if dof > 0 else float("nan")
        ss_tot = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
        self._pinv = np.linalg.pinv(M.T @ M)
        return self

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        if self.coef is None:
            raise RuntimeError("fit the model first")
        M = model_matrix(2.0 * np.atleast_2d(np.asarray(X, dtype=float)) - 1.0)
        mean = M @ self.coef
        leverage = np.einsum("ij,jk,ik->i", M, self._pinv, M)
        sd = self.residual_sd * np.sqrt(np.maximum(leverage, 0.0))
        return mean, sd

    def describe(self, factors: list[str] | None = None) -> dict:
        d = self._X.shape[1]
        names = term_names(factors or [f"x{i + 1}" for i in range(d)])
        return {
            "model": "quadratic",
            "terms": len(names),
            "runs": len(self._y),
            "underdetermined": bool(self.underdetermined),
            "r2": self.r2,
            "residual_sd": self.residual_sd,
            "coefficients": {name: float(c) for name, c in zip(names, self.coef)},
        }
