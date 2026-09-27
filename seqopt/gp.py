"""Gaussian process regression (Kriging), written out so it can be read.

The model is

    y = m + f(x) + e,    f ~ GP(0, s2 * k(x, x')),    e ~ N(0, n2)

with inputs on the unit cube (the caller scales them) and the response
standardized (mean subtracted, divided by its SD) before fitting, so m is the
sample mean and the default bounds below make sense for any response.

Kernels use a separate length scale per input (automatic relevance
determination), r = sqrt(sum_j ((x_j - x'_j) / l_j)^2):

    "rbf":       k = exp(-r^2 / 2)
    "matern32":  k = (1 + sqrt(3) r) exp(-sqrt(3) r)
    "matern52":  k = (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)     (default)

A short length scale means the response changes quickly along that input; a
long one means the input hardly matters.

The hyperparameters (log length scales, log s2, log n2) maximize the log
marginal likelihood

    log p(y) = -1/2 y' K^-1 y - sum(log diag L) - n/2 log(2 pi),   K = L L'

using L-BFGS-B with its analytic gradient, from a fixed start plus a few
seeded random restarts. The noise variance n2 is estimated by default
(a "nugget"); `noise="none"` fits an interpolating model for noiseless data,
and a number fixes the noise SD in response units.

Prediction at new points x*:

    mean = k*' K^-1 y            var = s2 - k*' K^-1 k*   (+ n2 for a new observation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

KERNELS = ("matern52", "matern32", "rbf")
JITTER = 1e-8  # added to the diagonal, relative to s2, to keep K positive definite
LOG_2PI = np.log(2.0 * np.pi)


@dataclass(frozen=True)
class GPParams:
    """Hyperparameters on the standardized scale: length scales on the unit cube, variances in SD units."""

    lengthscales: tuple[float, ...]
    signal_var: float
    noise_var: float

    def to_vector(self) -> np.ndarray:
        return np.log(np.r_[self.lengthscales, self.signal_var, self.noise_var])

    @classmethod
    def from_vector(cls, theta: np.ndarray) -> GPParams:
        values = np.exp(theta)
        return cls(tuple(float(v) for v in values[:-2]), float(values[-2]), float(values[-1]))


def _kernel_parts(kernel: str, X1: np.ndarray, X2: np.ndarray, lengthscales: np.ndarray):
    """k(r) and g(r), where dk/dlog(l_j) = g * ((x_j - x'_j) / l_j)^2, plus the scaled squared differences."""
    scaled = (X1[:, None, :] - X2[None, :, :]) / lengthscales  # (n1, n2, d)
    sq = scaled**2
    r2 = sq.sum(axis=-1)
    if kernel == "rbf":
        k = np.exp(-0.5 * r2)
        g = k
    else:
        r = np.sqrt(r2)
        if kernel == "matern32":
            a = np.sqrt(3.0) * r
            e = np.exp(-a)
            k = (1.0 + a) * e
            g = 3.0 * e
        else:
            a = np.sqrt(5.0) * r
            e = np.exp(-a)
            k = (1.0 + a + a * a / 3.0) * e
            g = (5.0 / 3.0) * (1.0 + a) * e
    return k, g, sq


class GaussianProcess:
    """GP regression on the unit cube. Fit with `fit`, then `predict` mean and SD in response units."""

    def __init__(
        self,
        kernel: str = "matern52",
        noise: str | float = "estimate",
        restarts: int = 4,
        seed: int = 0,
        lengthscale_bounds: tuple[float, float] = (0.01, 10.0),
        signal_bounds: tuple[float, float] = (0.01, 100.0),
        noise_bounds: tuple[float, float] = (1e-6, 1.0),
    ):
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}; choose one of {', '.join(KERNELS)}")
        if not (noise in ("estimate", "none") or (isinstance(noise, (int, float)) and noise >= 0)):
            raise ValueError("noise must be 'estimate', 'none', or a noise SD >= 0 in response units")
        self.kernel = kernel
        self.noise = noise
        self.restarts = int(restarts)
        self.seed = int(seed)
        self.lengthscale_bounds = lengthscale_bounds
        self.signal_bounds = signal_bounds
        self.noise_bounds = noise_bounds
        self.params: GPParams | None = None
        self.log_likelihood: float | None = None

    # --- Likelihood --------------------------------------------------------------

    def _fixed_noise_var(self) -> float | None:
        """Noise variance on the standardized scale when it is not estimated."""
        if self.noise == "estimate":
            return None
        if self.noise == "none":
            return 0.0
        return (float(self.noise) / self._y_sd) ** 2

    def _bounds(self, d: int) -> list[tuple[float, float]]:
        lo, hi = np.log(self.lengthscale_bounds)
        bounds = [(lo, hi)] * d + [tuple(np.log(self.signal_bounds))]
        fixed = self._fixed_noise_var()
        if fixed is None:
            bounds.append(tuple(np.log(self.noise_bounds)))
        else:
            v = np.log(max(fixed, 1e-12))
            bounds.append((v, v))
        return bounds

    def _nll(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """Negative log marginal likelihood and its gradient with respect to theta."""
        n, d = X.shape
        ls = np.exp(theta[:d])
        s2, n2 = np.exp(theta[d]), np.exp(theta[d + 1])
        k, g, sq = _kernel_parts(self.kernel, X, X, ls)
        Kf = s2 * k
        K = Kf + (n2 + JITTER * s2) * np.eye(n)
        try:
            L = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros_like(theta)
        alpha = cho_solve(L, y, check_finite=False)
        nll = 0.5 * y @ alpha + np.log(np.diag(L[0])).sum() + 0.5 * n * LOG_2PI
        W = np.outer(alpha, alpha) - cho_solve(L, np.eye(n), check_finite=False)
        grad = np.empty_like(theta)
        G = s2 * g
        for j in range(d):
            grad[j] = -0.5 * np.sum(W * G * sq[:, :, j])
        grad[d] = -0.5 * np.sum(W * Kf)
        grad[d + 1] = -0.5 * n2 * np.trace(W)
        if self._fixed_noise_var() is not None:
            grad[d + 1] = 0.0
        return float(nll), grad

    # --- Fitting -----------------------------------------------------------------

    def fit(self, X, y, init: GPParams | None = None, restarts: int | None = None) -> GaussianProcess:
        """Fit to points X (n, d) on the unit cube and responses y (n,).

        `init` warm-starts from earlier hyperparameters (the loop passes the
        previous fit's), and `restarts` overrides the number of random restarts.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError(f"{len(X)} points but {len(y)} responses")
        if len(y) == 0 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(X)):
            raise ValueError("fit needs at least one point, and finite values")
        n, d = X.shape
        self._y_mean = float(y.mean())
        sd = float(y.std())
        self._y_sd = sd if sd > 0 else 1.0
        ys = (y - self._y_mean) / self._y_sd
        bounds = self._bounds(d)
        lower, upper = np.array(bounds).T

        start = np.r_[np.full(d, np.log(0.3)), 0.0, np.log(1e-2)]
        if init is not None and len(init.lengthscales) == d:
            start = init.to_vector()
        starts = [np.clip(start, lower, upper)]
        rng = np.random.default_rng(self.seed)
        for _ in range(self.restarts if restarts is None else int(restarts)):
            starts.append(rng.uniform(lower, upper))

        best = None
        for theta0 in starts:
            result = minimize(self._nll, theta0, args=(X, ys), jac=True, method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result
        self._set(X, ys, GPParams.from_vector(best.x))
        self.log_likelihood = -float(best.fun)
        return self

    def fit_fixed(self, X, y, params: GPParams, y_scale: tuple[float, float] | None = None) -> GaussianProcess:
        """Condition on data with given hyperparameters (no optimization).

        `y_scale` = (mean, sd) keeps an earlier standardization, which the
        batch proposals use when they add "fantasy" points.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if y_scale is None:
            sd = float(y.std())
            y_scale = (float(y.mean()), sd if sd > 0 else 1.0)
        self._y_mean, self._y_sd = y_scale
        ys = (y - self._y_mean) / self._y_sd
        self._set(X, ys, params)
        self.log_likelihood = -self._nll(params.to_vector(), X, ys)[0]
        return self

    def _set(self, X: np.ndarray, ys: np.ndarray, params: GPParams) -> None:
        self.params = params
        self._X = X
        self._ys = ys
        ls = np.array(params.lengthscales)
        k, _, _ = _kernel_parts(self.kernel, X, X, ls)
        K = params.signal_var * k + (params.noise_var + JITTER * params.signal_var) * np.eye(len(X))
        self._L = np.linalg.cholesky(K)
        self._alpha = cho_solve((self._L, True), ys, check_finite=False)

    # --- Prediction --------------------------------------------------------------

    @property
    def y_scale(self) -> tuple[float, float]:
        return self._y_mean, self._y_sd

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y_mean + self._y_sd * self._ys

    def predict(self, X, include_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Mean and SD at points X (n, d), in response units.

        The SD is for the underlying function; `include_noise` adds the noise
        variance, giving the spread of a new measurement.
        """
        if self.params is None:
            raise RuntimeError("fit the model first")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        p = self.params
        k, _, _ = _kernel_parts(self.kernel, X, self._X, np.array(p.lengthscales))
        Ks = p.signal_var * k  # (m, n)
        mean = Ks @ self._alpha
        v = solve_triangular(self._L, Ks.T, lower=True, check_finite=False)
        var = np.maximum(p.signal_var - (v * v).sum(axis=0), 0.0)
        if include_noise:
            var = var + p.noise_var
        return self._y_mean + self._y_sd * mean, self._y_sd * np.sqrt(var)

    def describe(self) -> dict:
        """The fitted hyperparameters in readable units, for logs and the browser."""
        p = self.params
        return {
            "kernel": self.kernel,
            "lengthscales": list(p.lengthscales),
            "signal_sd": float(np.sqrt(p.signal_var) * self._y_sd),
            "noise_sd": float(np.sqrt(p.noise_var) * self._y_sd),
            "log_likelihood": self.log_likelihood,
        }
