"""Acquisition functions, and how the next point or batch is chosen.

The response is minimized; every acquisition value is to be maximized. From a
surrogate's mean mu(x) and SD s(x), and the incumbent f* (the best value so
far), with z = (f* - xi - mu) / s:

- "ei"       expected improvement      (f* - xi - mu) Phi(z) + s phi(z)
- "pi"       probability of improvement Phi(z)
- "ucb"      confidence bound           -(mu - kappa s)   (a lower bound, since we minimize)
- "exploit"  pure exploitation          -mu               (go where the model predicts the minimum)
- "explore"  pure exploration           s                 (go where the model knows least)

The incumbent. With noisy measurements the smallest observed value is biased
low (it is partly a lucky draw), so by default f* is the lowest predicted mean
at the points already measured (the "plug-in" choice in Picheny et al., 2013).
`incumbent="observed"` uses the smallest measurement instead.

Maximizing an acquisition: evaluate it on a seeded cloud of candidates
(uniform over the unit cube, plus points scattered around the best
measurements), then polish the best few with L-BFGS-B inside the cube. Points
closer than `min_distance` to a measured point are not allowed.

Batches of q points are picked one at a time. After each pick the model is
conditioned on a pretend result there, with its hyperparameters unchanged, so
the next pick goes somewhere else:
- "believer" (kriging believer): pretend the result is the model's mean there.
- "liar" (constant liar): pretend it is the incumbent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from .gp import GaussianProcess
from .rsm import QuadraticRSM

ACQUISITIONS = ("ei", "pi", "ucb", "exploit", "explore")
BATCH_RULES = ("believer", "liar")
LOCAL_SHARE = 0.25  # share of candidates scattered around the best measured points
LOCAL_SD = 0.05  # their spread, on the unit cube

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def acquisition_values(name: str, mean, sd, best: float, xi: float = 0.0, kappa: float = 2.0) -> np.ndarray:
    """Acquisition values (to maximize) from predicted means and SDs."""
    mean, sd = np.asarray(mean, dtype=float), np.asarray(sd, dtype=float)
    if name == "exploit":
        return -mean
    if name == "explore":
        return sd.copy()
    if name == "ucb":
        return -(mean - kappa * sd)
    if name not in ("ei", "pi"):
        raise ValueError(f"unknown acquisition {name!r}; choose one of {', '.join(ACQUISITIONS)}")
    gain = best - xi - mean
    safe = np.where(sd > 1e-12, sd, 1.0)
    z = gain / safe
    if name == "pi":
        return np.where(sd > 1e-12, ndtr(z), (gain > 0).astype(float))
    ei = gain * ndtr(z) + sd * _INV_SQRT_2PI * np.exp(-0.5 * z * z)
    return np.where(sd > 1e-12, np.maximum(ei, 0.0), np.maximum(gain, 0.0))


def incumbent(surrogate, how: str = "mean") -> float:
    """The best value so far: the lowest predicted mean at measured points, or the lowest measurement."""
    if how == "observed":
        return float(np.min(surrogate.y))
    if how != "mean":
        raise ValueError("incumbent must be 'mean' or 'observed'")
    mean, _ = surrogate.predict(surrogate.X)
    return float(np.min(mean))


def _too_close(X: np.ndarray, X_obs: np.ndarray | None, min_distance: float) -> np.ndarray:
    if X_obs is None or len(X_obs) == 0 or min_distance <= 0:
        return np.zeros(len(X), dtype=bool)
    d2 = ((X[:, None, :] - X_obs[None, :, :]) ** 2).sum(axis=-1)
    return d2.min(axis=1) < min_distance**2


def maximize(
    fn: Callable[[np.ndarray], np.ndarray],
    d: int,
    rng: np.random.Generator,
    centers: np.ndarray | None = None,
    X_obs: np.ndarray | None = None,
    min_distance: float = 0.0,
    n_candidates: int = 2000,
    n_polish: int = 4,
) -> tuple[np.ndarray, float]:
    """The point of the unit cube where fn (vectorized over rows) is largest, and its value."""
    n_local = int(LOCAL_SHARE * n_candidates) if centers is not None and len(centers) else 0
    cand = rng.random((n_candidates - n_local, d))
    if n_local:
        picks = centers[rng.integers(len(centers), size=n_local)]
        cand = np.vstack([cand, np.clip(picks + rng.normal(0.0, LOCAL_SD, (n_local, d)), 0.0, 1.0)])
    values = np.asarray(fn(cand), dtype=float)
    values = np.where(_too_close(cand, X_obs, min_distance) | ~np.isfinite(values), -np.inf, values)
    order = np.argsort(-values, kind="stable")[:n_polish]
    best_x, best_v = cand[order[0]].copy(), float(values[order[0]])
    if not np.isfinite(best_v):
        raise ValueError("no candidate point is far enough from the measured points; lower min_distance")

    def negative(x):
        return -float(fn(x[None, :])[0])

    for i in order:
        if not np.isfinite(values[i]):
            continue
        result = minimize(negative, cand[i], method="L-BFGS-B", bounds=[(0.0, 1.0)] * d,
                          options={"maxiter": 50})
        x = np.clip(result.x, 0.0, 1.0)
        v = -negative(x)
        if v > best_v and not _too_close(x[None, :], X_obs, min_distance)[0]:
            best_x, best_v = x, v
    return best_x, best_v


def _condition(surrogate, X: np.ndarray, y: np.ndarray):
    """The same kind of model refitted to (X, y); a GP keeps its hyperparameters and scaling."""
    if isinstance(surrogate, GaussianProcess):
        model = GaussianProcess(kernel=surrogate.kernel, noise=surrogate.noise)
        return model.fit_fixed(X, y, surrogate.params, y_scale=surrogate.y_scale)
    if isinstance(surrogate, QuadraticRSM):
        return QuadraticRSM().fit(X, y)
    raise TypeError(f"cannot condition a {type(surrogate).__name__}")


def propose(
    surrogate,
    acquisition: str = "ei",
    q: int = 1,
    rng: np.random.Generator | None = None,
    xi: float = 0.0,
    kappa: float = 2.0,
    batch: str = "believer",
    incumbent_rule: str = "mean",
    min_distance: float = 0.0,
    n_candidates: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """The next q points (unit cube, shape (q, d)) and their acquisition values when picked."""
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {acquisition!r}; choose one of {', '.join(ACQUISITIONS)}")
    if batch not in BATCH_RULES:
        raise ValueError(f"unknown batch rule {batch!r}; choose one of {', '.join(BATCH_RULES)}")
    if int(q) < 1:
        raise ValueError("q must be at least 1")
    rng = np.random.default_rng() if rng is None else rng
    best = incumbent(surrogate, incumbent_rule)
    model = surrogate
    X_all, y_all = surrogate.X.copy(), surrogate.y.copy()
    d = X_all.shape[1]
    centers = X_all[np.argsort(y_all)[: max(1, min(5, len(y_all)))]]
    picks, values = [], []
    for j in range(int(q)):
        def fn(Xc, model=model):
            mean, sd = model.predict(Xc)
            return acquisition_values(acquisition, mean, sd, best, xi, kappa)

        x, v = maximize(fn, d, rng, centers=centers, X_obs=X_all, min_distance=min_distance,
                        n_candidates=n_candidates)
        picks.append(x)
        values.append(v)
        if j < q - 1:
            fantasy = float(model.predict(x[None, :])[0][0]) if batch == "believer" else best
            X_all = np.vstack([X_all, x])
            y_all = np.append(y_all, fantasy)
            model = _condition(surrogate, X_all, y_all)
            # A pretend result counts as data: if it beats the incumbent, it becomes the incumbent,
            # otherwise the same spot would still promise a sure improvement and be picked again.
            best = min(best, fantasy)
    return np.array(picks), np.array(values)
