"""The Hartmann 6D test function: the deterministic core of the oracle.

    f(x) = -sum_{i=1..4} alpha_i * exp( -sum_{j=1..6} A_ij * (x_j - P_ij)^2 ),   x in [0, 1]^6

Reference: https://www.sfu.ca/~ssurjano/hart6.html. The global minimum is
f(x*) = -3.32237 at x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573).

Three forms share the same inner sum S(x) = sum_i alpha_i exp(...), so they
share their minimizers:

- "standard":      f = -S                          min -3.32237
- "rescaled":      f = -(2.58 + ln S) / 1.94       min -1.94880
- "sfu_rescaled":  f = -(2.58 + S) / 1.94          min -3.04246

"rescaled" is the form from Picheny et al. (2013), which the SFU page describes
as having mean 0 and variance 1 over the unit cube. The formula printed on the
SFU page has no logarithm, and as printed its mean is about -1.46 and its
variance about 0.04. Only the log version has mean 0 and variance 1 (checked by
Monte Carlo in the tests), so "rescaled" is the log version and the printed
formula is kept, under its own name, for anyone reproducing the page exactly.

Local minima: the SFU page says there are 6. A multistart search (40,000+
bounded L-BFGS-B starts, starts from each well center, and unconstrained
Newton) finds only two in [0, 1]^6, listed in LOCAL_MINIMA. The second one is
very flat in two directions, so it is an easy place for an optimizer to stop.

Everything here is pure NumPy and vectorized. A single point (shape (6,))
returns a float; a batch (shape (n, 6)) returns an array of n values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 6

ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
A = np.array([
    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
])
P = 1e-4 * np.array([
    [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
])
for _constant in (ALPHA, A, P):
    _constant.setflags(write=False)

FORMS = ("standard", "rescaled", "sfu_rescaled")

# Published optimum (SFU), kept exactly as printed.
X_STAR = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
F_STAR = -3.32237
X_STAR.setflags(write=False)

RESCALE_SHIFT = 2.58
RESCALE_SCALE = 1.94


@dataclass(frozen=True)
class LocalMinimum:
    x: tuple[float, ...]  # location in [0, 1]^6, Newton-polished to 1e-8
    f: float  # standard-form value
    label: str


LOCAL_MINIMA: tuple[LocalMinimum, ...] = (
    LocalMinimum((0.20168951, 0.15001069, 0.47687397, 0.27533243, 0.31165162, 0.65730053), -3.3223680114,
                 "global minimum"),
    LocalMinimum((0.40465313, 0.88244492, 0.84610157, 0.57398969, 0.1389266, 0.03849589), -3.2031619184,
                 "second minimum (flat basin)"),
)


def _as_points(x) -> tuple[np.ndarray, bool]:
    """(n, 6) float array, plus whether the input was a single point."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == DIM:
        return arr[None, :], True
    if arr.ndim == 2 and arr.shape[1] == DIM:
        return arr, False
    raise ValueError(f"expected a point of length {DIM} or an (n, {DIM}) array, got shape {arr.shape}")


def _check_form(form: str) -> None:
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; choose one of {', '.join(FORMS)}")


def _wells(X: np.ndarray) -> np.ndarray:
    """Each well's term alpha_i * exp(-sum_j A_ij (x_j - P_ij)^2), shape (n, 4)."""
    d = X[:, None, :] - P[None, :, :]
    return ALPHA * np.exp(-np.einsum("ij,nij->ni", A, d * d))


def _from_sum(s: np.ndarray, form: str) -> np.ndarray:
    if form == "standard":
        return -s
    if form == "rescaled":
        return -(RESCALE_SHIFT + np.log(s)) / RESCALE_SCALE
    return -(RESCALE_SHIFT + s) / RESCALE_SCALE


def hartmann6(x, form: str = "standard"):
    """Hartmann 6D value(s). A point gives a float; an (n, 6) array gives n values."""
    _check_form(form)
    X, single = _as_points(x)
    y = _from_sum(_wells(X).sum(axis=1), form)
    return float(y[0]) if single else y


def gradient(x, form: str = "standard") -> np.ndarray:
    """Analytic gradient with respect to x; shape (6,) for a point, (n, 6) for a batch."""
    _check_form(form)
    X, single = _as_points(x)
    terms = _wells(X)  # (n, 4)
    d = X[:, None, :] - P[None, :, :]  # (n, 4, 6)
    grad_standard = np.einsum("ni,ij,nij->nj", terms, 2.0 * A, d)  # gradient of -S
    if form == "rescaled":
        grad = grad_standard / (RESCALE_SCALE * terms.sum(axis=1, keepdims=True))
    elif form == "sfu_rescaled":
        grad = grad_standard / RESCALE_SCALE
    else:
        grad = grad_standard
    return grad[0] if single else grad


def minimum_value(form: str = "standard") -> float:
    """The global minimum value of a form (at the same x* for every form)."""
    _check_form(form)
    return hartmann6(LOCAL_MINIMA[0].x, form)
