"""JSON API for front ends: the browser app (through Pyodide) and, later, JMP.

One entry point, `handle(request_json) -> response_json`, so a caller in any
language needs nothing but JSON:

    handle('{"fn": "catalog", "args": {}}')  ->  '{"ok": true, "result": {...}}'

Errors come back as {"ok": false, "error": "..."} instead of raising. Every
decision lives here or deeper in the engine; the front end only displays.

The calls are stateless: the caller sends the oracle's config and the runs
measured so far (points in the oracle's units, and their responses) with each
request. The one piece of memory is a cache of the last fitted model, so
moving a slider does not refit it.

Designing and measuring need only NumPy. Modeling calls (fit, slice, profile,
suggest, benchmark) import SciPy when first used, which lets the browser start
before SciPy has finished loading.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from seqopt import designs

from . import csvio, process
from .function import FORMS
from .noise import NoiseModel
from .oracle import HartmannOracle

FUNCTIONS: dict[str, Callable] = {}
GRID_MAX = 81
MODELS = ("gp", "rsm")

_ORDER = [
    {"name": "randomize", "label": "Randomize run order", "kind": "bool", "default": False},
    {"name": "seed", "label": "Seed", "kind": "int", "default": 1, "min": 0, "max": 1_000_000},
]
_RUNS = {"name": "runs", "label": "Runs", "kind": "int", "default": 20, "min": 2, "max": 500}
_SEED = {"name": "seed", "label": "Seed", "kind": "int", "default": 1, "min": 0, "max": 1_000_000}
_CENTERS = {"name": "center_points", "label": "Center points", "kind": "int", "default": 0, "min": 0, "max": 20}

DESIGN_TYPES = [
    {"type": "maximin_lhs", "label": "Maximin Latin hypercube (space filling)", "params": [_RUNS, _SEED]},
    {"type": "latin_hypercube", "label": "Latin hypercube", "params": [_RUNS, _SEED]},
    {"type": "random", "label": "Random points", "params": [_RUNS, _SEED]},
    {"type": "box_behnken", "label": "Box-Behnken (response surface)",
     "params": [{**_CENTERS, "default": 6}, *_ORDER]},
    {"type": "central_composite", "label": "Central composite (response surface)",
     "params": [{"name": "alpha", "label": "Axial points", "kind": "choice", "choices": ["face", "rotatable"],
                 "default": "face"},
                {"name": "inscribed", "label": "Keep axial points inside the ranges", "kind": "bool",
                 "default": True},
                {**_CENTERS, "default": 4}, *_ORDER]},
    {"type": "fractional_factorial", "label": "Fractional factorial (2-level)",
     "params": [{"name": "runs", "label": "Runs", "kind": "choice", "choices": [8, 16, 32], "default": 16},
                _CENTERS, *_ORDER]},
    {"type": "full_factorial", "label": "Full factorial",
     "params": [{"name": "levels", "label": "Levels", "kind": "choice", "choices": [2, 3], "default": 2},
                _CENTERS, *_ORDER]},
    {"type": "plackett_burman", "label": "Plackett-Burman (screening)",
     "params": [{"name": "runs", "label": "Runs", "kind": "choice", "choices": [12, 20, 24], "default": 12},
                _CENTERS, *_ORDER]},
]

STRATEGY_TYPES = [
    {"method": "bo", "label": "Bayesian optimization (Gaussian process)"},
    {"method": "rsm", "label": "Quadratic RSM: go to the predicted minimum"},
    {"method": "random", "label": "Random point(s)"},
]
ACQUISITION_TYPES = [
    {"key": "ei", "label": "Expected improvement"},
    {"key": "pi", "label": "Probability of improvement"},
    {"key": "ucb", "label": "Confidence bound (UCB)"},
    {"key": "exploit", "label": "Exploit: lowest predicted mean"},
    {"key": "explore", "label": "Explore: highest uncertainty"},
]
KERNEL_TYPES = [
    {"key": "matern52", "label": "Matern 5/2"},
    {"key": "matern32", "label": "Matern 3/2"},
    {"key": "rbf", "label": "RBF (squared exponential)"},
]
BENCHMARK_STRATEGIES = [
    {"key": "random", "strategy": {"method": "random"}},
    {"key": "rsm", "strategy": {"method": "rsm"}},
    {"key": "bo-ei", "strategy": {"method": "bo", "acquisition": "ei"}},
    {"key": "bo-ucb", "strategy": {"method": "bo", "acquisition": "ucb"}},
    {"key": "bo-pi", "strategy": {"method": "bo", "acquisition": "pi"}},
    {"key": "exploit", "strategy": {"method": "bo", "acquisition": "exploit"}},
]
DEFAULTS = {"units": "process", "noise_sd": 0.05, "hetero": 0.0, "budget": 60, "seed": 1, "scenario": None}

_CACHE: dict = {"key": None, "model": None}


def _api(fn: Callable) -> Callable:
    FUNCTIONS[fn.__name__] = fn
    return fn


def handle(request_json: str) -> str:
    try:
        request = json.loads(request_json)
        fn = FUNCTIONS.get(request.get("fn"))
        if fn is None:
            raise ValueError(f"unknown function {request.get('fn')!r}")
        result = fn(**(request.get("args") or {}))
        return json.dumps({"ok": True, "result": _jsonable(result)}, allow_nan=False)
    except Exception as exc:  # report every failure to the caller rather than crash the page
        message = str(exc) if isinstance(exc, (ValueError, TypeError)) else f"{type(exc).__name__}: {exc}"
        return json.dumps({"ok": False, "error": message})


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# --- Helpers ---------------------------------------------------------------------------------


def _oracle(config: Mapping | None, budget: bool = True) -> HartmannOracle:
    config = dict(config or {})
    if not budget:
        config["budget"] = None
    return HartmannOracle.from_config(config)


def _data(oracle: HartmannOracle, X: Sequence | None, y: Sequence | None) -> tuple[np.ndarray, np.ndarray]:
    """Measured points on the unit cube and their responses."""
    X = np.asarray(X if X is not None else [], dtype=float).reshape(-1, oracle.dim)
    y = np.asarray(y if y is not None else [], dtype=float).ravel()
    if len(X) != len(y):
        raise ValueError(f"{len(X)} points but {len(y)} responses")
    if not np.all(np.isfinite(y)):
        raise ValueError("every measured run needs a numeric response")
    U = np.atleast_2d(process.to_unit(X, oracle.units)) if len(X) else np.empty((0, oracle.dim))
    return U, y


def _response(oracle: HartmannOracle) -> dict:
    r = oracle.response
    return {"name": r.name, "label": r.label, "unit": r.unit, "goal": r.goal, "header": csvio.response_header(oracle)}


def _fitted(oracle: HartmannOracle, U: np.ndarray, y: np.ndarray, model: str, kernel: str):
    """The model fitted to these runs, from the cache when nothing changed."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")
    need = 2
    if len(y) < need:
        raise ValueError(f"measure at least {need} runs before fitting a model")
    key = (model, kernel, U.tobytes(), y.tobytes())
    if _CACHE["key"] != key:
        if model == "gp":
            from seqopt.gp import GaussianProcess

            fitted = GaussianProcess(kernel=kernel, seed=0).fit(U, y)
        else:
            from seqopt.rsm import QuadraticRSM

            fitted = QuadraticRSM().fit(U, y)
        _CACHE.update(key=key, model=fitted)
    return _CACHE["model"]


def _describe(model, oracle: HartmannOracle) -> dict:
    names = [f["label"] for f in oracle.factors()]
    if hasattr(model, "kernel"):
        info = model.describe()
        info["lengthscales"] = dict(zip(names, info["lengthscales"]))
        return {"type": "gp", **info}
    info = model.describe([f["name"] for f in oracle.factors()])
    return {"type": "rsm", **info}


def _center_unit(oracle: HartmannOracle, center: Sequence | None) -> np.ndarray:
    if center is None:
        return np.full(oracle.dim, 0.5)
    return np.asarray(process.to_unit(np.asarray(center, dtype=float), oracle.units), dtype=float)


def _grid_size(n: int) -> int:
    n = int(n)
    if not 3 <= n <= GRID_MAX:
        raise ValueError(f"grid size must be 3-{GRID_MAX}")
    return n


def _layer(model, U: np.ndarray, layer: str, y: np.ndarray, acquisition: str = "ei") -> np.ndarray:
    mean, sd = model.predict(U)
    if layer == "mean":
        return mean
    if layer == "sd":
        return sd
    if layer == "acquisition":
        from seqopt.acquisition import acquisition_values, incumbent

        return acquisition_values(acquisition, mean, sd, incumbent(model, "mean"))
    raise ValueError("layer must be mean, sd or acquisition")


# --- Setup ------------------------------------------------------------------------------------


@_api
def catalog() -> dict:
    """Everything a front end needs to build its controls."""
    return {
        "process": process.PROCESS_NAME,
        "factors": process.table(),
        "units": list(process.UNITS),
        "forms": list(FORMS),
        "designs": DESIGN_TYPES,
        "strategies": STRATEGY_TYPES,
        "acquisitions": ACQUISITION_TYPES,
        "kernels": KERNEL_TYPES,
        "benchmark_strategies": BENCHMARK_STRATEGIES,
        "defaults": DEFAULTS,
    }


@_api
def lab(units: str = "process", noise_sd: float = 0.0, hetero: float = 0.0, budget: int | None = None,
        seed: int | None = None, scenario: int | None = None, form: str = "standard") -> dict:
    """An oracle config from the lab settings, with the seed filled in, plus what the student sees."""
    oracle = HartmannOracle(units=units, form=form if units != "process" else "standard",
                            noise=NoiseModel(sd=float(noise_sd), hetero=float(hetero)),
                            seed=seed, budget=budget, scenario=scenario)
    return oracle_info(oracle.config)


@_api
def oracle_info(config: Mapping) -> dict:
    oracle = _oracle(config)
    return {"config": oracle.config, "factors": oracle.factors(), "response": _response(oracle),
            "bounds": oracle.bounds}


# --- Designs and measuring --------------------------------------------------------------------


@_api
def design(config: Mapping, kind: str, params: Mapping | None = None) -> dict:
    """A design's runs in the oracle's units, with what an analyst should know about it."""
    oracle = _oracle(config)
    d = designs.make(kind, [f["name"] for f in oracle.factors()], **dict(params or {}))
    if np.any(d.unit < -1e-12) or np.any(d.unit > 1 + 1e-12):
        raise ValueError("Some runs fall outside the factor ranges (axial points beyond the limits). "
                         "Keep axial points inside the ranges, or use face-centered axial points.")
    points = process.from_unit(np.clip(d.unit, 0.0, 1.0), oracle.units)
    info = {k: d.info[k] for k in ("type", "runs", "resolution", "generators", "min_distance", "alpha",
                                   "inscribed", "cube_runs", "center_points", "note") if k in d.info}
    return {"points": points, "info": info}


@_api
def measure(config: Mapping, X: Sequence, y: Sequence, new: Sequence) -> dict:
    """Measure new points after the runs already measured (which count against the budget)."""
    oracle = _oracle(config)
    if len(X):
        oracle.record(X, y)
    new = np.asarray(new, dtype=float).reshape(-1, oracle.dim)
    if not len(new):
        raise ValueError("nothing to measure")
    values = np.atleast_1d(oracle.evaluate(new))
    b = oracle.budget
    return {
        "y": values,
        "y_true": [e.y_true for e in oracle.ledger[-len(new):]],
        "budget": {"used": b.used, "total": b.total, "remaining": b.remaining},
    }


@_api
def import_table(config: Mapping, text: str) -> dict:
    """Runs from a pasted or loaded table (JMP, CSV): points, and responses where present."""
    oracle = _oracle(config)
    table = csvio.parse_table(text)
    X, y, _, warnings = csvio.read_points(table, oracle)
    return {"points": X, "y": [None if np.isnan(v) else float(v) for v in y],
            "y_true": np.atleast_1d(oracle.true_response(X)), "warnings": warnings}


@_api
def export_table(config: Mapping, X: Sequence, y: Sequence) -> dict:
    oracle = _oracle(config)
    X = np.asarray(X, dtype=float).reshape(-1, oracle.dim)
    return {"csv": csvio.format_table(csvio.results_table(X, y, oracle))}


@_api
def truth(config: Mapping) -> dict:
    """Instructor view: the optimum and the other local minimum, in the oracle's units."""
    oracle = _oracle(config)
    return {"optimum": oracle.optimum(), "local_minima": oracle.local_minima(),
            "scenario": None if oracle.scenario.is_identity else oracle.scenario.to_dict()}


# --- Models -------------------------------------------------------------------------------------


@_api
def fit(config: Mapping, X: Sequence, y: Sequence, model: str = "gp", kernel: str = "matern52") -> dict:
    oracle = _oracle(config)
    U, yy = _data(oracle, X, y)
    return _describe(_fitted(oracle, U, yy, model, kernel), oracle)


@_api
def slice2d(config: Mapping, X: Sequence, y: Sequence, i: int, j: int, center: Sequence | None = None,
            n: int = 41, model: str | None = "gp", kernel: str = "matern52", layer: str = "mean",
            acquisition: str = "ei", truth: bool = False) -> dict:
    """Values on an n x n grid over factors i (across) and j (up), the others held at `center`.

    Grids are lists of rows: row r is the r-th value of factor j, column c the
    c-th value of factor i.
    """
    oracle = _oracle(config, budget=False)
    i, j, n = int(i), int(j), _grid_size(n)
    if i == j or not (0 <= i < oracle.dim and 0 <= j < oracle.dim):
        raise ValueError("choose two different factors")
    base = _center_unit(oracle, center)
    g = np.linspace(0.0, 1.0, n)
    gi, gj = np.meshgrid(g, g)  # gi varies along columns, gj along rows
    U = np.tile(base, (n * n, 1))
    U[:, i], U[:, j] = gi.ravel(), gj.ravel()
    axes = process.from_unit(np.column_stack([g] * oracle.dim), oracle.units)
    out = {"x": axes[:, i], "y": axes[:, j], "i": i, "j": j}
    if truth:
        out["true"] = oracle.true_response(process.from_unit(U, oracle.units)).reshape(n, n)
    if model:
        Um, ym = _data(oracle, X, y)
        fitted = _fitted(oracle, Um, ym, model, kernel)
        out["model"] = _layer(fitted, U, layer, ym, acquisition).reshape(n, n)
        out["layer"] = layer
    return out


@_api
def profile(config: Mapping, X: Sequence, y: Sequence, center: Sequence | None = None, n: int = 41,
            model: str | None = "gp", kernel: str = "matern52", truth: bool = False) -> dict:
    """One trace per factor through `center`: model mean and SD, and the true response if asked."""
    oracle = _oracle(config, budget=False)
    n = _grid_size(n)
    base = _center_unit(oracle, center)
    g = np.linspace(0.0, 1.0, n)
    axes = process.from_unit(np.column_stack([g] * oracle.dim), oracle.units)
    fitted = None
    if model:
        Um, ym = _data(oracle, X, y)
        fitted = _fitted(oracle, Um, ym, model, kernel)
    traces = []
    for k in range(oracle.dim):
        U = np.tile(base, (n, 1))
        U[:, k] = g
        trace = {"x": axes[:, k]}
        if fitted is not None:
            trace["mean"], trace["sd"] = fitted.predict(U)
        if truth:
            trace["true"] = oracle.true_response(process.from_unit(U, oracle.units))
        traces.append(trace)
    at = {}
    if fitted is not None:
        mean, sd = fitted.predict(base[None, :])
        at = {"mean": float(mean[0]), "sd": float(sd[0])}
    if truth:
        at["true"] = oracle.true_response(process.from_unit(base, oracle.units))
    return {"traces": traces, "at": at}


@_api
def suggest(config: Mapping, X: Sequence, y: Sequence, method: str = "bo", acquisition: str = "ei", q: int = 1,
            kernel: str = "matern52", xi: float = 0.0, kappa: float = 2.0, seed: int = 0) -> dict:
    """The next q points a strategy would measure (not measured yet), in the oracle's units."""
    from seqopt.acquisition import propose

    oracle = _oracle(config)
    U, yy = _data(oracle, X, y)
    q = int(q)
    if q < 1:
        raise ValueError("q must be at least 1")
    remaining = oracle.budget.remaining
    if remaining is not None:
        remaining -= len(yy)
        if remaining <= 0:
            raise ValueError("the budget is used up")
        q = min(q, remaining)
    rng = np.random.default_rng([int(seed), len(yy)])
    warnings, values, info = [], None, None
    if method == "random":
        picks = rng.random((q, oracle.dim))
    elif method in MODELS or method == "bo":
        kind = "gp" if method == "bo" else "rsm"
        fitted = _fitted(oracle, U, yy, kind, kernel)
        if kind == "rsm":
            from seqopt.rsm import n_terms

            if len(yy) < n_terms(oracle.dim):
                warnings.append(f"A full quadratic has {n_terms(oracle.dim)} terms but only {len(yy)} runs are "
                                "measured, so the fit is not unique.")
            picks, values = propose(fitted, "exploit", q, rng, incumbent_rule="observed", min_distance=0.02)
        else:
            picks, values = propose(fitted, acquisition, q, rng, xi=float(xi), kappa=float(kappa))
        info = _describe(fitted, oracle)
    else:
        raise ValueError(f"unknown method {method!r}; choose bo, rsm or random")
    return {"points": process.from_unit(picks, oracle.units), "acquisition": values, "model": info,
            "warnings": warnings}


# --- Benchmark -------------------------------------------------------------------------------


@_api
def benchmark_run(config: Mapping, strategy: Mapping, replicate: int, budget: int, seed: int = 0) -> dict:
    """One strategy on one replicate: its best-so-far curves (observed, and true at the best point)."""
    from seqopt.benchmark import run_benchmark
    from seqopt.loop import Strategy

    base = _oracle(config, budget=False)
    replicate = int(replicate)
    cfg = {**base.config, "seed": base.seed + replicate}
    result = run_benchmark(lambda r: HartmannOracle.from_config(cfg), [Strategy.from_dict(dict(strategy))],
                           int(budget), replicates=1, seed=int(seed) + replicate)
    label = result.labels[0]
    return {"label": label, "observed": result.observed[label][0], "true": result.true[label][0]}


@_api
def benchmark_summary(curves: Mapping, optimum: float | None = None) -> dict:
    """Median and quartiles across replicates, per strategy, for curves gathered by benchmark_run."""
    out = {"labels": list(curves), "summary": {"observed": {}, "true": {}}, "optimum": optimum}
    for label, c in curves.items():
        for which in ("observed", "true"):
            data = np.asarray(c[which], dtype=float)
            if data.ndim != 2 or not len(data):
                raise ValueError(f"{label}: no {which} curves")
            q25, median, q75 = np.percentile(data, [25, 50, 75], axis=0)
            out["summary"][which][label] = {"median": median, "q25": q25, "q75": q75}
        out["replicates"] = len(np.asarray(c["observed"]))
        out["budget"] = np.asarray(c["observed"]).shape[1]
    return out
