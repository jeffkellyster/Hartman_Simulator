"""The sequential optimization loop: initial design, fit, propose, evaluate, update.

Three methods, each a `Strategy`:
- "random": every point uniformly at random, batch_size at a time. The baseline.
- "rsm":    a space-filling initial design big enough for the full quadratic
            (terms + 2 runs), then each step fits the quadratic and measures
            where it predicts the minimum, at least `min_distance` away from
            points already measured so it cannot repeat itself.
- "bo":     a space-filling initial design (2d runs by default), then each
            step fits a GP and measures the point(s) that maximize the
            acquisition function. q > 1 gives batches.

Everything happens on the unit cube; the oracle's bounds map it to the
oracle's own units, so any `seqopt.oracle.Oracle` can be driven. Each step is
logged as a `StepRecord` (the initial design is step 0), and every random
choice comes from the optimizer's seed, so a run repeats exactly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import designs
from .acquisition import ACQUISITIONS, BATCH_RULES, propose
from .gp import KERNELS, GaussianProcess
from .oracle import Oracle
from .rsm import QuadraticRSM, n_terms

METHODS = ("random", "rsm", "bo")
ACQUISITION_LABELS = {"ei": "EI", "pi": "PI", "ucb": "UCB", "exploit": "exploit", "explore": "explore"}


@dataclass(frozen=True)
class Strategy:
    method: str = "bo"
    name: str | None = None  # label for logs and plots; a default is built from the settings
    initial_design: str = "maximin_lhs"
    initial_runs: int | None = None  # default: 2d for bo, quadratic terms + 2 for rsm
    acquisition: str = "ei"
    batch_size: int = 1
    kernel: str = "matern52"
    noise: str | float = "estimate"
    xi: float = 0.0
    kappa: float = 2.0
    batch_rule: str = "believer"
    incumbent: str = "mean"
    restarts: int = 4  # GP random restarts on the first fit; later fits warm-start with one
    min_distance: float | None = None  # default: 0.02 for rsm, 0 for bo
    n_candidates: int = 2000

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose one of {', '.join(METHODS)}")
        if self.acquisition not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition {self.acquisition!r}; choose one of {', '.join(ACQUISITIONS)}")
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {self.kernel!r}; choose one of {', '.join(KERNELS)}")
        if self.batch_rule not in BATCH_RULES:
            raise ValueError(f"unknown batch rule {self.batch_rule!r}")
        if self.initial_design not in designs.KINDS:
            raise ValueError(f"unknown initial design {self.initial_design!r}")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be at least 1")

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.method == "random":
            return "Random"
        if self.method == "rsm":
            return "LHS + RSM"
        q = f", q={self.batch_size}" if self.batch_size > 1 else ""
        return f"BO ({ACQUISITION_LABELS[self.acquisition]}{q})"

    def initial_count(self, d: int) -> int:
        if self.initial_runs is not None:
            return int(self.initial_runs)
        return n_terms(d) + 2 if self.method == "rsm" else 2 * d

    def distance(self) -> float:
        if self.min_distance is not None:
            return float(self.min_distance)
        return 0.02 if self.method == "rsm" else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> Strategy:
        return cls(**values)


@dataclass
class StepRecord:
    step: int  # 0 is the initial design
    phase: str  # "initial", "sequential", or "random"
    evaluations: list[int]  # this optimizer's evaluation numbers, from 0
    x: list[list[float]]  # points in the oracle's units
    y: list[float]
    best_y: float  # best measurement so far, after this step
    acquisition: list[float] | None = None
    model: dict | None = None  # the surrogate that proposed these points
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Optimizer:
    oracle: Oracle
    strategy: Strategy = field(default_factory=Strategy)
    seed: int = 0
    factors: list[str] | None = None  # names for RSM coefficients and logs

    def __post_init__(self):
        self.d = int(self.oracle.dim)
        bounds = np.asarray(self.oracle.bounds, dtype=float)
        self._low, self._high = bounds[:, 0], bounds[:, 1]
        self.U = np.empty((0, self.d))  # measured points, unit cube
        self.y = np.empty(0)
        self.records: list[StepRecord] = []
        self._gp_params = None

    # --- Units ---------------------------------------------------------------------

    def to_native(self, U) -> np.ndarray:
        U = np.atleast_2d(U)
        return self._low * (1.0 - U) + self._high * U

    # --- Loop ------------------------------------------------------------------------

    def _remaining(self, limit: int | None) -> int | None:
        left = [r for r in (self.oracle.budget.remaining, None if limit is None else limit - len(self.y))
                if r is not None]
        return min(left) if left else None

    def _measure(self, U: np.ndarray, phase: str, started: float, acquisition=None, model=None) -> StepRecord:
        X = self.to_native(U)
        y = np.asarray(self.oracle.evaluate(X), dtype=float)
        first = len(self.y)
        self.U = np.vstack([self.U, U])
        self.y = np.append(self.y, y)
        record = StepRecord(
            step=len(self.records),
            phase=phase,
            evaluations=list(range(first, len(self.y))),
            x=X.tolist(),
            y=y.tolist(),
            best_y=float(self.y.min()),
            acquisition=None if acquisition is None else [float(a) for a in acquisition],
            model=model,
            seconds=time.perf_counter() - started,
        )
        self.records.append(record)
        return record

    def step(self, limit: int | None = None) -> StepRecord | None:
        """Run one step (the initial design first); None when the budget or `limit` is used up."""
        started = time.perf_counter()
        room = self._remaining(limit)
        if room is not None and room <= 0:
            return None
        s = self.strategy
        rng = np.random.default_rng([self.seed, len(self.records)])

        if s.method == "random":
            n = s.batch_size if room is None else min(s.batch_size, room)
            return self._measure(rng.random((n, self.d)), "random", started)

        if not self.records:
            n = s.initial_count(self.d)
            n = n if room is None else min(n, room)
            if n < 2:
                U = rng.random((n, self.d))
            else:
                U = designs.make(s.initial_design, self.d, runs=n, seed=self.seed).unit
            return self._measure(U, "initial", started)

        q = s.batch_size if room is None else min(s.batch_size, room)
        if s.method == "rsm":
            model = QuadraticRSM().fit(self.U, self.y)
            U, values = propose(model, "exploit", q, rng, batch=s.batch_rule, incumbent_rule="observed",
                                min_distance=s.distance(), n_candidates=s.n_candidates)
            info = model.describe(self.factors)
        else:
            model = GaussianProcess(kernel=s.kernel, noise=s.noise, seed=int(rng.integers(2**31)))
            restarts = s.restarts if self._gp_params is None else 1
            model.fit(self.U, self.y, init=self._gp_params, restarts=restarts)
            self._gp_params = model.params
            U, values = propose(model, s.acquisition, q, rng, xi=s.xi, kappa=s.kappa, batch=s.batch_rule,
                                incumbent_rule=s.incumbent, min_distance=s.distance(),
                                n_candidates=s.n_candidates)
            info = model.describe()
        return self._measure(U, "sequential", started, values, info)

    def run(self, evaluations: int | None = None) -> list[StepRecord]:
        """Step until the oracle's budget, or `evaluations` measurements by this optimizer, is used up."""
        if evaluations is None and self.oracle.budget.remaining is None:
            raise ValueError("the oracle has no budget; say how many evaluations to run")
        while self.step(evaluations) is not None:
            pass
        return self.records

    # --- Results ---------------------------------------------------------------------

    def best(self) -> tuple[np.ndarray, float]:
        """The best measured point (oracle units) and its measurement."""
        if not len(self.y):
            raise ValueError("nothing measured yet")
        i = int(np.argmin(self.y))
        return self.to_native(self.U[i])[0], float(self.y[i])

    def best_so_far(self) -> np.ndarray:
        return np.minimum.accumulate(self.y)

    def log_rows(self) -> list[dict]:
        """One row per evaluation, for CSV: evaluation, step, phase, the point, the response."""
        rows = []
        for r in self.records:
            for k, x, y in zip(r.evaluations, r.x, r.y):
                rows.append({"evaluation": k + 1, "step": r.step, "phase": r.phase, "x": x, "y": y})
        return rows

    def log(self) -> dict:
        return {
            "strategy": self.strategy.to_dict(),
            "label": self.strategy.label,
            "seed": self.seed,
            "steps": [r.to_dict() for r in self.records],
        }
