"""Strategy comparison: several strategies, many seeded replicates, the same budget.

For replicate r and each strategy, a fresh oracle from `make_oracle(r)` is
optimized by `Optimizer(oracle, strategy, seed=seed + r)` for `budget`
evaluations. Every strategy in a replicate gets the same seed (common random
numbers). They see the same measurement noise stream when the oracle keys
noise to the evaluation number, as the Hartmann and test-function oracles do,
and the same initial design when they share one. So a difference between
strategies is less likely to be the luck of the draw.

After each evaluation n two curves are recorded:
- best observed: the lowest measurement among the first n.
- best true: the noiseless value at that best-observed point, if the oracle
  has `true_response`. It is what you would get by running the recommended
  setting again. With noise the observed curve flatters every strategy, so
  the true curve is the honest one.

`summary()` gives the median and 25th/75th percentiles across replicates at
each n: the median line and spread band of best-so-far vs budget.

Runs are sequential in one process. Many small linear-algebra calls from
several processes oversubscribe BLAS threads; set OPENBLAS_NUM_THREADS=1 if
you parallelize this across processes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .loop import Optimizer, Strategy
from .oracle import Oracle

CURVES = ("observed", "true")


@dataclass
class BenchmarkResult:
    labels: list[str]
    strategies: list[dict]
    budget: int
    replicates: int
    seed: int
    observed: dict[str, np.ndarray]  # label -> (replicates, budget) best observed so far
    true: dict[str, np.ndarray] | None  # label -> (replicates, budget) true value at the best observed point
    optimum: float | None
    seconds: dict[str, float]

    def _curves(self, which: str) -> dict[str, np.ndarray]:
        if which not in CURVES:
            raise ValueError("which must be 'observed' or 'true'")
        if which == "true":
            if self.true is None:
                raise ValueError("this oracle has no true_response, so there is no true curve")
            return self.true
        return self.observed

    def summary(self, which: str = "true") -> dict[str, dict[str, list[float]]]:
        """Per strategy: median, 25th and 75th percentile of best-so-far after each evaluation."""
        out = {}
        for label, curves in self._curves(which).items():
            q25, median, q75 = np.percentile(curves, [25, 50, 75], axis=0)
            out[label] = {"median": median.tolist(), "q25": q25.tolist(), "q75": q75.tolist()}
        return out

    def at(self, n: int, which: str = "true") -> dict[str, dict[str, float]]:
        """The same statistics after n evaluations."""
        if not 1 <= n <= self.budget:
            raise ValueError(f"n must be 1-{self.budget}")
        return {label: {k: v[n - 1] for k, v in stats.items()} for label, stats in self.summary(which).items()}

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "strategies": self.strategies,
            "budget": self.budget,
            "replicates": self.replicates,
            "seed": self.seed,
            "optimum": self.optimum,
            "seconds": self.seconds,
            "summary": {w: self.summary(w) for w in CURVES if w == "observed" or self.true is not None},
            "curves": {
                label: {
                    "observed": self.observed[label].tolist(),
                    **({"true": self.true[label].tolist()} if self.true is not None else {}),
                }
                for label in self.labels
            },
        }

    def rows(self) -> list[dict]:
        """Long format, one row per strategy, replicate and evaluation count (for JMP's Graph Builder)."""
        rows = []
        for label in self.labels:
            for r in range(self.replicates):
                for n in range(self.budget):
                    row = {"strategy": label, "replicate": r + 1, "evaluations": n + 1,
                           "best_observed": float(self.observed[label][r, n])}
                    if self.true is not None:
                        row["best_true"] = float(self.true[label][r, n])
                    rows.append(row)
        return rows


def run_benchmark(
    make_oracle: Callable[[int], Oracle],
    strategies: Sequence[Strategy],
    budget: int,
    replicates: int = 10,
    seed: int = 0,
    optimum: float | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> BenchmarkResult:
    """Run every strategy on `replicates` fresh oracles for `budget` evaluations each.

    `progress(done, total, label)` is called after each run, for progress bars.
    """
    strategies = list(strategies)
    labels = [s.label for s in strategies]
    if len(set(labels)) != len(labels):
        raise ValueError(f"strategy labels must differ (give them names): {labels}")
    budget, replicates = int(budget), int(replicates)
    if budget < 1 or replicates < 1:
        raise ValueError("budget and replicates must be at least 1")

    observed = {label: np.empty((replicates, budget)) for label in labels}
    true: dict[str, np.ndarray] | None = {label: np.empty((replicates, budget)) for label in labels}
    seconds = {label: 0.0 for label in labels}
    total, done = replicates * len(strategies), 0
    for r in range(replicates):
        for strategy, label in zip(strategies, labels):
            oracle = make_oracle(r)
            started = time.perf_counter()
            opt = Optimizer(oracle, strategy, seed=seed + r)
            opt.run(budget)
            seconds[label] += time.perf_counter() - started
            if len(opt.y) < budget:
                raise ValueError(f"the oracle's budget ({oracle.budget.total}) is smaller than the benchmark's ({budget})")
            observed[label][r] = np.minimum.accumulate(opt.y)
            if true is not None and hasattr(oracle, "true_response"):
                best_index = _running_argmin(opt.y)
                true[label][r] = np.asarray(oracle.true_response(opt.to_native(opt.U)))[best_index]
            else:
                true = None
            done += 1
            if progress is not None:
                progress(done, total, label)
    return BenchmarkResult(labels, [s.to_dict() for s in strategies], budget, replicates, int(seed),
                           observed, true, optimum, seconds)


def _running_argmin(y: np.ndarray) -> np.ndarray:
    """Index of the smallest value among the first n, for every n (earliest on ties)."""
    out = np.empty(len(y), dtype=np.intp)
    best = 0
    for i in range(len(y)):
        if y[i] < y[best]:
            best = i
        out[i] = best
    return out
