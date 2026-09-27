"""The oracle: evaluate points the way a fab measures wafers.

Every call to `evaluate` costs one evaluation per point against the budget,
and every evaluation is written to the ledger. The Hartmann function is
deterministic; only the optional noise term (see `noise.py`) varies, and it is
seeded, so any run can be rebuilt from `config` and replayed exactly.

Settings, all recorded in `config`:
- units: what points are given in and what comes back.
    "process" (default): PECVD engineering units in; non-uniformity (%) out.
    "unit" or "coded":   [0, 1] or [-1, 1] per factor in; the function value out.
- form: "standard", "rescaled" or "sfu_rescaled" (see `function.py`). Process
  units always use the standard form, which their % mapping is calibrated to.
- noise: a NoiseModel, or a dict of its settings; SD is in response units.
- seed: the noise seed. None picks one at random and records it.
- budget: the maximum number of evaluations; None means unlimited.
- scenario: None for no disguise, a seed for a blind scenario (see
  `scenario.py`), or a Scenario.

The ledger and `optimum()` include the noiseless response. Hiding them from
students is the front end's job (blind mode); the oracle has no secrets.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from seqopt.oracle import Budget

from . import process
from .function import DIM, FORMS, LOCAL_MINIMA, hartmann6
from .noise import NoiseModel
from .process import RESPONSE, UNITS, Response
from .scenario import IDENTITY, Scenario

UNIT_RESPONSE = Response("y", "Response", "", offset=0.0, scale=1.0)


@dataclass(frozen=True)
class Evaluation:
    k: int  # evaluation number, counting from 0
    x: tuple[float, ...]  # the point as given, in the oracle's units
    y: float  # observed response (noise included)
    y_true: float  # noiseless response (instructor view only)


class HartmannOracle:
    """Hartmann 6D as a budgeted, optionally noisy, optionally disguised measurement."""

    dim = DIM

    def __init__(
        self,
        *,
        units: str = "process",
        form: str = "standard",
        noise: NoiseModel | Mapping | None = None,
        seed: int | None = None,
        budget: int | None = None,
        scenario: Scenario | int | None = None,
    ):
        if units not in UNITS:
            raise ValueError(f"unknown units {units!r}; choose one of {', '.join(UNITS)}")
        if form not in FORMS:
            raise ValueError(f"unknown form {form!r}; choose one of {', '.join(FORMS)}")
        if units == "process" and form != "standard":
            raise ValueError("process units report non-uniformity (%), which is defined on the standard form")
        if seed is None:
            seed = secrets.randbits(32)
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError(f"seed must be a non-negative whole number, got {seed!r}")

        self.units = units
        self.form = form
        self.noise = noise if isinstance(noise, NoiseModel) else NoiseModel.from_dict(noise)
        self.seed = int(seed)
        self.budget = Budget(budget)
        if scenario is None:
            self.scenario = IDENTITY
        elif isinstance(scenario, Scenario):
            self.scenario = scenario
        else:
            self.scenario = Scenario.from_seed(scenario)
        self.ledger: list[Evaluation] = []

    # --- Description -------------------------------------------------------

    @property
    def response(self) -> Response:
        return RESPONSE if self.units == "process" else UNIT_RESPONSE

    @property
    def bounds(self) -> np.ndarray:
        """(6, 2) array of [low, high] per factor, in the oracle's units."""
        if self.units == "process":
            return np.column_stack([process.LOW, process.HIGH])
        low = -1.0 if self.units == "coded" else 0.0
        return np.tile([low, 1.0], (DIM, 1))

    def factors(self) -> list[dict]:
        """The factors as the student sees them, in order."""
        if self.units == "process":
            return process.table()
        low, high = self.bounds[0]
        return [
            {"name": f"x{j + 1}", "label": f"x{j + 1}", "unit": self.units, "low": low, "high": high,
             "center": 0.5 * (low + high), "description": ""}
            for j in range(DIM)
        ]

    @property
    def config(self) -> dict:
        """Everything needed to rebuild this oracle (not its ledger)."""
        return {
            "units": self.units,
            "form": self.form,
            "noise": self.noise.to_dict(),
            "seed": self.seed,
            "budget": self.budget.total,
            "scenario": None if self.scenario.is_identity else self.scenario.to_dict(),
        }

    @classmethod
    def from_config(cls, config: Mapping) -> HartmannOracle:
        unknown = set(config) - {"units", "form", "noise", "seed", "budget", "scenario"}
        if unknown:
            raise ValueError(f"unknown oracle settings: {sorted(unknown)}")
        scenario = config.get("scenario")
        return cls(
            units=config.get("units", "process"),
            form=config.get("form", "standard"),
            noise=config.get("noise"),
            seed=config.get("seed"),
            budget=config.get("budget"),
            scenario=Scenario.from_dict(scenario) if isinstance(scenario, Mapping) else scenario,
        )

    # --- Evaluation --------------------------------------------------------

    def _true(self, U: np.ndarray) -> np.ndarray:
        f = hartmann6(self.scenario.to_hartmann(U), self.form)
        return self.response.from_f(f)

    def evaluate(self, X):
        """Measure point(s): a point gives a float, an (n, 6) array gives n values.

        Costs one evaluation per point. Points are checked before anything is
        charged, and a batch that would overrun the budget is refused whole.
        """
        X = np.asarray(X, dtype=float)
        U = np.atleast_2d(process.to_unit(X, self.units))
        first = self.budget.charge(len(U))
        y_true = self._true(U)
        y = y_true + self.noise.draw(self.seed, first, U)
        for i, (x_row, yi, ti) in enumerate(zip(np.atleast_2d(X), y, y_true)):
            self.ledger.append(Evaluation(first + i, tuple(float(v) for v in x_row), float(yi), float(ti)))
        return float(y[0]) if X.ndim == 1 else y

    def record(self, X, y) -> None:
        """Enter measurements made earlier, such as rows of a results table, without measuring again.

        They count against the budget and take the next evaluation numbers, so
        measurements made afterwards continue the same noise stream.
        """
        X = np.asarray(X, dtype=float)
        U = np.atleast_2d(process.to_unit(X, self.units))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if len(y) != len(U):
            raise ValueError(f"{len(U)} points but {len(y)} responses")
        if not np.all(np.isfinite(y)):
            raise ValueError("recorded responses must be numbers")
        first = self.budget.charge(len(U))
        y_true = self._true(U)
        for i, (x_row, yi, ti) in enumerate(zip(np.atleast_2d(X), y, y_true)):
            self.ledger.append(Evaluation(first + i, tuple(float(v) for v in x_row), float(yi), float(ti)))

    def true_response(self, X):
        """Noiseless response at point(s); free, for the instructor view and for scoring."""
        X = np.asarray(X, dtype=float)
        y = self._true(np.atleast_2d(process.to_unit(X, self.units)))
        return float(y[0]) if X.ndim == 1 else y

    def replay(self) -> HartmannOracle:
        """A fresh oracle with the same config that re-measures the ledger's points in order."""
        other = HartmannOracle.from_config(self.config)
        for e in self.ledger:
            other.evaluate(np.array(e.x))
        return other

    # --- Truth (instructor view) -------------------------------------------

    def _minimum(self, x_hartmann, f_label: str) -> dict:
        u = self.scenario.from_hartmann(np.asarray(x_hartmann))
        x = process.from_unit(u, self.units)
        return {"x": [float(v) for v in x], "y": self.true_response(x), "label": f_label}

    def optimum(self) -> dict:
        """Where the global minimum is, in this oracle's units and factor order, and its response."""
        best = LOCAL_MINIMA[0]
        return self._minimum(best.x, best.label)

    def local_minima(self) -> list[dict]:
        return [self._minimum(m.x, m.label) for m in LOCAL_MINIMA]

    def best_observed(self) -> Evaluation | None:
        """The ledger entry with the lowest observed response, if any."""
        return min(self.ledger, key=lambda e: e.y, default=None)
