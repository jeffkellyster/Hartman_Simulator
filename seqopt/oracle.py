"""What the optimization kernel needs from an oracle, and the evaluation budget.

The kernel never knows which function it is optimizing. Anything with a
dimension, box bounds in its own units, a `Budget`, and an `evaluate` method
that turns an (n, dim) array of points into n responses can be driven by it:
the Hartmann oracle today, Branin-Hoo or Rosenbrock next, the race car
simulator later. Responses are minimized.

Every evaluation costs one unit of budget, like measuring one wafer, and a
batch of q points costs q. A batch that would overrun the budget is refused
whole, before anything is evaluated, so a refused batch never leaves a partial
result behind.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class BudgetExhausted(RuntimeError):
    """Raised when an evaluation would exceed the budget. Nothing is evaluated."""


class Budget:
    """Evaluation counter with an optional limit (None means unlimited)."""

    def __init__(self, total: int | None = None, used: int = 0):
        if total is not None and (isinstance(total, bool) or int(total) != total or total < 0):
            raise ValueError(f"budget must be a non-negative whole number or None, got {total!r}")
        if int(used) != used or used < 0:
            raise ValueError(f"used must be a non-negative whole number, got {used!r}")
        self.total = None if total is None else int(total)
        self.used = int(used)
        if self.total is not None and self.used > self.total:
            raise ValueError(f"used ({self.used}) exceeds the budget ({self.total})")

    @property
    def remaining(self) -> int | None:
        return None if self.total is None else self.total - self.used

    def charge(self, n: int) -> int:
        """Spend n evaluations and return the index of the first (evaluations count from 0)."""
        if int(n) != n or n < 0:
            raise ValueError(f"cannot charge {n!r} evaluations")
        n = int(n)
        if self.total is not None and self.used + n > self.total:
            raise BudgetExhausted(
                f"{n} evaluation(s) requested but only {self.remaining} of {self.total} remain"
            )
        first = self.used
        self.used += n
        return first

    def __repr__(self) -> str:
        return f"Budget(total={self.total}, used={self.used})"


@runtime_checkable
class Oracle(Protocol):
    """The contract between the kernel and a function it optimizes."""

    dim: int
    budget: Budget

    @property
    def bounds(self) -> np.ndarray:
        """(dim, 2) array of [low, high] per input, in the oracle's own units."""
        ...

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Responses (to minimize) for an (n, dim) array of points; charges n evaluations."""
        ...
