"""Sequential optimization kernel: designs, surrogates, acquisitions and the loop.

Oracle-agnostic by design: nothing in this package imports `hartmann`, so the
same kernel can drive any function that satisfies `seqopt.oracle.Oracle`.
NumPy and SciPy only.
"""

from .benchmark import BenchmarkResult, run_benchmark
from .gp import GaussianProcess
from .loop import Optimizer, StepRecord, Strategy
from .oracle import Budget, BudgetExhausted, Oracle
from .rsm import QuadraticRSM

__all__ = [
    "BenchmarkResult",
    "Budget",
    "BudgetExhausted",
    "GaussianProcess",
    "Optimizer",
    "Oracle",
    "QuadraticRSM",
    "StepRecord",
    "Strategy",
    "run_benchmark",
]
