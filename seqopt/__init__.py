"""Sequential optimization kernel: designs, surrogates, acquisitions and the loop.

Oracle-agnostic by design: nothing in this package imports `hartmann`, so the
same kernel can drive any function that satisfies `seqopt.oracle.Oracle`.
NumPy and SciPy only.
"""

from .oracle import Budget, BudgetExhausted, Oracle

__all__ = ["Budget", "BudgetExhausted", "Oracle"]
