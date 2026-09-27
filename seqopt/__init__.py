"""Sequential optimization kernel: designs, surrogates, acquisitions and the loop.

Oracle-agnostic by design: nothing in this package imports `hartmann`, so the
same kernel can drive any function that satisfies `seqopt.oracle.Oracle`.
NumPy and SciPy only.

The names below load on first use, so importing `seqopt.oracle` or
`seqopt.designs` needs only NumPy. The browser app relies on this: it can
design and measure while SciPy is still downloading.
"""

import importlib

_EXPORTS = {
    "BenchmarkResult": ".benchmark",
    "run_benchmark": ".benchmark",
    "GaussianProcess": ".gp",
    "Optimizer": ".loop",
    "StepRecord": ".loop",
    "Strategy": ".loop",
    "Budget": ".oracle",
    "BudgetExhausted": ".oracle",
    "Oracle": ".oracle",
    "QuadraticRSM": ".rsm",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module 'seqopt' has no attribute {name!r}")
