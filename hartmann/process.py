"""Process mode: the six factors as a fictional PECVD film-deposition process.

Students work in engineering units and never see [0, 1]^6. Each factor maps
linearly onto one unit-cube coordinate (low end -> 0, high end -> 1), and the
response is thickness non-uniformity in percent:

    non-uniformity (%) = 6.0 + 1.5 * f(x)        (standard form, lower is better)

so the best achievable is about 1.02 % and the worst about 6 %. The mapping is
linear, so coded analysis in JMP is unchanged by it. The process is fiction:
the response is the Hartmann function, not deposition physics, and the factor
descriptions only set the scene.

Three input unit systems are supported everywhere:
- "process": engineering units within each factor's range
- "unit":    [0, 1] per factor
- "coded":   [-1, +1] per factor (JMP's coded levels)
Coded -1, 0 and +1 land exactly on a factor's low end, center and high end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 6
UNITS = ("process", "unit", "coded")
TOLERANCE = 1e-9  # share of a factor's range allowed outside it (rounding in CSV round-trips)

PROCESS_NAME = "PECVD film deposition"


@dataclass(frozen=True)
class ProcessFactor:
    name: str
    label: str
    unit: str
    low: float
    high: float
    description: str = ""

    @property
    def center(self) -> float:
        return 0.5 * (self.low + self.high)


FACTORS: tuple[ProcessFactor, ...] = (
    ProcessFactor("susceptor_temp_c", "Susceptor temperature", "°C", 250.0, 450.0,
                  "Heater set point under the wafer."),
    ProcessFactor("pressure_torr", "Chamber pressure", "Torr", 0.5, 5.0,
                  "Process pressure during deposition."),
    ProcessFactor("dep_time_s", "Deposition time", "s", 30.0, 180.0,
                  "Plasma-on time for the film."),
    ProcessFactor("sih4_flow_sccm", "SiH4 flow", "sccm", 50.0, 400.0,
                  "Silane precursor flow."),
    ProcessFactor("rf_power_w", "RF power", "W", 100.0, 800.0,
                  "Plasma power."),
    ProcessFactor("electrode_spacing_mm", "Electrode spacing", "mm", 8.0, 25.0,
                  "Gap between the showerhead and the susceptor."),
)
FACTORS_BY_NAME = {f.name: f for f in FACTORS}
LOW = np.array([f.low for f in FACTORS])
HIGH = np.array([f.high for f in FACTORS])
for _constant in (LOW, HIGH):
    _constant.setflags(write=False)


@dataclass(frozen=True)
class Response:
    name: str
    label: str
    unit: str
    offset: float
    scale: float
    goal: str = "minimize"

    def from_f(self, f):
        """Response for function value(s) f (a float or an array)."""
        return self.offset + self.scale * f

    def to_f(self, y):
        return (y - self.offset) / self.scale


RESPONSE = Response("non_uniformity_pct", "Thickness non-uniformity", "%", offset=6.0, scale=1.5)


def _as_points(x) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == DIM:
        return arr[None, :], True
    if arr.ndim == 2 and arr.shape[1] == DIM:
        return arr, False
    raise ValueError(f"expected a point of length {DIM} or an (n, {DIM}) array, got shape {arr.shape}")


def _shape_like(arr: np.ndarray, single: bool) -> np.ndarray:
    return arr[0] if single else arr


def _check_range(values: np.ndarray, low: np.ndarray, high: np.ndarray, names: list[str], units: str) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite numbers (no missing values)")
    slack = TOLERANCE * (high - low)
    bad = (values < low - slack) | (values > high + slack)
    if bad.any():
        row, col = map(int, np.argwhere(bad)[0])
        raise ValueError(
            f"row {row + 1}: {names[col]} = {values[row, col]:g} is outside its {units} range "
            f"[{low[col]:g}, {high[col]:g}]"
        )
    return np.clip(values, low, high)


def to_unit(x, units: str = "process") -> np.ndarray:
    """Unit-cube coordinates for points given in `units`; checks every value is in range."""
    if units not in UNITS:
        raise ValueError(f"unknown units {units!r}; choose one of {', '.join(UNITS)}")
    X, single = _as_points(x)
    labels = [f"{f.label} ({f.unit})" if units == "process" else f.label for f in FACTORS]
    if units == "process":
        X = _check_range(X, LOW, HIGH, labels, units)
        U = (X - LOW) / (HIGH - LOW)
    elif units == "coded":
        X = _check_range(X, -np.ones(DIM), np.ones(DIM), labels, units)
        U = (X + 1.0) / 2.0
    else:
        U = _check_range(X, np.zeros(DIM), np.ones(DIM), labels, units)
    return _shape_like(U, single)


def from_unit(u, units: str = "process") -> np.ndarray:
    """Points in `units` for unit-cube coordinates. Ends and centers map exactly."""
    if units not in UNITS:
        raise ValueError(f"unknown units {units!r}; choose one of {', '.join(UNITS)}")
    U, single = _as_points(u)
    U = _check_range(U, np.zeros(DIM), np.ones(DIM), [f.label for f in FACTORS], "unit")
    if units == "process":
        out = LOW * (1.0 - U) + HIGH * U
    elif units == "coded":
        out = 2.0 * U - 1.0
    else:
        out = U.copy()
    return _shape_like(out, single)


def table() -> list[dict]:
    """One row per factor, for display and for the JMP add-in."""
    return [
        {"name": f.name, "label": f.label, "unit": f.unit, "low": f.low, "high": f.high,
         "center": f.center, "description": f.description}
        for f in FACTORS
    ]
