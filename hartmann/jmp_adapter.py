"""Use the simulator from JMP.

JMP 18 embeds Python 3.11 and JMP 19 embeds 3.13. The add-in's "Install or
Update Engine" installs this package (bundled as a wheel) plus NumPy and SciPy
into JMP's Python. Developers can install a checkout instead; jpip splits its
argument on spaces, so use a path without any (a symlink works):

    Python Submit( "import jmputils; jmputils.jpip('install', '-e /Users/Jeff/hartmann-sim')" );

A JMP table is the lab's record, just like a results CSV for the command line:
- Factor columns are matched by name or label, ignoring case and units in
  parentheses: "RF power", "rf_power_w" and "RF power (W)" all work. A factor
  without a column is held at the center of its range.
- The response column is "Thickness non-uniformity (%)" (or its name, or JMP's
  default "Y"). Rows that already have a value were measured earlier and count
  against the budget; rows without one are measured now, and their noise
  continues the same seeded stream.
- The lab's settings (seed, noise, budget, blind scenario) live in the table
  variable "Hartmann lab" as JSON, so a table carries its own lab.

JSL talks to this module through Python Send / Python Submit / Python Get,
passing JSON strings. Everything that touches a data table is in the three
helpers at the bottom (`_column_names`, `_column_values`, `_write_values`); if
a JMP version changes the `jmp.DataTable` API, only those should need changing.
The rest is plain Python, which is what the tests use (with a fake table).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

import numpy as np

from . import app, csvio
from .noise import NoiseModel
from .oracle import HartmannOracle

LAB_VARIABLE = "Hartmann lab"


# --- Lab settings -----------------------------------------------------------------------------


def _config(config: Mapping | str) -> dict:
    """A lab config from a dict or its JSON text (as stored in the table variable)."""
    if isinstance(config, str):
        if not config.strip():
            raise ValueError("This table has no lab settings yet. Use Lab Settings or Measure This Table first.")
        config = json.loads(config)
    return HartmannOracle.from_config(config).config


def lab_config(noise_sd: float = 0.05, hetero: float = 0.0, budget: int | None = 60, seed: int | None = None,
               scenario: int | None = None, units: str = "process") -> dict:
    """A lab's config; a missing seed is drawn and recorded so the lab can be repeated."""
    budget = None if budget is None or (isinstance(budget, float) and math.isnan(budget)) else int(budget)
    seed = None if seed is None or (isinstance(seed, float) and math.isnan(seed)) else int(seed)
    scenario = None if scenario is None or (isinstance(scenario, float) and math.isnan(scenario)) else int(scenario)
    oracle = HartmannOracle(units=units, noise=NoiseModel(sd=float(noise_sd), hetero=float(hetero)), seed=seed,
                            budget=budget, scenario=scenario)
    return oracle.config


def lab_json(**settings) -> str:
    return json.dumps(lab_config(**settings))


def describe_lab(config: Mapping | str) -> str:
    """The settings in one readable line, for dialogs and the table variable's companion note."""
    c = _config(config)
    oracle = HartmannOracle.from_config(c)
    unit = oracle.response.unit
    parts = ["PECVD process units" if c["units"] == "process" else f"{c['units']} units",
             f"noise SD {c['noise']['sd']:g}{' ' + unit if unit else ''}"]
    if c["noise"]["hetero"]:
        factor = oracle.factors()[c["noise"]["hetero_factor"]]["label"]
        parts.append(f"noise grows x{1 + c['noise']['hetero']:g} along {factor}")
    parts.append("no budget" if c["budget"] is None else f"budget {c['budget']} runs")
    parts.append(f"seed {c['seed']}")
    if c.get("scenario"):
        parts.append(f"blind scenario {c['scenario']['seed']}")
    return "; ".join(parts)


def factor_definitions(units: str = "process") -> list[dict]:
    """One row per factor: name, label, unit, range and center (what Set Up a Design loads into JMP's DOE)."""
    return HartmannOracle(units=units, seed=0).factors()


def catalog_json() -> str:
    return json.dumps(app.catalog())


# --- Pure functions on column values ------------------------------------------------------------


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return repr(float(value))
    return str(value).strip()


def _table(columns: Mapping[str, Sequence]) -> csvio.Table:
    names = list(columns)
    n = max((len(v) for v in columns.values()), default=0)
    rows = [[_cell(columns[name][i]) if i < len(columns[name]) else "" for name in names] for i in range(n)]
    return csvio.Table(names, rows)


def measure_columns(columns: Mapping[str, Sequence], config: Mapping | str) -> dict:
    """Measure every row with no response; the rows that have one are the lab's history.

    Returns the response column's name (and whether it has to be created),
    the values to write by row (0-based), and a summary for the message.
    """
    c = _config(config)
    oracle = HartmannOracle.from_config(c)
    table = _table(columns)
    if not table.rows:
        raise ValueError("The table has no rows.")
    factor_cols, response_col, _ = csvio.match_columns(table.header, oracle)
    result = csvio.measure_table(table, oracle)
    name = table.header[response_col] if response_col is not None else csvio.response_header(oracle)
    y = result.y
    best = int(np.argmin(y))
    b = oracle.budget
    unit = f" {oracle.response.unit}" if oracle.response.unit else ""
    lines = [f"Measured {result.measured} row(s); {result.prior} already had a response."]
    if b.total is not None:
        lines.append(f"Budget: {b.used} of {b.total} runs used, {b.remaining} left.")
    lines.append(f"Best so far: {y[best]:.4g}{unit} in row {best + 1}.")
    lines += result.warnings
    return {
        "message": "\n".join(lines),
        "response_column": name,
        "new_column": response_col is None,
        "values": {str(r): float(y[r]) for r in result.rows_measured},
        "measured": result.measured,
        "prior": result.prior,
        "budget": {"used": b.used, "total": b.total, "remaining": b.remaining},
        "best": {"row": best + 1, "y": float(y[best])},
        "factor_columns": [table.header[factor_cols[j]] for j in sorted(factor_cols)],
        "warnings": result.warnings,
    }


def _measured(columns: Mapping[str, Sequence], oracle: HartmannOracle):
    """Points and responses of the measured rows, how many rows are unmeasured, and the factor columns."""
    table = _table(columns)
    factor_cols, _, _ = csvio.match_columns(table.header, oracle)
    missing = [f["label"] for j, f in enumerate(oracle.factors()) if j not in factor_cols]
    if missing:
        raise ValueError("Suggest Next Runs needs a column for every factor, so the suggestions can be written "
                         f"back. Missing: {', '.join(missing)}.")
    X, y, _, _ = csvio.read_points(table, oracle)
    done = ~np.isnan(y)
    return X[done], y[done], int((~done).sum()), [table.header[factor_cols[j]] for j in range(oracle.dim)]


def suggest_columns(columns: Mapping[str, Sequence], config: Mapping | str, method: str = "bo",
                    acquisition: str = "ei", q: int = 1, kernel: str = "matern52") -> dict:
    """The next runs a strategy would measure, as rows to append (factor columns in factor order)."""
    c = _config(config)
    oracle = HartmannOracle.from_config(c)
    X, y, planned, names = _measured(columns, oracle)
    if len(y) < 2:
        raise ValueError(f"Measure at least 2 rows before asking for suggestions ({len(y)} measured so far).")
    q = int(q)
    if c["budget"] is not None:
        room = c["budget"] - len(y) - planned
        if room <= 0:
            raise ValueError(f"The budget is spoken for: {len(y)} rows measured and {planned} waiting to be measured, "
                             f"out of {c['budget']}. Measure or delete the waiting rows first.")
        q = min(q, room)
    result = app.suggest(c, X, y, method=method, acquisition=acquisition, q=q, kernel=kernel, seed=c["seed"])
    n = len(result["points"])
    lines = [f"Added {n} suggested row(s) at the bottom of the table (selected). "
             "Measure This Table measures them."]
    model = result["model"]
    if model and model.get("type") == "gp":
        ls = ", ".join(f"{name} {v:.2g}" for name, v in model["lengthscales"].items())
        lines.append(f"Gaussian process length scales (short = changes fast, 10 = hardly matters): {ls}.")
    elif model:
        lines.append(f"Quadratic RSM: R-square {model['r2']:.3f} from {model['runs']} runs.")
    lines += result["warnings"]
    return {"message": "\n".join(lines), "columns": names, "points": np.asarray(result["points"]).tolist(),
            "acquisition": None if result["acquisition"] is None else np.asarray(result["acquisition"]).tolist(),
            "model": result["model"], "warnings": result["warnings"]}


def truth_columns(columns: Mapping[str, Sequence], config: Mapping | str) -> dict:
    """Instructor view: the optimum and second minimum, and how the best measured row compares."""
    c = _config(config)
    oracle = HartmannOracle.from_config(c)
    info = app.truth(c)
    out = {"optimum": info["optimum"], "local_minima": info["local_minima"], "scenario": info["scenario"],
           "factors": [f["label"] for f in oracle.factors()], "unit": oracle.response.unit, "best": None}
    table = _table(columns)
    if table.rows:
        X, y, _, _ = csvio.read_points(table, oracle)
        done = np.flatnonzero(~np.isnan(y))
        if len(done):
            r = int(done[np.argmin(y[done])])
            true = oracle.true_response(X[r])
            out["best"] = {"row": r + 1, "measured": float(y[r]), "true": true,
                           "gap": true - info["optimum"]["y"]}
    unit = f" {oracle.response.unit}" if oracle.response.unit else ""

    def where(x):
        return ", ".join(f"{f['label']} {v:.4g}{' ' + f['unit'] if c['units'] == 'process' else ''}"
                         for f, v in zip(oracle.factors(), x))

    lines = ["Instructor view: keep this from students in blind mode.", ""]
    for m in info["local_minima"]:
        lines.append(f"{m['label'].capitalize()}: {m['y']:.4g}{unit} at {where(m['x'])}.")
    if out["best"]:
        b = out["best"]
        lines.append(f"Best measured row {b['row']}: measured {b['measured']:.4g}{unit}, true {b['true']:.4g}{unit}, "
                     f"{b['gap']:.4g}{unit} above the optimum.")
    if info["scenario"]:
        lines.append(f"Blind scenario {info['scenario']['seed']}: the factors are reordered and some reversed, "
                     "so the published Hartmann optimum does not apply.")
    out["message"] = "\n".join(lines)
    return out


# --- Data table entry points (called from the add-in) ---------------------------------------------


def _columns(dt) -> dict[str, list]:
    return {name: _column_values(dt, name) for name in _column_names(dt)}


def run_table(dt, config: Mapping | str) -> dict:
    """Measure the table's unmeasured rows and write the response column. Returns the summary."""
    result = measure_columns(_columns(dt), config)
    _write_values(dt, result["response_column"], {int(r): v for r, v in result["values"].items()})
    return result


def suggest_table(dt, config: Mapping | str, method: str = "bo", acquisition: str = "ei", q: int = 1,
                  kernel: str = "matern52") -> dict:
    """Suggestions for this table; the add-in appends them as new rows."""
    return suggest_columns(_columns(dt), config, method=method, acquisition=acquisition, q=q, kernel=kernel)


def truth_table(dt, config: Mapping | str) -> dict:
    return truth_columns(_columns(dt), config)


def to_json(value) -> str:
    """JSON for JSL's Parse JSON (NumPy values converted, non-finite numbers as null)."""
    return json.dumps(app._jsonable(value), allow_nan=False)


# --- JMP-specific helpers ---------------------------------------------------------------------------


def _column_names(dt) -> list[str]:
    # jmp.DataTable iterates over its columns (JMP 18); older or fake tables may index instead.
    try:
        return [column.name for column in dt]
    except TypeError:
        return [dt[i].name for i in range(dt.ncols)]


def _column_values(dt, name: str) -> list:
    column = dt[name]
    return [column[i] for i in range(dt.nrows)]


def _write_values(dt, name: str, values: Mapping[int, float]) -> None:
    """Write values (by 0-based row) into a numeric column, creating it if needed."""
    if name in _column_names(dt):
        column = dt[name]
    else:
        numeric = None
        try:
            import jmp  # only meaningful inside JMP

            numeric = jmp.DataType.Numeric
        except Exception:  # noqa: BLE001 - no JMP, or an unrelated module named jmp
            pass
        column = dt.new_column(name, numeric) if numeric is not None else dt.new_column(name)
    for row, value in values.items():
        column[row] = float(value)


__all__ = [
    "LAB_VARIABLE",
    "catalog_json",
    "describe_lab",
    "factor_definitions",
    "lab_config",
    "lab_json",
    "measure_columns",
    "run_table",
    "suggest_columns",
    "suggest_table",
    "to_json",
    "truth_columns",
    "truth_table",
]
