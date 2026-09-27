"""Design and results tables as text, so a design made in JMP can be measured here and go back.

Reading accepts CSV, tab-separated text (what JMP puts on the clipboard when
you copy a table) or semicolon-separated text. Columns are matched to factors
by name or label, ignoring case, spaces, underscores and units in parentheses,
so "RF power", "rf_power_w" and "RF Power (W)" all mean the same factor. The
response column is matched the same way, and JMP's default response name "Y"
counts too. Other columns (JMP's "Pattern", run numbers, notes) are kept as
they are and otherwise ignored.

The table is the ledger. Rows that already have a response were measured
earlier: they count against the budget, and new rows continue the evaluation
count after them. So measuring a design in two sittings gives exactly the
numbers of measuring it in one, and an augmented design picks up where the
first left off.

A factor the table leaves out is held constant, at a value given in `hold`
or else at the center of its range (with a warning).
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from seqopt.oracle import BudgetExhausted

from .oracle import HartmannOracle

DELIMITERS = "\t,;"
RESPONSE_ALIASES = ("y", "response", "nonuniformity")
QUIET_COLUMNS = ("run", "pattern")  # ignored without a warning


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]


@dataclass
class MeasureResult:
    table: Table  # the input table with the response column filled in
    y: np.ndarray  # every row's response, in table order
    measured: int  # rows measured now
    prior: int  # rows that already had a response
    rows_measured: list[int] = field(default_factory=list)  # 0-based row numbers measured now
    warnings: list[str] = field(default_factory=list)


# --- Text in and out -------------------------------------------------------------


def normalize(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", str(text))  # drop units like "(W)"
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_table(text: str) -> Table:
    """Header and rows from delimited text. Blank lines and all-blank rows are skipped."""
    text = text.lstrip("﻿")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("The table needs a header row and at least one row.")
    sample = "\n".join(lines[:20])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:
        counts = {d: lines[0].count(d) for d in DELIMITERS}
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            # A single-column table has no separator at all.
            return Table([lines[0].strip()], [[line.strip()] for line in lines[1:]])
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    header = [h.strip() for h in next(reader)]
    width = len(header)
    rows = []
    for row in reader:
        cells = [cell.strip() for cell in row]
        if any(cells):
            rows.append(cells + [""] * (width - len(cells)))
    if not rows:
        raise ValueError("The table has a header row but no data rows.")
    return Table(header, rows)


def format_table(table: Table) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(table.header)
    writer.writerows(table.rows)
    return out.getvalue()


def read_text(path: str) -> str:
    """A table file's text: UTF-8 (with or without a byte-order mark), or Windows-1252 as JMP may write."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def _fmt(value: float) -> str:
    return format(float(value), ".6g")


# --- Columns -----------------------------------------------------------------------


def factor_header(factor: Mapping) -> str:
    unit = factor["unit"]
    return f"{factor['label']} ({unit})" if unit not in ("unit", "coded", "") else factor["label"]


def response_header(oracle: HartmannOracle) -> str:
    r = oracle.response
    return f"{r.label} ({r.unit})" if r.unit else r.name


def _lookup(oracle: HartmannOracle) -> dict[str, int | None]:
    """Normalized column name -> factor index, or None for the response."""
    table: dict[str, int | None] = {}
    for j, f in enumerate(oracle.factors()):
        table[normalize(f["name"])] = j
        table[normalize(f["label"])] = j
    r = oracle.response
    for alias in (r.name, r.label, *RESPONSE_ALIASES):
        table.setdefault(normalize(alias), None)
    return table


def match_columns(header: list[str], oracle: HartmannOracle) -> tuple[dict[int, int], int | None, list[str]]:
    """(factor index -> column, response column or None, ignored column names)."""
    lookup = _lookup(oracle)
    factors: dict[int, int] = {}
    response = None
    ignored, duplicates = [], []
    for c, name in enumerate(header):
        key = normalize(name)
        if key not in lookup:
            ignored.append(name)
        elif lookup[key] is None:
            if response is not None:
                duplicates.append(name)
            response = c
        elif lookup[key] in factors:
            duplicates.append(name)
        else:
            factors[lookup[key]] = c
    if duplicates:
        raise ValueError(f"These columns repeat a factor or the response: {duplicates}")
    return factors, response, ignored


def _hold_values(hold: Mapping | None, oracle: HartmannOracle) -> dict[int, float]:
    lookup = _lookup(oracle)
    out = {}
    for key, value in (hold or {}).items():
        j = lookup.get(normalize(key))
        if j is None:
            raise ValueError(f"{key!r} in hold is not a factor")
        out[j] = float(value)
    return out


def _number(cell: str, row: int, column: str) -> float:
    try:
        value = float(cell)
    except ValueError:
        raise ValueError(f"Row {row + 1}, column {column!r}: {cell!r} is not a number.") from None
    if not np.isfinite(value):
        raise ValueError(f"Row {row + 1}, column {column!r}: {cell!r} is not a finite number.")
    return value


# --- Designs and measuring --------------------------------------------------------------


def design_table(points, oracle: HartmannOracle) -> Table:
    """A table to measure: a run number, the factors in the oracle's units, an empty response column."""
    X = np.atleast_2d(np.asarray(points, dtype=float))
    header = ["Run"] + [factor_header(f) for f in oracle.factors()] + [response_header(oracle)]
    rows = [[str(i + 1)] + [_fmt(v) for v in x] + [""] for i, x in enumerate(X)]
    return Table(header, rows)


def read_points(table: Table, oracle: HartmannOracle, hold: Mapping | None = None):
    """Points (n, 6) in the oracle's units, responses (NaN where empty), the response column, and warnings."""
    factors, response_col, ignored = match_columns(table.header, oracle)
    held = _hold_values(hold, oracle)
    names = [f["label"] for f in oracle.factors()]
    both = sorted(set(factors) & set(held))
    if both:
        raise ValueError(f"{', '.join(names[j] for j in both)} is in the table and also held; choose one.")
    if not factors:
        raise ValueError(
            "No column matched a factor. Column names must be factor names or labels, for example "
            f"{factor_header(oracle.factors()[0])!r}. Found: {table.header}"
        )
    warnings = []
    loud = [c for c in ignored if normalize(c) not in QUIET_COLUMNS]
    if loud:
        warnings.append(f"Ignored columns that are not a factor or the response: {', '.join(loud)}.")
    center = {j: f["center"] for j, f in enumerate(oracle.factors())}
    for j in range(oracle.dim):
        if j not in factors and j not in held:
            held[j] = center[j]
            warnings.append(f"{names[j]} is not in the table, so it is held at its center, {center[j]:g}.")

    n = len(table.rows)
    X = np.empty((n, oracle.dim))
    y = np.full(n, np.nan)
    for r, row in enumerate(table.rows):
        for j in range(oracle.dim):
            if j in factors:
                c = factors[j]
                if row[c] == "":
                    raise ValueError(f"Row {r + 1}, column {table.header[c]!r} is empty.")
                X[r, j] = _number(row[c], r, table.header[c])
            else:
                X[r, j] = held[j]
        if response_col is not None and row[response_col] != "":
            y[r] = _number(row[response_col], r, table.header[response_col])
    return X, y, response_col, warnings


def measure_table(table: Table, oracle: HartmannOracle, hold: Mapping | None = None) -> MeasureResult:
    """Measure every row that has no response yet; the rows that do are the run's history.

    The oracle must be fresh: the table is the whole ledger. Everything is
    checked before anything is measured, and a table that would overrun the
    budget is refused whole.
    """
    if oracle.budget.used or oracle.ledger:
        raise ValueError("measure_table needs a fresh oracle; the table itself is the ledger")
    X, y, response_col, warnings = read_points(table, oracle, hold)
    done = ~np.isnan(y)
    todo = np.flatnonzero(~done)
    total = oracle.budget.total
    if total is not None and len(y) > total:
        raise BudgetExhausted(
            f"the table has {len(y)} rows ({int(done.sum())} already measured) but the budget is {total}"
        )
    if done.any():
        oracle.record(X[done], y[done])
    if len(todo):
        oracle.evaluate(X[todo])  # validates every point before charging anything
        y[todo] = [e.y for e in oracle.ledger[-len(todo):]]

    header = list(table.header)
    rows = [list(row) for row in table.rows]
    if response_col is None:
        header.append(response_header(oracle))
        response_col = len(header) - 1
        for row in rows:
            row.append("")
    for r in todo:
        rows[r][response_col] = _fmt(y[r])
    return MeasureResult(Table(header, rows), y, len(todo), int(done.sum()), [int(r) for r in todo], warnings)
