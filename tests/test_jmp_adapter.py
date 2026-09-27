import json
import math

import numpy as np
import pytest

from hartmann import HartmannOracle, csvio, jmp_adapter, process
from seqopt import BudgetExhausted, designs

NAN = float("nan")


class FakeColumn:
    def __init__(self, name, values):
        self.name, self.values = name, list(values)

    def __getitem__(self, i):
        return self.values[i]

    def __setitem__(self, i, value):
        self.values[i] = value

    def __len__(self):
        return len(self.values)


class FakeTable:
    """Just enough of JMP 18's jmp.DataTable for the adapter."""

    def __init__(self, columns):
        self.columns = [FakeColumn(n, v) for n, v in columns.items()]

    @property
    def nrows(self):
        return len(self.columns[0])

    @property
    def ncols(self):
        return len(self.columns)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.columns[key]
        return next(c for c in self.columns if c.name == key)

    def new_column(self, name, *_):
        column = FakeColumn(name, [NAN] * self.nrows)
        self.columns.append(column)
        return column

    def column(self, name):
        return self[name].values


def jmp_design_table(runs=8, seed=0, response="Y", with_pattern=True):
    """A table as JMP's DOE makes it: labels as column names, a Pattern column, an empty response."""
    X = process.from_unit(designs.latin_hypercube(6, runs, seed=seed).unit, "process")
    columns = {}
    if with_pattern:
        columns["Pattern"] = ["+-" * 3] * runs
    for j, f in enumerate(process.FACTORS):
        columns[f.label] = list(X[:, j])
    if response:
        columns[response] = [NAN] * runs
    return FakeTable(columns)


LAB = jmp_adapter.lab_config(noise_sd=0.05, budget=20, seed=4)


# --- Lab settings ---------------------------------------------------------------------------------


def test_lab_config_records_a_seed_and_round_trips_as_json():
    lab = jmp_adapter.lab_config(noise_sd=0.1, budget=30, scenario=7)
    assert isinstance(lab["seed"], int) and lab["scenario"]["seed"] == 7
    assert jmp_adapter._config(json.dumps(lab)) == lab
    assert jmp_adapter.lab_config(budget=NAN)["budget"] is None  # JSL's missing value
    with pytest.raises(ValueError, match="no lab settings"):
        jmp_adapter._config("")


def test_describe_lab_reads_like_a_sentence():
    text = jmp_adapter.describe_lab(jmp_adapter.lab_config(noise_sd=0.05, hetero=1, budget=60, seed=3, scenario=2))
    assert text == ("PECVD process units; noise SD 0.05 %; noise grows x2 along RF power; budget 60 runs; seed 3; "
                    "blind scenario 2")


def test_factor_definitions_and_catalog():
    rows = jmp_adapter.factor_definitions()
    assert [r["label"] for r in rows][4] == "RF power" and rows[0]["low"] == 250
    assert json.loads(jmp_adapter.catalog_json())["process"] == "PECVD film deposition"


# --- Measuring a table ------------------------------------------------------------------------------


def test_measuring_fills_jmps_response_column_and_matches_the_oracle():
    dt = jmp_design_table()
    result = jmp_adapter.run_table(dt, LAB)
    assert result["response_column"] == "Y" and not result["new_column"]
    assert result["measured"] == 8 and result["prior"] == 0
    assert result["budget"] == {"used": 8, "total": 20, "remaining": 12}
    X = np.column_stack([dt.column(f.label) for f in process.FACTORS])
    expected = HartmannOracle.from_config(LAB).evaluate(X)
    assert dt.column("Y") == pytest.approx(expected)
    assert result["best"]["y"] == pytest.approx(min(expected))
    assert dt.column("Pattern")[0] == "+-" * 3  # other columns untouched
    json.loads(jmp_adapter.to_json(result))


def test_a_missing_response_column_is_created():
    dt = jmp_design_table(response=None)
    result = jmp_adapter.run_table(dt, LAB)
    assert result["new_column"] and result["response_column"] == "Thickness non-uniformity (%)"
    assert not any(math.isnan(v) for v in dt.column("Thickness non-uniformity (%)"))


def test_measured_rows_are_history_and_new_rows_continue_the_stream():
    whole = jmp_design_table(runs=10)
    jmp_adapter.run_table(whole, LAB)

    part = jmp_design_table(runs=10)
    first = FakeTable({c.name: c.values[:6] for c in part.columns})
    jmp_adapter.run_table(first, LAB)
    later = FakeTable({c.name: c.values[:6] + part[c.name].values[6:] for c in first.columns})
    result = jmp_adapter.run_table(later, LAB)
    assert result["prior"] == 6 and result["measured"] == 4
    assert later.column("Y") == pytest.approx(whole.column("Y"))


def test_the_budget_is_enforced_across_sessions():
    dt = jmp_design_table(runs=12)
    jmp_adapter.run_table(dt, {**LAB, "budget": 15})
    bigger = FakeTable({c.name: c.values + c.values[:4] for c in dt.columns})
    for i in range(12, 16):
        bigger["Y"][i] = NAN
    with pytest.raises(BudgetExhausted, match="16 rows"):
        jmp_adapter.run_table(bigger, {**LAB, "budget": 15})


def test_factors_without_a_column_are_held_at_center_with_a_warning():
    dt = FakeTable({"RF power (W)": [100.0, 800.0], "Y": [NAN, NAN]})
    result = jmp_adapter.run_table(dt, LAB)
    assert len(result["warnings"]) == 5 and "held at its center" in result["warnings"][0]
    assert result["factor_columns"] == ["RF power (W)"]


# --- Suggestions ---------------------------------------------------------------------------------------


def test_suggestions_come_back_as_rows_for_the_table():
    dt = jmp_design_table(runs=10)
    jmp_adapter.run_table(dt, LAB)
    result = jmp_adapter.suggest_table(dt, LAB, q=3)
    assert result["columns"] == [f.label for f in process.FACTORS]
    assert len(result["points"]) == 3 and len(result["acquisition"]) == 3
    low, high = process.LOW, process.HIGH
    assert all(np.all((np.array(p) >= low) & (np.array(p) <= high)) for p in result["points"])
    assert result["model"]["type"] == "gp"
    assert jmp_adapter.suggest_table(dt, LAB, q=3)["points"] == result["points"]  # repeatable
    rsm = jmp_adapter.suggest_table(dt, LAB, method="rsm")
    assert "not unique" in rsm["warnings"][0]


def test_suggestions_leave_room_for_rows_still_waiting_to_be_measured():
    dt = jmp_design_table(runs=10)
    jmp_adapter.run_table(dt, {**LAB, "budget": 14})
    waiting = FakeTable({c.name: c.values + c.values[:3] for c in dt.columns})
    for i in range(10, 13):
        waiting["Y"][i] = NAN
    assert len(jmp_adapter.suggest_table(waiting, {**LAB, "budget": 14}, q=5)["points"]) == 1
    with pytest.raises(ValueError, match="spoken for"):
        jmp_adapter.suggest_table(waiting, {**LAB, "budget": 13})


def test_suggestions_need_every_factor_and_two_measured_rows():
    with pytest.raises(ValueError, match="column for every factor"):
        jmp_adapter.suggest_table(FakeTable({"RF power (W)": [100.0], "Y": [5.0]}), LAB)
    dt = jmp_design_table(runs=3)
    with pytest.raises(ValueError, match="at least 2"):
        jmp_adapter.suggest_table(dt, LAB)


# --- Instructor view -----------------------------------------------------------------------------------


def test_truth_reports_the_optimum_and_scores_the_best_row():
    dt = jmp_design_table(runs=8)
    jmp_adapter.run_table(dt, LAB)
    t = jmp_adapter.truth_table(dt, LAB)
    assert t["optimum"]["y"] == pytest.approx(1.0164, abs=1e-4)
    assert len(t["local_minima"]) == 2 and t["unit"] == "%"
    ys = dt.column("Y")
    assert t["best"]["row"] == int(np.argmin(ys)) + 1
    assert t["best"]["gap"] > 0
    assert jmp_adapter.truth_table(jmp_design_table(runs=2), LAB)["best"] is None


def test_the_csv_and_jmp_paths_agree():
    dt = jmp_design_table(runs=6)
    table = csvio.parse_table(csvio.format_table(jmp_adapter._table({c.name: c.values for c in dt.columns})))
    via_csv = csvio.measure_table(table, HartmannOracle.from_config(LAB)).y
    jmp_adapter.run_table(dt, LAB)
    assert dt.column("Y") == pytest.approx(via_csv)
