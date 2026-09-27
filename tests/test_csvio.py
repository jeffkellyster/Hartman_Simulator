import numpy as np
import pytest

from hartmann import HartmannOracle, NoiseModel, csvio, process
from seqopt import BudgetExhausted, designs

NOISY = dict(noise=NoiseModel(sd=0.1), seed=5)


def _oracle(**kw):
    return HartmannOracle(**{**NOISY, **kw})


def _design_text(runs=10, seed=0, units="process"):
    oracle = HartmannOracle(units=units, seed=0)
    points = process.from_unit(designs.latin_hypercube(6, runs, seed=seed).unit, units)
    return csvio.format_table(csvio.design_table(points, oracle))


def _responses(table):
    col = len(table.header) - 1
    return [float(r[col]) for r in table.rows]


# --- Reading and writing ------------------------------------------------------------


def test_a_design_table_names_every_factor_with_its_unit():
    header = csvio.parse_table(_design_text()).header
    assert header[0] == "Run"
    assert header[1] == "Susceptor temperature (°C)"
    assert header[5] == "RF power (W)"
    assert header[-1] == "Thickness non-uniformity (%)"
    unit_header = csvio.parse_table(_design_text(units="unit")).header
    assert unit_header == ["Run", "x1", "x2", "x3", "x4", "x5", "x6", "y"]


def test_tables_parse_from_csv_tabs_or_semicolons_with_or_without_a_bom():
    for text in ("a,b\n1,2\n", "a\tb\n1\t2\n", "a;b\n1;2\n", "﻿a,b\n1,2\n"):
        table = csvio.parse_table(text)
        assert table.header == ["a", "b"] and table.rows == [["1", "2"]]
    assert csvio.parse_table("a,b\n1,2\n,,\n\n").rows == [["1", "2"]]
    with pytest.raises(ValueError, match="header row"):
        csvio.parse_table("a,b\n")


def test_columns_match_by_name_or_label_ignoring_case_spacing_and_units():
    oracle = _oracle()
    header = ["Pattern", "rf_power_w", "RF Power (W)"]
    with pytest.raises(ValueError, match="repeat"):
        csvio.match_columns(header, oracle)
    factors, response, ignored = csvio.match_columns(
        ["Pattern", "SUSCEPTOR TEMPERATURE", "pressure_torr", "Y"], oracle
    )
    assert factors == {0: 1, 1: 2} and response == 3 and ignored == ["Pattern"]


def test_cp1252_files_from_jmp_are_read(tmp_path):
    path = tmp_path / "jmp.csv"
    path.write_bytes("Susceptor temperature (°C),RF power (W)\n300,400\n".encode("cp1252"))
    assert csvio.parse_table(csvio.read_text(str(path))).header[0] == "Susceptor temperature (°C)"


# --- Measuring ------------------------------------------------------------------------


def test_measuring_fills_the_response_and_matches_the_oracle():
    table = csvio.parse_table(_design_text())
    result = csvio.measure_table(table, _oracle())
    assert result.measured == 10 and result.prior == 0
    X, _, _, _ = csvio.read_points(table, _oracle())
    assert _responses(result.table) == pytest.approx(_oracle().evaluate(X), abs=5e-6)


def test_two_sittings_give_the_same_numbers_as_one():
    whole = csvio.measure_table(csvio.parse_table(_design_text(15)), _oracle()).table

    first = csvio.parse_table(_design_text(15))
    later_rows = first.rows[10:]
    first.rows = first.rows[:10]
    sitting1 = csvio.measure_table(first, _oracle()).table
    sitting1.rows += later_rows
    sitting2 = csvio.measure_table(csvio.parse_table(csvio.format_table(sitting1)), _oracle())

    assert sitting2.prior == 10 and sitting2.measured == 5
    assert sitting2.table.rows == whole.rows


def test_the_budget_covers_rows_already_measured():
    table = csvio.measure_table(csvio.parse_table(_design_text(8)), _oracle(budget=10)).table
    table.rows += csvio.parse_table(_design_text(3, seed=1)).rows
    with pytest.raises(BudgetExhausted, match="11 rows"):
        csvio.measure_table(table, _oracle(budget=10))


def test_missing_factors_are_held_at_center_or_at_a_given_value():
    text = "RF power (W),Electrode spacing (mm)\n100,8\n800,25\n"
    result = csvio.measure_table(csvio.parse_table(text), _oracle())
    assert len(result.warnings) == 4 and "held at its center, 350" in result.warnings[0]
    X, _, _, _ = csvio.read_points(csvio.parse_table(text), _oracle(), hold={"Susceptor temperature": 300})
    assert X[0, 0] == 300 and X[0, 1] == 2.75
    with pytest.raises(ValueError, match="also held"):
        csvio.read_points(csvio.parse_table(text), _oracle(), hold={"rf_power_w": 300})
    with pytest.raises(ValueError, match="not a factor"):
        csvio.read_points(csvio.parse_table(text), _oracle(), hold={"flux capacitor": 1})


def test_a_jmp_style_table_keeps_its_extra_columns():
    text = ("Pattern\tSusceptor temperature\tChamber pressure\tDeposition time\tSiH4 flow\tRF power\t"
            "Electrode spacing\tY\n"
            "−−−−−−\t250\t0.5\t30\t50\t100\t8\t\n"
            "++++++\t450\t5\t180\t400\t800\t25\t\n")
    result = csvio.measure_table(csvio.parse_table(text), _oracle())
    assert result.table.header[-1] == "Y"
    assert [r[0] for r in result.table.rows] == ["−−−−−−", "++++++"]
    assert result.warnings == []
    assert all(r[-1] for r in result.table.rows)


def test_coded_tables_measure_in_coded_units():
    text = "x1,x2,x3,x4,x5,x6\n-1,-1,-1,-1,-1,-1\n0,0,0,0,0,0\n"
    oracle = HartmannOracle(units="coded", seed=0)
    result = csvio.measure_table(csvio.parse_table(text), oracle)
    assert result.table.header[-1] == "y"
    assert _responses(result.table)[1] == pytest.approx(oracle.true_response([0.0] * 6), abs=5e-6)


def test_bad_tables_are_refused_before_anything_is_measured():
    oracle = _oracle(budget=5)
    bad_value = "RF power (W)\n400\n9000\n"
    with pytest.raises(ValueError, match="RF power"):
        csvio.measure_table(csvio.parse_table(bad_value), oracle)
    with pytest.raises(ValueError, match="not a number"):
        csvio.measure_table(csvio.parse_table("RF power (W)\nlots\n"), _oracle())
    with pytest.raises(ValueError, match="is empty"):
        csvio.measure_table(csvio.parse_table("RF power (W),SiH4 flow (sccm)\n400,\n"), _oracle())
    with pytest.raises(ValueError, match="No column matched"):
        csvio.measure_table(csvio.parse_table("a,b\n1,2\n"), _oracle())
    with pytest.raises(ValueError, match="fresh oracle"):
        used = _oracle()
        used.evaluate(process.from_unit([0.5] * 6))
        csvio.measure_table(csvio.parse_table(_design_text()), used)


def test_measured_responses_are_rounded_for_reading_but_kept_exact_in_the_result():
    result = csvio.measure_table(csvio.parse_table(_design_text(3)), _oracle())
    written = _responses(result.table)
    assert written == pytest.approx(result.y, abs=5e-6)
    assert np.isfinite(result.y).all()
