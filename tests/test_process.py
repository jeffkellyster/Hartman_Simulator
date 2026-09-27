import numpy as np
import pytest

from hartmann import FACTORS, F_STAR, RESPONSE, process

PECVD = [
    ("susceptor_temp_c", "Susceptor temperature", "°C", 250, 450),
    ("pressure_torr", "Chamber pressure", "Torr", 0.5, 5.0),
    ("dep_time_s", "Deposition time", "s", 30, 180),
    ("sih4_flow_sccm", "SiH4 flow", "sccm", 50, 400),
    ("rf_power_w", "RF power", "W", 100, 800),
    ("electrode_spacing_mm", "Electrode spacing", "mm", 8, 25),
]


def test_factor_table_is_the_agreed_pecvd_process():
    assert [(f.name, f.label, f.unit, f.low, f.high) for f in FACTORS] == PECVD
    assert [row["name"] for row in process.table()] == [p[0] for p in PECVD]


def test_coded_levels_land_exactly_on_low_center_and_high():
    coded = np.array([[-1.0] * 6, [0.0] * 6, [1.0] * 6])
    actual = process.from_unit(process.to_unit(coded, "coded"), "process")
    assert np.array_equal(actual[0], process.LOW)
    assert np.array_equal(actual[1], [f.center for f in FACTORS])
    assert np.array_equal(actual[2], process.HIGH)


@pytest.mark.parametrize("units", ["process", "unit", "coded"])
def test_every_unit_system_round_trips(units):
    U = np.random.default_rng(1).random((50, 6))
    X = process.from_unit(U, units)
    assert process.to_unit(X, units) == pytest.approx(U, abs=1e-12)


def test_a_single_point_keeps_its_shape():
    point = process.from_unit([0.5] * 6, "process")
    assert point.shape == (6,)
    assert process.to_unit(point, "process").shape == (6,)


def test_out_of_range_values_are_refused_with_the_factor_named():
    x = [f.center for f in FACTORS]
    x[4] = 900.0
    with pytest.raises(ValueError, match=r"RF power \(W\) = 900 is outside"):
        process.to_unit(x, "process")
    with pytest.raises(ValueError, match="outside its coded range"):
        process.to_unit([1.5] + [0.0] * 5, "coded")


def test_tiny_rounding_outside_a_range_is_accepted_and_clipped():
    x = [f.high for f in FACTORS]
    x[0] += 1e-12
    assert process.to_unit(x, "process")[0] == 1.0


def test_missing_values_and_unknown_units_are_refused():
    with pytest.raises(ValueError, match="missing"):
        process.to_unit([np.nan] + [0.5] * 5, "unit")
    with pytest.raises(ValueError, match="unknown units"):
        process.to_unit([0.5] * 6, "kelvin")


def test_response_is_non_uniformity_percent_and_inverts():
    assert RESPONSE.name == "non_uniformity_pct" and RESPONSE.unit == "%"
    assert RESPONSE.from_f(F_STAR) == pytest.approx(1.0164, abs=1e-4)
    assert RESPONSE.from_f(0.0) == 6.0
    assert RESPONSE.to_f(RESPONSE.from_f(-2.5)) == pytest.approx(-2.5)
