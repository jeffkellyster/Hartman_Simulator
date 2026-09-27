import numpy as np
import pytest

from hartmann import F_STAR, LOCAL_MINIMA, HartmannOracle, NoiseModel, Scenario, hartmann6, process
from seqopt import Budget, BudgetExhausted, Oracle

NOISY = NoiseModel(sd=0.2, hetero=1.0)


def _design(n=8, seed=0, units="process"):
    return process.from_unit(np.random.default_rng(seed).random((n, 6)), units)


# --- What the student sees ----------------------------------------------------


def test_default_oracle_speaks_pecvd_process_units():
    o = HartmannOracle(seed=0)
    assert o.response.name == "non_uniformity_pct"
    assert np.array_equal(o.bounds[:, 0], process.LOW) and np.array_equal(o.bounds[:, 1], process.HIGH)
    assert [f["label"] for f in o.factors()][4] == "RF power"
    y = o.evaluate([f["center"] for f in o.factors()])
    assert isinstance(y, float) and 1.0 < y < 6.0


def test_unit_and_coded_oracles_return_the_function_itself():
    x = np.array(LOCAL_MINIMA[0].x)
    assert HartmannOracle(units="unit", seed=0).evaluate(x) == pytest.approx(hartmann6(x))
    assert HartmannOracle(units="coded", seed=0).evaluate(2 * x - 1) == pytest.approx(hartmann6(x))
    assert HartmannOracle(units="unit", form="rescaled", seed=0).evaluate(x) == pytest.approx(-1.948803, abs=1e-6)
    assert [f["name"] for f in HartmannOracle(units="unit").factors()] == ["x1", "x2", "x3", "x4", "x5", "x6"]


def test_process_units_require_the_standard_form():
    with pytest.raises(ValueError, match="standard form"):
        HartmannOracle(units="process", form="rescaled")


def test_the_oracle_satisfies_the_kernel_contract():
    assert isinstance(HartmannOracle(seed=0), Oracle)


# --- Noise and reproducibility ------------------------------------------------


def test_without_noise_a_measurement_is_the_true_response():
    o = HartmannOracle(seed=1)
    X = _design()
    assert np.array_equal(o.evaluate(X), o.true_response(X))


def test_the_same_seed_reproduces_a_run_and_another_seed_does_not():
    X = _design()
    a = HartmannOracle(noise=NOISY, seed=42).evaluate(X)
    b = HartmannOracle(noise=NOISY, seed=42).evaluate(X)
    c = HartmannOracle(noise=NOISY, seed=43).evaluate(X)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_batching_does_not_change_the_measurements():
    X = _design()
    whole = HartmannOracle(noise=NOISY, seed=7).evaluate(X)
    one_at_a_time = HartmannOracle(noise=NOISY, seed=7)
    assert np.array_equal(whole, [one_at_a_time.evaluate(x) for x in X])


def test_repeating_a_point_gives_a_fresh_replicate():
    o = HartmannOracle(noise=NOISY, seed=3)
    x = _design(1)[0]
    first, second = o.evaluate(x), o.evaluate(x)
    assert first != second
    assert o.ledger[0].y_true == o.ledger[1].y_true


def test_a_run_replays_exactly_from_its_config():
    o = HartmannOracle(noise=NOISY, seed=11, budget=20, scenario=5)
    o.evaluate(_design(6, seed=1))
    o.evaluate(_design(3, seed=2))
    again = o.replay()
    assert again.ledger == o.ledger
    assert again.budget.used == o.budget.used == 9
    assert HartmannOracle.from_config(o.config).config == o.config


def test_a_seed_is_picked_and_recorded_when_not_given():
    o = HartmannOracle(noise=NOISY)
    assert isinstance(o.seed, int)
    X = _design(4)
    assert np.array_equal(o.evaluate(X), HartmannOracle.from_config(o.config).evaluate(X))


def test_bad_seeds_are_refused():
    for seed in (-1, 1.5, True):
        with pytest.raises(ValueError, match="seed"):
            HartmannOracle(seed=seed)


# --- Budget ---------------------------------------------------------------------


def test_each_point_costs_one_evaluation_and_an_overrun_is_refused_whole():
    o = HartmannOracle(seed=0, budget=5)
    o.evaluate(_design(3))
    assert o.budget.used == 3 and o.budget.remaining == 2
    with pytest.raises(BudgetExhausted, match="only 2 of 5 remain"):
        o.evaluate(_design(3))
    assert o.budget.used == 3 and len(o.ledger) == 3
    o.evaluate(_design(2))
    with pytest.raises(BudgetExhausted):
        o.evaluate(_design(1)[0])
    assert [e.k for e in o.ledger] == [0, 1, 2, 3, 4]


def test_recorded_measurements_count_and_later_ones_continue_the_stream():
    X = _design(5)
    straight = HartmannOracle(noise=NOISY, seed=4).evaluate(X)
    resumed = HartmannOracle(noise=NOISY, seed=4, budget=5)
    resumed.record(X[:3], straight[:3])
    assert resumed.budget.used == 3 and [e.k for e in resumed.ledger] == [0, 1, 2]
    assert np.array_equal(resumed.evaluate(X[3:]), straight[3:])
    with pytest.raises(BudgetExhausted):
        resumed.record(X[:1], straight[:1])
    with pytest.raises(ValueError, match="responses"):
        HartmannOracle(seed=0).record(X[:2], straight[:1])


def test_invalid_points_are_refused_before_anything_is_charged():
    o = HartmannOracle(seed=0, budget=5)
    X = _design(2)
    X[1, 0] = 1000.0
    with pytest.raises(ValueError, match="Susceptor temperature"):
        o.evaluate(X)
    assert o.budget.used == 0 and o.ledger == []


def test_budget_validation():
    assert Budget().remaining is None
    Budget(None).charge(10_000)
    with pytest.raises(ValueError):
        Budget(-1)
    with pytest.raises(ValueError):
        Budget(2.5)
    with pytest.raises(ValueError):
        Budget(3).charge(-1)


# --- Truth and blind scenarios ----------------------------------------------------


def test_the_process_optimum_is_about_one_percent_non_uniformity():
    o = HartmannOracle(seed=0)
    best = o.optimum()
    assert best["y"] == pytest.approx(6.0 + 1.5 * F_STAR, abs=1e-5)
    assert np.all((best["x"] >= process.LOW) & (best["x"] <= process.HIGH))


def test_a_blind_scenario_moves_the_optimum_but_keeps_its_value():
    plain = HartmannOracle(units="unit", seed=0)
    blind = HartmannOracle(units="unit", seed=0, scenario=8)
    assert np.abs(np.subtract(plain.optimum()["x"], blind.optimum()["x"])).max() > 0.05
    assert blind.optimum()["y"] == pytest.approx(plain.optimum()["y"], abs=1e-12)
    assert blind.scenario.to_hartmann(blind.optimum()["x"]) == pytest.approx(LOCAL_MINIMA[0].x, abs=1e-12)
    X = np.random.default_rng(0).random((50_000, 6))
    assert blind.true_response(X).min() > blind.optimum()["y"]


def test_local_minima_are_reported_in_the_students_coordinates():
    o = HartmannOracle(seed=0, scenario=Scenario.from_seed(2))
    minima = o.local_minima()
    assert [m["label"] for m in minima] == [m.label for m in LOCAL_MINIMA]
    for m, ref in zip(minima, LOCAL_MINIMA):
        assert m["y"] == pytest.approx(6.0 + 1.5 * ref.f, abs=1e-8)
        assert o.true_response(m["x"]) == pytest.approx(m["y"])


def test_best_observed_uses_the_observed_values():
    o = HartmannOracle(seed=0)
    assert o.best_observed() is None
    o.evaluate(_design(10))
    assert o.best_observed().y == min(e.y for e in o.ledger)


def test_unknown_config_keys_are_refused():
    with pytest.raises(ValueError, match="unknown oracle settings"):
        HartmannOracle.from_config({"units": "unit", "sigma": 1})
