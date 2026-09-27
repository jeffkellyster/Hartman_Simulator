import numpy as np
import pytest

from seqopt import designs
from seqopt.rsm import QuadraticRSM, n_terms, term_names


def _quadratic(U):
    c = 2 * U - 1
    return 3 + 2 * c[:, 0] - c[:, 1] + 0.5 * c[:, 0] * c[:, 2] + 1.5 * c[:, 1] ** 2


def test_an_exact_quadratic_is_recovered():
    U = designs.box_behnken(3).unit
    model = QuadraticRSM().fit(U, _quadratic(U))
    coef = model.describe(["A", "B", "C"])["coefficients"]
    assert coef["Intercept"] == pytest.approx(3)
    assert coef["A"] == pytest.approx(2) and coef["B"] == pytest.approx(-1)
    assert coef["A*C"] == pytest.approx(0.5) and coef["B^2"] == pytest.approx(1.5)
    assert coef["C^2"] == pytest.approx(0, abs=1e-10)
    assert model.r2 == pytest.approx(1)
    Ut = np.random.default_rng(0).random((10, 3))
    assert model.predict(Ut)[0] == pytest.approx(_quadratic(Ut))


def test_term_count_and_names():
    assert n_terms(6) == 28
    assert term_names(["a", "b"]) == ["Intercept", "a", "b", "a*b", "a^2", "b^2"]


def test_too_few_runs_is_flagged_and_has_no_error_estimate():
    U = np.random.default_rng(1).random((20, 6))
    model = QuadraticRSM().fit(U, U.sum(axis=1))
    assert model.underdetermined
    assert np.isnan(model.predict(U[:2])[1]).all()


def test_standard_errors_grow_away_from_the_data():
    U = designs.box_behnken(3).unit
    y = _quadratic(U) + 0.1 * np.random.default_rng(2).standard_normal(len(U))
    model = QuadraticRSM().fit(U, y)
    _, sd_center = model.predict([[0.5, 0.5, 0.5]])
    _, sd_corner = model.predict([[1.0, 1.0, 1.0]])
    assert 0 < sd_center[0] < sd_corner[0]
