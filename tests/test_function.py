import numpy as np
import pytest
from scipy.optimize import minimize

from hartmann import F_STAR, LOCAL_MINIMA, X_STAR, gradient, hartmann6, minimum_value

# Reference values from https://www.sfu.ca/~ssurjano/hart6.html
SFU_F_STAR = -3.32237
SFU_X_STAR = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)


def _hessian(x, form="standard", h=1e-6):
    """Central differences of the analytic gradient."""
    x = np.asarray(x, dtype=float)
    cols = []
    for j in range(6):
        step = np.zeros(6)
        step[j] = h
        cols.append((gradient(x + step, form) - gradient(x - step, form)) / (2 * h))
    H = np.column_stack(cols)
    return 0.5 * (H + H.T)


# --- The published optimum --------------------------------------------------


def test_global_optimum_matches_the_sfu_reference():
    assert hartmann6(SFU_X_STAR) == pytest.approx(SFU_F_STAR, abs=1e-5)
    assert F_STAR == SFU_F_STAR
    assert tuple(X_STAR) == SFU_X_STAR


def test_the_published_optimum_is_a_true_minimum():
    x = np.array(LOCAL_MINIMA[0].x)
    f0 = hartmann6(x)
    assert np.abs(gradient(x)).max() < 1e-6
    rng = np.random.default_rng(0)
    for radius in (1e-4, 1e-3, 1e-2):
        nearby = np.clip(x + rng.uniform(-radius, radius, (500, 6)), 0, 1)
        assert hartmann6(nearby).min() >= f0
    # The published x* is the polished minimizer rounded to about 5 decimals.
    assert np.abs(X_STAR - x).max() < 1e-5


def test_nothing_in_a_large_random_sample_beats_the_optimum():
    X = np.random.default_rng(1).random((200_000, 6))
    assert hartmann6(X).min() > hartmann6(LOCAL_MINIMA[0].x)


# --- Local minima -----------------------------------------------------------


@pytest.mark.parametrize("minimum", LOCAL_MINIMA, ids=lambda m: m.label)
def test_each_local_minimum_is_stationary_with_a_positive_definite_hessian(minimum):
    x = np.array(minimum.x)
    assert np.all((x > 0) & (x < 1))
    assert hartmann6(x) == pytest.approx(minimum.f, abs=1e-9)
    assert np.abs(gradient(x)).max() < 1e-6
    assert np.linalg.eigvalsh(_hessian(x)).min() > 0


def test_a_multistart_search_finds_the_listed_minima_and_nothing_else():
    starts = np.random.default_rng(6).random((80, 6))
    hits = [0] * len(LOCAL_MINIMA)
    for start in starts:
        result = minimize(hartmann6, start, jac=gradient, bounds=[(0, 1)] * 6, method="L-BFGS-B",
                          options={"ftol": 1e-13, "gtol": 1e-9})
        if np.abs(gradient(result.x)).max() > 1e-5:
            continue  # stalled start, not a minimum
        matches = [i for i, m in enumerate(LOCAL_MINIMA) if np.abs(result.x - m.x).max() < 1e-4]
        assert matches, f"found an unlisted minimum {result.fun:.6f} at {np.round(result.x, 4)}"
        hits[matches[0]] += 1
    assert all(hits)


# --- Forms ------------------------------------------------------------------


def test_each_form_has_its_documented_minimum_value():
    assert minimum_value("standard") == pytest.approx(-3.322368, abs=1e-6)
    assert minimum_value("rescaled") == pytest.approx(-1.948803, abs=1e-6)
    assert minimum_value("sfu_rescaled") == pytest.approx(-3.042458, abs=1e-6)


def test_rescaled_form_has_mean_zero_and_variance_one():
    y = hartmann6(np.random.default_rng(2012).random((100_000, 6)), "rescaled")
    assert abs(y.mean()) < 0.03
    assert y.var() == pytest.approx(1.0, abs=0.05)


def test_the_formula_as_printed_on_the_sfu_page_is_not_mean_zero_variance_one():
    # Documents why "rescaled" means the log form (see function.py).
    y = hartmann6(np.random.default_rng(2012).random((100_000, 6)), "sfu_rescaled")
    assert y.mean() == pytest.approx(-1.46, abs=0.02)
    assert y.var() < 0.1


@pytest.mark.parametrize("form", ["standard", "rescaled", "sfu_rescaled"])
def test_gradient_matches_finite_differences(form):
    rng = np.random.default_rng(3)
    for x in rng.random((5, 6)):
        numeric = np.array([
            (hartmann6(x + h, form) - hartmann6(x - h, form)) / 2e-6 for h in np.eye(6) * 1e-6
        ])
        assert gradient(x, form) == pytest.approx(numeric, rel=1e-5, abs=1e-8)


# --- Shapes and validation --------------------------------------------------


def test_a_point_gives_a_float_and_a_batch_gives_an_array():
    X = np.random.default_rng(4).random((7, 6))
    batch = hartmann6(X)
    assert batch.shape == (7,)
    assert isinstance(hartmann6(X[0]), float)
    assert [hartmann6(x) for x in X] == pytest.approx(batch, abs=0)
    assert gradient(X).shape == (7, 6)
    assert gradient(X[0]).shape == (6,)


def test_bad_shapes_and_forms_are_refused():
    with pytest.raises(ValueError, match="shape"):
        hartmann6([0.5] * 5)
    with pytest.raises(ValueError, match="shape"):
        hartmann6(np.zeros((2, 3, 6)))
    with pytest.raises(ValueError, match="unknown form"):
        hartmann6(X_STAR, form="log")


def test_constants_are_read_only():
    with pytest.raises(ValueError):
        X_STAR[0] = 0.5
