import numpy as np
import pytest

from seqopt.acquisition import acquisition_values, incumbent, maximize, propose
from seqopt.gp import GaussianProcess
from seqopt.rsm import QuadraticRSM


def _branin_unit(U):
    x1, x2 = -5 + 15 * U[:, 0], 15 * U[:, 1]
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    return (x2 - b * x1**2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10


U0 = np.random.default_rng(0).random((12, 2))
GP = GaussianProcess(seed=0).fit(U0, _branin_unit(U0))


def test_expected_improvement_matches_monte_carlo():
    rng = np.random.default_rng(1)
    for mean, sd, best in [(1.0, 0.5, 1.2), (2.0, 1.0, 0.5), (0.0, 0.1, 0.0)]:
        draws = rng.normal(mean, sd, 400_000)
        mc = np.maximum(best - draws, 0).mean()
        assert acquisition_values("ei", [mean], [sd], best)[0] == pytest.approx(mc, rel=0.02, abs=1e-4)
        assert acquisition_values("pi", [mean], [sd], best)[0] == pytest.approx((draws < best).mean(), abs=0.005)


def test_simple_acquisitions_and_their_limits():
    mean, sd = np.array([1.0, 2.0]), np.array([0.5, 0.0])
    assert acquisition_values("exploit", mean, sd, 0).tolist() == [-1.0, -2.0]
    assert acquisition_values("explore", mean, sd, 0).tolist() == [0.5, 0.0]
    assert acquisition_values("ucb", mean, sd, 0, kappa=2).tolist() == [0.0, -2.0]
    # With no uncertainty, EI is the plain improvement and PI is 0 or 1.
    assert acquisition_values("ei", [1.0, 3.0], [0.0, 0.0], 2.0).tolist() == [1.0, 0.0]
    assert acquisition_values("pi", [1.0, 3.0], [0.0, 0.0], 2.0).tolist() == [1.0, 0.0]
    assert np.all(acquisition_values("ei", np.linspace(-5, 5, 50), np.full(50, 0.3), 0.0) >= 0)
    with pytest.raises(ValueError, match="unknown acquisition"):
        acquisition_values("thompson", mean, sd, 0)


def test_the_incumbent_is_the_best_predicted_or_observed_value():
    assert incumbent(GP, "observed") == pytest.approx(_branin_unit(U0).min())
    assert incumbent(GP, "mean") == pytest.approx(GP.predict(U0)[0].min())


def test_maximize_finds_the_peak_and_respects_min_distance():
    rng = np.random.default_rng(2)
    peak = np.array([0.3, 0.7])
    x, v = maximize(lambda X: -((X - peak) ** 2).sum(axis=1), 2, rng)
    assert x == pytest.approx(peak, abs=1e-3)
    x, _ = maximize(lambda X: -((X - peak) ** 2).sum(axis=1), 2, rng, X_obs=peak[None, :], min_distance=0.1)
    assert np.linalg.norm(x - peak) >= 0.1 - 1e-9


def test_proposals_repeat_from_a_seed_and_stay_in_the_cube():
    a, va = propose(GP, "ei", 1, np.random.default_rng(5))
    b, vb = propose(GP, "ei", 1, np.random.default_rng(5))
    assert np.array_equal(a, b) and np.array_equal(va, vb)
    assert a.shape == (1, 2) and np.all((a >= 0) & (a <= 1))


@pytest.mark.parametrize("rule", ["believer", "liar"])
def test_a_batch_spreads_out(rule):
    X, _ = propose(GP, "ei", 4, np.random.default_rng(6), batch=rule)
    assert X.shape == (4, 2)
    gaps = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))[np.triu_indices(4, 1)]
    assert gaps.min() > 1e-3


def test_exploit_on_a_quadratic_goes_to_its_minimum():
    U = np.random.default_rng(7).random((20, 2))
    model = QuadraticRSM().fit(U, ((U - [0.6, 0.2]) ** 2).sum(axis=1))
    X, _ = propose(model, "exploit", 1, np.random.default_rng(8))
    assert X[0] == pytest.approx([0.6, 0.2], abs=1e-3)


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="acquisition"):
        propose(GP, "magic")
    with pytest.raises(ValueError, match="batch rule"):
        propose(GP, "ei", 2, batch="optimist")
    with pytest.raises(ValueError, match="q must"):
        propose(GP, "ei", 0)
