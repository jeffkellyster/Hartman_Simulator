import numpy as np
import pytest

from hartmann import hartmann6
from seqopt.gp import GaussianProcess, GPParams

RNG = np.random.default_rng(0)
X = RNG.random((25, 6))
Y = hartmann6(X) + 0.01 * RNG.standard_normal(25)
THETA = np.r_[np.log([0.3, 0.5, 0.2, 0.8, 0.4, 0.6]), np.log(1.3), np.log(0.02)]


def _standardized(gp):
    gp._y_mean, gp._y_sd = Y.mean(), Y.std()
    return (Y - Y.mean()) / Y.std()


@pytest.mark.parametrize("kernel", ["matern52", "matern32", "rbf"])
def test_likelihood_gradient_matches_finite_differences(kernel):
    gp = GaussianProcess(kernel=kernel)
    ys = _standardized(gp)
    _, grad = gp._nll(THETA, X, ys)
    numeric = np.array([(gp._nll(THETA + h, X, ys)[0] - gp._nll(THETA - h, X, ys)[0]) / 2e-6
                        for h in np.eye(len(THETA)) * 1e-6])
    assert grad == pytest.approx(numeric, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("kernel", ["matern52", "matern32", "rbf"])
def test_agrees_with_scikit_learn_at_the_same_hyperparameters(kernel):
    sklearn = pytest.importorskip("sklearn.gaussian_process")
    k = sklearn.kernels
    ls = np.exp(THETA[:6])
    base = k.RBF(ls) if kernel == "rbf" else k.Matern(ls, nu=2.5 if kernel == "matern52" else 1.5)
    reference = sklearn.GaussianProcessRegressor(
        k.ConstantKernel(np.exp(THETA[6])) * base + k.WhiteKernel(np.exp(THETA[7])),
        optimizer=None, normalize_y=True, alpha=0.0,
    ).fit(X, Y)
    ours = GaussianProcess(kernel=kernel).fit_fixed(X, Y, GPParams.from_vector(THETA))
    assert ours.log_likelihood == pytest.approx(reference.log_marginal_likelihood_value_, rel=1e-6)
    Xt = np.random.default_rng(1).random((20, 6))
    mean, sd = ours.predict(Xt, include_noise=True)
    ref_mean, ref_sd = reference.predict(Xt, return_std=True)
    assert mean == pytest.approx(ref_mean, abs=1e-7)
    assert sd == pytest.approx(ref_sd, rel=1e-5)


def test_a_noiseless_fit_interpolates_the_data():
    Xn = RNG.random((15, 6))
    yn = hartmann6(Xn)
    gp = GaussianProcess(noise="none").fit(Xn, yn)
    mean, sd = gp.predict(Xn)
    assert mean == pytest.approx(yn, abs=1e-4)
    assert np.all(sd < 1e-2)


def test_the_noise_estimate_recovers_the_noise_level():
    x = np.linspace(0, 1, 120)[:, None]
    y = np.sin(6 * x[:, 0]) + 0.1 * np.random.default_rng(3).standard_normal(120)
    gp = GaussianProcess(seed=1).fit(x, y)
    assert gp.describe()["noise_sd"] == pytest.approx(0.1, rel=0.25)
    fixed = GaussianProcess(noise=0.1).fit(x, y)
    assert fixed.describe()["noise_sd"] == pytest.approx(0.1, rel=1e-6)


def test_an_irrelevant_input_gets_a_long_length_scale():
    Xr = np.random.default_rng(4).random((40, 2))
    y = np.sin(5 * Xr[:, 0])
    ls = GaussianProcess(seed=0).fit(Xr, y).params.lengthscales
    assert ls[1] > 5 * ls[0]


def test_fits_repeat_from_a_seed_and_predict_in_the_right_shapes():
    a = GaussianProcess(seed=2).fit(X, Y)
    b = GaussianProcess(seed=2).fit(X, Y)
    assert a.params == b.params
    mean, sd = a.predict(X[:4])
    assert mean.shape == sd.shape == (4,)
    assert np.all(sd >= 0)
    warm = GaussianProcess(seed=2).fit(X, Y, init=a.params, restarts=0)
    assert warm.log_likelihood == pytest.approx(a.log_likelihood, abs=1e-3)


def test_bad_settings_and_data_are_refused():
    with pytest.raises(ValueError, match="kernel"):
        GaussianProcess(kernel="cubic")
    with pytest.raises(ValueError, match="noise"):
        GaussianProcess(noise="lots")
    with pytest.raises(ValueError, match="responses"):
        GaussianProcess().fit(X, Y[:3])
    with pytest.raises(ValueError, match="finite"):
        GaussianProcess().fit(X[:2], [1.0, np.nan])
    with pytest.raises(RuntimeError, match="fit"):
        GaussianProcess().predict(X)
