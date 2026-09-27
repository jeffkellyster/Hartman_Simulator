import numpy as np
import pytest

from hartmann import NoiseModel

U4 = np.random.default_rng(0).random((4, 6))


def test_no_noise_unless_asked():
    model = NoiseModel()
    assert not model.active
    assert np.array_equal(model.draw(seed=1, first=0, U=U4), np.zeros(4))


def test_a_draw_depends_only_on_the_seed_and_the_evaluation_number():
    model = NoiseModel(sd=0.3)
    whole = model.draw(seed=5, first=0, U=U4)
    split = np.concatenate([model.draw(seed=5, first=0, U=U4[:2]), model.draw(seed=5, first=2, U=U4[2:])])
    assert np.array_equal(whole, split)
    assert np.array_equal(whole, model.draw(seed=5, first=0, U=U4))
    assert not np.array_equal(whole, model.draw(seed=6, first=0, U=U4))
    assert not np.array_equal(whole, model.draw(seed=5, first=1, U=U4))


def test_heteroscedastic_sd_follows_its_formula():
    model = NoiseModel(sd=0.2, hetero=1.5, hetero_factor=4)
    low, high = np.full((1, 6), 0.5), np.full((1, 6), 0.5)
    low[0, 4], high[0, 4] = 0.0, 1.0
    assert model.sd_at(low) == pytest.approx([0.2])
    assert model.sd_at(high) == pytest.approx([0.5])
    n = 4000
    at_low = model.draw(seed=9, first=0, U=np.repeat(low, n, axis=0))
    at_high = model.draw(seed=9, first=n, U=np.repeat(high, n, axis=0))
    assert at_low.std(ddof=1) == pytest.approx(0.2, rel=0.05)
    assert at_high.std(ddof=1) == pytest.approx(0.5, rel=0.05)
    assert abs(at_low.mean()) < 3 * 0.2 / np.sqrt(n)


def test_homoscedastic_sd_is_the_same_everywhere():
    model = NoiseModel(sd=0.1)
    assert np.all(model.sd_at(U4) == 0.1)


def test_settings_round_trip_and_are_validated():
    model = NoiseModel(sd=0.1, hetero=0.5, hetero_factor=2)
    assert NoiseModel.from_dict(model.to_dict()) == model
    assert NoiseModel.from_dict(None) == NoiseModel()
    with pytest.raises(ValueError, match="sd"):
        NoiseModel(sd=-0.1)
    with pytest.raises(ValueError, match="hetero"):
        NoiseModel(sd=0.1, hetero=-1)
    with pytest.raises(ValueError, match="hetero_factor"):
        NoiseModel(sd=0.1, hetero_factor=6)
    with pytest.raises(ValueError, match="unknown noise settings"):
        NoiseModel.from_dict({"sigma": 0.1})
