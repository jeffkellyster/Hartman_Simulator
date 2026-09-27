import numpy as np
import pytest

from hartmann import Scenario
from hartmann.scenario import IDENTITY


def test_a_seed_always_gives_the_same_scenario():
    assert Scenario.from_seed(3) == Scenario.from_seed(3)
    assert Scenario.from_seed(3).seed == 3


def test_scenarios_are_never_the_identity_and_seeds_mostly_differ():
    drawn = [Scenario.from_seed(s) for s in range(300)]
    assert not any(s.is_identity for s in drawn)
    assert len({(s.perm, s.flip) for s in drawn}) > 290
    assert IDENTITY.is_identity


def test_the_disguise_is_one_to_one_on_the_unit_cube():
    U = np.random.default_rng(0).random((100, 6))
    for seed in range(20):
        s = Scenario.from_seed(seed)
        Z = s.to_hartmann(U)
        assert np.all((Z >= 0) & (Z <= 1))
        assert s.from_hartmann(Z) == pytest.approx(U, abs=1e-15)
        assert s.to_hartmann(s.from_hartmann(U)) == pytest.approx(U, abs=1e-15)


def test_a_worked_example():
    s = Scenario(perm=(1, 0, 2, 3, 4, 5), flip=(True, False, False, False, False, False))
    z = s.to_hartmann([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert z == pytest.approx([0.2, 0.9, 0.3, 0.4, 0.5, 0.6])
    assert z.shape == (6,)


def test_settings_round_trip_and_are_validated():
    s = Scenario.from_seed(11)
    assert Scenario.from_dict(s.to_dict()) == s
    assert Scenario.from_dict(None) is IDENTITY
    with pytest.raises(ValueError, match="permutation"):
        Scenario(perm=(0, 0, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="flip"):
        Scenario(flip=(True,))
    with pytest.raises(ValueError, match="seed"):
        Scenario.from_seed(-1)
