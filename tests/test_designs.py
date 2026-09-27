import itertools

import numpy as np
import pytest

from seqopt import designs

NAMES = [f"x{i}" for i in range(24)]


def is_orthogonal(matrix):
    return np.allclose(matrix.T @ matrix, len(matrix) * np.eye(matrix.shape[1]))


def quadratic_model(matrix):
    k = matrix.shape[1]
    cols = [np.ones(len(matrix))] + [matrix[:, i] for i in range(k)]
    cols += [matrix[:, i] * matrix[:, j] for i, j in itertools.combinations(range(k), 2)]
    cols += [matrix[:, i] ** 2 for i in range(k)]
    return np.column_stack(cols)


def min_distance(unit):
    d = np.sqrt(((unit[:, None, :] - unit[None, :, :]) ** 2).sum(axis=-1))
    d[np.diag_indices(len(unit))] = np.inf
    return d.min()


# --- Space filling ---------------------------------------------------------------


@pytest.mark.parametrize("kind", ["random", "latin_hypercube", "maximin_lhs"])
def test_space_filling_designs_reproduce_from_a_seed_and_stay_in_the_cube(kind):
    a = designs.make(kind, 6, runs=20, seed=3)
    assert a.matrix.shape == (20, 6)
    assert np.all((a.unit >= 0) & (a.unit <= 1))
    assert np.array_equal(a.matrix, designs.make(kind, 6, runs=20, seed=3).matrix)
    assert not np.array_equal(a.matrix, designs.make(kind, 6, runs=20, seed=4).matrix)
    assert a.info["min_distance"] == pytest.approx(min_distance(a.unit))


@pytest.mark.parametrize("kind", ["latin_hypercube", "maximin_lhs"])
def test_latin_hypercubes_use_every_stratum_once(kind):
    d = designs.make(kind, 5, runs=12, seed=1)
    strata = np.floor(d.unit * 12).astype(int)
    for col in strata.T:
        assert sorted(col) == list(range(12))


def test_maximin_spreads_runs_further_apart_than_plain_latin_hypercubes():
    maximin = designs.maximin_lhs(6, 30, seed=0).info["min_distance"]
    plain = [designs.latin_hypercube(6, 30, seed=s).info["min_distance"] for s in range(10)]
    assert maximin > max(plain)


def test_factors_can_be_names_or_a_count():
    assert designs.random_design(3, 4).factors == ["x1", "x2", "x3"]
    assert designs.random_design(["a", "b"], 4).factors == ["a", "b"]
    assert designs.random_design(["a", "b"], 4).rows[0].keys() == {"a", "b"}
    with pytest.raises(ValueError, match="unique"):
        designs.random_design(["a", "a"], 4)
    with pytest.raises(ValueError, match="whole number"):
        designs.latin_hypercube(3, 2.5)
    with pytest.raises(ValueError, match="at least 2"):
        designs.latin_hypercube(3, 1)


# --- Factorials (ported from the race car simulator) -------------------------------------


def test_full_factorial_two_and_three_levels():
    d = designs.full_factorial(NAMES[:3])
    assert d.matrix.shape == (8, 3)
    assert len({tuple(r) for r in d.matrix}) == 8
    assert is_orthogonal(d.matrix)
    assert d.matrix[:, 0].tolist() == [-1, 1] * 4  # first factor changes fastest
    assert designs.full_factorial(NAMES[:3], levels=3).matrix.shape == (27, 3)
    with pytest.raises(ValueError, match="fractional"):
        designs.full_factorial(NAMES[:13])


@pytest.mark.parametrize(
    "k, runs, resolution",
    [(4, 8, 4), (5, 16, 5), (6, 16, 4), (6, 32, 6), (7, 16, 4), (7, 8, 3), (8, 16, 4), (20, 64, 4), (11, 16, 3)],
)
def test_fractional_factorial_resolution(k, runs, resolution):
    d = designs.fractional_factorial(NAMES[:k], runs)
    m = d.matrix
    assert m.shape == (runs, k)
    assert np.all(m.sum(axis=0) == 0)
    assert is_orthogonal(m)
    assert d.info["resolution"] == resolution


def test_resolution_iv_keeps_main_effects_clear_and_aliases_interactions():
    d = designs.fractional_factorial(NAMES[:6], 16)
    aliases = d.info["aliases"]
    assert aliases["main_effects"] == {}
    assert any(len(chain) > 1 for chain in aliases["two_factor_interactions"])
    assert len(d.info["generators"]) == 2


def test_resolution_iii_aliases_main_effects_with_interactions():
    d = designs.fractional_factorial(NAMES[:6], 8)
    assert d.info["resolution"] == 3
    assert "x0" in d.info["aliases"]["main_effects"]


def test_given_generators_are_used_and_checked():
    d = designs.fractional_factorial(NAMES[:5], 8, generators=["AB", "AC"])
    assert d.info["generators"] == ["D = AB", "E = AC"]
    assert np.array_equal(d.matrix[:, 3], d.matrix[:, 0] * d.matrix[:, 1])
    with pytest.raises(ValueError):
        designs.fractional_factorial(NAMES[:5], 8, generators=["AD", "AC"])
    with pytest.raises(ValueError, match="power of two"):
        designs.fractional_factorial(NAMES[:5], 12)


def test_bit_counts_use_the_native_index_type():
    # In Pyodide (32-bit WebAssembly) np.bincount rejects int64, so counts must be np.intp.
    counts = designs._popcount(np.array([0b1011, 0b1], dtype=np.uint64))
    assert counts.dtype == np.intp
    assert counts.tolist() == [3, 1]


@pytest.mark.parametrize("runs", [12, 20, 24])
def test_plackett_burman_is_orthogonal(runs):
    d = designs.plackett_burman(NAMES[: runs - 1], runs)
    assert is_orthogonal(d.matrix)


# --- Response surface designs ---------------------------------------------------------


def test_central_composite_face_and_rotatable():
    face = designs.central_composite(NAMES[:3], center_points=4)
    assert face.runs == 8 + 6 + 4
    assert np.abs(face.matrix).max() == 1
    six = designs.central_composite(6, center_points=0)
    assert six.info["cube_runs"] == 32 and six.info["cube_resolution"] >= 5
    assert np.linalg.matrix_rank(quadratic_model(six.matrix)) == 28
    rotatable = designs.central_composite(6, alpha="rotatable", center_points=0)
    assert rotatable.info["alpha"] == pytest.approx(32 ** 0.25)
    assert np.abs(rotatable.matrix).max() == pytest.approx(32 ** 0.25)
    assert "note" in rotatable.info


def test_an_inscribed_central_composite_stays_inside_the_ranges():
    d = designs.central_composite(6, alpha="rotatable", inscribed=True, center_points=2)
    a = d.info["alpha"]
    assert np.abs(d.matrix).max() == pytest.approx(1.0)
    cube = d.matrix[np.all(np.abs(d.matrix) > 0, axis=1)]
    assert np.allclose(np.abs(cube), 1 / a)
    assert "note" not in d.info


@pytest.mark.parametrize("k, runs", [(3, 15), (4, 27), (5, 46), (6, 54), (7, 62)])
def test_box_behnken_matches_jmp_run_counts_and_fits_a_full_quadratic(k, runs):
    d = designs.box_behnken(k)
    m = d.matrix
    assert m.shape == (runs, k)
    assert set(np.unique(m)) == {-1.0, 0.0, 1.0}
    assert not np.any(np.all(np.abs(m) == 1, axis=1))  # no corner runs
    assert np.all(m.sum(axis=0) == 0)
    p = quadratic_model(m).shape[1]
    assert np.linalg.matrix_rank(quadratic_model(m)) == p


def test_box_behnken_limits_and_center_points():
    assert designs.box_behnken(6, center_points=0).runs == 48
    with pytest.raises(ValueError, match="3 to 7"):
        designs.box_behnken(8)


def test_center_points_and_randomization():
    plain = designs.box_behnken(4, center_points=3)
    shuffled = designs.box_behnken(4, center_points=3, randomize=True, seed=9)
    assert sorted(map(tuple, plain.matrix)) == sorted(map(tuple, shuffled.matrix))
    assert shuffled.info["standard_order"] != list(range(1, 28))
    assert np.array_equal(shuffled.matrix, plain.matrix[np.array(shuffled.info["standard_order"]) - 1])


def test_make_dispatches_and_rejects_unknown_kinds():
    assert designs.make("box_behnken", 6).runs == 54
    with pytest.raises(ValueError, match="unknown design type"):
        designs.make("taguchi", 6)
