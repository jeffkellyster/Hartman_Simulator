"""Experimental design generators.

Each generator takes the factors to vary (a list of names, or a count k for
x1..xk) and returns a `Design`: a matrix of coded levels (-1 low, 0 center,
+1 high), one row per run, plus what an analyst needs to know about it. The
optimization kernel works on the unit cube, so `Design.unit` gives the same
runs on [0, 1]. Pure NumPy, deterministic given a seed.

Space filling:
- Random: uniform over the cube.
- Latin hypercube: each factor's range cut into `runs` equal strata, each
  stratum used once, at a random spot inside it.
- Maximin Latin hypercube: the best of several Latin hypercubes, then
  improved by swapping values within a column, toward the largest smallest
  distance between runs (Morris-Mitchell phi_p criterion).

Classical:
- Full factorial: every combination of 2 or 3 levels.
- Fractional factorial: 2^(k-p) runs. Generators are chosen for the highest
  resolution and then the fewest short defining words (minimum aberration),
  and the alias structure of main effects and two-factor interactions is
  reported.
- Plackett-Burman: 12, 20 or 24 runs for main-effect screening.
- Central composite: a full or resolution V cube, axial points and center
  points. `inscribed=True` shrinks the whole design so the axial points sit
  on the factor limits, for regions (like Hartmann's) that end at the box.
- Box-Behnken: three levels, no corner runs, for 3 to 7 factors, from
  Box and Behnken's (1960) tables (6 factors: 48 runs + 6 centers, as in JMP).

The factorial, Plackett-Burman and central composite generators are ported
from the race car DOE simulator (github.com/jeffkellyster/DOE_Racecar_sim).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_RUNS = 4096
MAX_EXHAUSTIVE_COMBINATIONS = 5000
PLACKETT_BURMAN_GENERATORS = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}

# Box-Behnken blocks: each block's factors run a 2^m factorial, the rest sit at center.
# 3-5 factors use every pair; 6 and 7 use Box and Behnken's (1960) triples.
BOX_BEHNKEN_BLOCKS = {
    3: list(itertools.combinations(range(3), 2)),
    4: list(itertools.combinations(range(4), 2)),
    5: list(itertools.combinations(range(5), 2)),
    6: [(0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 3, 4), (1, 4, 5), (0, 2, 5)],
    7: [(3, 4, 5), (0, 5, 6), (1, 4, 6), (0, 1, 3), (2, 3, 6), (0, 2, 4), (1, 2, 5)],
}
BOX_BEHNKEN_CENTERS = {3: 3, 4: 3, 5: 6, 6: 6, 7: 6}  # JMP's defaults

PHI_P = 15  # Morris-Mitchell exponent: large p ranks designs by their closest pairs


@dataclass
class Design:
    kind: str
    factors: list[str]
    matrix: np.ndarray  # coded levels, one row per run, in run order
    info: dict = field(default_factory=dict)

    @property
    def runs(self) -> int:
        return len(self.matrix)

    @property
    def unit(self) -> np.ndarray:
        """The runs on the unit cube: coded -1, 0, +1 become 0, 0.5, 1."""
        return (self.matrix + 1.0) / 2.0

    @property
    def rows(self) -> list[dict[str, float]]:
        return [{name: float(v) for name, v in zip(self.factors, row)} for row in self.matrix]


KINDS = (
    "random",
    "latin_hypercube",
    "maximin_lhs",
    "full_factorial",
    "fractional_factorial",
    "plackett_burman",
    "central_composite",
    "box_behnken",
)


def make(kind: str, factors: Sequence[str] | int, **params) -> Design:
    generators = {
        "random": random_design,
        "latin_hypercube": latin_hypercube,
        "maximin_lhs": maximin_lhs,
        "full_factorial": full_factorial,
        "fractional_factorial": fractional_factorial,
        "plackett_burman": plackett_burman,
        "central_composite": central_composite,
        "box_behnken": box_behnken,
    }
    if kind not in generators:
        raise ValueError(f"unknown design type {kind!r}; choose one of {', '.join(KINDS)}")
    return generators[kind](factors, **params)


# --- Space filling ---------------------------------------------------------------


def random_design(factors: Sequence[str] | int, runs: int, seed: int = 0) -> Design:
    factors = _check_factors(factors)
    runs = _check_runs(runs, minimum=1)
    unit = np.random.default_rng(int(seed)).random((runs, len(factors)))
    info = {"seed": int(seed), **_spacing(unit)}
    return _finish("random", factors, 2.0 * unit - 1.0, info, 0, False, seed)


def latin_hypercube(factors: Sequence[str] | int, runs: int, seed: int = 0) -> Design:
    factors = _check_factors(factors)
    runs = _check_runs(runs, minimum=2)
    unit = _lhs(np.random.default_rng(int(seed)), runs, len(factors))
    info = {"seed": int(seed), **_spacing(unit)}
    return _finish("latin_hypercube", factors, 2.0 * unit - 1.0, info, 0, False, seed)


def maximin_lhs(
    factors: Sequence[str] | int, runs: int, seed: int = 0, candidates: int = 20, swaps: int | None = None
) -> Design:
    """A Latin hypercube pushed toward the largest smallest distance between runs.

    Starts from the best of `candidates` random Latin hypercubes, then tries
    `swaps` random exchanges of two runs' values in one column (default
    100 per run), keeping each exchange that lowers phi_p. Swaps within a
    column keep the design a Latin hypercube.
    """
    factors = _check_factors(factors)
    runs = _check_runs(runs, minimum=2)
    k = len(factors)
    rng = np.random.default_rng(int(seed))
    candidates = max(1, int(candidates))
    swaps = 100 * runs if swaps is None else max(0, int(swaps))

    best, best_phi = None, np.inf
    for _ in range(candidates):
        unit = _lhs(rng, runs, k)
        phi = _phi_p(_squared_distances(unit))
        if phi < best_phi:
            best, best_phi = unit, phi
    unit, phi = best, best_phi
    d2 = _squared_distances(unit)
    for _ in range(swaps):
        j = int(rng.integers(k))
        a, b = (int(i) for i in rng.choice(runs, size=2, replace=False))
        trial = unit.copy()
        trial[[a, b], j] = trial[[b, a], j]
        trial_d2 = d2.copy()
        for i in (a, b):
            row = ((trial - trial[i]) ** 2).sum(axis=1)
            trial_d2[i, :] = row
            trial_d2[:, i] = row
        trial_phi = _phi_p(trial_d2)
        if trial_phi < phi:
            unit, d2, phi = trial, trial_d2, trial_phi
    info = {"seed": int(seed), "criterion": "maximin (phi_p)", "candidates": candidates, "swaps": swaps,
            **_spacing(unit)}
    return _finish("maximin_lhs", factors, 2.0 * unit - 1.0, info, 0, False, seed)


# --- Classical -------------------------------------------------------------------


def full_factorial(
    factors: Sequence[str] | int, levels: int = 2, center_points: int = 0, randomize: bool = False, seed: int = 0
) -> Design:
    factors = _check_factors(factors)
    levels = int(levels)
    if levels not in (2, 3):
        raise ValueError("levels must be 2 or 3")
    runs = levels ** len(factors)
    if runs > MAX_RUNS:
        raise ValueError(
            f"a {levels}-level full factorial in {len(factors)} factors needs {runs} runs; use a fractional design"
        )
    info = {"levels": levels, "resolution": None, "aliases": _no_aliases()}
    return _finish("full_factorial", factors, _full_matrix(len(factors), levels), info, center_points, randomize, seed)


def fractional_factorial(
    factors: Sequence[str] | int,
    runs: int,
    generators: Sequence[str] | None = None,
    center_points: int = 0,
    randomize: bool = False,
    seed: int = 0,
) -> Design:
    """A two-level fraction in `runs` (a power of two) runs.

    `generators`, if given, are letter strings for each added factor in order,
    made from the first log2(runs) factors' letters, e.g. ["ABC", "BCD"].
    """
    factors = _check_factors(factors)
    matrix, info = _fractional(len(factors), int(runs), factors, generators)
    return _finish("fractional_factorial", factors, matrix, info, center_points, randomize, seed)


def plackett_burman(
    factors: Sequence[str] | int, runs: int | None = None, center_points: int = 0, randomize: bool = False,
    seed: int = 0,
) -> Design:
    factors = _check_factors(factors)
    k = len(factors)
    sizes = sorted(PLACKETT_BURMAN_GENERATORS)
    if runs is None:
        fitting = [n for n in sizes if n - 1 >= k]
        if not fitting:
            raise ValueError(f"Plackett-Burman designs here handle up to {sizes[-1] - 1} factors")
        runs = fitting[0]
    runs = int(runs)
    if runs not in PLACKETT_BURMAN_GENERATORS:
        raise ValueError(f"Plackett-Burman runs must be one of {sizes}")
    if k > runs - 1:
        raise ValueError(f"a {runs}-run Plackett-Burman design handles at most {runs - 1} factors")
    generator = np.array([1.0 if c == "+" else -1.0 for c in PLACKETT_BURMAN_GENERATORS[runs]])
    rows = [np.roll(generator, -i) for i in range(runs - 1)] + [-np.ones(runs - 1)]
    matrix = np.vstack(rows)[:, :k]
    info = {
        "resolution": 3,
        "aliases": _no_aliases(),
        "note": "Main effects are estimated clear of each other. Two-factor interactions are partially "
        "aliased with main effects and with each other.",
    }
    return _finish("plackett_burman", factors, matrix, info, center_points, randomize, seed)


def central_composite(
    factors: Sequence[str] | int,
    alpha: str | float = "face",
    center_points: int = 4,
    inscribed: bool = False,
    randomize: bool = False,
    seed: int = 0,
) -> Design:
    """Factorial cube (full up to 5 factors, otherwise resolution V or better) plus axial and center points.

    `alpha` is "face" (axial points at +/-1), "rotatable" (cube runs ^ 1/4),
    or a number. With alpha > 1 the axial points fall outside the factor
    limits unless `inscribed` is set, which divides the whole design by alpha
    so the axial points land on the limits and the cube sits inside them.
    """
    factors = _check_factors(factors)
    k = len(factors)
    if k < 2:
        raise ValueError("a central composite design needs at least 2 factors")
    if k <= 5:
        cube = _full_matrix(k, 2)
        info = {"cube": "full factorial", "cube_runs": len(cube), "cube_resolution": None}
    else:
        for m in range(k.bit_length(), k + 1):
            cube, cube_info = _fractional(k, 1 << m, factors, None)
            if cube_info["resolution"] is None or cube_info["resolution"] >= 5:
                break
        info = {
            "cube": "fractional factorial",
            "cube_runs": len(cube),
            "cube_resolution": cube_info["resolution"],
            "generators": cube_info["generators"],
            "letters": cube_info["letters"],
        }
    if alpha == "face":
        a = 1.0
    elif alpha == "rotatable":
        a = len(cube) ** 0.25
    else:
        a = float(alpha)
    if not a > 0:
        raise ValueError("alpha must be positive")
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = -a
        axial[2 * i + 1, i] = a
    matrix = np.vstack([cube, axial])
    if inscribed and a > 1:
        matrix = matrix / a
    info.update({"alpha": a, "inscribed": bool(inscribed), "axial_runs": 2 * k, "resolution": None,
                 "aliases": _no_aliases()})
    if a > 1 and not inscribed:
        info["note"] = ("Axial points sit beyond each factor's low and high, so the ranges must leave room "
                        "(or use inscribed=True).")
    return _finish("central_composite", factors, matrix, info, center_points, randomize, seed)


def box_behnken(
    factors: Sequence[str] | int, center_points: int | None = None, randomize: bool = False, seed: int = 0
) -> Design:
    """Box-Behnken design for 3 to 7 factors: edge midpoints and centers, no corners."""
    factors = _check_factors(factors)
    k = len(factors)
    if k not in BOX_BEHNKEN_BLOCKS:
        raise ValueError(f"Box-Behnken designs here cover 3 to 7 factors, not {k}")
    blocks = BOX_BEHNKEN_BLOCKS[k]
    rows = []
    for block in blocks:
        for signs in _full_matrix(len(block), 2):
            row = np.zeros(k)
            row[list(block)] = signs
            rows.append(row)
    if center_points is None:
        center_points = BOX_BEHNKEN_CENTERS[k]
    info = {"blocks": [[factors[i] for i in b] for b in blocks], "resolution": None, "aliases": _no_aliases()}
    return _finish("box_behnken", factors, np.array(rows), info, center_points, randomize, seed)


# --- Internals -------------------------------------------------------------------


def _check_factors(factors: Sequence[str] | int) -> list[str]:
    if isinstance(factors, int) and not isinstance(factors, bool):
        factors = [f"x{i + 1}" for i in range(factors)]
    factors = [str(f) for f in factors]
    if not factors:
        raise ValueError("choose at least one factor")
    if len(set(factors)) != len(factors):
        raise ValueError("factor names must be unique")
    if len(factors) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} factors")
    return factors


def _check_runs(runs, minimum: int) -> int:
    if isinstance(runs, bool) or int(runs) != runs:
        raise ValueError(f"runs must be a whole number, got {runs!r}")
    runs = int(runs)
    if runs < minimum:
        raise ValueError(f"need at least {minimum} runs")
    if runs > MAX_RUNS:
        raise ValueError(f"at most {MAX_RUNS} runs")
    return runs


def _lhs(rng: np.random.Generator, runs: int, k: int) -> np.ndarray:
    return np.column_stack([(rng.permutation(runs) + rng.random(runs)) / runs for _ in range(k)])


def _squared_distances(unit: np.ndarray) -> np.ndarray:
    return ((unit[:, None, :] - unit[None, :, :]) ** 2).sum(axis=-1)


def _phi_p(d2: np.ndarray) -> float:
    """Morris-Mitchell phi_p from squared distances; smaller is better spread."""
    upper = d2[np.triu_indices(len(d2), 1)]
    if np.any(upper == 0):
        return np.inf
    # Scaled by the smallest distance so large p cannot overflow.
    d = np.sqrt(upper)
    dmin = d.min()
    return float((np.sum((dmin / d) ** PHI_P)) ** (1.0 / PHI_P) / dmin)


def _spacing(unit: np.ndarray) -> dict:
    """Smallest distance between two runs, on the unit cube."""
    if len(unit) < 2:
        return {"min_distance": None}
    d2 = _squared_distances(unit)
    d2[np.diag_indices(len(unit))] = np.inf
    return {"min_distance": float(np.sqrt(d2.min()))}


def _finish(kind, factors, matrix, info, center_points, randomize, seed) -> Design:
    if isinstance(center_points, bool) or int(center_points) != center_points or center_points < 0:
        raise ValueError("center_points must be a whole number >= 0")
    center_points = int(center_points)
    if center_points:
        matrix = np.vstack([matrix, np.zeros((center_points, len(factors)))])
    order = np.random.default_rng(int(seed)).permutation(len(matrix)) if randomize else np.arange(len(matrix))
    matrix = np.asarray(matrix, dtype=float)[order] + 0.0  # + 0.0 turns -0.0 into 0.0
    info = {
        "type": kind,
        "runs": len(matrix),
        "factors": list(factors),
        "center_points": center_points,
        "randomized": bool(randomize),
        "standard_order": [int(i) + 1 for i in order],
        **info,
    }
    return Design(kind, list(factors), matrix, info)


def _full_matrix(k: int, levels: int) -> np.ndarray:
    values = (-1.0, 1.0) if levels == 2 else (-1.0, 0.0, 1.0)
    # Standard (Yates) order: the first factor changes fastest.
    return np.array([combo[::-1] for combo in itertools.product(values, repeat=k)]).reshape(-1, k)


def _fractional(k: int, runs: int, names: Sequence[str], generators: Sequence[str] | None):
    m = runs.bit_length() - 1
    if runs < 2 or 1 << m != runs:
        raise ValueError("runs must be a power of two")
    if runs > 1 << k:
        raise ValueError(f"{k} factors need at most {1 << k} runs (the full factorial)")
    if runs < k + 1:
        raise ValueError(f"{k} factors need at least {1 << k.bit_length()} runs")
    if runs > MAX_RUNS:
        raise ValueError(f"at most {MAX_RUNS} runs")
    base = _full_matrix(m, 2)
    p = k - m
    letters = {LETTERS[i]: names[i] for i in range(k)}
    if p == 0:
        return base, {"resolution": None, "base_factors": list(names), "letters": letters, "generators": [],
                      "word_length_pattern": {}, "aliases": _no_aliases()}

    masks = _parse_generators(generators, m, p) if generators else _choose_generators(m, p, k)
    added = np.column_stack([np.prod(base[:, [b for b in range(m) if mask >> b & 1]], axis=1) for mask in masks])
    words = _defining_words(m, masks)
    lengths = _popcount(words)
    info = {
        "resolution": int(lengths.min()),
        "base_factors": list(names[:m]),
        "letters": letters,
        "generators": [f"{LETTERS[m + j]} = {_mask_letters(mask)}" for j, mask in enumerate(masks)],
        "word_length_pattern": {int(n): int(c) for n, c in zip(*np.unique(lengths, return_counts=True))},
        "aliases": _aliases(k, words, names),
    }
    return np.hstack([base, added]), info


def _parse_generators(generators: Sequence[str], m: int, p: int) -> list[int]:
    if len(generators) != p:
        raise ValueError(f"need {p} generators, one per added factor")
    masks = []
    for text in generators:
        mask = 0
        for ch in text.replace(" ", "").upper():
            index = LETTERS.find(ch)
            if not 0 <= index < m:
                raise ValueError(f"generator {text!r} may only use the first {m} letters")
            mask ^= 1 << index
        if bin(mask).count("1") < 2:
            raise ValueError(f"generator {text!r} must use at least two letters")
        masks.append(mask)
    return masks


def _choose_generators(m: int, p: int, k: int) -> list[int]:
    """Generators with the best word length pattern found: exhaustive when small, greedy otherwise.

    Also tries only odd-length generators, which is how resolution IV designs
    with up to runs/2 factors are built.
    """
    weight = lambda x: bin(x).count("1")  # noqa: E731
    candidates = sorted((x for x in range(1, 1 << m) if weight(x) >= 2), key=lambda x: (-weight(x), x))
    pools = [candidates]
    odd = [x for x in candidates if weight(x) % 2 == 1]
    if len(odd) >= p:
        pools.append(odd)
    best_key, best = None, None
    for pool in pools:
        if math.comb(len(pool), p) <= MAX_EXHAUSTIVE_COMBINATIONS:
            options = itertools.combinations(pool, p)
        else:
            options = [_greedy_generators(m, p, k, pool)]
        for combo in options:
            key = _pattern_key(m, combo, k)
            if best_key is None or key < best_key:
                best_key, best = key, list(combo)
    return best


def _greedy_generators(m: int, p: int, k: int, pool: Sequence[int]) -> list[int]:
    chosen: list[int] = []
    words = np.zeros(1, dtype=np.uint64)
    counts = np.zeros(k + 1, dtype=np.int64)
    for j in range(p):
        best = None
        for mask in pool:
            if mask in chosen:
                continue
            word = np.uint64(mask | (1 << (m + j)))
            trial = counts + np.bincount(_popcount(words ^ word), minlength=k + 1)
            key = tuple(trial[1:].tolist())
            if best is None or key < best[0]:
                best = (key, mask, trial, word)
        _, mask, counts, word = best
        chosen.append(mask)
        words = np.concatenate([words, words ^ word])
    return chosen


def _pattern_key(m: int, masks: Sequence[int], k: int) -> tuple:
    counts = np.bincount(_popcount(_defining_words(m, masks)), minlength=k + 1)
    return tuple(counts[1:].tolist())


def _defining_words(m: int, masks: Sequence[int]) -> np.ndarray:
    """Every word in the defining relation except the identity, as bitmasks over all factors."""
    words = np.zeros(1, dtype=np.uint64)
    for j, mask in enumerate(masks):
        words = np.concatenate([words, words ^ np.uint64(mask | (1 << (m + j)))])
    return words[1:]


def _popcount(values: np.ndarray) -> np.ndarray:
    """Set bits per value, as NumPy's native index type.

    np.intp rather than int64: in Pyodide (32-bit WebAssembly) np.bincount
    refuses int64 input.
    """
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values).astype(np.intp)
    values = values.astype(np.uint64)
    count = np.zeros(values.shape, dtype=np.intp)
    one = np.uint64(1)
    while values.any():
        count += (values & one).astype(np.intp)
        values = values >> one
    return count


def _aliases(k: int, words: np.ndarray, names: Sequence[str]) -> dict:
    """Aliases among main effects and two-factor interactions."""
    weight = lambda x: bin(x).count("1")  # noqa: E731

    def label(mask: int) -> str:
        return " × ".join(names[i] for i in range(k) if mask >> i & 1)

    def aliased(mask: int) -> list[int]:
        others = words ^ np.uint64(mask)
        short = others[_popcount(others) <= 2]
        return sorted({int(x) for x in short}, key=lambda x: (weight(x), x))

    main = {}
    for i in range(k):
        found = aliased(1 << i)
        if found:
            main[names[i]] = [label(x) for x in found]
    chains, seen = [], set()
    for i, j in itertools.combinations(range(k), 2):
        mask = (1 << i) | (1 << j)
        if mask in seen:
            continue
        found = aliased(mask)
        seen.update(x for x in found if weight(x) == 2)
        if found:
            chains.append([label(mask)] + [label(x) for x in found])
    return {"main_effects": main, "two_factor_interactions": chains}


def _no_aliases() -> dict:
    return {"main_effects": {}, "two_factor_interactions": []}


def _mask_letters(mask: int) -> str:
    return "".join(LETTERS[i] for i in range(mask.bit_length()) if mask >> i & 1)
