import json

import numpy as np
import pytest

from hartmann import HartmannOracle, NoiseModel
from seqopt import testfunctions as tf
from seqopt.benchmark import _running_argmin, run_benchmark
from seqopt.loop import Optimizer, Strategy


# --- The loop --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["random", "rsm", "bo"])
def test_each_method_spends_exactly_the_budget(method):
    oracle = tf.branin(budget=14)
    opt = Optimizer(oracle, Strategy(method, initial_runs=8 if method != "random" else None), seed=1)
    opt.run()
    assert len(opt.y) == 14 and oracle.budget.remaining == 0
    assert opt.records[0].phase == ("random" if method == "random" else "initial")
    assert opt.best_so_far()[-1] == opt.y.min()
    assert np.all((opt.U >= 0) & (opt.U <= 1))


def test_a_run_repeats_exactly_from_its_seed():
    def run(seed):
        opt = Optimizer(tf.branin(budget=12, noise_sd=0.5, seed=3), Strategy("bo", initial_runs=6), seed=seed)
        opt.run()
        return opt
    a, b, c = run(4), run(4), run(5)
    assert np.array_equal(a.U, b.U) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.U, c.U)


def test_the_initial_design_defaults_follow_the_method():
    assert Strategy("bo").initial_count(6) == 12
    assert Strategy("rsm").initial_count(6) == 30
    assert Strategy("bo", initial_runs=5).initial_count(6) == 5


def test_batches_and_a_limit_on_evaluations():
    opt = Optimizer(tf.branin(), Strategy("bo", initial_runs=6, batch_size=3), seed=0)
    opt.run(evaluations=13)
    assert [len(r.y) for r in opt.records] == [6, 3, 3, 1]
    with pytest.raises(ValueError, match="no budget"):
        Optimizer(tf.branin(), Strategy("random")).run()


def test_rsm_never_measures_the_same_point_twice():
    opt = Optimizer(tf.rosenbrock(2, budget=16), Strategy("rsm", initial_runs=8), seed=0)
    opt.run()
    gaps = np.sqrt(((opt.U[:, None] - opt.U[None]) ** 2).sum(-1))[np.triu_indices(16, 1)]
    assert gaps.min() >= 0.02 - 1e-9


def test_bo_finds_the_branin_minimum_where_random_search_does_not():
    bo = Optimizer(tf.branin(budget=30), Strategy("bo", initial_runs=6), seed=0)
    bo.run()
    rand = Optimizer(tf.branin(budget=30), Strategy("random"), seed=0)
    rand.run()
    assert bo.best()[1] < 0.41 < rand.best()[1]


def test_bo_drives_the_hartmann_oracle_in_process_units():
    oracle = HartmannOracle(seed=0, budget=20, noise=NoiseModel(sd=0.02))
    opt = Optimizer(oracle, Strategy("bo"), seed=0, factors=[f["name"] for f in oracle.factors()])
    opt.run()
    x, y = opt.best()
    assert np.all((x >= oracle.bounds[:, 0]) & (x <= oracle.bounds[:, 1]))
    assert len(oracle.ledger) == 20 and y == min(e.y for e in oracle.ledger)
    log = opt.log()
    json.dumps(log)
    assert log["steps"][1]["model"]["kernel"] == "matern52"
    assert len(opt.log_rows()) == 20


def test_strategy_settings_are_checked_and_labelled():
    assert Strategy("bo", acquisition="ucb", batch_size=4).label == "BO (UCB, q=4)"
    assert Strategy("rsm").label == "LHS + RSM"
    assert Strategy("bo", name="Mine").label == "Mine"
    assert Strategy.from_dict(Strategy("bo", kappa=3).to_dict()) == Strategy("bo", kappa=3)
    for bad in ({"method": "annealing"}, {"acquisition": "magic"}, {"kernel": "cubic"}, {"batch_size": 0}):
        with pytest.raises(ValueError):
            Strategy(**bad)


# --- Benchmark ---------------------------------------------------------------------------


def test_benchmark_curves_summaries_and_rows():
    strategies = [Strategy("random"), Strategy("bo", initial_runs=5)]
    result = run_benchmark(lambda r: tf.branin(noise_sd=0.3, seed=r), strategies, budget=10, replicates=3,
                           optimum=0.397887)
    assert result.labels == ["Random", "BO (EI)"]
    for label in result.labels:
        assert result.observed[label].shape == (3, 10)
        assert np.all(np.diff(result.observed[label], axis=1) <= 0)  # best so far never gets worse
        assert np.all(result.true[label] >= 0.397887 - 1e-6)
    stats = result.at(10)
    assert set(stats["Random"]) == {"median", "q25", "q75"}
    assert stats["Random"]["q25"] <= stats["Random"]["median"] <= stats["Random"]["q75"]
    rows = result.rows()
    assert len(rows) == 2 * 3 * 10 and set(rows[0]) == {"strategy", "replicate", "evaluations",
                                                          "best_observed", "best_true"}
    json.dumps(result.to_dict())


def test_benchmarks_repeat_and_share_random_numbers_across_strategies():
    def bench():
        return run_benchmark(lambda r: tf.branin(noise_sd=0.3, seed=r),
                             [Strategy("random", name="A"), Strategy("random", name="B")], budget=6, replicates=2)
    a, b = bench(), bench()
    assert np.array_equal(a.observed["A"], b.observed["A"])
    # Same seed per replicate: two copies of a strategy see identical runs.
    assert np.array_equal(a.observed["A"], a.observed["B"])


def test_benchmark_checks_its_inputs():
    with pytest.raises(ValueError, match="labels must differ"):
        run_benchmark(lambda r: tf.branin(), [Strategy("random"), Strategy("random")], 5, 1)
    with pytest.raises(ValueError, match="smaller than the benchmark"):
        run_benchmark(lambda r: tf.branin(budget=3), [Strategy("random")], 5, 1)


def test_running_argmin_keeps_the_earliest_best():
    assert _running_argmin(np.array([3.0, 1.0, 2.0, 1.0, 0.5])).tolist() == [0, 1, 1, 1, 4]


def test_test_function_oracles():
    b = tf.branin()
    for m in b.minimizers:
        assert b.true_response(m) == pytest.approx(0.397887, abs=1e-5)
    r = tf.rosenbrock(4)
    assert r.true_response([1.0] * 4) == 0.0
    with pytest.raises(ValueError, match="inside the bounds"):
        b.evaluate([20.0, 0.0])
    with pytest.raises(ValueError, match="at least 2"):
        tf.rosenbrock(1)
    noisy = tf.branin(noise_sd=1.0, seed=2)
    assert noisy.evaluate([0.0, 5.0]) != noisy.evaluate([0.0, 5.0])
