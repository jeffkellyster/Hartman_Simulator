import json

import pytest

from hartmann import csvio
from hartmann.cli import main


def run(*argv):
    return main([str(a) for a in argv])


def read(path):
    return csvio.parse_table(csvio.read_text(str(path)))


def test_info_lists_the_process(capsys):
    assert run("info") == 0
    out = capsys.readouterr().out
    assert "PECVD" in out and "RF power" in out and "lower is better" in out


def test_design_then_run_then_augment(tmp_path, capsys):
    design, lab, results = tmp_path / "d.csv", tmp_path / "lab.json", tmp_path / "r.csv"
    assert run("design", "maximin", "--runs", 12, "--seed", 1, "-o", design) == 0
    assert len(read(design).rows) == 12
    assert run("setup", "--noise-sd", 0.05, "--seed", 3, "--budget", 20, "--scenario", 7, "-o", lab) == 0
    assert json.loads(lab.read_text())["budget"] == 20

    assert run("run", design, "--config", lab, "-o", results) == 0
    err = capsys.readouterr().err
    assert "Measured 12 row(s)" in err and "12 of 20 used" in err
    table = read(results)
    assert all(row[-1] for row in table.rows)

    # Augment by hand: append 4 unmeasured rows and run in place.
    more = read(design).rows[:4]
    text = csvio.format_table(csvio.Table(table.header, table.rows + more))
    results.write_text(text, encoding="utf-8")
    assert run("run", results, "--config", lab, "--in-place") == 0
    err = capsys.readouterr().err
    assert "Measured 4 row(s); 12 already had a response" in err and "16 of 20 used" in err
    assert len(read(results).rows) == 16


def test_the_same_config_gives_the_same_measurements(tmp_path):
    design = tmp_path / "d.csv"
    run("design", "lhs", "--runs", 5, "-o", design)
    run("run", design, "--noise-sd", 0.1, "--seed", 9, "-o", tmp_path / "a.csv")
    run("run", design, "--noise-sd", 0.1, "--seed", 9, "-o", tmp_path / "b.csv")
    run("run", design, "--noise-sd", 0.1, "--seed", 10, "-o", tmp_path / "c.csv")
    assert read(tmp_path / "a.csv").rows == read(tmp_path / "b.csv").rows
    assert read(tmp_path / "a.csv").rows != read(tmp_path / "c.csv").rows


def test_a_random_seed_is_reported_so_the_run_can_be_repeated(tmp_path, capsys):
    design = tmp_path / "d.csv"
    run("design", "random", "--runs", 3, "-o", design)
    run("run", design, "--noise-sd", 0.1, "-o", tmp_path / "r.csv")
    assert "pass --seed" in capsys.readouterr().err


def test_design_options_are_checked(tmp_path, capsys):
    assert run("design", "ccd", "--runs", 10) == 2
    assert "don't take --runs" in capsys.readouterr().err
    assert run("design", "lhs") == 2
    assert "need --runs" in capsys.readouterr().err
    assert run("design", "ccd", "--alpha", "rotatable") == 2
    assert "--inscribed" in capsys.readouterr().err
    assert run("design", "ccd", "--alpha", "rotatable", "--inscribed", "-o", tmp_path / "c.csv") == 0
    assert run("design", "fractional", "--runs", 16, "-o", tmp_path / "f.csv") == 0
    assert "Resolution 4" in capsys.readouterr().err


def test_errors_are_reported_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("RF power (W)\n9000\n", encoding="utf-8")
    assert run("run", bad) == 2
    assert "RF power" in capsys.readouterr().err
    assert run("run", tmp_path / "missing.csv") == 2
    assert run("run", bad, "--hold", "nonsense") == 2


def test_truth_reveals_the_optimum_and_scores_a_table(tmp_path, capsys):
    design, results = tmp_path / "d.csv", tmp_path / "r.csv"
    run("design", "bbd", "-o", design)
    run("run", design, "--scenario", 7, "--seed", 1, "-o", results)
    capsys.readouterr()
    assert run("truth", "--scenario", 7, "--table", results) == 0
    out = capsys.readouterr().out
    assert "Global minimum: 1.016 %" in out
    assert "Blind scenario 7" in out
    assert "above the optimum" in out


@pytest.mark.parametrize("units", ["unit", "coded"])
def test_unit_and_coded_tables_round_trip(tmp_path, units):
    design, results = tmp_path / "d.csv", tmp_path / "r.csv"
    assert run("design", "bbd", "--units", units, "-o", design) == 0
    assert run("run", design, "--units", units, "-o", results) == 0
    assert read(results).header[-1] == "y"


def test_optimize_spends_the_budget_and_logs_every_run(tmp_path, capsys):
    log, steps = tmp_path / "log.csv", tmp_path / "log.json"
    assert run("optimize", "--budget", 16, "--seed", 1, "--noise-sd", 0.02, "--truth", "-o", log, "--json", steps) == 0
    err = capsys.readouterr().err
    assert "BO (EI): 16 evaluations" in err and "the optimum is 1.016 %" in err
    table = read(log)
    assert table.header[:3] == ["Run", "Step", "Phase"] and len(table.rows) == 16
    assert json.loads(steps.read_text())["label"] == "BO (EI)"
    assert run("optimize", "--strategy", "random", "--seed", 1) == 2
    assert "needs a budget" in capsys.readouterr().err


def test_benchmark_prints_a_table_and_writes_json_and_csv(tmp_path, capsys):
    out, curves = tmp_path / "bench.json", tmp_path / "curves.csv"
    assert run("benchmark", "--budget", 8, "--replicates", 2, "--strategies", "random,bo", "--seed", 3,
               "--units", "unit", "-o", out, "--csv", curves) == 0
    printed = capsys.readouterr().out
    assert "Random" in printed and "BO (EI)" in printed and "Optimum -3.322" in printed
    result = json.loads(out.read_text())
    assert result["labels"] == ["Random", "BO (EI)"] and result["replicates"] == 2
    assert len(read(curves).rows) == 2 * 2 * 8
    assert run("benchmark", "--budget", 8, "--strategies", "annealing") == 2
    assert "unknown strategies" in capsys.readouterr().err
