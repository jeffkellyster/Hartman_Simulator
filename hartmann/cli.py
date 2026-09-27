"""Command line: make a design, measure it, and (for the instructor) see the answer.

    hartmann info                                   the factors, their ranges, the response
    hartmann design lhs --runs 20 -o design.csv     a design table to measure, or to edit in JMP
    hartmann setup --noise-sd 0.05 --budget 60 --scenario 7 -o lab.json
    hartmann run design.csv --config lab.json -o results.csv
    hartmann truth --config lab.json --table results.csv
    hartmann optimize --strategy bo --budget 40 --config lab.json -o log.csv
    hartmann benchmark --budget 60 --replicates 10 --noise-sd 0.05 -o bench.json

`run` measures every row that has no response yet and writes the table back
with the response filled in. Rows that already have a response count as
earlier measurements, so a results table can be augmented (in JMP or by hand)
and run again. `setup` saves the oracle's settings (seed, noise, budget,
blind scenario) to a file that every later `run` uses, so a class can share
one scenario. `optimize` lets one strategy spend a budget on its own (a demo,
or an answer key), and `benchmark` compares strategies over seeded replicates.
Tables are written to stdout unless -o is given; the summary goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from seqopt import designs
from seqopt.acquisition import ACQUISITIONS
from seqopt.benchmark import run_benchmark
from seqopt.gp import KERNELS
from seqopt.loop import METHODS, Optimizer, Strategy
from seqopt.oracle import BudgetExhausted

from . import csvio, process
from .function import FORMS
from .noise import NoiseModel
from .oracle import HartmannOracle

DESIGN_ALIASES = {
    "random": "random",
    "lhs": "latin_hypercube",
    "latin_hypercube": "latin_hypercube",
    "maximin": "maximin_lhs",
    "maximin_lhs": "maximin_lhs",
    "full": "full_factorial",
    "full_factorial": "full_factorial",
    "fractional": "fractional_factorial",
    "fractional_factorial": "fractional_factorial",
    "pb": "plackett_burman",
    "plackett_burman": "plackett_burman",
    "ccd": "central_composite",
    "central_composite": "central_composite",
    "bbd": "box_behnken",
    "box_behnken": "box_behnken",
}
DESIGN_PARAMS = {
    "random": {"runs", "seed"},
    "latin_hypercube": {"runs", "seed"},
    "maximin_lhs": {"runs", "seed"},
    "full_factorial": {"levels", "center_points", "randomize", "seed"},
    "fractional_factorial": {"runs", "generators", "center_points", "randomize", "seed"},
    "plackett_burman": {"runs", "center_points", "randomize", "seed"},
    "central_composite": {"alpha", "inscribed", "center_points", "randomize", "seed"},
    "box_behnken": {"center_points", "randomize", "seed"},
}
DESIGN_NAMES = {
    "random": "Random design",
    "latin_hypercube": "Latin hypercube",
    "maximin_lhs": "Maximin Latin hypercube",
    "full_factorial": "Full factorial",
    "fractional_factorial": "Fractional factorial",
    "plackett_burman": "Plackett-Burman",
    "central_composite": "Central composite",
    "box_behnken": "Box-Behnken",
}
REQUIRED = {"random": {"runs"}, "latin_hypercube": {"runs"}, "maximin_lhs": {"runs"},
            "fractional_factorial": {"runs"}}
BENCHMARK_STRATEGIES = {
    "random": {"method": "random"},
    "rsm": {"method": "rsm"},
    "bo": {"method": "bo", "acquisition": "ei"},
    "bo-ei": {"method": "bo", "acquisition": "ei"},
    "bo-pi": {"method": "bo", "acquisition": "pi"},
    "bo-ucb": {"method": "bo", "acquisition": "ucb"},
    "exploit": {"method": "bo", "acquisition": "exploit"},
    "explore": {"method": "bo", "acquisition": "explore"},
}


class UsageError(ValueError):
    """A mistake in the command, reported without a traceback."""


# --- Parser ------------------------------------------------------------------------------


def _oracle_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("oracle settings (override --config)")
    g.add_argument("--units", choices=process.UNITS, help="process (default), unit or coded")
    g.add_argument("--form", choices=FORMS, help="function form, for unit or coded units (default standard)")
    g.add_argument("--seed", type=int, help="noise seed; the same seed repeats the same measurements")
    g.add_argument("--noise-sd", type=float, help="measurement noise SD, in response units (default 0)")
    g.add_argument("--hetero", type=float, help="SD growth along one factor: SD x (1 + hetero) at its high end")
    g.add_argument("--hetero-factor", help="factor the SD grows along: a name, label or number 1-6 (default RF power)")
    g.add_argument("--budget", type=int, help="maximum number of measurements (default unlimited)")
    g.add_argument("--scenario", type=int, help="blind scenario seed: disguises where the optimum is")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hartmann", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="list the factors and the response")
    p.add_argument("--units", choices=process.UNITS, default="process")

    p = sub.add_parser("design", help="make a design table")
    p.add_argument("kind", choices=sorted(DESIGN_ALIASES), metavar="kind",
                   help="random, lhs, maximin, full, fractional, pb, ccd or bbd")
    p.add_argument("--runs", type=int)
    p.add_argument("--seed", type=int, help="for random designs and run-order randomization (default 0)")
    p.add_argument("--center-points", type=int)
    p.add_argument("--levels", type=int, choices=(2, 3))
    p.add_argument("--alpha", help="central composite axial distance: face (default), rotatable or a number")
    p.add_argument("--inscribed", action="store_true", help="central composite: keep axial points inside the ranges")
    p.add_argument("--generators", nargs="+", help="fractional factorial generators, e.g. ABCD BCDE")
    p.add_argument("--randomize", action="store_true", help="randomize the run order")
    p.add_argument("--units", choices=process.UNITS, default="process")
    p.add_argument("-o", "--out", help="output file (default stdout)")

    p = sub.add_parser("setup", help="save oracle settings for later runs")
    p.add_argument("--config", help="start from an existing settings file")
    _oracle_options(p)
    p.add_argument("-o", "--out", help="settings file to write (default stdout)")

    p = sub.add_parser("run", help="measure the rows of a table that have no response yet")
    p.add_argument("table", help="CSV or tab-separated table; - reads stdin")
    p.add_argument("--config", help="settings file from `hartmann setup`")
    _oracle_options(p)
    p.add_argument("--hold", action="append", default=[], metavar="FACTOR=VALUE",
                   help="value for a factor the table leaves out (default: its center)")
    out = p.add_mutually_exclusive_group()
    out.add_argument("-o", "--out", help="output file (default stdout)")
    out.add_argument("--in-place", action="store_true", help="write the results back into the table file")

    p = sub.add_parser("truth", help="instructor view: where the optimum is, and how a results table did")
    p.add_argument("--config", help="settings file from `hartmann setup`")
    _oracle_options(p)
    p.add_argument("--table", help="a results table to score against the truth")

    p = sub.add_parser("optimize", help="let one strategy spend the budget on its own")
    p.add_argument("--config", help="settings file from `hartmann setup`")
    _oracle_options(p)
    s = p.add_argument_group("strategy")
    s.add_argument("--strategy", choices=METHODS, default="bo", help="bo (default), rsm or random")
    s.add_argument("--acquisition", choices=ACQUISITIONS, default="ei", help="for bo (default ei)")
    s.add_argument("--initial", type=int, help="initial design runs (default: 12 for bo, 30 for rsm)")
    s.add_argument("--batch", type=int, default=1, help="points proposed per step (default 1)")
    s.add_argument("--kernel", choices=KERNELS, default="matern52")
    s.add_argument("--xi", type=float, default=0.0, help="EI/PI improvement margin (default 0)")
    s.add_argument("--kappa", type=float, default=2.0, help="UCB width in SDs (default 2)")
    s.add_argument("--opt-seed", type=int, default=0, help="seed for the strategy's own random choices")
    p.add_argument("--truth", action="store_true", help="also report how far the best point is from the optimum")
    p.add_argument("-o", "--out", help="evaluation log as CSV (default stdout)")
    p.add_argument("--json", help="also write the step-by-step log (model fits, acquisition values) as JSON")

    p = sub.add_parser("benchmark", help="compare strategies over seeded replicates")
    p.add_argument("--config", help="settings file from `hartmann setup`")
    _oracle_options(p)
    p.add_argument("--strategies", default="random,rsm,bo",
                   help=f"comma-separated, from {', '.join(BENCHMARK_STRATEGIES)} (default random,rsm,bo)")
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--batch", type=int, default=1, help="points proposed per step by BO strategies")
    p.add_argument("--kernel", choices=KERNELS, default="matern52")
    p.add_argument("--opt-seed", type=int, default=0, help="seed for the strategies' own random choices")
    p.add_argument("-o", "--out", help="full results as JSON (the browser app can load it)")
    p.add_argument("--csv", help="best-so-far curves in long format, for JMP's Graph Builder")
    return parser


# --- Helpers ------------------------------------------------------------------------------


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise UsageError(f"{path} is not a settings file")
    return config


def _hetero_factor(value: str, units: str) -> int:
    if value.isdigit():
        index = int(value) - 1
        if not 0 <= index < 6:
            raise UsageError("--hetero-factor as a number must be 1-6")
        return index
    oracle = HartmannOracle(units=units, seed=0)
    matched = csvio._lookup(oracle).get(csvio.normalize(value))
    if matched is None:
        raise UsageError(f"--hetero-factor {value!r} is not a factor")
    return matched


def _oracle(args, fresh_seed_note: list[str]) -> HartmannOracle:
    config = _load_config(getattr(args, "config", None))
    noise = dict(config.get("noise") or {})
    if args.units is not None:
        config["units"] = args.units
    if args.form is not None:
        config["form"] = args.form
    if args.seed is not None:
        config["seed"] = args.seed
    if args.budget is not None:
        config["budget"] = args.budget
    if args.scenario is not None:
        config["scenario"] = args.scenario
    if args.noise_sd is not None:
        noise["sd"] = args.noise_sd
    if args.hetero is not None:
        noise["hetero"] = args.hetero
    if args.hetero_factor is not None:
        noise["hetero_factor"] = _hetero_factor(args.hetero_factor, config.get("units", "process"))
    config["noise"] = noise
    oracle = HartmannOracle.from_config(config)
    if config.get("seed") is None and oracle.noise.active:
        fresh_seed_note.append(f"Noise seed {oracle.seed} (pass --seed {oracle.seed} to repeat these measurements).")
    return oracle


def _parse_hold(items: Sequence[str]) -> dict[str, float]:
    hold = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise UsageError(f"--hold needs FACTOR=VALUE, got {item!r}")
        try:
            hold[name.strip()] = float(value)
        except ValueError:
            raise UsageError(f"--hold {item!r}: {value!r} is not a number") from None
    return hold


def _write(text: str, path: str | None, bom: bool = False) -> None:
    """Write to a file or stdout. Tables get a UTF-8 byte-order mark so Excel shows °C correctly."""
    if path:
        with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def _say(*lines: str) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def _fmt_point(oracle: HartmannOracle, x) -> str:
    return ", ".join(
        f"{f['label']} {v:.4g}{' ' + f['unit'] if oracle.units == 'process' else ''}"
        for f, v in zip(oracle.factors(), x)
    )


# --- Commands ------------------------------------------------------------------------------


def cmd_info(args) -> None:
    oracle = HartmannOracle(units=args.units, seed=0)
    print(f"{process.PROCESS_NAME if args.units == 'process' else 'Hartmann 6D'} ({args.units} units)")
    for i, f in enumerate(oracle.factors(), 1):
        unit = f" {f['unit']}" if args.units == "process" else ""
        print(f"  {i}. {f['label']:<24} {f['low']:g} to {f['high']:g}{unit}")
    print(f"Response: {csvio.response_header(oracle)}, lower is better")


def cmd_design(args) -> None:
    kind = DESIGN_ALIASES[args.kind]
    given = {name: getattr(args, name) for name in ("runs", "seed", "center_points", "levels", "alpha", "generators")
             if getattr(args, name) is not None}
    for flag in ("inscribed", "randomize"):
        if getattr(args, flag):
            given[flag] = True
    extra = set(given) - DESIGN_PARAMS[kind]
    if extra:
        raise UsageError(f"{args.kind} designs don't take {', '.join('--' + e.replace('_', '-') for e in sorted(extra))}")
    missing = REQUIRED.get(kind, set()) - set(given)
    if missing:
        raise UsageError(f"{args.kind} designs need {', '.join('--' + m for m in sorted(missing))}")
    if "alpha" in given and given["alpha"] not in ("face", "rotatable"):
        try:
            given["alpha"] = float(given["alpha"])
        except ValueError:
            raise UsageError("--alpha must be face, rotatable or a number") from None

    oracle = HartmannOracle(units=args.units, seed=0)
    design = designs.make(kind, [f["name"] for f in oracle.factors()], **given)
    try:
        points = process.from_unit(design.unit, args.units)
    except ValueError:
        raise UsageError(
            "some runs fall outside the factor ranges (axial points with alpha > 1); "
            "use --inscribed or --alpha face"
        ) from None
    _write(csvio.format_table(csvio.design_table(points, oracle)), args.out, bom=True)

    info = design.info
    lines = [f"{DESIGN_NAMES[kind]}: {design.runs} runs."]
    if info.get("resolution"):
        lines.append(f"Resolution {info['resolution']}; generators {', '.join(info['generators'])}.")
    if info.get("min_distance") is not None:
        lines.append(f"Smallest distance between runs (unit cube): {info['min_distance']:.3f}.")
    if kind == "central_composite":
        lines.append(f"Alpha {info['alpha']:.4g}{' (inscribed)' if info['inscribed'] else ''}; "
                     f"cube {info['cube_runs']} runs, {info['center_points']} center points.")
    if info.get("note"):
        lines.append(info["note"])
    _say(*lines)


def cmd_setup(args) -> None:
    notes: list[str] = []
    oracle = _oracle(args, notes)
    _write(json.dumps(oracle.config, indent=2) + "\n", args.out)
    budget = "unlimited" if oracle.budget.total is None else oracle.budget.total
    _say(f"Settings: {oracle.units} units, noise SD {oracle.noise.sd:g}, seed {oracle.seed}, budget {budget}, "
         f"{'blind scenario ' + str(oracle.scenario.seed) if not oracle.scenario.is_identity else 'no disguise'}.")


def cmd_run(args) -> None:
    notes: list[str] = []
    oracle = _oracle(args, notes)
    path = args.table
    text = sys.stdin.read() if path == "-" else csvio.read_text(path)
    result = csvio.measure_table(csvio.parse_table(text), oracle, _parse_hold(args.hold))
    target = path if args.in_place and path != "-" else args.out
    _write(csvio.format_table(result.table), target, bom=True)

    lines = list(result.warnings)
    lines.append(f"Measured {result.measured} row(s); {result.prior} already had a response.")
    budget = oracle.budget
    if budget.total is not None:
        lines.append(f"Budget: {budget.used} of {budget.total} used, {budget.remaining} left.")
    best = int(np.argmin(result.y))
    unit = f" {oracle.response.unit}" if oracle.response.unit else ""
    lines.append(f"Best so far: {result.y[best]:.4g}{unit} in row {best + 1}.")
    _say(*lines, *notes)


def cmd_truth(args) -> None:
    notes: list[str] = []
    oracle = _oracle(args, notes)
    unit = f" {oracle.response.unit}" if oracle.response.unit else ""
    print("Instructor view (keep this from students in blind mode)")
    if not oracle.scenario.is_identity:
        s = oracle.scenario
        print(f"Blind scenario {s.seed}: factors 1-6 drive Hartmann coordinates {[p + 1 for p in s.perm]}; "
              f"factors {[j + 1 for j, f in enumerate(s.flip) if f]} run reversed.")
    for m in oracle.local_minima():
        print(f"{m['label'].capitalize()}: {m['y']:.4g}{unit} at {_fmt_point(oracle, m['x'])}")
    if args.table:
        table = csvio.parse_table(csvio.read_text(args.table))
        X, y, _, _ = csvio.read_points(table, oracle)
        measured = ~np.isnan(y)
        if not measured.any():
            print("The table has no measured rows yet.")
            return
        rows = np.flatnonzero(measured)
        best = rows[np.argmin(y[rows])]
        true = oracle.true_response(X[best])
        gap = true - oracle.optimum()["y"]
        print(f"Best measured row {best + 1}: observed {y[best]:.4g}{unit}, true {true:.4g}{unit}, "
              f"{gap:.4g}{unit} above the optimum.")
        print(f"  at {_fmt_point(oracle, X[best])}")


def cmd_optimize(args) -> None:
    notes: list[str] = []
    oracle = _oracle(args, notes)
    if oracle.budget.total is None:
        raise UsageError("optimize needs a budget: --budget N, or one in the settings file")
    strategy = Strategy(method=args.strategy, acquisition=args.acquisition, initial_runs=args.initial,
                        batch_size=args.batch, kernel=args.kernel, xi=args.xi, kappa=args.kappa)
    opt = Optimizer(oracle, strategy, seed=args.opt_seed, factors=[f["name"] for f in oracle.factors()])
    opt.run()

    header = ["Run", "Step", "Phase"] + [csvio.factor_header(f) for f in oracle.factors()]
    header.append(csvio.response_header(oracle))
    rows = [[str(r["evaluation"]), str(r["step"]), r["phase"]] + [csvio._fmt(v) for v in r["x"]] + [csvio._fmt(r["y"])]
            for r in opt.log_rows()]
    _write(csvio.format_table(csvio.Table(header, rows)), args.out, bom=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"oracle": oracle.config, **opt.log()}, fh, indent=1)

    unit = f" {oracle.response.unit}" if oracle.response.unit else ""
    x, y = opt.best()
    run = int(np.argmin(opt.y)) + 1
    lines = [f"{strategy.label}: {len(opt.y)} evaluations in {len(opt.records)} steps.",
             f"Best measured: {y:.4g}{unit} at run {run}: {_fmt_point(oracle, x)}."]
    if args.truth:
        true = oracle.true_response(x)
        lines.append(f"True value there: {true:.4g}{unit}; the optimum is {oracle.optimum()['y']:.4g}{unit} "
                     f"(gap {true - oracle.optimum()['y']:.4g}{unit}).")
    _say(*lines, *notes)


def cmd_benchmark(args) -> None:
    notes: list[str] = []
    base = _oracle(args, notes)
    budget = base.budget.total
    if budget is None:
        raise UsageError("benchmark needs a budget: --budget N, or one in the settings file")
    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    unknown = [n for n in names if n not in BENCHMARK_STRATEGIES]
    if unknown or not names:
        raise UsageError(f"unknown strategies {unknown}; choose from {', '.join(BENCHMARK_STRATEGIES)}")
    strategies = []
    for n in dict.fromkeys(names):
        settings = dict(BENCHMARK_STRATEGIES[n])
        if settings["method"] == "bo":
            settings.update(batch_size=args.batch, kernel=args.kernel)
        strategies.append(Strategy(**settings))
    config = {**base.config, "budget": None}

    def make_oracle(r: int) -> HartmannOracle:
        return HartmannOracle.from_config({**config, "seed": base.seed + r})

    interactive = sys.stderr.isatty()

    def progress(done: int, total: int, label: str) -> None:
        if interactive:
            print(f"\rbenchmark: {done}/{total} runs ({label})".ljust(60), end="", file=sys.stderr, flush=True)

    result = run_benchmark(make_oracle, strategies, budget, args.replicates, seed=args.opt_seed,
                           optimum=base.optimum()["y"], progress=progress)
    if interactive:
        print(file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"oracle": config, **result.to_dict()}, fh)
    if args.csv:
        rows = result.rows()
        table = csvio.Table(list(rows[0]), [[csvio._fmt(v) if isinstance(v, float) else str(v) for v in r.values()]
                                            for r in rows])
        _write(csvio.format_table(table), args.csv, bom=True)

    unit = f" {base.response.unit}" if base.response.unit else ""
    checkpoints = sorted({max(1, round(budget * f)) for f in (0.25, 0.5, 0.75, 1.0)})
    print(f"True value at the best measured point, median [25th-75th percentile] over {args.replicates} "
          f"replicates. Optimum {base.optimum()['y']:.4g}{unit}.")
    print("Strategy".ljust(16) + "".join(f"after {n}".rjust(24) for n in checkpoints))
    for label in result.labels:
        cells = []
        for n in checkpoints:
            st = result.at(n)[label]
            cells.append(f"{st['median']:.3f} [{st['q25']:.3f}, {st['q75']:.3f}]".rjust(24))
        print(label.ljust(16) + "".join(cells))
    _say(*notes, *(f"{label}: {sec:.1f} s" for label, sec in result.seconds.items()))


COMMANDS = {"info": cmd_info, "design": cmd_design, "setup": cmd_setup, "run": cmd_run, "truth": cmd_truth,
            "optimize": cmd_optimize, "benchmark": cmd_benchmark}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except (UsageError, ValueError, BudgetExhausted, OSError, json.JSONDecodeError) as exc:
        print(f"hartmann {args.command}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
