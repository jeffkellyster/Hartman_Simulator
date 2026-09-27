# Hartmann 6D Optimization Simulator

A teaching and testing tool for surrogate modeling, response surface methods (RSM) and Bayesian optimization (BO), built on the Hartmann 6D function. It follows the pattern of Jeff's race car DOE simulator ([DOE_Racecar_sim](https://github.com/jeffkellyster/DOE_Racecar_sim)): a Python engine, a browser app for exploring, and a JMP integration for design and analysis.

**Top priority: a colleague can open it cold and use it.**

## Hard requirements
1. **Cold start.**
   - Clone to running in under 10 minutes, or open the hosted browser app with nothing to install.
   - Dependencies stay light: NumPy and SciPy at runtime, nothing else.
2. **Deterministic engine.**
   - Noise enters only as an optional, seeded, additive term.
   - Any run can be reproduced from its seed and config.
3. **Budget.** Each point evaluated costs one evaluation, like measuring a wafer. A batch of q costs q.
4. **Reusable kernel.** `seqopt` is not tied to Hartmann, so it can later drive Branin-Hoo, Rosenbrock and the race car sim. It never imports `hartmann`, and a test enforces that.
5. **Runs inside JMP's Python.** JMP 18 embeds Python 3.11 and JMP 19 embeds 3.13. CI tests both.

## The function
f(x) = −Σᵢ αᵢ exp(−Σⱼ Aᵢⱼ (xⱼ − Pᵢⱼ)²) on [0,1]⁶. The constants are in `hartmann/function.py`, from https://www.sfu.ca/~ssurjano/hart6.html.
- **Global minimum:** f(x*) = −3.32237 at x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573). A unit test checks this against the SFU values.
- **Forms:**
  - `standard`
  - `rescaled`: −(2.58 + ln S)/1.94, which has mean 0 and variance 1. Its minimum is −1.94880.
  - `sfu_rescaled`: −(2.58 + S)/1.94, the formula exactly as printed on the SFU page. Its mean is −1.46 and variance 0.04; its minimum is −3.04246.
- **Local minima:**
  - The SFU page says 6. A 40,729-start L-BFGS-B search finds **2** in [0,1]⁶:
    - the global minimum;
    - a second at −3.20316, near (0.405, 0.882, 0.846, 0.574, 0.139, 0.039).
  - About 31% of random starts end in the second one. Its basin is flat in two directions (Hessian eigenvalues 0.30 and 0.65, against 18 and up at the global minimum), so it's an easy place for an optimizer to stop.

## Features (Jeff's brief)
1. **Oracle.**
   - Deterministic, with optional seeded additive Gaussian noise (settable SD, optionally heteroscedastic).
   - An evaluation budget and counter.
   - **Process mode:** six PECVD factors in engineering units, so students never see [0,1].
   - **Blind mode:** the optimum and the function are hidden from students and visible to the instructor.
2. **Designs.**
   - Space-filling: Latin hypercube, maximin LHS, random.
   - Classical: central composite, Box–Behnken, 2-level full and fractional factorials.
   - CSV export and import, so a design made in JMP can be run here.
3. **Sequential optimization kernel** (`seqopt`).
   - Surrogates: a GP (Kriging) with a Matern or RBF kernel, and quadratic RSM for comparison.
   - Acquisitions: EI, UCB, PI, and pure exploit and explore baselines, with batch proposals.
   - Loop: initial design → fit → propose → evaluate → update, with every step logged.
4. **Browser app.**
   - A two-factor contour of true vs surrogate side by side, with sliders for the other four factors. The true view is instructor only.
   - A profiler, the evaluated points, and a convergence plot.
   - Strategy comparison over N seeded replicates: random vs LHS+RSM vs BO, showing the median and spread of best-so-far vs budget.
5. **JMP.**
   - The engine can be imported as Python modules in a JMP Python project.
   - An add-in takes a design table, evaluates it through the oracle, and returns the responses, the same way the race car add-in does.

## Build order (one phase at a time; check in with Jeff at the end of each)
1. ~~Engine~~. Done: the function, noise, budget, process mapping, seeds, and tests (the optimum check and reproducibility).
2. **Current:** Designs module and CSV in and out. A CLI to run a design and get the responses back.
3. The optimization kernel (GP, acquisitions, loop) and the benchmark runner for strategy comparison.
4. The browser app: contours, profiler, convergence, comparison, and the instructor/blind toggle.
5. The JMP add-in, plus a short lab outline for Part 2 of the race car DOE class: RSM, then BO, on the same budget.

## Architecture
```
seqopt/      reusable kernel. oracle.py (Oracle protocol, Budget, BudgetExhausted)
             designs.py    random, LHS, maximin LHS, full/fractional factorial, Plackett-Burman,
                           central composite (face, rotatable, inscribed), Box-Behnken; coded matrix + .unit
             Later: gp, rsm, acquisition, loop, benchmark, testfunctions
hartmann/    function.py   the function, its forms, gradient, X_STAR / F_STAR, LOCAL_MINIMA
             noise.py      NoiseModel: seeded additive noise; sd·(1 + hetero·u_k)
             process.py    PECVD factors and the response map; process ↔ unit ↔ coded
             scenario.py   blind-mode disguise (seeded permutation + reflection)
             oracle.py     HartmannOracle: budget, ledger, config/replay, record(), truth for the instructor
             csvio.py      parse/format tables (CSV, TSV, ;), match columns, measure_table (the table is the ledger)
             cli.py        `hartmann info | design | setup | run | truth` (also `python -m hartmann`)
             Later: app (JSON dispatcher), jmp_adapter
web/ jmp/ scripts/ docs/lab/   later phases
tests/       pytest
```
- **Oracle units:** `process` (the default) takes PECVD engineering units and returns non-uniformity (%) = 6.0 + 1.5·f, where lower is better and the best is ≈1.02%. `unit` ([0,1]) and `coded` ([−1,1]) return f itself.
- **Noise:** the draw for evaluation k comes from `default_rng([seed, k])`. It's independent of batching, and a repeated point gives a fresh replicate. The SD is in the units of the reported response.
- **Browser app (Phase 4):** a static site. The same engine runs in Pyodide in a Web Worker, as in the race car sim, with no server at runtime. It's hosted on GitHub Pages; the repo is public.
- **JMP (Phase 5):**
  - A `.jmpaddin` bundles the wheel, which is installed with `jmputils.jpip`.
  - JSL calls Python with `Python Send`, then `Python Submit`, then `Python Get` and `Parse JSON`, all guarded by `Try( Python Init(), 0 )`.
  - A "Suggest Next Points" menu item lets JMP drive the BO loop.

## Decision log
| Date | Decision |
|---|---|
| 2026-09-27 | Runtime deps are **NumPy + SciPy only**. The GP is our own readable code; scikit-learn is dev-only, to cross-check it. BoTorch is ruled out: about 125 MB of torch per JMP Python version, and it doesn't run in Pyodide |
| 2026-09-27 | Support Python 3.11 (JMP 18) and 3.13 (JMP 19), and test both. Keep version floors loose: numpy ≥2.5 and scipy ≥1.18 need Python 3.12+ |
| 2026-09-27 | "Rescaled" means the log form, which really does have mean 0 and variance 1. The SFU formula as printed is kept as `sfu_rescaled` |
| 2026-09-27 | Blind mode is a seeded disguise (a factor permutation plus reflection) with an honor-system instructor toggle. It's a static site, so there is no real secrecy, and the docs say so |
| 2026-09-27 | The process is PECVD film deposition. The factors are susceptor temperature 250–450 °C, pressure 0.5–5 Torr, deposition time 30–180 s, SiH4 flow 50–400 sccm, RF power 100–800 W and electrode spacing 8–25 mm. The response is thickness non-uniformity (%) = 6.0 + 1.5·f |
| 2026-09-27 | The import packages are `hartmann` and `seqopt`, never `engine`. JMP has one shared Python environment, and the race car sim already installs `engine` |
| 2026-09-27 | Browser: a Pyodide worker, pinned to the race car's version (0.27.7). NumPy loads first and SciPy in the background. Canvas charts, no chart library |
| 2026-09-27 | Noise for evaluation k comes from `default_rng([seed, k])`. A seed of None picks one at random and records it in `config` |
| 2026-09-27 | Heteroscedastic noise grows along one factor, in the student's factor order. The default is RF power |
| 2026-09-27 | A batch that would overrun the budget is refused whole, and points are validated before anything is charged |
| 2026-09-27 | Git: push straight to `main`, with check-ins in chat (Jeff's choice) |
| 2026-09-27 | There are 2 local minima in [0,1]⁶, not the 6 the SFU page says. Both are stored in `LOCAL_MINIMA`, and a test checks that a multistart finds nothing else |
| 2026-09-27 | Phase 2: the race car sim's design generators are ported into `seqopt.designs`, generic over factor count, with the design as a coded matrix plus `.unit` |
| 2026-09-27 | Box–Behnken uses Box and Behnken's (1960) tables for 3 to 7 factors (6 factors: 48 runs + 6 centers = 54, matching JMP's run counts) |
| 2026-09-27 | Maximin LHS takes the best of 20 random LHS by phi_p (p = 15), then keeps within-column swaps that lower phi_p. For 20–60 runs in 6D this about doubles the closest-pair distance of a plain LHS |
| 2026-09-27 | The CCD gets an `inscribed` option (divide by alpha), because Hartmann's region ends at the box. With alpha > 1 and no inscribing, the CLI refuses and says why |
| 2026-09-27 | Results tables are the ledger. Rows that already have a response count against the budget and take the first evaluation numbers, and new rows continue from there, so two sittings give the same numbers as one |
| 2026-09-27 | CSV import: missing factors are held at their center (with a warning) or at `--hold`. `Y` (JMP's default response name) is accepted as the response column. Other columns pass through untouched |
| 2026-09-27 | Tables are written as UTF-8 with a BOM, so Excel shows °C. Tables are read as UTF-8, falling back to Windows-1252. Responses are written to 6 significant figures |
| 2026-09-27 | `hartmann setup` writes the oracle config as JSON, which a class shares. It includes the blind scenario (honor system) |

## Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
- Tests: `pytest` (a few seconds)
- CLI: `hartmann design maximin --runs 20 -o d.csv`, `hartmann run d.csv --config lab.json -o r.csv`, `hartmann truth --config lab.json --table r.csv`

## Conventions
- Follow the race car engine style:
  - a prose docstring at the top of each module;
  - frozen dataclasses;
  - readable `ValueError`s that name the factor and its range;
  - a fresh `default_rng` per draw, never global RNG state.
- Tests: pytest, plain `def test_<behavior>()` functions, fixed seeds, fast (the whole suite runs in seconds).
- A point (shape (6,)) gives a float, and a batch (shape (n, 6)) gives an array.
- **BLAS threads:** many tiny linear-algebra calls from several processes oversubscribe OpenBLAS and run up to 70× slower. Set `OPENBLAS_NUM_THREADS=1`, or use `threadpoolctl`, for multi-process benchmarks.
- Pyodide is 32-bit, so don't pass int64 arrays to NumPy functions that expect indices (see the race car sim's decision 29).
