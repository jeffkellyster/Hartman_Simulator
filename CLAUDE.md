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
2. ~~Designs module and CSV in and out. A CLI to run a design and get the responses back~~. Done.
3. ~~The optimization kernel (GP, acquisitions, loop) and the benchmark runner for strategy comparison~~. Done.
4. ~~The browser app: contours, profiler, convergence, comparison, and the instructor/blind toggle~~. Done.
5. **Current:** The JMP add-in, plus a short lab outline for Part 2 of the race car DOE class: RSM, then BO, on the same budget.

## Architecture
```
seqopt/      reusable kernel. oracle.py (Oracle protocol, Budget, BudgetExhausted)
             designs.py    random, LHS, maximin LHS, full/fractional factorial, Plackett-Burman,
                           central composite (face, rotatable, inscribed), Box-Behnken; coded matrix + .unit
             gp.py         GaussianProcess: Matern 5/2 / 3/2 / RBF, ARD, estimated nugget, MLE (L-BFGS-B,
                           analytic gradient, seeded restarts, warm start); fit_fixed for fantasies
             rsm.py        QuadraticRSM: full quadratic OLS in coded units, same predict(X) -> (mean, sd)
             acquisition.py  EI, PI, UCB, exploit, explore; maximize (candidates + L-BFGS-B polish);
                           propose q points (kriging believer / constant liar)
             loop.py       Strategy (random | rsm | bo + settings), Optimizer (step/run, StepRecord log)
             benchmark.py  run_benchmark: strategies x seeded replicates -> best-so-far observed/true, median/IQR
             testfunctions.py  FunctionOracle, Branin-Hoo, Rosenbrock
hartmann/    function.py   the function, its forms, gradient, X_STAR / F_STAR, LOCAL_MINIMA
             noise.py      NoiseModel: seeded additive noise; sd·(1 + hetero·u_k)
             process.py    PECVD factors and the response map; process ↔ unit ↔ coded
             scenario.py   blind-mode disguise (seeded permutation + reflection)
             oracle.py     HartmannOracle: budget, ledger, config/replay, record(), truth for the instructor
             csvio.py      parse/format tables (CSV, TSV, ;), match columns, measure_table (the table is the ledger)
             cli.py        `hartmann info | design | setup | run | truth | optimize | benchmark` (also `python -m hartmann`)
             app.py        JSON dispatcher handle(json) -> json for the browser (and JMP): stateless calls
                           plus a cache of the last fitted model
             jmp_adapter.py  JMP: lab config (table variable "Hartmann lab"), run_table / suggest_table /
                           truth_table on a jmp.DataTable; dt access only in _column_names/_column_values/_write_values
web/         index.html, style.css (palette tokens), charts.js (Canvas charts), app.js (thin client),
             worker.js (Pyodide 0.27.7: NumPy -> "basic", then SciPy -> "full"), data/benchmark.json
scripts/     serve.py (--native: engine in-process at /api, worker_native.js), build_site.py, build_benchmark.py
jmp/         addin/ (addin.def id com.hartmann.optsim, addin.jmpcust menu, *.jsl), README.md (guide = Help),
             HartmannSimulator.jmpaddin (built, committed, engine wheel bundled)
docs/lab/    part2-rsm-then-bo.md (the Part 2 lab outline)
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
| 2026-09-27 | Phase 3 GP: our own ~250 lines. Its log likelihood matches scikit-learn's to 1e-6 and its predictions to 1e-7 at the same hyperparameters, and a test checks this. Bounds: length scales 0.05–10 on the unit cube (raised from 0.01 in Phase 4; see below), signal variance 0.01–100 and noise variance 1e-6–1 on the standardized scale. The first fit uses 4 random restarts; later fits in a loop warm-start from the previous fit, with 1 restart |
| 2026-09-27 | EI and PI use the plug-in incumbent (the lowest predicted mean at measured points) by default, because the lowest measurement is biased low under noise |
| 2026-09-27 | Batch proposals pick one point at a time and add a pretend result there: the model's mean (kriging believer) or the incumbent (constant liar). **A pretend result that beats the incumbent becomes the incumbent**; without that, a believer batch piles up on one spot (a test catches this) |
| 2026-09-27 | Default initial designs: maximin LHS of 2d runs for BO (12 in 6D), and quadratic terms + 2 for RSM (30 in 6D). RSM measures the fitted minimum, but at least 0.02 (unit cube) from any measured point, so it can't keep repeating one spot |
| 2026-09-27 | Benchmark replicate r gives every strategy the same seed (common random numbers). It records the best observed value and, more honestly under noise, the true value at the best observed point. Median and 25th/75th percentiles go to JSON (for the browser) and to long-format CSV (for JMP's Graph Builder) |
| 2026-09-27 | Phase 4 browser API: `hartmann/app.py` calls are stateless. The page sends the config and the runs with every call, and keeps them itself (in localStorage, so a reload keeps the lab). The only memory is a cache of the last fitted model, so moving a slider doesn't refit |
| 2026-09-27 | `seqopt/__init__` loads its names lazily, and `hartmann.app` imports SciPy only inside model calls. So the page can design and measure as soon as NumPy is in, while SciPy loads (a test checks that importing the app leaves SciPy unloaded) |
| 2026-09-27 | `scripts/serve.py --native` serves the same page but swaps the worker for one that POSTs to `/api` in the server process. Use it for development and for offline use. It sets BLAS to 1 thread, because threaded BLAS fighting the browser for the CPU made auto-run 17× slower (108 s vs 6.5 s) |
| 2026-09-27 | Charts: hand-drawn Canvas with no library. Series colors are palette slots 1–4 (blue, orange, aqua, yellow), validated for colorblind separation in light and dark mode. In light mode aqua and yellow are under 3:1 contrast, so the comparison has direct end labels (when they don't collide) and a table view. Surfaces use a one-hue blue ramp whose direction flips in dark mode. Series colors follow the strategy, not its rank |
| 2026-09-27 | GP length-scale floor raised from 0.01 to 0.05. With a handful of runs, fits hit 0.01 and spiked at the data points. Hartmann's narrowest well is about 0.17. On 10 replicates of 40 runs, BO did the same or slightly better (median EI −3.114 vs −3.058; UCB −2.854 vs −2.784) |
| 2026-09-27 | The Compare tab opens with a saved comparison, `web/data/benchmark.json` (about 11 KB, summaries only), built by `scripts/build_benchmark.py`: 20 replicates, 60 runs, noise 0.05 %, oracle seeds 100 + r. Rebuild it whenever the kernel changes. The page can also run a smaller live comparison, one run per call, with progress and Stop |
| 2026-09-27 | Suggest moves the slice to the first suggested point, so the contours and profiler show where it's going. Auto-run redraws the contours every 5 steps |
| 2026-09-27 | The Pages workflow always builds the site. It publishes only if GitHub Pages is switched on (it checks the API); otherwise it posts a notice with the one-time setting, instead of failing |
| 2026-09-27 | This build environment's network policy blocks cdn.jsdelivr.net, so the Pyodide path couldn't be run here. The page was tested end to end in headless Chromium through `--native` (light and dark mode, every tab, auto-run, import, live comparison) |
| 2026-09-27 | Phase 5 JMP add-in (id `com.hartmann.optsim`, JMP 18+). It mirrors the race car add-in's verified JSL patterns: `Python Send`, then `Python Submit` with a raw string, then `Python Get` and `Parse JSON`, after `Try( Python Init(), 0 )`. The menu has Install or Update Engine, Set Up a Design, Measure This Table, Suggest Next Runs, Lab Settings, Instructor View and Help |
| 2026-09-27 | In JMP the table is the ledger, as with the command line. The lab's config is stored in the table variable "Hartmann lab" (JSON), with a readable "Hartmann lab settings" beside it, so a table carries its own lab. Measure asks for the settings the first time |
| 2026-09-27 | The adapter builds every dialog message in Python and returns strings. JSL never has to interpret JSON nulls, and only reads `response_column`, `factor_columns`, `columns` and `points` |
| 2026-09-27 | Suggest Next Runs needs a column for every factor, so suggestions can be written back as rows. It caps q so that measured rows, waiting rows and the new ones together stay within the budget. JSL appends and selects the rows |
| 2026-09-27 | Measure attaches three scripts: an RSM full quadratic (`& RS` effects, with the profiler), a Gaussian process (nugget estimated), and a convergence Graph Builder over added `Run` and `Best so far` formula columns. Each is wrapped in `Try`, so a failing script never undoes a measurement |
| 2026-09-27 | The built `.jmpaddin` is committed with the engine wheel inside, so a colleague can download it and go. `tests/test_addin.py` fails when the committed add-in's scripts, README or bundled engine modules differ from the source. After changing the engine or `jmp/addin/`, run `scripts/build_release.py` |
| 2026-09-27 | JSL can't run here (no JMP). `tests/test_addin.py` checks what it can: bracket balance with strings and comments stripped, that every embedded Python block compiles, that every `jmp_adapter.<name>` it calls exists, the add-in id, includes and menu targets. The bundled wheel was installed into a fresh Python 3.13 the way Install or Update Engine does it, and checked. **The add-in still needs a first run in JMP 18 and 19** |
| 2026-09-27 | The Part 2 lab (docs/lab) uses 60 wafers for each of classical RSM (screen 18, then a CCD of about 27 on the vital few, then 3 confirmation runs, then 12 at the team's discretion) and BO (12 space-filling runs, then the suggest/measure loop, keeping 2 to confirm). The debrief uses the Instructor View and the Compare tab |

## Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
- Tests: `pytest` (a few seconds)
- CLI: `hartmann design maximin --runs 20 -o d.csv`, `hartmann run d.csv --config lab.json -o r.csv`, `hartmann truth --config lab.json --table r.csv`
- Browser app: `python scripts/serve.py` (Pyodide) or `python scripts/serve.py --native` (engine in-process); `python scripts/build_site.py` builds `site/`; `OPENBLAS_NUM_THREADS=1 python scripts/build_benchmark.py` rebuilds the saved comparison (about 4 minutes)
- JMP add-in: edit `jmp/addin/` or `jmp/README.md`, then `python scripts/build_release.py` (wheel, add-in, site, `dist/` zip); `python scripts/build_addin.py` repackages only the scripts
- Optimize and compare: `hartmann optimize --strategy bo --budget 40 --truth`, `hartmann benchmark --budget 60 --replicates 20 --noise-sd 0.05 -o bench.json --csv curves.csv`

## Conventions
- Follow the race car engine style:
  - a prose docstring at the top of each module;
  - frozen dataclasses;
  - readable `ValueError`s that name the factor and its range;
  - a fresh `default_rng` per draw, never global RNG state.
- Tests: pytest, plain `def test_<behavior>()` functions, fixed seeds, fast (the whole suite runs in seconds).
- A point (shape (6,)) gives a float, and a batch (shape (n, 6)) gives an array.
- **BLAS threads:** the GP's matrices are small, so OpenBLAS threads only spin.
  - A BO benchmark runs at the same wall-clock speed with `OPENBLAS_NUM_THREADS=1`, on a quarter of the CPU (measured: 4.6 s vs 4.7 s wall-clock; 4.4 s vs 17.6 s CPU).
  - Alongside another busy process, default threading slows small linear algebra by 10–70×.
  - Recommend the variable for benchmarks in the docs. Don't set it inside the package, because JMP's Python environment is shared.
- Pyodide is 32-bit, so don't pass int64 arrays to NumPy functions that expect indices (see the race car sim's decision 29).
