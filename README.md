# Hartmann 6D Optimization Simulator

A teaching and testing tool for surrogate modeling, response surface methods
and Bayesian optimization. Students optimize a fictional PECVD deposition
process with six factors and a limited budget of "wafers". Behind it is the
Hartmann 6D test function, which has a sharp global minimum and a broad,
flat second basin that is easy to settle into.

It is a sibling of the [race car DOE simulator](https://github.com/jeffkellyster/DOE_Racecar_sim).
There are three ways to use the same engine:

| | For | Needs | Status |
|---|---|---|---|
| **Command line** | running designs from JMP or a spreadsheet | Python 3.11+, NumPy, SciPy | available now |
| **Python package** | scripts, notebooks, JMP's Python | Python 3.11+, NumPy, SciPy | available now |
| **Browser app** | anyone; open a link, no install | a modern browser | available now |
| **JMP add-in** | students designing and analyzing in JMP | JMP 18 or 19 | Phase 5 |

## Browser app

- **Hosted:** <https://jeffkellyster.github.io/Hartman_Simulator/> (this works once GitHub Pages is switched on for the repository). Python runs inside the page through Pyodide, so there's nothing to install. The first load downloads about 30 MB and takes a few seconds; after that it's cached.
- **Locally**, after the quick start below:

  ```
  python scripts/serve.py            # then open http://localhost:8000/
  python scripts/serve.py --native   # same page; the engine runs in this Python process (no internet needed, faster)
  ```

What's in it:
- **Lab:**
  - Choose the settings: units, noise, budget, seed, and a blind scenario.
  - Plan runs: a design, a table pasted from JMP, or runs the optimizer suggests. Then measure them, which spends budget.
  - Watch the model: a Gaussian process or quadratic RSM, shown as a two-factor contour of its prediction, uncertainty or acquisition value. The other four factors sit at the slider values.
  - The profiler shows one factor at a time, and a convergence chart tracks the best run.
  - *Auto-run* lets Bayesian optimization spend the rest of the budget.
  - Runs export as CSV for JMP.
  - The lab is saved in the browser, so a reload keeps it.
- **Compare strategies:** median and interquartile range of the best-so-far value vs. runs, over 20 seeded replicates, for random search, LHS + RSM, BO with expected improvement, and BO with a confidence bound. You can also run your own smaller comparison in the page.
- **Instructor view:** shows the true surface, the optimum, the second minimum, and how far the best run is from the optimum. It works on the honor system, because everything runs in the browser.
- **About:** a 10-minute first session for someone opening it cold.

## Quick start (about 2 minutes)

```
git clone https://github.com/jeffkellyster/Hartman_Simulator.git
cd Hartman_Simulator
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                  # a few seconds
```

## Command line: design, measure, repeat

```
hartmann info                                        # the six factors, their ranges, the response
hartmann design maximin --runs 20 --seed 1 -o design.csv
hartmann setup --noise-sd 0.05 --budget 60 --scenario 7 -o lab.json   # instructor, once per class
hartmann run design.csv --config lab.json -o results.csv
```

- **`design`** writes a table with a run number, the six factors in engineering units, and an empty response column.
  - Space filling: `random`, `lhs` (Latin hypercube), `maximin` (maximin Latin hypercube).
  - Classical: `full` and `fractional` factorials, `pb` (Plackett–Burman), `ccd` (central composite; `--alpha face|rotatable`, `--inscribed`) and `bbd` (Box–Behnken).
  - `hartmann design --help` lists the options.
- **`run`** measures every row that has no response yet and writes the table back with the response filled in.
  - Rows that already have a response count as earlier measurements. So you can augment a results table (in JMP or by hand), run it again with `--in-place`, and the new rows continue the same noise stream against the same budget.
  - Any other columns, such as JMP's Pattern column, pass through unchanged.
  - A factor the table leaves out is held at its center, or at `--hold "RF power=400"`.
- **Designs from JMP:** save or copy a JMP design table (CSV, or tab-separated from the clipboard) whose columns are named after the factors, for example `RF power` or `RF power (W)`. Then `hartmann run` it and open the results back in JMP. A response column called `Y`, JMP's default, is filled in too.
- **`setup`** saves the seed, noise, budget and blind scenario to a settings file, so every student run uses the same lab. Anything can also be passed straight to `run` (`--seed`, `--noise-sd`, `--hetero`, `--budget`, `--scenario`).
- **`truth`** is the instructor view: `hartmann truth --config lab.json --table results.csv` shows where the optimum is and how far the best measured row is from it.

`python -m hartmann` works the same as `hartmann`.

## Optimization: surrogates, acquisitions, and a benchmark

```
hartmann optimize --strategy bo --budget 40 --noise-sd 0.05 --truth -o log.csv
hartmann benchmark --budget 60 --replicates 20 --noise-sd 0.05 -o bench.json --csv curves.csv
```

- **`optimize`** lets one strategy spend the whole budget on its own and logs every run: its step, phase, point and response.
  - It's a demo, or an answer key to set beside a class's results.
  - `--json` also saves each step's model fit and acquisition values.
- **Strategies** (`--strategy`):
  - `random`: points chosen at random.
  - `rsm`: a maximin Latin hypercube big enough for the full quadratic (30 runs in 6D). Each step then fits the quadratic and measures its predicted minimum.
  - `bo`: a 12-run maximin Latin hypercube, then Bayesian optimization.
    - Model: a Gaussian process (Matern 5/2 by default; `--kernel matern32|rbf`).
    - Acquisition (`--acquisition`): `ei` expected improvement, `pi` probability of improvement, `ucb` confidence bound, and the pure baselines `exploit` and `explore`.
    - `--batch q` proposes q points per step.
- **`benchmark`** runs several strategies over seeded replicates. It prints the median and interquartile range of the best-so-far value at four budget checkpoints.
  - The value reported is the *true* (noiseless) value at the best measured point.
  - `-o` writes JSON for the browser app. `--csv` writes the curves in long format, for JMP's Graph Builder.
  - Strategy names: `random`, `rsm`, `bo`, `bo-pi`, `bo-ucb`, `exploit`, `explore`.
  - For long benchmarks, run with `OPENBLAS_NUM_THREADS=1`. It's the same speed on a quarter of the CPU, because the matrices are small.
  - A 20-replicate, 60-wafer run of five strategies takes about 5 minutes.

In Python, the same kernel (`seqopt`) drives any function that has a dimension, bounds, a budget and an `evaluate` method:

```python
from seqopt.loop import Optimizer, Strategy
from seqopt import testfunctions

oracle = testfunctions.branin(budget=30)
opt = Optimizer(oracle, Strategy("bo", acquisition="ei"), seed=0)
opt.run()
opt.best()          # best point and value; opt.log() has every step's model and acquisition
```

## Python

Measure some wafers:

```python
from hartmann import HartmannOracle, NoiseModel

fab = HartmannOracle(noise=NoiseModel(sd=0.05), seed=1, budget=30)
fab.factors()                    # the six process factors, their units and ranges
fab.evaluate([350, 2.75, 105, 225, 450, 16.5])  # one wafer -> non-uniformity in %
fab.budget                       # Budget(total=30, used=1)
fab.ledger                       # every measurement, in order
```

- **Seeds:** the same `seed` always gives the same measurements.
- **Replays:** `HartmannOracle.from_config(fab.config)` rebuilds the oracle, and `fab.replay()` re-measures the ledger to show the run reproduces.

## The process students see

| Factor | Unit | Range |
|---|---|---|
| Susceptor temperature | °C | 250–450 |
| Chamber pressure | Torr | 0.5–5.0 |
| Deposition time | s | 30–180 |
| SiH4 flow | sccm | 50–400 |
| RF power | W | 100–800 |
| Electrode spacing | mm | 8–25 |

- **Response:** thickness non-uniformity (%) = 6.0 + 1.5·f(x). Lower is better, and the best achievable is about 1.02%.
- **Units:** `HartmannOracle(units="unit")` or `units="coded"` works on [0,1]⁶ or [−1,1]⁶ instead and returns f itself.

### Options
- **Noise:** Gaussian, added to the reported response, with an SD you choose.
  - With `NoiseModel(sd=..., hetero=h)`, the SD grows along RF power (or any other factor you pick), so high-power wafers are noisier.
  - Evaluation k's noise comes from `(seed, k)`. It doesn't matter how the evaluations were batched, and repeating a point gives a fresh replicate.
- **Budget:** each point costs one evaluation. A batch that would overrun the budget is refused whole.
- **Blind mode:** `scenario=<seed>` reorders and reverses the factors, so the published optimum can't be looked up.
  - `fab.optimum()` and `fab.local_minima()` give the instructor the answers in the student's own units.
  - It's a disguise on the honor system, not security.

## The function

f(x) = −Σᵢ αᵢ exp(−Σⱼ Aᵢⱼ (xⱼ − Pᵢⱼ)²) on [0,1]⁶, from the
[SFU library](https://www.sfu.ca/~ssurjano/hart6.html). Its global minimum is
f(x*) = −3.32237 at x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652,
0.6573), and the test suite checks this.

- **Rescaled form:** `hartmann6(x, form="rescaled")` is Picheny et al.'s version, with mean 0 and variance 1.
  - The formula printed on the SFU page leaves out a logarithm, so as printed it has a mean of about −1.46.
  - The version with the logarithm really does have mean 0 and variance 1.
  - The printed version is available as `form="sfu_rescaled"`.
- **Local minima:** the SFU page says 6, but a 40,000-start search finds 2 in the unit cube:
  - the global minimum;
  - a second at −3.20316, near (0.405, 0.882, 0.846, 0.574, 0.139, 0.039), which about 31% of random starts end in.

  Both are in `hartmann.LOCAL_MINIMA`.

## Layout

- `hartmann/` is the engine: the function, noise, process mapping, blind scenarios, the oracle, CSV tables (`csvio`) and the command line (`cli`).
- `seqopt/` is the reusable optimization kernel, not tied to Hartmann. It contains:
  - the oracle contract;
  - the design generators;
  - the Gaussian process and quadratic RSM surrogates;
  - the acquisition functions;
  - the optimization loop;
  - the benchmark runner;
  - the Branin and Rosenbrock test functions.
- `web/` is the browser app. It's static, and the engine runs in a Pyodide Web Worker. `web/data/benchmark.json` is the saved comparison; rebuild it with `scripts/build_benchmark.py`.
- `scripts/` has `serve.py` (local server, with a `--native` mode), `build_site.py` (the GitHub Pages build) and `build_benchmark.py`.
- `tests/` is the pytest suite.
- `CLAUDE.md` holds the requirements, build order and decision log.

## License

MIT
