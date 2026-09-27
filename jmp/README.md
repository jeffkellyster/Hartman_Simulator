# Using the Hartmann simulator from JMP

JMP 18 (Python 3.11) and JMP 19 (Python 3.13) both work. The engine needs NumPy and SciPy,
which the add-in installs for you.

## Install once

1. Open `jmp/HartmannSimulator.jmpaddin` in JMP (File > Open, or drag it onto the Home window).
2. **Add-Ins > Hartmann Simulator > Install or Update Engine.**
   - It installs the engine bundled in the add-in, plus NumPy and SciPy, into JMP's Python.
   - It then checks the engine: it should report f(x*) = -3.32237.
   - Repeat it after updating the add-in, and restart JMP afterwards.
   - JMP 18 and JMP 19 keep separate Python packages, so install in each one you use.

## The lab, step by step

1. **Set Up a Design...**
   - Choose a design platform: Custom, Response Surface, Space Filling, Screening, Definitive
     Screening or Full Factorial.
   - It opens with the six process factors and their ranges, and the response *Thickness
     non-uniformity (%)* set to Minimize.
   - Uncheck a factor to leave it out, or narrow a range to zoom in. A factor with no column is held
     at the center of its range; to hold it somewhere else, add a column for it filled with that value.
   - Set the number of runs in the platform, then Make Table.
2. **Measure This Table...** measures every row that has no response yet. Each row costs one run of
   the budget.
   - The first time, you choose the lab: noise SD, how much the noise grows along RF power, budget,
     seed and an optional blind scenario.
   - The settings are stored in the table (table variable *Hartmann lab*, with a readable
     companion *Hartmann lab settings*) and used for every later measurement.
   - Three scripts are attached: *RSM: full quadratic with profiler*, *Gaussian process*, and
     *Convergence: best so far*. For the last one, *Run* and *Best so far* columns are added.
3. **Augment and repeat.** Rows that already have a response are earlier measurements: they count
   against the budget, and new rows continue the same seeded noise stream. So you can:
   - use JMP's Augment Design, add rows by hand, or add suggested rows;
   - then Measure This Table again, which measures only the new rows.
4. **Suggest Next Runs...** fits a model to the measured rows and appends the rows it would measure
   next, selected and without a response. Then Measure This Table.
   - Methods: Bayesian optimization, with expected improvement, probability of improvement, a
     confidence bound, or pure exploit / explore; quadratic RSM (go to the predicted minimum); or
     random.
   - Repeating Suggest then Measure is the Bayesian optimization loop, driven from JMP.
   - Suggestions never go past the budget; rows still waiting to be measured count.
   - Every factor needs a column.
5. **Instructor View...** shows where the optimum and the second minimum are, in this table's
   units, and how far the best measured row is from the optimum. It works on the honor system.

**Lab Settings for This Table...** shows or changes the stored settings. Change them before
measuring: rows already measured keep their values.

## Table conventions

- **Factor columns:** named after the factors, by label or name, with or without units. For
  example `RF power`, `rf_power_w` or `RF power (W)`. Other columns (Pattern, notes) are left alone.
- **The response column:** `Thickness non-uniformity (%)`, `non_uniformity_pct`, or JMP's default
  `Y`. It is created if missing.
- **The same class, the same numbers:** give everyone the same lab settings and seed, and the same
  rows measure the same. Different seeds give each team its own noise; different blind scenarios
  give each team its own optimum location.

| Factor | Unit | Range |
|---|---|---|
| Susceptor temperature | °C | 250–450 |
| Chamber pressure | Torr | 0.5–5.0 |
| Deposition time | s | 30–180 |
| SiH4 flow | sccm | 50–400 |
| RF power | W | 100–800 |
| Electrode spacing | mm | 8–25 |

The best possible non-uniformity is about 1.02 %. A second, broader basin bottoms out at
about 1.20 %.

## From Python inside JMP, without the menus

```python
from hartmann import jmp_adapter
lab = jmp_adapter.lab_config(noise_sd=0.05, budget=60, seed=7)
jmp_adapter.run_table(dt, lab)          # dt: a jmp.DataTable from Python Send( dt )
jmp_adapter.suggest_table(dt, lab, q=2) # points to add as rows
```

The whole engine is there too, for example `from hartmann import HartmannOracle` and
`from seqopt import Optimizer, Strategy`.

## The other direction

- **Browser app:** it imports a JMP table. Copy the table in JMP (it goes onto the clipboard as
  tab-separated text) and paste it into *Import from JMP or a CSV*.
- **Command line:** `hartmann run` measures a CSV saved from JMP the same way this add-in does.
- **Exports:** results from both come back to JMP as CSV.

## If something fails

- **Install errors on Windows about NumPy:** JMP loads NumPy when Python starts, so an upgrade
  can't replace it while JMP is running. Close JMP and install from the command line. JMP's
  *Python Create JPIP CMD()* makes a `jpip` script for this.
- **SSL or certificate errors during install:** in a JMP script window, run
  `Python Submit( "import jmputils; jmputils.jpip('install --upgrade', 'pip setuptools certifi')" );`
  then try again.
- **Data table errors:** the adapter uses JMP 18's `jmp.DataTable`: iterating `dt` for columns,
  `dt[name][i]`, `dt.new_column(name, jmp.DataType.Numeric)` and `dt.nrows`. If a later version
  differs, only `_column_names`, `_column_values` and `_write_values` in `hartmann/jmp_adapter.py`
  need changing.

Rebuild the add-in after editing `jmp/addin/` or the engine with `python scripts/build_release.py`.
It rebuilds the bundled wheel, this add-in, the browser site, and a zip for colleagues in `dist/`.
