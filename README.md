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
| **Python package** | scripts, notebooks, JMP's Python | Python 3.11+, NumPy, SciPy | available now |
| **Browser app** | anyone; open a link, no install | a modern browser | Phase 4 |
| **JMP add-in** | students designing and analyzing in JMP | JMP 18 or 19 | Phase 5 |

## Quick start (about 2 minutes)

```
git clone https://github.com/jeffkellyster/Hartman_Simulator.git
cd Hartman_Simulator
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                  # about 1 second
```

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

- `hartmann/` is the engine: the function, noise, process mapping, blind scenarios and the oracle.
- `seqopt/` is the reusable optimization kernel. It isn't tied to Hartmann; designs, GP, RSM, acquisitions and the loop arrive in Phases 2–3.
- `tests/` is the pytest suite.
- `CLAUDE.md` holds the requirements, build order and decision log.

## License

MIT
