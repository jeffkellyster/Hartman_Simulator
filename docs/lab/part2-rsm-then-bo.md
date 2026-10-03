# Part 2 lab: response surfaces, then Bayesian optimization, on the same budget

A follow-on to Part 1 of the race car DOE class (screening and aliasing). Teams tune a fictional
PECVD deposition process for the most uniform film, first the classical way (screening, a response
surface, the profiler) and then with Bayesian optimization, each with the same budget of 60 wafers.
The debrief compares the two against the truth.

**Time:** about 3 hours, or two 90-minute sessions (Parts A and B in the first, B and C in the second).
**Tools:** JMP 18 or 19 with the Hartmann Simulator add-in (`jmp/README.md`), or the browser app. Teams
can mix the two: a JMP table pastes straight into the browser app.

## Learning objectives

By the end, participants can:

1. Run a sequential RSM study (screen, fit a second-order model, optimize, confirm) and read its lack of fit.
2. Explain what a Gaussian process surrogate is, and what its prediction and uncertainty mean.
3. Explain how an acquisition function trades exploration against exploitation.
4. Say when each approach fits a problem, and why "best measured" overstates results under noise.

## The process

The six factors and their ranges:

| Factor | Unit | Range |
|---|---|---|
| Susceptor temperature | °C | 250–450 |
| Chamber pressure | Torr | 0.5–5.0 |
| Deposition time | s | 30–180 |
| SiH4 flow | sccm | 50–400 |
| RF power | W | 100–800 |
| Electrode spacing | mm | 8–25 |

- **Response:** thickness non-uniformity (%), to minimize.
- **Hidden from the students:**
  - the best possible is about 1.02 %;
  - a broader, flatter second basin sits at about 1.20 %;
  - most of the region is 4–6 %.

## Instructor setup (before class)

1. **Pick the lab:**
   - noise SD 0.05 %, budget 60;
   - one **blind scenario** for the class, so the published Hartmann optimum can't be looked up;
   - a **seed per team**, so teams get their own noise but the same landscape.

   Use a different scenario per team if you'd rather they couldn't compare notes.
2. **Give each team** their settings: noise, budget, seed, scenario.
   - **JMP:** they enter the settings the first time they run *Measure This Table*.
   - **Browser app:** Lab setup, then *Start a new lab*.
   - **Command line:** `hartmann setup --noise-sd 0.05 --budget 60 --seed 12 --scenario 7 -o team12.json`.
3. **Each part uses a fresh table** (JMP) or a new lab (browser), so each gets its own 60 runs.
4. **Answer key:** let the automated strategies play the same lab:
   `hartmann optimize --config team12.json --strategy bo --truth`, and the same with `--strategy rsm`.

## Part A: the classical route (about 70 minutes, 60 runs)

1. **Screen (18 runs).**
   - *Set Up a Design* > Screening Design: all six factors, a 16-run resolution IV fraction plus 2
     center points.
   - *Measure This Table*, then fit main effects.
   - Discuss:
     - Which factors look active?
     - What do the center points say about curvature?
     - A factor with a small main effect can still matter here. Why?
2. **Response surface (about 27 runs).**
   - On the 3–4 factors that matter, run a face-centered central composite (Response Surface Design)
     in their full ranges.
   - Hold the rest at their best screening level: add a column for each, filled with that value. A
     factor with no column is held at the center of its range.
   - Fit the full quadratic (the attached *RSM* script) and check lack of fit and R².
3. **Optimize and confirm (3 runs).**
   - Minimize in the profiler, then measure 3 replicates at the predicted optimum.
   - Compare the prediction with the confirmation runs.
4. **Spend the rest (about 12 runs)** as the team sees fit: a smaller RSM around the new point, a
   path of steepest descent, or more confirmation.

   Record: **the best setting, its confirmed value, and the runs spent at each stage.**

## Part B: Bayesian optimization (about 60 minutes, 60 runs)

1. **Space-filling start (12 runs).** *Set Up a Design* > Space Filling Design (Latin hypercube),
   all six factors. Measure.
2. **The loop.**
   - *Suggest Next Runs* (Bayesian optimization, expected improvement, 1–3 runs), then *Measure This
     Table*. Repeat.
   - Every 10 runs or so, open the *Gaussian process* script (or the browser app's Explore panel)
     and look at:
     - where the model is sure, and where it isn't (*Show: Uncertainty*);
     - the length scales: which factors matter, and do they match Part A?
     - where the next suggestion goes, and why there (*Show: Acquisition*).
3. **Try a change** once progress stalls:
   - a confidence bound (UCB) instead of EI;
   - or 3 runs per step, as if a tool processes a batch of wafers at once.
4. **Keep 2 runs to confirm** the final choice.

   Record the same things as in Part A.

### Discussion: how *Suggest Next Runs* picks a run (about 10 minutes, during the loop)

Suggest fits the add-in's own Gaussian process in Python (not JMP's platform), then asks it where
one more run is worth the most:

1. **Fit.**
   - The factors are scaled to 0–1.
   - The GP (Matern 5/2 by default) gets a length scale per factor and an estimated noise term.
   - These are chosen to make the measured runs most likely.
2. **The bar to beat** is the lowest *predicted* value at the measured runs, not the lowest
   measurement, which is partly a lucky draw.
3. **Score every candidate** with the chosen acquisition:
   - expected improvement: the average amount a run there would beat the bar;
   - or the confidence bound: the prediction minus 2 SD.
4. **Search.**
   - It scores 2,000 candidate settings, a quarter of them near the 5 best runs.
   - It polishes the top 4 with a local optimizer, inside the ranges.
5. **Batches.**
   - It picks the runs one at a time.
   - After each pick, the model pretends the result equals its prediction (the "kriging believer").
     That removes the uncertainty there, so the next pick goes elsewhere.

**Notes for the instructor:**
- Suggestions are seeded by the lab seed and the number of measured rows, so the same table always
  gets the same suggestion.
- The attached *Gaussian process* script runs JMP's own platform, for viewing only. Its length scales
  can differ a little from the add-in's fit.

**Ask the class:**
- Expected improvement is high in two kinds of places. What are they, and which kind is the next
  suggestion? Compare the browser app's *Show: Uncertainty* and *Show: Acquisition* maps.
- Why is the bar the best *prediction*, not the best measurement?
- A batch of 3 spreads out. Where would all three go without the pretend results?

## Part C: debrief (about 30 minutes)

1. **Reveal the truth.**
   - JMP: *Instructor View* on each team's tables.
   - Command line: `hartmann truth --config team12.json --table partB.csv`.
   - Browser: tick *Instructor view*.

   Put each team's best *true* value from A and B on the board.
2. **Show the Compare strategies tab** (20 replicates of 60 runs, noise 0.05 %). The median true
   value at 60 runs:
   - random search: about 3.5 %;
   - LHS + RSM: about 2.8 %;
   - BO with EI: about 1.1 %;
   - BO with a confidence bound: about 1.1 %.
3. **Discussion:**
   - **Why the quadratic misses.** One smooth bowl can't follow a surface with two basins and
     sharp walls. Lack of fit said so; did anyone listen?
   - **Screening in a curved world.** Main effects at the center of the region can hide a factor
     that matters a lot somewhere else.
   - **Exploration and exploitation.** Several BO runs settle in the flat second basin (1.20 %). What
     would have pulled them out? Compare the uncertainty and acquisition maps.
   - **Noise.** The best *measured* value is partly luck; the true value at that setting is higher.
     That is why both parts end with confirmation runs.
   - **When to use which.**
     - RSM: the region is known to be near the optimum, there are few factors, and you want an
       interpretable model.
     - BO: runs are expensive, there are more factors, the surface is irregular, and runs can be
       sequential.
     - Both need a sensible region to search.

## Variations

- **Harder:** budget 40 or noise SD 0.1 %.
- **Heteroscedastic:** noise grows along RF power (Lab Settings, growth 1–2). Replicates at the
  extremes reveal it. Does the "best" setting sit where the noise is worst?
- **Statistics class:** unit-cube units (browser app or command line) take the engineering away and
  show the function itself.
- **Assessment:** grade on the gap between the true value at the team's final choice and the
  optimum (Instructor View), and on the reasoning in a one-page summary. That summary should cover
  the final settings, the predicted and confirmed values, the runs spent per stage, and one plot.
