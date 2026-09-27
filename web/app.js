"use strict";

// Thin client over the Python engine (hartmann/app.py, running in a Web Worker).
// Every number comes from the engine. This file keeps the lab's state (the
// settings and the runs measured so far), asks the engine, and draws.

// --- Python worker --------------------------------------------------------------------------

const worker = new Worker("worker.js");
const waiting = new Map();
let nextId = 1;

function call(fn, args = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, fn, args });
  });
}

worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "status") setStatus(msg.text);
  else if (msg.type === "fatal") setStatus(`Could not start Python: ${msg.error}`, true);
  else if (msg.type === "ready") onReady(msg.level).catch((err) => setStatus(err.message, true));
  else if (msg.type === "result") {
    const waiter = waiting.get(msg.id);
    waiting.delete(msg.id);
    if (msg.response.ok) waiter.resolve(msg.response.result);
    else waiter.reject(new Error(msg.response.error));
  }
};

// --- State --------------------------------------------------------------------------------

const STORE_KEY = "hartmann-sim-lab-v1";
// Series colors follow the strategy, never its rank (index into --s1..--s6).
const STRATEGY_COLORS = { "Random": 0, "LHS + RSM": 1, "BO (EI)": 2, "BO (UCB)": 3, "BO (PI)": 4, "BO (exploit)": 5 };
const LIVE_DEFAULT = ["random", "rsm", "bo-ei", "bo-ucb"];

const state = {
  catalog: null,
  engine: "loading", // "basic" (NumPy: design and measure) or "full" (SciPy: models too)
  lab: null, // {config, factors, response, bounds}
  runs: [], // {x, y, yTrue, step, source}
  plan: null, // {points, title, info, source}
  center: null, // the slice point, in the lab's units
  axes: [0, 1],
  model: "gp",
  kernel: "matern52",
  layer: "mean",
  step: 0,
  instructor: false,
  truth: null,
  slice: null,
  profile: null,
  modelInfo: null,
  hits: {},
  auto: { running: false, stop: false },
  compare: { saved: null, live: null, which: "true", running: false, stop: false },
};

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

function setStatus(text, error = false) {
  const status = $("status");
  status.textContent = text;
  status.classList.toggle("error", error);
}

const fmt = Charts.fmt;
const unitOf = () => state.lab?.response.unit ?? "";
const withUnit = (v, unit = unitOf()) => `${fmt(v)}${unit ? ` ${unit}` : ""}`;
const factorUnit = (f) => (state.lab.config.units === "process" ? f.unit : "");
const factorTitle = (f) => (factorUnit(f) ? `${f.label} (${factorUnit(f)})` : f.label);
const responseTitle = () => state.lab.response.header;
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const runX = () => state.runs.map((r) => r.x);
const runY = () => state.runs.map((r) => r.y);

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({
      config: state.lab.config, runs: state.runs, center: state.center, axes: state.axes, model: state.model,
      kernel: state.kernel, layer: state.layer, step: state.step,
    }));
  } catch { /* storage unavailable: the lab still works, it just won't survive a reload */ }
}

function restore() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || "null");
  } catch {
    return null;
  }
}

// --- Startup ----------------------------------------------------------------------------------

let started = false;

async function onReady(level) {
  if (level === "full") state.engine = "full";
  else if (state.engine !== "full") state.engine = "basic";
  if (!started) {
    started = true;
    await start();
  }
  if (level === "full") {
    setStatus("Ready");
    updateControls();
    scheduleExplore();
  } else {
    setStatus("Ready to design and measure. Loading the models (SciPy)…");
  }
}

async function start() {
  state.catalog = await call("catalog");
  buildStaticControls();
  loadSavedComparison();
  const saved = restore();
  let restored = false;
  if (saved?.config) {
    try {
      state.lab = await call("oracle_info", { config: saved.config });
      state.runs = Array.isArray(saved.runs) ? saved.runs : [];
      state.center = saved.center ?? null;
      state.axes = saved.axes ?? [0, 1];
      state.model = saved.model ?? "gp";
      state.kernel = saved.kernel ?? "matern52";
      state.layer = saved.layer ?? "mean";
      state.step = saved.step ?? 0;
      restored = true;
    } catch {
      restored = false;
    }
  }
  if (!restored) {
    const d = state.catalog.defaults;
    state.lab = await call("lab", { units: d.units, noise_sd: d.noise_sd, hetero: d.hetero, budget: d.budget, seed: d.seed });
  }
  if (!state.center || state.center.length !== state.lab.factors.length) state.center = state.lab.factors.map((f) => f.center);
  afterLabChange();
}

function buildStaticControls() {
  const cat = state.catalog;
  const designSelect = $("design-type");
  designSelect.replaceChildren(...cat.designs.map((d) => el("option", { value: d.type }, d.label)));
  designSelect.addEventListener("change", renderDesignParams);
  renderDesignParams();
  $("method").replaceChildren(...cat.strategies.map((s) => el("option", { value: s.method }, s.label)));
  $("acquisition").replaceChildren(...cat.acquisitions.map((a) => el("option", { value: a.key }, a.label)));
  $("method").addEventListener("change", updateControls);
  $("live-strategies").replaceChildren(...cat.benchmark_strategies.map((s) =>
    el("label", {}, el("input", { type: "checkbox", value: s.key, checked: LIVE_DEFAULT.includes(s.key) }), s.key)));
}

function renderDesignParams() {
  const type = state.catalog.designs.find((d) => d.type === $("design-type").value);
  const box = $("design-params");
  box.replaceChildren();
  for (const p of type.params) {
    let input;
    if (p.kind === "bool") input = el("input", { type: "checkbox", "data-param": p.name, "data-kind": p.kind, checked: p.default });
    else if (p.kind === "choice") {
      input = el("select", { "data-param": p.name, "data-kind": p.kind },
        ...p.choices.map((c) => el("option", { value: c, selected: c === p.default }, String(c))));
    } else input = el("input", { type: "number", "data-param": p.name, "data-kind": p.kind, value: p.default, min: p.min, max: p.max, step: 1 });
    box.append(el("label", {}, p.label, input));
  }
}

function designParams() {
  const params = {};
  for (const input of $("design-params").querySelectorAll("[data-param]")) {
    const kind = input.dataset.kind;
    const name = input.dataset.param;
    if (kind === "bool") params[name] = input.checked;
    else if (kind === "choice") params[name] = Number.isNaN(Number(input.value)) ? input.value : Number(input.value);
    else params[name] = Number(input.value);
  }
  return params;
}

// --- Lab settings -------------------------------------------------------------------------------

function afterLabChange() {
  const c = state.lab.config;
  $("units").value = c.units;
  $("noise-sd").value = c.noise.sd;
  $("hetero").value = c.noise.hetero;
  $("budget").value = c.budget ?? "";
  $("seed").value = c.seed;
  $("blind").checked = Boolean(c.scenario);
  $("scenario").value = c.scenario?.seed ?? 1;
  $("noise-unit").textContent = unitOf();
  const hetero = state.lab.factors[c.noise.hetero_factor];
  $("hetero").parentElement.firstChild.textContent = `Noise growth along ${hetero.label} `;
  buildAxisControls();
  buildSliders();
  buildProfiler();
  state.truth = null;
  if (state.instructor) loadTruth();
  renderAll();
}

async function newLab() {
  if (state.runs.length && !confirm("Start a new lab? The measured runs will be cleared.")) return;
  try {
    const blind = $("blind").checked;
    const seed = $("seed").value === "" ? null : Number($("seed").value);
    state.lab = await call("lab", {
      units: $("units").value,
      noise_sd: Number($("noise-sd").value) || 0,
      hetero: Number($("hetero").value) || 0,
      budget: $("budget").value === "" ? null : Number($("budget").value),
      seed,
      scenario: blind ? Number($("scenario").value) || 0 : null,
    });
    state.runs = [];
    state.plan = null;
    state.step = 0;
    state.center = state.lab.factors.map((f) => f.center);
    save();
    afterLabChange();
    setStatus("New lab started");
  } catch (err) {
    setStatus(err.message, true);
  }
}

function budgetLeft() {
  const total = state.lab.config.budget;
  return total === null || total === undefined ? Infinity : total - state.runs.length;
}

function renderBudget() {
  const total = state.lab.config.budget;
  const used = state.runs.length;
  $("budget-fill").style.width = total ? `${Math.min(100, (100 * used) / total)}%` : "0";
  $("budget-text").textContent = total ? `${used} of ${total} runs used, ${Math.max(total - used, 0)} left` : `${used} runs, no budget`;
}

function updateControls() {
  const full = state.engine === "full";
  const busy = state.auto.running;
  const left = budgetLeft();
  $("suggest").disabled = !full || busy || left <= 0;
  $("auto").disabled = !full || busy || left <= 0;
  $("generate").disabled = busy;
  $("measure").disabled = busy;
  $("acquisition").disabled = $("method").value !== "bo";
  $("live-run").disabled = !full || state.compare.running;
  $("engine-note").textContent = full ? "" : "The models are still loading; designs and measuring already work.";
}

// --- Planning and measuring -------------------------------------------------------------------------

function setPlan(plan) {
  state.plan = plan;
  renderPlan();
  drawContours();
}

function renderPlan() {
  const box = $("pending");
  if (!state.plan) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  $("pending-title").textContent = state.plan.title;
  const left = budgetLeft();
  const n = state.plan.points.length;
  const over = n > left ? ` The budget has only ${left} left.` : "";
  $("pending-info").textContent = (state.plan.info ?? "") + over;
  $("measure").textContent = n === 1 ? "Measure 1 run" : `Measure ${n} runs`;
}

async function planDesign() {
  try {
    const kind = $("design-type").value;
    const result = await call("design", { config: state.lab.config, kind, params: designParams() });
    const info = result.info;
    const bits = [];
    if (info.resolution) bits.push(`Resolution ${info.resolution}.`);
    if (info.min_distance) bits.push(`Closest pair ${fmt(info.min_distance)} apart on the unit cube.`);
    if (info.note) bits.push(info.note);
    const label = state.catalog.designs.find((d) => d.type === kind).label;
    setPlan({ points: result.points, title: `${label}: ${result.points.length} runs planned`, info: bits.join(" "), source: "design" });
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function suggestRuns() {
  const method = $("method").value;
  const result = await call("suggest", {
    config: state.lab.config, X: runX(), y: runY(), method, acquisition: $("acquisition").value,
    q: Number($("batch").value) || 1, kernel: state.model === "gp" ? state.kernel : "matern52", seed: state.lab.config.seed,
  });
  const n = result.points.length;
  const what = method === "bo" ? state.catalog.acquisitions.find((a) => a.key === $("acquisition").value).label
    : state.catalog.strategies.find((s) => s.method === method).label;
  return { points: result.points, title: `${n} suggested run${n > 1 ? "s" : ""}`, info: `${what}. ${result.warnings.join(" ")}`.trim(), source: "suggested" };
}

async function suggest() {
  try {
    setStatus("Fitting the model and searching…");
    const plan = await suggestRuns();
    state.center = [...plan.points[0]];
    setPlan(plan);
    save();
    renderSliderValues();
    scheduleExplore();
    setStatus("Ready");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function measure() {
  if (!state.plan) return;
  const result = await call("measure", { config: state.lab.config, X: runX(), y: runY(), new: state.plan.points });
  state.step += 1;
  state.plan.points.forEach((x, k) => {
    state.runs.push({ x, y: result.y[k], yTrue: result.y_true[k], step: state.step, source: state.plan.source });
  });
  state.plan = null;
  save();
}

async function measureClicked() {
  try {
    await measure();
    renderAll();
    setStatus("Measured");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function autoRun() {
  state.auto = { running: true, stop: false };
  $("stop-auto").hidden = false;
  updateControls();
  try {
    while (!state.auto.stop && budgetLeft() > 0) {
      setStatus(`Auto-run: ${state.runs.length} of ${state.lab.config.budget} runs…`);
      state.plan = await suggestRuns();
      await measure();
      renderBudget();
      renderRuns();
      renderConvergence();
      if (state.step % 5 === 0) scheduleExplore(); // every step would slow the run down
    }
    setStatus(state.auto.stop ? "Stopped" : "Budget used up");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    state.auto = { running: false, stop: false };
    $("stop-auto").hidden = true;
    renderAll();
  }
}

async function importTable() {
  const text = $("import-text").value;
  if (!text.trim()) return;
  try {
    const result = await call("import_table", { config: state.lab.config, text });
    const measured = [];
    const planned = [];
    result.points.forEach((x, k) => (result.y[k] === null ? planned : measured).push({ x, y: result.y[k], yTrue: result.y_true[k] }));
    if (measured.length) {
      state.step += 1;
      for (const r of measured) state.runs.push({ ...r, step: state.step, source: "imported" });
    }
    const notes = [...result.warnings];
    if (measured.length) notes.push(`${measured.length} measured runs added.`);
    if (planned.length) {
      setPlan({ points: planned.map((r) => r.x), title: `${planned.length} imported runs to measure`, info: "", source: "imported" });
      notes.push(`${planned.length} runs without a response are planned.`);
    }
    if (budgetLeft() < 0) notes.push("The imported runs exceed the budget.");
    $("import-status").textContent = notes.join(" ");
    save();
    renderAll();
  } catch (err) {
    $("import-status").textContent = err.message;
  }
}

async function exportRuns() {
  if (!state.runs.length) return;
  const result = await call("export_table", { config: state.lab.config, X: runX(), y: runY() });
  const blob = new Blob(["﻿" + result.csv], { type: "text/csv" });
  const link = el("a", { href: URL.createObjectURL(blob), download: "hartmann-runs.csv" });
  document.body.append(link);
  link.click();
  link.remove();
}

// --- Rendering: everything ----------------------------------------------------------------------------

function renderAll() {
  renderBudget();
  renderPlan();
  renderRuns();
  renderConvergence();
  renderTruth();
  updateControls();
  scheduleExplore();
}

// --- Explore: contours, sliders, profiler -------------------------------------------------------------

function buildAxisControls() {
  const options = () => state.lab.factors.map((f, k) => el("option", { value: k }, f.label));
  $("axis-x").replaceChildren(...options());
  $("axis-y").replaceChildren(...options());
  $("axis-x").value = state.axes[0];
  $("axis-y").value = state.axes[1];
  $("model").value = `${state.model}:${state.model === "gp" ? state.kernel : ""}`;
  $("layer").value = state.layer;
}

function buildSliders() {
  const box = $("sliders");
  box.replaceChildren();
  state.lab.factors.forEach((f, k) => {
    const input = el("input", { type: "range", min: 0, max: 1000, step: 1, "aria-label": f.label, "data-k": k });
    input.addEventListener("input", () => {
      state.center[k] = f.low + (Number(input.value) / 1000) * (f.high - f.low);
      renderSliderValues();
      scheduleExplore();
    });
    input.addEventListener("change", save);
    box.append(el("div", { class: "slider", "data-k": k },
      el("span", { class: "name" }, f.label), el("span", { class: "value" }), input));
  });
  renderSliderValues();
}

function renderSliderValues() {
  state.lab.factors.forEach((f, k) => {
    const row = $("sliders").querySelector(`.slider[data-k="${k}"]`);
    row.querySelector("input").value = Math.round((1000 * (state.center[k] - f.low)) / (f.high - f.low));
    row.querySelector(".value").textContent = `${fmt(state.center[k])}${factorUnit(f) ? ` ${factorUnit(f)}` : ""}`;
    row.classList.toggle("on-axis", state.axes.includes(k));
  });
}

let exploreBusy = false;
let exploreAgain = false;

function scheduleExplore() {
  if (exploreBusy) {
    exploreAgain = true;
    return;
  }
  exploreBusy = true;
  renderExplore().catch((err) => setStatus(err.message, true)).finally(() => {
    exploreBusy = false;
    if (exploreAgain) {
      exploreAgain = false;
      scheduleExplore();
    }
  });
}

function canModel() {
  return state.engine === "full" && state.runs.length >= 2;
}

async function renderExplore() {
  if (!state.lab) return;
  const truth = state.instructor;
  const model = canModel();
  const [i, j] = state.axes;
  const args = { config: state.lab.config, X: runX(), y: runY(), center: state.center, model: model ? state.model : null, kernel: state.kernel, truth };
  state.slice = null;
  state.profile = null;
  state.modelError = null;
  if (model || truth) {
    try {
      state.slice = await call("slice2d", { ...args, i, j, n: 41, layer: state.layer });
      state.profile = await call("profile", { ...args, n: 41 });
      state.modelInfo = model ? await call("fit", { config: state.lab.config, X: runX(), y: runY(), model: state.model, kernel: state.kernel }) : null;
    } catch (err) {
      state.modelError = err.message;
    }
  } else state.modelInfo = null;
  drawContours();
  drawProfiler();
  renderModelInfo();
}

function layerLabel() {
  if (state.layer === "sd") return `Model uncertainty, SD${unitOf() ? ` (${unitOf()})` : ""}`;
  if (state.layer === "acquisition") return "Acquisition value (higher = more worth measuring)";
  return `Predicted ${state.lab.response.label.toLowerCase()}${unitOf() ? ` (${unitOf()})` : ""}`;
}

function drawContours() {
  if (!state.lab) return;
  const [i, j] = state.axes;
  const fi = state.lab.factors[i];
  const fj = state.lab.factors[j];
  const s = state.slice;
  const lastStep = state.runs.length ? state.runs[state.runs.length - 1].step : null;
  const points = state.runs.map((r, k) => ({ x: r.x[i], y: r.x[j], kind: r.step === lastStep ? "latest" : "run", run: k }));
  for (const p of state.plan?.points ?? []) points.push({ x: p[i], y: p[j], kind: "planned" });
  const markers = state.instructor && state.truth
    ? state.truth.local_minima.map((m, k) => ({ x: m.x[i], y: m.x[j], major: k === 0, label: m.label })) : [];
  const center = [state.center[i], state.center[j]];
  const common = { xLabel: factorTitle(fi), yLabel: factorTitle(fj), points, center };

  const trueGrid = s?.true;
  const modelGrid = s?.model;
  let trueDomain = trueGrid ? Charts.extent(trueGrid.flat()) : null;
  let modelDomain = modelGrid ? Charts.extent(modelGrid.flat()) : null;
  if (trueGrid && modelGrid && state.layer === "mean") {
    const both = Charts.extent([...trueGrid.flat(), ...modelGrid.flat()]);
    trueDomain = both;
    modelDomain = both;
  }

  $("true-cover").hidden = Boolean(trueGrid);
  if (!state.instructor) $("true-cover").innerHTML = "Hidden in the student view.<br>Measure runs and let the model show you the surface.";
  else if (!trueGrid) $("true-cover").textContent = "Loading…";
  if (trueGrid) {
    state.hits.true = Charts.heatmap($("contour-true"), { ...common, grid: trueGrid, x: s.x, y: s.y, domain: trueDomain, markers });
    Charts.colorbar($("colorbar-true"), trueDomain, `True ${state.lab.response.label.toLowerCase()}${unitOf() ? ` (${unitOf()})` : ""}`);
  } else {
    state.hits.true = null;
    Charts.setup($("contour-true"), Math.min($("contour-true").clientWidth, 420));
    Charts.setup($("colorbar-true"), 46);
  }

  const cover = $("model-cover");
  $("model-caption").textContent = state.model === "gp" ? "Gaussian process model" : "Quadratic RSM model";
  if (modelGrid) {
    cover.hidden = true;
    state.hits.model = Charts.heatmap($("contour-model"), { ...common, grid: modelGrid, x: s.x, y: s.y, domain: modelDomain, markers });
    Charts.colorbar($("colorbar-model"), modelDomain, layerLabel());
  } else {
    state.hits.model = null;
    cover.hidden = false;
    cover.textContent = state.modelError ? `The model could not be fitted: ${state.modelError}`
      : state.engine !== "full" ? "The models are loading…"
        : `Measure at least 2 runs to fit a model (${state.runs.length} so far).`;
    Charts.setup($("contour-model"), Math.min($("contour-model").clientWidth, 420));
    Charts.setup($("colorbar-model"), 46);
  }

  const items = [{ label: "Measured run", kind: "dot", color: "ink" }];
  if (state.runs.length) items.push({ label: "Latest step", kind: "dot", color: 1 });
  if (state.plan) items.push({ label: "Planned run", kind: "ring", color: 1 });
  if (markers.length) items.push({ label: "Optimum and second minimum", kind: "cross", color: "ink" });
  items.push({ label: "Crosshair: the slider values", kind: "line", color: "muted" });
  Charts.legend($("contour-legend"), items);
}

function contourHover(which, event) {
  const hit = state.hits[which];
  if (!hit) return Charts.hideTip();
  const [px, py] = Charts.position(event.target, event);
  const d = hit.toData(px, py);
  if (!d) return Charts.hideTip();
  const [i, j] = state.axes;
  const fi = state.lab.factors[i];
  const fj = state.lab.factors[j];
  const near = hit.nearest(px, py);
  const value = which === "true" ? `True: ${withUnit(d.value)}`
    : state.layer === "mean" ? `Predicted: ${withUnit(d.value)}` : state.layer === "sd" ? `SD: ${withUnit(d.value)}` : `Acquisition: ${fmt(d.value)}`;
  let html = `<div class="t-title">${escapeHtml(value)}</div>`
    + `<div class="t-muted">${escapeHtml(fi.label)} ${fmt(d.x)}${factorUnit(fi) ? ` ${factorUnit(fi)}` : ""}, ${escapeHtml(fj.label)} ${fmt(d.y)}${factorUnit(fj) ? ` ${factorUnit(fj)}` : ""}</div>`;
  if (near && near.kind !== "planned") {
    const run = state.runs[near.run];
    html += `<div>Run ${near.run + 1}: measured ${escapeHtml(withUnit(run.y))}</div><div class="t-muted">Plotted where its two factors put it; its other four differ from the sliders.</div>`;
  } else if (near) html += `<div>A planned run (not measured yet)</div>`;
  html += `<div class="t-muted">Click to move the slice here.</div>`;
  Charts.showTip(html, event);
}

function contourClick(which, event) {
  const hit = state.hits[which];
  if (!hit) return;
  const d = hit.toData(...Charts.position(event.target, event));
  if (!d) return;
  state.center[state.axes[0]] = d.x;
  state.center[state.axes[1]] = d.y;
  save();
  renderSliderValues();
  scheduleExplore();
}

function buildProfiler() {
  const box = $("profiler");
  box.replaceChildren();
  state.lab.factors.forEach((f, k) => {
    const canvas = el("canvas", { "data-k": k, "aria-label": `Profile of ${f.label}` });
    canvas.addEventListener("mousemove", (e) => profileHover(k, e));
    canvas.addEventListener("mouseleave", () => {
      Charts.hideTip();
      drawProfiler();
    });
    canvas.addEventListener("click", (e) => {
      const x = state.hits[`profile${k}`]?.xAt(...Charts.position(canvas, e));
      if (x === null || x === undefined) return;
      state.center[k] = x;
      save();
      renderSliderValues();
      scheduleExplore();
    });
    box.append(el("figure", {}, el("figcaption", {}, factorTitle(f)), canvas));
  });
}

function profileSeries(trace) {
  const series = [];
  if (trace.mean) {
    series.push({ label: "Model ± 2 SD", color: 0, points: [], band: trace.x.map((x, n) => [x, trace.mean[n] - 2 * trace.sd[n], trace.mean[n] + 2 * trace.sd[n]]) });
    series.push({ label: "Model prediction", color: 0, points: trace.x.map((x, n) => [x, trace.mean[n]]) });
  }
  if (trace.true) series.push({ label: "True response", color: 1, points: trace.x.map((x, n) => [x, trace.true[n]]) });
  return series;
}

function drawProfiler(hover = {}) {
  if (!state.lab) return;
  const p = state.profile;
  const legend = [];
  if (p?.traces[0]?.mean) legend.push({ label: "Model prediction", kind: "line", color: 0 }, { label: "± 2 SD", kind: "band", color: 0 });
  if (p?.traces[0]?.true) legend.push({ label: "True response", kind: "line", color: 1 });
  Charts.legend($("profile-legend"), legend);
  let domain = null;
  if (p) {
    const values = p.traces.flatMap((t) => [
      ...(t.mean ? t.mean.map((m, n) => m - 2 * t.sd[n]) : []), ...(t.mean ? t.mean.map((m, n) => m + 2 * t.sd[n]) : []),
      ...(t.true ?? []),
    ]);
    domain = Charts.extent(values);
    const pad = 0.05 * (domain[1] - domain[0] || 1);
    domain = [domain[0] - pad, domain[1] + pad];
  }
  state.lab.factors.forEach((f, k) => {
    const canvas = $("profiler").querySelector(`canvas[data-k="${k}"]`);
    if (!p) {
      const { ctx, w, h } = Charts.setup(canvas, 150);
      const th = Charts.theme();
      ctx.fillStyle = th.ink2;
      ctx.font = `12px ${th.font}`;
      ctx.textAlign = "center";
      ctx.fillText(k === 0 ? "Measure 2 runs to fit a model" : "", w / 2, h / 2);
      state.hits[`profile${k}`] = null;
      return;
    }
    state.hits[`profile${k}`] = Charts.lines(canvas, {
      series: profileSeries(p.traces[k]), xDomain: [f.low, f.high], yDomain: domain, height: 150,
      vline: state.center[k], hoverX: hover.k === k ? hover.x : null, xCount: 3, yCount: 3,
      margin: { left: 40, right: 10, bottom: 22 },
    });
  });
}

function profileHover(k, event) {
  const hit = state.hits[`profile${k}`];
  const trace = state.profile?.traces[k];
  if (!hit || !trace) return;
  const x = hit.xAt(...Charts.position(event.target, event));
  if (x === null) return Charts.hideTip();
  const n = trace.x.reduce((best, v, idx) => (Math.abs(v - x) < Math.abs(trace.x[best] - x) ? idx : best), 0);
  const f = state.lab.factors[k];
  let html = `<div class="t-title">${escapeHtml(f.label)} ${fmt(trace.x[n])}${factorUnit(f) ? ` ${factorUnit(f)}` : ""}</div>`;
  if (trace.mean) html += `<div class="t-row"><span class="key key-line" style="--key:${Charts.theme().series[0]}"></span>Predicted ${escapeHtml(withUnit(trace.mean[n]))} ± ${fmt(2 * trace.sd[n])}</div>`;
  if (trace.true) html += `<div class="t-row"><span class="key key-line" style="--key:${Charts.theme().series[1]}"></span>True ${escapeHtml(withUnit(trace.true[n]))}</div>`;
  html += `<div class="t-muted">Click to set ${escapeHtml(f.label)} here.</div>`;
  drawProfiler({ k, x: trace.x[n] });
  Charts.showTip(html, event);
}

function renderModelInfo() {
  const info = state.modelInfo;
  const box = $("model-info");
  if (!info) {
    box.textContent = "";
    return;
  }
  if (info.type === "gp") {
    const ls = Object.entries(info.lengthscales).map(([name, v]) => `${name} ${fmt(v)}`).join(", ");
    box.textContent = `Length scales (0 to 1 is the whole range; short means the response changes quickly along that factor, 10 means it hardly matters): ${ls}. `
      + `Estimated noise SD ${withUnit(info.noise_sd)}.`;
  } else {
    const r2 = info.r2 === null || info.r2 === undefined ? "–" : fmt(info.r2);
    box.textContent = `Full quadratic, ${info.terms} terms from ${info.runs} runs; R² ${r2}.`
      + (info.underdetermined ? " Fewer runs than terms, so the fit is not unique." : "");
  }
}

// --- Convergence and the runs table ---------------------------------------------------------------------

function bestIndexes() {
  const out = [];
  let best = 0;
  state.runs.forEach((r, k) => {
    if (r.y < state.runs[best].y) best = k;
    out.push(best);
  });
  return out;
}

function renderConvergence(hoverX = null) {
  const canvas = $("convergence");
  const runs = state.runs;
  const legend = [{ label: "Each run", kind: "dot", color: "muted" }, { label: "Best measured so far", kind: "line", color: 0 }];
  if (state.instructor) legend.push({ label: "True value at that best run", kind: "line", color: 1 });
  Charts.legend($("convergence-legend"), legend);
  if (!runs.length) {
    const { ctx, w, h } = Charts.setup(canvas, 240);
    const th = Charts.theme();
    ctx.fillStyle = th.ink2;
    ctx.font = `12px ${th.font}`;
    ctx.textAlign = "center";
    ctx.fillText("No runs measured yet", w / 2, h / 2);
    state.hits.convergence = null;
    return;
  }
  const best = bestIndexes();
  const series = [
    { label: "Each run", color: "muted", dotsOnly: true, points: runs.map((r, k) => [k + 1, r.y]) },
    { label: "Best so far", color: 0, step: true, points: runs.map((r, k) => [k + 1, runs[best[k]].y]) },
  ];
  const refs = [];
  if (state.instructor) {
    series.push({ label: "True at best", color: 1, step: true, points: runs.map((r, k) => [k + 1, runs[best[k]].yTrue]) });
    if (state.truth) refs.push({ y: state.truth.optimum.y, label: `Optimum ${withUnit(state.truth.optimum.y)}` });
  }
  const total = Math.max(state.lab.config.budget ?? runs.length, runs.length, 2);
  state.hits.convergence = Charts.lines(canvas, {
    series, refs, xDomain: [1, total], height: 240, xLabel: "Runs measured", yLabel: responseTitle(), hoverX,
  });
}

function convergenceHover(event) {
  const hit = state.hits.convergence;
  if (!hit) return;
  const x = hit.xAt(...Charts.position(event.target, event));
  if (x === null) return Charts.hideTip();
  const k = Math.min(Math.max(Math.round(x), 1), state.runs.length) - 1;
  const best = bestIndexes()[k];
  const th = Charts.theme();
  let html = `<div class="t-title">Run ${k + 1}</div>`
    + `<div class="t-row"><span class="key key-dot" style="--key:${th.muted}"></span>Measured ${escapeHtml(withUnit(state.runs[k].y))}</div>`
    + `<div class="t-row"><span class="key key-line" style="--key:${th.series[0]}"></span>Best so far ${escapeHtml(withUnit(state.runs[best].y))} (run ${best + 1})</div>`;
  if (state.instructor) html += `<div class="t-row"><span class="key key-line" style="--key:${th.series[1]}"></span>True at that run ${escapeHtml(withUnit(state.runs[best].yTrue))}</div>`;
  renderConvergence(k + 1);
  Charts.showTip(html, event);
}

function renderRuns() {
  const table = $("runs-table");
  const factors = state.lab.factors;
  const head = el("tr", {}, el("th", {}, "Run"), el("th", {}, "Step"),
    ...factors.map((f) => el("th", { title: f.label }, factorUnit(f) ? `${shortLabel(f)} (${factorUnit(f)})` : f.label)),
    el("th", {}, state.lab.response.label + (unitOf() ? ` (${unitOf()})` : "")),
    state.instructor ? el("th", {}, "True") : null);
  const bestK = state.runs.length ? bestIndexes()[state.runs.length - 1] : -1;
  const lastStep = state.runs.length ? state.runs[state.runs.length - 1].step : null;
  const rows = state.runs.map((r, k) => el("tr", { class: [k === bestK ? "best" : "", r.step === lastStep ? "latest" : ""].join(" ") },
    el("td", {}, String(k + 1)), el("td", {}, String(r.step)),
    ...r.x.map((v) => el("td", {}, fmt(v))), el("td", {}, fmt(r.y)),
    state.instructor ? el("td", {}, fmt(r.yTrue)) : null)).reverse();
  table.replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows));
  $("export").disabled = !state.runs.length;
  $("clear-runs").disabled = !state.runs.length || state.auto.running;
}

function shortLabel(f) {
  return { "Susceptor temperature": "Temp", "Chamber pressure": "Pressure", "Deposition time": "Time",
    "SiH4 flow": "SiH4", "RF power": "RF", "Electrode spacing": "Gap" }[f.label] ?? f.label;
}

// --- Instructor view ---------------------------------------------------------------------------------

async function loadTruth() {
  try {
    state.truth = await call("truth", { config: state.lab.config });
  } catch (err) {
    setStatus(err.message, true);
  }
  renderTruth();
  renderConvergence();
  drawContours();
}

function renderTruth() {
  const box = $("truth-box");
  if (!state.instructor || !state.truth) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const t = state.truth;
  const pointText = (x) => state.lab.factors.map((f, k) => `${f.label} ${fmt(x[k])}${factorUnit(f) ? ` ${factorUnit(f)}` : ""}`).join(", ");
  const items = [el("dt", {}, "Optimum"), el("dd", {}, `${withUnit(t.optimum.y)} at ${pointText(t.optimum.x)}`)];
  const second = t.local_minima[1];
  if (second) items.push(el("dt", {}, "Second minimum"), el("dd", {}, `${withUnit(second.y)} at ${pointText(second.x)}`));
  if (state.runs.length) {
    const best = state.runs[bestIndexes()[state.runs.length - 1]];
    items.push(el("dt", {}, "Best run"), el("dd", {}, `measured ${withUnit(best.y)}, true ${withUnit(best.yTrue)}: ${withUnit(best.yTrue - t.optimum.y)} above the optimum`));
  }
  if (t.scenario) items.push(el("dt", {}, "Blind scenario"), el("dd", {}, `seed ${t.scenario.seed}: the factors are reordered and some reversed, so the published optimum doesn't apply`));
  box.replaceChildren(el("strong", {}, "Instructor view"), el("dl", {}, ...items),
    el("div", { class: "buttons" }, el("button", { onclick: () => goTo(t.optimum.x) }, "Slice through the optimum"),
      second ? el("button", { onclick: () => goTo(second.x) }, "Slice through the second minimum") : null));
}

function goTo(x) {
  state.center = [...x];
  save();
  renderSliderValues();
  scheduleExplore();
}

// --- Compare strategies -----------------------------------------------------------------------------

async function loadSavedComparison() {
  try {
    const response = await fetch("data/benchmark.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.compare.saved = await response.json();
  } catch {
    state.compare.saved = null;
  }
  renderCompare();
}

function currentComparison() {
  return state.compare.live ?? state.compare.saved;
}

function renderCompare(hoverX = null) {
  const data = currentComparison();
  const canvas = $("compare-chart");
  if ($("tab-compare").hidden) return;
  if (!data) {
    $("compare-caption").textContent = "No saved comparison is available. Run your own below.";
    Charts.setup(canvas, 360);
    $("compare-table").replaceChildren();
    return;
  }
  const which = state.compare.which;
  const summary = data.summary[which];
  const labels = data.labels.filter((l) => summary[l]);
  const colorOf = (label, k) => STRATEGY_COLORS[label] ?? k;
  const series = labels.map((label, k) => {
    const s = summary[label];
    const xs = s.median.map((_, n) => n + 1);
    return { label, color: colorOf(label, k), points: xs.map((x, n) => [x, s.median[n]]), band: xs.map((x, n) => [x, s.q25[n], s.q75[n]]) };
  });
  const unit = data.oracle?.units === "unit" ? "" : "%";
  const refs = data.optimum !== null && data.optimum !== undefined ? [{ y: data.optimum, label: `Optimum ${fmt(data.optimum)}${unit ? ` ${unit}` : ""}` }] : [];
  state.hits.compare = Charts.lines(canvas, {
    series, refs, xDomain: [1, data.budget], height: 360, endLabels: true, hoverX,
    xLabel: "Runs measured", yLabel: `${which === "true" ? "True value at the best run" : "Best measured"}${unit ? ` (${unit})` : ""}`,
  });
  Charts.legend($("compare-legend"), labels.flatMap((label, k) => [{ label, kind: "line", color: colorOf(label, k) }])
    .concat([{ label: "Middle half of the replicates", kind: "band", color: "muted" }]));
  const noise = data.oracle?.noise?.sd;
  $("compare-caption").textContent = `${state.compare.live ? "Your run" : "Saved comparison"}: median (line) and middle 50% (band) over `
    + `${data.replicates} replicates of ${data.budget} runs${noise !== undefined ? `, noise SD ${fmt(noise)}${unit ? ` ${unit}` : ""}` : ""}. `
    + (which === "true" ? "The true value at the best measured run is what you'd get running that setting again." : "The best measurement flatters every strategy when there is noise.");
  const checkpoints = [...new Set([0.25, 0.5, 0.75, 1].map((f) => Math.max(1, Math.round(f * data.budget))))];
  $("compare-table").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", {}, "Strategy"), ...checkpoints.map((n) => el("th", {}, `After ${n} runs`)))),
    el("tbody", {}, ...labels.map((label) => el("tr", {}, el("td", {}, label),
      ...checkpoints.map((n) => el("td", {}, `${fmt(summary[label].median[n - 1])} [${fmt(summary[label].q25[n - 1])}–${fmt(summary[label].q75[n - 1])}]`))))));
  $("live-reset").hidden = !state.compare.live || state.compare.running;
}

function compareHover(event) {
  const hit = state.hits.compare;
  const data = currentComparison();
  if (!hit || !data) return;
  const x = hit.xAt(...Charts.position(event.target, event));
  if (x === null) return Charts.hideTip();
  const n = Math.min(Math.max(Math.round(x), 1), data.budget);
  const summary = data.summary[state.compare.which];
  const th = Charts.theme();
  let html = `<div class="t-title">After ${n} runs</div>`;
  data.labels.filter((l) => summary[l]).forEach((label, k) => {
    const s = summary[label];
    html += `<div class="t-row"><span class="key key-line" style="--key:${th.series[STRATEGY_COLORS[label] ?? k]}"></span>`
      + `${escapeHtml(label)}: ${fmt(s.median[n - 1])} [${fmt(s.q25[n - 1])}–${fmt(s.q75[n - 1])}]</div>`;
  });
  renderCompare(n);
  Charts.showTip(html, event);
}

async function runLiveComparison() {
  const keys = [...$("live-strategies").querySelectorAll("input:checked")].map((i) => i.value);
  if (!keys.length) return setStatus("Choose at least one strategy", true);
  const budget = Number($("live-budget").value) || 40;
  const reps = Number($("live-reps").value) || 5;
  const strategies = state.catalog.benchmark_strategies.filter((s) => keys.includes(s.key));
  const optimum = (await call("truth", { config: state.lab.config })).optimum.y;
  const curves = {};
  const c = state.compare;
  c.running = true;
  c.stop = false;
  $("live-stop").hidden = false;
  $("live-progress").hidden = false;
  updateControls();
  const total = reps * strategies.length;
  let done = 0;
  try {
    for (let r = 0; r < reps && !c.stop; r++) {
      for (const s of strategies) {
        if (c.stop) break;
        setStatus(`Comparison: run ${done + 1} of ${total} (${s.key}, replicate ${r + 1})…`);
        const result = await call("benchmark_run", { config: state.lab.config, strategy: s.strategy, replicate: r, budget, seed: 0 });
        curves[result.label] ??= { observed: [], true: [] };
        curves[result.label].observed.push(result.observed);
        curves[result.label].true.push(result.true);
        done += 1;
        $("live-progress").value = done / total;
      }
      const summary = await call("benchmark_summary", { curves, optimum });
      c.live = { ...summary, oracle: state.lab.config };
      renderCompare();
    }
    setStatus(c.stop ? "Comparison stopped" : "Comparison finished");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    c.running = false;
    $("live-stop").hidden = true;
    $("live-progress").hidden = true;
    updateControls();
    renderCompare();
  }
}

// --- Tabs and events --------------------------------------------------------------------------------

function showTab(name) {
  for (const button of document.querySelectorAll(".tabs button")) {
    const on = button.dataset.tab === name;
    button.setAttribute("aria-selected", on);
    $(`tab-${button.dataset.tab}`).hidden = !on;
  }
  if (name === "compare") renderCompare();
  if (name === "lab" && state.lab) redraw();
}

function redraw() {
  if (!state.lab) return;
  drawContours();
  drawProfiler();
  renderConvergence();
  renderCompare();
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
$("new-lab").addEventListener("click", newLab);
$("generate").addEventListener("click", planDesign);
$("suggest").addEventListener("click", suggest);
$("auto").addEventListener("click", autoRun);
$("stop-auto").addEventListener("click", () => { state.auto.stop = true; });
$("measure").addEventListener("click", measureClicked);
$("discard").addEventListener("click", () => setPlan(null));
$("import-run").addEventListener("click", importTable);
$("import-file-btn").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) $("import-text").value = await file.text();
  e.target.value = "";
});
$("export").addEventListener("click", () => exportRuns().catch((err) => setStatus(err.message, true)));
$("clear-runs").addEventListener("click", () => {
  if (!confirm("Clear all measured runs? The lab settings stay.")) return;
  state.runs = [];
  state.plan = null;
  state.step = 0;
  save();
  renderAll();
});
$("instructor").addEventListener("change", (e) => {
  state.instructor = e.target.checked;
  if (state.instructor && !state.truth) loadTruth();
  renderAll();
});
$("axis-x").addEventListener("change", (e) => setAxis(0, Number(e.target.value)));
$("axis-y").addEventListener("change", (e) => setAxis(1, Number(e.target.value)));
$("model").addEventListener("change", (e) => {
  const [model, kernel] = e.target.value.split(":");
  state.model = model;
  if (kernel) state.kernel = kernel;
  save();
  scheduleExplore();
});
$("layer").addEventListener("change", (e) => {
  state.layer = e.target.value;
  save();
  scheduleExplore();
});
$("compare-which").addEventListener("change", (e) => {
  state.compare.which = e.target.value;
  renderCompare();
});
$("live-run").addEventListener("click", runLiveComparison);
$("live-stop").addEventListener("click", () => { state.compare.stop = true; });
$("live-reset").addEventListener("click", () => {
  state.compare.live = null;
  renderCompare();
});

function setAxis(which, k) {
  const other = state.axes[1 - which];
  if (k === other) state.axes[1 - which] = state.axes[which];
  state.axes[which] = k;
  $("axis-x").value = state.axes[0];
  $("axis-y").value = state.axes[1];
  save();
  renderSliderValues();
  scheduleExplore();
}

for (const which of ["true", "model"]) {
  const canvas = $(`contour-${which}`);
  canvas.addEventListener("mousemove", (e) => contourHover(which, e));
  canvas.addEventListener("mouseleave", Charts.hideTip);
  canvas.addEventListener("click", (e) => contourClick(which, e));
}
$("convergence").addEventListener("mousemove", convergenceHover);
$("convergence").addEventListener("mouseleave", () => {
  Charts.hideTip();
  renderConvergence();
});
$("compare-chart").addEventListener("mousemove", compareHover);
$("compare-chart").addEventListener("mouseleave", () => {
  Charts.hideTip();
  renderCompare();
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(redraw, 120);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
