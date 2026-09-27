"use strict";

// Canvas charts for the simulator: heatmaps with contour lines, line charts
// with bands, a color bar, and a shared tooltip. No dependencies.
//
// Colors come from CSS custom properties (see style.css), so light and dark
// mode each use their own validated steps. Marks follow one spec: 2px lines,
// 8px markers with a 2px surface ring, 10% washes for bands, hairline grid.

const Charts = (() => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function theme() {
    const ramp = [];
    for (let i = 1; i <= 7; i++) ramp.push(hexToRgb(css(`--ramp-${i}`)));
    return {
      surface: css("--surface"),
      ink: css("--ink"),
      ink2: css("--ink-2"),
      muted: css("--muted"),
      grid: css("--grid"),
      axis: css("--axis"),
      series: [1, 2, 3, 4, 5, 6].map((i) => css(`--s${i}`)),
      ramp, // low value -> high value, already flipped for dark mode in CSS
      font: getComputedStyle(document.body).fontFamily,
    };
  }

  function hexToRgb(hex) {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }

  function rgba(color, alpha) {
    const [r, g, b] = hexToRgb(color);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // Size a canvas to its CSS box at the device pixel ratio.
  function setup(canvas, height) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(canvas.clientWidth, 50);
    const h = height ?? Math.max(canvas.clientHeight, 50);
    canvas.style.height = `${h}px`;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  // Round tick values covering [min, max].
  function ticks(min, max, count = 5) {
    if (!(max > min)) return [min];
    const raw = (max - min) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
    const out = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }

  function fmt(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return v.toPrecision(2);
  }

  // Decimals needed to show every tick exactly: 5 not 5.00, 2.5 not 3.
  function tickDecimals(values) {
    const step = values.length > 1 ? Math.abs(values[1] - values[0]) : Math.abs(values[0]) || 1;
    for (let d = 0; d <= 6; d++) {
      const scaled = step * 10 ** d;
      if (Math.abs(scaled - Math.round(scaled)) < 1e-6 * Math.max(1, scaled)) return d;
    }
    return 6;
  }

  function tickLabel(v, decimals) {
    return v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  }

  const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

  function extent(values) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (Number.isFinite(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    return lo <= hi ? [lo, hi] : [0, 1];
  }

  function rampColor(t, ramp) {
    const x = Math.min(Math.max(t, 0), 1) * (ramp.length - 1);
    const i = Math.min(Math.floor(x), ramp.length - 2);
    const f = x - i;
    return ramp[i].map((c, k) => Math.round(c + f * (ramp[i + 1][k] - c)));
  }

  // --- Axes ------------------------------------------------------------------------------

  function axes(ctx, th, box, xs, ys, xDomain, yDomain, xLabel, yLabel, opts = {}) {
    ctx.font = `11px ${th.font}`;
    ctx.fillStyle = th.muted;
    ctx.strokeStyle = th.grid;
    ctx.lineWidth = 1;
    const xt = opts.xTicks ?? ticks(xDomain[0], xDomain[1], opts.xCount ?? 5);
    const yt = opts.yTicks ?? ticks(yDomain[0], yDomain[1], opts.yCount ?? 5);
    const xd = tickDecimals(xt);
    const yd = tickDecimals(yt);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of xt) {
      const x = Math.round(xs(v)) + 0.5;
      if (opts.grid) {
        ctx.beginPath();
        ctx.moveTo(x, box.top);
        ctx.lineTo(x, box.bottom);
        ctx.stroke();
      }
      ctx.fillText(tickLabel(v, xd), x, box.bottom + 4);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of yt) {
      const y = Math.round(ys(v)) + 0.5;
      if (opts.grid) {
        ctx.beginPath();
        ctx.moveTo(box.left, y);
        ctx.lineTo(box.right, y);
        ctx.stroke();
      }
      ctx.fillText(tickLabel(v, yd), box.left - 5, y);
    }
    ctx.strokeStyle = th.axis;
    ctx.beginPath();
    ctx.moveTo(box.left, box.bottom + 0.5);
    ctx.lineTo(box.right, box.bottom + 0.5);
    ctx.stroke();
    ctx.fillStyle = th.ink2;
    if (xLabel) {
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(xLabel, (box.left + box.right) / 2, box.bottom + 34);
    }
    if (yLabel) {
      ctx.save();
      ctx.translate(12, (box.top + box.bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
  }

  function dot(ctx, th, x, y, fill, r = 4) {
    ctx.beginPath();
    ctx.arc(x, y, r + 2, 0, 2 * Math.PI);
    ctx.fillStyle = th.surface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = fill;
    ctx.fill();
  }

  function ring(ctx, th, x, y, stroke, r = 5) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.lineWidth = 4;
    ctx.strokeStyle = th.surface;
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }

  function cross(ctx, th, x, y, color, r = 6) {
    for (const [width, stroke] of [[5, th.surface], [2, color]]) {
      ctx.lineWidth = width;
      ctx.strokeStyle = stroke;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x - r, y - r);
      ctx.lineTo(x + r, y + r);
      ctx.moveTo(x + r, y - r);
      ctx.lineTo(x - r, y + r);
      ctx.stroke();
    }
  }

  // --- Heatmap with contour lines ------------------------------------------------------------

  // Marching squares: line segments where the grid crosses `level`.
  function isoSegments(grid, level) {
    const segs = [];
    const rows = grid.length;
    const cols = grid[0].length;
    const lerp = (a, b) => (level - a) / (b - a);
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const v = [grid[r][c], grid[r][c + 1], grid[r + 1][c + 1], grid[r + 1][c]];
        const pts = [];
        const edges = [[0, 1, [c, r], [c + 1, r]], [1, 2, [c + 1, r], [c + 1, r + 1]],
          [2, 3, [c + 1, r + 1], [c, r + 1]], [3, 0, [c, r + 1], [c, r]]];
        for (const [a, b, pa, pb] of edges) {
          if ((v[a] < level) !== (v[b] < level)) {
            const t = lerp(v[a], v[b]);
            pts.push([pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])]);
          }
        }
        if (pts.length === 2) segs.push(pts);
        else if (pts.length === 4) segs.push([pts[0], pts[1]], [pts[2], pts[3]]);
      }
    }
    return segs;
  }

  // opts: {grid (rows = y), x, y (axis values), domain, xLabel, yLabel, points: [{x, y, kind}],
  //        center: [cx, cy], markers: [{x, y, label}], hover: {px, py}}
  // Returns a map from canvas pixels to data, for hover and clicks.
  function heatmap(canvas, opts) {
    const th = theme();
    const size = Math.min(canvas.clientWidth, 420);
    const { ctx, w, h } = setup(canvas, size);
    const box = { left: 48, right: w - 10, top: 8, bottom: h - 42 };
    const xs = scale(opts.x[0], opts.x[opts.x.length - 1], box.left, box.right);
    const ys = scale(opts.y[0], opts.y[opts.y.length - 1], box.bottom, box.top);
    const grid = opts.grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const [lo, hi] = opts.domain ?? extent(grid.flat());

    // Cells, drawn small and scaled up with smoothing.
    const off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    const octx = off.getContext("2d");
    const img = octx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [R, G, B] = rampColor((grid[r][c] - lo) / (hi - lo || 1), th.ramp);
        const k = ((rows - 1 - r) * cols + c) * 4;
        img.data[k] = R;
        img.data[k + 1] = G;
        img.data[k + 2] = B;
        img.data[k + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, box.left, box.top, box.right - box.left, box.bottom - box.top);

    // Contour lines at round levels, in a translucent ink.
    const gx = (c) => box.left + (c / (cols - 1)) * (box.right - box.left);
    const gy = (r) => box.bottom - (r / (rows - 1)) * (box.bottom - box.top);
    ctx.strokeStyle = rgba(th.ink.startsWith("#") ? th.ink : "#000000", 0.28);
    ctx.lineWidth = 1;
    for (const level of ticks(lo, hi, 8)) {
      if (level <= lo || level >= hi) continue;
      ctx.beginPath();
      for (const [[c0, r0], [c1, r1]] of isoSegments(grid, level)) {
        ctx.moveTo(gx(c0), gy(r0));
        ctx.lineTo(gx(c1), gy(r1));
      }
      ctx.stroke();
    }

    // Slice crosshair (where the other sliders sit).
    if (opts.center) {
      ctx.strokeStyle = rgba(th.ink.startsWith("#") ? th.ink : "#000000", 0.35);
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      const cx = Math.round(xs(opts.center[0])) + 0.5;
      const cy = Math.round(ys(opts.center[1])) + 0.5;
      ctx.beginPath();
      ctx.moveTo(cx, box.top);
      ctx.lineTo(cx, box.bottom);
      ctx.moveTo(box.left, cy);
      ctx.lineTo(box.right, cy);
      ctx.stroke();
    }

    for (const p of opts.points ?? []) {
      const px = xs(p.x);
      const py = ys(p.y);
      if (p.kind === "planned") ring(ctx, th, px, py, th.series[1]);
      else dot(ctx, th, px, py, p.kind === "latest" ? th.series[1] : th.ink);
    }
    for (const m of opts.markers ?? []) cross(ctx, th, xs(m.x), ys(m.y), th.ink, m.major ? 7 : 5);

    axes(ctx, th, box, xs, ys, [opts.x[0], opts.x[opts.x.length - 1]], [opts.y[0], opts.y[opts.y.length - 1]],
      opts.xLabel, opts.yLabel);

    const inv = (v, a, b, r0, r1) => a + ((v - r0) / (r1 - r0)) * (b - a);
    return {
      box,
      toData(px, py) {
        if (px < box.left || px > box.right || py < box.top || py > box.bottom) return null;
        const x = inv(px, opts.x[0], opts.x[opts.x.length - 1], box.left, box.right);
        const y = inv(py, opts.y[0], opts.y[opts.y.length - 1], box.bottom, box.top);
        const c = Math.round(((px - box.left) / (box.right - box.left)) * (cols - 1));
        const r = Math.round(((box.bottom - py) / (box.bottom - box.top)) * (rows - 1));
        return { x, y, value: grid[r][c] };
      },
      nearest(px, py, radius = 12) {
        let best = null;
        let bestD = radius;
        (opts.points ?? []).forEach((p, index) => {
          const d = Math.hypot(xs(p.x) - px, ys(p.y) - py);
          if (d <= bestD) {
            bestD = d;
            best = { ...p, index };
          }
        });
        return best;
      },
    };
  }

  // The scale's name on its own line, then the bar, then its low and high values.
  function colorbar(canvas, domain, label) {
    const th = theme();
    const { ctx, w } = setup(canvas, 46);
    const left = 48;
    const right = w - 10;
    ctx.font = `11px ${th.font}`;
    ctx.fillStyle = th.ink2;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(label, left, 2);
    for (let x = left; x < right; x++) {
      const [r, g, b] = rampColor((x - left) / (right - left), th.ramp);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, 17, 1, 10);
    }
    ctx.fillStyle = th.ink2;
    ctx.fillText(fmt(domain[0]), left, 31);
    ctx.textAlign = "right";
    ctx.fillText(fmt(domain[1]), right, 31);
  }

  // --- Line charts --------------------------------------------------------------------------

  // opts: {series: [{label, color (index into series palette or "ink"/"muted"), points: [[x, y]],
  //                   band: [[x, lo, hi]], step, dots, dotsOnly}],
  //        xDomain, yDomain, xLabel, yLabel, refs: [{y, label}], vline, height, hoverX, endLabels,
  //        margin}
  function lines(canvas, opts) {
    const th = theme();
    const { ctx, w, h } = setup(canvas, opts.height ?? 240);
    const m = { left: 48, right: 12, top: 10, bottom: opts.xLabel ? 40 : 24, ...opts.margin };
    const box = { left: m.left, right: w - m.right, top: m.top, bottom: h - m.bottom };
    const all = opts.series.flatMap((s) => [
      ...s.points.map((p) => p[1]),
      ...(s.band ?? []).flatMap((b) => [b[1], b[2]]),
    ]).concat((opts.refs ?? []).map((r) => r.y));
    let yDomain = opts.yDomain ?? extent(all);
    if (yDomain[0] === yDomain[1]) yDomain = [yDomain[0] - 1, yDomain[1] + 1];
    const pad = 0.05 * (yDomain[1] - yDomain[0]);
    yDomain = opts.yDomain ?? [yDomain[0] - pad, yDomain[1] + pad];
    const xDomain = opts.xDomain ?? extent(opts.series.flatMap((s) => s.points.map((p) => p[0])));
    const ys = scale(yDomain[0], yDomain[1], box.bottom, box.top);
    const color = (c) => (c === "ink" ? th.ink : c === "muted" ? th.muted : th.series[c]);

    // Direct labels at line ends, only when none collide; otherwise the legend and table carry
    // identity. Their room on the right is kept only when they are drawn.
    const ends = opts.endLabels ? opts.series.filter((s) => !s.dotsOnly && s.points.length)
      .map((s) => ({ s, y: ys(s.points[s.points.length - 1][1]) })).sort((a, b) => a.y - b.y) : [];
    const labelled = ends.length > 0 && ends.every((e, k) => k === 0 || e.y - ends[k - 1].y >= 14);
    if (labelled) box.right = w - 96;
    const xs = scale(xDomain[0], xDomain[1], box.left, box.right);

    axes(ctx, th, box, xs, ys, xDomain, yDomain, opts.xLabel, opts.yLabel,
      { grid: true, xCount: opts.xCount ?? 5, yCount: opts.yCount ?? 4 });

    for (const r of opts.refs ?? []) {
      const y = Math.round(ys(r.y)) + 0.5;
      ctx.strokeStyle = th.ink2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(box.left, y);
      ctx.lineTo(box.right, y);
      ctx.stroke();
      ctx.fillStyle = th.ink2;
      ctx.font = `11px ${th.font}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(r.label, box.left + 4, y - 2);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(box.left, box.top - 6, box.right - box.left, box.bottom - box.top + 12);
    ctx.clip();
    for (const s of opts.series) {
      if (!s.band) continue;
      ctx.fillStyle = rgba(color(s.color), 0.12);
      ctx.beginPath();
      s.band.forEach(([x, , hi], k) => (k ? ctx.lineTo(xs(x), ys(hi)) : ctx.moveTo(xs(x), ys(hi))));
      for (let k = s.band.length - 1; k >= 0; k--) ctx.lineTo(xs(s.band[k][0]), ys(s.band[k][1]));
      ctx.closePath();
      ctx.fill();
    }
    for (const s of opts.series) {
      const c = color(s.color);
      if (!s.dotsOnly && s.points.length) {
        ctx.strokeStyle = c;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        s.points.forEach(([x, y], k) => {
          if (!k) ctx.moveTo(xs(x), ys(y));
          else if (s.step) {
            ctx.lineTo(xs(x), ys(s.points[k - 1][1]));
            ctx.lineTo(xs(x), ys(y));
          } else ctx.lineTo(xs(x), ys(y));
        });
        ctx.stroke();
      }
      if (s.dots || s.dotsOnly) for (const [x, y] of s.points) dot(ctx, th, xs(x), ys(y), c, s.dotRadius ?? 3);
    }
    ctx.restore();

    if (opts.vline !== undefined && opts.vline !== null) {
      const x = Math.round(xs(opts.vline)) + 0.5;
      ctx.strokeStyle = th.ink2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, box.top);
      ctx.lineTo(x, box.bottom);
      ctx.stroke();
    }

    if (labelled) {
      ctx.font = `12px ${th.font}`;
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      for (const { s, y } of ends) {
        dot(ctx, th, box.right + 8, y, color(s.color), 3);
        ctx.fillStyle = th.ink2;
        ctx.fillText(s.label, box.right + 16, y);
      }
    }

    if (opts.hoverX !== undefined && opts.hoverX !== null) {
      const x = Math.round(xs(opts.hoverX)) + 0.5;
      ctx.strokeStyle = th.ink2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, box.top);
      ctx.lineTo(x, box.bottom);
      ctx.stroke();
    }

    const inv = (px) => xDomain[0] + ((px - box.left) / (box.right - box.left)) * (xDomain[1] - xDomain[0]);
    return {
      box,
      xAt(px, py) {
        if (px < box.left - 4 || px > box.right + 4 || py < box.top - 4 || py > box.bottom + 4) return null;
        return Math.min(Math.max(inv(px), xDomain[0]), xDomain[1]);
      },
    };
  }

  // --- Legend and tooltip -------------------------------------------------------------------

  // items: [{label, kind: "line" | "dot" | "ring" | "band" | "cross", color}]
  function legend(container, items) {
    container.replaceChildren();
    const th = theme();
    for (const item of items) {
      const key = document.createElement("span");
      key.className = `key key-${item.kind}`;
      const c = item.color === "ink" ? th.ink : item.color === "muted" ? th.muted : th.series[item.color];
      key.style.setProperty("--key", c);
      const entry = document.createElement("span");
      entry.className = "legend-item";
      entry.append(key, document.createTextNode(item.label));
      container.append(entry);
    }
  }

  const tip = () => document.getElementById("tooltip");

  function showTip(html, event) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const pad = 14;
    const { innerWidth: W, innerHeight: H } = window;
    const r = t.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + r.width > W - 8) x = event.clientX - r.width - pad;
    if (y + r.height > H - 8) y = event.clientY - r.height - pad;
    t.style.left = `${x}px`;
    t.style.top = `${y}px`;
  }

  function hideTip() {
    tip().hidden = true;
  }

  function position(canvas, event) {
    const r = canvas.getBoundingClientRect();
    return [event.clientX - r.left, event.clientY - r.top];
  }

  return { theme, setup, ticks, fmt, extent, heatmap, colorbar, lines, legend, showTip, hideTip, position };
})();
