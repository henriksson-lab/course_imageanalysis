// k-means (Lloyd's algorithm) step by step on 2D points, or on points on a line (data-mode="1d").
// Click to add points, drag them, or use "remove points". Centres are the large crossed circles:
// drag them to place them by hand. "assign" colours every point by its nearest centre (the shaded
// Voronoi cells), "update" moves every centre to the mean of its points (with a trail), "step"
// does both, "run" repeats until nothing changes. Right: J (the within-cluster sum of squares)
// per step, and "100 restarts": a histogram of the final J of 100 runs from different starts.
// The y-scale slider stretches the y axis before distances are measured (units matter).
//
// <div data-demo="kmeans2d" data-preset="blobs3|hand1d|stuck|plusplus|elongated|unequal|rescale"
//      data-mode="2d|1d" data-k="3" data-steps="0" data-restarts=""></div>
import { autoMount, button, htmlEl, presets, slider, svgEl, draggable, type Slider } from "../lib/ui";
import { pointPlayground, type Playground } from "../lib/playground";
import { DATASETS, anisotropic, type LPoint } from "../lib/datasets2d";
import { Plot, seriesColor } from "../lib/plot";
import { QUALITATIVE, hexToRgb } from "../lib/colour";
import { rng } from "../lib/random";
import { assignStep, updateStep, initCentres, kmeans, type KMeansInit } from "../lib/kmeans";

const NAME = "kmeans2d";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; }
.qb-demo[data-demo="${NAME}"] .km-left { flex: 0 1 470px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .km-right { flex: 1 1 420px; min-width: 260px; max-width: 560px; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .km-plots { display: flex; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .km-plots > div { flex: 1 1 50%; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .km-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] select { font: inherit; font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .km-left { flex: 0 0 500px; }
.reveal .qb-demo[data-demo="${NAME}"] .km-right { flex: 1 1 600px; max-width: 640px; gap: .25em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .5em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const GREY = "#9a9a9a";
const COLS = QUALITATIVE.slice(0, 8);
/** 1D mode: data values 0..12 on the horizontal line y = 0.5. */
const R1 = 12;

interface Preset {
  id: string; label: string; mode: "2d" | "1d"; k: number; init: KMeansInit | "hand";
  points: () => LPoint[]; centres?: number[]; yscale?: number; seed?: number;
}

const line = (vals: number[]): LPoint[] => vals.map(v => ({ x: v / R1, y: 0.5, c: 0 }));
const ds = (id: string, seed = 1) => () => DATASETS.find(d => d.id === id)!.make({ seed }).points.map(p => ({ ...p, c: 0 }));
const bands = () => anisotropic([{ centre: [0.5, 0.36], sd: [0.2, 0.035], angle: 0 }, { centre: [0.5, 0.64], sd: [0.2, 0.035], angle: 0 }], { seed: 4, n: 140 }).points.map(p => ({ ...p, c: 0 }));

const PRESETS: Preset[] = [
  { id: "blobs3", label: "three blobs", mode: "2d", k: 3, init: "kmeans++", points: ds("blobs3"), seed: 1 },
  { id: "hand1d", label: "by hand (1D)", mode: "1d", k: 2, init: "hand", points: () => line([1, 2, 4, 7, 9, 10]), centres: [1, 2] },
  { id: "stuck", label: "stuck: two centres in one group", mode: "1d", k: 3, init: "hand", points: () => line([0, 1, 5, 6, 10, 11]), centres: [0, 1, 8] },
  { id: "plusplus", label: "k-means++ avoids it", mode: "1d", k: 3, init: "kmeans++", points: () => line([0, 1, 5, 6, 10, 11]), seed: 3 },
  { id: "elongated", label: "elongated: cut across", mode: "2d", k: 2, init: "kmeans++", points: ds("aniso", 2), seed: 1 },
  { id: "unequal", label: "big and small group", mode: "2d", k: 2, init: "kmeans++", points: ds("unequal", 1), seed: 1 },
  { id: "rescale", label: "rescale the y axis", mode: "2d", k: 2, init: "kmeans++", points: bands, seed: 2, yscale: 1 },
];

const f2 = (v: number) => v.toFixed(2), f3 = (v: number) => v.toFixed(3);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? (host.dataset.mode === "1d" ? PRESETS[1] : PRESETS[0]);
  const st = {
    mode: p0.mode, k: Number(host.dataset.k) || p0.k, init: p0.init as KMeansInit | "hand", seed: p0.seed ?? 1, yscale: p0.yscale ?? 1,
    C: [] as number[],            // centres in data units (k·2; 1D: k·1 values)
    C0: [] as number[],           // start centres (for reset)
    assign: null as Int32Array | null,
    next: "assign" as "assign" | "update",
    Js: [] as number[], iters: [] as number[], trails: [] as number[][][],
    iter: 0, done: false, restarts: null as number[] | null,
  };
  let timer = 0;

  // ---- layout ----
  const left = htmlEl("div", "km-left", "", host);
  const right = htmlEl("div", "km-right", "", host);
  const pg: Playground = pointPlayground(left, {
    data: { points: p0.points(), k: 1 }, k: 1, colors: [GREY, ...COLS], controls: false, size: 480, r: 8,
    ariaLabel: "Points and k-means centres", onChange: (_pts, final) => { if (st.mode === "1d") flatten(); pointsChanged(final); },
  });
  const plots = htmlEl("div", "km-plots", "", right);
  const jDiv = htmlEl("div", "", "", plots), hDiv = htmlEl("div", "", "", plots);
  const jPlot = new Plot(jDiv, { width: 300, height: 190, x: [0, 6], y: [0, 1], xlabel: "step", ylabel: "J", font: 17 });
  const hPlot = new Plot(hDiv, { width: 300, height: 190, x: [0, 1], y: [0, 10], xlabel: "final J (100 runs)", ylabel: "runs", font: 17 });
  const jLine = jPlot.line([], [], { color: seriesColor(0), width: 2.5 });
  const jDots = jPlot.scatter([], [], { color: seriesColor(0), r: 5 });
  const hBars = hPlot.histogram([], [], { color: seriesColor(3) });
  const hCur = hPlot.vline(NaN, { color: seriesColor(1), text: "this run" });
  const ro = htmlEl("div", "qb-readout", "", right);

  const row1 = htmlEl("div", "qb-presets", "", right);
  button(row1, "assign", () => { stop(); doAssign(); draw(); });
  button(row1, "update", () => { stop(); doUpdate(); draw(); });
  button(row1, "step", () => { stop(); step(); draw(); });
  const runBtn = button(row1, "run", () => { if (timer) stop(); else run(); });
  button(row1, "reset", () => { stop(); restart(false); });
  button(row1, "new start", () => { stop(); st.seed++; restart(true); });
  button(row1, "100 restarts", () => { stop(); restarts(); });
  const row2 = htmlEl("div", "qb-presets", "", right);
  htmlEl("span", "km-lab", "start:", row2);
  const initBtns = (["random", "kmeans++", "hand"] as const).map(m => button(row2, m === "hand" ? "by hand" : m === "kmeans++" ? "k-means++" : "random points", () => { st.init = m; restart(true); }));
  const remBtn = button(row2, "remove points", () => { pg.tool = pg.tool === "remove" ? "add" : "remove"; sync(); });
  button(row2, "clear", () => { stop(); pg.setPoints([]); });
  const kS: Slider = slider(right, "K (centres)", 1, 8, 1, st.k, v => { st.k = v; restart(true); });
  const yS: Slider = slider(right, "y axis × (2D)", 0.25, 4, 0.05, st.yscale, v => { st.yscale = v; restart(true); });
  htmlEl("div", "qb-note", "Click to add a point, drag to move it. Drag a centre to place it by hand. Shading: the Voronoi cell of each centre (the points it would get).", right);
  const presetBtns = presets(right, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const resetAll = button(right, "Reset demo", () => load(p0));
  resetAll.style.alignSelf = "flex-start";

  // ---- centre handles, trails, 1D axis ----
  const gTrail = svgEl("g", {}, pg.overlay);
  const gAxis = svgEl("g", {}, pg.overlay);
  const gHandles = svgEl("g", {}, pg.top);

  // ---- data helpers ----
  const D = () => (st.mode === "1d" ? 1 : 2);
  function dataX(): { X: Float64Array; n: number } {
    const pts = pg.points, d = D(), X = new Float64Array(pts.length * d);
    pts.forEach((p, i) => { if (d === 1) X[i] = p.x * R1; else { X[2 * i] = p.x; X[2 * i + 1] = p.y * st.yscale; } });
    return { X, n: pts.length };
  }
  const toScreen = (j: number): [number, number] => (st.mode === "1d" ? [st.C[j] / R1, 0.5] : [st.C[2 * j], st.C[2 * j + 1] / st.yscale]);
  function flatten(): void { for (const p of pg.points) p.y = 0.5; pg.redraw(); }

  function restart(newCentres: boolean): void {
    stop();
    const { X, n } = dataX(), d = D();
    if (newCentres || st.C0.length !== st.k * d) {
      if (st.init === "hand" || n === 0) {
        // spread the centres along a diagonal (or keep the preset's)
        st.C0 = [];
        for (let j = 0; j < st.k; j++) {
          const t = (j + 0.5) / st.k;
          if (d === 1) st.C0.push(+(t * R1).toFixed(2)); else st.C0.push(0.15 + 0.7 * t, (0.2 + 0.6 * t) * st.yscale);
        }
      } else {
        st.C0 = Array.from(initCentres(X, n, d, Math.min(st.k, n), st.init, rng(st.seed)));
        while (st.C0.length < st.k * d) st.C0.push(...st.C0.slice(0, d));
      }
    }
    st.C = st.C0.slice();
    st.assign = null; st.next = "assign"; st.Js = []; st.iters = []; st.iter = 0; st.done = false;
    st.trails = Array.from({ length: st.k }, (_, j) => [toScreen(j)]);
    buildHandles();
    draw();
  }

  function load(p: Preset): void {
    stop();
    st.mode = p.mode; st.k = p.k; st.init = p.init; st.seed = p.seed ?? 1; st.yscale = p.yscale ?? 1; st.restarts = null;
    kS.set(st.k); yS.set(st.yscale);
    st.assign = null; st.C = [];
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    pg.setPoints({ points: p.points(), k: 1 });      // triggers pointsChanged
    st.C0 = p.centres ? p.centres.slice() : [];
    restart(!p.centres);
  }

  function pointsChanged(final: boolean): void {
    st.restarts = null;
    if (!st.C.length) return;
    if (st.assign) { doAssignOnly(); }
    draw();
    void final;
  }

  // ---- algorithm ----
  function doAssignOnly(): void {
    const { X, n } = dataX();
    const a = assignStep(X, n, D(), st.C, st.k);
    st.assign = a.assign;
  }
  function doAssign(): void {
    const { X, n } = dataX();
    if (!n || (st.next === "update" && st.assign)) return;
    const prev = st.assign;
    const a = assignStep(X, n, D(), st.C, st.k);
    st.done = !!prev && prev.length === n && a.assign.every((v, i) => v === prev[i]) && st.next === "assign";
    st.assign = a.assign;
    st.Js.push(a.J); st.iters.push(st.iter);
    st.next = "update";
  }
  function doUpdate(): void {
    if (!st.assign) { doAssign(); return; }
    const { X, n } = dataX();
    if (!n) return;
    const u = updateStep(X, n, D(), st.assign, st.k, undefined, st.C);
    st.C = Array.from(u.centres);
    st.iter++;
    st.trails.forEach((t, j) => t.push(toScreen(j)));
    st.next = "assign";
  }
  function step(): void { if (st.next === "assign") doAssign(); if (!st.done) { doUpdate(); doAssign(); } }
  function run(): void {
    runBtn.textContent = "pause";
    timer = window.setInterval(() => {
      step(); draw();
      if (st.done || st.iter > 200) stop();
    }, 350);
  }
  function stop(): void { if (timer) { clearInterval(timer); timer = 0; } runBtn.textContent = "run"; }

  function restarts(): void {
    const { X, n } = dataX(), d = D();
    if (!n) return;
    const how: KMeansInit = st.init === "hand" ? "random" : st.init;
    const r = rng(1000 + st.seed), Js: number[] = [];
    for (let t = 0; t < 100; t++) Js.push(kmeans(X, n, d, st.k, { init: initCentres(X, n, d, Math.min(st.k, n), how, r) }).J);
    st.restarts = Js;
    draw();
  }

  // ---- drawing ----
  function buildHandles(): void {
    gHandles.replaceChildren();
    for (let j = 0; j < st.k; j++) {
      const g = svgEl("g", {}, gHandles);
      const c = svgEl("circle", { r: 13, "stroke-width": 4 }, g);
      c.style.setProperty("fill", "var(--qb-bg, #fff)");
      c.style.setProperty("stroke", COLS[j]);
      svgEl("path", { d: "M-7,0H7M0,-7V7", "stroke-width": 3, stroke: COLS[j] }, g);
      g.addEventListener("pointerdown", ev => ev.stopPropagation());
      draggable(g as unknown as SVGElement, ev => pg.toData(ev), (x, y) => {
        x = Math.min(1, Math.max(0, x)); y = Math.min(1, Math.max(0, y));
        if (st.mode === "1d") st.C[j] = +(x * R1).toFixed(2); else { st.C[2 * j] = x; st.C[2 * j + 1] = y * st.yscale; }
        st.C0 = st.C.slice(); st.init = "hand";
        st.Js = []; st.iters = []; st.iter = 0; st.done = false; st.next = "assign";
        st.trails = Array.from({ length: st.k }, (_, i) => [toScreen(i)]);
        if (st.assign) doAssign();
        draw();
      });
    }
  }

  function draw(): void {
    const d = D(), pts = pg.points;
    // point colours = cluster (0 = not assigned)
    pts.forEach((p, i) => { p.c = st.assign && st.assign.length === pts.length ? st.assign[i] + 1 : 0; });
    // Voronoi cells (distances with the y scale; in 1D vertical bands)
    const rgb = COLS.map(h => hexToRgb(h).map(v => 255 * v));
    pg.setBackgroundColour(st.C.length ? (x, y) => {
      let b = 0, bd = Infinity;
      for (let j = 0; j < st.k; j++) {
        const q = d === 1 ? (x * R1 - st.C[j]) ** 2 : (x - st.C[2 * j]) ** 2 + (y * st.yscale - st.C[2 * j + 1]) ** 2;
        if (q < bd) { bd = q; b = j; }
      }
      return [rgb[b][0], rgb[b][1], rgb[b][2], 0.16];
    } : null);
    pg.redraw();
    // handles and trails
    Array.from(gHandles.children).forEach((g, j) => {
      const [x, y] = toScreen(j);
      g.setAttribute("transform", `translate(${pg.sx(x).toFixed(1)},${pg.sy(y).toFixed(1)})`);
    });
    gTrail.replaceChildren();
    st.trails.forEach((t, j) => {
      if (t.length < 2) return;
      const pl = svgEl("polyline", { points: t.map(([x, y]) => `${pg.sx(x).toFixed(1)},${pg.sy(y).toFixed(1)}`).join(" "), fill: "none", "stroke-width": 2.5, "stroke-dasharray": "5 4" }, gTrail);
      pl.style.setProperty("stroke", COLS[j]);
      for (const [x, y] of t.slice(0, -1)) { const c = svgEl("circle", { cx: pg.sx(x).toFixed(1), cy: pg.sy(y).toFixed(1), r: 4 }, gTrail); c.style.setProperty("fill", COLS[j]); }
    });
    // 1D axis with values
    gAxis.replaceChildren();
    if (st.mode === "1d") {
      const y0 = pg.sy(0.5);
      const l = svgEl("line", { x1: pg.sx(0), x2: pg.sx(1), y1: y0, y2: y0, "stroke-width": 1.5 }, gAxis);
      l.style.setProperty("stroke", "var(--qb-axis, #444)");
      for (let v = 0; v <= R1; v++) {
        svgEl("line", { x1: pg.sx(v / R1), x2: pg.sx(v / R1), y1: y0 + 14, y2: y0 + 22, "stroke-width": 1.5, stroke: "#666" }, gAxis);
        const t = svgEl("text", { x: pg.sx(v / R1), y: y0 + 44, "text-anchor": "middle" }, gAxis);
        t.textContent = String(v); t.style.fontSize = "18px";
      }
    }
    // J plot
    const xs = st.Js.map((_, i) => i);
    jPlot.setDomain([0, Math.max(4, xs.length)], [0, st.Js.length ? Math.max(1e-9, ...st.Js) * 1.1 : 1]);
    jLine.set(xs, st.Js); jDots.set(xs, st.Js);
    // restarts
    if (st.restarts) {
      const Js = st.restarts, lo = Math.min(...Js), hi = Math.max(...Js), span = hi - lo || Math.max(1e-3, hi * 0.05);
      const nb = 20, e = Array.from({ length: nb + 1 }, (_, i) => lo - span * 0.05 + i * span * 1.1 / nb), c = new Array(nb).fill(0);
      Js.forEach(v => { c[Math.min(nb - 1, Math.floor((v - e[0]) / (e[nb] - e[0]) * nb))]++; });
      hPlot.setDomain([e[0], e[nb]], [0, Math.max(...c) * 1.15]);
      hBars.set(e, c);
    } else { hBars.set([], []); }
    const Jnow = st.Js.length ? st.Js[st.Js.length - 1] : NaN;
    hCur.set(st.restarts ? Jnow : NaN);
    // readout
    const cs = Array.from({ length: st.k }, (_, j) => d === 1 ? f2(st.C[j]) : `(${f2(st.C[2 * j])}, ${f2(st.C[2 * j + 1] / st.yscale)})`);
    let s = `K = ${st.k}   points: ${pts.length}   iterations: ${st.iter}` + (st.done ? "   converged" : "") + "\n";
    s += `centres: ${cs.join("  ")}\n`;
    s += `J = ${Number.isFinite(Jnow) ? (d === 1 ? f2(Jnow) : f3(Jnow)) : "–"}` + (st.next === "update" ? "   (next: update)" : st.assign ? "   (next: assign)" : "   (press assign or step)");
    if (st.restarts) {
      const Js = st.restarts, best = Math.min(...Js);
      const atBest = Js.filter(v => v <= best * (1 + 1e-6) + 1e-12).length;
      s += `\n100 restarts: best J = ${d === 1 ? f2(best) : f3(best)}, reached by ${atBest} runs; worst ${d === 1 ? f2(Math.max(...Js)) : f3(Math.max(...Js))}`;
    }
    ro.textContent = s;
    sync();
  }

  function sync(): void {
    initBtns.forEach((b, i) => b.classList.toggle("active", ["random", "kmeans++", "hand"][i] === st.init));
    remBtn.classList.toggle("active", pg.tool === "remove");
    yS.input.parentElement!.style.display = st.mode === "1d" ? "none" : "";
  }

  load(p0);
  // optional: run a number of steps at load (data-steps), e.g. for a figure
  for (let i = 0; i < (Number(host.dataset.steps) || 0); i++) step();
  if (Number(host.dataset.steps)) draw();
  if (host.dataset.restarts) restarts();
}

autoMount(NAME, mount);
