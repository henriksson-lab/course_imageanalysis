// EM for a 1D mixture of two (data-k="3": three) Gaussians, one half-step at a time. The points
// lie on a line (drag them); each is coloured by its responsibilities (a mix of the component
// colours). Drawn: each weighted component π_k N(x | μ_k, σ_k²) (dashed) and their sum, the
// mixture density (black). Tables: r_ik per point, and N_k, μ_k, σ_k², π_k per component, plus
// log p(D) after every half-step. "E-step" recomputes the responsibilities from the current
// parameters; "M-step" re-estimates the parameters from the responsibilities. Options: hard
// assignments (0/1), equal variances and weights (with both: k-means), and a variance floor.
//
// <div data-demo="em1d" data-preset="hand|ambiguous|collapse|kmeans|three" data-k="2"></div>
import { autoMount, button, checkbox, htmlEl, presets, draggable, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { QUALITATIVE, hexToRgb } from "../lib/colour";
import { type Mixture, makeMixture, eStep, mStep, normalPdf, cloneMixture } from "../lib/gmm";

const NAME = "em1d";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; }
.qb-demo[data-demo="${NAME}"] .e1-left { flex: 0 1 660px; min-width: 280px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] table { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] td, .qb-demo[data-demo="${NAME}"] th { padding: .1em .45em; text-align: right; border-bottom: 1px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] th { font-weight: 600; text-align: left; }
.qb-demo[data-demo="${NAME}"] .e1-tables { display: flex; flex-wrap: wrap; gap: .4em 1.2em; align-items: flex-start; overflow-x: auto; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .e1-stale td { color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .e1-left { flex: 0 0 680px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 420px; gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] table, .reveal .qb-demo[data-demo="${NAME}"] td, .reveal .qb-demo[data-demo="${NAME}"] th { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .25em .5em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const COLS = QUALITATIVE.slice(0, 3);
const X0 = 0, X1 = 12, FLOOR = 0.1;

interface Preset { id: string; label: string; x: number[]; mu: number[]; v: number[]; pi: number[]; hard?: boolean; equal?: boolean; floor?: boolean }
const PRESETS: Preset[] = [
  { id: "hand", label: "the by-hand example", x: [1, 2, 4, 7, 9, 10], mu: [2, 8], v: [4, 4], pi: [0.5, 0.5] },
  { id: "ambiguous", label: "an ambiguous point in the middle", x: [1, 2, 3, 5.5, 8, 9, 10], mu: [2, 9], v: [1, 1], pi: [0.5, 0.5] },
  { id: "collapse", label: "collapse: one component on one point", x: [1, 1.5, 2.5, 3, 4, 4.5, 10], mu: [3, 10], v: [2, 8], pi: [0.5, 0.5] },
  { id: "kmeans", label: "hard EM = k-means", x: [1, 2, 4, 7, 9, 10], mu: [1, 2], v: [4, 4], pi: [0.5, 0.5], hard: true, equal: true },
  { id: "three", label: "three components", x: [1, 1.5, 2.5, 5, 5.5, 6.5, 9.5, 10, 11], mu: [2, 4, 9], v: [3, 3, 3], pi: [1 / 3, 1 / 3, 1 / 3] },
];

const f2 = (v: number) => v.toFixed(2), f3 = (v: number) => v.toFixed(3);

function mount(host: HTMLElement): void {
  injectStyle();
  const kAttr = Number(host.dataset.k);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? (kAttr === 3 ? PRESETS[4] : PRESETS[0]);
  let xs: number[] = [], m: Mixture = makeMixture([1], [[0]], [1]), m0 = m;
  let resp: Float64Array | null = null, next: "E" | "M" = "E", hard = false, equal = false, floor = false;
  let log: { what: string; L: number }[] = [];

  // ---- layout ----
  const left = htmlEl("div", "e1-left", "", host);
  const plot = new Plot(left, { width: 660, height: 290, x: [X0, X1], y: [0, 0.3], xlabel: "x", ylabel: "density", font: 18, margin: { b: 56 } });
  const tables = htmlEl("div", "e1-tables", "", left);
  const tPts = htmlEl("table", "", "", tables), tPar = htmlEl("table", "", "", tables);
  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  const eBtn = button(row, "E-step", () => { doE(); draw(); });
  const mBtn = button(row, "M-step", () => { doM(); draw(); });
  button(row, "run to convergence", () => { for (let i = 0; i < 500; i++) { const L0 = log.length ? log[log.length - 1].L : -Infinity; if (next === "E") doE(); doM(); doE(); if (Math.abs(log[log.length - 1].L - L0) < 1e-9) break; } draw(); });
  button(row, "reset", () => { m = cloneMixture(m0); resp = null; next = "E"; log = []; draw(); });
  const hardCb = checkbox(panel, "hard assignments (0 or 1)", hard, v => { hard = v; restartFromParams(); });
  const eqCb = checkbox(panel, "equal variances and weights", equal, v => { equal = v; restartFromParams(); });
  const flCb = checkbox(panel, `variance floor ε = ${FLOOR}`, floor, v => { floor = v; restartFromParams(); });
  htmlEl("div", "qb-note", "Drag the points along the line. Colours: the responsibilities (blue = component 1, red = component 2).", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "Reset demo", () => load(p0)).style.alignSelf = "flex-start";

  // curves
  const curves = [0, 1, 2].map(j => plot.fn(() => NaN, { color: COLS[j], width: 3, dash: "7 5" }, 400));
  const total = plot.fn(() => NaN, { color: "var(--qb-fg, #222)", width: 3.5 }, 400);
  const gMu = svgEl("g", {}, plot.over);
  const gPts = svgEl("g", {}, plot.over);

  function load(p: Preset): void {
    xs = p.x.slice();
    m0 = makeMixture(p.pi, p.mu.map(v => [v]), p.v);
    m = cloneMixture(m0);
    hard = !!p.hard; equal = !!p.equal; floor = !!p.floor;
    hardCb.checked = hard; eqCb.checked = equal; flCb.checked = floor;
    resp = null; next = "E"; log = [];
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    buildPoints();
    draw();
  }
  function restartFromParams(): void { resp = null; next = "E"; log = []; draw(); }

  const X = () => Float64Array.from(xs);
  function doE(): void {
    const e = eStep(m, X(), xs.length, undefined, hard);
    resp = e.resp; next = "M";
    log.push({ what: "E", L: e.logLik });
  }
  function doM(): void {
    if (!resp) doE();
    m = mStep(X(), xs.length, 1, resp!, m.k, { covType: equal ? "tied" : "full", equalWeights: equal, floor: floor ? FLOOR : 0, minVar: 1e-10 }, undefined, m);
    next = "E";
    log.push({ what: "M", L: eStep(m, X(), xs.length).logLik });
  }

  function buildPoints(): void {
    gPts.replaceChildren();
    xs.forEach((_, i) => {
      const c = svgEl("circle", { r: 11, stroke: "#fff", "stroke-width": 2 }, gPts);
      draggable(c, ev => plot.toData(ev), x => {
        xs[i] = Math.round(Math.max(X0, Math.min(X1, x)) * 10) / 10;
        if (resp) { resp = null; next = "E"; log = []; }
        draw();
      });
    });
  }

  function draw(): void {
    const k = m.k, rgb = COLS.map(h => hexToRgb(h).map(v => 255 * v));
    // density curves
    let top = 0;
    for (let x = X0; x <= X1; x += 0.02) {
      let s = 0;
      for (let j = 0; j < k; j++) s += m.weights[j] * normalPdf(x, m.means[j], m.covs[j]);
      top = Math.max(top, s);
    }
    plot.setDomain(undefined, [0, Math.min(1.2, Math.max(0.12, top * 1.12))]);
    curves.forEach((c, j) => c.set(j < k ? (x: number) => m.weights[j] * normalPdf(x, m.means[j], m.covs[j]) : () => NaN));
    total.set((x: number) => { let s = 0; for (let j = 0; j < k; j++) s += m.weights[j] * normalPdf(x, m.means[j], m.covs[j]); return s; });
    // points coloured by responsibilities
    const y0 = plot.y(0);
    Array.from(gPts.children).forEach((c, i) => {
      let r = 0, g = 0, b = 0;
      if (resp) for (let j = 0; j < k; j++) { const w = resp[i * k + j]; r += w * rgb[j][0]; g += w * rgb[j][1]; b += w * rgb[j][2]; }
      else { r = g = b = 140; }
      c.setAttribute("cx", plot.x(xs[i]).toFixed(1)); c.setAttribute("cy", y0.toFixed(1));
      c.setAttribute("fill", `rgb(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)})`);
    });
    // mean markers
    gMu.replaceChildren();
    for (let j = 0; j < k; j++) {
      const x = plot.x(m.means[j]), spike = m.covs[j] < 1e-3;
      svgEl("line", { x1: x, x2: x, y1: plot.y(0), y2: plot.box.t, stroke: COLS[j], "stroke-width": spike ? 4 : 2, "stroke-dasharray": spike ? "" : "3 5" }, gMu);
      const t = svgEl("text", { x: x + 4, y: plot.box.t + 18, fill: COLS[j], class: "qb-halo", style: "font-size:18px" }, gMu);
      t.textContent = spike ? `μ${j + 1} (spike)` : `μ${j + 1}`;
    }
    // tables
    const tr = (cells: string[], head = false, cls = "") => `<tr class="${cls}">` + cells.map((c, i) => (head || i === 0 ? `<th>${c}</th>` : `<td>${c}</td>`)).join("") + "</tr>";
    let h = tr(["x", ...xs.map(v => v.toFixed(1))], true);
    for (let j = 0; j < k; j++) h += tr([`<span style="color:${COLS[j]}">r<sub>i${j + 1}</sub></span>`, ...xs.map((_, i) => (resp ? f2(resp[i * k + j]) : "–"))], false, next === "E" && resp ? "e1-stale" : "");
    tPts.innerHTML = h;
    const N = Array.from({ length: k }, (_, j) => (resp ? xs.reduce((s, _x, i) => s + resp![i * k + j], 0) : NaN));
    let h2 = tr(["", "N<sub>k</sub>", "μ<sub>k</sub>", "σ<sub>k</sub>²", "π<sub>k</sub>"], true);
    for (let j = 0; j < k; j++) h2 += tr([`<span style="color:${COLS[j]}">${j + 1}</span>`, Number.isFinite(N[j]) ? f2(N[j]) : "–", f2(m.means[j]), m.covs[j] < 1e-3 ? m.covs[j].toExponential(1) : f2(m.covs[j]), f2(m.weights[j])]);
    tPar.innerHTML = h2;
    // readout
    const Lnow = eStep(m, X(), xs.length).logLik;
    const collapsed = Array.from(m.covs).some(v => v < 1e-3);
    let s = `log p(D) = ${f3(Lnow)}` + (collapsed ? "\ncollapsed: σ² hit the numerical minimum 1e-10; as σ² → 0, log p → ∞" : "") + "\n";
    s += `next: ${next === "E" ? "E-step (responsibilities)" : "M-step (parameters)"}\n`;
    const Ms = log.filter(l => l.what === "M");
    const seq = [log.length ? log[0].L : Lnow, ...Ms.map(l => l.L)];
    s += "log p: start, then after each M-step:\n" + seq.slice(-8).map(f2).join(" → ");
    const drops = seq.some((v, i) => i > 0 && v < seq[i - 1] - 1e-9);
    if (drops) s += "\n(a drop: hard EM does not maximise this likelihood)";
    ro.textContent = s;
    eBtn.classList.toggle("active", next === "E");
    mBtn.classList.toggle("active", next === "M");
  }

  load(p0);
  for (const c of (host.dataset.steps ?? "").split("")) { if (c === "E") doE(); if (c === "M") doM(); }
  if (host.dataset.steps) draw();
}

autoMount(NAME, mount);
