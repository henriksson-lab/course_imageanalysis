// Pseudoreplication: a simulated experiment with two conditions, E experiments and m cells
// per experiment and condition, a day-to-day SD and a cell-to-cell SD. SuperPlot: cells as
// small dots coloured by experiment, experiment means as big markers (joined in the paired
// design). Three analyses side by side: cells as n, experiment means as n, paired experiment
// means: effect, 95 % CI and p. 1000 repetitions give the fraction of p < 0.05 (false
// positives when the true effect is 0, power otherwise). A small plot shows the standard error
// of the effect against the number of cells: what it really is and what cells-as-n claims.
// Auto-mounts on <div data-demo="pseudorep">. Options: data-preset="cells|more-cells|more-exp|
// paired|confounded", data-panels="superplot,ci,runs,se,controls".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor, plotFigure } from "../lib/plot";
import { QUALITATIVE } from "../lib/colour";
import { rng } from "../lib/random";
import { mean, TestResult } from "../lib/stats";
import * as M from "./model";

const NAME = "pseudorep";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .pr-sp { flex: 0 1 360px; min-width: 240px; }
[data-demo="${NAME}"] .pr-mid { flex: 1 1 330px; max-width: 420px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 260px; gap: .35em; }
[data-demo="${NAME}"] .pr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .pr-hidden { display: none !important; }
[data-demo="${NAME}"] .qb-readout { font-size: 13.5px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-readout { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// lib/plot.ts numbers its clip paths per bundle ("qb-clip-1", …), so two demos on one page
// share ids and clip each other's plots. Give every plot of this demo a unique clip id.
let clipCount = 0;
function uniqueClip<T extends { svg: SVGSVGElement; layer: SVGGElement }>(p: T): T {
  const cp = p.svg.querySelector("clipPath");
  if (cp) { cp.id = `${NAME}-clip-${++clipCount}`; p.layer.setAttribute("clip-path", `url(#${cp.id})`); }
  return p;
}

const fp = (p: number) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));
const DESIGNS: { id: M.Settings["design"]; label: string }[] = [{ id: "independent", label: "independent" }, { id: "paired", label: "paired" }, { id: "confounded", label: "confounded" }];

function mount(host: HTMLElement): void {
  injectStyle();
  const panels = new Set((host.dataset.panels ?? "superplot,ci,runs,se,controls").split(/[\s,]+/));
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[0];
  let st: M.Settings = { ...M.DEFAULTS, ...p0.settings }, seed = p0.seed ?? 1;
  const slide = !!host.closest(".reveal"), fs = slide ? 24 : 18;

  const spF = plotFigure(host, "SuperPlot: cells and experiment means", "pr-sp qb-figure");
  const mid = htmlEl("div", "pr-mid", "", host);
  const ciF = plotFigure(mid, "effect with 95 % CI, three analyses");
  const runsF = plotFigure(mid, "1000 repetitions: share with p < 0.05");
  const runsCap = runsF.querySelector("figcaption")!;
  const seF = plotFigure(host, "SE of the effect vs cells per experiment", "qb-plot qb-figure");
  seF.style.maxWidth = "360px";
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pr-lab", "presets", panel);
  const pbtn = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "pr-lab", "design", panel);
  const drow = htmlEl("div", "qb-presets", "", panel);
  const dbtn = DESIGNS.map(d => button(drow, d.label, () => set({ design: d.id })));
  const eS = slider(panel, "experiments E", 2, 10, 1, st.E, v => set({ E: v }));
  const mOf = (u: number) => Math.round(10 ** u), uOf = (m: number) => Math.log10(m);
  const mS = slider(panel, "cells per exp. m", 0.7, 3, 0.05, uOf(st.m), u => set({ m: mOf(u) }), u => String(mOf(u)));
  const bS = slider(panel, "day-to-day SD", 0, 2, 0.1, st.sdBetween, v => set({ sdBetween: v }));
  const wS = slider(panel, "cell-to-cell SD", 0.5, 4, 0.1, st.sdWithin, v => set({ sdWithin: v }));
  const fS = slider(panel, "true effect", 0, 2, 0.1, st.effect, v => set({ effect: v }));
  const batS = slider(panel, "batch shift", 0, 2, 0.1, st.batch, v => set({ batch: v }));
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "new experiment", () => { seed++; drawOnce(); });
  button(row, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);

  const hide = (el: HTMLElement, key: string) => { if (!panels.has(key)) el.classList.add("pr-hidden"); };
  hide(spF, "superplot"); hide(ciF, "ci"); hide(runsF, "runs"); hide(seF, "se"); hide(panel, "controls");

  function set(p: Partial<M.Settings>): void { st = { ...st, ...p }; pbtn.forEach(b => b.classList.remove("active")); draw(); }
  function load(p: M.Preset): void {
    st = { ...M.DEFAULTS, ...p.settings }; seed = p.seed ?? 1;
    eS.set(st.E); mS.set(uOf(st.m)); bS.set(st.sdBetween); wS.set(st.sdWithin); fS.set(st.effect); batS.set(st.batch);
    pbtn.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i] === p));
    draw();
  }

  function drawOnce(): void {
    dbtn.forEach((b, i) => b.classList.toggle("active", DESIGNS[i].id === st.design));
    batS.input.disabled = st.design !== "confounded";
    const { data, res } = M.runOnce(st, seed);
    // SuperPlot
    spF.querySelector("svg")?.remove();
    const all = data.flatMap(d => Array.from(d.values));
    const lo = Math.min(...all), hi = Math.max(...all), pad = 0.05 * (hi - lo);
    const sp = new Plot(spF, { x: [0.4, 2.6], y: [lo - pad, hi + pad], ylabel: "measured feature", height: 440, width: 420, font: fs, xticks: [1, 2], xfmt: v => (v === 1 ? "control" : v === 2 ? "treated" : "") });
    uniqueClip(sp);
    const jr = rng(99);
    const nExp = Math.max(...data.map(d => d.exp)) + 1;
    const colour = (e: number) => QUALITATIVE[e % QUALITATIVE.length];
    const cap = Math.min(st.m, 80);
    for (const d of data) {
      const xs: number[] = [], ys: number[] = [];
      for (let i = 0; i < cap; i++) { xs.push(1 + d.cond + 0.32 * (jr.unif() - 0.5) * 2 * 0.8); ys.push(d.values[i]); }
      sp.scatter(xs, ys, { color: colour(d.exp), r: 2.6, fillOpacity: 0.55, stroke: "none" });
    }
    const means = data.map(d => ({ x: 1 + d.cond + (d.exp / Math.max(1, nExp - 1) - 0.5) * 0.3, y: mean(d.values), e: d.exp, c: d.cond }));
    if (st.design === "paired") for (let e = 0; e < nExp; e++) {
      const a = means.find(q => q.e === e && q.c === 0), b = means.find(q => q.e === e && q.c === 1);
      if (a && b) sp.line([a.x, b.x], [a.y, b.y], { color: colour(e), width: 2.5 });
    }
    sp.scatter(means.map(q => q.x), means.map(q => q.y), { colors: means.map(q => colour(q.e)), r: 10, stroke: "var(--qb-fg)" });
    // CI panel
    ciF.querySelector("svg")?.remove();
    const rows: [string, TestResult | null][] = [["cells as n", res.cells], ["experiments as n", res.means], ["paired", res.paired]];
    const lim = Math.max(1, ...rows.flatMap(([, r]) => (r ? [Math.abs(r.ci[0]), Math.abs(r.ci[1])] : [])), st.effect + 0.5);
    const ci = new Plot(ciF, { x: [-lim, lim], y: [0.4, 4.1], xlabel: "treated − control", width: 420, height: slide ? 300 : 270, font: fs, yticks: [], margin: { l: 14 } });
    uniqueClip(ci);
    ci.vline(0, { color: "var(--qb-muted)", width: 1.5, dash: "" });
    ci.vline(st.effect, { color: seriesColor(2), width: 2, text: "true" });
    rows.forEach(([lab, r], k) => {
      const y = 3 - k;
      if (!r) { ci.text(-lim * 0.95, y + 0.25, lab + ": needs the paired design", { size: 17, color: "var(--qb-muted)" }); return; }
      ci.line([r.ci[0], r.ci[1]], [y, y], { color: seriesColor(k === 0 ? 1 : 0), width: 5 });
      ci.scatter([r.estimate], [y], { color: seriesColor(k === 0 ? 1 : 0), r: 7 });
      ci.text(-lim * 0.97, y + 0.38, `${lab}: n = ${r.n}, p = ${fp(r.p)}`, { size: 17 });
    });
    const line = (lab: string, r: TestResult | null) => (r ? `${lab}: ${r.estimate.toFixed(2)}, p ${fp(r.p)}\n  CI [${r.ci[0].toFixed(2)}, ${r.ci[1].toFixed(2)}]` : `${lab}: paired design only`);
    ro.textContent = [line("cells as n", res.cells), line("experiments as n", res.means), line("paired", res.paired)].join("\n");
  }

  function drawRuns(): void {
    const many = M.runMany(st, 1000, 11);
    runsF.querySelector("svg")?.remove();
    runsCap.textContent = st.effect === 0 && st.design !== "confounded" ? "1000 repetitions: share with p < 0.05 (no effect: false positives)" : st.effect === 0 ? "1000 repetitions: share with p < 0.05 (no effect, only the batch)" : "1000 repetitions: share with p < 0.05 (true effect: power)";
    const pl = new Plot(runsF, { x: [0.4, 3.6], y: [0, 1.15], yticks: [0, 0.2, 0.4, 0.6, 0.8, 1], ylabel: "share p < 0.05", width: 420, height: 250, font: fs, xticks: [1, 2, 3], xfmt: v => ["", "cells as n", "exp. as n", "paired"][Math.round(v)] ?? "" });
    uniqueClip(pl);
    const vals = [many.cells, many.means, Number.isFinite(many.paired) ? many.paired : 0];
    pl.bars([1, 2, 3], vals, { colors: [seriesColor(1), seriesColor(0), seriesColor(0)], barWidth: 0.6 });
    pl.hline(0.05, { color: seriesColor(2), width: 2, text: st.effect === 0 ? "5 %" : "" });
    vals.forEach((v, k) => { if (k < 2 || Number.isFinite(many.paired)) pl.text(k + 1, v + 0.05, (100 * v).toFixed(0) + " %", { anchor: "middle", size: 18 }); });
  }

  function drawSE(): void {
    seF.querySelector("svg")?.remove();
    const ms = Array.from({ length: 60 }, (_, i) => 10 ** (0.5 + 2.5 * i / 59));
    const ind = { ...st, design: "independent" as const }, par = { ...st, design: "paired" as const };
    const ti = ms.map(m => M.seCurves(ind, m).true), tp = ms.map(m => M.seCurves(par, m).true), cl = ms.map(m => M.seCurves(st, m).claimed);
    const top = Math.max(...ti, ...cl) * 1.1;
    const pl = new Plot(seF, { x: [3, 1000], xlog: true, y: [0, top], xlabel: "cells per experiment m", ylabel: "SE of the effect", width: 420, height: 330, font: fs });
    uniqueClip(pl);
    pl.line(ms, ti, { color: seriesColor(0), label: "experiments as n" });
    pl.line(ms, tp, { color: seriesColor(2), label: "paired" });
    pl.line(ms, cl, { color: seriesColor(1), dash: "6 4", label: "cells as n claims" });
    const c = M.seCurves(st, st.m);
    pl.scatter([st.m, st.m], [c.true, c.claimed], { colors: [st.design === "paired" ? seriesColor(2) : seriesColor(0), seriesColor(1)], r: 7 });
    pl.legend(undefined, "tr");
  }

  function draw(): void { drawOnce(); drawRuns(); drawSE(); }
  load(p0);
}

autoMount(NAME, mount);
