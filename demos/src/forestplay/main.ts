// Random-forest playground. Left: points of two or three classes in the plane, with the
// forest's averaged class probabilities painted behind them (colour = majority, strength =
// confidence). Middle: the maps of six individual trees, and the out-of-bag and test error
// against the number of trees. Sliders: number of trees B, maximum depth, minimum leaf size,
// features per split m; the data can carry extra derived features (x + y, x − y) and pure
// noise features, so that m has something to choose from. Readouts: OOB error, test error,
// and the average correlation of the trees' maps (how alike the trees are).
// Trees come from lib/forest.ts. Auto-mounts on <div data-demo="forestplay">.
// Options: data-preset="single|bagged|malike|mone|noise|far|boundary".
import { htmlEl, slider, presets, checkbox, autoMount, Slider } from "../lib/ui";
import { pointPlayground, Playground } from "../lib/playground";
import { PRESETS, Preset, makeData, featureTable } from "./data";
import { Plot, seriesColor } from "../lib/plot";
import { hexToRgb, QUALITATIVE } from "../lib/colour";
import * as FO from "../lib/forest";

const NAME = "forestplay";
const RES = 64;
const STRIP = 6;

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .fp-left { flex: 0 1 min(380px, calc(100vh - 230px)); min-width: 240px; }
[data-demo="${NAME}"] .fp-left .qb-plot { max-width: none; }
[data-demo="${NAME}"] .fp-mid { flex: 1 1 300px; min-width: 220px; max-width: 400px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .fp-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
[data-demo="${NAME}"] .fp-strip figure { margin: 0; }
[data-demo="${NAME}"] .fp-strip canvas { width: 100%; aspect-ratio: 1; display: block; border: 1px solid var(--qb-grid); image-rendering: pixelated; }
[data-demo="${NAME}"] .fp-strip figcaption { font-size: 14px; text-align: center; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-panel { flex: 0 0 270px; gap: .3em; font-size: 15px; }
[data-demo="${NAME}"] .fp-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-line; }
[data-demo="${NAME}"] .fp-cap { font-size: 15px; color: var(--qb-muted); text-align: center; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .fp-left { flex-basis: 380px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .05em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 12px; padding: .3em .5em; }
.reveal [data-demo="${NAME}"] .fp-mid .qb-plot-svg { max-height: 190px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const pct = (v: number) => (Number.isFinite(v) ? (100 * v).toFixed(1) + " %" : "–");

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const st = { ...p0, test: makeData(p0.data, true) };
  let forest: FO.Forest | null = null;

  const left = htmlEl("div", "fp-left", "", host);
  const mid = htmlEl("div", "fp-mid", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "fp-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const classBox = htmlEl("div", "", "", panel);
  htmlEl("div", "fp-lab", "add points of class", classBox);
  const pg: Playground = pointPlayground(left, {
    data: makeData(p0.data, false), k: 3, resolution: RES, controlsParent: classBox, r: 6,
    names: ["class 1", "class 2", "class 3"], onChange: (_, final) => retrain(final),
  });
  const cap = htmlEl("div", "fp-cap", "forest: average of the trees' votes", left);

  htmlEl("div", "fp-lab", "the forest", panel);
  const bS: Slider = slider(panel, "trees B", 1, 200, 1, st.B, v => { st.B = v; retrain(true); });
  const mS: Slider = slider(panel, "features per split m", 1, 8, 1, st.m, v => { st.m = v; retrain(true); });
  const dS: Slider = slider(panel, "max depth", 1, 20, 1, st.depth, v => { st.depth = v; retrain(true); });
  const lS: Slider = slider(panel, "min leaf size", 1, 20, 1, st.minLeaf, v => { st.minLeaf = v; retrain(true); });
  const bootCb = checkbox(panel, "bootstrap samples", st.bootstrap, v => { st.bootstrap = v; retrain(true); });
  htmlEl("div", "fp-lab", "features", panel);
  const extraCb = checkbox(panel, "add x + y and x − y", st.extra, v => { st.extra = v; retrain(true); });
  const nS: Slider = slider(panel, "noise features", 0, 4, 1, st.noise, v => { st.noise = v; retrain(true); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  // strip of single trees
  htmlEl("div", "fp-lab", "the first six trees, each on its own bootstrap sample", mid);
  const strip = htmlEl("div", "fp-strip", "", mid);
  const minis = Array.from({ length: STRIP }, (_, b) => {
    const fig = htmlEl("figure", "", "", strip);
    const c = htmlEl("canvas", "", "", fig);
    c.width = RES; c.height = RES;
    htmlEl("figcaption", "", `tree ${b + 1}`, fig);
    return c;
  });
  const errPlot = new Plot(mid, { width: 440, height: 230, x: [1, 200], y: [0, 0.5], xlabel: "number of trees", ylabel: "error", yfmt: v => (100 * v).toFixed(0) + "%", margin: { l: 70 }, ariaLabel: "error against the number of trees" });
  const oobLine = errPlot.line([1], [0], { color: seriesColor(0), label: "out of bag" });
  const testLine = errPlot.line([1], [0], { color: seriesColor(1), dash: "7 4", label: "test" });
  errPlot.legend(undefined, "tr");

  // grid of map points
  const gx: number[] = [], gy: number[] = [];
  for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) { gx.push((i + 0.5) / RES); gy.push((j + 0.5) / RES); }
  const rgb = QUALITATIVE.map(h => hexToRgb(h).map(v => 255 * v));

  let pending = 0;
  function retrain(final: boolean): void {
    if (!final) { cancelAnimationFrame(pending); pending = requestAnimationFrame(() => compute(false)); return; }
    compute(true);
  }

  function compute(full: boolean): void {
    const pts = pg.points;
    if (!pts.length) { forest = null; pg.setBackground(null); ro.textContent = "no points"; return; }
    const F = { extra: st.extra, noise: st.noise }, d = 2 + (F.extra ? 2 : 0) + F.noise;
    if (st.m > d) { st.m = d; mS.set(d); }
    mS.input.max = String(d);
    const T = featureTable(pts.map(p => p.x), pts.map(p => p.y), F, 1);
    const k = Math.max(2, ...pts.map(p => p.c + 1));
    const y = pts.map(p => p.c);
    const B = full ? st.B : Math.min(st.B, 40);
    forest = FO.trainForest(FO.binTable(T, 1024), FO.classTarget(y, k), null, {
      nTrees: B, mtry: st.m, maxDepth: st.depth, minLeaf: st.minLeaf, bootstrap: st.bootstrap, seed: 7,
    });
    const G = featureTable(gx, gy, F, 2);
    const P = FO.predictForest(forest, G.X, G.n);
    pg.setBackground((x, yy) => {
      const i = Math.min(RES - 1, Math.max(0, Math.floor(x * RES))), j = Math.min(RES - 1, Math.max(0, Math.floor(yy * RES)));
      const q = (j * RES + i) * k;
      return P.subarray(q, q + k);
    });
    // single trees
    minis.forEach((c, b) => {
      const ctx = c.getContext("2d")!, id = ctx.createImageData(RES, RES);
      const t = forest!.trees[b];
      for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
        const g = j * RES + i, o = 4 * ((RES - 1 - j) * RES + i);
        if (!t) { id.data[o + 3] = 0; continue; }
        const leaf = FO.leafOfRow(t, G.X, G.n, g);
        let r = 0, gg = 0, bb = 0;
        for (let cc = 0; cc < k; cc++) { const w = t.value[leaf * k + cc]; r += w * rgb[cc][0]; gg += w * rgb[cc][1]; bb += w * rgb[cc][2]; }
        id.data[o] = 255 - 0.55 * (255 - r); id.data[o + 1] = 255 - 0.55 * (255 - gg); id.data[o + 2] = 255 - 0.55 * (255 - bb); id.data[o + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      c.style.opacity = t ? "1" : "0.25";
    });
    // curves and readouts
    const Tt = featureTable(st.test.points.map(p => p.x), st.test.points.map(p => p.y), F, 3);
    const yt = st.test.points.map(p => p.c);
    const oc = FO.oobErrorCurve(forest, T.X, y), tc = FO.testErrorCurve(forest, Tt.X, Tt.n, yt);
    const xsB = Array.from({ length: B }, (_, b) => b + 1);
    errPlot.setDomain([1, Math.max(2, st.B)], [0, Math.max(0.1, ...Array.from(tc), ...Array.from(oc).filter(Number.isFinite)) * 1.1]);
    oobLine.set(xsB.filter((_, b) => Number.isFinite(oc[b])), Array.from(oc).filter(Number.isFinite));
    testLine.set(xsB, tc);
    const rho = FO.treeCorrelation(forest, G.X, G.n, k > 2 ? 0 : 1, 30);
    const oobE = forest.trees.length > 1 && st.bootstrap ? FO.oobError(forest, T.X, y) : NaN;
    const P2 = FO.predictForest(forest, T.X, T.n);
    const trainE = FO.errorOf(P2, k, y);
    ro.textContent = `d = ${d} features, m = ${st.m}${full ? "" : " (preview)"}\ntraining error ${pct(trainE)}\nOOB error ${st.bootstrap ? pct(oobE) : "– (no bootstrap)"}\ntest error ${pct(tc[B - 1])}\ntree correlation ρ̂ ${Number.isFinite(rho) ? rho.toFixed(2) : "–"}`;
    cap.textContent = forest.trees.length === 1 ? "a single tree" : `forest of ${forest.trees.length} trees: average of their votes`;
  }

  function load(p: Preset): void {
    Object.assign(st, p, { test: makeData(p.data, true) });
    bS.set(p.B); mS.input.max = "8"; mS.set(p.m); dS.set(p.depth); lS.set(p.minLeaf); nS.set(p.noise);
    bootCb.checked = p.bootstrap; extraCb.checked = p.extra;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    pg.setPoints(makeData(p.data, false));
  }

  presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
  compute(true);
}

autoMount(NAME, mount);
