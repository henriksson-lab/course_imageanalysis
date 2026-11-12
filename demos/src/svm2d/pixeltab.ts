// The SVM day's plug-in tab for the pixelclass shell (registered in pixelclass/main.ts,
// shown with data-tabs="svmcv"). Two parts:
//  - SVM settings in one click: unscaled features, a too large γ, linear vs RBF, one-vs-rest
//    vs one-vs-one (they set the SVM's parameters through the shell and retrain);
//  - cross-validation of (C, γ) on the painted pixels, with folds BY STROKE (whole strokes
//    held out) or BY PIXEL (random pixels held out: neighbours of held-out pixels stay in
//    training, so the error is optimistic). The map shows the CV error for every cell of a
//    log grid; "use the best" copies the best (C, γ) to the classifier.
// The cross-validation trains on a class-stratified random subset of at most 300 painted
// pixels (seeded) and runs cell by cell, so painting stays responsive.
import { htmlEl, button, presets } from "../lib/ui";
import { heatmapGrid } from "../lib/plot";
import { viridis } from "../lib/colour";
import { svgEl } from "../lib/ui";
import { makeFolds, cvError, fitPredictor, logGrid } from "../lib/svm";
import { rng } from "../lib/random";
import type { PixelclassTab, ShellContext } from "../pixelclass/plugins";

const CV_MAX = 300;
const fmt = (v: number) => (v >= 1 ? String(Math.round(v)) : String(v));

export const SVM_CV_TAB: PixelclassTab = {
  id: "svmcv",
  label: "SVM: settings and cross-validation of (C, γ)",
  mount(parent: HTMLElement, ctx: ShellContext) {
    const msg = htmlEl("div", "qb-note", "", parent);
    const setRow = htmlEl("div", "", "", parent);
    presets(setRow, [
      { label: "standardised", apply: () => ctx.setParams({ standardise: 1 }) },
      { label: "unscaled features", apply: () => ctx.setParams({ standardise: 0 }) },
      { label: "γ = 10 (too large)", apply: () => ctx.setParams({ kernel: 1, gamma: 1 }) },
      { label: "linear", apply: () => ctx.setParams({ kernel: 0 }) },
      { label: "RBF", apply: () => ctx.setParams({ kernel: 1 }) },
      { label: "one-vs-rest", apply: () => ctx.setParams({ ovo: 0 }) },
      { label: "one-vs-one", apply: () => ctx.setParams({ ovo: 1 }) },
    ]);
    const cvRow = htmlEl("div", "qb-presets", "", parent);
    let mode: "stroke" | "pixel" = "stroke";
    const modeBtns = [
      button(cvRow, "folds by stroke", () => { mode = "stroke"; sync(); }),
      button(cvRow, "folds by pixel", () => { mode = "pixel"; sync(); }),
    ];
    button(cvRow, "run CV", () => run());
    const prog = htmlEl("div", "qb-note", "", parent);
    const mapBox = htmlEl("div", "", "", parent);
    const cmp = htmlEl("div", "qb-note", "", parent);
    const Cs = logGrid(-1, 3), gs = logGrid(-3, 1);
    const results: Partial<Record<"stroke" | "pixel", { E: Float64Array; best: number }>> = {};
    let job = 0;

    function sync(): void {
      modeBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (mode === "stroke")));
      const isSvm = ctx.classifier().id === "svm";
      msg.textContent = isSvm ? "" : "Choose the SVM as classifier to use this panel.";
      draw();
    }

    function draw(): void {
      mapBox.replaceChildren();
      const r = results[mode];
      if (!r) return;
      const hm = heatmapGrid(mapBox, r.E, gs.length, Cs.length, {
        map: viridis, lo: 0, hi: 0.5, fmt: v => (100 * v).toFixed(0) + "%", cell: 46, font: 16,
        rowLabels: gs.map(g => "γ " + fmt(g)), colLabels: Cs.map(c => fmt(c)), title: `CV error, folds by ${mode} (columns: C)`,
      });
      const cell = 46, f = 16, lw = Math.max(...gs.map(g => ("γ " + fmt(g)).length)) * f * 0.6 + 10, th = f * 1.5 + f * 1.4;
      const row = Math.floor(r.best / Cs.length), col = r.best % Cs.length;
      const box = svgEl("rect", { x: lw + col * cell + 1.5, y: th + row * cell + 1.5, width: cell - 3, height: cell - 3, fill: "none" }, hm.svg);
      box.style.setProperty("stroke", "#e6a700"); box.style.setProperty("stroke-width", "4");
      button(mapBox, `use the best: C = ${fmt(Cs[col])}, γ = ${fmt(gs[row])}`, () => ctx.setParams({ kernel: 1, C: Math.log10(Cs[col]), gamma: Math.log10(gs[row]) }));
      const a = results.stroke, b = results.pixel;
      cmp.textContent = a && b ? `best cell: by stroke ${(100 * a.E[a.best]).toFixed(1)}% error, by pixel ${(100 * b.E[b.best]).toFixed(1)}% error.` : "";
    }

    function run(): void {
      const data = ctx.training();
      if (ctx.classifier().id !== "svm") { sync(); return; }
      if (!data) { prog.textContent = "paint at least two classes first"; return; }
      const my = ++job, m0 = mode;
      // stratified subset, row-major
      const r = rng(7), keep = Math.min(1, CV_MAX / data.n), idx: number[] = [];
      for (let i = 0; i < data.n; i++) if (r.unif() < keep) idx.push(i);
      const n = idx.length, d = data.d, X = new Float64Array(n * d), cls = new Int32Array(n), grp = new Int32Array(n);
      idx.forEach((i, q) => { for (let j = 0; j < d; j++) X[q * d + j] = data.X[j * data.n + i]; cls[q] = data.y[i]; grp[q] = data.groups ? data.groups[i] : i; });
      const nGroups = new Set(grp).size;
      const nF = m0 === "stroke" ? Math.min(5, nGroups) : 5;
      if (nF < 2) { prog.textContent = "paint at least two strokes"; return; }
      const folds = m0 === "stroke" ? makeFolds(n, nF, cls, grp) : makeFolds(n, nF, cls, undefined, 3);
      const p = ctx.params(), std = (p.standardise ?? 1) >= 0.5, ovo = (p.ovo ?? 1) >= 0.5;
      const E = new Float64Array(Cs.length * gs.length).fill(NaN);
      let cell = 0;
      const step = () => {
        if (my !== job) return;
        const t0 = performance.now();
        while (cell < E.length && performance.now() - t0 < 60) {
          const gi = Math.floor(cell / Cs.length), ci = cell % Cs.length;
          E[cell] = cvError(n, cls, folds, nF, tr => fitPredictor(X, d, cls, data.k, tr, { kernel: { type: "rbf", gamma: gs[gi] }, C: Cs[ci], tol: 1e-3 }, { standardise: std, mode: ovo ? "ovo" : "ovr" }));
          cell++;
        }
        prog.textContent = cell < E.length ? `cross-validating … ${cell} of ${E.length} cells` : `${nF} folds by ${m0}, ${n} pixels from ${nGroups} strokes${std ? "" : ", features not standardised"}`;
        if (cell < E.length) { setTimeout(step, 0); return; }
        let best = 0;
        E.forEach((v, i) => { if (v < E[best] - 1e-12) best = i; });
        results[m0] = { E, best };
        if (mode === m0) draw();
      };
      prog.textContent = "cross-validating …";
      setTimeout(step, 0);
    }
    ctx.onTrained(() => { if (!msg.textContent && ctx.classifier().id !== "svm") sync(); else if (msg.textContent && ctx.classifier().id === "svm") sync(); });
    sync();
  },
};
