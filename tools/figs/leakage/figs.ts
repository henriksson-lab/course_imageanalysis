// Extra figures for leakage/ (slides). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/leakage/figs.ts
// then python3 -I tools/figs/leakage/thumbs.py <dir>/leakage-set.json (and tools/figs/leakage/augment.py). Numbers are printed.
import * as fs from "fs";
import * as os from "os";
import * as LK from "../../../demos/src/leakage/model";
import * as TP from "../../../demos/src/testpeek/model";
import { ROOT, C, svg, text, line, rect, path, axes, arrow } from "../evaluation/svglib";
const TMP = process.env.FIGTMP || os.tmpdir();

// ------------------------------------------------ data of the leakage demo: images and the accuracy of each split
{
  const base = { ...LK.DEFAULTS }, strong = { ...LK.DEFAULTS, dayEffect: 2 };
  const s1 = LK.makeSet(base), s2 = LK.makeSet(strong);
  const r1 = LK.evaluateAll(s1, base), r2 = LK.evaluateAll(s2, strong);
  const maj1 = LK.majority(s1), maj2 = LK.majority(s2);
  console.log("leakage default:", r1.map(r => `${r.strategy} ${(100 * r.mean).toFixed(1)}`).join(", "), "majority", (100 * maj1).toFixed(1));
  console.log("leakage newday:", r2.map(r => `${r.strategy} ${(100 * r.mean).toFixed(1)}`).join(", "), "majority", (100 * maj2).toFixed(1));
  fs.writeFileSync(TMP + "/leakage-set.json", JSON.stringify({
    S: LK.S, days: Array.from(s1.day), images: s1.images.map(g => Array.from(g.data, v => Math.round(v * 10) / 10)), labels: s1.labels.map(l => Array.from(l)),
    acc: r1.map(r => ({ id: r.strategy, label: r.label, mean: r.mean })), acc2: r2.map(r => ({ id: r.strategy, label: r.label, mean: r.mean })), maj: maj1, maj2,
  }));
  // autocorrelation of the image intensities against the shift (mean over the 12 images, x and y shifts)
  const S = LK.S, maxd = 16, Cd = new Array(maxd + 1).fill(0);
  for (const g of s1.images) {
    let m = 0; for (const v of g.data) m += v; m /= g.data.length;
    let v0 = 0; for (const v of g.data) v0 += (v - m) ** 2; v0 /= g.data.length;
    for (let d = 0; d <= maxd; d++) {
      let c = 0, n = 0;
      for (let y = 0; y < S; y++) for (let x = 0; x + d < S; x++) { c += (g.data[y * S + x] - m) * (g.data[y * S + x + d] - m); n++; }
      for (let y = 0; y + d < S; y++) for (let x = 0; x < S; x++) { c += (g.data[y * S + x] - m) * (g.data[(y + d) * S + x] - m); n++; }
      Cd[d] += c / n / v0 / s1.images.length;
    }
  }
  const ell = Cd.findIndex(v => v < 0.2);
  console.log("autocorr", Cd.map(v => v.toFixed(2)).join(" "), "falls below 0.2 at", ell);
  const ax = axes(80, 20, 330, 220, [0, maxd], [-0.2, 1], [0, 4, 8, 12, 16], [0, 0.5, 1], "shift δ (pixels)", "correlation C(δ)");
  let body = ax.s + line(ax.X(0), ax.Y(0), ax.X(maxd), ax.Y(0), "#999", 1.5);
  body += path(Cd.map((v, d) => [ax.X(d), ax.Y(v)] as [number, number]), { stroke: C.blue, w: 3.5 });
  body += line(ax.X(ell), ax.Y(-0.2), ax.X(ell), ax.Y(1), C.fp, 2, "6 4") + text(ax.X(ell) + 8, ax.Y(0.85), `ℓ ≈ ${ell} px`, { size: 19, anchor: "start", fill: C.fp });
  fs.writeFileSync(ROOT + "leakage/fig-autocorr.svg", svg(440, 310, body));
}

// ------------------------------------------------ fig-kfold.svg: 4-fold cross-validation
{
  let body = "";
  const x0 = 110, w = 80, h = 40;
  for (let f = 0; f < 4; f++) {
    body += text(x0 - 15, 45 + f * 55 + 27, `fold ${f + 1}`, { size: 19, anchor: "end" });
    for (let k = 0; k < 4; k++) body += rect(x0 + k * (w + 6), 45 + f * 55, w, h, { fill: k === f ? C.fp : C.blue, op: 0.75 });
  }
  body += text(x0 + 2 * (w + 6), 28, "the data, in 4 parts", { size: 19 });
  body += rect(x0, 280, 20, 20, { fill: C.blue, op: 0.75 }) + text(x0 + 28, 296, "train", { size: 19, anchor: "start" });
  body += rect(x0 + 110, 280, 20, 20, { fill: C.fp, op: 0.75 }) + text(x0 + 138, 296, "test", { size: 19, anchor: "start" });
  body += text(x0 + 2 * (w + 6), 335, "score = mean over the 4 folds", { size: 19 });
  fs.writeFileSync(ROOT + "leakage/fig-kfold.svg", svg(470, 350, body));
}

// ------------------------------------------------ fig-bestofk.svg: best of 50 test scores vs the same method on fresh data
{
  const p = TP.PRESETS.find(q => q.id === "fifty")!, st = { ...TP.DEFAULTS, ...p.settings };
  const m = TP.runMany(st, 1000, 7);
  console.log("bestofk: reported", (100 * m.meanReported).toFixed(1), "fresh", (100 * m.meanFresh).toFixed(1), "optimism", (100 * m.optimism).toFixed(1));
  const lo = 0.66, hi = 0.98, nb = 32, bw = (hi - lo) / nb;
  const hist = (x: Float64Array) => { const c = new Array(nb).fill(0); x.forEach(v => { const k = Math.floor((v - lo) / bw + 1e-9); if (k >= 0 && k < nb) c[k]++; }); return c; };
  const hr = hist(m.reported), hf = hist(m.fresh), top = Math.max(...hr, ...hf) * 1.15;
  const ax = axes(70, 50, 400, 220, [lo, hi], [0, top], [0.7, 0.8, 0.9], [], "accuracy", "how often (1000 runs)", v => `${Math.round(v * 100)} %`);
  let body = ax.s;
  const bars = (c: number[], col: string) => c.forEach((v, k) => { if (v) body += rect(ax.X(lo + k * bw) + 0.5, ax.Y(v), ax.X(lo + (k + 1) * bw) - ax.X(lo + k * bw) - 1, ax.Y(0) - ax.Y(v), { fill: col, op: 0.55 }); });
  bars(hf, C.tp); bars(hr, C.fp);
  body += line(ax.X(m.meanFresh), ax.Y(0), ax.X(m.meanFresh), 45, C.tp, 3) + line(ax.X(m.meanReported), ax.Y(0), ax.X(m.meanReported), 45, C.fp, 3);
  body += text(ax.X(m.meanFresh) - 6, 38, `fresh data ${(100 * m.meanFresh).toFixed(1)} %`, { size: 19, anchor: "end", fill: "#1b7a1b" });
  body += text(ax.X(m.meanReported) + 6, 38, `reported ${(100 * m.meanReported).toFixed(1)} %`, { size: 19, anchor: "start", fill: C.fp });
  fs.writeFileSync(ROOT + "leakage/fig-bestofk.svg", svg(560, 340, body));
}

// ------------------------------------------------ fig-tvt.svg: training, validation, test
{
  let body = "";
  const parts: [string, number, string, string, string][] = [["training", 250, C.blue, "fit the model", ""], ["validation", 210, C.purple, "make every choice", "(settings, features …)"], ["test", 170, C.fp, "report once,", "at the end"]];
  let x = 20;
  for (const [name, w, col, l1, l2] of parts) {
    body += rect(x, 30, w - 6, 60, { fill: col, op: 0.75 }) + text(x + (w - 6) / 2, 68, name, { fill: "#fff", weight: "bold" });
    body += text(x + (w - 6) / 2, 125, l1, { size: 19 }) + (l2 ? text(x + (w - 6) / 2, 150, l2, { size: 18 }) : "");
    x += w;
  }
  fs.writeFileSync(ROOT + "leakage/fig-tvt.svg", svg(670, 165, body));
}
