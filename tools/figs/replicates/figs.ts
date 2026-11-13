// Extra figures for replicates/ (slides). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/replicates/figs.ts
// Numbers are printed so the text can quote them.
import * as fs from "fs";
import * as PR from "../../../demos/src/pseudorep/model";
import * as FSm from "../../../demos/src/featurescan/model";
import * as ST from "../../../demos/src/lib/stats";
import { rng } from "../../../demos/src/lib/random";
import { ROOT, C, svg, text, line, circle, rect, path, axes } from "../evaluation/svglib";
const OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"];   // Okabe–Ito

// ------------------------------------------------ fig-tree.svg: experiments > wells > images > cells
{
  let body = "";
  const r = rng(3);
  const ex = [130, 400], wellDx = 70, imgDx = 32;
  ex.forEach((x, e) => {
    body += rect(x - 70, 20, 140, 42, { fill: OI[e], op: 0.85, rx: 6 }) + text(x, 48, `experiment ${e + 1}`, { fill: "#fff", weight: "bold", size: 18 });
    [-1, 1].forEach((sw, w) => {
      const wx = x + sw * wellDx;
      body += line(x, 62, wx, 100, "#888", 2) + circle(wx, 118, 18, { fill: "#fff", stroke: OI[e], w: 3 }) + text(wx, 124, `w${w + 1}`, { size: 16 });
      [-1, 1].forEach(si => {
        const ix = wx + si * imgDx;
        body += line(wx, 136, ix, 168, "#888", 2) + rect(ix - 22, 168, 44, 44, { fill: "#eee", stroke: "#555", w: 1.5 });
        for (let c = 0; c < 6; c++) body += circle(ix - 15 + 30 * r.unif(), 175 + 30 * r.unif(), 3.2, { fill: OI[e] });
      });
    });
  });
  body += text(20, 48, "", {});
  const lab = (y: number, s: string) => text(560, y, s, { size: 18, anchor: "start" });
  body += lab(47, "day, animal, patient") + lab(124, "wells") + lab(195, "images, cells");
  body += text(290, 250, "variation at every level; day-to-day usually largest", { size: 18, fill: "#444" });
  fs.writeFileSync(ROOT + "replicates/fig-tree.svg", svg(760, 265, body));
}

// ------------------------------------------------ fig-semean.svg: the mean of n values wobbles by SD/√n
{
  const r = rng(8), R = 10000, sims = [4, 16].map(n => Float64Array.from({ length: R }, () => { let s = 0; for (let i = 0; i < n; i++) s += r.norm(); return s / n; }));
  console.log("semean SD of means n=4", ST.sd(sims[0]).toFixed(3), "n=16", ST.sd(sims[1]).toFixed(3));
  const lo = -2, hi = 2, nb = 40, bw = (hi - lo) / nb;
  const hs = sims.map(x => { const c = new Array(nb).fill(0); x.forEach(v => { const k = Math.floor((v - lo) / bw); if (k >= 0 && k < nb) c[k]++; }); return c; });
  const top = Math.max(...hs[1]) * 1.1;
  const ax = axes(60, 30, 380, 220, [lo, hi], [0, top], [-2, -1, 0, 1, 2], [], "mean of the n values (true mean 0, SD 1)", "");
  let body = ax.s;
  const cols = [C.fn, C.blue];
  hs.forEach((c, j) => c.forEach((v, k) => { if (v) body += rect(ax.X(lo + k * bw) + 0.5, ax.Y(v), ax.X(lo + (k + 1) * bw) - ax.X(lo + k * bw) - 1, ax.Y(0) - ax.Y(v), { fill: cols[j], op: 0.5 }); }));
  body += rect(300, 40, 18, 18, { fill: C.fn, op: 0.6 }) + text(326, 56, `n = 4: SE ${ST.sd(sims[0]).toFixed(2)}`, { size: 18, anchor: "start" });
  body += rect(300, 70, 18, 18, { fill: C.blue, op: 0.6 }) + text(326, 86, `n = 16: SE ${ST.sd(sims[1]).toFixed(2)}`, { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "replicates/fig-semean.svg", svg(470, 320, body));
}

// ------------------------------------------------ fig-cis.svg: 20 repeated experiments, estimate ± 2 SE
{
  // first seed whose 20 intervals miss the truth exactly once (the typical 1 in 20)
  const se = Math.SQRT1_2, truth = 1, n = 20;
  let est: number[] = [];
  for (let seed = 31; ; seed++) { const r = rng(seed); est = Array.from({ length: n }, () => truth + se * r.norm()); if (est.filter(e => Math.abs(e - truth) > 2 * se).length === 1) break; }
  const miss = est.filter(e => Math.abs(e - truth) > 2 * se).length;
  console.log("cis: misses", miss, "of", n);
  const ax = axes(60, 15, 380, 300, [-1.5, 3.5], [0, n + 1], [-1, 0, 1, 2, 3], [], "effect (treated − control)", "");
  let body = ax.s + line(ax.X(0), ax.Y(0), ax.X(0), ax.Y(n + 1), "#999", 1.5) + line(ax.X(truth), ax.Y(0), ax.X(truth), ax.Y(n + 1), C.tp, 2.5, "6 4");
  est.forEach((e, i) => {
    const y = ax.Y(i + 1), bad = Math.abs(e - truth) > 2 * se, col = bad ? C.fp : C.blue;
    body += line(ax.X(e - 2 * se), y, ax.X(e + 2 * se), y, col, 3) + circle(ax.X(e), y, 4.5, { fill: col });
  });
  body += text(ax.X(truth) + 6, ax.Y(n + 1) + 2, "true effect", { size: 18, anchor: "start", fill: "#1b7a1b" });
  body += text(250, 395, `${n - miss} of ${n} intervals contain the truth`, { size: 18 });
  fs.writeFileSync(ROOT + "replicates/fig-cis.svg", svg(470, 410, body));
}

// ------------------------------------------------ fig-fpr.svg: false-positive rate of the cells-as-n test
{
  const ms = Array.from({ length: 121 }, (_, i) => 10 ** (3 * i / 120));
  const ax = axes(80, 20, 330, 230, [1, 1000], [0, 1], [1, 10, 100, 1000], [0, 0.5, 1], "cells per experiment m", "false-positive rate", v => String(v), { logx: true });
  let body = ax.s;
  [[0.1, C.blue], [0.02, C.purple]].forEach(([rho, col]) => { body += path(ms.map(m => [ax.X(m), ax.Y(ST.pseudoRepFPR(m, rho as number))] as [number, number]), { stroke: col as string, w: 3.5 }); });
  for (const m of [10, 100, 1000]) { const v = ST.pseudoRepFPR(m, 0.1); body += circle(ax.X(m), ax.Y(v), 6.5, { fill: C.fp }) + text(ax.X(m) - 8, ax.Y(v) - 10, v.toFixed(2), { size: 18, anchor: "end", fill: C.fp }); }
  console.log("fpr rho 0.1", [10, 100, 1000].map(m => ST.pseudoRepFPR(m, 0.1).toFixed(3)).join(" "));
  body += line(ax.X(1), ax.Y(0.05), ax.X(1000), ax.Y(0.05), "#999", 1.5, "5 4");
  body += text(ax.X(1000) - 4, ax.Y(0.05) - 6, "nominal 5 %", { size: 17, anchor: "end", fill: "#666" });
  body += line(95, 40, 125, 40, C.blue, 3.5) + text(132, 46, "ρ = 0.1", { size: 18, anchor: "start" });
  body += line(95, 68, 125, 68, C.purple, 3.5) + text(132, 74, "ρ = 0.02", { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "replicates/fig-fpr.svg", svg(440, 305, body));
}

// ------------------------------------------------ fig-superplot.svg: cells coloured by experiment, experiment means
{
  const st = { ...PR.DEFAULTS, E: 3, m: 100, effect: 0 };
  // first seed where cells-as-n is "highly significant" (p < 0.001) although there is no effect
  let seed = 1; while (PR.runOnce(st, seed).res.cells.p >= 0.001) seed++;
  const { data, res } = PR.runOnce(st, seed);
  console.log("superplot: cells p", res.cells.p.toExponential(2), "means p", res.means.p.toFixed(2), "estimate", res.means.estimate?.toFixed?.(2));
  const all = data.flatMap(d => Array.from(d.values)), lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all));
  const ax = axes(70, 20, 330, 290, [0, 2], [lo, hi], [], [lo, Math.round((lo + hi) / 2), hi], "", "cell measurement");
  let body = ax.s;
  const r = rng(5);
  data.forEach(d => {
    const col = OI[d.exp], xc = 0.5 + d.cond;
    d.values.forEach(v => (body += circle(ax.X(xc + 0.6 * (r.unif() - 0.5)), ax.Y(v), 2.6, { fill: col, op: 0.45 })));
  });
  data.forEach(d => { const m = ST.mean(d.values), xc = 0.5 + d.cond + 0.25 * ((d.exp % 3) - 1); body += circle(ax.X(xc), ax.Y(m), 11, { fill: OI[d.exp], stroke: "#222", w: 2 }); });
  body += text(ax.X(0.5), 345, "control", { size: 19 }) + text(ax.X(1.5), 345, "treated", { size: 19 });
  body += text(235, 380, `cells as n: p = ${res.cells.p < 1e-3 ? res.cells.p.toExponential(0).replace("e-", "·10⁻").replace(/⁻(\d+)/, (_, d) => "⁻" + d.split("").map((c: string) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+c]).join("")) : res.cells.p.toFixed(3)}`, { size: 18 });
  body += text(235, 405, `experiments as n: p = ${res.means.p.toFixed(2)}`, { size: 18 });
  fs.writeFileSync(ROOT + "replicates/fig-superplot.svg", svg(450, 420, body));
}

// ------------------------------------------------ fig-secells.svg: true and claimed SE against cells per experiment
{
  const st = { ...PR.DEFAULTS, E: 3 };
  const ms = Array.from({ length: 121 }, (_, i) => 10 ** (0.5 + 2.5 * i / 120));
  console.log("secells", [5, 10, 100, 1000].map(m => { const c = PR.seCurves(st, m); return `${m}: ${c.true.toFixed(2)}/${c.claimed.toFixed(2)}`; }).join(" "));
  const ax = axes(80, 20, 330, 230, [3, 1000], [0, 1.3], [3, 10, 100, 1000], [0, 0.5, 1], "cells per experiment", "SE of the effect", v => String(v), { logx: true });
  let body = ax.s;
  body += path(ms.map(m => [ax.X(m), ax.Y(PR.seCurves(st, m).true)] as [number, number]), { stroke: C.blue, w: 3.5 });
  body += path(ms.map(m => [ax.X(m), ax.Y(PR.seCurves(st, m).claimed)] as [number, number]), { stroke: C.fp, w: 3.5, dash: "8 5" });
  for (const m of [5, 10, 100, 1000]) { const c = PR.seCurves(st, m); body += circle(ax.X(m), ax.Y(c.true), 5.5, { fill: C.blue }) + circle(ax.X(m), ax.Y(c.claimed), 5.5, { fill: C.fp }); }
  body += line(200, 40, 235, 40, C.blue, 3.5) + text(242, 46, "true", { size: 18, anchor: "start" });
  body += line(200, 68, 235, 68, C.fp, 3.5, "8 5") + text(242, 74, "claimed by cells as n", { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "replicates/fig-secells.svg", svg(440, 305, body));
}

// ------------------------------------------------ fig-plates.svg: confounded vs randomised layout on a plate with a gradient
{
  let body = "";
  const draw = (x0: number, id: string, title: string) => {
    const p = FSm.PRESETS.find(q => q.id === id)!, set = { ...FSm.DEFAULTS, ...p.settings };
    const pl = FSm.makePlate(set), a = FSm.analyse(pl);
    let best = 0; a.p.forEach((v, i) => { if (v < a.p[best]) best = i; });
    const v = pl.values[best], vs = Array.from(v).sort((x, y) => x - y), lo = vs[Math.floor(0.02 * vs.length)], hi = vs[Math.floor(0.98 * vs.length)];
    console.log("plates", id, "raw", a.raw, "bonf", a.bonf, "BH", a.bh, "false bonf", a.falseBonf);
    const cs = 13;
    body += text(x0 + 12 * cs, 25, title, { weight: "bold", size: 20 });
    for (let y = 0; y < pl.rows; y++) for (let x = 0; x < pl.cols; x++) {
      const i = y * pl.cols + x, t = Math.min(1, Math.max(0, (v[i] - lo) / (hi - lo)));
      const g = Math.round(235 - 175 * t), col = `rgb(${g},${Math.round(g + 10 * (1 - t))},${Math.round(250 - 80 * t)})`;
      body += rect(x0 + x * cs, 40 + y * cs, cs - 1, cs - 1, { fill: col });
      if (pl.cond[i]) body += circle(x0 + x * cs + (cs - 1) / 2, 40 + y * cs + (cs - 1) / 2, 2.6, { fill: C.fp });
    }
    body += text(x0 + 12 * cs, 40 + 16 * cs + 28, `hits after Bonferroni: ${a.bonf} of 300`, { size: 20 });
    body += text(x0 + 12 * cs, 40 + 16 * cs + 52, `(none real)`, { size: 20 });
  };
  draw(15, "confounded", "treated on the left half");
  draw(345, "randomised", "randomised layout");
  body += circle(30, 362, 4, { fill: C.fp }) + text(42, 368, "treated well", { size: 19, anchor: "start" });
  body += rect(180, 355, 12, 12, { fill: "rgb(235,245,250)" }) + rect(193, 355, 12, 12, { fill: "rgb(148,155,210)" }) + rect(206, 355, 12, 12, { fill: "rgb(60,60,170)" }) + text(226, 368, "one feature: a left-to-right gradient", { size: 19, anchor: "start" });
  fs.writeFileSync(ROOT + "replicates/fig-plates.svg", svg(670, 380, body));
}
