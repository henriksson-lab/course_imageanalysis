// Figures for marked-patterns/ (SVG). Run from the repo root (about a minute):
//   node_modules/.bin/esbuild tools/figs/marked-patterns/figs.ts --bundle --platform=node --log-level=warning | node
import { svg, text, line, rect, circle, out, C, T0, T1, patternPanel, curvePanel, greyImage } from "../point-patterns/kit";
import * as PP from "../../../demos/src/lib/pointpattern";
import { rng } from "../../../demos/src/lib/random";
import * as M from "../../../demos/src/celltypes/model";

const D = "marked-patterns/";
const tw = M.tissueWindow(), iw = M.imageWindow();
const tissue = (gen: M.Gen, o: Partial<M.TissueOptions> = {}) => M.makeTissue({ ...M.DEFAULT_TISSUE, gen, ...o });
const typeCol = (p: PP.Points) => (i: number) => (p.marks![i] === 1 ? T1 : T0);
const vessel = (X: (v: number) => number, Y: (v: number) => number, k: number) => circle(X(M.VESSEL[0]), Y(M.VESSEL[1]), M.VESSEL[2] * k, { stroke: "#c0392b", w: 2.5 });

/** Observed curve, 39 null curves, envelopes and the MAD test (as in the demo, seed 1). */
function test(p: PP.Points, win: PP.Window, c: M.Curve, nul: M.Null, bw = 40) {
  const grid = c === "inhom" ? M.rhoGrid(p, win, bw) : undefined;
  const obs = M.curveOf(p, win, c, bw), ref = M.referenceOf(p, c), r = rng(78), sims: Float64Array[] = [];
  for (let k = 0; k < 39; k++) sims.push(M.curveOf(M.simulateNull(r, p, win, nul, bw, grid), win, c, bw));
  const idx = PP.rangeIdx(M.RADII, [2, M.RMAX]), env = PP.pointwiseEnvelope(sims), mad = PP.madTest(obs, sims, env.mean, idx);
  return { obs, ref, env, mad, idx };
}
function envPlot(x0: number, y0: number, w: number, h: number, t: ReturnType<typeof test>, yd: [number, number], yt: number[], yl: string, title: string) {
  const xs = t.idx.map(k => M.RADII[k]);
  let s = curvePanel(x0, y0, w, h, [0, 40], yd, [0, 20, 40], yt, "r (µm)", yl,
    [{ xs: M.RADII, ys: t.ref, color: "#999", dash: "2 4", w: 1.5 }, { xs: M.RADII, ys: t.env.mean, color: "#000", dash: "7 5", w: 2 }, { xs: M.RADII, ys: t.obs, color: C.blue }],
    { bands: [{ xs, lo: t.idx.map(k => t.mad.lo[k]), hi: t.idx.map(k => t.mad.hi[k]), color: C.tp, opacity: 0.18 }, { xs, lo: t.idx.map(k => t.env.lo[k]), hi: t.idx.map(k => t.env.hi[k]), color: C.grey, opacity: 0.4 }] });
  s += text(x0 + w / 2, y0 - 8, title, { size: 18 }) + text(x0 + w / 2, y0 + h + 78, `MAD p = ${t.mad.p.toFixed(3)}`, { size: 18, weight: "bold" });
  return s;
}

// ---- a two-type tissue
{
  const p = tissue("attraction");
  let n1 = 0; for (const m of p.marks!) if (m === 1) n1++;
  let b = patternPanel(30, 30, 400, tw, p, { r: 4.2, colors: typeCol(p), extra: vessel });
  b += circle(470, 120, 8, { fill: T0 }) + text(486, 127, `tumour cells (${p.x.length - n1})`, { anchor: "start" });
  b += circle(470, 160, 8, { fill: T1 }) + text(486, 167, `immune cells (${n1})`, { anchor: "start" });
  b += circle(470, 200, 10, { stroke: "#c0392b", w: 2.5 }) + text(486, 207, "vessel (outside the tissue)", { anchor: "start" });
  b += text(470, 260, "grey: outside the tissue mask", { anchor: "start", fill: "#777" });
  b += text(470, 320, "synthetic tissue, 200 × 200 µm", { anchor: "start", fill: "#777" });
  out(D + "fig-tissue.svg", svg(780, 460, b));
  console.log("tissue: tumour", p.x.length - n1, "immune", n1, "tissue area", tw.area);
}

// ---- cross-L for attraction, independence, segregation
{
  const cases: [M.Gen, M.Null, string][] = [["attraction", "relocate", "immune near tumour"], ["independent", "relocate", "independent"], ["segregation", "permute", "segregated domains"]];
  let b = "";
  const res: string[] = [];
  cases.forEach(([g, nl, t], k) => {
    const p = tissue(g), x0 = 30 + k * 380, tt = test(p, tw, "cross", nl);
    b += patternPanel(x0 + 50, 30, 240, tw, p, { r: 2.8, colors: typeCol(p), extra: vessel, title: t });
    b += envPlot(x0 + 70, 320, 270, 200, tt, [-12, 12], [-10, 0, 10], "L₀₁ − r (µm)", nl === "relocate" ? "null: relocate immune cells" : "null: permute labels");
    res.push(`${g}: p ${tt.mad.p} L01-r@10 ${tt.obs[10].toFixed(2)}`);
  });
  out(D + "fig-cross-l.svg", svg(1150, 630, b));
  console.log("cross-L:", res.join(" | "));
}

// ---- the two nulls drawn: packed tissue, one relocation, one permutation
{
  const p = tissue("packed"), r = rng(5);
  const reloc = M.simulateNull(r, p, tw, "relocate", 40), perm = M.simulateNull(r, p, tw, "permute", 40);
  let b = "";
  [[p, "observed (packed, random types)"], [reloc, "relocate immune cells"], [perm, "permute labels"]].forEach(([q, t], k) => {
    b += patternPanel(30 + k * 360, 40, 320, tw, q as PP.Points, { r: 3.4, colors: typeCol(q as PP.Points), title: t as string });
  });
  b += text(550, 400, "relocated immune cells can sit on top of tumour cells; permuted labels keep every cell where it is", { size: 19 });
  out(D + "fig-two-nulls.svg", svg(1110, 420, b));
}

// ---- packed tissue: the two nulls disagree
{
  const p = tissue("packed"), a = test(p, tw, "cross", "relocate"), c = test(p, tw, "cross", "permute");
  let b = envPlot(90, 40, 400, 260, a, [-10, 6], [-10, -5, 0, 5], "L₀₁(r) − r  (µm)", "null: relocate immune cells");
  b += envPlot(660, 40, 400, 260, c, [-10, 6], [-10, -5, 0, 5], "L₀₁(r) − r  (µm)", "null: permute labels");
  out(D + "fig-packed.svg", svg(1100, 420, b));
  console.log("packed: relocate p", a.mad.p, "permute p", c.mad.p, "n", p.x.length);
}

// ---- mark connection function: segregation vs random labels
{
  const s = tissue("segregation"), rl = tissue("random"), ts = test(s, tw, "mcf", "permute"), tr = test(rl, tw, "mcf", "permute");
  let b = envPlot(90, 40, 400, 260, tr, [0, 0.8], [0, 0.2, 0.4, 0.6, 0.8], "p₀₁(r)", "random types");
  b += envPlot(660, 40, 400, 260, ts, [0, 0.8], [0, 0.2, 0.4, 0.6, 0.8], "p₀₁(r)", "segregated domains");
  out(D + "fig-mcf.svg", svg(1100, 420, b));
  console.log("mcf: ref", ts.ref[0].toFixed(3), "segregation at 5,10,20:", [5, 10, 20].map(k => ts.obs[k].toFixed(3)), "p", ts.mad.p, "random p", tr.mad.p);
}

// ---- density gradient: plain L vs inhomogeneous L
{
  const p = tissue("gradient"), a = test(p, tw, "L", "csr"), c = test(p, tw, "inhom", "inhomPoisson", 40);
  let b = patternPanel(30, 40, 300, tw, p, { r: 3, colors: () => C.grey, extra: vessel, title: "denser near the vessel" });
  b += envPlot(450, 40, 300, 240, a, [-4, 6], [-4, 0, 4], "L(r) − r  (µm)", "plain L, null CSR");
  b += envPlot(860, 40, 300, 240, c, [-4, 6], [-4, 0, 4], "L_inhom − r  (µm)", "inhomogeneous L (σ = 40 µm)");
  out(D + "fig-gradient.svg", svg(1200, 400, b));
  console.log("gradient: plain p", a.mad.p, "L-r@20", a.obs[20].toFixed(2), "inhom p", c.mad.p, "Linhom-r@20", c.obs[20].toFixed(2));
}

// ---- the window: image vs tissue mask
{
  const p = tissue("random"), a = test(p, iw, "L", "csr"), c = test(p, tw, "L", "csr");
  let b = patternPanel(30, 40, 300, tw, p, { r: 3, colors: () => C.grey, extra: (X, Y, k) => rect(X(0), Y(0), 200 * k, 200 * k, { stroke: C.fp, w: 3, dash: "8 5" }), title: "cells at random in the tissue" });
  b += envPlot(450, 40, 300, 240, a, [-4, 16], [0, 5, 10, 15], "L(r) − r  (µm)", `window: whole image (${iw.area} µm²)`);
  b += envPlot(860, 40, 300, 240, c, [-4, 16], [0, 5, 10, 15], "L(r) − r  (µm)", `window: tissue mask (${tw.area} µm²)`);
  out(D + "fig-window.svg", svg(1200, 400, b));
  console.log("window: image p", a.mad.p, "L-r@20", a.obs[20].toFixed(2), "mask p", c.mad.p, "L-r@20", c.obs[20].toFixed(2), "areas", iw.area, tw.area);
}

// ---- bandwidth dilemma: clustered cells, inhomogeneous L at three bandwidths
{
  const p = tissue("labelled");
  let b = "";
  const res: string[] = [];
  [5, 12, 40].forEach((bw, k) => {
    const g = M.rhoGrid(p, tw, bw), N = M.GRID_N, fin = Array.from(g).filter(Number.isFinite), hi = Math.max(...fin), x0 = 30 + k * 380;
    const u = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) u[i] = Number.isFinite(g[i]) ? Math.round(40 + 215 * g[i] / hi) : 0;
    b += greyImage(u, N, 0, 0, N, N, x0, 40, 5, 0, 255);
    for (let i = 0; i < p.x.length; i++) b += circle(x0 + p.x[i] * 1.25, 40 + p.y[i] * 1.25, 2.4, { fill: C.orange });
    const t = test(p, tw, "inhom", "inhomPoisson", bw);
    const verdict = t.mad.p > 0.05 ? "no interaction left" : t.obs[10] > 0 ? "clustered beyond the density" : "below: over-corrected";
    b += text(x0 + 125, 30, `ρ̂ with σ = ${bw} µm`) + text(x0 + 125, 330, `L_inhom(10) − 10 = ${t.obs[10].toFixed(1)} µm`, { size: 19 }) + text(x0 + 125, 356, `p = ${t.mad.p.toFixed(3)}: ${verdict}`, { size: 19, weight: "bold" });
    res.push(`bw ${bw}: p ${t.mad.p} L@10 ${t.obs[10].toFixed(2)}`);
  });
  out(D + "fig-bandwidth.svg", svg(1150, 380, b));
  console.log("bandwidth:", res.join(" | "), "n", p.x.length);
}
void line;
