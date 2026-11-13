// Slide figures for spot-crowd/ (second image pass). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/spot-crowd/figs.ts --bundle --platform=node --log-level=warning | node
// Writes spot-crowd/fig-density.svg, fig-split.svg, fig-gate.svg, fig-gatef1.svg; prints the numbers.
// The density and gate figures use the spotmatch demo's image and detector (seed 21, zero-mean
// Gaussian, maxima r = 2, k = 4); the split figure is the crowdfit demo's "d = 2σ" preset.
import { spotImage, spotFilter, estimateNoise, noiseGain, matchPoints, Pt } from "../../../demos/src/lib/spotdetect";
import { localMaxima2D } from "../../../demos/src/lib/edges";
import { Grey } from "../../../demos/src/lib/imageops";
import { analyse, W as CW, H as CH } from "../../../demos/src/crowdfit/model";
import { C, svg, text, line, circle, rect, path, cross, axes, greyImage, out } from "../pointdetection/svgkit";

const N = 256, PSF = 1.5;
/** The spotmatch detector without fitting: zero-mean Gaussian, maxima r = 2 (border 3), score in noise sd. */
function detect(img: Grey, k: number): Pt[] {
  const resp = spotFilter(img, "zmgauss", PSF), rn = estimateNoise(img) * noiseGain("zmgauss", PSF);
  return localMaxima2D(resp, k * rn, 2, 3).map(p => [p.x, p.y] as Pt);
}
const scene = (count: number) => spotImage({ size: N, count, sigma: PSF, amplitude: [8, 80], background: 40, noise: 10, seed: 21 });

// ---------------------------------------------------------------- fig-density.svg
{
  const counts = [80, 300, 1000], cs = 72, px = 3.4, cx = 60, cy = 100;
  let b = "";
  counts.forEach((n, k) => {
    const s = scene(n), T: Pt[] = s.truth.map(t => [t.x, t.y]), D = detect(s.image, 4), m = matchPoints(T, D, 2, "optimal");
    const md = new Set(m.pairs.map(p => p.d)), mt = new Set(m.pairs.map(p => p.t)), X0 = 10 + k * (cs * px + 20), Y0 = 60;
    b += greyImage(s.image.data, N, cx, cy, cs, cs, X0, Y0, px);
    const inC = (x: number, y: number) => x >= cx && x < cx + cs && y >= cy && y < cy + cs;
    s.truth.forEach((t, i) => { if (!mt.has(i) && inC(t.x, t.y)) b += circle(X0 + (t.x - cx + 0.5) * px, Y0 + (t.y - cy + 0.5) * px, 7, { stroke: C.fn, w: 2.5 }); });
    D.forEach((d, j) => { if (!inC(d[0], d[1])) return; const x = X0 + (d[0] - cx + 0.5) * px, y = Y0 + (d[1] - cy + 0.5) * px; b += md.has(j) ? circle(x, y, 4, { fill: C.tp, stroke: "#000", w: 1 }) : cross(x, y, 4.5, C.fp, 3); });
    b += text(X0 + cs * px / 2, 24, `${n} spots`, { weight: "bold", size: 22 });
    b += text(X0 + cs * px / 2, 48, `spacing ≈ ${(N / Math.sqrt(n)).toFixed(0)} px`, { size: 21 });
    b += text(X0 + cs * px / 2, Y0 + cs * px + 28, `recall ${m.recall.toFixed(2)}, F1 ${m.f1.toFixed(2)}`, { size: 21, weight: "bold" });
    console.log("density", n, "TP", m.tp, "FP", m.fp, "FN", m.fn, "recall", m.recall.toFixed(3), "F1", m.f1.toFixed(3));
  });
  const yL = 60 + cs * px + 62;
  b += circle(30, yL - 6, 5, { fill: C.tp, stroke: "#000", w: 1 }) + text(42, yL, "found", { anchor: "start", size: 21 });
  b += cross(140, yL - 6, 5, C.fp, 3) + text(152, yL, "false", { anchor: "start", size: 21 });
  b += circle(240, yL - 6, 7, { stroke: C.fnDark, w: 2.5 }) + text(254, yL, "missed   (crops; scores for the whole image)", { anchor: "start", size: 21 });
  out("spot-crowd/fig-density.svg", svg(3 * (cs * px + 20), yL + 14, b));
}

// ---------------------------------------------------------------- fig-split.svg (crowdfit "d = 2σ")
{
  const st = { sigma: 1.5, dr: 2, q: 1, noise: 6, c: 2, seed: 1 }, A = analyse(st, 1000 + st.seed), d = st.dr * st.sigma;
  const x0 = 4, x1 = 27, y0 = 3, y1 = 16, cw = x1 - x0, ch = y1 - y0, px = 9.5, gap = 24;
  const res1 = A.img.map((v, i) => v - A.f1.model[i]), res2 = A.img.map((v, i) => v - A.f2.model[i]);
  const panels: [string, Float64Array, number, number][] = [["data: 1 candidate", A.img, 20, 20 + 60], ["residual: 1 spot", res1, -25, 25], ["residual: 2 spots", res2, -25, 25]];
  let b = "";
  panels.forEach(([lab, img, lo, hi], k) => {
    const X0 = k === 0 ? 10 + (cw * px + gap) / 2 : 10 + (k - 1) * (cw * px + gap), Y0 = 34 + (k ? 1 : 0) * (ch * px + 50);
    b += text(X0 + cw * px / 2, Y0 - 10, lab, { size: 21, weight: "bold" });
    b += greyImage(img, CW, x0, y0, cw, ch, X0, Y0, px, lo, hi);
    const P = (x: number, y: number): [number, number] => [X0 + (x - x0 + 0.5) * px, Y0 + (y - y0 + 0.5) * px];
    if (k === 0) {
      for (const sx of [-d / 2, d / 2]) { const [x, y] = P(15 + sx, 9); b += circle(x, y, 4, { fill: C.fn, stroke: "#000", w: 1 }); }
      for (const c of A.cands) { const [x, y] = P(c.x, c.y); b += circle(x, y, 13, { stroke: C.sky, w: 3 }); }
    } else {
      const f = k === 1 ? A.f1 : A.f2;
      for (const sp of f.spots) { const [x, y] = P(sp.x, sp.y); b += cross(x, y, 7, C.tp, 3.5); }
    }
  });
  const yT = 34 + 2 * (ch * px + 50) + 4;
  b += text(10, yT, `gain RSS₁ − RSS₂ = ${A.sel.gain.toFixed(0)}`, { anchor: "start", size: 21 });
  b += text(10, yT + 26, `penalty 3·c·σn² = ${A.sel.penalty.toFixed(0)} (c = 2): split`, { anchor: "start", size: 21 });
  console.log("split: candidates", A.cands.length, "rss1", A.f1.rss.toFixed(0), "rss2", A.f2.rss.toFixed(0), "gain", A.sel.gain.toFixed(0), "penalty", A.sel.penalty.toFixed(0), "keepTwo", A.sel.keepTwo, "fit2", A.f2.spots.map(s => s.x.toFixed(2)).join(","));
  out("spot-crowd/fig-split.svg", svg(2 * cw * px + gap + 20, yT + 36, b));
}

// ---------------------------------------------------------------- fig-gate.svg (matching within a gate)
{
  const g = 2, T: Pt[] = [[1.5, 2], [7, 1.6], [12.2, 2.2]], D: Pt[] = [[2.2, 2.5], [9.4, 0.9], [11.6, 1.2], [13.3, 3]];
  const m = matchPoints(T, D, g, "optimal"), sc = 38, X = (x: number) => 20 + x * sc, Y = (y: number) => 20 + y * sc;
  let b = "";
  const mt = new Set(m.pairs.map(p => p.t)), md = new Set(m.pairs.map(p => p.d));
  T.forEach(t => b += circle(X(t[0]), Y(t[1]), g * sc, { stroke: "#999", w: 1.5, dash: "5 4", fill: "#f4f4f4" }));
  for (const p of m.pairs) b += line(X(T[p.t][0]), Y(T[p.t][1]), X(D[p.d][0]), Y(D[p.d][1]), C.tp, 4);
  T.forEach((t, i) => b += circle(X(t[0]), Y(t[1]), 11, { fill: "#fff", stroke: mt.has(i) ? C.tp : C.fnDark, w: 4 }));
  D.forEach((d, j) => b += md.has(j) ? circle(X(d[0]), Y(d[1]), 7, { fill: C.tp, stroke: "#000", w: 1.5 }) : cross(X(d[0]), Y(d[1]), 7, C.fp, 3.5));
  // labels
  b += text(X(1.5), Y(4.6), "TP", { weight: "bold", fill: C.tp, size: 22 });
  b += text(X(7), Y(4.6), "FN + FP", { weight: "bold", size: 22 }) + text(X(7), Y(5.3), "(outside the gate)", { size: 20 });
  b += text(X(12.4), Y(4.6), "TP + FP", { weight: "bold", size: 22 }) + text(X(12.4), Y(5.3), "(one-to-one)", { size: 20 });
  b += line(X(1.5), Y(2), X(1.5 + g * 0.7071), Y(2 - g * 0.7071), C.fg, 1.5) + text(X(1.5 + g * 0.71) + 4, Y(2 - g * 0.71) - 4, "r_g", { size: 20, anchor: "start", italic: true });
  const yL = Y(6.4);
  b += circle(30, yL - 6, 9, { fill: "#fff", stroke: C.tp, w: 3.5 }) + text(46, yL, "truth", { anchor: "start", size: 20 });
  b += circle(130, yL - 6, 6, { fill: C.tp, stroke: "#000", w: 1.5 }) + text(144, yL, "matched", { anchor: "start", size: 20 });
  b += cross(250, yL - 6, 6, C.fp, 3.5) + text(264, yL, "false", { anchor: "start", size: 20 });
  b += circle(340, yL - 6, 9, { fill: "#fff", stroke: C.fnDark, w: 3.5 }) + text(356, yL, `missed;  TP ${m.tp}, FP ${m.fp}, FN ${m.fn}`, { anchor: "start", size: 20 });
  console.log("gate example: TP", m.tp, "FP", m.fp, "FN", m.fn, "P", m.precision.toFixed(2), "R", m.recall.toFixed(2), "F1", m.f1.toFixed(2));
  out("spot-crowd/fig-gate.svg", svg(X(15.2), yL + 14, b));
}

// ---------------------------------------------------------------- fig-gatef1.svg (F1 and RMSE against the gate)
{
  const s = scene(80), T: Pt[] = s.truth.map(t => [t.x, t.y]), D = detect(s.image, 4);
  const gates = Array.from({ length: 40 }, (_, i) => 0.125 * (i + 1)), ms = gates.map(g => matchPoints(T, D, g, "optimal"));
  const ax = axes(80, 20, 430, 230, [0, 5], [0, 1.02], [0, 1, 2, 3, 4, 5], [0, 0.5, 1], "gate radius r_g (px)", "", v => String(v), false, false, 20);
  let b = ax.s;
  b += rect(ax.X(1), 20, ax.X(5) - ax.X(1), 230, { fill: "#ececec" }) + text(ax.X(3.4), ax.Y(0.3), "flat part", { size: 21 });
  b += path(gates.map((g, i) => [ax.X(g), ax.Y(ms[i].f1)] as [number, number]), { stroke: C.blue, w: 3.5 });
  b += path(gates.map((g, i) => [ax.X(g), ax.Y(Math.min(1.02, (ms[i].rmse || 0)))] as [number, number]), { stroke: C.orange, w: 3.5, dash: "8 5" });
  b += line(ax.X(2), 20, ax.X(2), 250, C.fg, 1.5, "3 3");
  b += text(ax.X(4.2), ax.Y(0.9) + 26, "F1", { size: 21, fill: C.blue, weight: "bold" });
  b += text(ax.X(4.2), ax.Y(ms[39].rmse) - 12, "RMSE (px)", { size: 21, fill: "#a86b00", weight: "bold" });
  const at = (g: number) => ms[gates.indexOf(g)];
  for (const g of [0.5, 1, 2, 3, 5]) { const q = at(g); console.log(`gatef1 g=${g} TP ${q.tp} FP ${q.fp} FN ${q.fn} F1 ${q.f1.toFixed(2)} RMSE ${q.rmse.toFixed(2)}`); }
  b += circle(ax.X(0.5), ax.Y(at(0.5).f1), 5, { fill: C.blue }) + line(ax.X(0.5), ax.Y(at(0.5).f1), ax.X(1.15), ax.Y(0.14), C.fg, 1.2) + text(ax.X(1.2), ax.Y(0.12), `gate 0.5 px: F1 ${at(0.5).f1.toFixed(2)}`, { size: 20, anchor: "start" });
  out("spot-crowd/fig-gatef1.svg", svg(540, 320, b));
}
