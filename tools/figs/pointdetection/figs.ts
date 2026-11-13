// Static figures for the point-detection day (SVG), print fallbacks for the demos. Run from the
// repo root:
//   node_modules/.bin/esbuild tools/figs/pointdetection/figs.ts --bundle --platform=node --log-level=warning | node
// Outputs go into spot-filters/, spot-threshold/, spot-fit/, spot-crowd/. Numbers are printed.
import * as fs from "fs";
import { spotImage, spotFilter, filterKernel, filterResponseNoise, spotGain, noiseMaximaValues, pixelFraction, matchPoints, Pt, atrous1d } from "../../../demos/src/lib/spotdetect";
import { pixelGaussian, centroid, threePoint, gauss3Point, parabola3Point, fitSpots, crlbGaussian, thompsonVar } from "../../../demos/src/lib/spotfit";
import { localMaxima2D } from "../../../demos/src/lib/edges";
import { rng } from "../../../demos/src/lib/ui";

const C = { tp: "#2ca02c", fp: "#d62728", fn: "#e6b800", blue: "#1f77b4", orange: "#ff7f0e", purple: "#9467bd", grey: "#777", fg: "#222" };
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const f1 = (v: number) => v.toFixed(1);
function svg(w: number, h: number, body: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
}
const text = (x: number, y: number, s: string, o: { size?: number; anchor?: string; fill?: string; weight?: string } = {}) =>
  `<text x="${f1(x)}" y="${f1(y)}" font-size="${o.size ?? 20}" text-anchor="${o.anchor ?? "middle"}" fill="${o.fill ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}>${esc(s)}</text>`;
const line = (x1: number, y1: number, x2: number, y2: number, st = C.fg, w = 2, dash = "") => `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" stroke="${st}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const circle = (x: number, y: number, r: number, o: { fill?: string; stroke?: string; w?: number } = {}) => `<circle cx="${f1(x)}" cy="${f1(y)}" r="${f1(r)}" fill="${o.fill ?? "none"}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"/>`;
const path = (pts: [number, number][], o: { stroke?: string; w?: number; dash?: string } = {}) => `<path d="M${pts.map(p => f1(p[0]) + "," + f1(p[1])).join("L")}" fill="none" stroke="${o.stroke ?? C.fg}" stroke-width="${o.w ?? 2.5}" stroke-linejoin="round"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
function axes(x0: number, y0: number, w: number, h: number, xd: [number, number], yd: [number, number], xt: number[], yt: number[], xl: string, yl: string, fmt = (v: number) => String(v), logx = false, logy = false) {
  const tx = (v: number) => (logx ? Math.log10(v) : v), ty = (v: number) => (logy ? Math.log10(v) : v);
  const X = (v: number) => x0 + (tx(v) - tx(xd[0])) / (tx(xd[1]) - tx(xd[0])) * w, Y = (v: number) => y0 + h - (ty(v) - ty(yd[0])) / (ty(yd[1]) - ty(yd[0])) * h;
  let s = "";
  for (const t of xt) s += line(X(t), y0, X(t), y0 + h, "#e3e3e3", 1) + line(X(t), y0 + h, X(t), y0 + h + 6, C.fg, 1.5) + text(X(t), y0 + h + 25, fmt(t), { size: 18 });
  for (const t of yt) s += line(x0, Y(t), x0 + w, Y(t), "#e3e3e3", 1) + line(x0 - 6, Y(t), x0, Y(t), C.fg, 1.5) + text(x0 - 10, Y(t) + 6, fmt(t), { size: 18, anchor: "end" });
  s += path([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], { w: 1.5 });
  if (xl) s += text(x0 + w / 2, y0 + h + 50, xl);
  if (yl) s += `<text x="${x0 - 52}" y="${y0 + h / 2}" font-size="20" text-anchor="middle" transform="rotate(-90 ${x0 - 52} ${y0 + h / 2})">${esc(yl)}</text>`;
  return { X, Y, s };
}
/** Grey image as SVG rects (small images only). */
function greyImage(data: ArrayLike<number>, w: number, h: number, x0: number, y0: number, px: number, lo?: number, hi?: number): string {
  let a = Infinity, b = -Infinity; for (let i = 0; i < w * h; i++) { a = Math.min(a, data[i]); b = Math.max(b, data[i]); }
  lo = lo ?? a; hi = hi ?? b;
  let s = "";
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = Math.round(255 * Math.max(0, Math.min(1, (data[y * w + x] - lo) / (hi - lo || 1))));
    s += `<rect x="${f1(x0 + x * px)}" y="${f1(y0 + y * px)}" width="${f1(px + 0.3)}" height="${f1(px + 0.3)}" fill="rgb(${v},${v},${v})"/>`;
  }
  return s;
}
const ROOT = "./";
const out = (p: string, s: string) => { fs.writeFileSync(ROOT + p, s); console.log("wrote", p); };

// ---------------------------------------------------------------- spot-filters/fig-pipeline.svg
{
  const n = 64, s = spotImage({ size: n, count: 9, sigma: 1.5, amplitude: [25, 70], noise: 10, background: 40, slope: 30, margin: 6, minDistance: 8, seed: 5 });
  const r = spotFilter(s.image, "log", 1.5);
  const m = localMaxima2D(r, 0, 2, 2), rn = 10 * filterResponseNoise(filterKernel("log", 1.5).data), kept = m.filter(p => p.value > 4 * rn);
  const px = 3.6, W = n * px; let b = "";
  const xs = [20, 20 + W + 50, 20 + 2 * (W + 50), 20 + 3 * (W + 50)];
  b += greyImage(s.image.data, n, n, xs[0], 40, px) + text(xs[0] + W / 2, 28, "1. image");
  b += greyImage(r.data, n, n, xs[1], 40, px) + text(xs[1] + W / 2, 28, "2. filter (LoG)");
  b += greyImage(r.data, n, n, xs[2], 40, px);
  for (const p of m) b += circle(xs[2] + (p.x + 0.5) * px, 40 + (p.y + 0.5) * px, 4, { stroke: p.value > 4 * rn ? C.tp : C.fp, w: 2 });
  b += text(xs[2] + W / 2, 28, "3. local maxima");
  b += greyImage(s.image.data, n, n, xs[3], 40, px);
  for (const p of kept) b += circle(xs[3] + (p.x + 0.5) * px, 40 + (p.y + 0.5) * px, 7, { stroke: C.tp, w: 2.5 });
  b += text(xs[3] + W / 2, 28, "4. threshold");
  for (let i = 0; i < 3; i++) b += `<path d="M${xs[i] + W + 10},${40 + W / 2}h28l-8,-6m8,6l-8,6" stroke="${C.fg}" stroke-width="2.5" fill="none"/>`;
  b += text(xs[1] + W + 25, 40 + W + 32, `${m.length} candidates (red: below 4 noise sd), ${kept.length} spots kept; truth: ${s.truth.length} spots`, { size: 18 });
  out("spot-filters/fig-pipeline.svg", svg(xs[3] + W + 20, 40 + W + 46, b));
  console.log("pipeline: candidates", m.length, "kept", kept.length, "truth", s.truth.length);
}

// ---------------------------------------------------------------- spot-filters/fig-matched.svg
{
  const L = 80, sp = 1.5, A = 20, sn = 10, r = rng(3), sig = Array.from({ length: L }, (_, i) => A * Math.sqrt(2 * Math.PI) * sp * pixelFraction(i, 40, sp) + sn * r.norm());
  let b = "";
  const rows: [string, number][] = [["raw (single pixel)", 0], ["Gaussian σ = 0.5", 0.5], ["σ = 1.5 (matched)", 1.5], ["σ = 4", 4]];
  rows.forEach(([lab, s], k) => {
    const y0 = 20 + k * 120, ax = axes(70, y0, 520, 90, [0, L - 1], [-30, 50], [], [0], "", "");
    let f: number[];
    if (s === 0) f = sig;
    else { const R = Math.ceil(4 * s), g = Array.from({ length: 2 * R + 1 }, (_, i) => Math.exp(-((i - R) ** 2) / (2 * s * s))), z = g.reduce((a, v) => a + v, 0); f = sig.map((_, i) => { let t = 0; for (let j = -R; j <= R; j++) t += g[j + R] / z * sig[Math.min(L - 1, Math.max(0, i + j))]; return t; }); }
    // 2D SNR (the matched-filter number quoted in the deck): A·Σw p / (σn √Σw²)
    const snr = s === 0 ? A / sn : A * spotGain(filterKernel("gauss", s), sp) / (sn * filterResponseNoise(filterKernel("gauss", s).data));
    b += ax.s + path(f.map((v, i) => [ax.X(i), ax.Y(Math.max(-30, Math.min(50, v)))] as [number, number]), { stroke: k === 2 ? C.tp : C.blue, w: 2.5 });
    b += line(ax.X(40), y0, ax.X(40), y0 + 90, C.orange, 1.5, "4 4");
    b += text(600, y0 + 40, lab, { anchor: "start", size: 19 }) + text(600, y0 + 66, `2D spot SNR ${snr.toFixed(1)}`, { anchor: "start", size: 19, fill: k === 2 ? C.tp : C.fg });
    console.log("matched fig", lab, snr.toFixed(2));
  });
  out("spot-filters/fig-matched.svg", svg(820, 500, b));
}

// ---------------------------------------------------------------- spot-filters/fig-atrous.svg
{
  const L = 64, r = rng(8), base = Array.from({ length: L }, (_, i) => 30 + 0.4 * i + 40 * Math.exp(-((i - 20) ** 2) / 4.5) + 25 * Math.exp(-((i - 44) ** 2) / 4.5));
  const sig = base.map(v => v + 4 * r.norm()), D = atrous1d(sig, 4);
  const rows: [string, number[]][] = [["input", sig], ["w1 (noise)", Array.from(D.w[0])], ["w2", Array.from(D.w[1])], ["w3", Array.from(D.w[2])], ["a4 (background)", Array.from(D.a[3])]];
  let b = "";
  rows.forEach(([lab, f], k) => {
    const lo = Math.min(...f), hi = Math.max(...f), y0 = 14 + k * 78, X = (i: number) => 60 + i * 8, Y = (v: number) => y0 + 60 - (v - lo) / (hi - lo || 1) * 56;
    if (lo < 0 && hi > 0) b += line(60, Y(0), 60 + 63 * 8, Y(0), "#bbb", 1);
    b += path(f.map((v, i) => [X(i), Y(v)] as [number, number]), { stroke: k === 0 ? C.fg : k === 4 ? C.grey : C.blue, w: 2.5 });
    b += text(590, y0 + 38, lab, { anchor: "start", size: 19 });
  });
  b += line(60 + 20 * 8, 10, 60 + 20 * 8, 400, C.orange, 1.5, "4 4") + line(60 + 44 * 8, 10, 60 + 44 * 8, 400, C.orange, 1.5, "4 4");
  out("spot-filters/fig-atrous.svg", svg(780, 410, b));
}

// ---------------------------------------------------------------- spot-threshold/fig-maxima.svg
{
  const vals: number[] = [];
  for (let s = 1; s <= 4; s++) vals.push(...noiseMaximaValues(256, 1.5, 500 + s, 1).values);
  const lo = -1, hi = 6, bw = 0.25, nb = (hi - lo) / bw, cnt = new Array(nb).fill(0);
  for (const v of vals) { const k = Math.floor((v - lo) / bw); if (k >= 0 && k < nb) cnt[k]++; }
  const pdfS = vals.length * bw / Math.sqrt(2 * Math.PI), ymax = Math.max(...cnt, pdfS) * 1.1;
  const ax = axes(80, 20, 600, 260, [lo, hi], [0, ymax], [-1, 0, 1, 2, 3, 4, 5, 6], [], "value (noise sd of the filtered image)", "count");
  let b = ax.s;
  cnt.forEach((c, k) => { const x0 = ax.X(lo + k * bw), x1 = ax.X(lo + (k + 1) * bw); b += `<rect x="${f1(x0)}" y="${f1(ax.Y(c))}" width="${f1(x1 - x0 - 1)}" height="${f1(ax.Y(0) - ax.Y(c))}" fill="${C.blue}" fill-opacity="0.55"/>`; });
  const pts: [number, number][] = []; for (let x = lo; x <= hi; x += 0.05) pts.push([ax.X(x), ax.Y(pdfS * Math.exp(-x * x / 2))]);
  b += path(pts, { stroke: C.fg, w: 2.5, dash: "7 5" });
  b += line(ax.X(3), 20, ax.X(3), 280, C.fp, 2.5, "6 4") + text(ax.X(3) + 8, 44, "k = 3", { anchor: "start", fill: C.fp });
  b += text(ax.X(4.4), 110, "local maxima", { fill: C.blue, anchor: "start", size: 19 }) + text(ax.X(-0.9), 60, "ordinary pixels", { anchor: "start", size: 19 });
  const above = vals.filter(v => v > 3).length / 4;
  console.log("maxima per 256² image", vals.length / 4, "above 3:", above, "mean maximum", vals.reduce((a, v) => a + v, 0) / vals.length);
  out("spot-threshold/fig-maxima.svg", svg(720, 360, b));
}

// ---------------------------------------------------------------- spot-fit/fig-bias.svg
{
  const G = 13, Cc = 6, s = 1, Nn = 1000, b0 = 20, xs = Array.from({ length: 41 }, (_, i) => -0.5 + i / 40);
  const est = (img: Float64Array) => {
    const cen = centroid(img, G, G, Cc, Cc, 3, 0)[0] - Cc, par = threePoint(img, G, G, Cc, Cc, parabola3Point)[0] - Cc, lg = threePoint(img, G, G, Cc, Cc, gauss3Point, b0)[0] - Cc;
    const f = fitSpots(img, G, G, s, [{ x: Cc, y: Cc, N: Nn }], b0); return [cen, par, lg, f.spots[0].x - Cc];
  };
  const ys = xs.map(t => est(pixelGaussian(Cc + t, Cc, s, Nn, b0, G)).map(v => v - t));
  const ax = axes(90, 20, 560, 300, [-0.5, 0.5], [-0.35, 0.35], [-0.5, -0.25, 0, 0.25, 0.5], [-0.3, -0.15, 0, 0.15, 0.3], "true offset from the pixel centre (px)", "estimate − truth (px)");
  let b = ax.s;
  const cols = [C.blue, C.orange, C.purple, C.tp], labs = ["centroid (7 × 7, background 20)", "parabola", "log-parabola (b subtracted)", "least-squares fit"];
  cols.forEach((c, k) => { b += path(xs.map((x, i) => [ax.X(x), ax.Y(ys[i][k])] as [number, number]), { stroke: c, w: 3, dash: k === 3 ? "8 5" : "" }); b += text(680, 60 + k * 30, labs[k], { anchor: "start", size: 18, fill: c }); });
  console.log("bias at 0.3:", ys[32].map(v => v.toFixed(3)).join(" "));
  out("spot-fit/fig-bias.svg", svg(990, 390, b));
}

// ---------------------------------------------------------------- spot-fit/fig-precision.svg
{
  const Ns = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000], s = 100, a = 100;
  const ax = axes(90, 20, 560, 300, [30, 10000], [0.8, 100], [30, 100, 1000, 10000], [1, 3, 10, 30, 100], "detected photons N", "localisation precision (nm)", v => String(v), true, true);
  let b = ax.s;
  const curve = (f: (N: number) => number) => Ns.map(N => [ax.X(N), ax.Y(Math.sqrt(f(N)))] as [number, number]);
  b += path(curve(N => crlbGaussian(s, N)), { stroke: C.fg, w: 3 });
  b += path(curve(N => thompsonVar(s, a, 0, N)), { stroke: C.blue, w: 2.5, dash: "7 4" });
  b += path(curve(N => thompsonVar(s, a, 10, N)), { stroke: C.fp, w: 3 });
  b += text(680, 70, "s/√N (bound, ideal camera)", { anchor: "start", size: 18 });
  b += text(680, 100, "100 nm pixels", { anchor: "start", size: 18, fill: C.blue });
  b += text(680, 130, "+ background 10 photons/pixel", { anchor: "start", size: 18, fill: C.fp });
  b += text(680, 175, "s = 100 nm", { anchor: "start", size: 18 });
  console.log("precision: N=100 b=10", Math.sqrt(thompsonVar(s, a, 10, 100)).toFixed(1), "N=1000 b=10", Math.sqrt(thompsonVar(s, a, 10, 1000)).toFixed(2));
  out("spot-fit/fig-precision.svg", svg(990, 390, b));
}

// ---------------------------------------------------------------- spot-crowd/fig-merge.svg
{
  const s = 1.5; let b = "";
  [1.5, 2, 2.5, 3].forEach((rr, k) => {
    const d = rr * s, x0 = 20 + k * 245, X = (x: number) => x0 + 110 + x * 18, Y = (v: number) => 210 - v * 75;
    const g = (x: number, ss: number) => Math.exp(-x * x / (2 * ss * ss));
    const raw: [number, number][] = [], flt: [number, number][] = [], se = Math.SQRT2 * s;
    for (let x = -6; x <= 6; x += 0.05) { raw.push([X(x), Y(g(x - d / 2, s) + g(x + d / 2, s))]); flt.push([X(x), Y((g(x - d / 2, se) + g(x + d / 2, se)) * 0.5 * 1.0)]); }
    b += line(X(-6), Y(0), X(6), Y(0), "#999", 1.5) + path(raw, { stroke: C.fg, w: 3 }) + path(flt, { stroke: C.blue, w: 3, dash: "7 4" });
    b += circle(X(-d / 2), Y(0), 5, { fill: C.orange }) + circle(X(d / 2), Y(0), 5, { fill: C.orange });
    b += text(X(0), 30, `d = ${rr}σ`, { weight: "bold" });
  });
  b += text(500, 252, "black: the two spots (σ)   ·   blue dashed: after the matched filter (σ√2), scaled", { size: 18 });
  out("spot-crowd/fig-merge.svg", svg(1000, 265, b));
}

// ---------------------------------------------------------------- spot-crowd/fig-matching.svg
{
  const T: Pt[] = [[0, 0], [2.5, 0]], D: Pt[] = [[1, 0], [-2, 0]];
  let b = "";
  ["greedy", "optimal"].forEach((meth, k) => {
    const m = matchPoints(T, D, 2.5, meth as "greedy" | "optimal"), X = (x: number) => 40 + k * 470 + (x + 4.5) * 45, Y = 150;
    T.forEach(t => b += circle(X(t[0]), Y, 2.5 * 45, { stroke: "#bbb", w: 1.5 }));
    for (const p of m.pairs) b += line(X(T[p.t][0]), Y, X(D[p.d][0]), Y, C.tp, 5);
    const mt = new Set(m.pairs.map(p => p.t)), md = new Set(m.pairs.map(p => p.d));
    T.forEach((t, i) => { b += circle(X(t[0]), Y, 14, { fill: "#fff", stroke: mt.has(i) ? C.tp : C.fn, w: 4 }) + text(X(t[0]), Y - 24, `T${i + 1}`); });
    D.forEach((d, i) => { b += circle(X(d[0]), Y, 9, { fill: md.has(i) ? C.tp : C.fp, stroke: "#000", w: 1.5 }) + text(X(d[0]), Y + 38, `D${i + 1}`); });
    b += text(X(0.5), 305, `${meth}: TP ${m.tp}, FP ${m.fp}, FN ${m.fn}, F1 ${m.f1.toFixed(2)}`, { weight: "bold" });
  });
  out("spot-crowd/fig-matching.svg", svg(940, 320, b));
}
