// Slide figures for spot-filters/ (second image pass). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/spot-filters/figs.ts --bundle --platform=node --log-level=warning | node
// Writes spot-filters/fig-psfspot.svg, fig-zeromean.svg, fig-maxwindow.svg, fig-nms.svg; prints the numbers.
import { spotImage, spotFilter, estimateNoise, noiseGain, matchPoints, pixelFraction, kernel1d, Pt } from "../../../demos/src/lib/spotdetect";
import { localMaxima2D } from "../../../demos/src/lib/edges";
import { C, svg, text, line, circle, rect, path, cross, arrow, axes, greyImage, out } from "../pointdetection/svgkit";

// ---------------------------------------------------------------- fig-psfspot.svg
{
  const n = 9, c = 4, s = 1.5, tot = 2 * Math.PI * s * s;
  const spot = (A: number) => { const d = new Float64Array(n * n); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) d[y * n + x] = A * tot * pixelFraction(x, c, s) * pixelFraction(y, c, s); return d; };
  const bright = spot(100), dim = spot(40), px = 24;
  let b = "";
  b += greyImage(bright, n, 0, 0, n, n, 30, 40, px, 0, bright[c * n + c]) + text(30 + n * px / 2, 28, "bright spot");
  b += greyImage(dim, n, 0, 0, n, n, 300, 40, px, 0, bright[c * n + c]) + text(300 + n * px / 2, 28, "dim spot (same scale)");
  const ax = axes(80, 300, 440, 190, [-4.6, 4.6], [0, 100], [-4, -2, 0, 2, 4], [0, 50, 100], "pixel (across the spot centre)", "value", String, false, false, 20);
  b += ax.s;
  const bw = (ax.X(1) - ax.X(0)) * 0.36;
  for (let i = 0; i < n; i++) {
    const x = i - c;
    b += rect(ax.X(x) - bw, ax.Y(bright[c * n + i]), bw, ax.Y(0) - ax.Y(bright[c * n + i]), { fill: C.blue });
    b += rect(ax.X(x), ax.Y(dim[c * n + i]), bw, ax.Y(0) - ax.Y(dim[c * n + i]), { fill: C.orange });
  }
  // continuous profiles (peak = pixel value at the centre) and the ±σ mark
  for (const [arr, col] of [[bright, C.blue], [dim, C.orange]] as [Float64Array, string][]) {
    const pk = arr[c * n + c], pts: [number, number][] = [];
    for (let x = -4.5; x <= 4.5; x += 0.05) pts.push([ax.X(x), ax.Y(pk * Math.exp(-x * x / (2 * (s * s + 1 / 12))))]);
    b += path(pts, { stroke: col, w: 2, dash: "5 4" });
  }
  const yS = ax.Y(bright[c * n + c] * Math.exp(-0.5)) ;
  b += arrow(ax.X(0), yS, ax.X(-s), yS, C.fg, 2) + arrow(ax.X(0), yS, ax.X(s), yS, C.fg, 2) + text(ax.X(s) + 8, yS + 7, "±σ = 1.5 px", { anchor: "start", size: 20 });
  b += text(300, 562, "brightness differs, the width does not", { size: 20 });
  out("spot-filters/fig-psfspot.svg", svg(560, 572, b));
  console.log("psfspot: centre pixel", bright[c * n + c].toFixed(1), dim[c * n + c].toFixed(1));
}

// ---------------------------------------------------------------- fig-zeromean.svg
{
  // kernels (1D, peak scaled to 1)
  let b = "";
  const R = 8, kG = kernel1d("gauss", 1.5), kZ = kernel1d("zmgauss", 1.5), kL = kernel1d("log", 1.5);
  const ax = axes(70, 30, 450, 150, [-R, R], [-0.45, 1.05], [-6, -3, 0, 3, 6], [0, 1], "offset (px)", "weight", String, false, false, 20);
  b += ax.s + line(ax.X(-R), ax.Y(0), ax.X(R), ax.Y(0), "#999", 1.5);
  const draw = (k: Float64Array, col: string, dash: string) => {
    const h = k.length >> 1; let pk = 0; for (const v of k) pk = Math.max(pk, v);
    const pts: [number, number][] = []; for (let i = 0; i < k.length; i++) { const x = i - h; if (Math.abs(x) <= R) pts.push([ax.X(x), ax.Y(k[i] / pk)]); }
    b += path(pts, { stroke: col, w: 3, dash });
  };
  draw(kG, C.blue, ""); draw(kZ, C.tp, "8 4"); draw(kL, C.fp, "2 3");
  b += text(320, 20, "kernels (σ = 1.5)", { size: 20, anchor: "start" });
  b += line(362, 46, 390, 46, C.blue, 3) + text(396, 53, "Gaussian", { size: 20, anchor: "start", fill: C.blue });
  b += line(362, 74, 390, 74, C.tp, 3, "8 4") + text(396, 81, "zero-mean", { size: 20, anchor: "start", fill: C.tp });
  b += line(362, 102, 390, 102, C.fp, 3, "2 3") + text(396, 109, "LoG", { size: 20, anchor: "start", fill: C.fp });
  // detection on the demo's uneven image (spotdetect, seed 7): Gaussian vs LoG, k = 4, r = 2
  const N = 256, s = spotImage({ size: N, count: 80, sigma: 1.5, amplitude: [20, 100], background: 40, slope: 60, bump: 90, noise: 10, seed: 7 });
  const noise = estimateNoise(s.image), T: Pt[] = s.truth.map(t => [t.x, t.y]);
  const px = 1.05, y0 = 280;
  (["gauss", "log"] as const).forEach((f, k) => {
    const resp = spotFilter(s.image, f, 1.5), rn = noise * noiseGain(f, 1.5);
    const dets = localMaxima2D(resp, -Infinity, 2, 2).filter(p => p.value > 4 * rn);
    const m = matchPoints(T, dets.map(d => [d.x, d.y] as Pt), 2, "optimal");
    const md = new Set(m.pairs.map(p => p.d)), mt = new Set(m.pairs.map(p => p.t));
    const X0 = 10 + k * (N * px + 22);
    b += greyImage(s.image.data, N, 0, 0, N, N, X0, y0, px);
    dets.forEach((d, j) => { const x = X0 + (d.x + 0.5) * px, y = y0 + (d.y + 0.5) * px; b += md.has(j) ? circle(x, y, 3.2, { stroke: C.tp, w: 2 }) : cross(x, y, 3, C.fp, 2); });
    s.truth.forEach((t, i) => { if (!mt.has(i)) b += circle(X0 + (t.x + 0.5) * px, y0 + (t.y + 0.5) * px, 5, { stroke: C.fn, w: 2 }); });
    b += text(X0 + N * px / 2, y0 - 32, f === "gauss" ? "Gaussian filter" : "LoG (zero-mean)", { weight: "bold", size: 21 });
    b += text(X0 + N * px / 2, y0 - 10, `${m.tp} found, ${m.fp} false`, { size: 20, fill: m.fp > 10 ? C.fp : C.fg });
    console.log("zeromean", f, "kept", dets.length, "TP", m.tp, "FP", m.fp, "FN", m.fn, "F1", m.f1.toFixed(2));
  });
  b += circle(30, y0 + N * px + 20, 5, { stroke: C.tp, w: 2.5 }) + text(42, y0 + N * px + 26, "found", { size: 20, anchor: "start" });
  b += cross(120, y0 + N * px + 20, 5, C.fp, 2.5) + text(132, y0 + N * px + 26, "false", { size: 20, anchor: "start" });
  b += circle(200, y0 + N * px + 20, 6, { stroke: C.fnDark, w: 2.5 }) + text(212, y0 + N * px + 26, "missed (of 80; k = 4)", { size: 20, anchor: "start" });
  out("spot-filters/fig-zeromean.svg", svg(570, y0 + N * px + 40, b));
}

// ---------------------------------------------------------------- fig-maxwindow.svg (the deck's 5 × 5 example)
{
  const G = [[2, 3, 1, 0, 1], [4, 9, 5, 2, 0], [3, 6, 7, 6, 1], [1, 2, 4, 8, 3], [0, 1, 3, 2, 2]], cs = 44;
  let b = "";
  const grid = (X0: number, Y0: number, cand: [number, number][], wins: { r: number; c: number; rad: number; col: string; dash?: string }[]) => {
    let s = "";
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      const isC = cand.some(([rr, cc]) => rr === r && cc === c);
      s += rect(X0 + c * cs, Y0 + r * cs, cs, cs, { fill: isC ? "#cfe8df" : "#fff", stroke: "#999", w: 1 });
      s += text(X0 + c * cs + cs / 2, Y0 + r * cs + cs / 2 + 8, String(G[r][c]), { size: 22, weight: isC ? "bold" : undefined });
    }
    for (const w of wins) {
      const c0 = Math.max(0, w.c - w.rad), c1 = Math.min(4, w.c + w.rad), r0 = Math.max(0, w.r - w.rad), r1 = Math.min(4, w.r + w.rad);
      s += rect(X0 + c0 * cs + 2, Y0 + r0 * cs + 2, (c1 - c0 + 1) * cs - 4, (r1 - r0 + 1) * cs - 4, { stroke: w.col, w: 3.5, dash: w.dash });
    }
    return s;
  };
  b += text(10 + 2.5 * cs, 24, "r = 1 (3 × 3)", { weight: "bold" });
  b += grid(10, 40, [[1, 1], [3, 3]], [{ r: 1, c: 1, rad: 1, col: C.blue }, { r: 3, c: 3, rad: 1, col: C.orange, dash: "7 4" }]);
  b += text(10 + 2.5 * cs, 40 + 5 * cs + 28, "2 candidates: 9 and 8", { size: 18 });
  const X2 = 10 + 5 * cs + 40;
  b += text(X2 + 2.5 * cs, 24, "r = 2 (5 × 5)", { weight: "bold" });
  b += grid(X2, 40, [[1, 1]], [{ r: 3, c: 3, rad: 2, col: C.orange, dash: "7 4" }]);
  b += text(X2 + 2.5 * cs, 40 + 5 * cs + 28, "8 sees the 9: 1 candidate", { size: 18 });
  out("spot-filters/fig-maxwindow.svg", svg(X2 + 5 * cs + 10, 40 + 5 * cs + 40, b));
}

// ---------------------------------------------------------------- fig-nms.svg (one operation, four neighbourhoods)
{
  let b = "";
  const cs = 22, W = 260;
  const cell = (x: number, y: number, fill: string, stroke = "#999") => rect(x, y, cs, cs, { fill, stroke, w: 1 });
  const HI = "#9ed3c4", NB = "#f6d7a7";
  // 1. Canny: across the edge
  { const X0 = 50, Y0 = 18; for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) b += cell(X0 + c * cs, Y0 + r * cs, r === 2 && c === 2 ? HI : (r === 1 && c === 1) || (r === 3 && c === 3) ? NB : "#fff");
    b += line(X0 + 5 * cs, Y0, X0, Y0 + 5 * cs, "#555", 2, "5 4"); b += arrow(X0 + 1.5 * cs, Y0 + 1.5 * cs, X0 + 4.2 * cs, Y0 + 4.2 * cs, C.blue, 2.5);
    b += text(X0 + 2.5 * cs, Y0 + 5 * cs + 26, "Canny: 2 pixels", { size: 18 }) + text(X0 + 2.5 * cs, Y0 + 5 * cs + 48, "across the edge", { size: 18 }); }
  // 2. blobs: 3 × 3 × 3 in (x, y, σ)
  { const X0 = W + 40, Y0 = 30; for (let k = 2; k >= 0; k--) for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) b += cell(X0 + c * cs + k * 18, Y0 + r * cs - k * 10 + 12, k === 1 && r === 1 && c === 1 ? HI : NB, "#666");
    b += text(X0 + 60, Y0 + 5 * cs + 14, "blobs: 26 neighbours", { size: 18 }) + text(X0 + 60, Y0 + 5 * cs + 36, "in x, y and σ", { size: 18 }); }
  // 3. boxes in a detector
  { const X0 = 2 * W + 10, Y0 = 22; b += rect(X0 + 10, Y0 + 10, 80, 70, { stroke: C.tp, w: 3.5 }) + rect(X0 + 34, Y0 + 26, 80, 70, { stroke: C.fp, w: 3, dash: "6 4" });
    b += text(X0 + 30, Y0 + 2, "0.9", { size: 18, fill: C.tp }) + text(X0 + 132, Y0 + 98, "0.7", { size: 18, fill: C.fp });
    b += text(X0 + 62, Y0 + 5 * cs + 22, "detectors: drop", { size: 18 }) + text(X0 + 62, Y0 + 5 * cs + 44, "overlapping boxes", { size: 18 }); }
  // 4. spots: (2r+1)² window
  { const X0 = 3 * W - 10, Y0 = 18; for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) b += cell(X0 + c * cs, Y0 + r * cs, r === 2 && c === 2 ? HI : NB);
    b += text(X0 + 2.5 * cs, Y0 + 5 * cs + 26, "spots: window", { size: 18, weight: "bold" }) + text(X0 + 2.5 * cs, Y0 + 5 * cs + 48, "of radius r = 2", { size: 18, weight: "bold" }); }
  out("spot-filters/fig-nms.svg", svg(4 * W - 100, 196, b));
}
