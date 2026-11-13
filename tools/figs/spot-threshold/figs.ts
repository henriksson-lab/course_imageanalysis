// Slide figures for spot-threshold/ (second image pass). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/spot-threshold/figs.ts --bundle --platform=node --log-level=warning | node
// Writes spot-threshold/fig-rawvsk.svg, fig-mad.svg, fig-noisefilter.svg, fig-detlimit.svg, fig-poisson.svg.
import { spotImage, spotFilter, noiseGain, matchPoints, whiteNoise, Pt } from "../../../demos/src/lib/spotdetect";
import { localMaxima2D } from "../../../demos/src/lib/edges";
import { gaussianBlur } from "../../../demos/src/lib/imageops";
import { poisson } from "../../../demos/src/lib/random";
import { rng } from "../../../demos/src/lib/ui";
import { C, svg, text, line, circle, rect, path, cross, axes, greyImage, out, Phi } from "../pointdetection/svgkit";

const G15 = noiseGain("gauss", 1.5);   // 0.188
console.log("noise gain gauss 1.5:", G15.toFixed(4));

// ---------------------------------------------------------------- fig-rawvsk.svg
{
  // one scene, two exposures: the second has 4× the signal and background (Gaussian noise 2× larger)
  const n = 112, base = { size: n, count: 22, sigma: 1.5, minDistance: 8, margin: 6, seed: 11 };
  const ex = [1, 4].map(g => spotImage({ ...base, amplitude: [15 * g, 60 * g], background: 40 * g, noise: 10 * Math.sqrt(g) }));
  const tRaw = 4 * 10 * G15;   // 7.5: k = 4 for the first exposure
  const px = 2, gap = 16, X0 = 44, Y0 = 62, W = n * px;
  let b = "";
  b += text(X0 + W / 2, 24, `raw threshold ${tRaw.toFixed(1)}`, { weight: "bold" }) + text(X0 + W / 2, 48, "(grey levels)", { size: 20 });
  b += text(X0 + W + gap + W / 2, 24, "k = 4 noise sd", { weight: "bold" }) + text(X0 + W + gap + W / 2, 48, "(of the filtered image)", { size: 20 });
  ex.forEach((s, row) => {
    const resp = spotFilter(s.image, "gauss", 1.5), sn = 10 * Math.sqrt(row ? 4 : 1), rn = sn * G15;
    const cand = localMaxima2D(resp, -Infinity, 2, 3), T: Pt[] = s.truth.map(t => [t.x, t.y]);
    const y = Y0 + row * (W + 46);
    b += `<text x="26" y="${y + W / 2}" font-size="20" text-anchor="middle" transform="rotate(-90 26 ${y + W / 2})"><tspan font-weight="bold">${row ? "exposure 4×" : "exposure 1×"}</tspan>, noise sd ${sn.toFixed(0)}</text>`;
    [tRaw, 4 * rn].forEach((t, col) => {
      const dets = cand.filter(p => p.value > t), m = matchPoints(T, dets.map(d => [d.x, d.y] as Pt), 2, "optimal"), md = new Set(m.pairs.map(p => p.d));
      const X = X0 + col * (W + gap);
      b += greyImage(s.image.data, n, 0, 0, n, n, X, y, px);
      dets.forEach((d, j) => { const cx = X + (d.x + 0.5) * px, cy = y + (d.y + 0.5) * px; b += md.has(j) ? circle(cx, cy, 6, { stroke: C.tp, w: 2.5 }) : cross(cx, cy, 4.5, C.fp, 2.5); });
      b += text(X + W / 2, y + W + 26, `${m.tp} of ${s.truth.length} found, ${m.fp} false`, { size: 20, fill: m.fp > 3 ? C.fp : C.fg });
      console.log("rawvsk exposure", row ? 4 : 1, col ? "k=4" : "raw", "t", t.toFixed(2), "TP", m.tp, "FP", m.fp, "FN", m.fn);
    });
  });
  out("spot-threshold/fig-rawvsk.svg", svg(X0 + 2 * W + gap + 10, Y0 + 2 * (W + 46) - 6, b));
}

// ---------------------------------------------------------------- fig-mad.svg (the deck's nine values)
{
  const v = [3, -1, 0, 2, -2, 1, 25, -1, 0], n = v.length;
  const sorted = [...v].sort((a, b) => a - b), med = sorted[4], dev = v.map(x => Math.abs(x - med)).sort((a, b) => a - b), mad = dev[4], sig = mad / 0.6745;
  const mean = v.reduce((a, x) => a + x, 0) / n, sd = Math.sqrt(v.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1));
  console.log("mad: median", med, "MAD", mad, "sigma", sig.toFixed(2), "mean", mean, "sd", sd.toFixed(2));
  const ax = axes(30, 20, 520, 190, [-8, 27], [0, 1], [-5, 0, 5, 10, 15, 20, 25], [], "value", "");
  let b = ax.s.replace(/<path d="M30.0,20.0L30.0,210.0/, '<path d="M30.0,210.0');
  const yA = ax.Y(0.72), yB = ax.Y(0.28);
  // robust: median ± σ̂
  b += rect(ax.X(med - sig), yA - 22, ax.X(med + sig) - ax.X(med - sig), 44, { fill: C.tp, opacity: 0.3 }) + line(ax.X(med), yA - 26, ax.X(med), yA + 26, C.tp, 3);
  b += text(ax.X(-7.5), yA - 32, `median ± MAD/0.6745 = 0 ± ${sig.toFixed(2)}`, { anchor: "start", size: 20, fill: "#00664a" });
  // ordinary: mean ± sd
  b += rect(ax.X(mean - sd), yB - 22, ax.X(mean + sd) - ax.X(mean - sd), 44, { fill: C.fp, opacity: 0.22 }) + line(ax.X(mean), yB - 26, ax.X(mean), yB + 26, C.fp, 3);
  b += text(ax.X(-7.5), yB - 32, `mean ± sd = ${mean} ± ${sd.toFixed(1)}`, { anchor: "start", size: 20, fill: "#a04000" });
  const count = new Map<number, number>();
  for (const x of v) { const k = count.get(x) ?? 0; count.set(x, k + 1); for (const yy of [yA, yB]) b += circle(ax.X(x), yy - 6 + k * 12, 6, { fill: C.fg }); }
  b += text(ax.X(25), yA + 34, "the spot", { size: 20 });
  out("spot-threshold/fig-mad.svg", svg(570, 285, b));
}

// ---------------------------------------------------------------- fig-noisefilter.svg
{
  const N = 256, w = whiteNoise(N, N, 41);
  for (let i = 0; i < w.data.length; i++) w.data[i] *= 10;
  const g = gaussianBlur(w, 1.5, "wrap");
  const sdOf = (d: ArrayLike<number>) => { let s = 0, q = 0; for (let i = 0; i < d.length; i++) { s += d[i]; q += d[i] * d[i]; } const m = s / d.length; return Math.sqrt(q / d.length - m * m); };
  const s0 = sdOf(w.data), s1 = sdOf(g.data);
  console.log("noisefilter: sd raw", s0.toFixed(2), "filtered", s1.toFixed(2), "predicted", (10 * G15).toFixed(2));
  const c = 48, px = 3.5;
  let b = "";
  b += greyImage(w.data, N, 0, 0, c, c, 20, 40, px, -30, 30) + text(20 + c * px / 2, 28, `noise, sd ${s0.toFixed(1)}`);
  b += greyImage(g.data, N, 0, 0, c, c, 210, 40, px, -30, 30) + text(210 + c * px / 2, 28, `filtered σ = 1.5`);
  b += text(210 + c * px / 2, 40 + c * px + 26, "same grey scale", { size: 20 });
  b += greyImage(g.data, N, 0, 0, c, c, 400, 40, px) + text(400 + c * px / 2, 28, `stretched`);
  b += text(400 + c * px / 2, 40 + c * px + 26, `sd ${s1.toFixed(2)}`, { size: 20 }) + text(400 + c * px / 2, 40 + c * px + 50, `(predicted ${(10 * G15).toFixed(2)})`, { size: 20 });
  out("spot-threshold/fig-noisefilter.svg", svg(590, 40 + c * px + 62, b));
}

// ---------------------------------------------------------------- fig-detlimit.svg
{
  // a 16 × 16 grid of equal spots (spacing 16 px, random sub-pixel offsets), noise 10, Gaussian filter σ = 1.5,
  // maxima r = 2, threshold k noise sd (noise known), detected = matched within 2 px
  const As = Array.from({ length: 29 }, (_, i) => 2 + i), ks = [3, 4, 5], seeds = [1, 2, 3, 4];
  const frac = ks.map(() => As.map(() => 0));
  for (const [ai, A] of As.entries()) for (const sd of seeds) {
    const r = rng(1000 * sd + A), list = [] as { x: number; y: number; amplitude: number }[];
    for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) list.push({ x: 8 + 16 * i + r.unif() - 0.5, y: 8 + 16 * j + r.unif() - 0.5, amplitude: A });
    const s = spotImage({ size: 256, list, sigma: 1.5, background: 40, noise: 10, seed: 50 * sd + A });
    const resp = spotFilter(s.image, "gauss", 1.5), cand = localMaxima2D(resp, -Infinity, 2, 2), T: Pt[] = list.map(t => [t.x, t.y]);
    ks.forEach((k, ki) => { const m = matchPoints(T, cand.filter(p => p.value > k * 10 * G15).map(p => [p.x, p.y] as Pt), 2, "optimal"); frac[ki][ai] += m.tp / list.length / seeds.length; });
  }
  const ax = axes(85, 40, 450, 250, [2, 30], [0, 1], [5, 10, 15, 20, 25, 30], [0, 0.5, 1], "spot amplitude A (noise sd 10)", "fraction found", v => String(v), false, false, 20);
  let b = ax.s;
  const cols = [C.sky, C.blue, C.purple];
  ks.forEach((k, ki) => {
    const pts: [number, number][] = []; for (let A = 2; A <= 30; A += 0.25) pts.push([ax.X(A), ax.Y(Phi(A * Math.sqrt(Math.PI * 2.25) / 10 - k))]);
    b += path(pts, { stroke: cols[ki], w: 2, dash: "6 4" });
    b += path(As.map((A, i) => [ax.X(A), ax.Y(frac[ki][i])] as [number, number]), { stroke: cols[ki], w: 3.5 });
    const i50 = As.findIndex((_, i) => frac[ki][i] >= 0.5);
    console.log("detlimit k", k, "A at 50% (measured)", As[i50], "frac at A=15:", frac[ki][13].toFixed(2), "model A_min", (k * 10 / Math.sqrt(Math.PI * 2.25)).toFixed(1));
  });
  b += line(ax.X(15), 40, ax.X(15), 290, C.fg, 1.5, "3 3") + text(ax.X(15), 30, "A_min = 15 (k = 4)", { size: 20 });
  b += text(ax.X(9.2), ax.Y(0.88), "k = 3", { size: 20, fill: cols[0], weight: "bold" }) + text(ax.X(19.5), ax.Y(0.88), "4", { size: 20, fill: cols[1], weight: "bold" }) + text(ax.X(23.4), ax.Y(0.62), "5", { size: 20, fill: cols[2], weight: "bold" });
  b += line(330, 382, 360, 382, C.fg, 3.5) + text(366, 388, "simulated", { anchor: "start", size: 20 }) + line(90, 382, 120, 382, C.fg, 2, "6 4") + text(126, 388, "SNR model", { anchor: "start", size: 20 });
  out("spot-threshold/fig-detlimit.svg", svg(560, 400, b));
}

// ---------------------------------------------------------------- fig-poisson.svg
{
  const L = 200, r = rng(77), bg = Array.from({ length: L }, (_, i) => 100 + 300 * Math.exp(-((i - 130) ** 2) / (2 * 22 * 22)));
  const x = bg.map(m => poisson(r, m)), k = 3;
  const ax = axes(85, 20, 470, 260, [0, L - 1], [40, 520], [0, 50, 100, 150], [100, 200, 300, 400, 500], "position (pixel)", "photons", v => String(v), false, false, 20);
  let b = ax.s;
  b += path(x.map((v, i) => [ax.X(i), ax.Y(v)] as [number, number]), { stroke: "#9a9a9a", w: 1.5 });
  b += path(bg.map((v, i) => [ax.X(i), ax.Y(v)] as [number, number]), { stroke: C.fg, w: 2.5 });
  const glob = bg.map(v => v + k * 10), loc = bg.map(v => v + k * Math.sqrt(v));
  b += path(glob.map((v, i) => [ax.X(i), ax.Y(v)] as [number, number]), { stroke: C.fp, w: 3, dash: "8 5" });
  b += path(loc.map((v, i) => [ax.X(i), ax.Y(v)] as [number, number]), { stroke: C.tp, w: 3 });
  let nG = 0, nL = 0;
  x.forEach((v, i) => { if (v > glob[i]) { nG++; b += cross(ax.X(i), ax.Y(v), 5, C.fp, 2.5); } if (v > loc[i]) nL++; });
  console.log("poisson: pixels above global", nG, "above local", nL);
  b += line(95, 372, 125, 372, C.fp, 3, "8 5") + text(131, 379, `background + 3·10 (global): ${nG} pixels above`, { anchor: "start", size: 20 });
  b += line(95, 402, 125, 402, C.tp, 3) + text(131, 409, `background + 3·√background (local): ${nL}`, { anchor: "start", size: 20 });
  b += text(ax.X(20), ax.Y(150), "sd 10", { size: 20 }) + text(ax.X(130), ax.Y(500), "sd 20", { size: 20 });
  out("spot-threshold/fig-poisson.svg", svg(580, 420, b));
}
