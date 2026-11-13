// Extra slide figures for histogram/lec.qmd (the first set is in tools/figs/thresholding/figs.ts).
// Run: node_modules/.bin/esbuild tools/figs/histogram/figs.ts --bundle --platform=node --outfile=/tmp/x.js && node /tmp/x.js
import { boxFilter } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/imageops";
import { intHistogram } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/histogram";
import * as HM from "/data/henriksson/github/course/imageanalysis/demos/src/histthresh/model";
import { rng, shuffle } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/random";
import { Svg, Panel, C, u8, writeRgbTiles, greyRgb } from "../thresholding/plot";

const truth = HM.blobTruth();
const base = { source: "blobs" as const, contrast: 100, noise: 8, median: false, blur: 0 };
const NPIX = 256 * 256;
const area = (h: ArrayLike<number>) => { const a = new Float64Array(256); let s = 0; for (let t = 255; t >= 0; t--) { a[t] = s / NPIX; s += h[t]; } return a; }; // a[t] = fraction > t
const xs = Array.from({ length: 256 }, (_, i) => i);

{ // cumulative histogram and the percentile threshold (clean blobs)
  const img = HM.makeImage(base, truth, null), h = intHistogram(img.data, 0, 255).counts;
  const P = new Float64Array(256); let s = 0; for (let t = 0; t < 256; t++) { s += h[t]; P[t] = s / NPIX; }
  let tq = 0; while (P[tq] < 0.75) tq++;
  console.log("percentile 25 % t =", tq, "P =", P[tq].toFixed(4));
  const S = new Svg(600, 360), p = new Panel(S, 90, 20, 480, 260, [0, 255], [0, 1]);
  p.axes([0, 50, 100, 150, 200, 250], [0, 0.25, 0.5, 0.75, 1], "threshold t", "fraction ≤ t");
  const mx = Math.max(...Array.from(h));
  p.bars(xs.map(x => x - 0.5).concat([255.5]), Array.from(h, v => 0.45 * v / mx), "#9ab", 0.6);
  p.line(xs, P, C.c1, 3.5);
  p.line([0, tq], [0.75, 0.75], C.c4, 2.5, "7 5"); p.line([tq, tq], [0.75, 0], C.c4, 2.5, "7 5"); p.dot(tq, 0.75, C.c4, 8);
  p.text(tq + 6, 0.06, `t = ${tq}`, C.c4, "start", 20);
  p.text(4, 0.80, "1 − q = 0.75", C.c4, "start", 20);
  p.text(185, 0.9, "P(t)", C.c1, "middle", 20);
  p.text(60, 0.5, "histogram", "#678", "middle", 19);
  S.out("histogram/fig-cumulative.svg");
}

{ // the mask area 1 − P(t): flat in a deep valley, steep in a shallow one
  const S = new Svg(600, 360), p = new Panel(S, 90, 20, 480, 260, [0, 255], [0, 1]);
  p.axes([0, 50, 100, 150, 200, 250], [0, 0.25, 0.5, 0.75, 1], "threshold t", "mask area 1 − P(t)");
  p.rect(100, 130, 0, 1, "#f2e6c9", 0.7);
  p.text(115, 0.93, "valley", "#8a6d1f", "middle", 19);
  const cols = [C.c1, C.c2];
  [8, 35].forEach((noise, k) => {
    const img = HM.makeImage({ ...base, noise }, truth, null), h = intHistogram(img.data, 0, 255).counts, a = area(h);
    console.log("noise", noise, "A(100)", a[100].toFixed(3), "A(130)", a[130].toFixed(3), "slope ‰", (1000 * (a[100] - a[130]) / 30).toFixed(2));
    p.line(xs, a, cols[k], 3.5, k ? "9 5" : "");
  });
  p.text(165, 0.40, "clean (σ = 8)", C.c1, "start", 20);
  p.text(165, 0.52, "noisy (σ = 35)", C.c2, "start", 20);
  S.out("histogram/fig-area.svg");
}

{ // the 5 × 5 example: image and its histogram with mean and ± one standard deviation
  const h = [3, 8, 4, 1, 1, 1, 5, 2];
  const vals: number[] = []; h.forEach((c, i) => { for (let k = 0; k < c; k++) vals.push(i); });
  shuffle(rng(11), vals);
  const mu = vals.reduce((a, b) => a + b, 0) / 25, v = vals.reduce((a, b) => a + (b - mu) ** 2, 0) / 25;
  console.log("5x5 mean", mu.toFixed(2), "var", v.toFixed(2));
  const S = new Svg(560, 480), cell = 40, ox = 30, oy = 10;
  vals.forEach((g, k) => {
    const x = ox + (k % 5) * cell, y = oy + Math.floor(k / 5) * cell, grey = Math.round(255 * g / 7);
    S.add(`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="rgb(${grey},${grey},${grey})" stroke="#fff" stroke-width="2"/>`);
    S.text(x + cell / 2, y + cell / 2 + 7, String(g), g > 3 ? C.fg : "#fff", "middle", 19);
  });
  S.text(ox + 5 * cell + 20, oy + 2.5 * cell + 7, "25 pixels, levels 0–7", C.fg, "start", 19);
  const p = new Panel(S, 90, 230, 440, 180, [-0.6, 7.6], [0, 0.45]);
  p.axes([0, 1, 2, 3, 4, 5, 6, 7], [0, 0.1, 0.2, 0.3], "grey level i", "pᵢ");
  h.forEach((c, i) => p.bars([i - 0.38, i + 0.38], [c / 25], C.c1, 0.8));
  const sd = Math.sqrt(v);
  p.line([mu - sd, mu + sd], [0.36, 0.36], C.c4, 3); p.dot(mu, 0.36, C.c4, 7);
  p.text(mu, 0.385, `μ = ${mu.toFixed(2)}, σ = ${sd.toFixed(2)}`, C.c4, "middle", 19);
  S.out("histogram/fig-toy.svg");
}

{ // smoothing first: noisy image and mask, the same after a 3 × 3 mean (t = 110)
  const noisy = HM.makeImage({ ...base, noise: 35 }, truth, null);
  const sm = boxFilter(noisy, 1);
  const t = 110;
  const dice = (m: ArrayLike<number>) => { let tp = 0, a = 0, b = 0; for (let i = 0; i < NPIX; i++) { tp += m[i] && truth[i] ? 1 : 0; a += m[i] ? 1 : 0; b += truth[i] ? 1 : 0; } return 2 * tp / (a + b); };
  const m1 = Uint8Array.from(noisy.data, v => (v > t ? 1 : 0)), m2 = Uint8Array.from(sm.data, v => (v > t ? 1 : 0));
  console.log("Dice noisy", dice(m1).toFixed(3), "after 3x3 mean", dice(m2).toFixed(3));
  const wrong = (m: ArrayLike<number>) => { let w = 0; for (let i = 0; i < NPIX; i++) w += (m[i] ? 1 : 0) !== truth[i] ? 1 : 0; return w / NPIX; };
  console.log("wrong pixels noisy", (100 * wrong(m1)).toFixed(1), "% after", (100 * wrong(m2)).toFixed(1), "%");
  const bw = (m: Uint8Array) => greyRgb(Uint8Array.from(m, v => (v ? 255 : 0)));
  writeRgbTiles("histogram/fig-smooth.png", [greyRgb(u8(noisy)), bw(m1), greyRgb(u8(sm)), bw(m2)], 2, 8);
}
console.log("histogram figures written");
