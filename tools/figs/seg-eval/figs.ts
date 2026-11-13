// Extra slide figures for seg-eval/lec.qmd (the first set is in tools/figs/thresholding/figs.ts).
// Run: node_modules/.bin/esbuild tools/figs/seg-eval/figs.ts --bundle --platform=node --outfile=/tmp/x.js && node /tmp/x.js
import { rocModel } from "/data/henriksson/github/course/imageanalysis/demos/src/threshroc/model";
import { intHistogram } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/histogram";
import { confusion, dice } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/metrics";
import { Svg, Panel, C, TPc, FPc, FNc, RGB, writeRgbTiles } from "../thresholding/plot";

const N = 256;
const rgbS = (c: RGB) => `rgb(${c.join(",")})`;
/** Overlay coloured by the four cases, like the threshroc demo (alpha 0.62). */
function cases(img: ArrayLike<number>, pred: ArrayLike<number>, truth: ArrayLike<number>, a = 0.62): Uint8Array {
  const out = new Uint8Array(N * N * 3);
  for (let i = 0; i < N * N; i++) {
    const g = img[i], p = pred[i], q = truth[i], col = p && q ? TPc : p ? FPc : q ? FNc : null;
    for (let c = 0; c < 3; c++) out[3 * i + c] = col ? (1 - a) * g + a * col[c] : g;
  }
  return out;
}
const f3 = (v: number) => v.toFixed(3);

{ // confusion overlay: "Otsu vs max Dice" preset at Otsu's t and at the max-Dice t
  const M = rocModel({ fraction: 0.1, contrast: 60, noise: 15, light: 0, smooth: false });
  const ims = [M.marks.otsu, M.marks.dice].map(t => {
    const pred = Uint8Array.from(M.img.data, v => (v > t ? 1 : 0)), c = confusion(pred, M.truth);
    console.log(`t=${t}: TP ${c.tp} FP ${c.fp} FN ${c.fn} TN ${c.tn} precision ${f3(c.tp / (c.tp + c.fp))} recall ${f3(c.tp / (c.tp + c.fn))} Dice ${f3(dice(c))}`);
    return cases(M.img.data, pred, M.truth);
  });
  writeRgbTiles("seg-eval/fig-overlay.png", ims, 2, 8);
}

{ // accuracy misleads: 3 % objects, the "all background" answer: every object pixel is a miss
  const M = rocModel({ fraction: 0.03, contrast: 50, noise: 20, light: 0, smooth: false });
  const none = new Uint8Array(N * N), c = confusion(none, M.truth);
  console.log(`rare: object fraction ${f3(c.fn / (N * N))} accuracy ${f3((c.tp + c.tn) / (N * N))}`);
  const t = M.marks.dice, pred = Uint8Array.from(M.img.data, v => (v > t ? 1 : 0)), c2 = confusion(pred, M.truth);
  console.log(`rare max Dice t=${t}: accuracy ${f3((c2.tp + c2.tn) / (N * N))} Dice ${f3(dice(c2))}`);
  writeRgbTiles("seg-eval/fig-rare.png", [cases(M.img.data, none, M.truth, 0.85), cases(M.img.data, pred, M.truth, 0.85)], 2, 8);
}

// the 5 × 5 example: truth 7 pixels; TP 6, FP 2, FN 1, TN 16
const T5 = [
  0, 0, 0, 0, 0,
  0, 1, 1, 0, 0,
  0, 1, 1, 1, 0,
  0, 0, 1, 1, 0,
  0, 0, 0, 0, 0];
const P5 = [
  0, 0, 0, 0, 0,
  0, 1, 1, 1, 0,
  0, 1, 1, 1, 0,
  0, 0, 1, 0, 1,
  0, 0, 0, 0, 0];
{
  const c = confusion(P5, T5); console.log("5x5", c, "Dice", f3(dice(c)));
  const S = new Svg(450, 300), cell = 52, ox = 10, oy = 10;
  for (let k = 0; k < 25; k++) {
    const p = P5[k], q = T5[k], col = p && q ? rgbS(TPc) : p ? rgbS(FPc) : q ? rgbS(FNc) : "#e4e4e4", lab = p && q ? "TP" : p ? "FP" : q ? "FN" : "";
    const x = ox + (k % 5) * cell, y = oy + Math.floor(k / 5) * cell;
    S.add(`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${col}" stroke="#fff" stroke-width="3"/>`);
    if (lab) S.text(x + cell / 2, y + cell / 2 + 7, lab, "#fff", "middle", 19, 'font-weight="bold"');
  }
  const rows: [string, string, number][] = [["TP", rgbS(TPc), c.tp], ["FP", rgbS(FPc), c.fp], ["FN", rgbS(FNc), c.fn], ["TN", "#bbb", c.tn]];
  rows.forEach(([l, col, n], k) => { S.add(`<rect x="295" y="${22 + k * 50}" width="30" height="30" fill="${col}"/>`); S.text(337, 45 + k * 50, `${l} = ${n}`, C.fg, "start", 23); });
  S.text(295, 245, "truth:", C.fg, "start", 21); S.text(295, 272, "7 pixels", C.fg, "start", 21);
  S.out("seg-eval/fig-5x5.svg");
}

{ // Dice and Jaccard as areas: the two masks as overlapping regions, numbers of the 5 × 5 example
  const S = new Svg(560, 340);
  S.add(`<circle cx="210" cy="160" r="120" fill="${rgbS(FNc)}" fill-opacity="0.35" stroke="${rgbS(FNc)}" stroke-width="3"/>`);
  S.add(`<circle cx="350" cy="160" r="120" fill="${rgbS(FPc)}" fill-opacity="0.35" stroke="${rgbS(FPc)}" stroke-width="3"/>`);
  S.add(`<clipPath id="cl"><circle cx="210" cy="160" r="120"/></clipPath><circle cx="350" cy="160" r="120" clip-path="url(#cl)" fill="${rgbS(TPc)}" fill-opacity="0.75"/>`);
  S.text(150, 168, "FN 1", C.fg, "middle", 24); S.text(280, 168, "TP 6", "#fff", "middle", 24, 'font-weight="bold"'); S.text(410, 168, "FP 2", C.fg, "middle", 24);
  S.text(150, 30, "truth", rgbS(FNc), "middle", 21); S.text(410, 30, "prediction", rgbS(FPc), "middle", 21);
  S.text(280, 315, "Dice = 2·6 / (7 + 8) = 0.80     Jaccard = 6 / 9 = 0.67", C.fg, "middle", 20);
  S.out("seg-eval/fig-dice.svg");
}

{ // sweeping t: the two class histograms; TPR and FPR are the areas above t
  const M = rocModel({ fraction: 0.1, contrast: 60, noise: 15, light: 0, smooth: false }), t = M.marks.otsu, q = M.at(t);
  const hO = intHistogram(M.img.data, 0, 255, M.truth).counts, hB = intHistogram(M.img.data, 0, 255, Uint8Array.from(M.truth, v => 1 - v)).counts;
  const nO = Array.from(hO).reduce((a, b) => a + b, 0), nB = Array.from(hB).reduce((a, b) => a + b, 0);
  const dO = Array.from(hO, v => v / nO), dB = Array.from(hB, v => v / nB);
  console.log(`sweep t=${t}: TPR ${f3(q.tpr)} FPR ${f3(q.fpr)} AUC ${f3(M.auc)}`);
  const S = new Svg(520, 610), mx = Math.max(...dB, ...dO) * 1.1, xs = Array.from({ length: 256 }, (_, i) => i);
  const p = new Panel(S, 70, 40, 430, 210, [0, 200], [0, mx]);
  p.axes([0, 50, 100, 150, 200], [], "grey level", "fraction of class");
  const above = (d: number[]) => [xs.filter(x => x > t), d.filter((_, i) => i > t)] as const;
  { const [ax, ay] = above(dB); p.area(ax, ay, C.c1, 0.5); } { const [ax, ay] = above(dO); p.area(ax, ay, C.c4, 0.5); }
  p.line(xs, dB, C.c1, 2.5); p.line(xs, dO, C.c4, 2.5);
  p.vline(t + 0.5, C.fg, "", 2.5); S.text(p.X(t) + 6, 32, `t = ${t} (Otsu)`, C.fg, "start", 19);
  S.text(p.X(60), p.Y(mx * 0.8), "background", C.c1, "end", 19); S.text(p.X(155), p.Y(mx * 0.55), "object", C.c4, "start", 19);
  S.text(20, 318, `right of t:  TPR ${q.tpr.toFixed(2)} of the object pixels,`, C.c4, "start", 19);
  S.text(20, 344, `FPR ${q.fpr.toFixed(2)} of the background pixels`, C.c1, "start", 19);
  const r = new Panel(S, 170, 380, 190, 170, [0, 1], [0, 1]);
  r.axes([0, 0.5, 1], [0, 0.5, 1], "FPR", "TPR");
  r.line([0, 1], [0, 1], "#bbb", 1.5, "5 5");
  r.line(M.pts.map(z => z.fpr), M.pts.map(z => z.tpr), C.c3, 3);
  r.dot(q.fpr, q.tpr, C.fg, 8);
  S.text(r.X(q.fpr) + 12, r.Y(q.tpr) + 26, `t = ${t}`, C.fg, "start", 19);
  S.text(r.X(0.55), r.Y(0.15), `AUC ${M.auc.toFixed(2)}`, C.c3, "start", 19);
  S.out("seg-eval/fig-sweep.svg");
}

{ // ROC hides class imbalance, precision–recall shows it: 50 % vs 3 % objects, same contrast and noise
  const S = new Svg(640, 340);
  const a = new Panel(S, 80, 30, 220, 220, [0, 1], [0, 1]), b = new Panel(S, 400, 30, 220, 220, [0, 1], [0, 1]);
  a.axes([0, 0.5, 1], [0, 0.5, 1], "FPR", "TPR (recall)"); b.axes([0, 0.5, 1], [0, 0.5, 1], "recall", "precision");
  a.line([0, 1], [0, 1], "#bbb", 1.5, "5 5");
  ([[0.5, C.c1, ""], [0.03, C.c2, "9 5"]] as const).forEach(([fr, col, dash]) => {
    const M = rocModel({ fraction: fr, contrast: 50, noise: 20, light: 0, smooth: false });
    const q = M.pts.reduce((best, z) => (Math.abs(z.tpr - 0.9) < Math.abs(best.tpr - 0.9) ? z : best));
    console.log(`fraction ${fr}: AUC ${f3(M.auc)}; at TPR≈0.9: FPR ${f3(q.fpr)} precision ${f3(q.precision)}; max Dice ${f3(Math.max(...M.pts.map(z => z.dice)))}`);
    a.line(M.pts.map(z => z.fpr), M.pts.map(z => z.tpr), col, 3, dash);
    b.line(M.pts.map(z => z.tpr), M.pts.map(z => z.precision), col, 3, dash);
  });
  S.text(140, 320, "— 50 % objects", C.c1, "start", 20); S.text(360, 320, "- - 3 % objects", C.c2, "start", 20);
  S.out("seg-eval/fig-imbalance.svg");
}

{ // pixels are not objects: two touching discs merged; a thin spur that Dice hardly sees
  const disc = (cx: number, cy: number, r: number) => Uint8Array.from({ length: N * N }, (_, i) => ((i % N - cx) ** 2 + (Math.floor(i / N) - cy) ** 2 <= r * r ? 1 : 0));
  const or = (...m: Uint8Array[]) => Uint8Array.from(m[0], (_, i) => (m.some(x => x[i]) ? 1 : 0));
  const A = disc(92, 128, 36), B = disc(164, 128, 36), truth2 = or(A, B);
  const bridge = Uint8Array.from({ length: N * N }, (_, i) => { const x = i % N, y = Math.floor(i / N); return x >= 116 && x <= 140 && Math.abs(y - 128) <= 14 ? 1 : 0; });
  const merged = or(truth2, bridge);
  const c1 = confusion(merged, truth2);
  const T = disc(128, 128, 60), spur = Uint8Array.from({ length: N * N }, (_, i) => { const x = i % N, y = Math.floor(i / N); return x >= 128 && x <= 238 && Math.abs(y - 128) <= 1 ? 1 : 0; });
  const P = or(T, spur), c2 = confusion(P, T);
  // Hausdorff distance between the two masks (brute force over foreground pixels)
  const pts = (m: Uint8Array) => { const o: number[][] = []; for (let i = 0; i < N * N; i++) if (m[i]) o.push([i % N, Math.floor(i / N)]); return o; };
  const directed = (X: number[][], Y: number[][]) => { let h = 0; for (const [x, y] of X) { let d = Infinity; for (const [u, v] of Y) d = Math.min(d, (x - u) ** 2 + (y - v) ** 2); h = Math.max(h, d); } return Math.sqrt(h); };
  const pp = pts(P), tt = pts(T), hd = Math.max(directed(pp, tt), directed(tt, pp));
  console.log(`merged: Dice ${f3(dice(c1))} FP ${c1.fp} (2 true objects, 1 predicted); spur: Dice ${f3(dice(c2))} Hausdorff ${hd.toFixed(1)}`);
  const white = new Uint8Array(N * N).fill(235);
  writeRgbTiles("seg-eval/fig-objects.png", [cases(white, merged, truth2, 0.85), cases(white, P, T, 0.85)], 2, 8);
}
console.log("seg-eval figures written");
