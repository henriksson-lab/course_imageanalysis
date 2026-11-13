// Tests of the cellnets-compare unit's comparison code (demos/src/cellcompare/compare.ts):
// the classical pipeline on synthetic nuclei with known answers, the scoring (F1(tau),
// TP/FP/FN, splits, merges, error marks) and the helpers. Run:
//   node_modules/.bin/esbuild demos/test/cellnets-compare-test.ts --bundle --platform=node --log-level=warning | node
import * as C from "../src/cellcompare/compare";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

/** Label image and grey image of discs (cx, cy, r); grey: 20 background, 200 inside, plus a little seeded noise. */
function discs(w: number, h: number, list: [number, number, number][], seed = 3): { lab: Int32Array; grey: Float32Array } {
  const lab = new Int32Array(w * h), grey = new Float32Array(w * h);
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    list.forEach(([cx, cy, r], k) => { if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) lab[i] = k + 1; });
    grey[i] = (lab[i] ? 200 : 20) + 10 * (rnd() - 0.5);
  }
  return { lab, grey };
}

current = "samples";
{
  check(C.SAMPLES.length === 7, "seven annotated images");
  check(new Set(C.SAMPLES.map((s) => s.id)).size === 7, "unique ids");
  check(C.SAMPLES.every((s) => s.labels.endsWith("-labels.png") && s.diameter > 5), "labels file and a diameter for each");
  check(C.SAMPLES.filter((s) => s.dark).map((s) => s.id).join() === "histology,brightfield", "dark nuclei: histology and bright field");
  check(C.TAUS.length === 10 && C.TAUS[0] === 0.5 && C.TAUS[4] === 0.7 && C.TAUS[9] === 0.95, "taus 0.5 … 0.95, index 4 = 0.7");
}

current = "helpers";
{
  const g = C.greyFromPixels([30, 60, 90, 10, 10, 10], 2, 1, 3);
  check(g[0] === 60 && g[1] === 10, "grey = mean of RGB");
  check(C.greyFromPixels([65535], 1, 1, 1, 16)[0] === 255, "16-bit scaled to 255");
  const inv = C.invertGrey(new Float32Array([0, 100, 255]));
  check(inv[0] === 255 && inv[1] === 155 && inv[2] === 0, "invert");
  check(C.compact([0, 7, 7, 3, 0, 9]).join() === "0,1,1,2,0,3", "compact relabels 1..n in order of appearance");
  check(close(C.perHour(2.13), 1690, 1) && close(C.perHour(7.7), 467.5, 0.1) && close(C.perHour(48.3), 74.5, 0.1), "images per hour");
  const rgb = C.rgbFromGrey(new Float32Array([12.4, 300]));
  check(rgb.join() === "12,12,12,255,255,255", "grey -> RGB bytes, clipped");
}

current = "classical/separate";
{
  const w = 120, h = 90;
  const { lab, grey } = discs(w, h, [[25, 25, 10], [70, 30, 12], [40, 65, 9], [95, 65, 11]]);
  const r = C.classical(grey, w, h, C.CLASSICAL_DEFAULT);
  check(r.count === 4, `four separate discs -> four objects (got ${r.count})`);
  check(r.thr > 0.05 && r.thr < 0.95, `Otsu threshold between the two levels (${r.thr.toFixed(2)})`);
  const s = C.score(lab, r.labels);
  check(s.f1[0] === 1 && s.tp5 === 4 && s.fp5 === 0 && s.fn5 === 0, `F1 at 0.5 = 1 (got ${s.f1[0]})`);
  check(s.splits === 0 && s.merges === 0, "no splits or merges");
}

current = "classical/touching";
{
  // Two discs of radius 14 whose centres are 22 px apart: a neck the distance transform splits.
  const w = 100, h = 60;
  const { lab, grey } = discs(w, h, [[36, 30, 14], [58, 30, 14]]);
  const split = C.classical(grey, w, h, { ...C.CLASSICAL_DEFAULT, h: 1 });
  check(split.count === 2, `h = 1 splits the touching pair (got ${split.count})`);
  const merged = C.classical(grey, w, h, { ...C.CLASSICAL_DEFAULT, h: 8 });
  check(merged.count === 1, `h = 8 keeps them as one (got ${merged.count})`);
  const s = C.score(lab, merged.labels);
  check(s.merges === 1 && s.tp5 <= 1, `one merge detected (merges ${s.merges})`);
  const marks = C.errorMarks(lab, merged.labels, w);
  check(marks.some((m) => m.kind === "merge") && !marks.some((m) => m.kind === "miss"), "merge mark, merged objects not also counted as misses");
  const small = C.classical(grey, w, h, { ...C.CLASSICAL_DEFAULT, minSize: 5000 });
  check(small.count === 0, "minimum size above every object removes all");
  const manual = C.classical(grey, w, h, { ...C.CLASSICAL_DEFAULT, method: "manual", thr: 0.5 });
  check(manual.count === 2 && manual.thr === 0.5, "manual threshold is used as given");
}

current = "score";
{
  const w = 80, h = 40;
  const { lab } = discs(w, h, [[20, 20, 10], [60, 20, 10]]);
  const same = C.score(lab, lab);
  check(same.f1.every((f) => f === 1), "identical labels: F1 = 1 at every tau");
  // A prediction shifted by 4 px: IoU about 0.6 -> matched at 0.5, not at 0.7.
  const shifted = new Int32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 4; x < w; x++) shifted[y * w + x] = lab[y * w + x - 4];
  const s = C.score(lab, shifted);
  check(s.tp5 === 2 && s.tp7 === 0 && s.f1[4] === 0, `shifted: TP 2 at 0.5, 0 at 0.7 (got ${s.tp5}, ${s.tp7})`);
  for (let i = 1; i < s.f1.length; i++) check(s.f1[i] <= s.f1[i - 1] + 1e-12, "F1(tau) does not increase");
  // An extra object and a missing one.
  const { lab: other } = discs(w, h, [[20, 20, 10], [70, 8, 4]]);
  const e = C.score(lab, other);
  check(e.tp5 === 1 && e.fp5 === 1 && e.fn5 === 1 && close(e.f1[0], 0.5, 1e-12), "one TP, one FP, one FN: F1 0.5");
  const marks = C.errorMarks(lab, other, w);
  check(marks.filter((m) => m.kind === "miss").length === 1 && marks.filter((m) => m.kind === "extra").length === 1, "one miss and one extra mark");
  // A true object split into two halves.
  const halves = new Int32Array(lab);
  for (let i = 0; i < halves.length; i++) if (halves[i] === 1 && (i % w) >= 20) halves[i] = 3;
  const sp = C.score(lab, halves);
  check(sp.splits === 1, `split detected (got ${sp.splits})`);
  check(C.errorMarks(lab, halves, w).some((m) => m.kind === "split"), "split mark");
}

current = "worked F1 (B1)";
{
  const f1 = (tp: number, np: number, nt: number) => (2 * tp) / (np + nt);
  check(close(f1(18, 21, 20), 0.878, 5e-4) && close(f1(12, 21, 20), 0.585, 5e-4), "method A: 0.878, 0.585");
  check(close(f1(16, 17, 20), 0.865, 5e-4) && close(f1(15, 17, 20), 0.811, 5e-4), "method B: 0.865, 0.811");
}

console.log(`cellnets-compare-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
