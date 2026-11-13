// Data for the slide figures of maxmargin/ (the svm2d demo's presets and solver, so the
// pictures and numbers match the demo). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/maxmargin/data.ts
// then python3 -I tools/figs/maxmargin/figs.py <dir>.
import * as fs from "fs";
import * as os from "os";
import { SVM_PRESETS, BY_HAND_POINTS } from "../../../demos/src/svm2d/presets";
import { trainSVM, marginWidth, decision, hingeLoss, logisticLoss, squaredLoss, zeroOneLoss } from "../../../demos/src/lib/svm";
import type { LPoint } from "../../../demos/src/lib/datasets2d";

const TMP = process.env.FIGTMP || os.tmpdir();
function fit(pts: LPoint[], C: number) {
  const X = new Float64Array(2 * pts.length);
  pts.forEach((p, i) => { X[2 * i] = p.x; X[2 * i + 1] = p.y; });
  const m = trainSVM(X, pts.length, 2, pts.map(p => (p.c === 1 ? 1 : -1)), { kernel: { type: "linear" }, C, tol: 1e-4 });
  const yf = pts.map((p, i) => (p.c === 1 ? 1 : -1) * decision(m, X, 2 * i));
  const errors = yf.filter(v => v <= 0).length;
  return { points: pts, C, w: Array.from(m.w!), b: m.b, alpha: Array.from(m.alpha), sv: Array.from(m.sv), width: marginWidth(m), errors, yf };
}
const preset = (id: string) => SVM_PRESETS.find(p => p.id === id)!;
const out: Record<string, unknown> = {};
out.byhand = fit(BY_HAND_POINTS.map(p => ({ ...p })), 1000);
const nonsv = preset("nonsv").points();
out.nonsv = fit(nonsv, 1000);
// move two points without a ring (not support vectors), staying outside the street
const moved = nonsv.map(p => ({ ...p }));
const mv: [number, number, number][] = [[3, 3.6, 1.0], [5, -0.6, 2.4]];   // index, new x, new y
for (const [i, x, y] of mv) { moved[i].x = x; moved[i].y = y; }
out.nonsvMoved = fit(moved, 1000);
out.moves = mv;
const outl = preset("outlier").points();
out.outlierHard = fit(outl, 1000);
out.outlierSoft = fit(outl, 1);
out.losses = [2, 1, 0.5, 0, -1].map(m => ({ m, zo: zeroOneLoss(m), hinge: hingeLoss(m), logistic: logisticLoss(m), squared: squaredLoss(m) }));
for (const k of ["byhand", "nonsv", "nonsvMoved", "outlierHard", "outlierSoft"]) {
  const r = out[k] as ReturnType<typeof fit>;
  console.log(k, "w", r.w.map(v => v.toFixed(3)), "b", r.b.toFixed(3), "width", r.width.toFixed(2), "SV", r.sv.length, "errors", r.errors);
}
fs.writeFileSync(TMP + "/maxmargin.json", JSON.stringify(out));
