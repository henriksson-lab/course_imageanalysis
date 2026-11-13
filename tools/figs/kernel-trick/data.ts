// Data for the slide figures of kernel-trick/ (the svmlift and svm2d demos' presets and the
// course's SVM solver, so the pictures and numbers match the demos).
// Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/kernel-trick/data.ts
// then python3 -I tools/figs/kernel-trick/figs.py <dir>.
import * as fs from "fs";
import * as os from "os";
import { PRESETS as LIFT, liftAndTrain } from "../../../demos/src/svmlift/presets";
import { SVM_PRESETS, BY_HAND_POINTS } from "../../../demos/src/svm2d/presets";
import { trainSVM, decision } from "../../../demos/src/lib/svm";
import { kernelFn } from "../../../demos/src/lib/svmkernels";
import type { LPoint } from "../../../demos/src/lib/datasets2d";

const TMP = process.env.FIGTMP || os.tmpdir();
const out: Record<string, unknown> = {};
const toX = (pts: LPoint[]) => { const X = new Float64Array(2 * pts.length); pts.forEach((p, i) => { X[2 * i] = p.x; X[2 * i + 1] = p.y; }); return X; };
const ys = (pts: LPoint[]) => pts.map(p => (p.c === 1 ? 1 : -1));

// 1. ring lifted onto the paraboloid; the demo's linear SVM in 3D (C = 10^2, the demo's start)
const ring = LIFT.find(p => p.id === "ring")!.points();
const m3 = liftAndTrain(ring, "paraboloid", 100)!;
out.lift = { points: ring, w: Array.from(m3.w!), b: m3.b, sv: Array.from(m3.sv) };
console.log("lift: w", Array.from(m3.w!).map(v => v.toFixed(3)), "b", m3.b.toFixed(3), "SV", m3.sv.length);

// 2. by-hand example: α of each point (linear, C = 1000 as in the demo)
const bh = BY_HAND_POINTS.map(p => ({ ...p }));
const mb = trainSVM(toX(bh), bh.length, 2, ys(bh), { kernel: { type: "linear" }, C: 1000, tol: 1e-6 });
out.byhand = { points: bh, alpha: Array.from(mb.alpha), w: Array.from(mb.w!), b: mb.b };
console.log("byhand alpha", Array.from(mb.alpha).map(v => v.toFixed(3)), "w", Array.from(mb.w!), "b", mb.b);

// 3. γ and C: the "C and γ both large" preset's points, RBF, a 2 × 2 grid of settings
const both = SVM_PRESETS.find(p => p.id === "both")!.points();
const X = toX(both), y = ys(both), R = 120, lo = -5, hi = 5;
const cells: unknown[] = [];
for (const g of [0.03, 5]) for (const C of [0.1, 1000]) {
  const m = trainSVM(X, both.length, 2, y, { kernel: { type: "rbf", gamma: g }, C, tol: 1e-4 });
  const kf = kernelFn(m.kernel), F: number[] = [];
  for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) F.push(decision(m, [lo + (hi - lo) * i / (R - 1), lo + (hi - lo) * j / (R - 1)], 0, kf));
  const err = both.filter((p, i) => y[i] * decision(m, X, 2 * i, kf) <= 0).length;
  cells.push({ gamma: g, C, nSV: m.sv.length, errors: err, sv: Array.from(m.sv), F });
  console.log(`gamma ${g} C ${C}: SV ${m.sv.length} errors ${err} of ${both.length}`);
}
out.grid = { points: both, R, lo, hi, cells };
fs.writeFileSync(TMP + "/kernel-trick.json", JSON.stringify(out));
