// Data for shape-models/lec.qmd, computed with the course's shape code (demos/src/lib/shapes.ts)
// and the demos' own settings: two hands (landmarks), the GPA set of the procrustes demo
// (10 leaves, seed 21), and the variance shares of the shapepca demo (40 hands; with 20 %
// misnumbered shapes for the "bad correspondence" preset).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/shape-models/data.ts --bundle --platform=node --outfile=<tmp>/shm-data.js
//   node <tmp>/shm-data.js <tmp>/shm-data.json
// then python3 -I tools/figs/shape-models/figs.py <tmp>/shm-data.json
import { writeFileSync } from "fs";
import { Shape, family, trainingSet, normalise, gpaStep, sumSq } from "../../../demos/src/lib/shapes";
import { buildModel, varianceShares } from "../../../demos/src/shapepca/model";

const arr = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 10000) / 10000);
const hand = family("hand");
const hands = [hand.make([-1.0, 0.25, 0.95]), hand.make([1.6, -0.2, 1.08]), hand.make(hand.params.map(p => p.mean))].map(arr);

// GPA exactly as the procrustes demo's "GPA on a set"
const ts = trainingSet(family("leaf"), { count: 10, seed: 21, noise: 0.02, rot: 0.6, logScale: 0.25, shift: 0.7 });
const raw: Shape[] = ts.shapes.map(s => { const o = new Float64Array(24); for (let i = 0; i < 12; i++) { o[2 * i] = s[4 * i] * 0.9; o[2 * i + 1] = -s[4 * i + 1] * 0.9; } return o; });
let al = raw.map(normalise); const ref = al[0].slice(); let mean: Shape = ref.slice();
const resid = (A: Shape[], m: Shape) => A.reduce((t, s) => t + sumSq(s, m), 0);
const gpa: any[] = [{ iter: 0, resid: resid(al, mean), al: al.map(arr), mean: arr(mean) }];
for (let it = 1; it <= 3; it++) {
  const r = gpaStep(al, mean, ref); al = r.aligned; mean = r.mean;
  gpa.push({ iter: it, resid: resid(al, mean), change: r.change, al: al.map(arr), mean: arr(mean) });
}
gpa.forEach(g => console.log("GPA iter", g.iter, "resid", g.resid.toFixed(4), "change", g.change));

const good = buildModel({ family: "hand", count: 40, noise: 0.01, align: true, corrupt: 0 });
const bad = buildModel({ family: "hand", count: 40, noise: 0.01, align: true, corrupt: 0.2 });
const sh = (m: any) => varianceShares(m.model, 8);
console.log("good", sh(good).share.map(v => v.toFixed(1)).join(" "), "bad", sh(bad).share.map(v => v.toFixed(1)).join(" "), "corrupted", bad.corrupted.filter(Boolean).length);
writeFileSync(process.argv[2], JSON.stringify({ hands, raw: raw.map(arr), gpa, shares: { good: sh(good), bad: sh(bad), nbad: bad.corrupted.filter(Boolean).length } }));
