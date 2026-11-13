// Readouts of the analytic flowtrack demo for the cellpose/ exercise (Explore: following the
// arrows) and the lecture notes. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/cellpose/demo_numbers.ts --bundle --platform=node --log-level=warning | node
import * as M from "../../../demos/src/flowtrack/model";
import { iouTable, match, objectScores } from "../../../demos/src/lib/eval";
const N = M.SIZE;
const E0: M.Errors = { noise: 0, blur: 0, shared: false, debris: false };

function run(shape: M.ShapeId, e: Partial<M.Errors>, steps: number, thr = 0.4, iters: number | null = null, eta = 1) {
  const lab = M.makeShape(shape);
  const pr = M.predict(lab, N, N, { ...E0, ...e }, iters);
  const d = M.decode(pr.fg, pr.vx, pr.vy, N, N, { steps, flowThreshold: thr, iters, eta });
  const sc = objectScores(match(iouTable(lab, d.masks), 0.5));
  let area = 0, assigned = 0; for (let p = 0; p < N * N; p++) { if (lab[p]) area++; if (d.masks[p]) assigned++; }
  return `${shape} ${JSON.stringify(e)} steps ${steps} eta ${eta} thr ${thr} k ${iters ?? "auto " + pr.target.iters.join("/")}: area ${area}, seeds ${d.seeds.length}, masks ${d.nRaw} (small ${d.small}), errors ${d.errors.map((x) => x.toFixed(2)).join(" ")}, final ${d.count}, assigned ${assigned}, TP ${sc.tp} FP ${sc.fp} FN ${sc.fn}`;
}

console.log("== shapes: area, source, auto k, longest path (steps until every pixel is within 1.5 px of its source)");
for (const s of M.SHAPES) {
  const lab = M.makeShape(s.id);
  const T = M.flowTargets(lab, N, N);
  let worst = 0, fewestK = 0;
  for (let p = 0; p < N * N; p++) if (lab[p]) {
    const src = T.sources[lab[p] - 1], sx = src % N, sy = Math.floor(src / N);
    const path = M.tracePath(T.vx, T.vy, N, N, p % N, Math.floor(p / N), 200, 1);
    let j = 0; while (j <= 200 && Math.hypot(path[2 * j] - sx, path[2 * j + 1] - sy) > 1.5) j++;
    worst = Math.max(worst, j);
  }
  // smallest k at which every pixel of every cell has heat > 0
  for (let k = 1; k < 300; k++) {
    let ok = true;
    let n = 0; for (const v of lab) if (v > n) n = v;
    for (let l = 1; l <= n && ok; l++) { const H = M.heatTarget(lab, N, N, l, k); for (let p = 0; p < N * N; p++) if (lab[p] === l && !(H[p] > 0)) { ok = false; break; } }
    if (ok) { fewestK = k; break; }
  }
  let area = 0; for (const v of lab) if (v) area++;
  console.log(`${s.id}: area ${area}, sources ${T.sources.map((p) => `(${p % N}, ${Math.floor(p / N)})`).join(" ")}, auto k ${T.iters.join("/")}, longest path ${worst} steps, heat everywhere from k = ${fewestK}`);
}

console.log("\n== heat spreads (C, k = 10, 20, 40)");
for (const k of [5, 10, 20, 40]) {
  const lab = M.makeShape("C");
  const T = M.flowTargets(lab, N, N, k);
  let arrows = 0, hot = 0, area = 0; for (let p = 0; p < N * N; p++) if (lab[p]) { area++; if (T.vx[p] || T.vy[p]) arrows++; if (T.heat[p] > 0) hot++; }
  console.log(`k ${k}: pixels with heat ${hot}/${area}, with an arrow ${arrows}; ${run("C", {}, 100, 0.4, k)}`);
}

console.log("\n== bend: clicked tip (30, 12) of the C");
{
  const lab = M.makeShape("C"), T = M.flowTargets(lab, N, N);
  const path = M.tracePath(T.vx, T.vy, N, N, 30, 12, 100, 1);
  const src = T.sources[0];
  let j = 0; while (j < 100 && Math.hypot(path[2 * j] - (src % N), path[2 * j + 1] - Math.floor(src / N)) > 1) j++;
  console.log(`end (${path[200].toFixed(1)}, ${path[201].toFixed(1)}), source (${src % N}, ${Math.floor(src / N)}), within 1 px after ${j} steps, straight distance ${Math.hypot(30 - (src % N), 12 - Math.floor(src / N)).toFixed(1)}`);
  // does the straight segment leave the cell?
  let out = 0; for (let t = 0; t <= 1; t += 0.01) { const x = Math.round(30 + t * ((src % N) - 30)), y = Math.round(12 + t * (Math.floor(src / N) - 12)); if (!lab[y * N + x]) out++; }
  console.log(`straight segment tip -> source: ${out} of 101 samples outside the cell`);
}

console.log("\n== too few steps");
for (const st of [5, 10, 15, 20, 25, 30, 40]) console.log(run("C", {}, st));
for (const st of [5, 10, 20]) console.log(run("rod", {}, st));
for (const eta of [0.5, 2]) console.log(run("C", {}, 10, 0.4, null, eta));

console.log("\n== noise");
for (const nz of [0.2, 0.3, 0.35, 0.5, 0.8]) { console.log(run("C", { noise: nz }, 100, 0)); console.log(run("C", { noise: nz }, 100, 0.4)); }
for (const nz of [0.35, 0.5]) console.log(run("disc", { noise: nz }, 100, 0.4));
console.log(run("C", { noise: 0.35, blur: 1 }, 100, 0));

console.log("\n== debris / flow check");
for (const thr of [0.4, 1.0, 1.25, 1.5, 0]) console.log(run("disc", { debris: true }, 100, thr));

console.log("\n== shared field / blur on touching");
console.log(run("touching", {}, 100));
console.log(run("touching", { shared: true }, 100));
for (const b of [1, 2, 3, 4]) console.log(run("touching", { blur: b }, 100));
console.log(run("dumbbell", {}, 100));
console.log(run("dumbbell", { noise: 0.45 }, 100, 0));
