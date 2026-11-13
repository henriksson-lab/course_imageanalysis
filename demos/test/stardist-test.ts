// Tests for the star-convex polygon unit (networks for cells and objects day, unit stardist/):
// the geometry of demos/src/starrays/geom.ts (rays, polygon area, polygon pixels and IoU,
// objectness, best centre, polygon NMS) against closed forms and brute force, the worked numbers
// of the lecture and exercise, the shape presets, and the simulated decoding of the demo's
// decode mode (each preset shows the effect its label promises).
// Run: node_modules/.bin/esbuild demos/test/stardist-test.ts --bundle --platform=node --log-level=warning | node
import * as G from "../src/starrays/geom";
import * as S from "../src/starrays/scene";
import { SHAPES } from "../src/starrays/shapes";
import { rasterise } from "../src/lib/shapecanvas";
import { nmsGeneric } from "../src/lib/boxes";
import { rng } from "../src/lib/ui";

let fails = 0, passes = 0;
function check(name: string, ok: boolean, info = ""): void {
  if (ok) passes++; else { fails++; console.log("FAIL", name, info); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const W = 64;
const shapeMask = (id: string) => rasterise(SHAPES.find(s => s.id === id)!.polys(), W, W);

// ---- closed forms: rectangle, ellipse, disc, given rays ----------------------------------
{
  const r8 = G.rectRays(4, 2, 8);
  check("rect rays", [4, 2 * Math.SQRT2, 2, 2 * Math.SQRT2, 4, 2 * Math.SQRT2, 2, 2 * Math.SQRT2].every((v, k) => near(r8[k], v, 1e-12)), Array.from(r8).join(","));
  check("rect 8 rays area 24", near(G.polyArea(r8), 24, 1e-9), String(G.polyArea(r8)));
  const a32 = G.polyArea(G.rectRays(4, 2, 32));
  check("rect 32 rays area 31.3, IoU 0.98", near(a32, 31.3, 0.05) && near(a32 / 32, 0.98, 0.005), a32.toFixed(3));
  const cov = [8, 16, 32].map(n => G.polyArea(G.ellipseRays(6, 3, n)) / (Math.PI * 18));
  check("ellipse 6x3 coverage 85.4/95.5/98.8 %", near(cov[0], 0.854, 5e-4) && near(cov[1], 0.955, 5e-4) && near(cov[2], 0.988, 5e-4), cov.map(c => c.toFixed(4)).join(" "));
  check("ellipse n=8 area 48.3", near(G.polyArea(G.ellipseRays(6, 3, 8)), 48.3, 0.05));
  check("given rays 5,4,3,4,... area 45.3", near(G.polyArea([5, 4, 3, 4, 5, 4, 3, 4]), 45.25, 0.01));
  for (const [n, c] of [[8, 0.900], [32, 0.994]]) {
    const r = new Array(n).fill(1);
    check(`disc n-gon coverage n=${n}`, near(G.polyArea(r) / Math.PI, c, 5e-4) && near(G.polyArea(r) / Math.PI, n / (2 * Math.PI) * Math.sin(2 * Math.PI / n), 1e-12));
  }
  check("4 equal rays: square of area 2r^2", near(G.polyArea([3, 3, 3, 3]), 18, 1e-12));
}

// ---- rays on the pixel grid ---------------------------------------------------------------
{
  const rect = shapeMask("rect");   // pixels 12..51 x 22..41: [12,52] x [22,42]
  const r = G.raysAt(rect, W, W, 32, 32, 8);
  check("pixel rect rays 20, 14.14, 10", [20, 10 * Math.SQRT2, 10, 10 * Math.SQRT2, 20, 10 * Math.SQRT2, 10, 10 * Math.SQRT2].every((v, k) => near(r[k], v, 1e-9)), Array.from(r).map(v => v.toFixed(3)).join(","));
  const f = G.fitAt(rect, W, W, 32, 32, 8);
  check("pixel rect: polygon 600 px, IoU 0.75", f.pix.length === 600 && near(f.iou, 0.75, 1e-12), `${f.pix.length} ${f.iou}`);
  // brute-force ray marching agrees with the exact traversal
  const disc = shapeMask("disc");
  const R = rng(3);
  let worst = 0;
  for (let t = 0; t < 200; t++) {
    const cx = 20 + 24 * R.unif(), cy = 20 + 24 * R.unif();
    if (!disc[Math.floor(cy) * W + Math.floor(cx)]) continue;
    const a = 2 * Math.PI * R.unif(), dx = Math.cos(a), dy = Math.sin(a);
    let s = 0; while (disc[Math.floor(cy + s * dy) * W + Math.floor(cx + s * dx)]) s += 1e-3;
    worst = Math.max(worst, Math.abs(G.rayLength(disc, W, W, cx, cy, dx, dy) - s));
  }
  check("ray traversal = fine marching", worst < 2e-3, worst.toExponential(2));
  const rd = G.raysAt(disc, W, W, 32, 32, 32);
  check("disc rays from the centre = radius (±1 px)", Array.from(rd).every(v => near(v, 18, 1)), Array.from(rd).map(v => v.toFixed(1)).join(","));
  check("start outside: rays 0", G.raysAt(disc, W, W, 2, 2, 8).every(v => v === 0));
}

// ---- polygon pixels, IoU, objectness ----------------------------------------------------
{
  const R = rng(5);
  let okArea = true, okSym = true;
  for (let t = 0; t < 40; t++) {
    const n = [5, 8, 16, 32][t % 4];
    const ra = Float64Array.from({ length: n }, () => 4 + 10 * R.unif()), rb = Float64Array.from({ length: n }, () => 4 + 10 * R.unif());
    const pa = G.polygonPixels(32.3, 31.7, ra, W, W), pb = G.polygonPixels(30.1, 33.2, rb, W, W);
    if (Math.abs(pa.length - G.polyArea(ra)) > 0.08 * G.polyArea(ra) + 4) okArea = false;
    if (!near(G.pixelIou(pa, pb, W * W), G.pixelIou(pb, pa, W * W), 1e-12)) okSym = false;
    if (G.pixelIou(pa, pa, W * W) !== 1) okSym = false;
  }
  check("polygon pixel count ≈ triangle-rule area", okArea);
  check("polygon IoU symmetric, 1 for identical", okSym);
  const ob = G.objectness(shapeMask("ellipse"), W, W);
  let mx = 0, outside = 0;
  const m = shapeMask("ellipse");
  for (let i = 0; i < ob.length; i++) { mx = Math.max(mx, ob[i]); if (!m[i] && ob[i]) outside++; }
  check("objectness: max 1, 0 outside", near(mx, 1, 1e-6) && outside === 0);
}

// ---- star-convexity of the shape presets (best centre) ----------------------------------
{
  const best: Record<string, number> = {};
  for (const id of ["disc", "ellipse", "rect", "L", "C", "ring", "rod", "star"]) best[id] = G.bestCentre(shapeMask(id), W, W, 32)!.iou;
  check("convex shapes: best IoU ≥ 0.97", best.disc >= 0.97 && best.ellipse >= 0.97 && best.rect >= 0.97, JSON.stringify(best));
  check("L from its corner ≥ 0.9", best.L >= 0.9, String(best.L));
  check("C, ring, bent rod: no centre works (< 0.8)", best.C < 0.8 && best.ring < 0.8 && best.rod < 0.8, `${best.C} ${best.ring} ${best.rod}`);
  const armEnd = G.fitAt(shapeMask("L"), W, W, 19.5, 12.5, 32).iou;
  check("L from an arm's end < 0.7", armEnd < 0.7, String(armEnd));
  const st = shapeMask("star");
  const f5 = G.fitAt(st, W, W, 32, 32, 5), f10 = G.fitAt(st, W, W, 32, 32, 10);
  check("star, 5 rays on the tips: polygon sticks out", f5.extra > 50, `extra ${f5.extra}`);
  check("star, 10 rays: tips and notches, IoU > 0.9", f10.iou > 0.9, String(f10.iou));
}

// ---- polygon NMS --------------------------------------------------------------------------
{
  // exercise B4: the given IoU table
  const names = ["P1", "P2", "P3", "P4", "P5", "P6"], prob = [0.92, 0.85, 0.80, 0.62, 0.55, 0.35];
  const tab: Record<string, number> = { "P1P2": 0.71, "P1P3": 0.18, "P1P4": 0.05, "P1P5": 0.33, "P2P3": 0.22, "P2P5": 0.41, "P3P4": 0.46, "P3P6": 0.12, "P4P6": 0.58 };
  const iou = (i: number, j: number) => tab[names[i] + names[j]] ?? tab[names[j] + names[i]] ?? 0;
  const run = (pmin: number, t: number) => {
    const idx = prob.map((p, i) => i).filter(i => prob[i] >= pmin);
    return nmsGeneric(idx.length, idx.map(i => prob[i]), (a, b) => iou(idx[a], idx[b]), t).keep.map(k => names[idx[k]]).join(",");
  };
  check("B4a (≥0.5, NMS 0.3): P1,P3", run(0.5, 0.3) === "P1,P3", run(0.5, 0.3));
  check("B4b (≥0.5, NMS 0.5): P1,P3,P4,P5", run(0.5, 0.5) === "P1,P3,P4,P5", run(0.5, 0.5));
  check("B4c (≥0.3, NMS 0.7): P1,P3,P4,P5,P6", run(0.3, 0.7) === "P1,P3,P4,P5,P6", run(0.3, 0.7));
  // bounding-box pre-check gives the same result as all pairwise IoUs
  const R = rng(9), cands: G.Candidate[] = [];
  for (let i = 0; i < 60; i++) {
    const rays = Float64Array.from({ length: 16 }, () => 3 + 5 * R.unif());
    cands.push(G.makeCandidate(10 + 44 * R.unif(), 10 + 44 * R.unif(), R.unif(), rays, W, W));
  }
  for (const t of [0.05, 0.3, 0.6]) {
    const a = G.polygonNms(cands, t, W * W);
    const b = nmsGeneric(cands.length, cands.map(c => c.score), (i, j) => G.pixelIou(cands[i].pix, cands[j].pix, W * W), t);
    check(`polygon NMS with box pre-check = brute force (t=${t})`, a.keep.join() === b.keep.join(), `${a.keep.length} vs ${b.keep.length}`);
    check(`pre-check skipped pairs (t=${t})`, a.iouComputed < cands.length * (cands.length - 1) / 2);
  }
}

// ---- simulated decoding (decode mode presets) ----------------------------------------------
{
  const sc = S.makeScene();
  check("scene: 10 nuclei, 3 debris", sc.nTrue === 10 && sc.nAll === 13);
  const sim = S.simulate(32, 0.3, 0.2), sim2 = S.simulate(32, 0.3, 0.2);
  check("simulation cached/deterministic", sim === sim2);
  const d = (p: number, t: number) => S.decode(sim, p, t);
  const def = d(0.5, 0.3);
  check("defaults (0.5, 0.3): all 10 found, F1 1", def.scores.tp === 10 && def.scores.fp === 0 && def.scores.fn === 0, JSON.stringify(def.scores));
  const strong = d(0.5, 0.02);
  check("NMS 0.02: a touching neighbour deleted (FN)", strong.scores.fn >= 1 && strong.scores.fp === 0, JSON.stringify(strong.scores));
  check("deleted neighbours are from touching pairs", strong.missed.every(l => [1, 2, 7, 8].includes(l)), strong.missed.join());
  const weak = d(0.5, 0.8);
  check("NMS 0.8: duplicates (FP), nothing missed", weak.scores.fp >= 3 && weak.scores.fn === 0, JSON.stringify(weak.scores));
  const low = d(0.3, 0.3);
  check("threshold 0.3: three debris specks (FP)", low.scores.fp === 3 && low.scores.fn === 0, JSON.stringify(low.scores));
  const high = d(0.7, 0.3);
  check("threshold 0.7: the dim nucleus (4) lost", high.scores.fn === 1 && high.missed[0] === 4, high.missed.join());
  check("candidates do not depend on the NMS threshold", strong.above.length === weak.above.length && def.above.length === strong.above.length);
  check("labels: kept polygons painted", def.labels.some(v => v === def.kept.length) && !def.labels.some(v => v > def.kept.length));
  console.log(`  decode: ${def.above.length} candidates at 0.5 → ${def.kept.length}; NMS 0.02 FN ${strong.scores.fn}; NMS 0.8 FP ${weak.scores.fp}; 0.3 FP ${low.scores.fp}; 0.7 FN ${high.scores.fn}`);
}

console.log(`stardist: ${passes}/${passes + fails} checks passed`);
if (fails) (globalThis as unknown as { process: { exit(c: number): void } }).process.exit(1);
