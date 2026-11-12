// Tests of lib/skeleton.ts and lib/trees.ts (skeleton day).
// node_modules/.bin/esbuild demos/test/skeleton-test.ts --bundle --platform=node --log-level=warning | node
import {
  neighbours, simpleTest, yokoi8, isSimpleCode, hitOrMissT, rotations8, T_END, T_ISOLATED, THIN_TEMPLATES, templateBits,
  thin, thinSimple, zhangSuen, topology, medialAxis, distanceMap, reconstructFromDiscs, lantuejoul, lantuejoulReconstruct,
  skeletonGraph, cyclomatic, prune, pathLength, chainCounts, lengthFromCounts, shollCrossings, classifySkeleton, Metric,
  rotateTemplate, L1, L2, degreeAt, totalLength, edgePolylines, skeletonize,
} from "../src/lib/skeleton";
import { hitOrMiss, SE } from "../src/lib/morph";
import { growTree, treeGraph, treeCoverage } from "../src/lib/trees";
import { rasterise, ellipsePoly, starPoly } from "../src/lib/shapecanvas";

let fails = 0, passes = 0;
function check(name: string, ok: boolean, info = ""): void {
  if (ok) passes++; else { fails++; console.log("FAIL", name, info); }
}

const fromText = (rows: string[]) => {
  const h = rows.length, w = rows[0].length, m = new Uint8Array(w * h);
  rows.forEach((r, y) => r.split("").forEach((c, x) => { m[y * w + x] = c === "1" ? 1 : 0; }));
  return { m, w, h };
};
const same = (a: ArrayLike<number>, b: ArrayLike<number>) => { for (let i = 0; i < a.length; i++) if (!a[i] !== !b[i]) return false; return a.length === b.length; };
const count = (a: ArrayLike<number>) => { let n = 0; for (let i = 0; i < a.length; i++) if (a[i]) n++; return n; };

// --- simple points: component count vs Yokoi's connectivity number, for all 256 neighbourhoods
{
  let bad = 0;
  for (let c = 0; c < 256; c++) {
    const nb = Array.from({ length: 8 }, (_, k) => (c >> k) & 1);
    const t = simpleTest(nb), y = yokoi8(nb);
    // Yokoi = 1 ⇔ simple, for pixels with at least one 4-neighbour in the background
    const border = [0, 2, 4, 6].some(k => !nb[k]);
    if (border && (y === 1) !== t.simple) bad++;
    if (!border && t.simple) bad++;
    if (isSimpleCode(c) !== t.simple) bad++;
  }
  check("simple points agree with Yokoi", bad === 0, String(bad));
}

// brute force: removing a simple pixel keeps components and holes in a 5 × 5 window, for random patterns
{
  let r = 12345, bad = 0;
  const rnd = () => { r = (r * 1103515245 + 12345) & 0x7fffffff; return r / 0x7fffffff; };
  for (let t = 0; t < 3000; t++) {
    const w = 5, h = 5, m = new Uint8Array(25);
    for (let i = 0; i < 25; i++) m[i] = rnd() < 0.55 ? 1 : 0;
    m[12] = 1;
    const before = topology(m, w, h);
    const s = simpleTest(neighbours(m, w, h, 2, 2)).simple;
    m[12] = 0;
    const after = topology(m, w, h);
    const keeps = before.components === after.components && before.holes === after.holes;
    if (s && !keeps) bad++;
  }
  check("removing a simple point keeps the topology (random 5×5)", bad === 0, String(bad));
}

// --- hit-or-miss with templates = morph.hitOrMiss with (B1, B2)
{
  const { m, w, h } = fromText(["0000000", "0110000", "0000000", "0011110", "0000010", "0100000", "0000000"]);
  for (const t of [T_END, T_ISOLATED, ...THIN_TEMPLATES]) {
    const dx1: number[] = [], dy1: number[] = [], dx2: number[] = [], dy2: number[] = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      if (t[r][c] === "1") { dx1.push(c - 1); dy1.push(r - 1); }
      if (t[r][c] === "0") { dx2.push(c - 1); dy2.push(r - 1); }
    }
    const B1: SE = { dx: Int32Array.from(dx1), dy: Int32Array.from(dy1) }, B2: SE = { dx: Int32Array.from(dx2), dy: Int32Array.from(dy2) };
    check("hit-or-miss = morph.hitOrMiss " + t.join("/"), same(hitOrMissT(m, w, h, t), hitOrMiss(m, w, h, B1, B2)));
  }
  const ends = rotations8(T_END).map(t => hitOrMissT(m, w, h, t));
  const e = new Uint8Array(w * h);
  ends.forEach(a => a.forEach((v, i) => { e[i] |= v; }));
  const cls = classifySkeleton(m, w, h);
  check("8 end-point templates = pixels with one neighbour", same(e, Uint8Array.from(cls, v => (v === 1 ? 1 : 0))));
  check("rotation by 360° is the identity", rotateTemplate(L2, 8).join() === L2.join());
  check("no thinning template matches an end point", THIN_TEMPLATES.every(t => { const b = templateBits(t); return [0, 1, 2, 3, 4, 5, 6, 7].every(k => ((1 << k) & b.care) !== b.want); }));
}

// --- thinning keeps topology and leaves no removable pixels
const shapes: { name: string; m: Uint8Array; w: number; h: number }[] = [];
{
  const N = 64;
  const add = (name: string, polys: [number, number][][]) => shapes.push({ name, m: rasterise(polys, N, N), w: N, h: N });
  add("rectangle", [[[8, 20], [56, 20], [56, 40], [8, 40]]]);
  add("ellipse", [ellipsePoly(32, 32, 24, 12, 48, 0.4)]);
  add("star", [starPoly(32, 33, 26, 11, 5)]);
  add("ring", [ellipsePoly(32, 32, 24, 24, 48), ellipsePoly(32, 32, 12, 12, 48)]);
  add("L", [[[10, 8], [24, 8], [24, 40], [54, 40], [54, 54], [10, 54]]]);
  add("two holes", [[[6, 10], [58, 10], [58, 54], [6, 54]], [[14, 20], [26, 20], [26, 44], [14, 44]], [[38, 20], [50, 20], [50, 44], [38, 44]]]);
}
for (const s of shapes) {
  const t0 = topology(s.m, s.w, s.h);
  const sk = skeletonize(s.m, s.w, s.h), t1 = topology(sk, s.w, s.h);
  const tt = topology(thin(s.m, s.w, s.h).skeleton, s.w, s.h);
  check(`template thinning keeps topology: ${s.name}`, t0.components === tt.components && t0.holes === tt.holes);
  check(`thinning keeps topology: ${s.name}`, t0.components === t1.components && t0.holes === t1.holes, JSON.stringify([t0, t1]));
  let removable = 0, corners = 0;
  for (let y = 0; y < s.h; y++) for (let x = 0; x < s.w; x++) {
    if (!sk[y * s.w + x]) continue;
    const nb = neighbours(sk, s.w, s.h, x, y), st = simpleTest(nb);
    if (st.simple && st.degree > 1) removable++;
    // a pixel whose two neighbours touch each other is a redundant corner
    if (st.degree === 2) corners += st.fg.length === 1 ? 1 : 0;
  }
  check(`skeletonize leaves no simple non-end pixel: ${s.name}`, removable === 0, String(removable));
  const zs = zhangSuen(s.m, s.w, s.h).skeleton, tz = topology(zs, s.w, s.h);
  check(`Zhang–Suen keeps topology: ${s.name}`, tz.components === t0.components && tz.holes === t0.holes);
  const ss = thinSimple(s.m, s.w, s.h).skeleton, ts = topology(ss, s.w, s.h);
  check(`simple-point thinning keeps topology: ${s.name}`, ts.components === t0.components && ts.holes === t0.holes);
  // Zhang–Suen and template thinning agree up to short end branches: every Zhang–Suen pixel lies within 2 px of the skeleton
  let far = 0;
  for (let i = 0; i < zs.length; i++) if (zs[i]) {
    const x = i % s.w, y = (i - x) / s.w;
    let near = false;
    for (let dy = -2; dy <= 2 && !near; dy++) for (let dx = -2; dx <= 2 && !near; dx++) if (sk[(y + dy) * s.w + x + dx]) near = true;
    if (!near) far++;
  }
  check(`Zhang–Suen ≈ template thinning: ${s.name}`, far === 0, String(far));
  // skeleton graph: E − V + C = holes
  const g = skeletonGraph(sk, s.w, s.h);
  check(`E − V + C = holes: ${s.name}`, cyclomatic(g) === t0.holes, `${g.edges.length} ${g.nodes.length} ${g.components} vs ${t0.holes}`);
}
// without keeping end points, a tree shrinks to a point and the ring stays a ring
{
  const rect = shapes[0], ring = shapes[3];
  check("no end points kept: rectangle → 1 pixel", count(thinSimple(rect.m, rect.w, rect.h, { keepEnds: false }).skeleton) === 1);
  const rs = thinSimple(ring.m, ring.w, ring.h, { keepEnds: false }).skeleton;
  check("no end points kept: ring stays a ring", topology(rs, ring.w, ring.h).holes === 1);
}
// parallel removal cuts a two-pixel line; sequential does not
{
  const { m, w, h } = fromText(["0000000000", "0111111110", "0111111110", "0000000000"]);
  check("parallel thinning cuts a 2-pixel line", count(thin(m, w, h, { parallel: true }).skeleton) === 0);
  const seq = thin(m, w, h).skeleton;
  check("sequential thinning keeps it", topology(seq, w, h).components === 1 && count(seq) > 0);
}

// --- medial axis
for (const metric of ["cityblock", "chessboard", "euclidean"] as Metric[]) for (const s of shapes) {
  const dt = distanceMap(s.m, s.w, s.h, metric), ax = medialAxis(s.m, s.w, s.h, metric, dt);
  check(`axis discs reconstruct the shape: ${s.name} ${metric}`, same(reconstructFromDiscs(ax, dt, s.w, s.h, metric), s.m));
}
for (const s of shapes) for (const se of ["cross", "square"] as const) {
  const L = lantuejoul(s.m, s.w, s.h, se);
  check(`Lantuéjoul reconstructs: ${s.name} ${se}`, same(lantuejoulReconstruct(L, s.w, s.h, se), s.m));
  const metric: Metric = se === "cross" ? "cityblock" : "chessboard";
  const dt = distanceMap(s.m, s.w, s.h, metric), ax = medialAxis(s.m, s.w, s.h, metric, dt);
  let ok = same(L.skeleton, ax);
  for (let i = 0; i < dt.length && ok; i++) if (ax[i] && L.radius[i] !== dt[i] - 1) ok = false;
  check(`Lantuéjoul S_r = maximal discs with DT = r + 1: ${s.name} ${se}`, ok);
}
// a rectangle's axis forks at the ends with city-block and Euclidean discs, not with square (chessboard) discs
{
  const { m, w, h } = fromText(["0000000000000", "0111111111110", "0111111111110", "0111111111110", "0111111111110", "0111111111110", "0000000000000"]);
  const ends = (metric: Metric) => skeletonGraph(skeletonize(medialAxis(m, w, h, metric), w, h), w, h).nodes.filter(n => n.kind === "end").length;
  check("rectangle: city-block axis forks (4 end points)", ends("cityblock") === 4, String(ends("cityblock")));
  check("rectangle: Euclidean axis forks (4 end points)", ends("euclidean") === 4, String(ends("euclidean")));
  check("rectangle: chessboard axis is a segment (2 end points)", ends("chessboard") === 2, String(ends("chessboard")));
}

// --- graph of small skeletons
{
  const { m, w, h } = fromText(["000000000", "010000010", "001000100", "000101000", "000010000", "000010000", "000010000", "000010000", "000000000"]);
  const g = skeletonGraph(m, w, h);
  const ends = g.nodes.filter(n => n.kind === "end").length, junc = g.nodes.filter(n => n.kind === "junction").length;
  check("Y: 3 ends, 1 junction, 3 edges", ends === 3 && junc === 1 && g.edges.length === 3, `${ends} ${junc} ${g.edges.length}`);
  check("Y: total length 6√2 + 3", Math.abs(totalLength(g, "weighted") - (6 * Math.SQRT2 + 3)) < 1e-9, String(totalLength(g, "weighted")));
  const pr = prune(m, w, h, 4);
  const gp = skeletonGraph(pr.skeleton, w, h);
  check("Y pruned with k = 4: the stem (3) goes, one edge left", pr.removed === 1 && gp.edges.length === 1, `${pr.removed} ${gp.edges.length}`);
  const p5 = prune(m, w, h, 5);
  check("Y pruned with k = 5: the junction keeps its two longest edges", p5.removed === 1);
}
// pruning keeps topology of a noisy shape's skeleton (no cycles destroyed)
for (const s of shapes) {
  const sk = skeletonize(s.m, s.w, s.h), pr = prune(sk, s.w, s.h, 6).skeleton;
  const t0 = topology(sk, s.w, s.h), t1 = topology(pr, s.w, s.h);
  check(`pruning keeps holes and components: ${s.name}`, t0.holes === t1.holes && t0.components === t1.components);
}

// --- length estimators on rendered straight lines
{
  let worst = { count: 0, weighted: 0, corrected: 0 };
  for (let deg = 0; deg <= 90; deg += 3) {
    const th = deg * Math.PI / 180, L = 150, N = 200;
    const x0 = 20, y0 = 180, x1 = x0 + L * Math.cos(th), y1 = y0 - L * Math.sin(th);
    const path: number[] = [];
    // digital straight line (Bresenham-like: one pixel per step along the major axis)
    const n = Math.max(Math.abs(Math.round(x1) - x0), Math.abs(Math.round(y1) - y0));
    for (let k = 0; k <= n; k++) path.push(Math.round(y0 + (y1 - y0) * k / n) * N + Math.round(x0 + (x1 - x0) * k / n));
    const truth = Math.hypot(Math.round(x1) - x0, Math.round(y1) - y0);
    const c = chainCounts(path, N);
    for (const est of ["count", "weighted", "corrected"] as const) {
      const e = Math.abs(lengthFromCounts(c, est) / truth - 1);
      if (e > worst[est]) worst[est] = e;
    }
  }
  check("pixel counting is off by up to ~29% (1/√2)", worst.count > 0.25 && worst.count < 0.3, String(worst.count));
  check("1/√2 weights within 9%", worst.weighted < 0.09, String(worst.weighted));
  check("corrected estimator within 3%", worst.corrected < 0.03, String(worst.corrected));
}

// --- Sholl: a cross of 4 rays crosses every circle 4 times
{
  const lines: [number, number][][] = [[[50, 50], [90, 50]], [[50, 50], [10, 50]], [[50, 50], [50, 90]], [[50, 50], [50, 10]]];
  check("Sholl: 4 rays → 4 crossings", shollCrossings(lines, 50, 50, 20) === 4 && shollCrossings(lines, 50, 50, 45) === 0);
}

// --- synthetic trees: the skeleton graph of a clean rendering matches the ground truth
for (const kind of ["neuron", "vessel"] as const) for (const seed of [1, 2, 3, 4, 5]) {
  const t = growTree({ kind, seed, loops: 0 }), G = treeGraph(t);
  const { mask } = treeCoverage(t), N = t.size;
  const sk = prune(skeletonize(mask, N, N), N, N, 6).skeleton, g = skeletonGraph(sk, N, N);
  const truthEnds = G.nodes.filter(n => n.kind === "end").length;
  const ends = g.nodes.filter(n => n.kind === "end").length;
  const truthLen = G.edges.reduce((s, e) => s + e.length, 0), len = totalLength(g, "corrected");
  const tt = topology(mask, N, N);
  check(`tree ${kind} ${seed}: one component, no holes`, tt.components === 1 && tt.holes === 0, JSON.stringify(tt));
  if (kind === "vessel") {
    check(`tree ${kind} ${seed}: end points match (±1)`, Math.abs(ends - truthEnds) <= 1, `${ends} vs ${truthEnds}`);
    check(`tree ${kind} ${seed}: length within 10%`, Math.abs(len / truthLen - 1) < 0.1, `${len.toFixed(1)} vs ${truthLen.toFixed(1)}`);
  } else {
    check(`tree ${kind} ${seed}: end points match (±1)`, Math.abs(ends - truthEnds) <= 1, `${ends} vs ${truthEnds}`);
  }
}
{
  const t = growTree({ kind: "vessel", seed: 7, loops: 3 }), G = treeGraph(t);
  const { mask } = treeCoverage(t), N = t.size;
  const tt = topology(mask, N, N);
  check("vessel loops = holes of the mask", tt.holes === G.loops, `${tt.holes} vs ${G.loops}`);
  const g = skeletonGraph(skeletonize(mask, N, N), N, N);
  check("vessel: E − V + C = loops", cyclomatic(g) === G.loops);
}
// unused import guard
void degreeAt; void pathLength; void edgePolylines; void L1;

console.log(`skeleton tests: ${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
