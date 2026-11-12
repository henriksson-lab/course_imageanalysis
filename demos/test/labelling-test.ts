// Reference checks for the labelling day (label-connectivity/, regionprops/, shape-features/,
// label-measure/): flood fill, two-pass and run-based labelling against lib/label.ts and a
// brute-force reachability; union–find heuristics; bit-quad Euler number; moments of
// digitised ellipses; perimeter estimators; convex hull, rotating calipers and minimum
// enclosing circle against brute force; area opening and the counting frame. Run:
//   node_modules/.bin/esbuild demos/test/labelling-test.ts --bundle --platform=node --log-level=warning | node
import { rng } from "../src/lib/random";
import { label2d, label3d } from "../src/lib/label";
import { floodFrames, twoPassFrames, runFrames } from "../src/morphgrid/mode-label";
import { Forest, Compression } from "../src/unionfind/forest";
import * as RP from "../src/moments/regionprops";
import * as CH from "../src/lib/convexhull";
import { shapeFeatures, fourierOutline, outline } from "../src/moments/features";
import { rasterise, ellipsePoly } from "../src/lib/shapecanvas";
import * as TOP from "../src/connparadox/topology";
import * as RF from "../src/regionfilter/pipeline";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

function randomMask(r: ReturnType<typeof rng>, w: number, h: number, p: number): Uint8Array {
  const m = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) m[i] = r.unif() < p ? 1 : 0;
  return m;
}

/** Same partition up to a permutation of labels (0 must map to 0). */
function samePartition(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  const f = new Map<number, number>(), g = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    if ((a[i] === 0) !== (b[i] === 0)) return false;
    if (a[i] === 0) continue;
    if (f.has(a[i]) && f.get(a[i]) !== b[i]) return false;
    if (g.has(b[i]) && g.get(b[i]) !== a[i]) return false;
    f.set(a[i], b[i]); g.set(b[i], a[i]);
  }
  return true;
}

/** Brute-force components: repeated relaxation of min-label over neighbours. */
function bruteLabels(m: ArrayLike<number>, w: number, h: number, conn: 4 | 8): Int32Array {
  const l = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) l[i] = m[i] ? i + 1 : 0;
  let changed = true;
  while (changed) {
    changed = false;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!l[i]) continue;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        if (conn === 4 && dx && dy) continue;
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (l[j] && l[j] < l[i]) { l[i] = l[j]; changed = true; }
      }
    }
  }
  return l;
}

function lastAux(gen: Iterable<{ aux?: ArrayLike<number> }>): Int32Array {
  let last: ArrayLike<number> | undefined;
  for (const f of gen) if (f.aux) last = f.aux;
  return Int32Array.from(last!);
}

// ---------------------------------------------------------------------------------------
current = "labelling 2D";
{
  const r = rng(11);
  for (let t = 0; t < 60; t++) {
    const w = 3 + Math.floor(r.unif() * 14), h = 3 + Math.floor(r.unif() * 14), p = 0.2 + 0.6 * r.unif();
    const m = randomMask(r, w, h, p);
    for (const conn of [4, 8] as const) {
      const ref = bruteLabels(m, w, h, conn), lib = label2d(m, w, h, conn);
      check(samePartition(ref, lib.data), `lib label2d vs brute force (${w}×${h}, ${conn})`);
      for (const kind of ["bfs", "dfs"] as const) check(samePartition(ref, lastAux(floodFrames(m, w, h, conn, kind))), `flood ${kind} (${w}×${h}, ${conn})`);
      const tp = lastAux(twoPassFrames(m, w, h, conn));
      check(samePartition(ref, tp), `two-pass (${w}×${h}, ${conn})`);
      check(Array.from(tp).every((v, i) => v === lib.data[i]), `two-pass numbering = raster order of first pixel (${conn})`);
      check(samePartition(ref, lastAux(runFrames(m, w, h, conn))), `runs (${w}×${h}, ${conn})`);
    }
  }
}

current = "labelling 3D";
{
  const r = rng(12);
  for (let t = 0; t < 20; t++) {
    const n = 3 + Math.floor(r.unif() * 4), m = randomMask(r, n * n * n, 1, 0.3 + 0.3 * r.unif());
    for (const conn of [6, 18, 26] as const) {
      const ref = TOP.bruteLabels3d(m, n, conn), lib = label3d(m, n, n, n, conn);
      check(samePartition(ref, lib.data), `label3d vs brute force (n=${n}, ${conn})`);
    }
  }
}

current = "union–find";
{
  const r = rng(13);
  for (let t = 0; t < 40; t++) {
    const n = 5 + Math.floor(r.unif() * 30), ops: [number, number][] = [];
    for (let k = 0; k < n; k++) ops.push([Math.floor(r.unif() * n), Math.floor(r.unif() * n)]);
    const parts: string[] = [];
    for (const byRank of [false, true]) for (const c of ["none", "full", "halving"] as Compression[]) {
      const f = new Forest(n, { byRank, compression: c });
      for (const [a, b] of ops) f.union(a, b);
      parts.push(JSON.stringify(f.classes()));
      if (byRank && c === "none") check(f.maxDepth() <= Math.floor(Math.log2(n)), `union by rank: depth ≤ log2 n (n=${n}, depth ${f.maxDepth()})`);
    }
    check(parts.every(p => p === parts[0]), "all heuristics give the same partition");
  }
  // naive chain: union(i, i+1) links i's root under i+1: depth n-1
  const f = new Forest(8, { byRank: false, compression: "none" });
  for (let i = 0; i < 7; i++) f.union(i, i + 1);
  check(f.maxDepth() === 7, `naive chain depth 7 (got ${f.maxDepth()})`);
  const s0 = f.steps; f.find(0); check(f.steps - s0 === 7, "find at the bottom of the chain follows 7 pointers");
  const g = new Forest(8, { byRank: false, compression: "full" });
  for (let i = 0; i < 7; i++) g.union(i, i + 1);
  g.find(0);
  check(g.maxDepth() === 1, "full compression after one find: depth 1");
  const hf = new Forest(8, { byRank: false, compression: "halving" });
  for (let i = 0; i < 7; i++) hf.union(i, i + 1);
  hf.find(0);
  check(hf.depth(0) === 4 && hf.maxDepth() === 4, `halving after one find: depth of 0 is 4 (got ${hf.depth(0)}, max ${hf.maxDepth()})`);
}

current = "Euler number";
{
  const r = rng(14);
  for (let t = 0; t < 80; t++) {
    const w = 3 + Math.floor(r.unif() * 15), h = 3 + Math.floor(r.unif() * 15);
    const m = randomMask(r, w, h, 0.25 + 0.5 * r.unif());
    for (const conn of [4, 8] as const) {
      const comps = label2d(m, w, h, conn).count, holes = RP.countHoles(m, w, h, conn);
      check(RP.eulerNumber(m, w, h, conn) === comps - holes, `bit-quads = components − holes (${conn}): ${RP.eulerNumber(m, w, h, conn)} vs ${comps} − ${holes}`);
    }
  }
}

current = "moments";
{
  for (const [a, b, ang] of [[30, 12, 0.3], [25, 10, -0.9], [20, 19, 1.2], [40, 8, 1.5]]) {
    const N = 128, poly = ellipsePoly(64, 64, a, b, 200, ang);
    const m = rasterise([poly], N, N);
    const p = RP.maskProps(m, N, N)!;
    // a solid ellipse has variance a²/4 along its major axis
    check(Math.abs(Math.sqrt(p.l1) * 2 - a) < 0.02 * a + 0.3, `semi-axis a = 2√λ1: ${2 * Math.sqrt(p.l1)} vs ${a}`);
    check(Math.abs(Math.sqrt(p.l2) * 2 - b) < 0.02 * b + 0.3, `semi-axis b = 2√λ2: ${2 * Math.sqrt(p.l2)} vs ${b}`);
    // ellipsePoly rotates counter-clockwise on screen; orientation is reported the same way
    const want = ((ang * 180 / Math.PI + 90) % 180 + 180) % 180 - 90;
    if (a / b > 1.2) check(Math.abs(((p.orientation - want + 90) % 180 + 180) % 180 - 90) < 1.5, `orientation ${p.orientation} vs ${want}`);
    check(Math.abs(p.cx - 63.5) < 0.3 && Math.abs(p.cy - 63.5) < 0.3, "centroid at the centre (pixel centres at integers)");
  }
  // raw/central moment consistency on a random mask
  const r = rng(15), w = 9, h = 7, m = randomMask(r, w, h, 0.5);
  const A = RP.rawMoment(m, w, h, 0, 0), m10 = RP.rawMoment(m, w, h, 1, 0), m01 = RP.rawMoment(m, w, h, 0, 1);
  const p = RP.maskProps(m, w, h)!;
  check(close(p.area, A) && close(p.cx, m10 / A) && close(p.cy, m01 / A), "centroid = m10/m00, m01/m00");
  check(close(p.mu20, RP.rawMoment(m, w, h, 2, 0) - m10 * m10 / A, 1e-9), "μ20 = m20 − m10²/m00");
  check(close(p.mu11, RP.rawMoment(m, w, h, 1, 1) - m10 * m01 / A, 1e-9), "μ11 = m11 − m10 m01/m00");
}

current = "perimeter";
{
  // digital discs: edge count / 2πr → 4/π; chain code within a few %; Crofton within 2 %
  for (const R of [40, 80, 160]) {
    const N = 2 * R + 6, m = new Uint8Array(N * N), c = N / 2;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if ((x + 0.5 - c) ** 2 + (y + 0.5 - c) ** 2 <= R * R) m[y * N + x] = 1;
    const e = RP.edgePerimeter(m, N, N) / (2 * Math.PI * R), ch = RP.chainPerimeter(m, N, N) / (2 * Math.PI * R), cr = RP.croftonPerimeter(m, N, N) / (2 * Math.PI * R);
    check(Math.abs(e - 4 / Math.PI) < 0.01, `edge count / 2πr = ${e.toFixed(4)} → 4/π (r=${R})`);
    check(ch > 1 && ch < 1.08, `chain code / 2πr = ${ch.toFixed(4)} (r=${R})`);
    check(Math.abs(cr - 1) < 0.02, `Crofton / 2πr = ${cr.toFixed(4)} (r=${R})`);
  }
  // simple shapes by hand: a 4 × 4 square
  const sq = new Uint8Array(36); for (let y = 1; y < 5; y++) for (let x = 1; x < 5; x++) sq[y * 6 + x] = 1;
  check(RP.edgePerimeter(sq, 6, 6) === 16, "4×4 square: 16 edges");
  check(close(RP.chainPerimeter(sq, 6, 6), 12), "4×4 square: chain code through centres = 12");
  // ring: chain code includes the hole
  const ring = new Uint8Array(25); for (let y = 1; y < 4; y++) for (let x = 1; x < 4; x++) ring[y * 5 + x] = 1; ring[12] = 0;
  check(close(RP.chainPerimeter(ring, 5, 5), 8 + 4 * Math.SQRT2), `3×3 ring: chain 8 (outer) + 4√2 (around the hole), got ${RP.chainPerimeter(ring, 5, 5)}`);
  check(RP.edgePerimeter(ring, 5, 5) === 16, "3×3 ring: 12 + 4 edges");
}

current = "convex hull";
{
  const r = rng(16);
  for (let t = 0; t < 60; t++) {
    const n = 3 + Math.floor(r.unif() * 40);
    const pts: CH.Pt[] = Array.from({ length: n }, () => [Math.round(r.unif() * 20), Math.round(r.unif() * 20)]);
    const hull = CH.convexHull(pts), brute = CH.bruteForceHullVertices(pts);
    const key = (ps: CH.Pt[]) => ps.map(p => p.join(",")).sort().join(";");
    if (hull.length >= 3) {
      check(key(hull) === key(brute), `hull vertices = brute force (n=${n})`);
      check(CH.shoelaceArea(hull) > 0, "hull counter-clockwise (y up): positive area");
      const cal = CH.rotatingCalipers(hull), bf = CH.bruteForceFeret(pts);
      check(close(cal.minWidth, bf.minWidth, 1e-9), `min Feret ${cal.minWidth} vs ${bf.minWidth}`);
      check(close(cal.maxDiameter, bf.maxDiameter, 1e-9), `max Feret ${cal.maxDiameter} vs ${bf.maxDiameter}`);
      // w(θ) ≥ min width everywhere and equals it at minAngle
      const prof = CH.widthProfile(hull, 360);
      check(Math.min(...prof.width) >= cal.minWidth - 1e-9, "w(θ) ≥ min width");
      check(close(CH.feretWidth(hull, cal.minAngle), cal.minWidth, 1e-9), "w(minAngle) = min width");
      const mec = CH.minEnclosingCircle(pts, 3);
      check(pts.every(p => Math.hypot(p[0] - mec.x, p[1] - mec.y) <= mec.r + 1e-7), "enclosing circle contains all points");
      check(mec.r <= cal.maxDiameter / Math.sqrt(3) + 1e-9 && mec.r >= cal.maxDiameter / 2 - 1e-9, "D/2 ≤ r ≤ D/√3 (Jung)");
    }
  }
  // corner hull of a digital disc → πr²
  const R = 100, N = 2 * R + 4, m = new Uint8Array(N * N), c = N / 2;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if ((x + 0.5 - c) ** 2 + (y + 0.5 - c) ** 2 <= R * R) m[y * N + x] = 1;
  const ha = Math.abs(CH.shoelaceArea(CH.maskHull(m, N, N)));
  check(Math.abs(ha / (Math.PI * R * R) - 1) < 0.01, `corner hull of a disc / πr² = ${(ha / (Math.PI * R * R)).toFixed(4)}`);
  // a 1-pixel line has centre-hull area 0, corner-hull area = its length
  const line = new Uint8Array(30); for (let x = 1; x < 9; x++) line[10 + x] = 1;
  check(Math.abs(CH.shoelaceArea(CH.maskHull(line, 10, 3, "centres"))) === 0, "line: centre hull area 0");
  check(Math.abs(CH.shoelaceArea(CH.maskHull(line, 10, 3, "corners"))) === 8, "line: corner hull area 8");
}

current = "shape features";
{
  // analytic disc polygon: form factor 4πA/P² → 1
  const poly = ellipsePoly(0, 0, 50, 50, 720);
  let A = 0, P = 0;
  for (let i = 0; i < poly.length; i++) { const p = poly[i], q = poly[(i + 1) % poly.length]; A += p[0] * q[1] - q[0] * p[1]; P += Math.hypot(q[0] - p[0], q[1] - p[1]); }
  A = Math.abs(A) / 2;
  check(Math.abs(4 * Math.PI * A / (P * P) - 1) < 1e-4, "form factor of a disc polygon = 1");
  // a digital disc: edge-count form factor ≈ π/4; solidity ≈ 1; inscribed radius ≈ r
  const R = 30, N = 70, m = new Uint8Array(N * N), c = N / 2;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if ((x + 0.5 - c) ** 2 + (y + 0.5 - c) ** 2 <= R * R) m[y * N + x] = 1;
  const f = shapeFeatures(m, N, N)!;
  // edge count overestimates the perimeter by 4/π, so the form factor is (π/4)² = π²/16 ≈ 0.62
  check(Math.abs(f.ffEdges - Math.PI ** 2 / 16) < 0.02, `disc form factor (edges) ${f.ffEdges.toFixed(3)} ≈ π²/16`);
  check(f.solidity > 0.97 && f.solidity <= 1, `disc solidity ${f.solidity.toFixed(3)}`);
  check(Math.abs(f.inscribedR - R) < 1.5, `disc inscribed radius ${f.inscribedR.toFixed(2)} ≈ ${R}`);
  check(Math.abs(f.calipers.maxDiameter - 2 * R) < 1.5 && Math.abs(f.calipers.minWidth - 2 * R) < 1.5, "disc Feret min ≈ max ≈ 2r");
  check(Math.abs(f.compactMoments - 1) < 0.02, `disc second-moment compactness ${f.compactMoments.toFixed(3)} ≈ 1`);
  check(Math.abs(f.enclosing.r - R) < 1, `disc enclosing radius ${f.enclosing.r.toFixed(2)}`);
  // Fourier outline: all coefficients reproduce the outline
  const out = outline(m, N, N, 64);
  const fo = fourierOutline(out, 32);
  check(out.every((p, i) => Math.hypot(p[0] - fo.smooth[i][0], p[1] - fo.smooth[i][1]) < 1e-6), "Fourier outline with all coefficients = outline");
  const f1 = fourierOutline(out, 1);
  const rr = f1.smooth.map(p => Math.hypot(p[0] - c, p[1] - c));
  check(Math.max(...rr) - Math.min(...rr) < 0.5, "k = 1 gives a circle (ellipse) for a disc");
}

current = "topology (connparadox)";
{
  // closed diagonal curve: 8/8 leaks, 4/4 breaks, 8/4 and 4/8 consistent
  const g = TOP.PARADOX_GRIDS.diamond, w = g[0].length, h = g.length;
  const m = new Uint8Array(w * h);
  g.forEach((row, y) => [...row].forEach((ch, x) => { m[y * w + x] = ch === "1" ? 1 : 0; }));
  const v84 = TOP.analyse2d(m, w, h, 8, 4), v88 = TOP.analyse2d(m, w, h, 8, 8), v44 = TOP.analyse2d(m, w, h, 4, 4), v48 = TOP.analyse2d(m, w, h, 4, 8);
  check(v84.fg === 1 && v84.bg === 2 && v84.enclosed === 1, `8/4: one curve, inside and outside (${JSON.stringify(v84)})`);
  check(v88.fg === 1 && v88.bg === 1, "8/8: the inside leaks out");
  check(v44.fg > 1 && v44.bg === 2, "4/4: broken curve, yet the inside is cut off");
  check(v48.fg > 1 && v48.bg === 1, "4/8: broken curve, no inside");
}

current = "label-measure pipeline";
{
  const r = rng(17);
  for (let t = 0; t < 30; t++) {
    const w = 6 + Math.floor(r.unif() * 14), h = 6 + Math.floor(r.unif() * 14), m = randomMask(r, w, h, 0.45);
    const a = 1 + Math.floor(r.unif() * 6);
    const ao = RF.areaOpening(m, w, h, a, 8);
    const lab = bruteLabels(m, w, h, 8), size = new Map<number, number>();
    for (const v of lab) if (v) size.set(v, (size.get(v) ?? 0) + 1);
    check(Array.from(ao).every((v, i) => v === (lab[i] && size.get(lab[i])! >= a ? 1 : 0)), `area opening = brute force (a=${a})`);
  }
  // counting frame (margin 2 → frame [2, 8) × [2, 8)): objects that hit the frame and do
  // not touch the exclusion line (right edge extended up, bottom edge extended left)
  const lab = new Int32Array(100);
  const put = (l: number, ...xy: number[]) => { for (let k = 0; k < xy.length; k += 2) lab[xy[k + 1] * 10 + xy[k]] = l; };
  put(1, 1, 5, 2, 5); put(2, 5, 1, 5, 2); put(3, 7, 5, 8, 5); put(4, 5, 7, 5, 8); put(5, 5, 5); put(6, 0, 0); put(7, 8, 1);
  const keep = RF.countingFrame(lab, 10, 10, 7, 2);
  check(Array.from(keep).slice(1).join("") === "1100100", `counting frame: ${Array.from(keep).slice(1).join("")}`);
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
