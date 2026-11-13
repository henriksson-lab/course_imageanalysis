// Tests for the ray-tracing day (lib/raytrace.ts, lib/bvh.ts, lib/rays2d.ts, lib/montecarlo.ts):
//  - sphere/plane/triangle hits against brute-force sampling along the ray; the stable quadratic
//    against the textbook one in float32 (cancellation); Möller–Trumbore against linalg.solve;
//    ray-cast barycentrics equal the rasteriser's perspective-correct ones;
//  - reflect keeps the length; refract obeys Snell and returns null exactly beyond the critical
//    angle; Fresnel R(0) = R0, R(90°) = 1, Schlick within 0.01 (absolute) of exact for glass below
//    45° (0.04 at any angle); R + T = 1 at every split of the 2D tracer (no power lost beyond the cut-off);
//  - the slab test against sampling; BVH (median, SAH) and grid give the same nearest hit as
//    brute force on random rays over random 2D scenes and a marching-cubes mesh; SAH root cost
//    never worse than "no split";
//  - Monte Carlo: estimators converge, RMS error over 500 repetitions within 10 % of σ/√N;
//    cosine sampling of a Lambertian surface under a constant sky has zero variance; Russian
//    roulette keeps the mean; the three path-tracer sampling modes agree within their error.
// Run: node_modules/.bin/esbuild demos/test/raytracing-test.ts --bundle --platform=node --log-level=warning | node
import * as RT from "../src/lib/raytrace";
import * as B from "../src/lib/bvh";
import * as R2 from "../src/lib/rays2d";
import * as MC from "../src/lib/montecarlo";
import * as M from "../src/lib/mat4";
import { solve } from "../src/lib/linalg";
import { perspectiveBary, barycentric, WinVertex } from "../src/lib/raster";
import { marchingCubes } from "../src/lib/marching";
import { nucleiStack } from "../src/lib/volume";
import { rng } from "../src/lib/random";
import type { Vec3 } from "../src/lib/mat4";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const r = rng(4242);
const U = (a: number, b: number) => a + (b - a) * r.unif();
const randDir = (): Vec3 => M.normalize([r.norm(), r.norm(), r.norm()]);

// ---------------------------------------------------------------- intersections
{
  // brute-force: first sign change of f(t) on a fine grid, refined by bisection
  const firstRoot = (f: (t: number) => number, tmax: number): number => {
    const n = 20000;
    let prev = f(1e-9);
    for (let i = 1; i <= n; i++) {
      const t = tmax * i / n, v = f(t);
      if ((v <= 0) !== (prev <= 0)) {
        let a = tmax * (i - 1) / n, b = t;
        for (let k = 0; k < 60; k++) { const m = (a + b) / 2; if ((f(m) <= 0) === (f(a) <= 0)) a = m; else b = m; }
        return (a + b) / 2;
      }
      prev = v;
    }
    return Infinity;
  };
  let sph = 0, pl = 0;
  for (let it = 0; it < 200; it++) {
    const o: Vec3 = [U(-3, 3), U(-3, 3), U(-3, 3)], c: Vec3 = [U(-2, 2), U(-2, 2), U(-2, 2)], rad = U(0.2, 2);
    const d = it % 2 ? randDir() : M.normalize(M.add(M.sub(c, o), M.scale3(randDir(), U(0, 2.5))));
    const t = RT.hitSphere(o, d, c, rad);
    const tb = firstRoot(s => { const p = RT.pointAt(o, d, s); return (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2 - rad * rad; }, 20);
    if (t < Infinity || tb < Infinity) { sph++; check(Math.abs(t - tb) < 1e-6, `sphere hit ${t} vs sampled ${tb}`); }
    const n = randDir(), p0: Vec3 = [U(-2, 2), U(-2, 2), U(-2, 2)];
    let tp = RT.hitPlane(o, d, p0, n); if (!(tp > 0)) tp = Infinity;
    const tpb = firstRoot(s => M.dot(M.sub(RT.pointAt(o, d, s), p0), n), 1e4);
    if (tp < 1e4 || tpb < Infinity) { pl++; check(Math.abs(tp - tpb) < 1e-5 * Math.max(1, tp), `plane hit ${tp} vs ${tpb}`); }
  }
  check(sph > 30 && pl > 30, `enough sphere (${sph}) and plane (${pl}) hits tested`);

  // stable vs textbook quadratic in float32: b' = 1e4, constant k = 1 (far-away sphere, unit d)
  const f = Math.fround;
  const o: Vec3 = [0, 0, 0], d: Vec3 = [0, 0, -1], c: Vec3 = [0, 0, -1e4], rad = Math.sqrt(1e8 - 1);   // ‖o − c‖² − r² = 1
  const exactSmall = 1 / (1e4 + Math.sqrt(1e8 - 1));   // k / (−b' + √disc) with b' = −1e4 (Vieta)
  const naive = RT.solveSphereNaive(o, d, c, rad, f);
  // stable in float32: q = −(b' + sign(b')√disc), t = k/q
  const bq = f(-1e4), disc = f(f(bq * bq) - 1), q = f(-(bq - f(Math.sqrt(disc)))), stable = f(1 / q);
  check(Math.abs(stable - exactSmall) / exactSmall < 1e-6, `stable float32 small root ${stable} vs ${exactSmall}`);
  check(Math.abs(naive[0] - exactSmall) / exactSmall > 0.5, `naive float32 small root should be badly wrong: ${naive[0]}`);
  const st64 = RT.solveSphere(o, d, c, rad);
  check(near(st64.t1, exactSmall, 1e-9), "stable float64 small root");

  // Möller–Trumbore against a 3 × 3 linear solve
  for (let it = 0; it < 200; it++) {
    const a: Vec3 = [U(-2, 2), U(-2, 2), U(-2, 2)], b: Vec3 = [U(-2, 2), U(-2, 2), U(-2, 2)], cc: Vec3 = [U(-2, 2), U(-2, 2), U(-2, 2)];
    const o2: Vec3 = [U(-3, 3), U(-3, 3), U(-3, 3)], d2 = randDir();
    const mt = RT.solveTriangle(o2, d2, a, b, cc);
    const e1 = M.sub(b, a), e2 = M.sub(cc, a);
    const A = new Float64Array([-d2[0], e1[0], e2[0], -d2[1], e1[1], e2[1], -d2[2], e1[2], e2[2]]);
    const x = solve(A, new Float64Array(M.sub(o2, a)), 3);
    if (!x) continue;
    check(near(mt.t, x[0], 1e-7) && near(mt.u, x[1], 1e-7) && near(mt.v, x[2], 1e-7), `MT (${mt.t}, ${mt.u}, ${mt.v}) vs solve (${x[0]}, ${x[1]}, ${x[2]})`);
    if (RT.insideTriangle(mt.u, mt.v) && mt.t > 0) {
      const p = RT.pointAt(o2, d2, mt.t), q = M.add(a, M.add(M.scale3(e1, mt.u), M.scale3(e2, mt.v)));
      check(M.length(M.sub(p, q)) < 1e-9, "hit point from t equals the point from (u, v)");
    }
  }

  // ray-cast barycentrics = the rasteriser's perspective-correct ones
  const W = 64, H = 48, fov = 50, P = M.perspective(fov * Math.PI / 180, W / H, 0.1, 100);
  for (let it = 0; it < 50; it++) {
    const tri: Vec3[] = [0, 1, 2].map(() => [U(-1.5, 1.5), U(-1, 1), U(-6, -2)] as Vec3);
    const wv: WinVertex[] = tri.map(p => { const c4 = M.transformVec4(P, [p[0], p[1], p[2], 1]); return { x: (c4[0] / c4[3] + 1) * 0.5 * W, y: (c4[1] / c4[3] + 1) * 0.5 * H, z: c4[2] / c4[3], w: c4[3], attr: [] }; });
    const px = U(0, W), py = U(0, H);
    const bs = barycentric(wv[0], wv[1], wv[2], px, py);
    if (!bs) continue;
    const pc = perspectiveBary(wv, bs);
    const ray = RT.cameraRay(0, 0, W, H, { eye: [0, 0, 0], fov }, px, H - py);
    const mt = RT.solveTriangle(ray.o, ray.d, tri[0], tri[1], tri[2]);
    check(near(pc[0], 1 - mt.u - mt.v, 1e-6) && near(pc[1], mt.u, 1e-6) && near(pc[2], mt.v, 1e-6), `perspective-correct (${pc.map(v => v.toFixed(5))}) vs ray cast (${[1 - mt.u - mt.v, mt.u, mt.v].map(v => v.toFixed(5))})`);
  }

  // camera ray for pixel (1, 0) of a 4 × 2 image, fov 90°: direction ∝ (−0.5, 0.5, −1)
  const cr = RT.cameraRay(1, 0, 4, 2, { eye: [0, 0, 0], fov: 90 });
  const ex = M.normalize([-0.5, 0.5, -1]);
  check(M.length(M.sub(cr.d, ex)) < 1e-12, `camera ray ${cr.d}`);
}

// ---------------------------------------------------------------- optics
{
  for (let it = 0; it < 200; it++) {
    const d = randDir(), n = randDir();
    const rf = RT.reflect(d, n);
    check(near(M.length(rf), 1, 1e-12) && near(M.dot(rf, n), -M.dot(d, n), 1e-12), "reflect keeps length, flips the normal part");
    const nf: Vec3 = M.dot(d, n) < 0 ? n : M.scale3(n, -1);
    const n1 = U(1, 2.4), n2 = U(1, 2.4), eta = n1 / n2;
    const t = RT.refract(d, nf, eta);
    const c1 = -M.dot(d, nf), s1 = Math.sqrt(Math.max(0, 1 - c1 * c1));
    const critical = n1 > n2 ? Math.asin(n2 / n1) : Infinity;
    const th1 = Math.acos(Math.min(1, c1));
    if (Math.abs(th1 - critical) < 1e-9) continue;
    check((t === null) === (th1 > critical), `refract returns null exactly beyond the critical angle (θ1 = ${th1}, θc = ${critical})`);
    if (t) {
      const s2 = M.length(M.cross(t, nf));
      check(near(M.length(t), 1, 1e-9), "refracted ray is unit");
      check(near(n1 * s1, n2 * s2, 1e-9), `Snell: ${n1 * s1} vs ${n2 * s2}`);
      check(M.dot(t, nf) < 0, "refracted ray continues through the surface");
    }
  }
  check(near(RT.fresnel(1, 1, 1.5), 0.04, 1e-12), "R0 air–glass = 0.04");
  check(near(RT.fresnel(1e-9, 1, 1.5), 1, 1e-6), "R → 1 at grazing incidence");
  check(near(RT.fresnel(1, 1.5, 1.5), 0, 1e-12), "R0 = 0 for equal indices");
  let worst = 0;
  for (let a = 0; a < 45; a += 0.5) { const c = Math.cos(a * Math.PI / 180); worst = Math.max(worst, Math.abs(RT.schlick(c, 1, 1.5) - RT.fresnel(c, 1, 1.5))); }
  check(worst < 0.01, `Schlick within 0.01 of exact below 45° (worst ${worst})`);
  worst = 0;
  for (let a = 0; a < 90; a += 0.25) { const c = Math.cos(a * Math.PI / 180); worst = Math.max(worst, Math.abs(RT.schlick(c, 1, 1.5) - RT.fresnel(c, 1, 1.5))); }
  check(worst < 0.04, `Schlick within 0.04 of exact at every angle (worst ${worst})`);
  check(RT.fresnel(Math.cos(50 * Math.PI / 180), 1.5, 1) === 1, "TIR from glass at 50° gives R = 1");

  // 2D tracer: Snell at every hit, and no power lost beyond the cut-off
  const sc: R2.Scene2D = {
    nAmbient: 1, mirrors: [],
    regions: [R2.discRegion("ball", 1.5, 0, 0, 1, 0), R2.rectRegion("slab", 1.33, 3, 0.5, 1, 3, 20, 0), R2.prismRegion("prism", 1.7, -3, 0, 2, 10, 0)],
  };
  for (let it = 0; it < 100; it++) {
    const ang = U(0, 2 * Math.PI), o: R2.P2 = [6 * Math.cos(ang), 6 * Math.sin(ang)], tgt: R2.P2 = [U(-4, 4), U(-1.5, 1.5)];
    const res = R2.trace2d(sc, o, [tgt[0] - o[0], tgt[1] - o[1]], 1, { fresnel: true, minPower: 1e-9, maxDepth: 60 });
    const out = res.segs.filter(s => s.escaped).reduce((s, g) => s + g.power, 0);
    check(out > 1 - 1e-5 && out < 1 + 1e-9, `power conserved in the 2D tracer: ${out}`);
    for (const h of res.hits) if (!h.mirror && h.theta2 !== null) check(near(h.n1 * Math.sin(h.theta1 * Math.PI / 180), h.n2 * Math.sin(h.theta2 * Math.PI / 180), 1e-7), "2D Snell");
  }
  check(near(R2.cauchy(1.52, 0.004, 550), 1.52, 1e-12) && R2.cauchy(1.52, 0.004, 450) > R2.cauchy(1.52, 0.004, 650), "Cauchy: n(550) given, blue bends more");
}

// ---------------------------------------------------------------- slab test, BVH, grid
{
  for (let it = 0; it < 300; it++) {
    const dim = it % 2 ? 3 : 2;
    const min = Array.from({ length: dim }, () => U(-2, 1)), max = min.map(v => v + U(0.1, 2));
    const o = Array.from({ length: dim }, () => U(-4, 4)), d = Array.from({ length: dim }, () => r.norm());
    if (it % 10 === 0) d[0] = 0;   // a ray parallel to a slab
    const res = B.slab(o, d, min, max);
    // sample along the ray
    let tIn = Infinity, tOut = -Infinity;
    for (let k = 0; k <= 60000; k++) {
      const t = k * 2e-3;
      if (o.every((v, i) => v + t * d[i] >= min[i] && v + t * d[i] <= max[i])) { tIn = Math.min(tIn, t); tOut = Math.max(tOut, t); }
    }
    if (tIn === Infinity) check(res === null || res[1] - res[0] < 4e-3, `slab miss: ${res}`);
    else check(res !== null && Math.abs(res[0] - tIn) < 4e-3 && Math.abs(res[1] - tOut) < 4e-3, `slab [${res}] vs sampled [${tIn}, ${tOut}]`);
  }
  // origin on a slab plane with d = 0 on that axis: 0·∞ = NaN, must be ignored
  const s0 = B.slab([0, 0.5], [0, 1], [0, 0], [2, 1]);
  check(s0 !== null && near(s0[1], 0.5), `slab with NaN (origin on the plane, d = 0): ${s0}`);

  // 2D scenes of discs and segments
  for (const layout of ["uniform", "clustered"]) {
    const prims: { kind: 0 | 1; a: number[] }[] = [];
    const N = 400;
    for (let i = 0; i < N; i++) {
      const cx = layout === "uniform" ? U(0, 100) : 30 + 10 * r.norm(), cy = layout === "uniform" ? U(0, 100) : 60 + 10 * r.norm();
      if (i % 3) prims.push({ kind: 0, a: [cx, cy, U(0.3, 2)] });
      else { const ang = U(0, Math.PI), L = U(1, 5); prims.push({ kind: 1, a: [cx, cy, cx + L * Math.cos(ang), cy + L * Math.sin(ang)] }); }
    }
    const boxes: B.AABB[] = prims.map(p => p.kind === 0 ? { min: [p.a[0] - p.a[2], p.a[1] - p.a[2]], max: [p.a[0] + p.a[2], p.a[1] + p.a[2]] } : { min: [Math.min(p.a[0], p.a[2]), Math.min(p.a[1], p.a[3])], max: [Math.max(p.a[0], p.a[2]), Math.max(p.a[1], p.a[3])] });
    const bvhM = B.buildBVH(boxes, { method: "median" }), bvhS = B.buildBVH(boxes, { method: "sah" });
    const lo = [Math.min(...boxes.map(b => b.min[0])), Math.min(...boxes.map(b => b.min[1]))], hi = [Math.max(...boxes.map(b => b.max[0])), Math.max(...boxes.map(b => b.max[1]))];
    const grid = B.buildGrid(boxes, lo, hi, [16, 16]);
    // every primitive in exactly one leaf
    for (const bv of [bvhM, bvhS]) {
      const seen = new Int32Array(N);
      for (const nd of bv.nodes) if (nd.left < 0) for (let i = nd.start; i < nd.start + nd.count; i++) seen[bv.order[i]]++;
      check(seen.every(v => v === 1), "every primitive in exactly one leaf");
    }
    const root = bvhS.nodes[0];
    check(B.bvhCost(bvhS) <= N + 1e-9, `SAH tree cost ${B.bvhCost(bvhS)} ≤ no split ${N}`);
    let sumS = 0, sumBrute = 0;
    for (let it = 0; it < 1000; it++) {
      const o = [U(-10, 110), U(-10, 110)], ang = U(0, 2 * Math.PI), d = [Math.cos(ang), Math.sin(ang)];
      const hit = (i: number, tmax: number) => { const p = prims[i]; return p.kind === 0 ? B.rayDisc2(o, d, p.a[0], p.a[1], p.a[2], tmax) : B.raySegment2(o, d, p.a[0], p.a[1], p.a[2], p.a[3], tmax); };
      const bf = B.bruteForce(N, hit), m = B.traverseBVH(bvhM, o, d, hit), s = B.traverseBVH(bvhS, o, d, hit), g = B.traverseGrid(grid, o, d, hit);
      check(bf.prim === m.prim && bf.prim === s.prim && bf.prim === g.prim && (bf.prim < 0 || (near(bf.t, m.t) && near(bf.t, g.t))), `same nearest hit (${layout}): brute ${bf.prim} median ${m.prim} sah ${s.prim} grid ${g.prim}`);
      sumS += s.primTests + s.boxTests; sumBrute += bf.primTests;
    }
    check(sumS < sumBrute / 4, `BVH saves tests (${layout}): ${sumS} vs ${sumBrute}`);
    void root;
  }

  // a marching-cubes mesh in 3D
  const v = nucleiStack({ noise: 0 });
  const mesh = marchingCubes(v.data, v.nx, v.ny, v.nz, 100, { spacing: v.spacing });
  const P = mesh.positions, I = mesh.indices, nt = I.length / 3;
  const tv = (k: number): Vec3 => [P[3 * k], P[3 * k + 1], P[3 * k + 2]];
  const boxes: B.AABB[] = [];
  for (let t = 0; t < nt; t++) { const a = tv(I[3 * t]), b = tv(I[3 * t + 1]), c = tv(I[3 * t + 2]); boxes.push({ min: [0, 1, 2].map(k => Math.min(a[k], b[k], c[k])), max: [0, 1, 2].map(k => Math.max(a[k], b[k], c[k])) }); }
  const bvh = B.buildBVH(boxes, { method: "sah" });
  let tests = 0, hits = 0;
  for (let it = 0; it < 300; it++) {
    const o: Vec3 = [U(-2, 15), U(-2, 15), U(-2, 12)], tgt: Vec3 = [U(3, 10), U(3, 10), U(2, 8)], d = M.normalize(M.sub(tgt, o));
    const hit = (t: number, tmax: number) => { const s = RT.solveTriangle(o, d, tv(I[3 * t]), tv(I[3 * t + 1]), tv(I[3 * t + 2])); return s.t > 0 && s.t < tmax && RT.insideTriangle(s.u, s.v) ? s.t : Infinity; };
    const bf = B.bruteForce(nt, hit), s = B.traverseBVH(bvh, o, d, hit);
    check(bf.prim === s.prim || near(bf.t, s.t, 1e-9), `mesh: same nearest hit ${bf.t} vs ${s.t}`);
    tests += s.primTests + s.boxTests; if (bf.prim >= 0) hits++;
  }
  check(hits > 50, `mesh rays hit (${hits})`);
  check(tests / 300 < nt / 20, `mesh BVH: ${(tests / 300).toFixed(0)} tests per ray vs ${nt} triangles`);
}

// ---------------------------------------------------------------- Monte Carlo
{
  // ∫0^1 x² dx = 1/3 with uniform samples: RMS over 500 repetitions ≈ σ/√N
  const f = (x: number) => x * x, sigma = Math.sqrt(1 / 5 - 1 / 9);
  for (const N of [16, 256]) {
    let se = 0;
    for (let rep = 0; rep < 500; rep++) { const e = MC.estimate(f, () => r.unif(), () => 1, N) - 1 / 3; se += e * e; }
    const rms = Math.sqrt(se / 500), pred = sigma / Math.sqrt(N);
    check(Math.abs(rms / pred - 1) < 0.1, `RMS ${rms} vs σ/√N ${pred} (N = ${N})`);
  }
  // importance sampling with p(x) = 2x: f/p = x/2, variance 1/8 − 1/9
  {
    let w = new MC.Welford();
    for (let k = 0; k < 20000; k++) { const x = Math.sqrt(r.unif()); w.add(f(x) / (2 * x)); }
    check(Math.abs(w.mean - 1 / 3) < 4 * w.sem(), `importance sampled mean ${w.mean}`);
    check(Math.abs(w.variance() - (1 / 8 - 1 / 9)) < 0.002, `importance sampled variance ${w.variance()}`);
  }
  // hemisphere: uniform estimate of ∫cos dω = π; cosine sampling of a Lambertian surface under a
  // constant sky L: estimate ρ L for every sample (zero variance)
  {
    const w = new MC.Welford();
    for (let k = 0; k < 40000; k++) { const v = MC.uniformHemisphere(r.unif(), r.unif()); w.add(v[2] * 2 * Math.PI); }
    check(Math.abs(w.mean - Math.PI) < 4 * w.sem(), `∫cos dω ≈ π: ${w.mean}`);
    const rho = 0.7, Lsky = 2, wc = new MC.Welford();
    for (let k = 0; k < 1000; k++) { const v = MC.cosineHemisphere(r.unif(), r.unif()); const fr = rho / Math.PI, pdf = v[2] / Math.PI; wc.add(fr * Lsky * v[2] / pdf); }
    check(near(wc.mean, rho * Lsky, 1e-9) && wc.variance() < 1e-20, `cosine sampling: zero variance (${wc.mean}, ${wc.variance()})`);
    // directions are unit and in the upper hemisphere; toWorld keeps them unit and on the normal's side
    for (let k = 0; k < 200; k++) { const n = randDir(), v = MC.toWorld(MC.cosineHemisphere(r.unif(), r.unif()), n); check(near(M.length(v), 1, 1e-9) && M.dot(v, n) >= -1e-12, "toWorld"); }
  }
  // Russian roulette keeps the mean, increases the variance
  {
    const plain = new MC.Welford(), rr = new MC.Welford(), q = 0.3;
    for (let k = 0; k < 200000; k++) { const x = r.unif(); plain.add(x); rr.add(MC.russianRoulette(q, r.unif()) ? x / q : 0); }
    check(Math.abs(rr.mean - 0.5) < 4 * rr.sem(), `roulette mean ${rr.mean}`);
    check(rr.variance() > plain.variance(), "roulette adds variance");
  }
  // 2D half-circle: ∫ cos θ dθ = 2 with uniform and cosine angles; tabulated pdf integrates to 1
  {
    const u = MC.estimate(Math.cos, () => MC.uniformAngle(r.unif()), MC.uniformAnglePdf, 100000);
    check(Math.abs(u - 2) < 0.01, `2D uniform ∫cos = ${u}`);
    const c = MC.estimate(Math.cos, () => MC.cosineAngle(r.unif()), MC.cosineAnglePdf, 1000);
    check(near(c, 2, 1e-9), `2D cosine ∫cos exact: ${c}`);
    const tp = new MC.TabulatedPdf(-1, 1, [1, 2, 3, 0, 4]);
    let s = 0; for (let k = 0; k < 1000; k++) { const x = -1 + 2 * (k + 0.5) / 1000; s += tp.pdf(x) * 2 / 1000; }
    check(near(s, 1, 1e-9), `tabulated pdf integrates to 1: ${s}`);
    for (let k = 0; k < 200; k++) { const x = tp.sample(r.unif()); check(tp.pdf(x) > 0, "inverse CDF never samples a zero bin"); }
  }
  // path tracer: the three sampling strategies estimate the same image (unbiased)
  {
    const sc = RT.boxScene({ glass: false, mirror: false });
    const W = 8, H = 6, means: number[] = [], sems: number[] = [];
    for (const sampling of ["uniform", "cosine", "light"] as RT.Sampling[]) {
      const w = new MC.Welford();
      for (let px = 0; px < W * H; px++) {
        const i = px % W, j = Math.floor(px / W);
        for (let s = 0; s < 400; s++) {
          const rnd = RT.pixelRng(px, s, 11 + sampling.length);
          const ray = RT.cameraRay(i, j, W, H, sc.camera, rnd(), rnd());
          const c = RT.tracePath(sc, ray.o, ray.d, rnd, { maxBounces: 3, russianRoulette: false, sampling });
          w.add(c[0] + c[1] + c[2]);
        }
      }
      means.push(w.mean); sems.push(w.sem());
    }
    for (let k = 1; k < 3; k++) check(Math.abs(means[k] - means[0]) < 4 * Math.hypot(sems[k], sems[0]), `path tracer means agree: ${means.map(v => v.toFixed(4))} ± ${sems.map(v => v.toFixed(4))}`);
    check(sems[2] < sems[0], `light sampling has less noise than uniform: ${sems[2]} vs ${sems[0]}`);
  }
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
