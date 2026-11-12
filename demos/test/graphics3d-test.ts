// Tests for the 3D-graphics day (lib/raster.ts, lib/volrender.ts, the shading demo's light and
// texture code, and the WebGL matrices of lib/mat4.ts used by pipeline3d):
//  - P maps the corners of the frustum to the corners of the NDC cube; lookAt is orthonormal
//    with V e = 0; the NDC depth formula matches the matrix;
//  - edge functions and barycentrics: sum to 1, reproduce the vertices, inside test against a
//    brute-force point-in-triangle; a quad split into two triangles covers every pixel exactly
//    once with the top-left rule (random quads with vertices on pixel centres);
//  - perspective-correct interpolation equals the attribute at the 3D point seen through the
//    pixel; affine interpolation does not;
//  - near-plane clipping keeps z ≥ −w and leaves a visible triangle unchanged;
//  - M⁻ᵀ n stays perpendicular to transformed tangents (M n does not, for non-uniform scale);
//  - Phong reflection and the Blinn half vector; the icosphere is closed and outward;
//  - front-to-back = back-to-front compositing; opacity correction makes a homogeneous slab
//    independent of the step and equal to 1 − e^(−τD); early termination never adds samples;
//    the isosurface crossing is exact on a linear ramp;
//  - mipmaps: total texels → 4/3 of level 0, box levels keep the mean.
// Run: node_modules/.bin/esbuild demos/test/graphics3d-test.ts --bundle --platform=node --log-level=warning | node
import * as M from "../src/lib/mat4";
import * as R from "../src/lib/raster";
import * as VR from "../src/lib/volrender";
import { rng } from "../src/lib/random";
import { reflect, halfVector, reflectance, unit, dot3, V3 } from "../src/shading/light";
import { icosphere, torus } from "../src/shading/meshes";
import { buildMips, mipTexels, checkerTexture, footprintLevel } from "../src/shading/texture";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol = 1e-5) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const r = rng(2026);
const U = (a: number, b: number) => a + (b - a) * r.unif();

// ------------------------------------------------------------------ matrices
{
  for (let it = 0; it < 20; it++) {
    const fov = U(0.3, 2), asp = U(0.5, 2), n = U(0.05, 2), f = n + U(1, 50);
    const P = M.perspective(fov, asp, n, f), th = Math.tan(fov / 2);
    for (const d of [n, f]) for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      const p = M.transformPoint(P, [sx * d * th * asp, sy * d * th, -d]);
      check(near(p[0], sx, 1e-4) && near(p[1], sy, 1e-4) && near(p[2], d === n ? -1 : 1, 1e-4), `frustum corner → NDC corner (${p.map(x => x.toFixed(4))})`);
    }
    // the depth formula of the slides: z_ndc = A + B / z_view with A, B from n and f
    const d = U(n, f), pz = M.transformPoint(P, [0, 0, -d])[2];
    check(near(pz, R.ndcDepth(d, n, f), 1e-4), "ndcDepth matches the matrix");
  }
  for (let it = 0; it < 20; it++) {
    const e: M.Vec3 = [U(-5, 5), U(-5, 5), U(-5, 5)], g: M.Vec3 = [U(-1, 1), U(-1, 1), U(-1, 1)];
    const V = M.lookAt(e, g, [0, 1, 0]);
    const ve = M.transformPoint(V, e);
    check(Math.hypot(...ve) < 1e-4, "V e = 0");
    const R3 = [0, 1, 2].map(i => [M.at(V, i, 0), M.at(V, i, 1), M.at(V, i, 2)] as M.Vec3);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) check(near(M.dot(R3[i], R3[j]), i === j ? 1 : 0, 1e-5), "rows of R orthonormal");
    const gv = M.transformPoint(V, g);
    check(Math.abs(gv[0]) < 1e-4 && Math.abs(gv[1]) < 1e-4 && gv[2] < 0, "the target lies on the −z axis");
  }
  // by-hand example of the exercise: n = 1, f = 3
  check(near(R.ndcDepth(1, 1, 3), -1) && near(R.ndcDepth(3, 1, 3), 1) && near(R.ndcDepth(2, 1, 3), 0.5), "n = 1, f = 3: z_ndc(2) = 0.5");
}

// ------------------------------------------------------------------ edge functions, barycentrics
const vtx = (x: number, y: number, z = 0, w = 1, attr: number[] = []): R.WinVertex => ({ x, y, z, w, attr });
function insideBrute(a: R.WinVertex, b: R.WinVertex, c: R.WinVertex, px: number, py: number): boolean {
  // same side of each edge as the opposite vertex (strict; points on edges skipped by the caller)
  const s = (p: R.WinVertex, q: R.WinVertex, o: R.WinVertex) => Math.sign(((q.x - p.x) * (py - p.y) - (q.y - p.y) * (px - p.x))) === Math.sign(((q.x - p.x) * (o.y - p.y) - (q.y - p.y) * (o.x - p.x)));
  return s(a, b, c) && s(b, c, a) && s(c, a, b);
}
{
  for (let it = 0; it < 200; it++) {
    const a = vtx(U(0, 20), U(0, 20)), b = vtx(U(0, 20), U(0, 20)), c = vtx(U(0, 20), U(0, 20));
    if (Math.abs(R.area2(a, b, c)) < 1) continue;
    const px = U(0, 20), py = U(0, 20), bc = R.barycentric(a, b, c, px, py)!;
    check(near(bc[0] + bc[1] + bc[2], 1, 1e-9), "barycentrics sum to 1");
    check(near(bc[0] * a.x + bc[1] * b.x + bc[2] * c.x, px, 1e-9) && near(bc[0] * a.y + bc[1] * b.y + bc[2] * c.y, py, 1e-9), "barycentrics reproduce the point");
    const va = R.barycentric(a, b, c, a.x, a.y)!;
    check(near(va[0], 1, 1e-9) && Math.abs(va[1]) < 1e-9 && Math.abs(va[2]) < 1e-9, "vertex a has (1, 0, 0)");
    check(insideBrute(a, b, c, px, py) === (bc[0] > 0 && bc[1] > 0 && bc[2] > 0), "inside ⇔ all barycentrics positive");
  }
  // the exercise triangle a = (1,1), b = (7,2), c = (3,6)
  const a = vtx(1, 1), b = vtx(7, 2), c = vtx(3, 6);
  check(R.area2(a, b, c) === 28, "area2 of the exercise triangle is 28");
  // rasteriser coverage against brute force (no pixel centre exactly on an edge here)
  for (let it = 0; it < 50; it++) {
    const tri = [vtx(U(0, 30), U(0, 20), 0, 1, [0]), vtx(U(0, 30), U(0, 20), 0, 1, [0]), vtx(U(0, 30), U(0, 20), 0, 1, [0])];
    const t = R.createTarget(30, 20);
    R.rasteriseTriangle(t, tri, { cull: false }, () => [1, 1, 1]);
    let ok = true;
    for (let j = 0; j < 20; j++) for (let i = 0; i < 30; i++) if ((t.id[j * 30 + i] === 0) !== insideBrute(tri[0], tri[1], tri[2], i + 0.5, j + 0.5)) ok = false;
    check(ok, "rasterised pixels = pixel centres inside");
  }
}

// ------------------------------------------------------------------ top-left rule: a quad split in two
{
  for (let it = 0; it < 200; it++) {
    // convex quad with vertices on pixel centres or corners (so many centres lie on edges)
    const cx = 6, cy = 6, ang = [0, 1, 2, 3].map(k => k * Math.PI / 2 + U(-0.6, 0.6));
    const q = ang.map(a => vtx(Math.round((cx + U(3, 5.5) * Math.cos(a)) * 2) / 2, Math.round((cy + U(3, 5.5) * Math.sin(a)) * 2) / 2));
    if (R.area2(q[0], q[1], q[2]) <= 0 || R.area2(q[0], q[2], q[3]) <= 0) continue;
    const cover = new Uint8Array(14 * 14);
    for (const tri of [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]) {
      const t = R.createTarget(14, 14), rec: R.Fragment[] = [];
      R.rasteriseTriangle(t, tri, { topLeft: true, depthTest: false }, () => [1, 1, 1], 0, rec);
      for (const f of rec) if (f.inside) cover[f.j * 14 + f.i]++;
    }
    // reference: centres strictly inside the quad, plus those on its boundary chosen by the rule
    let twice = 0, holes = 0;
    for (let j = 0; j < 14; j++) for (let i = 0; i < 14; i++) {
      const px = i + 0.5, py = j + 0.5;
      const inQuad = insideBrute(q[0], q[1], q[2], px, py) || insideBrute(q[0], q[2], q[3], px, py);
      if (cover[j * 14 + i] > 1) twice++;
      if (inQuad && cover[j * 14 + i] === 0) holes++;
    }
    check(twice === 0, "top-left rule: no pixel drawn twice across the shared diagonal");
    check(holes === 0, "top-left rule: no hole along the shared diagonal");
  }
  // without the rule the shared edge of the "shared edge" preset is drawn twice
  const A = vtx(1.5, 1.5), B = vtx(13.5, 1.5), C = vtx(1.5, 10.5), D = vtx(13.5, 10.5);
  const both = (tl: boolean) => {
    const cover = new Uint8Array(16 * 12);
    for (const tri of [[A, B, C], [B, D, C]]) { const rec: R.Fragment[] = []; R.rasteriseTriangle(R.createTarget(16, 12), tri, { topLeft: tl, depthTest: false }, () => [1, 1, 1], 0, rec); for (const f of rec) if (f.inside) cover[f.j * 16 + f.i]++; }
    return Array.from(cover).filter(x => x > 1).length;
  };
  check(both(true) === 0 && both(false) === 4, `shared edge: 0 doubles with the rule, 4 without (got ${both(true)}, ${both(false)})`);
}

// ------------------------------------------------------------------ perspective-correct interpolation
{
  for (let it = 0; it < 30; it++) {
    // a triangle in view space, attribute = its view-space x coordinate
    const P = M.perspective(1, 1, 0.5, 50), W = 64, H = 64;
    const pts: M.Vec3[] = [[U(-2, 2), U(-2, 2), -U(1.5, 3)], [U(-2, 2), U(-2, 2), -U(6, 12)], [U(-2, 2), U(-2, 2), -U(2, 10)]];
    const cl: R.ClipVertex[] = pts.map(p => ({ p: M.transformVec4(P, [...p, 1]), attr: [p[0], p[1], p[2]] }));
    const win = cl.map(c => R.toWindow(c, W, H));
    const t = R.createTarget(W, H);
    let worstP = 0, worstA = 0;
    const invP = M.invert(P)!;
    for (const persp of [true, false]) R.rasteriseTriangle(t, win, { perspective: persp, depthTest: false }, (a, i, j) => {
      // the 3D point under the pixel centre: intersect the eye ray with the triangle's plane
      const nd: M.Vec3 = [(i + 0.5) / W * 2 - 1, (j + 0.5) / H * 2 - 1, 0];
      const q = M.transformPoint(invP, nd), dir = M.normalize(q);
      const nrm = M.cross(M.sub(pts[1], pts[0]), M.sub(pts[2], pts[0]));
      const s = M.dot(nrm, pts[0]) / M.dot(nrm, dir), X = M.scale3(dir, s);
      const e = Math.abs(a[0] - X[0]) + Math.abs(a[2] - X[2]);
      if (persp) worstP = Math.max(worstP, e); else worstA = Math.max(worstA, e);
      return [0, 0, 0];
    });
    check(worstP < 1e-3, `perspective-correct = the 3D point (error ${worstP})`);
    check(worstA > 1e-2, `affine interpolation is wrong under perspective (error ${worstA})`);
  }
  // the exercise: an edge from w = 1 (u = 0) to w = 3 (u = 1), screen midpoint → u = 1/4
  const a = vtx(0, 0, 0, 1, [0]), b = vtx(10, 0, 0, 3, [1]);
  check(near(R.interpolate([a, b, b], [0.5, 0.5, 0], true)[0], 0.25), "perspective-correct midpoint u = 1/4");
}

// ------------------------------------------------------------------ near-plane clipping
{
  for (let it = 0; it < 100; it++) {
    const tri: R.ClipVertex[] = [0, 1, 2].map(() => { const w = U(-2, 4); return { p: [U(-3, 3), U(-3, 3), U(-4, 4), w] as R.V4, attr: [U(0, 1)] }; });
    const out = R.clipNear(tri);
    check(out.every(v => v.p[2] + v.p[3] >= -1e-9), "clipped polygon has z ≥ −w");
    if (tri.every(v => v.p[2] + v.p[3] >= 0)) check(out.length === 3, "a triangle in front is unchanged");
    if (tri.every(v => v.p[2] + v.p[3] < 0)) check(out.length === 0, "a triangle behind is removed");
  }
}

// ------------------------------------------------------------------ normals: inverse transpose
{
  for (let it = 0; it < 50; it++) {
    const Mm = M.mul(M.rotation([U(-1, 1), U(-1, 1), U(-1, 1)], U(0, 3)), M.scaling(U(0.2, 3), U(0.2, 3), U(0.2, 3)));
    const n = M.normalize([U(-1, 1), U(-1, 1), U(-1, 1)]);
    let t = M.cross(n, [U(-1, 1), U(-1, 1), U(-1, 1)]); t = M.normalize(t);
    const tw = M.transformDir(Mm, t), nGood = M.mat3Vec(M.normalMatrix(Mm), n);
    check(Math.abs(M.dot(tw, nGood)) < 1e-4 * M.length(tw) * M.length(nGood), "M⁻ᵀ n ⊥ M t");
  }
  // the exercise: M = diag(2, 1, 1), plane x + y = 1: n = (1, 1, 0), tangent (1, −1, 0)
  const Md = M.scaling(2, 1, 1), t: M.Vec3 = [1, -1, 0], n: M.Vec3 = [1, 1, 0];
  check(M.dot(M.transformDir(Md, t), M.transformDir(Md, n)) === 3, "M n · M t = 3 (not perpendicular)");
  check(near(M.dot(M.transformDir(Md, t), M.mat3Vec(M.normalMatrix(Md), n)), 0), "M⁻ᵀ n · M t = 0");
}

// ------------------------------------------------------------------ reflection models, meshes
{
  for (let it = 0; it < 50; it++) {
    const n = unit([U(-1, 1), U(-1, 1), U(-1, 1)] as V3), l = unit([U(-1, 1), U(-1, 1), U(-1, 1)] as V3);
    const rr = reflect(n, l);
    check(near(Math.hypot(...rr), 1, 1e-9) && near(dot3(rr, n), dot3(l, n), 1e-9), "reflection keeps length and angle");
    // mirror configuration: v = r → Phong and Blinn–Phong both give k_s
    if (dot3(n, l) > 0.1) {
      const m = { ka: 0, kd: 0, ks: 0.7, shininess: 30 };
      check(near(reflectance(n, l, rr, "phong", m).specular, 0.7, 1e-6) && near(reflectance(n, l, rr, "blinn", m).specular, 0.7, 1e-6), "mirror direction: specular = k_s");
      check(near(dot3(halfVector(l, rr), n), 1, 1e-6), "half vector = normal in the mirror configuration");
    }
  }
  for (const L of [0, 1, 2, 3]) {
    const m = icosphere(L), nv = m.positions.length / 3;
    check(nv === 10 * 4 ** L + 2 && m.indices.length / 3 === 20 * 4 ** L, `icosphere level ${L}: V = 10·4^L + 2, F = 20·4^L`);
    let vol = 0;
    for (let f = 0; f < m.indices.length; f += 3) {
      const [a, b, c] = [0, 1, 2].map(k => m.indices[f + k]).map(i => [m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]] as M.Vec3);
      vol += M.dot(a, M.cross(b, c)) / 6;
    }
    check(vol > 0 && vol < 4 / 3 * Math.PI, `icosphere level ${L} is outward and inside the unit ball (volume ${vol.toFixed(3)})`);
    // closed: every directed edge appears once and its reverse once
    const E = new Map<string, number>();
    for (let f = 0; f < m.indices.length; f += 3) for (let k = 0; k < 3; k++) { const a = m.indices[f + k], b = m.indices[f + (k + 1) % 3]; E.set(`${a},${b}`, (E.get(`${a},${b}`) ?? 0) + 1); }
    check([...E.entries()].every(([k, c]) => { const [a, b] = k.split(","); return c === 1 && E.get(`${b},${a}`) === 1; }), `icosphere level ${L}: consistently oriented, closed`);
  }
  const tm = torus();
  check(tm.positions.length / 3 * 2 === tm.indices.length / 3, "torus: F = 2V (Euler characteristic 0)");
}

// ------------------------------------------------------------------ compositing
{
  for (let it = 0; it < 100; it++) {
    const n = 1 + Math.floor(U(0, 12));
    const al = Array.from({ length: n }, () => U(0, 1)), co = Array.from({ length: n }, () => [U(0, 1), U(0, 1), U(0, 1)] as [number, number, number]);
    const f = VR.frontToBack(al, co), b = VR.backToFront(al, co);
    check(f.C.every((x, k) => near(x, b.C[k], 1e-9)) && near(f.T, b.T, 1e-9), "front-to-back = back-to-front");
    const e = VR.frontToBack(al, co, 0.05);
    check(e.used <= n && (e.used === n || e.T < 0.05), "early termination stops only when T < ε");
    check(e.C.every((x, k) => Math.abs(x - f.C[k]) <= 0.05 + 1e-12), "early termination changes C by less than ε");
  }
  // homogeneous slab: per-voxel opacity a0 = 1 − e^(−τ), thickness D voxels
  const tau = 0.3, D = 12, a0 = 1 - Math.exp(-tau);
  const tf: VR.TF = { points: [[0, a0], [255, a0]], map: "grey" };
  for (const ds of [0.25, 0.5, 1, 2, 3]) {
    const n = Math.round(D / ds), vals = new Array(n).fill(100);
    const T1 = VR.renderRay(vals, tf, { mode: "dvr", step: ds, correction: true }).T;
    const T0 = VR.renderRay(vals, tf, { mode: "dvr", step: ds, correction: false }).T;
    check(near(1 - T1, 1 - Math.exp(-tau * D), 1e-9), `opacity correction: slab opacity = 1 − e^(−τD) at Δs = ${ds}`);
    if (ds !== 1) check(!near(1 - T0, 1 - Math.exp(-tau * D), 1e-3), `without correction the slab opacity depends on Δs (${ds})`);
  }
  check(near(VR.segmentAlpha(0.5, 1), 0.3934693402873666) && near(VR.segmentAlpha(0.5, 0.5), 0.22119921692859512), "α for τ = 0.5 at Δs = 1 and 0.5");
  check(near(VR.correctAlpha(VR.segmentAlpha(0.5, 1), 0.5), VR.segmentAlpha(0.5, 0.5), 1e-12), "correctAlpha = 1 − e^(−τΔs)");
  // isosurface crossing on a linear ramp is exact
  const ramp = Array.from({ length: 10 }, (_, i) => 10 * i);
  const hit = VR.renderRay(ramp, tf, { mode: "iso", step: 1, iso: 37 }).hit;
  check(near(hit, 3.7, 1e-12), "iso crossing by linear interpolation");
  // MIP / MinIP / average
  const vals = [5, 40, 120, 3, 80, 60, 2];
  check(VR.renderRay(vals, tf, { mode: "mip", step: 1 }).value === 120, "MIP");
  check(VR.renderRay(vals, tf, { mode: "minip", step: 1, body: 30 }).value === 3, "MinIP between the first and last body sample");
  check(near(VR.renderRay(vals, tf, { mode: "avg", step: 1 }).value, 310 / 7), "average");
}

// ------------------------------------------------------------------ the phantom and the CPU caster
{
  const v = VR.headPhantom();
  check(v.nx === 64 && v.ny === 64 && v.nz === 40, "phantom size 64 × 64 × 40");
  let lo = Infinity, hi = -Infinity;
  for (const x of v.data) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
  check(lo >= 0 && hi <= 255, "phantom values in 0..255");
  const val = (x: number, y: number, zmm: number) => v.data[(Math.round(zmm / 1.6) * 64 + y) * 64 + x];
  check(val(32, 45, 31) > 180, "the vessel trunk is bright");
  check(val(32, 55, 24) < 30, "the cavity is dark");
  check(Math.abs(val(32, 29, 31) - 100) < 20 || val(32, 29, 31) > 180, "brain (or vessel) at the centre");
  const PV = M.multiply(M.perspective(0.6, 1, 0.1, 10), M.lookAt([0.3, 0.4, 2.2], [0, 0, 0]));
  const tf: VR.TF = { points: [[140, 0], [165, 0.9], [255, 0.9]], map: "bone" };
  const a = VR.castImage(v, tf, PV, 16, 16, { mode: "dvr", step: 1, eps: 0.05 });
  let ok = true;
  for (let i = 0; i < 256; i++) if (a.used[i] > a.full[i]) ok = false;
  check(ok, "early termination never takes more samples");
  const lut = VR.tfLUT(tf);
  check(lut[4 * 150 + 3] === Math.round(255 * VR.tfAlpha(tf, 150)) && lut[4 * 100 + 3] === 0, "transfer-function LUT");
}

// ------------------------------------------------------------------ mipmaps
{
  const ms = buildMips(checkerTexture(256, 16), 256);
  check(ms.length === 9, "256² texture has 9 levels");
  check(near(mipTexels(ms) / (256 * 256), 4 / 3, 1e-4) && mipTexels(ms) === (4 ** 9 - 1) / 3, "mipmaps cost a third more");
  const mean = (d: Float32Array) => d.reduce((s, x) => s + x, 0) / d.length;
  check(ms.every(m => near(mean(m.data), mean(ms[0].data), 1e-6)), "box levels keep the mean");
  check(near(footprintLevel(8, 0, 0, 8), 3), "8 × 8 texels per pixel → level 3");
}

console.log(`graphics3d: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
