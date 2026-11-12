// Tests for the marching-cubes day's libraries (lib/marching.ts, lib/mesh.ts, lib/volume.ts):
//  - marching squares: crossing points against the bilinear interpolant, closed contours,
//    shoelace area of a disk, saddle rules and the asymptotic decider against brute force;
//  - the cube: 24 rotations, 256 cases → 23 classes (rotations) and 15 (with complement),
//    Burnside's count; generated triangle table: every case's loops use each cut edge once;
//  - marching cubes / tetrahedra: closed, consistently oriented meshes on random volumes for
//    every consistent face rule; cracks for the complement rule; sphere area/volume converge;
//    torus χ = 0; the single-voxel octahedron (area √3, volume 1/6);
//  - volumes: voxel ↔ world round trip, axis-aligned reslice = indexing, resampling a constant
//    volume stays constant, trilinear exact on linear functions, blur keeps the mean.
// Run: node_modules/.bin/esbuild demos/test/marchingcubes-test.ts --bundle --platform=node --log-level=warning | node
import * as MS from "../src/lib/marching";
import { meshArea, meshVolume, meshTopology, vertexCluster, laplacianSmooth } from "../src/lib/mesh";
import * as V from "../src/lib/volume";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ------------------------------------------------------------------ marching squares
function bilinear(v: number[], x: number, y: number): number {
  return v[0] * (1 - x) * (1 - y) + v[1] * x * (1 - y) + v[2] * x * y + v[3] * (1 - x) * y;
}

function testSquares(): void {
  const r = rng(3);
  // crossing points lie on the level set of the (edge-linear) interpolant
  for (let k = 0; k < 500; k++) {
    const w = 6, h = 5, vals = Array.from({ length: w * h }, () => r.unif() * 10);
    const c = 5;
    const C = MS.marchingSquares(vals, w, h, c, { saddle: "decider" });
    for (let i = 0; i < C.points.length; i += 2) {
      const x = C.points[i], y = C.points[i + 1];
      const x0 = Math.floor(x), y0 = Math.floor(y);
      const val = x === x0 ? vals[y0 * w + x0] * (1 - (y - y0)) + vals[Math.min(h - 1, y0 + 1) * w + x0] * (y - y0)
        : vals[y0 * w + x0] * (1 - (x - x0)) + vals[y0 * w + Math.min(w - 1, x0 + 1)] * (x - x0);
      check(near(val, c, 1e-9), `crossing on level set (${val})`);
    }
    // each point is used by at most one outgoing and one incoming segment; padded grid → all closed
    const P = MS.padGrid(vals, w, h, -1);
    const Cp = MS.marchingSquares(P.values, P.w, P.h, c, { saddle: (["separate", "join", "decider"] as const)[k % 3] });
    check(Cp.lines.every(L => L.closed), "padded contours are closed");
    // area enclosed ≈ total positive, ≥ 0 because the border is below c
    check(MS.contourArea(Cp) > -1e-9, "enclosed area non-negative");
  }
  // shoelace area of a sampled disk (signed distance) converges to πR²
  for (const R of [5, 10, 20]) {
    const n = 2 * R + 6, vals: number[] = [];
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) vals.push(R - Math.hypot(x - (n - 1) / 2, y - (n - 1) / 2));
    const C = MS.marchingSquares(vals, n, n, 0);
    check(C.lines.length === 1 && C.lines[0].closed, "disk gives one closed contour");
    const A = MS.contourArea(C), L = MS.contourLength(C);
    check(near(A / (Math.PI * R * R), 1, 0.02), `disk area R=${R}: ${A} vs ${Math.PI * R * R}`);
    check(near(L / (2 * Math.PI * R), 1, 0.01), `disk perimeter R=${R}: ${L}`);
  }
  // a dark hole gives negative area
  {
    const n = 20, vals: number[] = [];
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) vals.push(Math.hypot(x - 9.5, y - 9.5) - 4);
    const P = MS.padGrid(vals, n, n, -100);
    const C = MS.marchingSquares(P.values, P.w, P.h, 0);
    const areas = C.lines.map(L => MS.lineArea(C, L)).sort((a, b) => a - b);
    check(areas.length === 2 && areas[0] < 0 && areas[1] > 0, "ring: hole negative, outer positive " + areas);
  }
  // saddle: the decider agrees with the connectivity of the bilinear interpolant (brute force)
  for (let k = 0; k < 2000; k++) {
    const c = 5;
    const hi = () => 5.01 + r.unif() * 5, lo = () => 4.99 - r.unif() * 5;
    const v = k % 2 ? [hi(), lo(), hi(), lo()] : [lo(), hi(), lo(), hi()];
    const joins = MS.deciderJoins(v, c);
    // brute force: is there a path through above-c points between the two above corners?
    // the bilinear interpolant restricted to the diagonal joining them; for a saddle, the
    // above corners are connected iff the saddle value exceeds c.
    const m = 200;
    let connected: boolean;
    {
      // flood fill on a fine grid
      const above = new Uint8Array((m + 1) * (m + 1));
      for (let j = 0; j <= m; j++) for (let i = 0; i <= m; i++) above[j * (m + 1) + i] = bilinear(v, i / m, j / m) > c ? 1 : 0;
      const start = k % 2 ? 0 : m, goal = k % 2 ? (m + 1) * (m + 1) - 1 : m * (m + 1);
      const seen = new Uint8Array(above.length), stack = [start];
      seen[start] = 1;
      while (stack.length) {
        const p = stack.pop()!, i = p % (m + 1), j = (p - i) / (m + 1);
        for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const a = i + di, b = j + dj;
          if (a < 0 || b < 0 || a > m || b > m) continue;
          const q = b * (m + 1) + a;
          if (!seen[q] && above[q]) { seen[q] = 1; stack.push(q); }
        }
      }
      connected = seen[goal] === 1;
    }
    const f = MS.saddleValue(v);
    if (Math.abs(f - c) > 0.05) check(joins === connected, `decider vs flood fill: f*=${f.toFixed(3)} ${joins} ${connected}`);
  }
  // the 16 cases: number of segments
  const nseg = [0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0];
  for (let cs = 0; cs < 16; cs++) {
    const v = [0, 1, 2, 3].map(k => ((cs >> k) & 1 ? 1 : 0));
    check(MS.squareSegments(v, 0.5, "separate").length === nseg[cs], `case ${cs} segments`);
    check(MS.squareCase(v, 0.5) === cs, `case index ${cs}`);
  }
  // midpoint interpolation puts points at edge midpoints
  {
    const C = MS.marchingSquares([0, 10, 0, 0], 2, 2, 2, { interp: "midpoint" });
    check(C.points.length === 4 && near(C.points[0], 1, 1e-12) && near(C.points[1], 0.5, 1e-12) && near(C.points[2], 0.5, 1e-12), "midpoint " + Array.from(C.points));
  }
}

// ------------------------------------------------------------------ the cube
function testCube(): void {
  check(MS.CUBE_ROTATIONS.length === 24, "24 rotations");
  const perms = new Set(MS.CUBE_ROTATIONS.map(R => R.perm.join(",")));
  check(perms.size === 24, "24 distinct corner permutations");
  check(new Set(MS.CLASSES_23.of.map(c => c.rep)).size === 23, "23 classes under rotation");
  check(new Set(MS.CLASSES_15.of.map(c => c.rep)).size === 15, "15 classes with complement");
  // Burnside: average number of fixed cases
  let fixed = 0;
  for (const R of MS.CUBE_ROTATIONS) for (let cs = 0; cs < 256; cs++) if (MS.permuteCase(cs, R.perm) === cs) fixed++;
  check(fixed / 24 === 23, "Burnside count " + fixed / 24);
  // the stored rotation maps a case to its representative
  for (let cs = 0; cs < 256; cs++) {
    const c = MS.CLASSES_23.of[cs];
    check(MS.permuteCase(cs, MS.CUBE_ROTATIONS[c.rotation].perm) === c.rep, `rotation to rep ${cs}`);
    const d = MS.CLASSES_15.of[cs];
    check(MS.permuteCase(d.complemented ? 255 - cs : cs, MS.CUBE_ROTATIONS[d.rotation].perm) === d.rep, `rotation to rep15 ${cs}`);
  }
  // rotation axis/angle reproduce the matrix (Rodrigues)
  for (const R of MS.CUBE_ROTATIONS) {
    const [x, y, z] = R.axis, c = Math.cos(R.angle), s = Math.sin(R.angle), C = 1 - c;
    const M = [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
    check(M.every((m, i) => near(m, R.matrix[i], 1e-9)), "axis-angle of rotation");
  }
  // loops: each cut edge exactly once, for all join masks
  for (let cs = 0; cs < 256; cs++) {
    const cut = MS.CUBE_EDGES.map(([a, b]) => ((cs >> a) & 1) !== ((cs >> b) & 1));
    for (const mask of [0, 63, 21, 42]) {
      const loops = MS.cubeLoops(cs, mask), used = new Array(12).fill(0);
      for (const L of loops) for (const e of L) used[e]++;
      check(used.every((u, e) => u === (cut[e] ? 1 : 0)), `loops use cut edges once, case ${cs}`);
    }
  }
  // classic counts: 15/23 class representatives; 6 classes (of 23 → under rotation, with
  // complement 15) contain an ambiguous face
  const ambClasses = new Set(MS.CLASSES_15.of.filter((_, cs) => MS.ambiguousFaces(cs).length > 0).map(c => c.rep));
  check(ambClasses.size === 6, "6 of the 15 classes have an ambiguous face: " + ambClasses.size);
  // max triangles per case = 4 with separate rule? (Lorensen–Cline: up to 4; with fan of 12-loops more)
  const tab = MS.triangleTable("separate");
  check(Math.max(...tab.map(t => t.length / 3)) <= 5, "at most 5 triangles per cube (separate rule)");
  check(tab[1].length === 3 && tab[0].length === 0 && tab[255].length === 0, "trivial cases");
}

// ------------------------------------------------------------------ marching cubes on volumes
function randomVolume(seed: number, n: number): Float32Array {
  const r = rng(seed), a = new Float32Array(n * n * n);
  for (let i = 0; i < a.length; i++) a[i] = r.unif();
  // a zero border so every surface closes
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
    if (x === 0 || y === 0 || z === 0 || x === n - 1 || y === n - 1 || z === n - 1) a[(z * n + y) * n + x] = 0;
  return a;
}

function testCubes(): void {
  for (let s = 1; s <= 12; s++) {
    const n = 7, d = randomVolume(s, n);
    for (const opts of [
      { faceRule: "separate" }, { faceRule: "join" }, { faceRule: "decider" },
      { algorithm: "mt6" }, { algorithm: "mt5" },
    ] as MS.CubesOptions[]) {
      const m = MS.marchingCubes(d, n, n, n, 0.5, opts), T = meshTopology(m);
      check(T.closed, `closed mesh seed ${s} ${JSON.stringify(opts)}: ${JSON.stringify(T)}`);
      check(meshVolume(m) > 0, "positive volume");
    }
  }
  // the complement trick produces cracks somewhere in random volumes
  let cracks = 0;
  for (let s = 1; s <= 6; s++) { const n = 8, d = randomVolume(100 + s, n); cracks += meshTopology(MS.marchingCubes(d, n, n, n, 0.5, { faceRule: "complement" })).boundaryEdges; }
  check(cracks > 0, "complement rule gives cracks: " + cracks);

  // single bright voxel: octahedron, area √3, volume 1/6 (at c = 0.5)
  {
    const n = 3, d = new Float32Array(27); d[13] = 1;
    const m = MS.marchingCubes(d, n, n, n, 0.5), T = meshTopology(m);
    check(T.faces === 8 && T.vertices === 6 && T.euler === 2 && T.closed, "octahedron topology " + JSON.stringify(T));
    check(near(meshArea(m), Math.sqrt(3), 1e-6), "octahedron area " + meshArea(m));
    check(near(meshVolume(m), 1 / 6, 1e-6), "octahedron volume " + meshVolume(m));
    for (const alg of ["mt5", "mt6"] as const) {
      const t = MS.marchingCubes(d, n, n, n, 0.5, { algorithm: alg });
      check(meshTopology(t).closed, "MT single voxel closed " + alg);
    }
  }
  // sphere: area and volume converge
  let prevErr = Infinity;
  for (const R of [4, 8, 16]) {
    const v = V.sphereSDF(2 * R + 5, R), m = MS.marchingCubes(v.data, v.nx, v.ny, v.nz, 0);
    const eA = Math.abs(meshArea(m) / (4 * Math.PI * R * R) - 1), eV = Math.abs(meshVolume(m) / (4 / 3 * Math.PI * R ** 3) - 1);
    check(eA < 0.03 && eV < (R < 5 ? 0.05 : 0.03), `sphere R=${R}: area err ${eA}, volume err ${eV}`);
    check(eA < prevErr, "area error decreases");
    prevErr = eA;
    check(meshTopology(m).euler === 2, "sphere χ = 2");
  }
  // torus χ = 0, one component
  {
    const v = V.torusSDF(32, 9, 4), m = MS.marchingCubes(v.data, v.nx, v.ny, v.nz, 0), T = meshTopology(m);
    check(T.euler === 0 && T.components === 1 && T.closed, "torus χ = 0 " + JSON.stringify(T));
    const mt = MS.marchingCubes(v.data, v.nx, v.ny, v.nz, 0, { algorithm: "mt6" });
    check(meshTopology(mt).euler === 0, "torus χ = 0 with tetrahedra");
  }
  // spacing scales the volume
  {
    const v = V.sphereSDF(21, 8);
    const m1 = MS.marchingCubes(v.data, 21, 21, 21, 0), m2 = MS.marchingCubes(v.data, 21, 21, 21, 0, { spacing: [1, 1, 2] });
    check(near(meshVolume(m2) / meshVolume(m1), 2, 1e-6), "spacing doubles volume");
  }
  // voxel faces: the area of exposed faces of a digitised ball is about 1.5× the sphere's
  {
    const R = 16, n = 2 * R + 4, c = (n - 1) / 2;
    let faces = 0;
    const inside = (x: number, y: number, z: number) => Math.hypot(x - c, y - c, z - c) <= R;
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (inside(x, y, z))
      for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) if (!inside(x + dx, y + dy, z + dz)) faces++;
    const ratio = faces / (4 * Math.PI * R * R);
    check(ratio > 1.4 && ratio < 1.6, "voxel-face area ratio " + ratio);
  }
  // decimation reduces triangles and keeps area roughly; smoothing shrinks
  {
    const v = V.sphereSDF(37, 16), m = MS.marchingCubes(v.data, 37, 37, 37, 0);
    const d = vertexCluster(m, 3);
    check(d.indices.length < m.indices.length / 4, "decimation reduces triangles");
    check(near(meshArea(d) / meshArea(m), 1, 0.1), "decimated area within 10%");
    const sm = laplacianSmooth(m, 20, 0.5);
    check(meshVolume({ positions: sm, indices: m.indices }) < meshVolume(m), "Laplacian smoothing shrinks");
  }
}

// ------------------------------------------------------------------ volumes
function testVolumes(): void {
  const v = V.nucleiStack();
  v.origin = [10, -5, 3];
  const A = V.voxelToWorld(v), Ai = V.worldToVoxel(v);
  const r = rng(5);
  for (let k = 0; k < 100; k++) {
    const p: V.Vec3 = [r.unif() * 60, r.unif() * 60, r.unif() * 18];
    const q = V.applyAffine(Ai, V.applyAffine(A, p));
    check(near(q[0], p[0], 1e-9) && near(q[1], p[1], 1e-9) && near(q[2], p[2], 1e-9), "voxel->world->voxel");
  }
  // rotated axes too
  const w = V.cloneVolume(v); w.axes = V.rotY(0.3);
  const B = V.voxelToWorld(w), Bi = V.worldToVoxel(w);
  const q = V.applyAffine(Bi, V.applyAffine(B, [3, 4, 5]));
  check(near(q[0], 3, 1e-9) && near(q[1], 4, 1e-9) && near(q[2], 5, 1e-9), "rotated affine round trip");
  // axis-aligned reslice = array indexing
  v.origin = [0, 0, 0];
  const k = 7, sl = V.sliceAxis(v, "z", k);
  const plane: V.Plane = { centre: [(v.nx - 1) / 2 * 0.2, (v.ny - 1) / 2 * 0.2, k * 0.5], u: [1, 0, 0], w: [0, 1, 0], width: v.nx, height: v.ny, step: 0.2 };
  const rs = V.reslice(v, plane);
  let maxd = 0;
  for (let i = 0; i < sl.data.length; i++) maxd = Math.max(maxd, Math.abs(sl.data[i] - rs.data[i]));
  check(maxd < 1e-3, "axis-aligned reslice equals indexing " + maxd);
  // tilted plane with zero tilt is an xy plane
  const tp = V.tiltedPlane([1, 2, 3], 0, 0, 4, 4, 1);
  check(tp.u.join() === "1,0,0" && near(tp.w[1], 1, 1e-12), "zero tilt");
  // resampling a constant stays constant
  const cst = V.createVolume(10, 10, 4, [0.2, 0.2, 0.5], 3.5);
  const iso = V.resampleIsotropic(cst);
  check(iso.nz === 10 && iso.data.every(a => near(a, 3.5, 1e-6)), "resampled constant");
  // trilinear exact on a linear function
  const lin = V.volumeFromFunction(5, 5, 5, [1, 1, 1], (x, y, z) => 2 * x - y + 3 * z + 1);
  for (let t = 0; t < 50; t++) {
    const x = r.unif() * 4, y = r.unif() * 4, z = r.unif() * 4;
    check(near(V.trilinear(lin, x, y, z), 2 * x - y + 3 * z + 1, 1e-5), "trilinear linear");
  }
  // gradient of a linear function in physical units
  const lin2 = V.volumeFromFunction(6, 6, 6, [0.5, 0.5, 2], (x, y, z) => 2 * x - y + 3 * z);
  const g = V.gradient3d(lin2), i0 = V.vidx(lin2, 2, 3, 3);
  check(near(g.gx[i0], 2, 1e-4) && near(g.gy[i0], -1, 1e-4) && near(g.gz[i0], 3, 1e-4), "physical gradient");
  // blur keeps the mean of a constant-border volume approximately and a constant exactly
  const b = V.gaussianBlur3d(cst, 0.6);
  check(b.data.every(a => near(a, 3.5, 1e-5)), "blur of constant");
  // MIP of the nuclei stack: A and B overlap in the projection, separate in 3D
  const m = V.mip(V.nucleiStack({ noise: 0 }), "z");
  check(m.width === 64 && m.height === 64, "mip size");
  // gradient normals on a sphere point outwards
  const sv = V.ballVolume(21, 7, [1, 1, 1], 1);
  const mesh = MS.marchingCubes(sv.data, 21, 21, 21, 0.5);
  const nrm = V.gradientNormals(sv, mesh.voxelPositions);
  let bad = 0;
  for (let i = 0; i < nrm.length; i += 3) {
    const p = [mesh.positions[i] - 10, mesh.positions[i + 1] - 10, mesh.positions[i + 2] - 10];
    if (p[0] * nrm[i] + p[1] * nrm[i + 1] + p[2] * nrm[i + 2] < 0.9 * Math.hypot(p[0], p[1], p[2])) bad++;
  }
  check(bad === 0, "gradient normals radial, bad = " + bad);
}

testSquares();
testCube();
testCubes();
testVolumes();
console.log(`marchingcubes: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
