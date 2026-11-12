// Marching squares and marching cubes/tetrahedra (owner: marching-cubes day). The only
// iso-contour code of the course: the level-set day draws its zero contours with
// `marchingSquares` too. No DOM: safe to import from tests.
//
//   const C = marchingSquares(values, w, h, c, { saddle: "decider" });   // 2D contour lines
//   C.lines            // chained polylines (closed loops, or open at the grid border)
//   contourLength(C), contourArea(C)                                    // shoelace area
//   const M = marchingCubes(vol.data, nx, ny, nz, c, { faceRule: "separate", spacing });
//   M.positions, M.indices                                              // indexed triangle mesh
//
// Conventions (printed on the exercise sheets)
//  - A grid point is "above" (inside a bright object) when its value is > c.
//  - Square cell: corners 0 = (x, y), 1 = (x+1, y), 2 = (x+1, y+1), 3 = (x, y+1) (image axes,
//    y down: top-left, top-right, bottom-right, bottom-left); edge k joins corner k and k+1
//    (0 top, 1 right, 2 bottom, 3 left). Case index = Σ 2^k over the above corners (0..15).
//  - Cube cell: corner k = (x, y, z) offsets (k & 1, (k >> 1) & 1, (k >> 2) & 1), i.e.
//    k = x + 2y + 4z; case index = Σ 2^k over the above corners (0..255). Edges 0–3 run along
//    x (corners 0-1, 2-3, 4-5, 6-7), 4–7 along y (0-2, 1-3, 4-6, 5-7), 8–11 along z
//    (0-4, 1-5, 2-6, 3-7).
//  - Crossing point on an edge from corner a to b: t = (c − I_a) / (I_b − I_a) ("linear"), or
//    t = 1/2 ("midpoint", the contour of the thresholded image).
//  - Orientation: 2D contours run with the above region on a fixed side, so the shoelace
//    sum is positive around bright blobs and negative around holes; 3D triangles are wound
//    counter-clockwise seen from OUTSIDE the bright object (normals point to lower values),
//    so the signed volume of a closed surface around a bright object is positive.
//
// The cube triangle table is not copied from anywhere: it is GENERATED from a face rule.
// The surface meets each face of a cube in a marching-squares contour; chaining the face
// segments of the 6 faces gives closed loops, and each loop is triangulated as a fan.

export type SaddleRule = "separate" | "join" | "decider";
export type FaceRule = "separate" | "join" | "decider" | "complement";
export type EdgeInterp = "linear" | "midpoint";

/** Crossing parameter on an edge from value a to value b. */
export function edgeT(a: number, b: number, c: number, interp: EdgeInterp = "linear"): number {
  if (interp === "midpoint" || a === b) return 0.5;
  const t = (c - a) / (b - a);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/**
 * The contour segments inside one polygonal cell, given the above/below state of its corners
 * in cyclic order. Edge k joins corner k and k + 1 (mod n). Returns segments [from, to] as
 * edge indices, oriented from the crossing where the boundary walk LEAVES the above region to
 * the one where it ENTERS it. With 4 crossings (a saddle), `join` decides whether the above
 * corners are connected through the cell (true) or separated (false).
 */
export function cycleSegments(above: ArrayLike<boolean>, join: boolean): [number, number][] {
  const n = above.length, xs: { e: number; out: boolean }[] = [];
  for (let k = 0; k < n; k++) {
    const a = above[k], b = above[(k + 1) % n];
    if (a !== b) xs.push({ e: k, out: a });
  }
  const m = xs.length, segs: [number, number][] = [];
  for (let i = 0; i < m; i++) {
    if (!xs[i].out) continue;
    const partner = join ? xs[(i + 1) % m] : xs[(i - 1 + m) % m];
    segs.push([xs[i].e, partner.e]);
  }
  return segs;
}

// =========================================================================================
// Marching squares

export const SQ_CORNERS: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
export const SQ_EDGES: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 0]];

/** Case index (0..15) of a square cell from its 4 corner values (order 0..3 as above). */
export function squareCase(v: ArrayLike<number>, c: number): number {
  return (v[0] > c ? 1 : 0) | (v[1] > c ? 2 : 0) | (v[2] > c ? 4 : 0) | (v[3] > c ? 8 : 0);
}

export const isSquareSaddle = (cs: number): boolean => cs === 5 || cs === 10;

/** Saddle value of the bilinear interpolant of a cell: f* = (I00 I11 − I10 I01) / (I00 + I11 − I10 − I01),
 *  with I00 = v0, I10 = v1, I11 = v2, I01 = v3. NaN when the denominator is 0 (no saddle). */
export function saddleValue(v: ArrayLike<number>): number {
  const den = v[0] + v[2] - v[1] - v[3];
  return den === 0 ? NaN : (v[0] * v[2] - v[1] * v[3]) / den;
}

/** Asymptotic decider (Nielson & Hamann 1991): are the above corners of a saddle cell connected? */
export function deciderJoins(v: ArrayLike<number>, c: number): boolean {
  const f = saddleValue(v);
  return Number.isFinite(f) ? f > c : false;
}

/** Should the above corners of this cell be joined, under a rule? (Only matters for saddles.) */
export function saddleJoin(v: ArrayLike<number>, c: number, rule: SaddleRule): boolean {
  return rule === "join" ? true : rule === "separate" ? false : deciderJoins(v, c);
}

/** The segments of one square cell, as pairs of cell edge indices (0..3), oriented. */
export function squareSegments(v: ArrayLike<number>, c: number, rule: SaddleRule = "decider"): [number, number][] {
  const ab = [v[0] > c, v[1] > c, v[2] > c, v[3] > c];
  return cycleSegments(ab, saddleJoin(v, c, rule));
}

/** Position of the crossing on edge e (0..3) of a unit cell, in cell coordinates. */
export function squareEdgePoint(v: ArrayLike<number>, c: number, e: number, interp: EdgeInterp = "linear"): [number, number] {
  const [a, b] = SQ_EDGES[e], t = edgeT(v[a], v[b], c, interp);
  const pa = SQ_CORNERS[a], pb = SQ_CORNERS[b];
  return [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
}

export interface ContourLine { ids: number[]; closed: boolean }

export interface Contour {
  /** Crossing points, x y pairs in grid coordinates (x = column, y = row). */
  points: Float64Array;
  /** Oriented segments as pairs of point ids. */
  segments: Int32Array;
  /** Segments chained into polylines. */
  lines: ContourLine[];
}

export interface SquaresOptions { saddle?: SaddleRule; interp?: EdgeInterp }

/**
 * Marching squares on a w × h grid of values (row-major, y down). Each cell is handled on
 * its own; a crossing point on an edge is computed once and shared by the two cells, so the
 * lines join up into closed loops (or end on the grid border).
 */
export function marchingSquares(values: ArrayLike<number>, w: number, h: number, c: number, opts: SquaresOptions = {}): Contour {
  const rule = opts.saddle ?? "decider", interp = opts.interp ?? "linear";
  const nH = (w - 1) * h;
  const idOf = new Int32Array(nH + w * (h - 1)).fill(-1);
  const pts: number[] = [], segs: number[] = [];
  const point = (gid: number, x0: number, y0: number, x1: number, y1: number): number => {
    if (idOf[gid] >= 0) return idOf[gid];
    const a = values[y0 * w + x0], b = values[y1 * w + x1], t = edgeT(a, b, c, interp);
    idOf[gid] = pts.length / 2;
    pts.push(x0 + t * (x1 - x0), y0 + t * (y1 - y0));
    return idOf[gid];
  };
  const v = [0, 0, 0, 0];
  for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
    v[0] = values[y * w + x]; v[1] = values[y * w + x + 1]; v[2] = values[(y + 1) * w + x + 1]; v[3] = values[(y + 1) * w + x];
    const cs = squareCase(v, c);
    if (cs === 0 || cs === 15) continue;
    const edgePt = (e: number): number =>
      e === 0 ? point(y * (w - 1) + x, x, y, x + 1, y)
        : e === 1 ? point(nH + y * w + x + 1, x + 1, y, x + 1, y + 1)
          : e === 2 ? point((y + 1) * (w - 1) + x, x, y + 1, x + 1, y + 1)
            : point(nH + y * w + x, x, y, x, y + 1);
    for (const [e0, e1] of squareSegments(v, c, rule)) segs.push(edgePt(e0), edgePt(e1));
  }
  const points = Float64Array.from(pts), segments = Int32Array.from(segs);
  return { points, segments, lines: chainSegments(points.length / 2, segments) };
}

/** Chain oriented segments (each point has at most one outgoing and one incoming) into lines. */
export function chainSegments(nPoints: number, segments: ArrayLike<number>): ContourLine[] {
  const next = new Int32Array(nPoints).fill(-1), hasIn = new Uint8Array(nPoints), used = new Uint8Array(nPoints);
  for (let i = 0; i < segments.length; i += 2) { next[segments[i]] = segments[i + 1]; hasIn[segments[i + 1]] = 1; }
  const lines: ContourLine[] = [];
  const follow = (s: number): number[] => {
    const ids = [s]; used[s] = 1;
    let p = next[s];
    while (p >= 0 && !used[p]) { ids.push(p); used[p] = 1; p = next[p]; }
    return ids;
  };
  for (let p = 0; p < nPoints; p++) if (!hasIn[p] && next[p] >= 0 && !used[p]) lines.push({ ids: follow(p), closed: false });
  for (let p = 0; p < nPoints; p++) if (!used[p] && next[p] >= 0) lines.push({ ids: follow(p), closed: true });
  return lines;
}

/** Length of one line (or of all lines). */
export function lineLength(C: Contour, L: ContourLine): number {
  const P = C.points, ids = L.ids;
  let s = 0;
  for (let i = 0; i + 1 < ids.length; i++) s += Math.hypot(P[2 * ids[i + 1]] - P[2 * ids[i]], P[2 * ids[i + 1] + 1] - P[2 * ids[i] + 1]);
  if (L.closed && ids.length > 1) { const a = ids[ids.length - 1], b = ids[0]; s += Math.hypot(P[2 * b] - P[2 * a], P[2 * b + 1] - P[2 * a + 1]); }
  return s;
}
export function contourLength(C: Contour): number { return C.lines.reduce((s, L) => s + lineLength(C, L), 0); }

/** Signed area of a closed line by the shoelace formula ½ Σ (xᵢ yᵢ₊₁ − xᵢ₊₁ yᵢ): positive
 *  around a bright blob, negative around a hole (with the orientation of this file). */
export function lineArea(C: Contour, L: ContourLine): number {
  if (!L.closed) return 0;
  const P = C.points, ids = L.ids;
  let s = 0;
  for (let i = 0; i < ids.length; i++) {
    const a = ids[i], b = ids[(i + 1) % ids.length];
    s += P[2 * a] * P[2 * b + 1] - P[2 * b] * P[2 * a + 1];
  }
  return s / 2;
}
/** Enclosed area: the sum of the signed areas of all closed lines (holes subtract). */
export function contourArea(C: Contour): number { return C.lines.reduce((s, L) => s + lineArea(C, L), 0); }

/** Copy of a grid with a border of `pad` cells of value `value`, so every contour closes. */
export function padGrid(values: ArrayLike<number>, w: number, h: number, value: number, pad = 1): { values: Float64Array; w: number; h: number } {
  const W = w + 2 * pad, H = h + 2 * pad, out = new Float64Array(W * H).fill(value);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[(y + pad) * W + x + pad] = values[y * w + x];
  return { values: out, w: W, h: H };
}

// =========================================================================================
// The cube: corners, edges, faces, rotations, classes

export const CUBE_CORNERS: [number, number, number][] = Array.from({ length: 8 }, (_, k) => [k & 1, (k >> 1) & 1, (k >> 2) & 1]);
export const CUBE_EDGES: [number, number][] = [
  [0, 1], [2, 3], [4, 5], [6, 7],
  [0, 2], [1, 3], [4, 6], [5, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
];
/** Faces as corner cycles, counter-clockwise seen from outside the cube: −z, +z, −x, +x, −y, +y. */
export const CUBE_FACES: number[][] = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 4, 6, 2], [5, 1, 3, 7], [0, 1, 5, 4], [6, 7, 3, 2]];
export const FACE_NAMES = ["−z", "+z", "−x", "+x", "−y", "+y"];

const EDGE_OF = new Int8Array(64).fill(-1);
CUBE_EDGES.forEach(([a, b], e) => { EDGE_OF[a * 8 + b] = e; EDGE_OF[b * 8 + a] = e; });
/** Cube edge joining corners a and b (−1 if they are not neighbours). */
export const cubeEdge = (a: number, b: number): number => EDGE_OF[a * 8 + b];

export const popcount = (x: number): number => { let n = 0; while (x) { n += x & 1; x >>= 1; } return n; };

/** Is face f ambiguous for this case (diagonal corners above, the other two below)? */
export function faceAmbiguous(cs: number, f: number): boolean {
  const q = CUBE_FACES[f], b = q.map(k => (cs >> k) & 1);
  return b[0] === b[2] && b[1] === b[3] && b[0] !== b[1];
}
export const ambiguousFaces = (cs: number): number[] => [0, 1, 2, 3, 4, 5].filter(f => faceAmbiguous(cs, f));

/**
 * Closed loops of cube edges for a case, given which faces join their above corners
 * (bit f of `joinMask`; only ambiguous faces care). Loops are chains of face segments.
 */
export function cubeLoops(cs: number, joinMask = 0): number[][] {
  const next = new Int8Array(12).fill(-1);
  for (let f = 0; f < 6; f++) {
    const q = CUBE_FACES[f], ab = q.map(k => ((cs >> k) & 1) === 1);
    for (const [e0, e1] of cycleSegments(ab, ((joinMask >> f) & 1) === 1))
      next[cubeEdge(q[e0], q[(e0 + 1) % 4])] = cubeEdge(q[e1], q[(e1 + 1) % 4]);
  }
  const used = new Uint8Array(12), loops: number[][] = [];
  for (let e = 0; e < 12; e++) {
    if (next[e] < 0 || used[e]) continue;
    const L: number[] = [];
    let p = e;
    while (!used[p]) { used[p] = 1; L.push(p); p = next[p]; }
    loops.push(L);
  }
  return loops;
}

/** The faces (0..5) each cube edge lies on. */
const EDGE_FACES: number[][] = CUBE_EDGES.map(([a, b]) => CUBE_FACES.map((q, f) => (q.includes(a) && q.includes(b) ? f : -1)).filter(f => f >= 0));
const shareFace = (e1: number, e2: number): boolean => EDGE_FACES[e1].some(f => EDGE_FACES[e2].includes(f));

/**
 * Triangulate the loops, wound so that normals point to the below (outside) side. Each loop
 * is a fan from one of its vertices, chosen so that no fan diagonal lies on a cube face (a
 * diagonal on a face could be repeated by the neighbouring cube and make the mesh
 * non-manifold). Where no such vertex exists (some long loops with saddle faces), the fan
 * is built around an extra vertex at the loop's centre, referenced as 12 + loop index.
 */
export function loopsToTriangles(loops: number[][]): number[] {
  const t: number[] = [];
  loops.forEach((L, li) => {
    const n = L.length;
    let start = -1;
    for (let s = 0; s < n && start < 0; s++) {
      let ok = true;
      for (let i = 2; i < n - 1 && ok; i++) if (shareFace(L[s], L[(s + i) % n])) ok = false;
      if (ok) start = s;
    }
    if (start >= 0) {
      const R = L.slice(start).concat(L.slice(0, start));
      for (let i = 1; i + 1 < n; i++) t.push(R[0], R[i + 1], R[i]);
    } else {
      for (let i = 0; i < n; i++) t.push(12 + li, L[(i + 1) % n], L[i]);
    }
  });
  return t;
}

/** The join mask a face rule gives for a case (for "decider", pass the corner values). */
export function faceJoinMask(cs: number, rule: FaceRule, values?: ArrayLike<number>, c = 0): number {
  if (rule === "separate") return 0;
  if (rule === "join") return 63;
  if (rule === "complement") return popcount(cs) > 4 || (popcount(cs) === 4 && !(cs & 1)) ? 63 : 0;
  let m = 0;
  for (let f = 0; f < 6; f++) {
    if (!faceAmbiguous(cs, f) || !values) continue;
    const q = CUBE_FACES[f];
    if (deciderJoins([values[q[0]], values[q[1]], values[q[2]], values[q[3]]], c)) m |= 1 << f;
  }
  return m;
}

export interface CaseTriangulation {
  /** Edge triples; an entry 12 + i means the centre of loops[i]. */
  tris: Int8Array;
  loops: number[][];
}
const TRI_CACHE = new Map<number, CaseTriangulation>();
/** Triangulation of a case with a face join mask; cached. */
export function caseTriangulation(cs: number, joinMask = 0): CaseTriangulation {
  const amb = ambiguousFaces(cs).reduce((m, f) => m | (1 << f), 0), key = cs * 64 + (joinMask & amb);
  let t = TRI_CACHE.get(key);
  if (!t) {
    const loops = cubeLoops(cs, joinMask & amb);
    t = { tris: Int8Array.from(loopsToTriangles(loops)), loops };
    TRI_CACHE.set(key, t);
  }
  return t;
}
/** Triangles (edge triples, with 12 + i = centre of loop i) of a case. */
export const caseTriangles = (cs: number, joinMask = 0): Int8Array => caseTriangulation(cs, joinMask).tris;

/** The generated triangle table for a fixed face rule: 256 lists of edge triples. */
export function triangleTable(rule: "separate" | "join" | "complement" = "separate"): Int8Array[] {
  return Array.from({ length: 256 }, (_, cs) => caseTriangles(cs, faceJoinMask(cs, rule)));
}

// ---- the 24 rotations of the cube as corner permutations ----

type IMat = number[]; // row-major 3×3 integer
const imul = (A: IMat, B: IMat): IMat => [0, 1, 2].flatMap(r => [0, 1, 2].map(c => A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]));
const RX90: IMat = [1, 0, 0, 0, 0, -1, 0, 1, 0], RY90: IMat = [0, 0, 1, 0, 1, 0, -1, 0, 0];

export interface CubeRotation {
  /** Row-major 3×3 rotation matrix (about the cube centre). */
  matrix: number[];
  /** perm[k] = corner that corner k moves to. */
  perm: number[];
  /** Axis (unit) and angle (radians), for animation. */
  axis: [number, number, number];
  angle: number;
}

function rotationFromMatrix(R: IMat): CubeRotation {
  const perm = CUBE_CORNERS.map(([x, y, z]) => {
    const p = [2 * x - 1, 2 * y - 1, 2 * z - 1];
    const q = [0, 1, 2].map(r => R[3 * r] * p[0] + R[3 * r + 1] * p[1] + R[3 * r + 2] * p[2]);
    return (q[0] > 0 ? 1 : 0) + (q[1] > 0 ? 2 : 0) + (q[2] > 0 ? 4 : 0);
  });
  const tr = R[0] + R[4] + R[8], angle = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
  let axis: [number, number, number] = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
  if (angle < 1e-9) axis = [0, 0, 1];
  else if (Math.abs(angle - Math.PI) < 1e-9) {
    // 180°: axis from R + I (any non-zero column)
    const B = [R[0] + 1, R[1], R[2], R[3], R[4] + 1, R[5], R[6], R[7], R[8] + 1];
    const col = [0, 1, 2].map(c => [B[c], B[3 + c], B[6 + c]]).find(v => Math.hypot(...v) > 1e-9)!;
    axis = col as [number, number, number];
  }
  const l = Math.hypot(...axis) || 1;
  return { matrix: R, perm, axis: [axis[0] / l, axis[1] / l, axis[2] / l], angle };
}

/** The 24 rotations of the cube (identity first). */
export const CUBE_ROTATIONS: CubeRotation[] = (() => {
  const seen = new Map<string, IMat>([["1,0,0,0,1,0,0,0,1", [1, 0, 0, 0, 1, 0, 0, 0, 1]]]);
  const queue: IMat[] = [[1, 0, 0, 0, 1, 0, 0, 0, 1]];
  while (queue.length) {
    const A = queue.shift()!;
    for (const G of [RX90, RY90]) {
      const B = imul(G, A), k = B.join(",");
      if (!seen.has(k)) { seen.set(k, B); queue.push(B); }
    }
  }
  return [...seen.values()].map(rotationFromMatrix);
})();

/** Apply a corner permutation to a case index. */
export function permuteCase(cs: number, perm: number[]): number {
  let out = 0;
  for (let k = 0; k < 8; k++) if ((cs >> k) & 1) out |= 1 << perm[k];
  return out;
}

export interface CaseClass {
  /** Representative case (the smallest index of the class). */
  rep: number;
  /** Class number (classes sorted by number of above corners, then representative). */
  id: number;
  /** Index into CUBE_ROTATIONS of a rotation taking this case to the representative. */
  rotation: number;
  /** For the 15-class version: whether the complement was used. */
  complemented: boolean;
}

function buildClasses(withComplement: boolean): { of: CaseClass[]; reps: number[] } {
  const repOf = new Int32Array(256).fill(-1), rotOf = new Int32Array(256), compOf = new Uint8Array(256);
  for (let cs = 0; cs < 256; cs++) {
    let best = 1e9, br = 0, bc = 0;
    for (const comp of withComplement ? [0, 1] : [0]) {
      const base = comp ? 255 - cs : cs;
      CUBE_ROTATIONS.forEach((R, r) => { const q = permuteCase(base, R.perm); if (q < best) { best = q; br = r; bc = comp; } });
    }
    repOf[cs] = best; rotOf[cs] = br; compOf[cs] = bc;
  }
  const reps = [...new Set(Array.from(repOf))].sort((a, b) => popcount(a) - popcount(b) || a - b);
  const of = Array.from({ length: 256 }, (_, cs) => ({ rep: repOf[cs], id: reps.indexOf(repOf[cs]), rotation: rotOf[cs], complemented: compOf[cs] === 1 }));
  return { of, reps };
}

/** Classes under the 24 rotations (23 classes) and under rotations + complement (15 classes). */
export const CLASSES_23 = buildClasses(false);
export const CLASSES_15 = buildClasses(true);

// =========================================================================================
// Marching cubes and marching tetrahedra on a volume

export type Algorithm = "mc" | "mt5" | "mt6";

export interface CubesOptions {
  faceRule?: FaceRule;
  interp?: EdgeInterp;
  algorithm?: Algorithm;
  /** Voxel spacing (Δx, Δy, Δz) applied to the output positions (default 1, 1, 1). */
  spacing?: [number, number, number];
  /** World position of voxel (0, 0, 0) (default 0). */
  origin?: [number, number, number];
}

export interface TriMesh {
  /** World positions x y z per vertex. */
  positions: Float32Array;
  /** Positions in voxel index coordinates (for gradient normals). */
  voxelPositions: Float32Array;
  /** Three vertex indices per triangle. */
  indices: Uint32Array;
}

/**
 * Marching cubes (or tetrahedra) on a volume stored (z, y, x). One pass over the cells; a
 * vertex on a grid edge is computed once and shared, so the mesh is indexed and, with a
 * consistent face rule, closed wherever the surface does not leave the volume.
 */
export function marchingCubes(data: ArrayLike<number>, nx: number, ny: number, nz: number, c: number, opts: CubesOptions = {}): TriMesh {
  const alg = opts.algorithm ?? "mc";
  const interp = opts.interp ?? "linear", rule = opts.faceRule ?? "separate";
  const sp = opts.spacing ?? [1, 1, 1], org = opts.origin ?? [0, 0, 0];
  const vp: number[] = [], tri: number[] = [];
  const P = (x: number, y: number, z: number) => (z * ny + y) * nx + x;
  const vertexAt = (pa: number, pb: number): [number, number, number] => {
    const a = data[pa], b = data[pb], t = edgeT(a, b, c, interp);
    const ax = pa % nx, ay = Math.floor(pa / nx) % ny, az = Math.floor(pa / (nx * ny));
    const bx = pb % nx, by = Math.floor(pb / nx) % ny, bz = Math.floor(pb / (nx * ny));
    return [ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az)];
  };
  const pushVertex = (p: [number, number, number]): number => { vp.push(p[0], p[1], p[2]); return vp.length / 3 - 1; };

  if (alg === "mc") {
    const N = nx * ny * nz, idOf = new Int32Array(3 * N).fill(-1);
    const corner = new Int32Array(8), val = new Float64Array(8), eid = new Int32Array(24);
    for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
      let cs = 0;
      for (let k = 0; k < 8; k++) {
        const p = P(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1));
        corner[k] = p; val[k] = data[p];
        if (val[k] > c) cs |= 1 << k;
      }
      if (cs === 0 || cs === 255) continue;
      const tr = caseTriangulation(cs, faceJoinMask(cs, rule, val, c)), t = tr.tris;
      eid.fill(-1);
      for (let i = 0; i < t.length; i++) {
        const e = t[i];
        if (e >= 12) {
          // centre of a loop: a vertex of this cube only
          if (eid[e] < 0) {
            const L = tr.loops[e - 12], m: [number, number, number] = [0, 0, 0];
            for (const f of L) { const [a, b] = CUBE_EDGES[f], q = vertexAt(corner[a], corner[b]); m[0] += q[0] / L.length; m[1] += q[1] / L.length; m[2] += q[2] / L.length; }
            eid[e] = pushVertex(m);
          }
          tri.push(eid[e]);
          continue;
        }
        if (eid[e] < 0) {
          const [a, b] = CUBE_EDGES[e], pa = corner[a], axis = e >> 2; // 0: x, 1: y, 2: z
          const key = 3 * pa + axis;
          if (idOf[key] < 0) idOf[key] = pushVertex(vertexAt(pa, corner[b]));
          eid[e] = idOf[key];
        }
        tri.push(eid[e]);
      }
    }
  } else {
    const ids = new Map<number, number>(), N = nx * ny * nz;
    const vid = (pa: number, pb: number): number => {
      const key = pa < pb ? pa * N + pb : pb * N + pa;
      let id = ids.get(key);
      if (id === undefined) { id = pushVertex(vertexAt(pa, pb)); ids.set(key, id); }
      return id;
    };
    const pos = (p: number): [number, number, number] => [p % nx, Math.floor(p / nx) % ny, Math.floor(p / (nx * ny))];
    const corner = new Int32Array(8);
    for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
      let any = 0, all = 1;
      for (let k = 0; k < 8; k++) {
        corner[k] = P(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1));
        const ab = data[corner[k]] > c ? 1 : 0; any |= ab; all &= ab;
      }
      if (!any || all) continue;
      const tets = alg === "mt6" ? TETS6 : (x + y + z) % 2 === 0 ? TETS5A : TETS5B;
      for (const T of tets) tetTriangles(T.map(k => corner[k]), data, c, vid, pos, tri);
    }
  }
  const voxelPositions = Float32Array.from(vp), positions = new Float32Array(vp.length);
  for (let i = 0; i < vp.length; i += 3) for (let a = 0; a < 3; a++) positions[i + a] = org[a] + sp[a] * vp[i + a];
  return { positions, voxelPositions, indices: Uint32Array.from(tri) };
}

/** Six tetrahedra around the cube diagonal 0–7 (the same split in every cube). */
export const TETS6: number[][] = [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]];
/** Five tetrahedra: four corner tetrahedra and a central one; the mirrored split (B) is used in
 *  every other cube so that the face diagonals of neighbours match. */
export const TETS5A: number[][] = [[0, 1, 2, 4], [3, 2, 1, 7], [5, 4, 7, 1], [6, 7, 4, 2], [1, 2, 4, 7]];
export const TETS5B: number[][] = [[1, 3, 0, 5], [2, 0, 3, 6], [4, 6, 5, 0], [7, 5, 6, 3], [0, 3, 5, 6]];

/** The triangles of one tetrahedron (corners = grid point ids), wound with normals towards
 *  the below corners. Exposed for the single-cube demo. */
export function tetTriangles(T: number[], data: ArrayLike<number>, c: number, vid: (a: number, b: number) => number,
  pos: (p: number) => [number, number, number], out: number[]): void {
  const up = T.filter(p => data[p] > c), dn = T.filter(p => !(data[p] > c));
  if (up.length === 0 || dn.length === 0) return;
  const cu = centroid(up.map(pos)), cd = centroid(dn.map(pos));
  const emit = (a: [number, number], b: [number, number], d: [number, number]) => {
    const pa = edgePos(a), pb = edgePos(b), pd = edgePos(d);
    const n = cross(sub(pb, pa), sub(pd, pa)), o = sub(cd, cu);
    const ia = vid(a[0], a[1]), ib = vid(b[0], b[1]), id = vid(d[0], d[1]);
    if (n[0] * o[0] + n[1] * o[1] + n[2] * o[2] >= 0) out.push(ia, ib, id); else out.push(ia, id, ib);
  };
  const edgePos = ([p, q]: [number, number]): [number, number, number] => {
    const t = edgeT(data[p], data[q], c), a = pos(p), b = pos(q);
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
  };
  if (up.length === 1 || dn.length === 1) {
    const [lone, rest] = up.length === 1 ? [up[0], dn] : [dn[0], up];
    emit([lone, rest[0]], [lone, rest[1]], [lone, rest[2]]);
  } else {
    const [a, b] = up, [d, e] = dn;
    // quad a-d, a-e, b-e, b-d (cyclic)
    emit([a, d], [a, e], [b, e]);
    emit([a, d], [b, e], [b, d]);
  }
}

const sub = (a: number[], b: number[]): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: number[], b: number[]): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const centroid = (ps: number[][]): [number, number, number] => {
  const s: [number, number, number] = [0, 0, 0];
  for (const p of ps) { s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; }
  return [s[0] / ps.length, s[1] / ps.length, s[2] / ps.length];
};

/** The surface pieces of a single cube (8 corner values), as positions in [0, 1]³ and triangles:
 *  for the one-cube demo. Each loop of the face rule is triangulated as a fan. */
export function singleCube(values: ArrayLike<number>, c: number, rule: FaceRule = "separate", interp: EdgeInterp = "linear"):
  { cs: number; loops: number[][]; edgePoints: ([number, number, number] | null)[]; triangles: number[] } {
  // edgePoints[e] for the 12 edges (null if not cut), then one centre per loop (index 12 + i);
  // triangles index into edgePoints.
  let cs = 0;
  for (let k = 0; k < 8; k++) if (values[k] > c) cs |= 1 << k;
  const loops = cubeLoops(cs, faceJoinMask(cs, rule, values, c) & ambiguousFaces(cs).reduce((m, f) => m | (1 << f), 0));
  const edgePoints = CUBE_EDGES.map(([a, b]) => {
    if ((values[a] > c) === (values[b] > c)) return null;
    const t = edgeT(values[a], values[b], c, interp), A = CUBE_CORNERS[a], B = CUBE_CORNERS[b];
    return [A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]), A[2] + t * (B[2] - A[2])] as [number, number, number];
  });
  for (const L of loops) {
    const m: [number, number, number] = [0, 0, 0];
    for (const e of L) for (let a = 0; a < 3; a++) m[a] += edgePoints[e]![a] / L.length;
    edgePoints.push(m);
  }
  return { cs, loops, edgePoints, triangles: loopsToTriangles(loops) };
}
