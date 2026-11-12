// Triangle meshes: measurement, topology checks, normals, decimation, smoothing (owner:
// marching-cubes day; the 3D-graphics day adds primitives). No DOM: safe for tests.
//
// A mesh is { positions: x y z per vertex, indices: three vertex ids per triangle }, wound
// counter-clockwise seen from outside (as marching cubes in lib/marching.ts produces it).
//
//   meshArea(m)              Σ ½ ‖(b − a) × (c − a)‖
//   meshVolume(m)            Σ ⅙ a · (b × c)  (divergence theorem; needs a closed mesh)
//   meshTopology(m)          vertices, edges, faces, Euler characteristic, boundary edges,
//                            components, orientation check
//   vertexCluster(m, h)      decimation by snapping vertices to a grid of cell size h
//   laplacianSmooth(m, k)    move each vertex towards the mean of its neighbours

export interface Mesh {
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
}

type V3 = [number, number, number];
const vtx = (P: ArrayLike<number>, i: number): V3 => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];

/** Total surface area. */
export function meshArea(m: Mesh): number {
  const P = m.positions, I = m.indices;
  let s = 0;
  for (let t = 0; t + 2 < I.length; t += 3) {
    const a = vtx(P, I[t]), b = vtx(P, I[t + 1]), c = vtx(P, I[t + 2]);
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    s += 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
  }
  return s;
}

/** Enclosed volume Σ ⅙ det[a b c] (signed tetrahedra from the origin). Positive for a closed
 *  mesh wound counter-clockwise from outside; meaningless for a mesh with holes. */
export function meshVolume(m: Mesh): number {
  const P = m.positions, I = m.indices;
  let s = 0;
  for (let t = 0; t + 2 < I.length; t += 3) {
    const a = vtx(P, I[t]), b = vtx(P, I[t + 1]), c = vtx(P, I[t + 2]);
    s += a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return s / 6;
}

export interface Topology {
  /** Vertices used by at least one triangle. */
  vertices: number;
  edges: number;
  faces: number;
  /** χ = V − E + F (2 per sphere-like closed component, 0 per torus). */
  euler: number;
  /** Edges used by exactly one triangle (cracks / holes / the volume border). */
  boundaryEdges: number;
  /** Edges used by more than two triangles. */
  nonManifoldEdges: number;
  /** Interior edges whose two triangles traverse them in the same direction. */
  misoriented: number;
  /** Connected components (via shared vertices). */
  components: number;
  /** Is every edge shared by exactly two oppositely oriented triangles? */
  closed: boolean;
}

/** Edge key of an undirected vertex pair. */
const ekey = (a: number, b: number, n: number) => (a < b ? a * n + b : b * n + a);

/** Topology of a mesh (degenerate triangles with repeated vertices are skipped). */
export function meshTopology(m: Mesh): Topology {
  const I = m.indices, nv = m.positions.length / 3;
  const count = new Map<number, number>(), dirSum = new Map<number, number>();
  const used = new Uint8Array(nv);
  const parent = Int32Array.from({ length: nv }, (_, i) => i);
  const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  let faces = 0;
  for (let t = 0; t + 2 < I.length; t += 3) {
    const a = I[t], b = I[t + 1], c = I[t + 2];
    if (a === b || b === c || a === c) continue;
    faces++;
    for (const [p, q] of [[a, b], [b, c], [c, a]]) {
      const k = ekey(p, q, nv);
      count.set(k, (count.get(k) ?? 0) + 1);
      dirSum.set(k, (dirSum.get(k) ?? 0) + (p < q ? 1 : -1));
      used[p] = 1;
      const rp = find(p), rq = find(q);
      if (rp !== rq) parent[rp] = rq;
    }
  }
  let boundaryEdges = 0, nonManifoldEdges = 0, misoriented = 0;
  for (const [k, n] of count) {
    if (n === 1) boundaryEdges++;
    else if (n > 2) nonManifoldEdges++;
    else if (dirSum.get(k) !== 0) misoriented++;
  }
  let vertices = 0, components = 0;
  for (let i = 0; i < nv; i++) if (used[i]) { vertices++; if (find(i) === i) components++; }
  const edges = count.size;
  return {
    vertices, edges, faces, euler: vertices - edges + faces, boundaryEdges, nonManifoldEdges, misoriented, components,
    closed: boundaryEdges === 0 && nonManifoldEdges === 0 && misoriented === 0,
  };
}

/** Boundary edges as line segments (x0 y0 z0 x1 y1 z1 each): where the mesh has cracks. */
export function boundarySegments(m: Mesh): Float32Array {
  const I = m.indices, P = m.positions, nv = P.length / 3;
  const count = new Map<number, number>();
  for (let t = 0; t + 2 < I.length; t += 3) {
    const a = I[t], b = I[t + 1], c = I[t + 2];
    if (a === b || b === c || a === c) continue;
    for (const [p, q] of [[a, b], [b, c], [c, a]]) { const k = ekey(p, q, nv); count.set(k, (count.get(k) ?? 0) + 1); }
  }
  const out: number[] = [];
  for (const [k, n] of count) if (n === 1) {
    const a = Math.floor(k / nv), b = k - a * nv;
    out.push(...vtx(P, a), ...vtx(P, b));
  }
  return Float32Array.from(out);
}

/** One unit normal per triangle (right-hand rule). */
export function faceNormals(m: Mesh): Float32Array {
  const P = m.positions, I = m.indices, n = new Float32Array(I.length);
  for (let t = 0; t + 2 < I.length; t += 3) {
    const a = vtx(P, I[t]), b = vtx(P, I[t + 1]), c = vtx(P, I[t + 2]);
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const x = u[1] * v[2] - u[2] * v[1], y = u[2] * v[0] - u[0] * v[2], z = u[0] * v[1] - u[1] * v[0], l = Math.hypot(x, y, z) || 1;
    n[t] = x / l; n[t + 1] = y / l; n[t + 2] = z / l;
  }
  return n;
}

/** Remove vertices no triangle uses and degenerate triangles; returns a compact mesh. */
export function compactMesh(m: Mesh): { positions: Float32Array; indices: Uint32Array; map: Int32Array } {
  const nv = m.positions.length / 3, map = new Int32Array(nv).fill(-1), P: number[] = [], I: number[] = [];
  for (let t = 0; t + 2 < m.indices.length; t += 3) {
    const a = m.indices[t], b = m.indices[t + 1], c = m.indices[t + 2];
    if (a === b || b === c || a === c) continue;
    for (const v of [a, b, c]) {
      if (map[v] < 0) { map[v] = P.length / 3; P.push(m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]); }
      I.push(map[v]);
    }
  }
  return { positions: Float32Array.from(P), indices: Uint32Array.from(I), map };
}

/**
 * Vertex-clustering decimation: snap every vertex to a grid of cell size h (cells anchored at
 * `origin`), merge the vertices of each cell into their mean, drop triangles that collapse
 * (two corners in the same cell) and duplicates. Simple and fast; may make the mesh
 * non-manifold where thin parts collapse.
 */
export function vertexCluster(m: Mesh, h: number, origin: [number, number, number] = [0, 0, 0]): { positions: Float32Array; indices: Uint32Array } {
  if (!(h > 0)) return compactMesh(m);
  const P = m.positions, nv = P.length / 3, cellOf = new Map<string, number>(), rep = new Int32Array(nv);
  const sum: number[] = [], cnt: number[] = [];
  for (let i = 0; i < nv; i++) {
    const k = [0, 1, 2].map(a => Math.floor((P[3 * i + a] - origin[a]) / h)).join(",");
    let id = cellOf.get(k);
    if (id === undefined) { id = cnt.length; cellOf.set(k, id); sum.push(0, 0, 0); cnt.push(0); }
    rep[i] = id; cnt[id]++;
    for (let a = 0; a < 3; a++) sum[3 * id + a] += P[3 * i + a];
  }
  const pos = new Float32Array(sum.length);
  for (let i = 0; i < cnt.length; i++) for (let a = 0; a < 3; a++) pos[3 * i + a] = sum[3 * i + a] / cnt[i];
  const seen = new Set<string>(), I: number[] = [];
  for (let t = 0; t + 2 < m.indices.length; t += 3) {
    const a = rep[m.indices[t]], b = rep[m.indices[t + 1]], c = rep[m.indices[t + 2]];
    if (a === b || b === c || a === c) continue;
    const k = [a, b, c].sort((x, y) => x - y).join(",") + (isEven(a, b, c) ? "+" : "-");
    if (seen.has(k)) continue;
    seen.add(k); I.push(a, b, c);
  }
  return compactMesh({ positions: pos, indices: I });
}
/** Parity of the cyclic order (a, b, c) relative to sorted order. */
function isEven(a: number, b: number, c: number): boolean { return (a < b ? 1 : 0) + (b < c ? 1 : 0) + (c < a ? 1 : 0) === 2; }

/** Neighbour lists of every vertex. */
export function vertexNeighbours(m: Mesh): number[][] {
  const nv = m.positions.length / 3, nb: Set<number>[] = Array.from({ length: nv }, () => new Set<number>());
  for (let t = 0; t + 2 < m.indices.length; t += 3) {
    const a = m.indices[t], b = m.indices[t + 1], c = m.indices[t + 2];
    nb[a].add(b); nb[a].add(c); nb[b].add(a); nb[b].add(c); nb[c].add(a); nb[c].add(b);
  }
  return nb.map(s => [...s]);
}

/** Laplacian smoothing: `iterations` times, move each vertex a fraction `lambda` towards the
 *  mean of its neighbours. Shrinks the surface (Taubin's two-step variant does not). */
export function laplacianSmooth(m: Mesh, iterations: number, lambda = 0.5): Float32Array {
  let P = Float32Array.from(m.positions);
  const nb = vertexNeighbours(m);
  for (let it = 0; it < iterations; it++) {
    const Q = new Float32Array(P);
    for (let i = 0; i < nb.length; i++) {
      const n = nb[i];
      if (!n.length) continue;
      for (let a = 0; a < 3; a++) {
        let s = 0;
        for (const j of n) s += P[3 * j + a];
        Q[3 * i + a] = P[3 * i + a] + lambda * (s / n.length - P[3 * i + a]);
      }
    }
    P = Q;
  }
  return P;
}
