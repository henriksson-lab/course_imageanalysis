// Small test meshes for the shading demo (indexed triangle lists, counter-clockwise seen from
// outside). Local to the demo: lib/mesh.ts belongs to the marching-cubes day.
export interface TriMesh { positions: number[]; indices: number[] }

/** Icosphere: an icosahedron subdivided `level` times, vertices pushed onto the unit sphere. */
export function icosphere(level: number): TriMesh {
  const t = (1 + Math.sqrt(5)) / 2;
  let P: number[][] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let F: number[][] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const norm = (p: number[]) => { const l = Math.hypot(p[0], p[1], p[2]); return [p[0] / l, p[1] / l, p[2] / l]; };
  P = P.map(norm);
  for (let s = 0; s < level; s++) {
    const cache = new Map<string, number>(), F2: number[][] = [];
    const mid = (a: number, b: number) => {
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      let i = cache.get(key);
      if (i === undefined) { i = P.length; P.push(norm([(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2, (P[a][2] + P[b][2]) / 2])); cache.set(key, i); }
      return i;
    };
    for (const [a, b, c] of F) { const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a); F2.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    F = F2;
  }
  return { positions: P.flat(), indices: F.flat() };
}

/** Torus around the y axis: major radius R, tube radius r. */
export function torus(R = 0.7, r = 0.3, nu = 32, nv = 16): TriMesh {
  const P: number[] = [], I: number[] = [];
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const u = 2 * Math.PI * i / nu, v = 2 * Math.PI * j / nv;
    P.push((R + r * Math.cos(v)) * Math.cos(u), r * Math.sin(v), -(R + r * Math.cos(v)) * Math.sin(u));
  }
  const id = (i: number, j: number) => ((i + nu) % nu) * nv + ((j + nv) % nv);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
    I.push(a, b, c, a, c, d);
  }
  return { positions: P, indices: I };
}

/** A low-poly "gem": an icosphere of level 1 with bumped vertices (strong creases). */
export function creaseBlob(): TriMesh {
  const m = icosphere(1);
  for (let v = 0; v < m.positions.length / 3; v++) {
    const s = 0.8 + 0.35 * Math.abs(Math.sin(3.1 * v + 0.7));
    for (let k = 0; k < 3; k++) m.positions[3 * v + k] *= s;
  }
  return m;
}

/** Area-weighted vertex normals (sum of the un-normalised face normals around each vertex). */
export function vertexNormals(m: TriMesh): number[] {
  const n = new Array<number>(m.positions.length).fill(0), P = m.positions;
  for (let t = 0; t < m.indices.length; t += 3) {
    const [a, b, c] = [m.indices[t], m.indices[t + 1], m.indices[t + 2]];
    const u = [0, 1, 2].map(k => P[3 * b + k] - P[3 * a + k]), w = [0, 1, 2].map(k => P[3 * c + k] - P[3 * a + k]);
    const f = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    for (const v of [a, b, c]) for (let k = 0; k < 3; k++) n[3 * v + k] += f[k];
  }
  for (let v = 0; v < n.length; v += 3) { const l = Math.hypot(n[v], n[v + 1], n[v + 2]) || 1; n[v] /= l; n[v + 1] /= l; n[v + 2] /= l; }
  return n;
}
