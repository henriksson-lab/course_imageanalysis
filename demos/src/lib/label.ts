// Connected components (no DOM). Masks are flat arrays (non-zero = foreground) with their
// sizes passed explicitly; label images are Int32Arrays with 0 = background and objects
// numbered 1..count in raster order of their first pixel (top-left first).
//
//  - UnionFind: disjoint sets with path halving and union by size.
//  - label2d (4- or 8-connectivity), label3d (6, 18 or 26), by two raster passes + union–find.
//  - neighbour offset tables, component sizes, bounding boxes, removing small objects.

export type Conn2 = 4 | 8;
export type Conn3 = 6 | 18 | 26;

/** A label image: 0 = background, objects 1..count. */
export interface Labels { width: number; height: number; data: Int32Array; count: number }

/** Disjoint-set forest over 0..n-1. */
export class UnionFind {
  readonly parent: Int32Array;
  readonly sizeOf: Int32Array;
  constructor(n: number) {
    this.parent = new Int32Array(n);
    this.sizeOf = new Int32Array(n).fill(1);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }
  /** Representative of x's set (path halving). */
  find(x: number): number {
    const p = this.parent;
    while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
  }
  /** Merge the sets of a and b; returns the new representative. */
  union(a: number, b: number): number {
    let ra = this.find(a), rb = this.find(b);
    if (ra === rb) return ra;
    if (this.sizeOf[ra] < this.sizeOf[rb]) { const t = ra; ra = rb; rb = t; }
    this.parent[rb] = ra;
    this.sizeOf[ra] += this.sizeOf[rb];
    return ra;
  }
  same(a: number, b: number): boolean { return this.find(a) === this.find(b); }
  /** Number of elements in x's set. */
  size(x: number): number { return this.sizeOf[this.find(x)]; }
}

/** Neighbour offsets [dx, dy] for 4- or 8-connectivity (4 first, then the diagonals). */
export function offsets2d(conn: Conn2): [number, number][] {
  const o: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  if (conn === 8) o.push([1, 1], [-1, 1], [1, -1], [-1, -1]);
  return o;
}

/** Neighbour offsets [dx, dy, dz]: 6 (faces), 18 (+ edges), 26 (+ corners). */
export function offsets3d(conn: Conn3): [number, number, number][] {
  const o: [number, number, number][] = [];
  for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const k = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
    if (k === 0) continue;
    if (k === 1 || (k === 2 && conn >= 18) || (k === 3 && conn === 26)) o.push([dx, dy, dz]);
  }
  return o;
}

/** Turn provisional union–find labels into 1..count in raster order of first appearance. */
function finish(prov: Int32Array, uf: UnionFind, n: number): { data: Int32Array; count: number } {
  const map = new Int32Array(uf.parent.length).fill(0);
  const out = new Int32Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const p = prov[i];
    if (p === 0) continue;
    const r = uf.find(p);
    if (map[r] === 0) map[r] = ++count;
    out[i] = map[r];
  }
  return { data: out, count };
}

/** Connected components of a 2D mask (non-zero = object). */
export function label2d(mask: ArrayLike<number>, w: number, h: number, conn: Conn2 = 8): Labels {
  const n = w * h, prov = new Int32Array(n);
  const uf = new UnionFind(Math.floor(n / 2) + 2);   // at most ⌈n/2⌉ provisional labels
  let next = 1;
  // previously visited neighbours: left, and the row above
  const back: [number, number][] = conn === 8 ? [[-1, 0], [-1, -1], [0, -1], [1, -1]] : [[-1, 0], [0, -1]];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!mask[i]) continue;
    let l = 0;
    for (const [dx, dy] of back) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || xx >= w || yy < 0) continue;
      const m = prov[yy * w + xx];
      if (m === 0) continue;
      if (l === 0) l = m; else uf.union(l, m);
    }
    prov[i] = l || next++;
  }
  const { data, count } = finish(prov, uf, n);
  return { width: w, height: h, data, count };
}

/** Connected components of a 3D mask nx × ny × nz (index z*nx*ny + y*nx + x). */
export function label3d(mask: ArrayLike<number>, nx: number, ny: number, nz: number, conn: Conn3 = 26):
  { data: Int32Array; count: number } {
  const n = nx * ny * nz, prov = new Int32Array(n), uf = new UnionFind(n + 1);
  const back = offsets3d(conn).filter(([dx, dy, dz]) => dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0))));
  let next = 1;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = (z * ny + y) * nx + x;
    if (!mask[i]) continue;
    let l = 0;
    for (const [dx, dy, dz] of back) {
      const xx = x + dx, yy = y + dy, zz = z + dz;
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0) continue;
      const m = prov[(zz * ny + yy) * nx + xx];
      if (m === 0) continue;
      if (l === 0) l = m; else uf.union(l, m);
    }
    prov[i] = l || next++;
  }
  return finish(prov, uf, n);
}

/** Pixel count of each label 0..count (index 0 = background). */
export function componentSizes(labels: ArrayLike<number>, count: number): Int32Array {
  const s = new Int32Array(count + 1);
  for (let i = 0; i < labels.length; i++) s[labels[i]]++;
  return s;
}

/** Inclusive bounding box per label 1..count (index 0 unused). */
export function boundingBoxes(lab: Labels): { x0: number; y0: number; x1: number; y1: number }[] {
  const b = Array.from({ length: lab.count + 1 }, () => ({ x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity }));
  for (let y = 0; y < lab.height; y++) for (let x = 0; x < lab.width; x++) {
    const l = lab.data[y * lab.width + x];
    if (!l) continue;
    const r = b[l];
    if (x < r.x0) r.x0 = x; if (x > r.x1) r.x1 = x;
    if (y < r.y0) r.y0 = y; if (y > r.y1) r.y1 = y;
  }
  return b;
}

/** Centroid (mean x, mean y) per label 1..count (index 0 unused). */
export function centroids(lab: Labels): { x: number; y: number; n: number }[] {
  const c = Array.from({ length: lab.count + 1 }, () => ({ x: 0, y: 0, n: 0 }));
  for (let y = 0; y < lab.height; y++) for (let x = 0; x < lab.width; x++) {
    const l = lab.data[y * lab.width + x];
    if (l) { c[l].x += x; c[l].y += y; c[l].n++; }
  }
  for (const r of c) if (r.n) { r.x /= r.n; r.y /= r.n; }
  return c;
}

/** Renumber labels to 1..count in raster order (0 stays background). */
export function relabel(labels: ArrayLike<number>, w: number, h: number): Labels {
  const map = new Map<number, number>(), data = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const l = labels[i];
    if (!l) continue;
    let m = map.get(l);
    if (m === undefined) { m = map.size + 1; map.set(l, m); }
    data[i] = m;
  }
  return { width: w, height: h, data, count: map.size };
}

/** Drop objects with fewer than minSize pixels and renumber. */
export function removeSmall(lab: Labels, minSize: number): Labels {
  const s = componentSizes(lab.data, lab.count);
  const kept = lab.data.map(l => (l && s[l] >= minSize ? l : 0));
  return relabel(kept, lab.width, lab.height);
}

/** 0/1 mask of the pixels with label l (or of all objects when l is omitted). */
export function maskOf(labels: ArrayLike<number>, l?: number): Uint8Array {
  const m = new Uint8Array(labels.length);
  for (let i = 0; i < labels.length; i++) m[i] = l === undefined ? (labels[i] ? 1 : 0) : (labels[i] === l ? 1 : 0);
  return m;
}
