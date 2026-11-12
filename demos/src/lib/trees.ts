// Seeded synthetic branching trees with a known ground-truth graph (owner: skeleton day;
// no DOM): neurons (a soma with branching dendrites) and vessel networks (a tree with
// radius decreasing by Murray's law, tortuous branches and a few loops). Each tree is a
// list of centreline points with parent links (like the standard tree files of neuron
// tracing: id, position, radius, parent), from which `treeGraph` builds the ground-truth
// graph and `renderTree` a synthetic image. Points carry a z coordinate (0 for the 2D
// trees used now), so a 3D demo can reuse the format.
import { Grey, createGrey, gaussianBlur } from "./imageops";
import { rng, Rng, gauss, uniform } from "./random";
import { label2d } from "./label";

export interface TreePoint {
  x: number; y: number; z: number;
  /** Radius of the tube at this point (pixels). */
  r: number;
  /** Index of the parent point (−1 for the root). */
  parent: number;
  /** Centrifugal branch order: 1 for the branches leaving the root, +1 at each branch point. */
  order: number;
}

export interface Tree {
  kind: "neuron" | "vessel";
  size: number;
  points: TreePoint[];
  /** Extra links (anastomoses) between two points: loops of a vessel network. */
  links: [number, number][];
  /** The soma of a neuron (centre and radius). */
  soma?: { x: number; y: number; r: number };
}

export interface TreeOptions {
  kind?: "neuron" | "vessel";
  seed?: number;
  /** Image size (square). */
  size?: number;
  /** Branching probability per pixel of length (after a minimum segment length). */
  branching?: number;
  /** Tortuosity: size of the random turning (0 = straight branches). */
  tortuosity?: number;
  /** Number of primary branches of a neuron. */
  primaries?: number;
  /** Radius of the root (vessel) or of the primary dendrites (neuron). */
  r0?: number;
  /** Number of loops (vessels). */
  loops?: number;
  /** Maximum total path length of a branch from the root. */
  reach?: number;
}

export const NEURON_DEFAULTS: Required<TreeOptions> = { kind: "neuron", seed: 7, size: 256, branching: 0.022, tortuosity: 0.5, primaries: 5, r0: 1.8, loops: 0, reach: 115 };
export const VESSEL_DEFAULTS: Required<TreeOptions> = { kind: "vessel", seed: 15, size: 256, branching: 0.02, tortuosity: 0.5, primaries: 1, r0: 5, loops: 2, reach: 330 };

interface Tip { p: number; angle: number; omega: number; r: number; order: number; dist: number; sinceBranch: number; turns?: number }

/** Grow a tree. Branches stop at the border, near other branches (or join them: a loop), or at their reach. */
export function growTree(opts: TreeOptions = {}): Tree {
  const kind = opts.kind ?? "neuron";
  const o: Required<TreeOptions> = { ...(kind === "neuron" ? NEURON_DEFAULTS : VESSEL_DEFAULTS), ...opts } as Required<TreeOptions>;
  const R: Rng = rng(o.seed), N = o.size, step = 2, margin = kind === "vessel" ? 12 : 8;
  const points: TreePoint[] = [], links: [number, number][] = [];
  // spatial hash of points (cell 8 px)
  const CELL = 8, nc = Math.ceil(N / CELL), grid: number[][] = Array.from({ length: nc * nc }, () => []);
  const add = (pt: TreePoint) => {
    points.push(pt);
    const k = Math.min(nc - 1, Math.max(0, Math.floor(pt.y / CELL))) * nc + Math.min(nc - 1, Math.max(0, Math.floor(pt.x / CELL)));
    grid[k].push(points.length - 1);
    return points.length - 1;
  };
  /** Steps from p up to each of its ancestors (at most n steps). */
  const ancestors = (p: number, n: number) => { const s = new Map<number, number>(); for (let q = p, k = 0; q >= 0 && k < n; q = points[q].parent, k++) s.set(q, k); return s; };
  /**
   * Nearest point closer than its clearance (−1 if none), and whether it is close in the tree
   * (a common ancestor within 60 steps). Points just after a shared branch point, where the
   * two branches are still separating, are ignored.
   */
  const collide = (x: number, y: number, r: number, from: number): { q: number; near: boolean } => {
    const anc = ancestors(from, 60), gap = kind === "neuron" ? 3 : 5, win = kind === "neuron" ? 18 : 6;
    const cx = Math.floor(x / CELL), cy = Math.floor(y / CELL);
    let best = -1, bd = Infinity, bestNear = false;
    for (let gy = cy - 3; gy <= cy + 3; gy++) for (let gx = cx - 3; gx <= cx + 3; gx++) {
      if (gx < 0 || gy < 0 || gx >= nc || gy >= nc) continue;
      for (const q of grid[gy * nc + gx]) {
        const pq = points[q], d = Math.hypot(pq.x - x, pq.y - y);
        if (d >= r + pq.r + gap || d >= bd) continue;
        // steps from the tip and from q to their common ancestor
        let kt = -1, kq = 0;
        for (let a = q; a >= 0 && kq < 60; a = points[a].parent, kq++) if (anc.has(a)) { kt = anc.get(a)!; break; }
        if (kt >= 0 && step * Math.min(kt + 1, kq) < 2 * (r + pq.r) + win) continue;
        bd = d; best = q; bestNear = kt >= 0;
      }
    }
    return { q: best, near: bestNear };
  };
  const tips: Tip[] = [];
  let loopsLeft = kind === "vessel" ? o.loops : 0;
  let soma: Tree["soma"];
  if (kind === "neuron") {
    const cx = N / 2 + uniform(R, -6, 6), cy = N / 2 + uniform(R, -6, 6), rs = 7;
    soma = { x: cx, y: cy, r: rs };
    const root = add({ x: cx, y: cy, z: 0, r: rs, parent: -1, order: 0 });
    const a0 = 2 * Math.PI * R.unif();
    for (let k = 0; k < o.primaries; k++) {
      const ang = a0 + 2 * Math.PI * k / o.primaries + uniform(R, -0.3, 0.3);
      const p = add({ x: cx + rs * Math.cos(ang), y: cy + rs * Math.sin(ang), z: 0, r: o.r0, parent: root, order: 1 });
      tips.push({ p, angle: ang, omega: 0, r: o.r0, order: 1, dist: rs, sinceBranch: 0 });
    }
  } else {
    const y0 = N * uniform(R, 0.35, 0.65);
    const root = add({ x: margin + 2, y: y0, z: 0, r: o.r0, parent: -1, order: 0 });
    tips.push({ p: root, angle: uniform(R, -0.2, 0.2), omega: 0, r: o.r0, order: 1, dist: 0, sinceBranch: 0 });
  }
  const minSeg = kind === "neuron" ? 16 : 22, rMin = kind === "neuron" ? 0.9 : 1.2;
  let guard = 0;
  while (tips.length && guard++ < 20000) {
    // grow tips in turn (breadth-first), so branches compete fairly for space
    const t = tips.shift()!;
    const prev = points[t.p];
    t.omega = 0.8 * t.omega + 0.06 * o.tortuosity * gauss(R, 0, 1);
    t.angle += t.omega;
    const nx = prev.x + step * Math.cos(t.angle), ny = prev.y + step * Math.sin(t.angle);
    if (nx < margin || ny < margin || nx > N - margin || ny > N - margin) {
      // the main vessel may leave the image on the right; other tips turn back towards the centre
      if (kind === "vessel" && (t.order > 1 || nx > N - margin)) continue;
      const toC = Math.atan2(N / 2 - prev.y, N / 2 - prev.x);
      let d = toC - t.angle;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      t.angle += Math.sign(d) * 0.35; t.omega = 0;
      t.turns = (t.turns ?? 0) + 1;
      if (t.turns > 12) continue;
      tips.push(t);
      continue;
    }
    t.turns = 0;
    const hit = collide(nx, ny, t.r, t.p);
    if (hit.q >= 0) {
      if (loopsLeft > 0 && !hit.near && t.dist > 40) { links.push([t.p, hit.q]); loopsLeft--; }
      continue;
    }
    const p = add({ x: nx, y: ny, z: 0, r: t.r, parent: t.p, order: t.order });
    const nt: Tip = { ...t, p, dist: t.dist + step, sinceBranch: t.sinceBranch + step };
    if (nt.dist > o.reach) continue;
    if (nt.sinceBranch > minSeg && R.unif() < o.branching * step) {
      // Murray's law: r³ = r1³ + r2³, split unevenly
      const f = uniform(R, 0.35, 0.65);
      const r1 = Math.max(rMin, t.r * Math.cbrt(f)), r2 = Math.max(rMin, t.r * Math.cbrt(1 - f));
      const spread = kind === "vessel" ? uniform(R, 0.7, 1.1) : uniform(R, 0.45, 0.8);
      const side = R.unif() < 0.5 ? 1 : -1;
      tips.push({ p, angle: t.angle + side * spread * (1 - f), omega: 0, r: r1, order: t.order + 1, dist: nt.dist, sinceBranch: 0 });
      tips.push({ p, angle: t.angle - side * spread * f, omega: 0, r: r2, order: t.order + 1, dist: nt.dist, sinceBranch: 0 });
    } else tips.push(nt);
  }
  return removeStubs({ kind, size: N, points, links, soma }, 14);
}

/** Remove terminal branches shorter than `minLen` (stubs that stopped right after a branch point). */
export function removeStubs(t: Tree, minLen: number): Tree {
  const n = t.points.length, children = new Int32Array(n), dead = new Uint8Array(n);
  t.points.forEach(p => { if (p.parent >= 0) children[p.parent]++; });
  const linked = new Set(t.links.flat());
  for (let i = 1; i < n; i++) {
    if (children[i] || linked.has(i)) continue;
    const chain: number[] = [];
    let len = 0, q = i;
    while (q > 0 && children[q] <= 1 && !linked.has(q)) {
      chain.push(q);
      const pa = t.points[q].parent;
      len += Math.hypot(t.points[q].x - t.points[pa].x, t.points[q].y - t.points[pa].y);
      q = pa;
    }
    // q is the branch point (or the root); keep a primary dendrite of the soma unless it is tiny
    if (len < minLen && q >= 0 && (children[q] >= 2 || (q === 0 && len < 3))) for (const c of chain) dead[c] = 1;
  }
  const map = new Int32Array(n).fill(-1), pts: TreePoint[] = [];
  t.points.forEach((p, i) => { if (!dead[i]) { map[i] = pts.length; pts.push({ ...p }); } });
  for (const p of pts) if (p.parent >= 0) p.parent = map[p.parent];
  return { ...t, points: pts, links: t.links.map(([a, b]) => [map[a], map[b]] as [number, number]) };
}

// ---------------------------------------------------------------------------------------
// Ground-truth graph

export interface TreeNode { x: number; y: number; kind: "root" | "soma" | "branch" | "end"; degree: number; point: number }
export interface TreeEdge {
  a: number; b: number;
  /** Centreline from node a to node b. */
  line: [number, number][];
  length: number;
  chord: number;
  /** Mean radius along the edge. */
  radius: number;
  /** Strahler-style centrifugal order of the edge (order of its points). */
  order: number;
}
export interface TreeGraph { nodes: TreeNode[]; edges: TreeEdge[]; loops: number }

/** Collapse chains of degree-2 points into edges; nodes are the root/soma, branch points and ends. */
export function treeGraph(t: Tree): TreeGraph {
  const n = t.points.length, adj: number[][] = Array.from({ length: n }, () => []);
  t.points.forEach((p, i) => { if (p.parent >= 0) { adj[i].push(p.parent); adj[p.parent].push(i); } });
  for (const [a, b] of t.links) { adj[a].push(b); adj[b].push(a); }
  const isNode = (i: number) => i === 0 || adj[i].length !== 2;
  const nodeId = new Int32Array(n).fill(-1), nodes: TreeNode[] = [];
  for (let i = 0; i < n; i++) if (isNode(i)) {
    nodeId[i] = nodes.length;
    const kind = i === 0 ? (t.kind === "neuron" ? "soma" : "root") : adj[i].length === 1 ? "end" : "branch";
    nodes.push({ x: t.points[i].x, y: t.points[i].y, kind, degree: adj[i].length, point: i });
  }
  const edges: TreeEdge[] = [], used = new Set<string>();
  const key = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);
  for (let s = 0; s < n; s++) {
    if (!isNode(s)) continue;
    for (const first of adj[s]) {
      if (used.has(key(s, first))) continue;
      const chain = [s];
      let prev = s, cur = first;
      used.add(key(s, first));
      while (!isNode(cur)) {
        chain.push(cur);
        const nx = adj[cur][0] === prev ? adj[cur][1] : adj[cur][0];
        used.add(key(cur, nx));
        prev = cur; cur = nx;
      }
      chain.push(cur);
      const line = chain.map(i => [t.points[i].x, t.points[i].y] as [number, number]);
      // the soma's edges start on its rim, not at its centre
      if (t.soma && chain[0] === 0) line.shift();
      if (t.soma && chain[chain.length - 1] === 0) line.pop();
      let length = 0, rs = 0;
      for (let k = 1; k < line.length; k++) length += Math.hypot(line[k][0] - line[k - 1][0], line[k][1] - line[k - 1][1]);
      const inner = chain.filter(i => !(t.soma && i === 0));
      for (const i of inner) rs += t.points[i].r;
      const chordLen = Math.hypot(line[line.length - 1][0] - line[0][0], line[line.length - 1][1] - line[0][1]);
      edges.push({ a: nodeId[s], b: nodeId[cur], line, length, chord: chordLen, radius: rs / Math.max(1, inner.length), order: Math.max(...inner.map(i => t.points[i].order)) });
    }
  }
  return { nodes, edges, loops: t.links.length };
}

// ---------------------------------------------------------------------------------------
// Rendering

export interface RenderOptions {
  seed?: number;
  /** Brightness of the tubes above the background. */
  contrast?: number;
  background?: number;
  /** Gaussian blur σ (the optics). */
  blur?: number;
  /** Gaussian noise standard deviation. */
  noise?: number;
  /** Darkening towards the corners (0 = none, 0.5 = corners at half brightness). */
  vignette?: number;
  /** Number of soft background blobs (uneven background). */
  blobs?: number;
  /** Bright spots on the tubes (varicosities, cells on a vessel). */
  spots?: number;
}

/** Fraction of pixel (x, y) covered by the tubes (anti-aliased), and the true mask (coverage ≥ 0.5). */
export function treeCoverage(t: Tree): { cover: Float32Array; mask: Uint8Array } {
  const N = t.size, cover = new Float32Array(N * N);
  const seg = (ax: number, ay: number, ar: number, bx: number, by: number, br: number) => {
    const R = Math.max(ar, br) + 1.5;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx) - R)), x1 = Math.min(N - 1, Math.ceil(Math.max(ax, bx) + R));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by) - R)), y1 = Math.min(N - 1, Math.ceil(Math.max(ay, by) + R));
    const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy || 1;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const px = x + 0.5, py = y + 0.5;
      const u = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L2));
      const d = Math.hypot(px - ax - u * dx, py - ay - u * dy), r = ar + u * (br - ar);
      const c = Math.max(0, Math.min(1, r + 0.5 - d));
      if (c > cover[y * N + x]) cover[y * N + x] = c;
    }
  };
  t.points.forEach(p => { if (p.parent >= 0) { const q = t.points[p.parent]; const rq = t.soma && p.parent === 0 ? p.r : q.r; seg(q.x, q.y, rq, p.x, p.y, p.r); } });
  for (const [a, b] of t.links) seg(t.points[a].x, t.points[a].y, t.points[a].r, t.points[b].x, t.points[b].y, t.points[b].r);
  if (t.soma) seg(t.soma.x, t.soma.y, t.soma.r, t.soma.x, t.soma.y, t.soma.r);
  const mask = Uint8Array.from(cover, v => (v >= 0.5 ? 1 : 0));
  // fill pinholes of a few pixels (rasterisation gaps in the fork of a branch point)
  const inv = Uint8Array.from(mask, v => 1 - v), L = label2d(inv, N, N, 4), size = new Int32Array(L.count + 1);
  for (let i = 0; i < N * N; i++) size[L.data[i]]++;
  for (let i = 0; i < N * N; i++) if (L.data[i] > 0 && size[L.data[i]] <= 4) { mask[i] = 1; cover[i] = Math.max(cover[i], 0.5); }
  return { cover, mask };
}

/** A synthetic microscopy-like image of the tree (bright tubes on a dark background). */
export function renderTree(t: Tree, o: RenderOptions = {}): { image: Grey; mask: Uint8Array } {
  const N = t.size, r = rng(o.seed ?? 11);
  const { cover, mask } = treeCoverage(t);
  const contrast = o.contrast ?? 120, bg = o.background ?? 30;
  let img = createGrey(N, N);
  for (let i = 0; i < N * N; i++) img.data[i] = contrast * cover[i];
  for (let k = 0; k < (o.spots ?? 0); k++) {
    const p = t.points[1 + Math.floor(r.unif() * (t.points.length - 1))], rs = p.r * 1.8 + 1;
    for (let y = Math.floor(p.y - rs - 1); y <= p.y + rs + 1; y++) for (let x = Math.floor(p.x - rs - 1); x <= p.x + rs + 1; x++) {
      if (x < 0 || y < 0 || x >= N || y >= N) continue;
      const c = Math.max(0, Math.min(1, rs + 0.5 - Math.hypot(x + 0.5 - p.x, y + 0.5 - p.y)));
      img.data[y * N + x] = Math.max(img.data[y * N + x], contrast * 1.4 * c);
    }
  }
  if ((o.blur ?? 1) > 0) img = gaussianBlur(img, o.blur ?? 1);
  const blobs = Array.from({ length: o.blobs ?? 3 }, () => ({ x: N * r.unif(), y: N * r.unif(), s: N * uniform(r, 0.08, 0.2), a: uniform(r, 10, 35) }));
  const vig = o.vignette ?? 0.2;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = bg + img.data[y * N + x];
    for (const b of blobs) v += b.a * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / (2 * b.s * b.s));
    const rr = ((x - N / 2) ** 2 + (y - N / 2) ** 2) / (N * N / 2);
    v *= 1 - vig * rr;
    img.data[y * N + x] = v + (o.noise ?? 6) * r.norm();
  }
  return { image: img, mask };
}
