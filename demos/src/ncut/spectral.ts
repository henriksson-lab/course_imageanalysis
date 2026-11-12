// Spectral graph partitioning. No DOM: also used by the test (demos/test/).
//  - WGraph: an undirected weighted graph as an edge list + adjacency lists.
//  - fiedler(): the second-smallest eigenvector of L x = mu x (unnormalized) or of
//    L x = mu D x (normalized cuts, Shi & Malik 2000), by shift-invert Lanczos with a
//    banded Cholesky factor, deflating the trivial eigenvector.
//  - sweep(): cut, volumes and Ncut for every split of the nodes sorted by a vector.
//  - stoerWagner(): the global minimum cut (no seeds, no balance), O(n^3) for small graphs.
//  - jacobiEigen(): dense symmetric eigen-decomposition (small matrices; also the test reference).

export interface WGraph { n: number; p: number[]; q: number[]; w: number[] }

export function degrees(G: WGraph): Float64Array {
  const d = new Float64Array(G.n);
  for (let k = 0; k < G.p.length; k++) { d[G.p[k]] += G.w[k]; d[G.q[k]] += G.w[k]; }
  return d;
}

/** y = L x with L = D - W. */
function lapMul(G: WGraph, d: Float64Array, x: Float64Array, y: Float64Array): void {
  for (let i = 0; i < G.n; i++) y[i] = d[i] * x[i];
  for (let k = 0; k < G.p.length; k++) {
    const a = G.p[k], b = G.q[k], w = G.w[k];
    y[a] -= w * x[b]; y[b] -= w * x[a];
  }
}

const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function normalize(a: Float64Array): number {
  const s = Math.sqrt(dot(a, a));
  if (s > 0) for (let i = 0; i < a.length; i++) a[i] /= s;
  return s;
}

/** Eigen-decomposition of a dense symmetric matrix (row-major n x n) by cyclic Jacobi.
 *  Returns eigenvalues ascending and eigenvectors as columns of V (row-major). */
export function jacobiEigen(Ain: ArrayLike<number>, n: number): { values: Float64Array; vectors: Float64Array } {
  const A = Float64Array.from(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, tot = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = A[i * n + j] ** 2; tot += v; if (i !== j) off += v; }
    if (off <= 1e-26 * Math.max(tot, 1e-300)) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {   // columns p, q
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {   // rows p, q
        const apk = A[p * n + k], aqk = A[q * n + k];
        A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [...Array(n).keys()].sort((a, b) => A[a * n + a] - A[b * n + b]);
  const values = Float64Array.from(order, i => A[i * n + i]);
  const vectors = new Float64Array(n * n);
  order.forEach((src, dst) => { for (let k = 0; k < n; k++) vectors[k * n + dst] = V[k * n + src]; });
  return { values, vectors };
}

/** Smallest eigenpair of the symmetric tridiagonal matrix with diagonal a and off-diagonal b
 *  (Sturm-sequence bisection, then inverse iteration; T - shift is positive definite, so the
 *  plain Thomas algorithm is stable). */
export function tridiagSmallest(a: number[], b: number[]): { value: number; vector: Float64Array } {
  const k = a.length;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < k; i++) {
    const r = (i > 0 ? Math.abs(b[i - 1]) : 0) + (i < k - 1 ? Math.abs(b[i]) : 0);
    lo = Math.min(lo, a[i] - r); hi = Math.max(hi, a[i] + r);
  }
  const below = (x: number) => {   // number of eigenvalues < x
    let c = 0, q = 1;
    for (let i = 0; i < k; i++) {
      q = a[i] - x - (i > 0 ? b[i - 1] * b[i - 1] / q : 0);
      if (q === 0) q = -1e-300;
      if (q < 0) c++;
    }
    return c;
  };
  for (let it = 0; it < 200 && hi - lo > 1e-15 * Math.max(1, Math.abs(lo) + Math.abs(hi)); it++) {
    const mid = (lo + hi) / 2;
    if (below(mid) >= 1) hi = mid; else lo = mid;
  }
  const value = lo;
  const shift = value - 1e-10 * Math.max(1, Math.abs(hi) + Math.abs(lo));
  let v = new Float64Array(k).fill(1);
  const c = new Float64Array(k), dd = new Float64Array(k);
  for (let rep = 0; rep < 3; rep++) {
    // Thomas: (T - shift) z = v
    dd[0] = a[0] - shift; const z = Float64Array.from(v);
    for (let i = 1; i < k; i++) {
      c[i] = b[i - 1] / dd[i - 1];
      dd[i] = a[i] - shift - c[i] * b[i - 1];
      z[i] -= c[i] * z[i - 1];
    }
    z[k - 1] /= dd[k - 1];
    for (let i = k - 2; i >= 0; i--) z[i] = (z[i] - b[i] * z[i + 1]) / dd[i];
    normalize(z);
    v = z;
  }
  return { value, vector: v };
}

/** Banded Cholesky factor of a symmetric positive definite matrix. Row i stores columns
 *  i - bw .. i at band[i * (bw + 1) + (j - i + bw)]. */
function bandCholesky(band: Float64Array, n: number, bw: number): void {
  const W = bw + 1;
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(0, i - bw); j <= i; j++) {
      let s = band[i * W + (j - i + bw)];
      for (let k = Math.max(0, i - bw); k < j; k++) s -= band[i * W + (k - i + bw)] * band[j * W + (k - j + bw)];
      if (i === j) band[i * W + bw] = Math.sqrt(Math.max(s, 1e-300));
      else band[i * W + (j - i + bw)] = s / band[j * W + bw];
    }
  }
}

/** Solve (C C^T) x = b in place, C from bandCholesky. */
function bandSolve(C: Float64Array, n: number, bw: number, x: Float64Array): void {
  const W = bw + 1;
  for (let i = 0; i < n; i++) {
    let s = x[i];
    for (let k = Math.max(0, i - bw); k < i; k++) s -= C[i * W + (k - i + bw)] * x[k];
    x[i] = s / C[i * W + bw];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k <= Math.min(n - 1, i + bw); k++) s -= C[k * W + (i - k + bw)] * x[k];
    x[i] = s / C[i * W + bw];
  }
}

export interface Fiedler { x: Float64Array; mu: number; residual: number }

/** Second-smallest eigenpair of L x = mu x, or of L x = mu D x when `normalized`.
 *  Method: the symmetric matrix A = L or D^-1/2 L D^-1/2 has the trivial eigenvector y0
 *  (eigenvalue 0). Lanczos on (A + eps I)^-1, restricted to the complement of y0, finds
 *  the largest eigenvalue 1 / (mu + eps), which converges in a few dozen steps even when
 *  the eigenvalues are tiny and close. (A + eps I) is factored once, banded (node order
 *  matters: bandwidth = max |p - q| over the edges).
 *  The returned x is scaled to max |x| = 1, with the sign chosen so that x[0] <= 0. */
export function fiedler(G: WGraph, normalized: boolean): Fiedler {
  const n = G.n;
  if (n < 2) return { x: new Float64Array(n), mu: 0, residual: 0 };
  const d = degrees(G);
  const deg = Float64Array.from(d, v => v + 1e-12);   // isolated nodes: avoid dividing by 0
  const isq = Float64Array.from(deg, v => 1 / Math.sqrt(v));
  const sc = (i: number) => (normalized ? isq[i] : 1);
  let bw = 0, dmax = 0;
  for (let k = 0; k < G.p.length; k++) bw = Math.max(bw, Math.abs(G.p[k] - G.q[k]));
  for (const v of d) dmax = Math.max(dmax, v);
  const eps = 1e-9 * (normalized ? 1 : Math.max(dmax, 1e-12));
  const W = bw + 1;
  const C = new Float64Array(n * W);
  for (let i = 0; i < n; i++) C[i * W + bw] = d[i] * sc(i) * sc(i) + eps;
  for (let k = 0; k < G.p.length; k++) {
    const i = Math.max(G.p[k], G.q[k]), j = Math.min(G.p[k], G.q[k]);
    C[i * W + (j - i + bw)] -= G.w[k] * sc(i) * sc(j);
  }
  bandCholesky(C, n, bw);

  const tmp = new Float64Array(n), tmp2 = new Float64Array(n);
  const op = (x: Float64Array, y: Float64Array) => {   // y = A x
    for (let i = 0; i < n; i++) tmp[i] = x[i] * sc(i);
    lapMul(G, d, tmp, tmp2);
    for (let i = 0; i < n; i++) y[i] = tmp2[i] * sc(i);
  };
  const y0 = Float64Array.from(deg, v => (normalized ? Math.sqrt(v) : 1));
  normalize(y0);
  const deflate = (v: Float64Array) => { const c = dot(v, y0); for (let i = 0; i < n; i++) v[i] -= c * y0[i]; };

  let start = Float64Array.from({ length: n }, (_, i) => Math.sin(1.3 * i + 0.7) + 0.31 * Math.cos(0.17 * i * i));
  const m = Math.min(n - 1, 40);
  let best: Fiedler = { x: new Float64Array(n), mu: 0, residual: Infinity };
  const w = new Float64Array(n);
  for (let it = 0; it < 6; it++) {
    const Q: Float64Array[] = [];
    const alpha: number[] = [], beta: number[] = [];
    let q = Float64Array.from(start);
    deflate(q); normalize(q);
    let ritz = tridiagSmallest([0], []);
    for (let j = 0; j < m; j++) {
      Q.push(q);
      w.set(q); bandSolve(C, n, bw, w);
      alpha.push(-dot(w, q));          // Lanczos on -(A + eps I)^-1: we want its smallest
      for (let i = 0; i < n; i++) w[i] = -w[i];
      for (let pass = 0; pass < 2; pass++) {
        deflate(w);
        for (const v of Q) { const c = dot(w, v); for (let i = 0; i < n; i++) w[i] -= c * v[i]; }
      }
      const b = Math.sqrt(dot(w, w));
      const k = alpha.length;
      if (k === m || b <= 1e-14 * Math.abs(alpha[0]) || k % 4 === 0) {
        ritz = tridiagSmallest(alpha, beta);
        if (k === m || b <= 1e-14 * Math.abs(alpha[0]) || b * Math.abs(ritz.vector[k - 1]) < 1e-12 * Math.abs(ritz.value)) break;
      }
      beta.push(b);
      q = Float64Array.from(w, v => v / b);
    }
    const y = new Float64Array(n);
    for (let j = 0; j < alpha.length; j++) { const c = ritz.vector[j]; for (let i = 0; i < n; i++) y[i] += c * Q[j][i]; }
    deflate(y); normalize(y);
    op(y, w);
    const mu = dot(w, y);
    let res = 0;
    for (let i = 0; i < n; i++) res += (w[i] - mu * y[i]) ** 2;
    res = Math.sqrt(res);
    best = { x: y, mu, residual: res };
    if (res < 1e-9 * Math.max(1, normalized ? 1 : dmax)) break;
    start = y;
  }
  // back to x: x = D^-1/2 y for the normalized problem
  const x = Float64Array.from(best.x, (v, i) => v * sc(i));
  let mx = 0;
  for (const v of x) mx = Math.max(mx, Math.abs(v));
  const sgn = x[0] > 0 ? -1 : 1;
  for (let i = 0; i < n; i++) x[i] = mx > 0 ? sgn * x[i] / mx : 0;
  return { x, mu: best.mu, residual: best.residual };
}

export interface Sweep {
  order: number[];        // nodes sorted by the vector, ascending
  cut: Float64Array;      // cut[k]: A = first k+1 nodes of `order` (k = 0..n-2)
  volA: Float64Array; volB: Float64Array;
  ncut: Float64Array;     // cut/volA + cut/volB
}

/** Adjacency lists of a WGraph. */
export function adjacency(G: WGraph): { nb: number[][]; wt: number[][] } {
  const nb: number[][] = Array.from({ length: G.n }, () => []), wt: number[][] = Array.from({ length: G.n }, () => []);
  for (let k = 0; k < G.p.length; k++) {
    nb[G.p[k]].push(G.q[k]); wt[G.p[k]].push(G.w[k]);
    nb[G.q[k]].push(G.p[k]); wt[G.q[k]].push(G.w[k]);
  }
  return { nb, wt };
}

/** Every split of the nodes sorted by x into A (smaller values) and B, with its cut and Ncut. */
export function sweep(G: WGraph, x: ArrayLike<number>): Sweep {
  const n = G.n;
  const order = [...Array(n).keys()].sort((a, b) => x[a] - x[b] || a - b);
  const d = degrees(G);
  const { nb, wt } = adjacency(G);
  const inA = new Uint8Array(n);
  const total = d.reduce((a, b) => a + b, 0);
  const cut = new Float64Array(Math.max(0, n - 1)), volA = new Float64Array(Math.max(0, n - 1));
  const volB = new Float64Array(Math.max(0, n - 1)), ncut = new Float64Array(Math.max(0, n - 1));
  let c = 0, va = 0;
  for (let k = 0; k < n - 1; k++) {
    const u = order[k];
    // moving u from B to A: its edges to A stop being cut, its edges to B start
    for (let j = 0; j < nb[u].length; j++) c += inA[nb[u][j]] ? -wt[u][j] : wt[u][j];
    inA[u] = 1; va += d[u];
    cut[k] = Math.max(0, c); volA[k] = va; volB[k] = total - va;
    ncut[k] = cut[k] / Math.max(va, 1e-300) + cut[k] / Math.max(total - va, 1e-300);
  }
  return { order, cut, volA, volB, ncut };
}

/** Global minimum cut of an undirected graph (Stoer-Wagner). Returns its value and one side. */
export function stoerWagner(G: WGraph): { value: number; side: Uint8Array } {
  const n = G.n;
  const W = new Float64Array(n * n);
  for (let k = 0; k < G.p.length; k++) { W[G.p[k] * n + G.q[k]] += G.w[k]; W[G.q[k] * n + G.p[k]] += G.w[k]; }
  const members: number[][] = Array.from({ length: n }, (_, i) => [i]);
  const alive = [...Array(n).keys()];
  let best = Infinity, bestSide: number[] = [];
  while (alive.length > 1) {
    const key = new Float64Array(n);
    const used = new Uint8Array(n);
    let prev = -1, last = -1;
    for (let step = 0; step < alive.length; step++) {
      let sel = -1;
      for (const v of alive) if (!used[v] && (sel < 0 || key[v] > key[sel])) sel = v;
      used[sel] = 1;
      if (step === alive.length - 1) {
        if (key[sel] < best) { best = key[sel]; bestSide = members[sel].slice(); }
      }
      prev = last; last = sel;
      for (const v of alive) if (!used[v]) key[v] += W[sel * n + v];
    }
    // merge last into prev
    for (const v of alive) { W[prev * n + v] += W[last * n + v]; W[v * n + prev] = W[prev * n + v]; }
    W[prev * n + prev] = 0;
    members[prev].push(...members[last]);
    alive.splice(alive.indexOf(last), 1);
  }
  const side = new Uint8Array(n);
  for (const v of bestSide) side[v] = 1;
  return { value: n > 1 ? best : 0, side };
}
