// Superpixels, no DOM. Owner: the graph-based segmentation day (superpixels/). Tests:
// demos/test/graphseg-test.ts.
//
//  - slic(): SLIC (Achanta et al. 2012): k-means on (colour, x, y) with centres on a grid of
//    spacing S = sqrt(N/K) (moved to the lowest gradient in a 3×3 neighbourhood), distance
//    D² = d_c² + (m d_s / S)², each centre searching only a 2S × 2S window. Channels are any
//    1 (grey) or 3 (e.g. Lab) arrays. Optional trace of labels and centres per iteration.
//  - enforceConnectivity(): every superpixel one 4-connected piece; orphan fragments (all but
//    the largest piece, and pieces below minSize) join the neighbouring superpixel they touch.
//  - gridMarkers(): one marker per grid cell, for compact watershed-style superpixels.
//  - metrics: boundaryRecall, undersegmentationError, achievableAccuracy.
//  - spGraph(): superpixel adjacency graph with mean intensities, for graph algorithms.

export interface SlicOptions {
  K: number;
  /** Compactness (in the units of the channels). */
  m: number;
  iters?: number;
  /** Move each initial centre to the lowest gradient of its 3×3 neighbourhood (default true). */
  perturb?: boolean;
  /** Run enforceConnectivity at the end (default true). */
  connectivity?: boolean;
  /** Record labels and centres after every iteration. */
  trace?: boolean;
}

export interface SlicResult {
  labels: Int32Array;          // 1..count
  count: number;
  S: number;
  /** Centres per superpixel before connectivity: [c0..cC−1, x, y] each. */
  centres: Float64Array[];
  /** Objective Σ D² after each iteration (before connectivity). */
  objective: number[];
  /** Per iteration (0 = initial assignment after one pass? no: 0 = grid start), if traced. */
  traceLabels?: Int32Array[];
  traceCentres?: Float64Array[][];
  /** Number of orphan pixels relabelled by the connectivity step. */
  orphans: number;
}

/** Initial centres on a regular grid with spacing S (half a step from the border). */
export function gridCentres(w: number, h: number, K: number): { xs: number[]; ys: number[]; S: number } {
  const S = Math.sqrt((w * h) / K);
  const nx = Math.max(1, Math.round(w / S)), ny = Math.max(1, Math.round(h / S));
  const xs: number[] = [], ys: number[] = [];
  const sx = w / nx, sy = h / ny;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { xs.push(Math.min(w - 1, Math.floor((i + 0.5) * sx))); ys.push(Math.min(h - 1, Math.floor((j + 0.5) * sy))); }
  return { xs, ys, S: Math.sqrt(sx * sy) };
}

export function slic(ch: ArrayLike<number>[], w: number, h: number, o: SlicOptions): SlicResult {
  const C = ch.length, n = w * h, iters = o.iters ?? 10;
  const g = gridCentres(w, h, o.K), S = g.S, K = g.xs.length;
  const grad = (x: number, y: number) => {
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) return Infinity;
    let s = 0;
    for (let c = 0; c < C; c++) {
      const dx = ch[c][y * w + x + 1] - ch[c][y * w + x - 1], dy = ch[c][(y + 1) * w + x] - ch[c][(y - 1) * w + x];
      s += dx * dx + dy * dy;
    }
    return s;
  };
  const cen: Float64Array[] = [];
  for (let k = 0; k < K; k++) {
    let x = g.xs[k], y = g.ys[k];
    if (o.perturb !== false) {
      let best = grad(x, y), bx = x, by = y;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const v = grad(x + dx, y + dy); if (v < best) { best = v; bx = x + dx; by = y + dy; } }
      x = bx; y = by;
    }
    const c = new Float64Array(C + 2);
    for (let j = 0; j < C; j++) c[j] = ch[j][y * w + x];
    c[C] = x; c[C + 1] = y;
    cen.push(c);
  }
  const lab = new Int32Array(n).fill(-1), dist = new Float64Array(n);
  const wxy = (o.m / S) ** 2;
  const objective: number[] = [];
  const traceLabels: Int32Array[] = [], traceCentres: Float64Array[][] = [];
  if (o.trace) { traceLabels.push(new Int32Array(n).fill(0)); traceCentres.push(cen.map(c => Float64Array.from(c))); }
  const F = Array.from(ch, c => (c instanceof Float64Array ? c : Float64Array.from(c)));
  const f0 = F[0], f1 = F[C > 1 ? 1 : 0], f2 = F[C > 2 ? 2 : 0];
  for (let it = 0; it < iters; it++) {
    dist.fill(Infinity); lab.fill(-1);
    for (let k = 0; k < K; k++) {
      const c = cen[k], cx = c[C], cy = c[C + 1];
      const x0 = Math.max(0, Math.floor(cx - S)), x1 = Math.min(w - 1, Math.ceil(cx + S));
      const y0 = Math.max(0, Math.floor(cy - S)), y1 = Math.min(h - 1, Math.ceil(cy + S));
      const c0 = c[0], c1 = C > 1 ? c[1] : 0, c2 = C > 2 ? c[2] : 0;
      for (let y = y0; y <= y1; y++) {
        const dy2 = wxy * (y - cy) * (y - cy);
        for (let x = x0, p = y * w + x0; x <= x1; x++, p++) {
          const d0 = f0[p] - c0;
          let D = d0 * d0 + dy2 + wxy * (x - cx) * (x - cx);
          if (C === 3) { const d1 = f1[p] - c1, d2 = f2[p] - c2; D += d1 * d1 + d2 * d2; }
          else if (C !== 1) { for (let j = 1; j < C; j++) { const d = F[j][p] - c[j]; D += d * d; } }
          if (D < dist[p]) { dist[p] = D; lab[p] = k; }
        }
      }
    }
    let obj = 0;
    const acc = cen.map(() => new Float64Array(C + 3));
    for (let p = 0; p < n; p++) {
      const k = lab[p];
      if (k < 0) continue;
      obj += dist[p];
      const a = acc[k], x = p % w, y = (p - x) / w;
      for (let j = 0; j < C; j++) a[j] += ch[j][p];
      a[C] += x; a[C + 1] += y; a[C + 2]++;
    }
    objective.push(obj);
    for (let k = 0; k < K; k++) { const a = acc[k]; if (a[C + 2] > 0) for (let j = 0; j < C + 2; j++) cen[k][j] = a[j] / a[C + 2]; }
    if (o.trace) { traceLabels.push(Int32Array.from(lab, v => v + 1)); traceCentres.push(cen.map(c => Float64Array.from(c))); }
  }
  // unassigned pixels (outside every window: rare) take the nearest centre in position
  for (let p = 0; p < n; p++) if (lab[p] < 0) {
    const x = p % w, y = (p - x) / w;
    let best = Infinity;
    for (let k = 0; k < K; k++) { const d = (x - cen[k][C]) ** 2 + (y - cen[k][C + 1]) ** 2; if (d < best) { best = d; lab[p] = k; } }
  }
  let labels: Int32Array = Int32Array.from(lab, v => v + 1), count = K, orphans = 0;
  if (o.connectivity !== false) {
    const r = enforceConnectivity(labels, w, h, Math.max(1, Math.floor(S * S / 4)));
    labels = r.labels; count = r.count; orphans = r.orphans;
  } else {
    const r = relabel(labels); labels = r.labels; count = r.count;
  }
  return { labels, count, S, centres: cen, objective, traceLabels: o.trace ? traceLabels : undefined, traceCentres: o.trace ? traceCentres : undefined, orphans };
}

/** Consecutive labels 1..count in raster order of first appearance (0 stays 0). */
export function relabel(lab: ArrayLike<number>): { labels: Int32Array; count: number } {
  const map = new Map<number, number>(), out = new Int32Array(lab.length);
  for (let i = 0; i < lab.length; i++) {
    const v = lab[i];
    if (v <= 0) continue;
    let m = map.get(v);
    if (m === undefined) { m = map.size + 1; map.set(v, m); }
    out[i] = m;
  }
  return { labels: out, count: map.size };
}

/** 4-connected pieces of equal label. */
function pieces(lab: ArrayLike<number>, w: number, h: number): { piece: Int32Array; sizes: number[]; labelOf: number[] } {
  const n = w * h, piece = new Int32Array(n).fill(-1), sizes: number[] = [], labelOf: number[] = [], st: number[] = [];
  for (let s = 0; s < n; s++) {
    if (piece[s] >= 0) continue;
    const id = sizes.length, L = lab[s];
    let cnt = 0;
    piece[s] = id; st.push(s);
    while (st.length) {
      const p = st.pop()!; cnt++;
      const x = p % w, y = (p - x) / w;
      if (x > 0 && piece[p - 1] < 0 && lab[p - 1] === L) { piece[p - 1] = id; st.push(p - 1); }
      if (x + 1 < w && piece[p + 1] < 0 && lab[p + 1] === L) { piece[p + 1] = id; st.push(p + 1); }
      if (y > 0 && piece[p - w] < 0 && lab[p - w] === L) { piece[p - w] = id; st.push(p - w); }
      if (y + 1 < h && piece[p + w] < 0 && lab[p + w] === L) { piece[p + w] = id; st.push(p + w); }
    }
    sizes.push(cnt); labelOf.push(L);
  }
  return { piece, sizes, labelOf };
}

/** Keep the largest piece of every label (if ≥ minSize); every other piece (an orphan) takes the
 *  label of the piece left of (or above) its first pixel in raster order, as in SLIC. Repeated
 *  until stable. Returns consecutive labels and the number of relabelled pixels. */
export function enforceConnectivity(lab: ArrayLike<number>, w: number, h: number, minSize = 1): { labels: Int32Array; count: number; orphans: number } {
  const n = w * h, cur = Int32Array.from(lab);
  let orphans = 0;
  for (let round = 0; round < 20; round++) {
    const { piece, sizes, labelOf } = pieces(cur, w, h);
    const np = sizes.length, largest = new Int32Array(np).fill(-1);
    let maxL = 0;
    for (let i = 0; i < np; i++) maxL = Math.max(maxL, labelOf[i]);
    const big = new Int32Array(maxL + 1).fill(-1);
    for (let i = 0; i < np; i++) { const L = labelOf[i]; if (big[L] < 0 || sizes[i] > sizes[big[L]]) big[L] = i; }
    for (let i = 0; i < np; i++) largest[i] = big[labelOf[i]];
    const target = new Int32Array(np).fill(-1), seen = new Uint8Array(np);
    let any = false;
    for (let p = 0; p < n; p++) {
      const a = piece[p];
      if (seen[a]) continue;
      seen[a] = 1;
      const bad = largest[a] !== a || (sizes[a] < minSize && np > 1);
      if (!bad) continue;
      const x = p % w;
      const q = x > 0 ? p - 1 : p >= w ? p - w : -1;
      if (q >= 0) { target[a] = labelOf[piece[q]]; any = true; }
      else {   // the very first pixel: take the right or lower neighbour
        const q2 = x + 1 < w && piece[p + 1] !== a ? p + 1 : p + w < n ? p + w : -1;
        if (q2 >= 0 && piece[q2] !== a) { target[a] = labelOf[piece[q2]]; any = true; }
      }
    }
    if (!any) break;
    for (let p = 0; p < n; p++) { const t = target[piece[p]]; if (t >= 0) cur[p] = t; }
  }
  for (let p = 0; p < n; p++) if (cur[p] !== lab[p]) orphans++;
  const r = relabel(cur);
  return { labels: r.labels, count: r.count, orphans };
}

/** Is every label one 4-connected piece? */
export function isConnected(lab: ArrayLike<number>, w: number, h: number): boolean {
  const { labelOf } = pieces(lab, w, h);
  return new Set(labelOf).size === labelOf.length;
}

/** About K markers on a regular grid (label image, 1..K) for marker-based (compact) watershed superpixels. */
export function gridMarkers(w: number, h: number, K: number): Int32Array {
  const g = gridCentres(w, h, K), out = new Int32Array(w * h);
  g.xs.forEach((x, k) => { out[g.ys[k] * w + x] = k + 1; });
  return out;
}

/** Boundary pixels of a label image: a 4-neighbour (right or down, and left/up) differs. */
export function labelBoundary(lab: ArrayLike<number>, w: number, h: number): Uint8Array {
  const b = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if ((x + 1 < w && lab[p + 1] !== lab[p]) || (y + 1 < h && lab[p + w] !== lab[p])) b[p] = 1;
  }
  return b;
}

/** Fraction of true boundary pixels with a superpixel boundary pixel within tol (Chebyshev distance). */
export function boundaryRecall(sp: ArrayLike<number>, truth: ArrayLike<number>, w: number, h: number, tol = 2): number {
  const bt = labelBoundary(truth, w, h), bs = labelBoundary(sp, w, h);
  let tot = 0, hit = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!bt[y * w + x]) continue;
    tot++;
    let found = false;
    for (let dy = -tol; dy <= tol && !found; dy++) for (let dx = -tol; dx <= tol && !found; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && bs[yy * w + xx]) found = true;
    }
    if (found) hit++;
  }
  return tot ? hit / tot : 1;
}

function overlaps(sp: ArrayLike<number>, truth: ArrayLike<number>): { table: Map<number, Map<number, number>>; spSize: Map<number, number> } {
  const table = new Map<number, Map<number, number>>(), spSize = new Map<number, number>();
  for (let p = 0; p < sp.length; p++) {
    const s = sp[p], t = truth[p];
    spSize.set(s, (spSize.get(s) ?? 0) + 1);
    let m = table.get(s); if (!m) { m = new Map(); table.set(s, m); }
    m.set(t, (m.get(t) ?? 0) + 1);
  }
  return { table, spSize };
}

/** Under-segmentation error (Neubert & Protzel): for each superpixel and each truth region it
 *  overlaps, the smaller of the part inside and the part outside; summed and divided by N. */
export function undersegmentationError(sp: ArrayLike<number>, truth: ArrayLike<number>): number {
  const { table, spSize } = overlaps(sp, truth);
  let e = 0;
  for (const [s, m] of table) { const S = spSize.get(s)!; for (const [, c] of m) e += Math.min(c, S - c); }
  return e / sp.length;
}

/** Achievable segmentation accuracy: label every superpixel with its majority truth label; the
 *  fraction of pixels that are then correct. An upper bound for any superpixel-level labelling. */
export function achievableAccuracy(sp: ArrayLike<number>, truth: ArrayLike<number>): number {
  const { table } = overlaps(sp, truth);
  let ok = 0;
  for (const [, m] of table) { let b = 0; for (const [, c] of m) b = Math.max(b, c); ok += b; }
  return ok / sp.length;
}

/** Superpixel adjacency (4-neighbours) with mean intensity per superpixel and boundary length per edge. */
export function spGraph(sp: ArrayLike<number>, w: number, h: number, img: ArrayLike<number>): { n: number; mean: Float64Array; size: Float64Array; cx: Float64Array; cy: Float64Array; a: Int32Array; b: Int32Array; len: Float64Array } {
  let n = 0;
  for (let i = 0; i < sp.length; i++) if (sp[i] > n) n = sp[i];
  const mean = new Float64Array(n), size = new Float64Array(n), cx = new Float64Array(n), cy = new Float64Array(n);
  const map = new Map<number, number>(), A: number[] = [], B: number[] = [], L: number[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x, s = sp[p] - 1;
    mean[s] += img[p]; size[s]++; cx[s] += x; cy[s] += y;
    const nb = [x + 1 < w ? p + 1 : -1, y + 1 < h ? p + w : -1];
    for (const q of nb) {
      if (q < 0) continue;
      const t = sp[q] - 1;
      if (t === s) continue;
      const a = Math.min(s, t), b = Math.max(s, t), key = a * n + b;
      let e = map.get(key);
      if (e === undefined) { e = A.length; map.set(key, e); A.push(a); B.push(b); L.push(0); }
      L[e]++;
    }
  }
  for (let s = 0; s < n; s++) { mean[s] /= size[s] || 1; cx[s] /= size[s] || 1; cy[s] /= size[s] || 1; }
  return { n, mean, size, cx, cy, a: Int32Array.from(A), b: Int32Array.from(B), len: Float64Array.from(L) };
}
