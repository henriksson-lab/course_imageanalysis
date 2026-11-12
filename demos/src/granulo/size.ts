// Granulometry by openings, computed once for all radii (no DOM; tested against
// lib/morph.ts openings in demos/test/morphology-test.ts).
//
// Opening function OF(p) = the largest r with p in the opening A ∘ B_r (−1 on the
// background). B_r is a disc (x² + y² ≤ r²), square (chessboard radius r), diamond
// (city-block radius r) or horizontal line (2r + 1 pixels); all are nested (B_r ⊆ B_r+1).
// Erosion by B_r is a threshold of a distance transform: c ∈ A ⊖ B_r ⇔ d(c) > r, with d the
// distance to the nearest background pixel in the matching metric. So r(c) = the largest r
// with d(c) > r, and A ∘ B_r = ∪ {c + B_r : r(c) ≥ r}; painting c + B_r(c) for every c gives OF.
// Pixels outside the image are ignored (neutral), as in lib/morph.ts.
// {OF ≥ r} is the union of the openings by B_r' over all r' ≥ r. For the square, diamond and
// line this is exactly A ∘ B_r (a larger one is a union of translates of a smaller one); a
// digital disc is not always a union of smaller digital discs, so for discs it can be a few
// pixels larger than A ∘ D_r. It is the usual "granulometry by discs of radius ≥ r", which
// keeps a(r) decreasing.
import { edt, chamfer } from "../lib/distance";

export type SEKind = "disc" | "square" | "diamond" | "hline";

/** Distance of each object pixel to the nearest background pixel in the metric of the SE
 *  (squared Euclidean for the disc). Infinity where there is no background (in that row for hline). */
export function seDistance(mask: ArrayLike<number>, w: number, h: number, kind: SEKind): Float32Array {
  if (kind === "disc") return edt(mask, w, h, { squared: true });
  if (kind === "square") return chamfer(mask, w, h, "chessboard");
  if (kind === "diamond") return chamfer(mask, w, h, "cityblock");
  const d = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    let last = -Infinity;
    for (let x = 0; x < w; x++) { const i = y * w + x; if (!mask[i]) { last = x; d[i] = 0; } else d[i] = x - last; }
    last = Infinity;
    for (let x = w - 1; x >= 0; x--) { const i = y * w + x; if (!mask[i]) last = x; else d[i] = Math.min(d[i], last - x); }
  }
  return d;
}

/** Largest r with c ∈ A ⊖ B_r (−1 on the background), capped at rmax. */
export function erosionRadius(mask: ArrayLike<number>, w: number, h: number, kind: SEKind, rmax: number): Int32Array {
  const d = seDistance(mask, w, h, kind), out = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) { out[i] = -1; continue; }
    const v = d[i];
    let r = !Number.isFinite(v) ? rmax : kind === "disc" ? Math.floor(Math.sqrt(v - 1) + 1e-9) : v - 1;
    if (kind === "disc") while ((r + 1) * (r + 1) < v) r++;   // guard against rounding
    out[i] = Math.min(rmax, r);
  }
  return out;
}

/** Offsets of B_r. */
export function seOffsets(kind: SEKind, r: number): [number, number][] {
  const o: [number, number][] = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    const ok = kind === "disc" ? dx * dx + dy * dy <= r * r : kind === "square" ? true : kind === "diamond" ? Math.abs(dx) + Math.abs(dy) <= r : dy === 0;
    if (ok) o.push([dx, dy]);
  }
  return o;
}

/** Opening function OF (see above), capped at rmax. */
export function openingFunction(mask: ArrayLike<number>, w: number, h: number, kind: SEKind, rmax: number): Int32Array {
  const rc = erosionRadius(mask, w, h, kind, rmax), of = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) of[i] = mask[i] ? 0 : -1;
  const offs: [number, number][][] = [];
  for (let r = 0; r <= rmax; r++) offs.push(seOffsets(kind, r));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const r = rc[y * w + x];
    if (r < 1) continue;
    for (const [dx, dy] of offs[r]) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const j = yy * w + xx;
      if (of[j] < r) of[j] = r;
    }
  }
  return of;
}

/** Area left a(r) = |A ∘ B_r| for r = 0..rmax + 1 (a(0) = |A|), and the pattern spectrum a(r) − a(r + 1). */
export function granulometry(of: ArrayLike<number>, rmax: number): { area: number[]; spectrum: number[] } {
  const hist = new Array(rmax + 2).fill(0);
  for (let i = 0; i < of.length; i++) if (of[i] >= 0) hist[Math.min(rmax, of[i])]++;
  const area: number[] = [];   // area[r] = #{OF ≥ r}
  for (let r = 0; r <= rmax + 1; r++) { let s = 0; for (let q = r; q <= rmax; q++) s += hist[q]; area[r] = s; }
  const spectrum = area.slice(0, rmax + 1).map((a, r) => a - area[r + 1]);
  return { area, spectrum };
}
