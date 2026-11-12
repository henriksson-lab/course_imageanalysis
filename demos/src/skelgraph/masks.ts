// Masks of the skelgraph demo (no DOM, so node scripts can use them).
import { rasterise, ellipsePoly, Poly } from "../lib/shapecanvas";
import { rng } from "../lib/random";
import { edt } from "../lib/distance";
import { label2d } from "../lib/label";
import { growTree, renderTree } from "../lib/trees";
import { skeletonize, prune, skeletonGraph, SkelGraph, LengthEstimator } from "../lib/skeleton";

export const N = 256;
export type MaskId = "Y" | "spurs" | "loop" | "line" | "vessel" | "draw";

/** A rectangle of half-width hw from (x0, y0) to (x1, y1), clockwise on screen. */
/** Make a polygon run clockwise on screen (positive shoelace sum with y down), for nonzero unions. */
export function orient(p: Poly): Poly {
  let a = 0;
  for (let i = 0; i < p.length; i++) { const u = p[i], v = p[(i + 1) % p.length]; a += u[0] * v[1] - v[0] * u[1]; }
  return a < 0 ? p.reverse() : p;
}

/** A tapering arm from (x0, y0) (half-width w0) to (x1, y1) (half-width w1). */
export function arm(x0: number, y0: number, x1: number, y1: number, w0: number, w1: number): Poly {
  const L = Math.hypot(x1 - x0, y1 - y0) || 1, nx = -(y1 - y0) / L, ny = (x1 - x0) / L;
  return orient([[x0 + nx * w0, y0 + ny * w0], [x1 + nx * w1, y1 + ny * w1], [x1 - nx * w1, y1 - ny * w1], [x0 - nx * w0, y0 - ny * w0]]);
}

export function bar(x0: number, y0: number, x1: number, y1: number, hw: number): Poly {
  const L = Math.hypot(x1 - x0, y1 - y0) || 1, nx = -(y1 - y0) / L * hw, ny = (x1 - x0) / L * hw;
  const p: Poly = [[x0 + nx, y0 + ny], [x1 + nx, y1 + ny], [x1 - nx, y1 - ny], [x0 - nx, y0 - ny]];
  // make every polygon run the same way (nonzero union)
  let a = 0;
  for (let i = 0; i < 4; i++) { const u = p[i], v = p[(i + 1) % 4]; a += u[0] * v[1] - v[0] * u[1]; }
  return a < 0 ? p.reverse() : p;
}

/** Digital straight line of `len` pixels' extent from (x0, y0) at angle θ (degrees, up on screen). */
export function digitalLine(x0: number, y0: number, len: number, thetaDeg: number): { path: number[]; truth: number } {
  const th = thetaDeg * Math.PI / 180, x1 = Math.round(x0 + len * Math.cos(th)), y1 = Math.round(y0 - len * Math.sin(th));
  const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)), path: number[] = [];
  for (let k = 0; k <= n; k++) path.push(Math.round(y0 + (y1 - y0) * k / n) * N + Math.round(x0 + (x1 - x0) * k / n));
  return { path, truth: Math.hypot(x1 - x0, y1 - y0) };
}

export function makeMask(id: MaskId, theta: number): Uint8Array {
  if (id === "Y") return rasterise([arm(128, 140, 128, 230, 10, 0.5), arm(128, 140, 56, 46, 10, 0.5), arm(128, 140, 200, 46, 10, 0.5), orient(ellipsePoly(128, 140, 13, 13, 24))], N, N, "nonzero");
  if (id === "spurs") {
    // a thick "T" with a rough boundary (seeded bumps)
    const m = rasterise([bar(36, 60, 220, 60, 16), bar(128, 60, 128, 228, 16)], N, N, "nonzero");
    const r = rng(5), dt = edt(m, N, N);
    const edge: number[] = [];
    for (let i = 0; i < N * N; i++) if (m[i] && dt[i] < 1.5) edge.push(i);
    for (let k = 0; k < 26; k++) {
      const i = edge[Math.floor(r.unif() * edge.length)], cx = i % N, cy = Math.floor(i / N), rad = 2 + 3 * r.unif();
      for (let y = Math.floor(cy - rad); y <= cy + rad; y++) for (let x = Math.floor(cx - rad); x <= cx + rad; x++)
        if (x > 0 && y > 0 && x < N - 1 && y < N - 1 && Math.hypot(x - cx, y - cy) < rad) m[y * N + x] = 1;
    }
    return m;
  }
  if (id === "loop") return rasterise([[[40, 40], [216, 40], [216, 216], [40, 216]], ellipsePoly(92, 128, 26, 52, 20), [[148, 70], [186, 70], [186, 110], [148, 110]], ellipsePoly(166, 170, 16, 20, 16)], N, N, "evenodd");
  if (id === "line") {
    const m = new Uint8Array(N * N);
    for (const i of digitalLine(40, 216, 170, theta).path) m[i] = 1;
    return m;
  }
  if (id === "vessel") {
    const t = growTree({ kind: "vessel", seed: 11 }), { image } = renderTree(t, { noise: 14, blur: 1.2, seed: 4, blobs: 2 });
    const m = Uint8Array.from(image.data, v => (v > 80 ? 1 : 0));
    // remove specks below 30 pixels (area filter, labelling day)
    const L = label2d(m, N, N, 8), size = new Int32Array(L.count + 1);
    for (let i = 0; i < N * N; i++) size[L.data[i]]++;
    for (let i = 0; i < N * N; i++) if (size[L.data[i]] < 30) m[i] = 0;
    // fill holes up to 20 pixels (noise inside the thick vessels)
    const H = label2d(Uint8Array.from(m, v => 1 - v), N, N, 4), hs = new Int32Array(H.count + 1);
    for (let i = 0; i < N * N; i++) hs[H.data[i]]++;
    for (let i = 0; i < N * N; i++) if (H.data[i] && hs[H.data[i]] <= 20) m[i] = 1;
    return m;
  }
  return rasterise([bar(60, 200, 128, 120, 10), bar(128, 120, 200, 70, 8)], N, N, "nonzero");
}


/** Skeleton (unless the mask is already a line), pruning with k (pixels, or k/4 × DT if relative), and the graph. */
export function analyseMask(mask: Uint8Array, isLine: boolean, k: number, relative: boolean, est: LengthEstimator): { dt: Float32Array; skel0: Uint8Array; skel: Uint8Array; removed: number; graph: SkelGraph } {
  const dt = edt(mask, N, N);
  const skel0 = isLine ? mask.slice() : skeletonize(mask, N, N);
  let skel = skel0, removed = 0;
  if (k > 0) { const pr = prune(skel0, N, N, relative ? k / 4 : k, { est, dt: relative ? dt : undefined }); skel = pr.skeleton; removed = pr.removed; }
  return { dt, skel0, skel, removed, graph: skeletonGraph(skel, N, N) };
}
