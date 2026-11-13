// Outer boundary of the largest 8-connected object of a mask, as the splineoutline demo needs
// it (no DOM): the crack boundary (pixel edges between object and background, followed in
// order; their midpoints lie on average on the true outline, so they are the data the
// spline is fitted to), the number of cracks (the "edge count" perimeter), and the 8-chain
// code length through the boundary pixel centres (Moore tracing from the labelling day).
// Pixel centres at integers; a pixel covers [x − ½, x + ½] × [y − ½, y + ½]; y down.
import { label2d, componentSizes } from "../lib/label";
import { traceContour } from "../moments/regionprops";
import type { Pt } from "../lib/spline";

export interface Boundary {
  /** crack midpoints in order (clockwise on screen) */
  points: Pt[];
  /** crack corners in order (the staircase polygon) */
  corners: Pt[];
  /** number of cracks = edge-count perimeter */
  edges: number;
  /** 8-chain code length through boundary pixel centres */
  chain: number;
  /** the boundary pixel centres (Moore tracing) */
  pixels: Pt[];
  /** the object mask (largest component only) */
  mask: Uint8Array;
}

export function traceBoundary(mask: ArrayLike<number>, w: number, h: number): Boundary {
  const lab = label2d(mask, w, h, 8);
  const empty: Boundary = { points: [], corners: [], edges: 0, chain: 0, pixels: [], mask: new Uint8Array(w * h) };
  if (lab.count === 0) return empty;
  const sizes = componentSizes(lab.data, lab.count);
  let best = 1;
  for (let l = 2; l <= lab.count; l++) if (sizes[l] > sizes[best]) best = l;
  const m = new Uint8Array(w * h);
  let start = -1;
  for (let i = 0; i < w * h; i++) if (lab.data[i] === best) { m[i] = 1; if (start < 0) start = i; }
  const at = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && m[y * w + x] === 1;
  // directed cracks with the object on the right (clockwise on screen, y down); vertices are
  // pixel corners, indexed by (x + ½, y + ½) → integer grid of (w + 1) × (h + 1)
  const W1 = w + 1, next = new Map<number, number[]>();
  const add = (x0: number, y0: number, x1: number, y1: number) => {
    const a = y0 * W1 + x0, b = y1 * W1 + x1;
    const l = next.get(a);
    if (l) l.push(b); else next.set(a, [b]);
  };
  let total = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!at(x, y)) continue;
    // corners of pixel (x, y) in vertex coordinates: (x, y) top-left … (x + 1, y + 1)
    if (!at(x, y - 1)) { add(x, y, x + 1, y); total++; }          // top edge, going right
    if (!at(x + 1, y)) { add(x + 1, y, x + 1, y + 1); total++; }  // right edge, going down
    if (!at(x, y + 1)) { add(x + 1, y + 1, x, y + 1); total++; }  // bottom edge, going left
    if (!at(x - 1, y)) { add(x, y + 1, x, y); total++; }          // left edge, going up
  }
  // follow the outer loop from the top-left corner of the first pixel
  const sx = start % w, sy = (start - sx) / w;
  const v0 = sy * W1 + sx;
  const corners: Pt[] = [];
  let v = v0, prevDir: Pt = [1, 0];
  for (let guard = 0; guard <= total + 2; guard++) {
    const vx = v % W1, vy = (v - vx) / W1;
    corners.push([vx - 0.5, vy - 0.5]);
    const outs = next.get(v)!;
    let u = outs[0];
    if (outs.length > 1) {
      // two pixels touch diagonally here: turn right (keep 8-connected objects together)
      let bestScore = -Infinity;
      for (const o of outs) {
        const ox = o % W1, oy = (o - ox) / W1, d: Pt = [ox - vx, oy - vy];
        const cross = prevDir[0] * d[1] - prevDir[1] * d[0];   // > 0: right turn on screen
        if (cross > bestScore) { bestScore = cross; u = o; }
      }
    }
    const ux = u % W1, uy = (u - ux) / W1;
    prevDir = [ux - vx, uy - vy];
    next.set(v, outs.filter(o => o !== u));
    v = u;
    if (v === v0) break;
  }
  const points: Pt[] = corners.map((c, i) => {
    const d = corners[(i + 1) % corners.length];
    return [(c[0] + d[0]) / 2, (c[1] + d[1]) / 2];
  });
  const tc = traceContour(m, w, h, start);
  const chain = tc.codes.reduce((s, c) => s + (c % 2 ? Math.SQRT2 : 1), 0);
  return { points, corners, edges: corners.length, chain, pixels: tc.points, mask: m };
}
