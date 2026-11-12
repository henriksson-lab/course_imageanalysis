// Images and local costs for the live-wire and fast-marching demos (no DOM).
//  - neuriteImage: a seeded synthetic neuron (a soma and branching thin bright neurites),
//    with its tips; used for "length along a neurite".
//  - liveWireCost: per-pixel features for the live-wire link cost of Mortensen & Barrett:
//    gradient magnitude, Laplacian-of-Gaussian zero crossings and gradient direction, or
//    a "bright ridge" cost (cheap on bright thin lines).
import { Grey, createGrey, gradient, logFilter } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";
import { rng } from "../lib/random";
import { EdgeCost } from "../lib/shortestpath";

export const SIZE = 256;

export interface Neurite { image: Grey; tips: [number, number][]; soma: [number, number] }

/** A soma with four primary neurites that branch; thin (σ ≈ 1.1 px), bright, on a dark noisy background. */
export function neuriteImage(seed = 3): Neurite {
  const r = rng(seed), w = SIZE, h = SIZE;
  const soma: [number, number] = [78, 170];
  const branches: { pts: [number, number][]; amp: number }[] = [];
  const tips: [number, number][] = [];
  const grow = (x: number, y: number, ang: number, len: number, depth: number, amp: number) => {
    const pts: [number, number][] = [[x, y]];
    let turn = 0;
    for (let s = 0; s < len; s++) {
      turn = 0.85 * turn + 0.06 * r.norm();
      ang += turn;
      x += 1.5 * Math.cos(ang); y += 1.5 * Math.sin(ang);
      if (x < 6 || y < 6 || x > w - 7 || y > h - 7) break;
      pts.push([x, y]);
      if (depth < 2 && s > 12 && s < len - 15 && r.unif() < 0.022) {
        const side = r.unif() < 0.5 ? -1 : 1;
        grow(x, y, ang + side * (0.5 + 0.4 * r.unif()), Math.floor(len * (0.35 + 0.3 * r.unif())), depth + 1, amp * (0.7 + 0.2 * r.unif()));
      }
    }
    branches.push({ pts, amp });
    tips.push(pts[pts.length - 1]);
  };
  const starts = [-0.35, -1.25, 0.45, -2.3];
  starts.forEach((a, k) => grow(soma[0] + 8 * Math.cos(a), soma[1] + 8 * Math.sin(a), a, [120, 95, 90, 60][k], 0, 120 + 20 * r.unif()));
  const img = createGrey(w, h, 28), val = new Float32Array(w * h), sg = 1.1;
  for (const b of branches) for (const [px, py] of b.pts) {
    const ix = Math.round(px), iy = Math.round(py);
    for (let y = iy - 4; y <= iy + 4; y++) for (let x = ix - 4; x <= ix + 4; x++) {
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      const v = b.amp * Math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sg * sg));
      if (v > val[y * w + x]) val[y * w + x] = v;
    }
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const d = Math.hypot(x - soma[0], y - soma[1]);
    const s = d < 11 ? 170 : d < 13 ? 170 * (13 - d) / 2 : 0;
    img.data[y * w + x] += Math.max(val[y * w + x], s);
  }
  return { image: addGaussianNoise(img, 12, seed + 101), tips, soma };
}

export interface CostWeights { wG: number; wZ: number; wD: number }
export type CostMode = "edges" | "ridge" | "valley";

export interface LiveWireFeatures {
  width: number;
  height: number;
  /** 1 − G/Gmax: low on strong edges. */
  fG: Float32Array;
  /** 0 on a zero crossing of the Laplacian of Gaussian, else 1. */
  fZ: Float32Array;
  /** Unit vector along the edge (the gradient turned by 90°), 0 where there is no gradient. */
  ux: Float32Array;
  uy: Float32Array;
  /** 1 − max(0, −LoG)/max: low on the centre line of bright thin lines. */
  fR: Float32Array;
  /** 1 − max(0, LoG)/max: low on the centre line of dark thin lines. */
  fV: Float32Array;
}

export function liveWireFeatures(img: Grey, sigma: number): LiveWireFeatures {
  const w = img.width, h = img.height, n = w * h;
  const g = gradient(img, sigma), lg = logFilter(img, sigma);
  let gmax = 0;
  for (let i = 0; i < n; i++) gmax = Math.max(gmax, g.mag.data[i]);
  const fG = new Float32Array(n), fZ = new Float32Array(n).fill(1), ux = new Float32Array(n), uy = new Float32Array(n), fR = new Float32Array(n), fV = new Float32Array(n);
  let lpos = 0, lneg = 0;
  for (let i = 0; i < n; i++) { lpos = Math.max(lpos, lg.data[i]); lneg = Math.max(lneg, -lg.data[i]); }
  for (let i = 0; i < n; i++) {
    fG[i] = 1 - (gmax > 0 ? g.mag.data[i] / gmax : 0);
    const m = g.mag.data[i];
    if (m > 1e-9) { ux[i] = g.gy.data[i] / m; uy[i] = -g.gx.data[i] / m; }
    fV[i] = 1 - Math.max(0, lg.data[i]) / Math.max(1e-9, lpos);
    fR[i] = 1 - Math.max(0, -lg.data[i]) / Math.max(1e-9, lneg);
  }
  // zero crossings: the pixel of a sign change that is closer to zero
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, a = lg.data[i];
    for (const [dx, dy] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const b = lg.data[yy * w + xx];
      if (a * b < 0 && Math.abs(a) <= Math.abs(b)) { fZ[i] = 0; break; }
    }
  }
  return { width: w, height: h, fG, fZ, ux, uy, fR, fV };
}

/** Per-pixel local cost (for display): the vertex part of the link cost. */
export function localCost(f: LiveWireFeatures, mode: CostMode, wts: CostWeights): Float32Array {
  const n = f.width * f.height, c = new Float32Array(n);
  for (let i = 0; i < n; i++) c[i] = mode === "ridge" ? f.fR[i] : mode === "valley" ? f.fV[i] : wts.wG * f.fG[i] + wts.wZ * f.fZ[i];
  return c;
}

/** Small cost per pixel of length, so that flat regions do not give zero-cost detours. */
export const EPS = 0.02;

/**
 * Link cost p → q (Mortensen & Barrett 1995): length × (ε + wG fG(q) + wZ fZ(q)) + wD fD(p, q),
 * with fD = (2/3π)(acos(u_p · L) + acos(L · u_q)) and L the unit link turned to agree with u_p.
 * "ridge" (bright line): length × (ε + fR(q)); "valley" (dark line): length × (ε + fV(q)).
 */
export function liveWireEdge(f: LiveWireFeatures, mode: CostMode, wts: CostWeights): EdgeCost {
  const w = f.width, { fG, fZ, ux, uy, fR, fV } = f, k = 2 / (3 * Math.PI);
  if (mode === "ridge") return (_p, q, len) => len * (EPS + fR[q]);
  if (mode === "valley") return (_p, q, len) => len * (EPS + fV[q]);
  return (p, q, len) => {
    let lx = ((q % w) - (p % w)) / len, ly = (Math.floor(q / w) - Math.floor(p / w)) / len;
    if (ux[p] * lx + uy[p] * ly < 0) { lx = -lx; ly = -ly; }
    const dp = Math.min(1, Math.max(-1, ux[p] * lx + uy[p] * ly)), dq = Math.min(1, Math.max(-1, lx * ux[q] + ly * uy[q]));
    const fD = k * (Math.acos(dp) + Math.acos(dq));
    return len * (EPS + wts.wG * fG[q] + wts.wZ * fZ[q]) + wts.wD * fD;
  };
}
