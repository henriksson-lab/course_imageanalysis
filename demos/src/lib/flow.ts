// Optical flow (no DOM; runs under node). Owner: the optical flow day (Motion elective):
// units flow-constraint/, lucas-kanade/, horn-schunck/. Shared by the demos kymoflow,
// aperture, flowviz, lkflow and hsflow; the tracking day may reuse lucasKanade for points.
//
// Conventions: x to the right, y down (rows); a flow (u, v) is in pixels per frame and says
// where the content of a pixel of frame 1 is in frame 2: I2(x + u, y + v) = I1(x, y).
// Screen angles are counter-clockwise as seen on the screen (as on the edges day).
//
//   flowDerivatives(I1, I2, σ)       Ix, Iy (mean of both frames), It = I2 − I1 (smoothed by σ)
//   lucasKanade(I1, I2, {σ, ρ})      J u = −b per pixel, J = G_ρ * (∇I ∇Iᵀ) = the structure tensor
//   lkIterative / lkPyramid          warping iterations and coarse-to-fine on Gaussian pyramids
//   HornSchunck                      data + λ·smoothness, red–black Gauss–Seidel sweeps, robust option
//   endpointError, angularError, warpResidual, flowToRGBA, quiver, divergence, curl
//   flow1d, windowLS1d               the 1D constraint −It/Ix and its window least squares
import {
  Grey, createGrey, fromFunction, gaussianBlur, gaussianKernel, sampleBilinear, gaussianPyramid, pyrUp, Border,
} from "./imageops";
import { hsvToRgb } from "./colour";
import { rng } from "./random";

/** A flow field: u (x component) and v (y component), row-major w × h, pixels per frame. */
export interface Flow { w: number; h: number; u: Float32Array; v: Float32Array }

export function createFlow(w: number, h: number): Flow {
  return { w, h, u: new Float32Array(w * h), v: new Float32Array(w * h) };
}

/** A flow as a function of position (continuous). */
export type FieldFn = (x: number, y: number) => [number, number];

/** Sample a field function at the pixel centres. */
export function sampleField(f: FieldFn, w: number, h: number): Flow {
  const F = createFlow(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [a, b] = f(x, y);
    F.u[y * w + x] = a; F.v[y * w + x] = b;
  }
  return F;
}

// ---------------------------------------------------------------------------------------
// Synthetic flow fields with ground truth

export type FieldKind = "translation" | "rotation" | "zoom" | "shear" | "swirl" | "saddle" | "regions" | "apart" | "wound" | "random";
export const FIELD_NAMES: Record<FieldKind, string> = {
  translation: "translation", rotation: "rotation", zoom: "expansion (zoom)", shear: "shear", swirl: "vortex (swirl)",
  saddle: "saddle", regions: "a disc moves, the rest is still", apart: "two cells moving apart", wound: "a wound closing", random: "smooth random",
};

export interface FieldParams {
  /** Overall strength: px per frame (translation/regions/apart/wound/random/swirl peak), or rad/frame (rotation), or 1/frame (zoom, shear, saddle). */
  s: number;
  /** Direction in degrees, counter-clockwise on screen (translation, regions). */
  angle?: number;
  /** Centre (default image centre). */
  cx?: number; cy?: number;
  /** Radius (swirl core, disc of `regions`, cells of `apart`). */
  r?: number;
  seed?: number;
}

/**
 * A synthetic flow field on a w × h image. Rotation is counter-clockwise on screen for s > 0,
 * zoom expands for s > 0, the vortex has its peak speed s at radius r.
 */
export function fieldFn(kind: FieldKind, w: number, h: number, p: FieldParams): FieldFn {
  const cx = p.cx ?? (w - 1) / 2, cy = p.cy ?? (h - 1) / 2, s = p.s, r = p.r ?? Math.min(w, h) / 4;
  const a = (p.angle ?? 0) * Math.PI / 180, dirx = Math.cos(a), diry = -Math.sin(a);
  switch (kind) {
    case "translation": return () => [s * dirx, s * diry];
    case "rotation": {
      const c = Math.cos(s), sn = Math.sin(s);
      // content at offset (dx, dy) goes to (c dx + sn dy, −sn dx + c dy): CCW on screen (y down)
      return (x, y) => { const dx = x - cx, dy = y - cy; return [c * dx + sn * dy - dx, -sn * dx + c * dy - dy]; };
    }
    case "zoom": return (x, y) => [s * (x - cx), s * (y - cy)];
    case "shear": return (_x, y) => [s * (y - cy), 0];
    case "saddle": return (x, y) => [s * (x - cx), -s * (y - cy)];
    case "swirl": {
      // tangential speed s·(ρ/r)·exp((1 − ρ²/r²)/2): peak s at ρ = r; CCW on screen = (dy, −dx)/ρ
      return (x, y) => {
        const dx = x - cx, dy = y - cy, q = (dx * dx + dy * dy) / (r * r);
        const k = s / r * Math.exp((1 - q) / 2);
        return [k * dy, -k * dx];
      };
    }
    case "regions": return (x, y) => ((x - cx) ** 2 + (y - cy) ** 2 <= r * r ? [s * dirx, s * diry] : [0, 0]);
    case "apart": {
      // two cells left and right of the centre moving away from each other (Gaussian envelopes)
      const off = Math.min(w, h) / 5;
      return (x, y) => {
        const gl = Math.exp(-((x - cx + off) ** 2 + (y - cy) ** 2) / (2 * r * r / 2));
        const gr = Math.exp(-((x - cx - off) ** 2 + (y - cy) ** 2) / (2 * r * r / 2));
        return [s * (gr - gl), 0];
      };
    }
    case "wound": {
      // two cell sheets moving into a vertical gap in the middle; fastest at the front
      return (x) => {
        const d = x - cx, edge = r / 2;
        const sp = s * Math.exp(-Math.max(0, Math.abs(d) - edge) / (w / 4));
        return Math.abs(d) <= edge ? [0, 0] : [d < 0 ? sp : -sp, 0];
      };
    }
    case "random": {
      // smooth random field: Gaussian-blurred noise, scaled to rms speed s
      const R = rng(p.seed ?? 1), nu = createGrey(w, h), nv = createGrey(w, h);
      for (let i = 0; i < w * h; i++) { nu.data[i] = R.norm(); nv.data[i] = R.norm(); }
      const su = gaussianBlur(nu, Math.min(w, h) / 8, "wrap"), sv = gaussianBlur(nv, Math.min(w, h) / 8, "wrap");
      let ss = 0;
      for (let i = 0; i < w * h; i++) ss += su.data[i] ** 2 + sv.data[i] ** 2;
      const k = s / Math.sqrt(ss / (w * h) || 1);
      return (x, y) => [k * sampleBilinear(su, x, y), k * sampleBilinear(sv, x, y)];
    }
  }
}

/** Frame 2 from frame 1 and a forward flow: I2(p + f(p)) = I1(p), by inverting p ↦ p + f(p) with fixed-point steps. */
export function advect(img: Grey, f: FieldFn, iters = 12): Grey {
  const w = img.width, h = img.height, out = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let px = x, py = y;
    for (let k = 0; k < iters; k++) { const [a, b] = f(px, py); px = x - a; py = y - b; }
    out.data[y * w + x] = sampleBilinear(img, px, py, "replicate");
  }
  return out;
}

/** A frame pair with ground truth. */
export interface FramePair { I1: Grey; I2: Grey; truth: Flow }

/**
 * Frame pair from a base image: frame 2 is the base moved by the field. For "regions" the
 * disc moves over the still base (the uncovered strip shows the base: an occlusion, as in
 * real data). Seeded Gaussian noise of standard deviation `noise` is added to both frames.
 */
export function makePair(base: Grey, kind: FieldKind, p: FieldParams, noise = 0, seed = 1): FramePair {
  const w = base.width, h = base.height, f = fieldFn(kind, w, h, p);
  let I2: Grey;
  if (kind === "regions") {
    const cx = p.cx ?? (w - 1) / 2, cy = p.cy ?? (h - 1) / 2, r = p.r ?? Math.min(w, h) / 4;
    const [dx, dy] = f(cx, cy);
    I2 = fromFunction(w, h, (x, y) => ((x - dx - cx) ** 2 + (y - dy - cy) ** 2 <= r * r ? sampleBilinear(base, x - dx, y - dy) : base.data[y * w + x]));
  } else I2 = advect(base, f);
  const truth = sampleField(f, w, h);
  return { I1: noise > 0 ? addNoise(base, noise, seed) : base, I2: noise > 0 ? addNoise(I2, noise, seed + 7919) : I2, truth };
}

function addNoise(img: Grey, sd: number, seed: number): Grey {
  const R = rng(seed), out = createGrey(img.width, img.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = img.data[i] + sd * R.norm();
  return out;
}

// ---------------------------------------------------------------------------------------
// Synthetic scenes (grey 0..255)

/** Smooth random texture: blurred seeded noise, rescaled to mean 128 and the given contrast (sd). */
export function texture(w: number, h: number, seed = 1, scale = 2, sd = 40): Grey {
  const R = rng(seed), n = createGrey(w, h);
  for (let i = 0; i < w * h; i++) n.data[i] = R.norm();
  const b = gaussianBlur(n, scale, "wrap");
  let m = 0, s2 = 0;
  for (const v of b.data) { m += v; s2 += v * v; }
  m /= w * h; const s = Math.sqrt(s2 / (w * h) - m * m) || 1;
  for (let i = 0; i < w * h; i++) b.data[i] = 128 + sd * (b.data[i] - m) / s;
  return b;
}

/** A textured ring (radii r1 < r2) around a large flat disc, on a flat background: the filling-in test. */
export function ringScene(w: number, h: number, seed = 3, r1?: number, r2?: number): Grey {
  const t = texture(w, h, seed, 1.5, 45), cx = (w - 1) / 2, cy = (h - 1) / 2;
  const a = r1 ?? Math.min(w, h) * 0.24, b = r2 ?? Math.min(w, h) * 0.38;
  return fromFunction(w, h, (x, y) => {
    const d = Math.hypot(x - cx, y - cy);
    const inRing = smoothstep(a - 1, a + 1, d) * (1 - smoothstep(b - 1, b + 1, d));
    const flat = d < a ? 150 : 90;
    return flat * (1 - inRing) + t.data[y * w + x] * inRing;
  });
}

/** Low-contrast textured substrate with a bright textured "cell" (ellipse) on it, for motion boundaries. */
export function cellScene(w: number, h: number, seed = 5): { bg: Grey; fg: Grey; r: number } {
  const bg = texture(w, h, seed, 2.5, 14), fg = texture(w, h, seed + 1, 1.5, 35);
  for (let i = 0; i < w * h; i++) { bg.data[i] = bg.data[i] - 40; fg.data[i] = fg.data[i] + 40; }
  return { bg, fg, r: Math.min(w, h) * 0.26 };
}

/** Composite: the fg disc (centre c, radius r) over bg; the disc is moved by (dx, dy). Smooth 1-px rim. */
export function composite(bg: Grey, fg: Grey, r: number, dx = 0, dy = 0): Grey {
  const w = bg.width, h = bg.height, cx = (w - 1) / 2, cy = (h - 1) / 2;
  return fromFunction(w, h, (x, y) => {
    const d = Math.hypot(x - dx - cx, y - dy - cy), a = 1 - smoothstep(r - 0.7, r + 0.7, d);
    return a > 0 ? a * sampleBilinear(fg, x - dx, y - dy) + (1 - a) * bg.data[y * w + x] : bg.data[y * w + x];
  });
}

/** Frame pair for a fg disc moving by (dx, dy) over a still bg; truth = (dx, dy) inside the disc of frame 1, 0 outside. */
export function compositePair(bg: Grey, fg: Grey, r: number, dx: number, dy: number, noise = 0, seed = 1): FramePair {
  const w = bg.width, h = bg.height, cx = (w - 1) / 2, cy = (h - 1) / 2;
  const I1 = composite(bg, fg, r), I2 = composite(bg, fg, r, dx, dy), truth = createFlow(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (Math.hypot(x - cx, y - cy) <= r) { truth.u[y * w + x] = dx; truth.v[y * w + x] = dy; }
  return { I1: noise > 0 ? addNoise(I1, noise, seed) : I1, I2: noise > 0 ? addNoise(I2, noise, seed + 7919) : I2, truth };
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------------------
// Warping and derivatives

/** Backward warp: out(x, y) = img(x + u, y + v) (bilinear). With the right flow, warp(I2) ≈ I1. */
export function warpByFlow(img: Grey, u: ArrayLike<number>, v: ArrayLike<number>, border: Border = "replicate"): Grey {
  const w = img.width, h = img.height, out = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    out.data[i] = sampleBilinear(img, x + u[i], y + v[i], border);
  }
  return out;
}

export interface FlowDerivatives { ix: Grey; iy: Grey; it: Grey }

/**
 * Ix, Iy: Gaussian derivatives at scale σ, averaged over both frames; It = G_σ*I2 − G_σ*I1.
 * σ = 0: central differences and the plain frame difference.
 */
export function flowDerivatives(I1: Grey, I2: Grey, sigma: number): FlowDerivatives {
  const w = I1.width, h = I1.height, n = w * h;
  const k0 = sigma > 0 ? gaussianKernel(sigma, 0) : new Float64Array([1]);
  const k1 = sigma > 0 ? gaussianKernel(sigma, 1) : new Float64Array([-0.5, 0, 0.5]);
  // mean of the frames for Ix, Iy (linear, so filter the mean); difference for It
  const m = new Float32Array(n), d = new Float32Array(n);
  for (let i = 0; i < n; i++) { m[i] = 0.5 * (I1.data[i] + I2.data[i]); d[i] = I2.data[i] - I1.data[i]; }
  return {
    ix: { width: w, height: h, data: sepFilter(m, w, h, k1, k0) },
    iy: { width: w, height: h, data: sepFilter(m, w, h, k0, k1) },
    it: { width: w, height: h, data: sepFilter(d, w, h, k0, k0) },
  };
}

/**
 * Separable correlation (rows with kx, then columns with ky; odd, centred kernels; replicate
 * border). A monomorphic copy of imageops.separable for speed (the LK and HS demos recompute live).
 */
export function sepFilter(src: Float32Array, w: number, h: number, kx: Float64Array, ky: Float64Array): Float32Array {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const rx = kx.length >> 1, ry = ky.length >> 1, nx = kx.length, ny = ky.length;
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      if (x - rx >= 0 && x + rx < w) { const b = o + x - rx; for (let i = 0; i < nx; i++) s += kx[i] * src[b + i]; }
      else for (let i = 0; i < nx; i++) { let xx = x - rx + i; xx = xx < 0 ? 0 : xx >= w ? w - 1 : xx; s += kx[i] * src[o + xx]; }
      tmp[o + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    const inside = y - ry >= 0 && y + ry < h;
    for (let x = 0; x < w; x++) {
      let s = 0;
      if (inside) { let b = (y - ry) * w + x; for (let j = 0; j < ny; j++, b += w) s += ky[j] * tmp[b]; }
      else for (let j = 0; j < ny; j++) { let yy = y - ry + j; yy = yy < 0 ? 0 : yy >= h ? h - 1 : yy; s += ky[j] * tmp[yy * w + x]; }
      out[y * w + x] = s;
    }
  }
  return out;
}

/** Gaussian blur with a kernel cut at 3σ (σ ≤ 0: copy). */
export function blur3(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (!(sigma > 0)) return Float32Array.from(src);
  const k = gaussianKernel(sigma, 0, Math.max(1, Math.ceil(3 * sigma)));
  return sepFilter(src, w, h, k, k);
}

// ---------------------------------------------------------------------------------------
// Lucas–Kanade

/**
 * Solve the 2×2 least-squares problem J d = −b (J = [a b; b c], rhs −(p, q)). Where J is
 * (near) singular, return the minimum-norm solution: only the component along the strong
 * eigenvector (the normal flow at an edge; 0 in a flat region).
 */
export function solveLK(a: number, b: number, c: number, p: number, q: number, rel = 1e-6, abs = 1e-9): [number, number, number, number] {
  const out = new Float64Array(4);
  solveLKInto(a, b, c, p, q, out, rel, abs);
  return [out[0], out[1], out[2], out[3]];
}

/** Allocation-free solveLK: writes du, dv, mu1, mu2 into out[0..3]. */
export function solveLKInto(a: number, b: number, c: number, p: number, q: number, out: Float64Array, rel = 1e-6, abs = 1e-9): void {
  const m = (a + c) / 2, d = Math.hypot((a - c) / 2, b);
  const mu1 = Math.max(0, m + d), mu2 = Math.max(0, m - d);
  const th = 0.5 * Math.atan2(2 * b, a - c), v1x = Math.cos(th), v1y = Math.sin(th), v2x = -v1y, v2y = v1x;
  const tol = Math.max(abs, rel * mu1);
  let du = 0, dv = 0;
  if (mu1 > tol) { const k = -(v1x * p + v1y * q) / mu1; du += k * v1x; dv += k * v1y; }
  if (mu2 > tol) { const k = -(v2x * p + v2y * q) / mu2; du += k * v2x; dv += k * v2y; }
  out[0] = du; out[1] = dv; out[2] = mu1; out[3] = mu2;
}

export interface LKResult extends Flow {
  /** Eigenvalues of J per pixel (mu1 ≥ mu2): the confidence is mu2 (Shi–Tomasi). */
  mu1: Float32Array; mu2: Float32Array;
  jxx: Grey; jxy: Grey; jyy: Grey;
}

export interface LKOptions { sigma: number; rho: number; init?: Flow }

/** One Lucas–Kanade solve (around `init` if given: frame 2 is warped back by it first). */
export function lucasKanade(I1: Grey, I2: Grey, o: LKOptions): LKResult {
  const w = I1.width, h = I1.height, n = w * h;
  const J2 = o.init ? warpByFlow(I2, o.init.u, o.init.v) : I2;
  const { ix, iy, it } = flowDerivatives(I1, J2, o.sigma);
  const X = ix.data, Y = iy.data, T = it.data;
  const pxx = new Float32Array(n), pxy = new Float32Array(n), pyy = new Float32Array(n), pxt = new Float32Array(n), pyt = new Float32Array(n);
  for (let i = 0; i < n; i++) { pxx[i] = X[i] * X[i]; pxy[i] = X[i] * Y[i]; pyy[i] = Y[i] * Y[i]; pxt[i] = X[i] * T[i]; pyt[i] = Y[i] * T[i]; }
  const G = (a: Float32Array): Grey => ({ width: w, height: h, data: blur3(a, w, h, o.rho) });
  const jxx = G(pxx), jxy = G(pxy), jyy = G(pyy), bx = G(pxt), by = G(pyt);
  const res: LKResult = { w, h, u: new Float32Array(n), v: new Float32Array(n), mu1: new Float32Array(n), mu2: new Float32Array(n), jxx, jxy, jyy };
  const tmp = new Float64Array(4);
  for (let i = 0; i < n; i++) {
    solveLKInto(jxx.data[i], jxy.data[i], jyy.data[i], bx.data[i], by.data[i], tmp);
    res.u[i] = (o.init ? o.init.u[i] : 0) + tmp[0];
    res.v[i] = (o.init ? o.init.v[i] : 0) + tmp[1];
    res.mu1[i] = tmp[2]; res.mu2[i] = tmp[3];
  }
  return res;
}

/** Iterative LK: `iters` solves, each around the previous estimate (warp, solve the rest, add). */
export function lkIterative(I1: Grey, I2: Grey, o: LKOptions & { iters: number }): LKResult {
  let r = lucasKanade(I1, I2, o);
  for (let k = 1; k < o.iters; k++) r = lucasKanade(I1, I2, { ...o, init: r });
  return r;
}

/** Coarse-to-fine LK: per level (coarsest first) `iters` warping iterations; the flow is doubled going down a level. */
export function lkPyramid(I1: Grey, I2: Grey, o: { sigma: number; rho: number; levels: number; iters: number }): LKResult & { perLevel: Flow[] } {
  const P1 = gaussianPyramid(I1, o.levels), P2 = gaussianPyramid(I2, o.levels), L = P1.length;
  let init: Flow | undefined;
  const perLevel: Flow[] = new Array(L);
  let r: LKResult | null = null;
  for (let l = L - 1; l >= 0; l--) {
    if (init && init.w !== P1[l].width) init = upsampleFlow(init, P1[l].width, P1[l].height);
    r = lkIterative(P1[l], P2[l], { sigma: o.sigma, rho: o.rho, iters: o.iters, init });
    perLevel[l] = { w: r.w, h: r.h, u: r.u, v: r.v };
    init = r;
  }
  return { ...r!, perLevel };
}

/** Double a flow onto the next finer level (pyrUp geometry: small pixel i sits at 2i). */
export function upsampleFlow(f: Flow, w: number, h: number): Flow {
  const ug = { width: f.w, height: f.h, data: f.u }, vg = { width: f.w, height: f.h, data: f.v };
  const U = pyrUp(ug as Grey, w, h), V = pyrUp(vg as Grey, w, h);
  const out = createFlow(w, h);
  for (let i = 0; i < w * h; i++) { out.u[i] = 2 * U.data[i]; out.v[i] = 2 * V.data[i]; }
  return out;
}

// ---------------------------------------------------------------------------------------
// Horn–Schunck

export interface HSOptions {
  /** Weight of the smoothness term (Horn and Schunck's α²). */
  lambda: number;
  /** Derivative scale (0: central differences). */
  sigma: number;
  /** Charbonnier (robust) smoothness instead of quadratic. */
  robust?: boolean;
  /** Charbonnier scale ε in px/frame (default 0.1). */
  eps?: number;
  /** Linearise around this flow (frame 2 warped back by it). */
  init?: Flow;
}

/**
 * Horn–Schunck as a red–black Gauss–Seidel solver of
 *   E = Σ_p (Ix u + Iy v + It)² + λ Σ_{p~q} ψ(|u_p − u_q|² + |v_p − v_q|²)
 * (4-neighbour pairs inside the image; ψ(s) = s, or Charbonnier 2ε²(√(1 + s/ε²) − 1) by
 * reweighting). Per pixel: average the neighbours (ū, v̄), then step towards the pixel's
 * constraint line: u = ū − Ix (Ix ū + Iy v̄ + It) / (λW + Ix² + Iy²), W = sum of the weights
 * (4 inside, fewer at the border). Pixels of one checkerboard colour have neighbours only of
 * the other, so each half-sweep is an exact minimisation: E never increases.
 */
export class HornSchunck implements Flow {
  readonly w: number; readonly h: number;
  u: Float32Array; v: Float32Array;
  ix: Float32Array; iy: Float32Array;
  /** It linearised around init: It0 = It(warped) − Ix u0 − Iy v0, so Ix u + Iy v + It0 = 0 is the constraint on the total flow. */
  it: Float32Array;
  iterations = 0;
  opts: HSOptions;

  constructor(I1: Grey, I2: Grey, o: HSOptions) {
    this.opts = o;
    const w = I1.width, h = I1.height, n = w * h;
    this.w = w; this.h = h;
    const J2 = o.init ? warpByFlow(I2, o.init.u, o.init.v) : I2;
    const d = flowDerivatives(I1, J2, o.sigma);
    this.ix = Float32Array.from(d.ix.data); this.iy = Float32Array.from(d.iy.data); this.it = Float32Array.from(d.it.data);
    this.u = o.init ? Float32Array.from(o.init.u) : new Float32Array(n);
    this.v = o.init ? Float32Array.from(o.init.v) : new Float32Array(n);
    if (o.init) for (let i = 0; i < n; i++) this.it[i] -= this.ix[i] * o.init.u[i] + this.iy[i] * o.init.v[i];
  }

  /** Update one pixel from its neighbours (exact minimiser of E over (u_p, v_p) with the weights fixed). */
  private updatePixel(x: number, y: number): void {
    const w = this.w, h = this.h, i = y * w + x, lam = this.opts.lambda, U = this.u, V = this.v;
    const robust = !!this.opts.robust, e2 = (this.opts.eps ?? 0.1) ** 2, ui = U[i], vi = V[i];
    let W = 0, su = 0, sv = 0;
    for (let k = 0; k < 4; k++) {
      let j: number;
      if (k === 0) { if (x === 0) continue; j = i - 1; }
      else if (k === 1) { if (x === w - 1) continue; j = i + 1; }
      else if (k === 2) { if (y === 0) continue; j = i - w; }
      else { if (y === h - 1) continue; j = i + w; }
      const uj = U[j], vj = V[j];
      const c = robust ? 1 / Math.sqrt(1 + ((ui - uj) * (ui - uj) + (vi - vj) * (vi - vj)) / e2) : 1;
      W += c; su += c * uj; sv += c * vj;
    }
    const ub = W > 0 ? su / W : ui, vb = W > 0 ? sv / W : vi;
    const Ix = this.ix[i], Iy = this.iy[i], It = this.it[i];
    const den = lam * W + Ix * Ix + Iy * Iy;
    if (den <= 0) return;
    const kk = (Ix * ub + Iy * vb + It) / den;
    U[i] = ub - Ix * kk;
    V[i] = vb - Iy * kk;
  }

  /** `n` full sweeps (each: all "black" pixels, then all "white" pixels). */
  step(n = 1): void {
    const w = this.w, h = this.h;
    for (let s = 0; s < n; s++) {
      for (let colour = 0; colour < 2; colour++)
        for (let y = 0; y < h; y++) for (let x = (y + colour) & 1; x < w; x += 2) this.updatePixel(x, y);
      this.iterations++;
    }
  }

  /** The energy (data, smoothness, total) of the current field. */
  energy(): { data: number; smooth: number; total: number } {
    const w = this.w, h = this.h, e = this.opts.eps ?? 0.1;
    let data = 0, smooth = 0;
    const psi = (s: number) => (this.opts.robust ? 2 * e * e * (Math.sqrt(1 + s / (e * e)) - 1) : s);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, r = this.ix[i] * this.u[i] + this.iy[i] * this.v[i] + this.it[i];
      data += r * r;
      if (x < w - 1) smooth += psi((this.u[i] - this.u[i + 1]) ** 2 + (this.v[i] - this.v[i + 1]) ** 2);
      if (y < h - 1) smooth += psi((this.u[i] - this.u[i + w]) ** 2 + (this.v[i] - this.v[i + w]) ** 2);
    }
    return { data, smooth, total: data + this.opts.lambda * smooth };
  }
}

/** Coarse-to-fine Horn–Schunck: `iters` sweeps per level (`warps` re-linearisations per level), flow doubled going down. */
export function hsPyramid(I1: Grey, I2: Grey, o: { lambda: number; sigma: number; robust?: boolean; levels: number; iters: number; warps?: number }): HornSchunck {
  const P1 = gaussianPyramid(I1, o.levels), P2 = gaussianPyramid(I2, o.levels), L = P1.length;
  let init: Flow | undefined;
  let hs: HornSchunck | null = null;
  for (let l = L - 1; l >= 0; l--) {
    if (init && init.w !== P1[l].width) init = upsampleFlow(init, P1[l].width, P1[l].height);
    for (let k = 0; k < (o.warps ?? 1); k++) {
      hs = new HornSchunck(P1[l], P2[l], { lambda: o.lambda, sigma: o.sigma, robust: o.robust, init });
      hs.step(o.iters);
      init = { w: hs.w, h: hs.h, u: hs.u, v: hs.v };
    }
  }
  return hs!;
}

// ---------------------------------------------------------------------------------------
// Evaluation

/** Mean over the pixels at least `margin` from the border. */
export function interiorMean(a: ArrayLike<number>, w: number, h: number, margin = 0): number {
  let s = 0, n = 0;
  for (let y = margin; y < h - margin; y++) for (let x = margin; x < w - margin; x++) { const v = a[y * w + x]; if (Number.isFinite(v)) { s += v; n++; } }
  return n ? s / n : NaN;
}

/** Endpoint error |u − u_true| per pixel and its mean (interior: `margin` px from the border). */
export function endpointError(est: Flow, truth: Flow, margin = 0): { map: Float32Array; mean: number } {
  const n = est.w * est.h, map = new Float32Array(n);
  for (let i = 0; i < n; i++) map[i] = Math.hypot(est.u[i] - truth.u[i], est.v[i] - truth.v[i]);
  return { map, mean: interiorMean(map, est.w, est.h, margin) };
}

/** Angular error (degrees) between (u, v, 1) and (u_true, v_true, 1), per pixel. */
export function angularErrorAt(u: number, v: number, ug: number, vg: number): number {
  const c = (u * ug + v * vg + 1) / (Math.sqrt(u * u + v * v + 1) * Math.sqrt(ug * ug + vg * vg + 1));
  return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
}

export function angularError(est: Flow, truth: Flow, margin = 0): { map: Float32Array; mean: number } {
  const n = est.w * est.h, map = new Float32Array(n);
  for (let i = 0; i < n; i++) map[i] = angularErrorAt(est.u[i], est.v[i], truth.u[i], truth.v[i]);
  return { map, mean: interiorMean(map, est.w, est.h, margin) };
}

/** Warping residual I2(x + u) − I1(x): needs no ground truth (small is necessary, not sufficient). */
export function warpResidual(I1: Grey, I2: Grey, f: Flow): Grey {
  const r = warpByFlow(I2, f.u, f.v);
  for (let i = 0; i < r.data.length; i++) r.data[i] -= I1.data[i];
  return r;
}

// ---------------------------------------------------------------------------------------
// Display helpers (no DOM)

/** Screen angle (CCW on screen, radians) of a flow vector. */
export const flowAngle = (u: number, v: number): number => Math.atan2(-v, u);

/** Flow colour wheel: hue = direction (CCW on screen, 0° = right = red), saturation = speed / maxSpeed, white = still. */
export function flowColour(u: number, v: number, maxSpeed: number): [number, number, number] {
  const sp = Math.hypot(u, v);
  if (!Number.isFinite(sp)) return [128, 128, 128];
  let deg = flowAngle(u, v) * 180 / Math.PI;
  deg = (deg + 360) % 360;
  const c = hsvToRgb([deg, Math.min(1, sp / (maxSpeed || 1)), 1]);
  return [255 * c[0], 255 * c[1], 255 * c[2]];
}

export function flowToRGBA(f: Flow, maxSpeed: number, mask?: ArrayLike<number>): Uint8ClampedArray {
  const n = f.w * f.h, out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    const c = mask && !mask[i] ? [150, 150, 150] : flowColour(f.u[i], f.v[i], maxSpeed);
    out[4 * i] = c[0]; out[4 * i + 1] = c[1]; out[4 * i + 2] = c[2]; out[4 * i + 3] = 255;
  }
  return out;
}

/** Largest speed in a flow (finite values; interior `margin`). */
export function maxSpeed(f: Flow, margin = 0): number {
  let m = 0;
  for (let y = margin; y < f.h - margin; y++) for (let x = margin; x < f.w - margin; x++) {
    const i = y * f.w + x, s = Math.hypot(f.u[i], f.v[i]);
    if (Number.isFinite(s) && s > m) m = s;
  }
  return m;
}

/** Arrows on a grid of pixel centres every `step` px (centred): {x, y, u, v}. */
export function quiver(f: Flow, step: number): { x: number; y: number; u: number; v: number }[] {
  const out: { x: number; y: number; u: number; v: number }[] = [];
  const o = Math.floor(((f.w - 1) % step) / 2), oy = Math.floor(((f.h - 1) % step) / 2);
  for (let y = oy; y < f.h; y += step) for (let x = o; x < f.w; x += step) {
    const i = y * f.w + x;
    out.push({ x, y, u: f.u[i], v: f.v[i] });
  }
  return out;
}

/** Divergence u_x + v_y (central differences; one-sided at the border). */
export function divergence(f: Flow): Float32Array {
  const { w, h } = f, out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = dx(f.u, w, x, y) + dy(f.v, w, h, x, y);
  return out;
}

/** Curl v_x − u_y with y down (positive = clockwise on screen); `curlScreen` gives the CCW-on-screen sign. */
export function curl(f: Flow): Float32Array {
  const { w, h } = f, out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = dx(f.v, w, x, y) - dy(f.u, w, h, x, y);
  return out;
}

/** Rotation counter-clockwise on screen: −(v_x − u_y) in y-down coordinates (2Ω for a CCW rotation by Ω rad/frame). */
export function curlScreen(f: Flow): Float32Array {
  const c = curl(f);
  for (let i = 0; i < c.length; i++) c[i] = -c[i];
  return c;
}

function dx(a: ArrayLike<number>, w: number, x: number, y: number): number {
  const i = y * w + x;
  if (w < 2) return 0;
  return x === 0 ? a[i + 1] - a[i] : x === w - 1 ? a[i] - a[i - 1] : (a[i + 1] - a[i - 1]) / 2;
}
function dy(a: ArrayLike<number>, w: number, h: number, x: number, y: number): number {
  const i = y * w + x;
  if (h < 2) return 0;
  return y === 0 ? a[i + w] - a[i] : y === h - 1 ? a[i] - a[i - w] : (a[i + w] - a[i - w]) / 2;
}

// ---------------------------------------------------------------------------------------
// 1D: the constraint −It/Ix and window least squares (the unit-1 by-hand example)

/**
 * 1D flow quantities between two profiles: Ix = central difference of frame 1 (one-sided at
 * the ends), It = f2 − f1, and the per-pixel estimate u = −It/Ix (NaN where |Ix| ≤ tol).
 */
export function flow1d(f1: ArrayLike<number>, f2: ArrayLike<number>, tol = 1e-9): { ix: Float64Array; it: Float64Array; u: Float64Array } {
  const n = f1.length, ix = new Float64Array(n), it = new Float64Array(n), u = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    ix[x] = n < 2 ? 0 : x === 0 ? f1[1] - f1[0] : x === n - 1 ? f1[n - 1] - f1[n - 2] : (f1[x + 1] - f1[x - 1]) / 2;
    it[x] = f2[x] - f1[x];
    u[x] = Math.abs(ix[x]) > tol ? -it[x] / ix[x] : NaN;
  }
  return { ix, it, u };
}

/** Window least squares u(x) = −Σ Ix It / Σ Ix² over [x − r, x + r] (NaN where Σ Ix² ≤ tol). */
export function windowLS1d(ix: ArrayLike<number>, it: ArrayLike<number>, r: number, tol = 1e-9): Float64Array {
  const n = ix.length, out = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    let a = 0, b = 0;
    for (let k = Math.max(0, x - r); k <= Math.min(n - 1, x + r); k++) { a += ix[k] * ix[k]; b += ix[k] * it[k]; }
    out[x] = a > tol ? -b / a : NaN;
  }
  return out;
}
