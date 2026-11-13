// Model of the celltypes demo (no DOM): a synthetic tissue (a blob-shaped mask with a round
// vessel) holding two cell types (0 = tumour, 1 = immune), several generators, the null models
// (relocation in the mask, toroidal shift, permuted labels, inhomogeneous Poisson from ρ̂) and a
// fast kernel intensity with the edge factor of the mask on a 4 µm grid.
import { rng, gauss } from "../lib/random";
import * as PP from "../lib/pointpattern";

export const SIDE = 200;
export const VESSEL: [number, number, number] = [78, 92, 13];   // centre x, y and radius (µm)
export const RMAX = 40;
export const RADII = PP.linspace(0, RMAX, 41);

let tissueWin: PP.Window | null = null, imageWin: PP.Window | null = null;
/** The tissue mask (the vessel is a hole) and the whole image as windows. */
export function tissueWindow(): PP.Window {
  if (!tissueWin) tissueWin = PP.maskWindow(PP.blobMask(rng(11), SIDE, SIDE, { radius: 80, wobble: 0.14, aspect: 0.82, hole: VESSEL }), SIDE, SIDE);
  return tissueWin;
}
export function imageWindow(): PP.Window {
  if (!imageWin) imageWin = PP.rectWindow(SIDE, SIDE);
  return imageWin;
}

export type Gen = "random" | "independent" | "attraction" | "segregation" | "labelled" | "packed" | "gradient";
export const GEN_LABELS: Record<Gen, string> = {
  random: "cells at random, random types", independent: "tumour nests, independent immune cells", attraction: "immune cells near tumour cells", segregation: "segregated domains",
  labelled: "clustered cells, random types", packed: "packed tissue, random types", gradient: "density gradient to the vessel",
};

export interface TissueOptions { gen: Gen; n: number; share1: number; strength: number; seed: number }
export const DEFAULT_TISSUE: TissueOptions = { gen: "attraction", n: 300, share1: 0.3, strength: 0.7, seed: 1 };

const vesselDist = (x: number, y: number) => Math.hypot(x - VESSEL[0], y - VESSEL[1]) - VESSEL[2];

/** Tumour nests: colonies of radius 18 µm, about 20 cells each, cells at least 5 µm apart. */
const nests = (r: ReturnType<typeof rng>, n0: number, win: PP.Window) => PP.colonies(r, 14 / (SIDE * SIDE), n0 / 14 / 0.6 * 1.0, 18, 5, win);

/** A tissue pattern: marks 0 (tumour) and 1 (immune). */
export function makeTissue(o: TissueOptions): PP.Points {
  const win = tissueWindow(), r = rng(100 * o.seed + Object.keys(GEN_LABELS).indexOf(o.gen));
  const n1 = Math.round(o.n * o.share1), n0 = o.n - n1;
  switch (o.gen) {
    case "random": return PP.labelRandom(r, PP.binomialProcess(r, o.n, win), o.share1);
    case "independent": {
      const a = nests(r, n0, win), b = PP.ssi(r, n1, 5, win);
      return PP.concatPoints([a, b]);
    }
    case "attraction": {
      const a = nests(r, n0, win), xs: number[] = [], ys: number[] = [];
      for (let t = 0; xs.length < n1 && t < 100000; t++) {
        let x: number, y: number;
        if (r.unif() < o.strength) { const k = Math.floor(r.unif() * a.x.length), ang = 2 * Math.PI * r.unif(), d = 5 + 4 * r.unif(); x = a.x[k] + d * Math.cos(ang); y = a.y[k] + d * Math.sin(ang); }
        else { x = r.unif() * SIDE; y = r.unif() * SIDE; }
        if (!PP.inside(win, x, y)) continue;
        if (xs.some((u, i) => (u - x) ** 2 + (ys[i] - y) ** 2 < 16)) continue;
        xs.push(x); ys.push(y);
      }
      return PP.concatPoints([a, PP.makePoints(xs, ys)]);
    }
    case "segregation": {
      // all cells as one hard-core pattern; types by a smooth random field (domains)
      const all = PP.ssi(r, o.n, 5, win), cx: number[] = [], cy: number[] = [], s: number[] = [];
      for (let k = 0; k < 8; k++) { cx.push(r.unif() * SIDE); cy.push(r.unif() * SIDE); s.push(k % 2 ? 1 : -1); }
      const field = (x: number, y: number) => cx.reduce((acc, c, k) => acc + s[k] * Math.exp(-((x - c) ** 2 + (y - cy[k]) ** 2) / (2 * 30 * 30)), 0);
      const vals = Array.from(all.x, (x, i) => field(x, all.y[i]) + gauss(r, 0, 0.3 * (1 - o.strength)));
      const thr = [...vals].sort((a, b) => b - a)[n1 - 1];
      return { x: all.x, y: all.y, marks: Int32Array.from(vals, v => (v >= thr ? 1 : 0)) };
    }
    case "labelled": {
      const all = PP.thomas(r, 26 / (SIDE * SIDE), o.n / 26 / 0.6, 6, win);
      return PP.labelRandom(r, all, o.share1);
    }
    case "packed": {
      const all = PP.ssi(r, o.n, 7.5, win, 5000);
      return PP.labelRandom(r, all, o.share1);
    }
    case "gradient": {
      const rhoFn = (x: number, y: number) => Math.exp(-Math.max(0, vesselDist(x, y)) / 35);
      const all = PP.inhomBinomial(r, o.n, rhoFn, 1, win);
      return PP.labelRandom(r, all, o.share1);
    }
  }
}

// ---------------------------------------------------------------------------------------
// Kernel intensity with a precomputed edge factor (4 µm grid)

const G = 4, NG = SIDE / G;
const edgeCache = new Map<string, Float64Array>();
/** Edge factor e(x) = ∫_W k_σ(x − y) dy on the 4 µm grid (cell centres). */
export function edgeGrid(win: PP.Window, sigma: number): Float64Array {
  const key = `${win.mask ? "m" : "r"}|${sigma}`;
  const hit = edgeCache.get(key);
  if (hit) return hit;
  const frac = new Float64Array(NG * NG);
  for (let gy = 0; gy < NG; gy++) for (let gx = 0; gx < NG; gx++) {
    let c = 0;
    for (let y = gy * G; y < gy * G + G; y++) for (let x = gx * G; x < gx * G + G; x++) c += win.mask ? win.mask[y * SIDE + x] : 1;
    frac[gy * NG + gx] = c / (G * G);
  }
  const out = new Float64Array(NG * NG), c0 = G * G / (2 * Math.PI * sigma * sigma), R = Math.ceil(4 * sigma / G);
  for (let gy = 0; gy < NG; gy++) for (let gx = 0; gx < NG; gx++) {
    let s = 0;
    for (let qy = Math.max(0, gy - R); qy <= Math.min(NG - 1, gy + R); qy++) for (let qx = Math.max(0, gx - R); qx <= Math.min(NG - 1, gx + R); qx++) {
      const f = frac[qy * NG + qx];
      if (f > 0) s += f * c0 * Math.exp(-(((qx - gx) * G) ** 2 + ((qy - gy) * G) ** 2) / (2 * sigma * sigma));
    }
    out[gy * NG + gx] = s;
  }
  edgeCache.set(key, out);
  return out;
}
const edgeAt = (e: Float64Array, x: number, y: number) => e[Math.min(NG - 1, Math.floor(y / G)) * NG + Math.min(NG - 1, Math.floor(x / G))];

/** Leave-one-out kernel intensity at the points, edge-corrected. */
export function rhoAtPoints(p: PP.Points, win: PP.Window, sigma: number): Float64Array {
  const e = edgeGrid(win, sigma), n = p.x.length, out = new Float64Array(n), c = 1 / (2 * Math.PI * sigma * sigma), cut = 16 * sigma * sigma;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d2 = (p.x[i] - p.x[j]) ** 2 + (p.y[i] - p.y[j]) ** 2;
      if (d2 < cut) s += c * Math.exp(-d2 / (2 * sigma * sigma));
    }
    out[i] = Math.max(s / Math.max(edgeAt(e, p.x[i], p.y[i]), 0.05), 1e-6);
  }
  return out;
}

/** Kernel intensity on the 4 µm grid (NaN outside the window). */
export function rhoGrid(p: PP.Points, win: PP.Window, sigma: number): Float64Array {
  const e = edgeGrid(win, sigma), out = new Float64Array(NG * NG), c = 1 / (2 * Math.PI * sigma * sigma), cut = 16 * sigma * sigma;
  for (let gy = 0; gy < NG; gy++) for (let gx = 0; gx < NG; gx++) {
    const x = (gx + 0.5) * G, y = (gy + 0.5) * G;
    if (!PP.inside(win, x, y)) { out[gy * NG + gx] = NaN; continue; }
    let s = 0;
    for (let j = 0; j < p.x.length; j++) { const d2 = (p.x[j] - x) ** 2 + (p.y[j] - y) ** 2; if (d2 < cut) s += c * Math.exp(-d2 / (2 * sigma * sigma)); }
    out[gy * NG + gx] = s / Math.max(e[gy * NG + gx], 0.05);
  }
  return out;
}
export const GRID_N = NG;

/** n points from a gridded intensity (piecewise constant on the 4 µm cells), inside the window. */
export function sampleFromGrid(r: ReturnType<typeof rng>, n: number, grid: Float64Array, win: PP.Window): PP.Points {
  let mx = 0;
  for (const v of grid) if (Number.isFinite(v)) mx = Math.max(mx, v);
  return PP.inhomBinomial(r, n, (x, y) => { const v = grid[Math.min(NG - 1, Math.floor(y / G)) * NG + Math.min(NG - 1, Math.floor(x / G))]; return Number.isFinite(v) ? v : 0; }, mx, win);
}

// ---------------------------------------------------------------------------------------
// Curves and nulls

export type Curve = "cross" | "L" | "mcf" | "inhom";
export type Null = "relocate" | "shift" | "permute" | "csr" | "inhomPoisson";

/** The curve of the current tab: cross L12 − r, L − r of all cells, p01(r), or inhomogeneous L − r. */
export function curveOf(p: PP.Points, win: PP.Window, c: Curve, bw: number): Float64Array {
  if (c === "inhom") {
    const pd = PP.pairData(p, win, RMAX, "translation");
    return PP.LminusR(PP.KinhomFromPairs(pd, rhoAtPoints(p, win, bw), RADII), RADII);
  }
  if (c === "mcf") return PP.markConnection(PP.pairData(p, win, RMAX + 3, "translation"), p.marks!, RADII, 3, 0, 1);
  const pd = PP.pairData(p, win, RMAX, "translation");
  if (c === "L") return PP.LminusR(PP.KfromPairs(pd, RADII), RADII);
  return PP.LminusR(PP.KfromPairs(pd, RADII, { marks: p.marks, a: 0, b: 1 }), RADII);
}

/** One simulation under the chosen null. Types: type 1 is moved (relocate, shift), labels are permuted, or all points are redrawn. */
export function simulateNull(r: ReturnType<typeof rng>, p: PP.Points, win: PP.Window, nul: Null, bw: number, gridRho?: Float64Array): PP.Points {
  const t0 = PP.ofType(p, 0), t1 = PP.ofType(p, 1);
  switch (nul) {
    case "relocate": return PP.concatPoints([t0, PP.binomialProcess(r, t1.x.length, win)]);
    case "shift": {
      const s = PP.toroidalShift(t1, r.unif() * SIDE, r.unif() * SIDE, PP.rectWindow(SIDE, SIDE));
      return PP.concatPoints([t0, PP.clip(s, win)]);
    }
    case "permute": return PP.relabel(r, p);
    case "csr": return { ...PP.binomialProcess(r, p.x.length, win), marks: p.marks };
    case "inhomPoisson": return { ...sampleFromGrid(r, p.x.length, gridRho ?? rhoGrid(p, win, bw), win), marks: p.marks };
  }
}

/** The reference curve under the null: 0 for L − r, 2·p0·p1 for the mark connection function. */
export function referenceOf(p: PP.Points, c: Curve): Float64Array {
  if (c !== "mcf") return new Float64Array(RADII.length);
  let n1 = 0;
  for (const m of p.marks!) if (m === 1) n1++;
  const n = p.x.length, v = 2 * n1 * (n - n1) / (n * (n - 1));
  return new Float64Array(RADII.length).fill(v);
}
