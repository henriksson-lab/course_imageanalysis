// Post-processing of SAM's decoder output, in TypeScript so each step can be shown and tested
// (demos/test/sam-wasm-test.ts compares it with the native sam2-rs pipeline). The network itself
// (image encoder, prompt encoder, mask decoder) runs in WebAssembly: demos/wasm/sam/.
//
// The decoder returns mask logits on a 256 x 256 grid that covers the whole image after it was
// resized (not padded) to a square. Restoring a mask = bilinear resize of the logits to the
// image size (half-pixel centres, as torch's interpolate with align_corners=False), then
// threshold at 0. Automatic mask generation = decode a grid of single positive points
// (three candidates each), keep candidates whose predicted IoU is high, keep those whose mask
// barely changes when the threshold moves (stability score), then remove duplicates by box NMS.

/** Side of the decoder's low-resolution logits. */
export const LOW = 256;

/** Bilinear resize of one plane (sw x sh -> dw x dh), half-pixel centres, edges clamped
 *  (torch.nn.functional.interpolate(mode="bilinear", align_corners=False, antialias=False)). */
export function upsampleBilinear(src: ArrayLike<number>, sw: number, sh: number, dw: number, dh: number,
  out: Float32Array = new Float32Array(dw * dh)): Float32Array {
  const sx = sw / dw, sy = sh / dh;
  const x0 = new Int32Array(dw), x1 = new Int32Array(dw), fx = new Float32Array(dw);
  for (let x = 0; x < dw; x++) {
    const s = Math.max(0, (x + 0.5) * sx - 0.5);
    const i = Math.min(Math.floor(s), sw - 1);
    x0[x] = i; x1[x] = Math.min(i + 1, sw - 1); fx[x] = s - i;
  }
  for (let y = 0; y < dh; y++) {
    const s = Math.max(0, (y + 0.5) * sy - 0.5);
    const j = Math.min(Math.floor(s), sh - 1), j1 = Math.min(j + 1, sh - 1), fy = s - j;
    const r0 = j * sw, r1 = j1 * sw, o = y * dw;
    for (let x = 0; x < dw; x++) {
      const a = src[r0 + x0[x]] + (src[r0 + x1[x]] - src[r0 + x0[x]]) * fx[x];
      const b = src[r1 + x0[x]] + (src[r1 + x1[x]] - src[r1 + x0[x]]) * fx[x];
      out[o + x] = a + (b - a) * fy;
    }
  }
  return out;
}

/** Plane k of a stack of 256 x 256 logit planes. */
export function plane(logits: Float32Array, k: number): Float32Array {
  return logits.subarray(k * LOW * LOW, (k + 1) * LOW * LOW);
}

/** Restore one low-resolution logit plane to the image size (w x h). */
export function restore(low: ArrayLike<number>, w: number, h: number): Float32Array {
  return upsampleBilinear(low, LOW, LOW, w, h);
}

/** Binary mask: logit > threshold. */
export function threshold(logits: ArrayLike<number>, thr = 0): Uint8Array {
  const m = new Uint8Array(logits.length);
  for (let i = 0; i < logits.length; i++) m[i] = logits[i] > thr ? 1 : 0;
  return m;
}

/** Stability score: area(logit > thr + offset) / area(logit > thr - offset). A mask whose
 *  edge sits on a steep logit slope keeps its shape when the threshold moves: score near 1. */
export function stabilityScore(logits: ArrayLike<number>, thr = 0, offset = 1): number {
  let inter = 0, union = 0;
  const hi = thr + offset, lo = thr - offset;
  for (let i = 0; i < logits.length; i++) { const v = logits[i]; if (v > hi) inter++; if (v > lo) union++; }
  return union > 0 ? inter / union : 0;
}

export type Box = [number, number, number, number];

/** Tight box [x0, y0, x1, y1] (inclusive pixel indices) and area of a binary mask; null if empty. */
export function maskBox(mask: ArrayLike<number>, w: number, h: number): { box: Box; area: number } | null {
  let x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0;
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 0; x < w; x++) if (mask[r + x]) {
      area++;
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  return area ? { box: [x0, y0, x1, y1], area } : null;
}

/** IoU of two boxes [x0, y0, x1, y1] (as the reference: widths x1 - x0, no +1). */
export function boxIoU(a: Box, b: Box): number {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

/** IoU of two binary masks of equal size. */
export function maskIoU(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let i = 0, u = 0;
  for (let p = 0; p < a.length; p++) { const x = a[p] ? 1 : 0, y = b[p] ? 1 : 0; i += x & y; u += x | y; }
  return u ? i / u : 0;
}

/** The regular grid of prompt points, in [0, 1]^2 (cell centres, as the reference). */
export function pointGrid(n: number): [number, number][] {
  const off = 1 / (2 * n), step = n === 1 ? 0 : (1 - 2 * off) / (n - 1);
  const g: [number, number][] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) g.push([off + x * step, off + y * step]);
  return g;
}

export interface AmgParams {
  /** Points per side of the prompt grid. */
  pointsPerSide: number;
  /** Keep a candidate when its predicted IoU is above this. */
  predIou: number;
  /** Keep a candidate when its stability score is at least this. */
  stability: number;
  /** Threshold offset of the stability score. */
  stabilityOffset: number;
  /** Box IoU above which the lower-scored of two candidates is removed. */
  nms: number;
}

export const AMG_DEFAULT: AmgParams = { pointsPerSide: 8, predIou: 0.8, stability: 0.95, stabilityOffset: 1, nms: 0.7 };

/** One candidate mask of automatic mask generation. */
export interface Candidate {
  /** Index of the grid point that prompted it, and which of the three outputs it is. */
  point: number;
  output: number;
  /** Grid point in image pixels. */
  px: number;
  py: number;
  iou: number;
  stability: number;
  /** Full-resolution mask, box and area; only kept for candidates that can still pass the
   *  filters at their loosest settings (see `keepMaskIou`, `keepMaskStability`). */
  mask: Uint8Array | null;
  box: Box | null;
  area: number;
}

/**
 * Turn one decoded batch (count points x 3 outputs) into candidates: predicted IoU from the
 * decoder, stability score and mask from the restored logits. The restored mask is only built
 * when iou > keepIou and stability >= keepStability (the loosest filter settings offered), so
 * memory stays bounded; the filters themselves are applied later by `filterCandidates`.
 */
export function batchCandidates(logits: Float32Array, iou: Float32Array, count: number, outputs: number,
  points: [number, number][], firstPoint: number, w: number, h: number, offset: number,
  keepIou = 0.5, keepStability = 0.5): Candidate[] {
  const out: Candidate[] = [];
  const full = new Float32Array(w * h);
  for (let b = 0; b < count; b++) for (let k = 0; k < outputs; k++) {
    const idx = b * outputs + k, s = iou[idx];
    const c: Candidate = { point: firstPoint + b, output: k, px: points[b][0], py: points[b][1], iou: s,
      stability: NaN, mask: null, box: null, area: 0 };
    if (s > keepIou) {
      upsampleBilinear(plane(logits, idx), LOW, LOW, w, h, full);
      c.stability = stabilityScore(full, 0, offset);
      if (c.stability >= keepStability) {
        const m = threshold(full, 0), mb = maskBox(m, w, h);
        if (mb) { c.mask = m; c.box = mb.box; c.area = mb.area; }
      }
    }
    out.push(c);
  }
  return out;
}

/** Why a candidate was dropped (or "kept"). */
export type Fate = "iou" | "stability" | "empty" | "nms" | "kept";

/** Apply the filters to all candidates: predicted IoU, stability, then greedy box NMS in order
 *  of predicted IoU. Returns the kept candidates (highest score first) and the fate of each. */
export function filterCandidates(cands: Candidate[], p: AmgParams): { kept: Candidate[]; fate: Fate[]; counts: Record<Fate, number> } {
  const fate: Fate[] = new Array(cands.length);
  const pass: number[] = [];
  cands.forEach((c, i) => {
    if (!(c.iou > p.predIou)) fate[i] = "iou";
    else if (!(c.stability >= p.stability)) fate[i] = "stability";
    else if (!c.mask || !c.box) fate[i] = "empty";
    else pass.push(i);
  });
  pass.sort((a, b) => cands[b].iou - cands[a].iou || a - b);
  const kept: Candidate[] = [];
  for (const i of pass) {
    const c = cands[i];
    if (kept.some((k) => boxIoU(c.box!, k.box!) > p.nms)) { fate[i] = "nms"; continue; }
    fate[i] = "kept";
    kept.push(c);
  }
  const counts: Record<Fate, number> = { iou: 0, stability: 0, empty: 0, nms: 0, kept: 0 };
  for (const f of fate) counts[f]++;
  return { kept, fate, counts };
}

/** Paint kept masks into a label image (largest first, so small objects stay visible on top). */
export function labelImage(kept: Candidate[], w: number, h: number): Int32Array {
  const lab = new Int32Array(w * h);
  const order = kept.map((c, i) => i).sort((a, b) => kept[b].area - kept[a].area);
  for (const i of order) {
    const m = kept[i].mask!;
    for (let p = 0; p < m.length; p++) if (m[p]) lab[p] = i + 1;
  }
  return lab;
}

/**
 * The image embedding (C channels x n x n, channel-major) as an RGB picture: project every
 * position's C-vector on the first three principal components (power iteration on the C x C
 * covariance, fixed start vectors), each scaled to its 1st..99th percentile.
 */
export function embeddingRGB(emb: Float32Array, C: number, n: number): Uint8ClampedArray {
  const N = n * n;
  const mean = new Float64Array(C);
  for (let c = 0; c < C; c++) { let s = 0; for (let p = 0; p < N; p++) s += emb[c * N + p]; mean[c] = s / N; }
  const X = new Float32Array(N * C); // position-major, centred
  for (let c = 0; c < C; c++) for (let p = 0; p < N; p++) X[p * C + c] = emb[c * N + p] - mean[c];
  const cov = new Float64Array(C * C);
  for (let p = 0; p < N; p++) {
    const r = p * C;
    for (let i = 0; i < C; i++) { const xi = X[r + i]; if (xi === 0) continue; const ci = i * C; for (let j = i; j < C; j++) cov[ci + j] += xi * X[r + j]; }
  }
  for (let i = 0; i < C; i++) for (let j = 0; j < i; j++) cov[i * C + j] = cov[j * C + i];
  const comps: Float64Array[] = [];
  for (let k = 0; k < 3; k++) {
    let v = new Float64Array(C).map((_, i) => Math.sin(1 + i * (k + 1) * 0.7));
    for (let it = 0; it < 60; it++) {
      const u = new Float64Array(C);
      for (let i = 0; i < C; i++) { let s = 0; const ci = i * C; for (let j = 0; j < C; j++) s += cov[ci + j] * v[j]; u[i] = s; }
      for (const q of comps) { let d = 0; for (let i = 0; i < C; i++) d += u[i] * q[i]; for (let i = 0; i < C; i++) u[i] -= d * q[i]; }
      let nrm = 0; for (let i = 0; i < C; i++) nrm += u[i] * u[i];
      nrm = Math.sqrt(nrm) || 1;
      v = u.map((x) => x / nrm);
    }
    comps.push(v);
  }
  const rgba = new Uint8ClampedArray(N * 4);
  for (let k = 0; k < 3; k++) {
    const q = comps[k], proj = new Float32Array(N);
    for (let p = 0; p < N; p++) { let s = 0; const r = p * C; for (let i = 0; i < C; i++) s += X[r + i] * q[i]; proj[p] = s; }
    const sorted = Float32Array.from(proj).sort();
    const lo = sorted[Math.floor(0.01 * (N - 1))], hi = sorted[Math.floor(0.99 * (N - 1))];
    for (let p = 0; p < N; p++) rgba[p * 4 + k] = 255 * (proj[p] - lo) / ((hi - lo) || 1);
  }
  for (let p = 0; p < N; p++) rgba[p * 4 + 3] = 255;
  return rgba;
}
