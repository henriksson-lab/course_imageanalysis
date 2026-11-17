// Augmentations and the crop experiment of the augmentation unit. Images are Grey with values
// in [0, 1]. Geometric changes are inverse mappings with bilinear sampling and mirrored borders
// (positive angles turn clockwise on screen, as in the course code). The experiment: 16 × 16
// crops of round or elongated cells; 20 training crops whose elongated cells all lie (within
// 15° of horizontal), 200 validation crops in any direction and brightness; a network
// 256 → 32 → 2 trained with Adam with and without augmentation (lib/nn.ts).
// No DOM, so the exercise numbers come from this code (demos/test/training-test.ts).
import { rng } from "../lib/ui";
import { createGrey, gaussianBlur, warp, type Grey } from "../lib/imageops";
import { mlp, gradients, applyGradients, Adam, Batcher, meanLoss, predict, type Mlp } from "../lib/nn";

type R = ReturnType<typeof rng>;

// ---------------------------------------------------------------------------------------
// Augmentations

export function flipH(img: Grey): Grey {
  return warp(img, img.width, img.height, (x, y) => [img.width - 1 - x, y], "nearest", "mirror");
}
export function flipV(img: Grey): Grey {
  return warp(img, img.width, img.height, (x, y) => [x, img.height - 1 - y], "nearest", "mirror");
}
/** Rotation by `deg` degrees about the centre, clockwise on screen, mirrored borders. */
export function rotateCw(img: Grey, deg: number): Grey {
  const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const cx = (img.width - 1) / 2, cy = (img.height - 1) / 2;
  return warp(img, img.width, img.height, (x, y) => {
    const dx = x - cx, dy = y - cy;
    return [cx + c * dx + s * dy, cy - s * dx + c * dy];
  }, "bilinear", "mirror");
}
/** Zoom by `f` about the centre (f > 1 enlarges: the same object seen with smaller pixels). */
export function zoom(img: Grey, f: number): Grey {
  const cx = (img.width - 1) / 2, cy = (img.height - 1) / 2;
  return warp(img, img.width, img.height, (x, y) => [cx + (x - cx) / f, cy + (y - cy) / f], "bilinear", "mirror");
}
/** A smooth random displacement field: normal noise blurred with σ, scaled to rms `alpha` pixels. */
export function displacement(w: number, h: number, alpha: number, sigma: number, r: R): Float32Array {
  const g = createGrey(w, h);
  for (let i = 0; i < g.data.length; i++) g.data[i] = r.norm();
  const s = gaussianBlur(g, sigma, "replicate");
  let ms = 0;
  for (const v of s.data) ms += v * v;
  const k = alpha / Math.sqrt(ms / s.data.length || 1);
  return s.data.map(v => v * k);
}
/** Elastic deformation: the output pixel (x, y) reads the input at (x + u, y + v). */
export function elastic(img: Grey, alpha: number, sigma: number, seed: number): Grey {
  if (alpha <= 0) return { width: img.width, height: img.height, data: img.data.slice() };
  const r = rng(seed), u = displacement(img.width, img.height, alpha, sigma, r), v = displacement(img.width, img.height, alpha, sigma, r);
  return warp(img, img.width, img.height, (x, y) => { const i = y * img.width + x; return [x + u[i], y + v[i]]; }, "bilinear", "mirror");
}
/** gain · v^gamma + offset on every grey value (negative values to 0 first). */
export function intensity(img: Grey, gain: number, offset: number, gamma: number): Grey {
  return { width: img.width, height: img.height, data: img.data.map(v => gain * Math.max(0, v) ** gamma + offset) };
}
export function addNoise(img: Grey, sd: number, r: R): Grey {
  return { width: img.width, height: img.height, data: img.data.map(v => v + sd * r.norm()) };
}

/** All settings of the explorer, applied in this order: flips, zoom, rotation, elastic,
 *  intensity, blur, noise. */
export interface Explore {
  flipH: boolean; flipV: boolean; angle: number; zoom: number;
  alpha: number; sigma: number; gain: number; offset: number; gamma: number;
  blur: number; noise: number; seed: number;
}
export const NO_CHANGE: Explore = { flipH: false, flipV: false, angle: 0, zoom: 1, alpha: 0, sigma: 8, gain: 1, offset: 0, gamma: 1, blur: 0, noise: 0, seed: 1 };

export function applyExplore(img: Grey, e: Explore): Grey {
  let out = img;
  if (e.flipH) out = flipH(out);
  if (e.flipV) out = flipV(out);
  if (e.zoom !== 1) out = zoom(out, e.zoom);
  if (e.angle !== 0) out = rotateCw(out, e.angle);
  if (e.alpha > 0) out = elastic(out, e.alpha, e.sigma, e.seed);
  if (e.gain !== 1 || e.offset !== 0 || e.gamma !== 1) out = intensity(out, e.gain, e.offset, e.gamma);
  if (e.blur > 0) out = gaussianBlur(out, e.blur, "mirror");
  if (e.noise > 0) out = addNoise(out, e.noise, rng(e.seed + 1000));
  return out;
}

/** A random draw within ranges: flips each with probability 1/2, angle in ±maxAngle, zoom in
 *  [1/(1+z), 1+z], elastic with the given alpha, gain in 1 ± gainRange, noise as given. */
export interface Ranges { flips: boolean; maxAngle: number; zoomRange: number; alpha: number; sigma: number; gainRange: number; noise: number }
export function randomExplore(rg: Ranges, r: R): Explore {
  const u = () => r.unif();
  return {
    flipH: rg.flips && u() < 0.5, flipV: rg.flips && u() < 0.5,
    angle: rg.maxAngle * (2 * u() - 1),
    zoom: rg.zoomRange > 0 ? Math.exp(Math.log(1 + rg.zoomRange) * (2 * u() - 1)) : 1,
    alpha: rg.alpha, sigma: rg.sigma,
    gain: 1 + rg.gainRange * (2 * u() - 1), offset: 0, gamma: 1, blur: 0, noise: rg.noise,
    seed: Math.floor(u() * 1e9),
  };
}

// ---------------------------------------------------------------------------------------
// The crop experiment

export type Task = "shape" | "orientation";
export const SIZE = 16;

/** A SIZE × SIZE crop with one smooth-edged ellipse (4 × 4 sub-pixel samples), background 0.1. */
export function ellipseCrop(cx: number, cy: number, a: number, b: number, angleDeg: number, bright: number): Grey {
  const t = angleDeg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const g = createGrey(SIZE, SIZE);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    let inside = 0;
    for (let sy = 0; sy < 4; sy++) for (let sx = 0; sx < 4; sx++) {
      const dx = x + (sx + 0.5) / 4 - 0.5 - cx, dy = y + (sy + 0.5) / 4 - 0.5 - cy;
      const u = c * dx + s * dy, v = -s * dx + c * dy;
      if ((u / a) ** 2 + (v / b) ** 2 <= 1) inside += 1 / 16;
    }
    g.data[y * SIZE + x] = 0.1 + (bright - 0.1) * inside;
  }
  return g;
}

export interface CropSet { imgs: Grey[]; y: number[] }

/** n crops, classes alternating. Shape: class 0 round, class 1 elongated; in the training set
 *  the elongated ones lie within 15° of horizontal and the brightness is 0.8–1.0, in the
 *  validation set any direction and 0.5–1.1. Orientation: class 0 lying, class 1 standing, both
 *  sets alike. Noise sd 0.03. */
export function cropSet(task: Task, n: number, train: boolean, seed: number): CropSet {
  const r = rng(seed), imgs: Grey[] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const cx = 8 + 2 * r.unif() - 1, cy = 8 + 2 * r.unif() - 1, jitter = 30 * r.unif() - 15;
    let a: number, b: number, ang: number;
    if (task === "shape" && cls === 0) { a = 3 + 1.2 * r.unif(); b = a * (0.9 + 0.1 * r.unif()); ang = 180 * r.unif(); }
    else if (task === "shape") { a = 5 + 1.5 * r.unif(); b = 2 + 0.7 * r.unif(); ang = train ? jitter : 180 * r.unif(); }
    else { a = 5 + 1.5 * r.unif(); b = 2 + 0.7 * r.unif(); ang = 90 * cls + jitter; }
    const bright = train || task === "orientation" ? 0.8 + 0.2 * r.unif() : 0.5 + 0.6 * r.unif();
    const g = ellipseCrop(cx, cy, a, b, ang, bright);
    for (let k = 0; k < g.data.length; k++) g.data[k] += 0.03 * r.norm();
    imgs.push(g); y.push(cls);
  }
  return { imgs, y };
}

export interface Aug { flips: boolean; maxAngle: number; gainRange: number; noise: number; elastic: number }
export const NO_AUG: Aug = { flips: false, maxAngle: 0, gainRange: 0, noise: 0, elastic: 0 };
export const isOff = (a: Aug) => !a.flips && a.maxAngle === 0 && a.gainRange === 0 && a.noise === 0 && a.elastic === 0;

/** One random augmented copy of a crop for training. */
export function augmentCrop(img: Grey, a: Aug, r: R): Grey {
  let out = img;
  if (a.flips) { if (r.unif() < 0.5) out = flipH(out); if (r.unif() < 0.5) out = flipV(out); }
  if (a.maxAngle > 0) out = rotateCw(out, a.maxAngle * (2 * r.unif() - 1));
  if (a.elastic > 0) out = elastic(out, a.elastic, 3, Math.floor(r.unif() * 1e9));
  if (a.gainRange > 0) { const g = 1 + a.gainRange * (2 * r.unif() - 1); out = intensity(out, g, 0, 1); }
  if (a.noise > 0) out = addNoise(out, a.noise, r);
  return out;
}

const onehot = (c: number): number[] => (c === 0 ? [1, 0] : [0, 1]);

export interface EpochPoint { epoch: number; train: number; val: number; trainAcc: number; valAcc: number }

/** One training run of the network 256 → 32 → 2 (ReLU, softmax, cross-entropy), Adam 0.003,
 *  batches of 10, with augmentation `aug` and weight decay `decay`. */
export class CropRun {
  net: Mlp; opt: Adam; batcher: Batcher; r: R;
  X: number[][]; Y: number[][]; Xv: number[][]; Yv: number[][];
  history: EpochPoint[] = [];
  epoch = 0;
  lastBatch: Grey[] = [];
  constructor(public tr: CropSet, public va: CropSet, public aug: Aug, public decay: number, seed = 5) {
    this.net = mlp([SIZE * SIZE, 32, 2], ["relu", "id"], seed);
    this.opt = new Adam(this.net, 0.003);
    this.batcher = new Batcher(tr.imgs.length, 10, seed + 1);
    this.r = rng(seed + 2);
    this.X = tr.imgs.map(g => Array.from(g.data)); this.Y = tr.y.map(onehot);
    this.Xv = va.imgs.map(g => Array.from(g.data)); this.Yv = va.y.map(onehot);
    this.record();
  }
  accuracy(X: number[][], y: number[]): number {
    let ok = 0;
    X.forEach((x, i) => { const o = predict(this.net, x); if ((o[1] > o[0] ? 1 : 0) === y[i]) ok++; });
    return ok / X.length;
  }
  record(): void {
    this.history.push({
      epoch: this.epoch, train: meanLoss(this.net, this.X, this.Y, "ce"), val: meanLoss(this.net, this.Xv, this.Yv, "ce"),
      trainAcc: this.accuracy(this.X, this.tr.y), valAcc: this.accuracy(this.Xv, this.va.y),
    });
  }
  /** One epoch: every training crop once, in batches, each augmented anew. */
  trainEpoch(): void {
    const n = this.X.length, steps = Math.ceil(n / 10);
    for (let s = 0; s < steps; s++) {
      const idx = this.batcher.next();
      const imgs = idx.map(i => (isOff(this.aug) ? this.tr.imgs[i] : augmentCrop(this.tr.imgs[i], this.aug, this.r)));
      this.lastBatch = imgs;
      const Xb = imgs.map(g => Array.from(g.data)), Yb = idx.map(i => this.Y[i]);
      const g = gradients(this.net, Xb, Yb, { loss: "ce", decay: this.decay });
      applyGradients(this.net, g, this.opt);
    }
    this.epoch++;
    this.record();
  }
  best(): EpochPoint { return this.history.reduce((b, p) => (p.val < b.val ? p : b), this.history[0]); }
}

/** The sample data of the experiment (fixed seeds). */
export function experiment(task: Task): { tr: CropSet; va: CropSet } {
  return task === "shape" ? { tr: cropSet("shape", 20, true, 11), va: cropSet("shape", 200, false, 12) }
    : { tr: cropSet("orientation", 20, true, 21), va: cropSet("orientation", 200, false, 22) };
}
