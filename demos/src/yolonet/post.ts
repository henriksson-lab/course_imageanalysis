// Pure post-processing of the YOLO11 head output (no DOM), shared by the yolonet demo and
// demos/test/yolo-wasm-test.ts. It is the TypeScript reference of the same functions in the
// Wasm module (demos/wasm/yolo/src/lib.rs: decode_raw, nms_decoded); the test checks that
// both agree with each other and with the YOLOv11-rs crate's own decoding and NMS.
//
// Layouts (A = anchor points = sum over the scales of (size/stride)^2; scales in the order
// stride 8, 16, 32, each row-major):
//   raw     A x 144: 4 sides x 16 distance-bin logits (left, top, right, bottom), 80 class logits
//   decoded A x 88:  x1, y1, x2, y2 (input pixels), distances l, t, r, b (grid cells), 80 probabilities

export const NC = 80;
export const REG_MAX = 16;
export const NO = 4 * REG_MAX + NC; // 144
export const ND = 8 + NC; // 88
export const STRIDES = [8, 16, 32] as const;

/** The 80 COCO class names, in the checkpoint's order. */
export const COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant " +
  "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella " +
  "handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard " +
  "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli " +
  "carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote " +
  "keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear " +
  "hair_drier toothbrush").split(" ").map((s) => s.replace(/_/g, " "));

export interface Scale { stride: number; g: number; start: number; count: number }

/** The three output grids for a square input of side `size` (a multiple of 32). */
export function scales(size: number): Scale[] {
  let start = 0;
  return STRIDES.map((stride) => {
    const g = size / stride, s = { stride, g, start, count: g * g };
    start += g * g;
    return s;
  });
}

export function anchorCount(size: number): number {
  return scales(size).reduce((a, s) => a + s.count, 0);
}

/** Scale index, grid row i and column j of anchor a. */
export function anchorCell(a: number, size: number): { scale: number; i: number; j: number; stride: number } {
  const sc = scales(size);
  for (let k = 0; k < sc.length; k++) {
    const s = sc[k];
    if (a < s.start + s.count) { const r = a - s.start; return { scale: k, i: Math.floor(r / s.g), j: r % s.g, stride: s.stride }; }
  }
  throw new Error("anchor out of range");
}

/** Softmax of 16 logits. */
export function softmax(v: ArrayLike<number>, off = 0, n = REG_MAX): number[] {
  let m = -Infinity;
  for (let k = 0; k < n; k++) m = Math.max(m, v[off + k]);
  const e: number[] = [];
  let z = 0;
  for (let k = 0; k < n; k++) { const p = Math.exp(v[off + k] - m); e.push(p); z += p; }
  return e.map((p) => p / z);
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Decode raw (A x 144) into A x 88 (see the header). */
export function decode(raw: Float32Array, size: number): Float32Array {
  const n = anchorCount(size);
  if (raw.length !== n * NO) throw new Error(`raw must have ${n} x ${NO} values`);
  const out = new Float32Array(n * ND);
  let a = 0;
  for (const s of scales(size)) {
    for (let i = 0; i < s.g; i++) for (let j = 0; j < s.g; j++, a++) {
      const r = a * NO, o = a * ND;
      const d = [0, 0, 0, 0];
      for (let side = 0; side < 4; side++) {
        const p = softmax(raw, r + side * REG_MAX);
        let e = 0;
        for (let k = 0; k < REG_MAX; k++) e += p[k] * k;
        d[side] = e;
      }
      const cx = j + 0.5, cy = i + 0.5;
      out[o] = (cx - d[0]) * s.stride; out[o + 1] = (cy - d[1]) * s.stride;
      out[o + 2] = (cx + d[2]) * s.stride; out[o + 3] = (cy + d[3]) * s.stride;
      for (let k = 0; k < 4; k++) out[o + 4 + k] = d[k];
      for (let c = 0; c < NC; c++) out[o + 8 + c] = sigmoid(raw[r + 4 * REG_MAX + c]);
    }
  }
  return out;
}

export type Box = [number, number, number, number];

export function iou(a: ArrayLike<number>, b: ArrayLike<number>, ao = 0, bo = 0): number {
  const iw = Math.max(0, Math.min(a[ao + 2], b[bo + 2]) - Math.max(a[ao], b[bo]));
  const ih = Math.max(0, Math.min(a[ao + 3], b[bo + 3]) - Math.max(a[ao + 1], b[bo + 1]));
  const inter = iw * ih;
  const union = (a[ao + 2] - a[ao]) * (a[ao + 3] - a[ao + 1]) + (b[bo + 2] - b[bo]) * (b[bo + 3] - b[bo + 1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** One candidate: an (anchor, class) pair with its probability. */
export interface Candidate { anchor: number; cls: number; score: number }

/** Every (anchor, class) pair with probability > conf, sorted by score (stable: anchor, then class). */
export function candidates(dec: Float32Array, conf: number): Candidate[] {
  const n = dec.length / ND, out: Candidate[] = [];
  for (let a = 0; a < n; a++) for (let c = 0; c < NC; c++) {
    const p = dec[a * ND + 8 + c];
    if (p > conf) out.push({ anchor: a, cls: c, score: p });
  }
  return out.sort((x, y) => y.score - x.score);
}

export interface NmsResult {
  /** Candidates, sorted by score. */
  cands: Candidate[];
  /** For each candidate: -1 kept, -2 not reached (max_det), else the index of the kept candidate that removed it. */
  by: Int32Array;
  /** Indices (into cands) of the kept ones, in order. */
  kept: number[];
}

/** Greedy NMS with the crate's semantics (see lib.rs nms_decoded); also records who removed whom. */
export function nms(dec: Float32Array, conf: number, iouThr: number, maxDet = 300, classAgnostic = false): NmsResult {
  const cands = candidates(dec, conf);
  const by = new Int32Array(cands.length).fill(-3);
  const kept: number[] = [];
  for (let i = 0; i < cands.length; i++) {
    if (by[i] !== -3) continue;
    if (kept.length >= maxDet) { for (let j = i; j < cands.length; j++) if (by[j] === -3) by[j] = -2; break; }
    const ai = cands[i].anchor * ND;
    for (let j = i + 1; j < cands.length; j++) {
      if (by[j] !== -3 || (!classAgnostic && cands[j].cls !== cands[i].cls)) continue;
      if (iou(dec, dec, ai, cands[j].anchor * ND) > iouThr) by[j] = i;
    }
    by[i] = -1;
    kept.push(i);
  }
  return { cands, by, kept };
}

/** Rows of 7 like the Wasm nms(): x1, y1, x2, y2, score, class, anchor. */
export function nmsRows(dec: Float32Array, r: NmsResult): Float32Array {
  const out = new Float32Array(r.kept.length * 7);
  r.kept.forEach((k, n) => {
    const c = r.cands[k], o = c.anchor * ND;
    out.set([dec[o], dec[o + 1], dec[o + 2], dec[o + 3], c.score, c.cls, c.anchor], n * 7);
  });
  return out;
}

/** Best class and its probability for every anchor. */
export function bestClass(dec: Float32Array): { cls: Uint8Array; p: Float32Array } {
  const n = dec.length / ND, cls = new Uint8Array(n), p = new Float32Array(n);
  for (let a = 0; a < n; a++) {
    let b = 0, bp = -1;
    for (let c = 0; c < NC; c++) { const v = dec[a * ND + 8 + c]; if (v > bp) { bp = v; b = c; } }
    cls[a] = b; p[a] = bp;
  }
  return { cls, p };
}

/** Letterbox geometry: the image (w x h) scaled by r and centred in a size x size square. */
export interface Letterbox { size: number; r: number; nw: number; nh: number; x0: number; y0: number }

export function letterbox(w: number, h: number, size: number): Letterbox {
  const r = Math.min(size / w, size / h);
  const nw = Math.max(1, Math.round(w * r)), nh = Math.max(1, Math.round(h * r));
  return { size, r, nw, nh, x0: Math.floor((size - nw) / 2), y0: Math.floor((size - nh) / 2) };
}

/** A box in input pixels -> image pixels. */
export function toImage(b: ArrayLike<number>, lb: Letterbox, off = 0): Box {
  return [(b[off] - lb.x0) / lb.r, (b[off + 1] - lb.y0) / lb.r, (b[off + 2] - lb.x0) / lb.r, (b[off + 3] - lb.y0) / lb.r];
}

/** RGBA pixels of the letterboxed size x size image -> network input, 3 x size x size in 0..1 (CHW). */
export function toInput(rgba: ArrayLike<number>, size: number): Float32Array {
  const n = size * size, x = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) for (let c = 0; c < 3; c++) x[c * n + p] = rgba[4 * p + c] / 255;
  return x;
}
