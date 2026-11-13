// Cellpose 3 around the network (no DOM, so node tests can use it): channels, percentile
// normalisation, rescaling to the training cell size, 224 x 224 tiles with 10 % overlap and
// tapered averaging, the size model, and the flow colour wheel. Follows cellpose 3.1
// (transforms.py, core.run_net, models.CellposeModel.eval, plot.dx_to_circ); the network
// itself and the decoding into masks run in Rust/Wasm (demos/wasm/cellpose).

export type ModelKey = "cyto3" | "nuclei";

export interface ModelInfo {
  key: ModelKey;
  label: string;
  /** key in demos/dist/models/index.json */
  file: string;
  /** mean cell diameter of the training set (pixels); images are rescaled to it */
  diamMean: number;
  /** size model (style vector -> diameter), cyto3 only */
  sizeFile?: string;
}

export const MODELS: Record<ModelKey, ModelInfo> = {
  cyto3: { key: "cyto3", label: "cyto3 (cells)", file: "cellpose/cyto3.safetensors", diamMean: 30, sizeFile: "cellpose/size_cyto3.json" },
  nuclei: { key: "nuclei", label: "nuclei", file: "cellpose/nuclei.safetensors", diamMean: 17 },
};

export const BSIZE = 224;
export const TILE_OVERLAP = 0.1;

/** numpy.percentile (linear interpolation) of an unsorted array. */
export function percentile(x: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(x as ArrayLike<number>).sort();
  if (s.length === 0) return NaN;
  const pos = (q / 100) * (s.length - 1);
  const i = Math.floor(pos), f = pos - i;
  return i + 1 < s.length ? s[i] + f * (s[i + 1] - s[i]) : s[i];
}

/** cellpose normalize99: 1st percentile -> 0, 99th -> 1 (constant channels stay as they are,
 *  a range below 1e-3 gives zeros). `invert` maps v -> 1 - v afterwards. */
export function normalize99(x: Float32Array, invert = false): Float32Array {
  const out = new Float32Array(x);
  let mn = Infinity, mx = -Infinity;
  for (const v of x) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (!(mx - mn > 0)) return out; // np.ptp == 0: left unnormalised (and not inverted)
  const lo = percentile(x, 1), hi = percentile(x, 99);
  if (hi - lo > 1e-3) for (let i = 0; i < out.length; i++) out[i] = (out[i] - lo) / (hi - lo);
  else out.fill(0);
  if (invert) for (let i = 0; i < out.length; i++) out[i] = 1 - out[i];
  return out;
}

/** An image as planes of floats (1 or 3 channels), row-major. */
export interface Img { w: number; h: number; planes: Float32Array[] }

/** cellpose transforms.reshape: channels = [segment, nuclei], 0 = grey (mean of all planes) /
 *  none, 1..3 = red, green, blue. Returns the network's two input planes (second may be 0). */
export function toChannels(img: Img, chan: [number, number]): [Float32Array, Float32Array] {
  const n = img.w * img.h;
  const zero = new Float32Array(n);
  if (img.planes.length === 1) return [new Float32Array(img.planes[0]), zero];
  if (chan[0] === 0) {
    const g = new Float32Array(n);
    for (const p of img.planes) for (let i = 0; i < n; i++) g[i] += p[i];
    for (let i = 0; i < n; i++) g[i] /= img.planes.length;
    return [g, zero];
  }
  const pick = (c: number) => new Float32Array(img.planes[Math.min(img.planes.length, c) - 1]);
  return [pick(chan[0]), chan[1] > 0 ? pick(chan[1]) : zero];
}

/** Bilinear resize with pixel-centre alignment (as OpenCV INTER_LINEAR). */
export function resizeBilinear(src: Float32Array, w: number, h: number, W: number, H: number): Float32Array {
  const out = new Float32Array(W * H);
  const sx = w / W, sy = h / H;
  const xi0 = new Int32Array(W), xi1 = new Int32Array(W), xf = new Float32Array(W);
  for (let x = 0; x < W; x++) {
    let fx = (x + 0.5) * sx - 0.5;
    if (fx < 0) fx = 0;
    let i = Math.floor(fx);
    if (i >= w - 1) { i = w - 1; fx = i; }
    xi0[x] = i; xi1[x] = Math.min(w - 1, i + 1); xf[x] = fx - i;
  }
  for (let y = 0; y < H; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    let j = Math.floor(fy);
    if (j >= h - 1) { j = h - 1; fy = j; }
    const j1 = Math.min(h - 1, j + 1), ty = fy - j;
    for (let x = 0; x < W; x++) {
      const a = src[j * w + xi0[x]], b = src[j * w + xi1[x]], c = src[j1 * w + xi0[x]], d = src[j1 * w + xi1[x]];
      const tx = xf[x];
      out[y * W + x] = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
    }
  }
  return out;
}

/** transforms.get_pad_yx: pad to a multiple of 16 plus 8 pixels on every side. */
export function padYX(Ly: number, Lx: number, div = 16, extra = 1): [number, number, number, number] {
  const py = div * Math.ceil(Ly / div) - Ly, px = div * Math.ceil(Lx / div) - Lx;
  const e = Math.floor((extra * div) / 2);
  return [e + Math.floor(py / 2), e + py - Math.floor(py / 2), e + Math.floor(px / 2), e + px - Math.floor(px / 2)];
}

/** Tile starts along one axis (transforms.make_tiles, no augmentation). */
export function tileStarts(L: number, bsize = BSIZE, overlap = TILE_OVERLAP): { starts: number[]; size: number } {
  const ov = Math.min(0.5, Math.max(0.05, overlap));
  const size = Math.min(bsize, L);
  const n = L <= bsize ? 1 : Math.ceil(((1 + 2 * ov) * L) / bsize);
  const starts: number[] = [];
  // np.linspace(0, L - size, n).astype(int): i * step, last element exactly L - size
  const step = n > 1 ? (L - size) / (n - 1) : 0;
  for (let i = 0; i < n; i++) starts.push(i === n - 1 ? L - size : Math.trunc(i * step));
  return { starts, size };
}

/** transforms._taper_mask: weights for averaging overlapping tiles (sigmoid fall-off at the edges). */
export function taperMask(ly: number, lx: number, sig = 7.5): Float32Array {
  const bs = Math.max(224, Math.max(ly, lx));
  const mean = (bs - 1) / 2;
  const m1 = new Float64Array(bs);
  for (let i = 0; i < bs; i++) m1[i] = 1 / (1 + Math.exp((Math.abs(i - mean) - (bs / 2 - 20)) / sig));
  const y0 = Math.floor(bs / 2) - Math.floor(ly / 2), x0 = Math.floor(bs / 2) - Math.floor(lx / 2);
  const out = new Float32Array(ly * lx);
  for (let y = 0; y < ly; y++) for (let x = 0; x < lx; x++) out[y * lx + x] = m1[y0 + y] * m1[x0 + x];
  return out;
}

/** Runs one tile (2, h, w) -> (3, h, w) + style (256); e.g. a Web Worker or the Wasm glue. */
export type TileRunner = (input: Float32Array, h: number, w: number) => Promise<{ y: Float32Array; style: Float32Array }>;

export interface NetOutput {
  w: number; h: number;
  dy: Float32Array; dx: Float32Array; cellprob: Float32Array;
  /** image style vector (sum of tile styles, unit length) */
  style: Float32Array;
  /** network input size after rescaling, tiles used */
  workW: number; workH: number; nTiles: number; tileH: number; tileW: number;
  rescale: number;
}

/** The network part of CellposeModel.eval for one 2D image: rescale by `rescale`
 *  (= diam_mean / diameter), pad, tile, run, average the tiles, crop, resize back. */
export async function runNetwork(ch: [Float32Array, Float32Array], w: number, h: number, rescale: number,
  run: TileRunner, onTile?: (done: number, total: number) => void): Promise<NetOutput> {
  let a = ch[0], b = ch[1], Lyr = h, Lxr = w;
  if (rescale !== 1) {
    Lyr = Math.trunc(h * rescale); Lxr = Math.trunc(w * rescale);
    a = resizeBilinear(a, w, h, Lxr, Lyr); b = resizeBilinear(b, w, h, Lxr, Lyr);
  }
  const [yp1, yp2, xp1, xp2] = padYX(Lyr, Lxr);
  const Ly = Lyr + yp1 + yp2, Lx = Lxr + xp1 + xp2;
  const padded = [new Float32Array(Ly * Lx), new Float32Array(Ly * Lx)];
  for (let y = 0; y < Lyr; y++) {
    padded[0].set(a.subarray(y * Lxr, (y + 1) * Lxr), (y + yp1) * Lx + xp1);
    padded[1].set(b.subarray(y * Lxr, (y + 1) * Lxr), (y + yp1) * Lx + xp1);
  }
  const ty = tileStarts(Ly), tx = tileStarts(Lx);
  const ly = ty.size, lx = tx.size;
  const mask = taperMask(ly, lx);
  const acc = [new Float32Array(Ly * Lx), new Float32Array(Ly * Lx), new Float32Array(Ly * Lx)];
  const nav = new Float32Array(Ly * Lx);
  const style = new Float32Array(256);
  const jobs: Promise<void>[] = [];
  const total = ty.starts.length * tx.starts.length;
  let done = 0;
  onTile?.(0, total);
  for (const y0 of ty.starts) for (const x0 of tx.starts) {
    const inp = new Float32Array(2 * ly * lx);
    for (let c = 0; c < 2; c++) for (let y = 0; y < ly; y++)
      inp.set(padded[c].subarray((y0 + y) * Lx + x0, (y0 + y) * Lx + x0 + lx), c * ly * lx + y * lx);
    jobs.push(run(inp, ly, lx).then(({ y: out, style: st }) => {
      for (let c = 0; c < 3; c++) for (let y = 0; y < ly; y++) for (let x = 0; x < lx; x++) {
        acc[c][(y0 + y) * Lx + x0 + x] += out[c * ly * lx + y * lx + x] * mask[y * lx + x];
      }
      for (let y = 0; y < ly; y++) for (let x = 0; x < lx; x++) nav[(y0 + y) * Lx + x0 + x] += mask[y * lx + x];
      for (let k = 0; k < 256; k++) style[k] += st[k];
      onTile?.(++done, total);
    }));
  }
  await Promise.all(jobs);
  let sn = 0;
  for (const v of style) sn += v * v;
  sn = Math.sqrt(sn);
  for (let k = 0; k < 256; k++) style[k] /= sn;
  const crop = (c: number) => {
    const o = new Float32Array(Lyr * Lxr);
    for (let y = 0; y < Lyr; y++) for (let x = 0; x < Lxr; x++) {
      const i = (y + yp1) * Lx + x + xp1;
      o[y * Lxr + x] = acc[c][i] / nav[i];
    }
    return rescale !== 1 ? resizeBilinear(o, Lxr, Lyr, w, h) : o;
  };
  return {
    w, h, dy: crop(0), dx: crop(1), cellprob: crop(2), style,
    workW: Lxr, workH: Lyr, nTiles: total, tileH: ly, tileW: lx, rescale,
  };
}

/** Euler steps used by Cellpose 3 at the original image size: 200 / rescale (truncated). */
export function defaultNiter(rescale: number): number {
  return Math.trunc(200 / rescale);
}

/** Size model of cyto3 (models.SizeModel._size_estimation): diameter from the style vector. */
export interface SizeModel { A: number[]; smean: number[]; diam_mean: number; ymean: number }
export function diameterFromStyle(style: ArrayLike<number>, sm: SizeModel): number {
  let s = 0;
  for (let k = 0; k < sm.A.length; k++) s += sm.A[k] * (style[k] - sm.smean[k]);
  return Math.max(5, Math.exp(s + Math.log(sm.diam_mean) + sm.ymean));
}

/** utils.diameters: median over masks of sqrt(area) / (sqrt(pi) / 2). */
export function medianDiameter(masks: ArrayLike<number>): number {
  const counts = new Map<number, number>();
  for (let i = 0; i < masks.length; i++) if (masks[i] > 0) counts.set(masks[i], (counts.get(masks[i]) ?? 0) + 1);
  if (counts.size === 0) return 0;
  const r = Array.from(counts.values(), (c) => Math.sqrt(c)).sort((p, q) => p - q);
  const m = r.length % 2 ? r[(r.length - 1) / 2] : (r[r.length / 2 - 1] + r[r.length / 2]) / 2;
  return m / (Math.sqrt(Math.PI) / 2);
}

/** plot.dx_to_circ: flow direction as hue, magnitude (99th percentile = full) as brightness. */
export function flowToRGBA(dy: Float32Array, dx: Float32Array, out?: Uint8ClampedArray): Uint8ClampedArray {
  const n = dy.length;
  const o = out ?? new Uint8ClampedArray(4 * n);
  const mag = new Float32Array(n);
  for (let i = 0; i < n; i++) mag[i] = Math.hypot(dy[i], dx[i]);
  const m = normalize99(mag);
  for (let i = 0; i < n; i++) {
    const v = (255 * Math.min(1, Math.max(0, m[i]))) / 2;
    const ang = Math.atan2(dx[i], dy[i]) + Math.PI;
    o[4 * i] = v * (Math.cos(ang) + 1);
    o[4 * i + 1] = v * (Math.cos(ang + (2 * Math.PI) / 3) + 1);
    o[4 * i + 2] = v * (Math.cos(ang + (4 * Math.PI) / 3) + 1);
    o[4 * i + 3] = 255;
  }
  return o;
}

/** Pixels on the border of a label (4-neighbour differs), for outlines. */
export function outlines(lab: ArrayLike<number>, w: number, h: number): Uint8Array {
  const o = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, v = lab[i];
    if (!v) continue;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || lab[i - 1] !== v || lab[i + 1] !== v || lab[i - w] !== v || lab[i + w] !== v) o[i] = 1;
  }
  return o;
}
