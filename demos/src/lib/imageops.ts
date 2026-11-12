// Grey-level images and linear filters for the demos.
//
// A `Grey` is a width × height image of Float32 values, row-major (index y*width + x,
// row 0 at the top, y pointing down). Values are whatever the demo needs: 0..255 grey
// levels, 12-bit counts, signed filter responses. Pixel centres sit at integer
// coordinates, so pixel (0,0) covers [-0.5, 0.5]².
//
// Filters are written as CORRELATION (the weight at offset (i,j) multiplies the pixel at
// (x+i, y+j)), which is what image software calls "filtering"; for symmetric kernels it
// equals convolution. `convolve2d` flips the kernel when the true convolution is meant.
//
// No DOM except the display/loading section at the end (marked). Synthetic test images are
// in lib/synth.ts; the course micrograph is loaded by `loadCell()` in lib/cell.ts.

export interface Grey {
  width: number;
  height: number;
  data: Float32Array;
}

/** How a filter reads pixels outside the image.
 *  zero: 0 · replicate: nearest edge pixel (a a a | a b c) · mirror: reflection about the
 *  edge pixel, without repeating it (c b | a b c) · wrap: periodic (b c | a b c), as the FFT assumes. */
export type Border = "zero" | "replicate" | "mirror" | "wrap";

// ---------------------------------------------------------------------------------------
// Construction and pixel-wise helpers

/** A new image filled with `fill`. */
export function createGrey(width: number, height: number, fill = 0): Grey {
  const data = new Float32Array(width * height);
  if (fill !== 0) data.fill(fill);
  return { width, height, data };
}

/** Wrap (a copy of) any numeric array as a Grey. */
export function fromArray(width: number, height: number, a: ArrayLike<number>): Grey {
  if (a.length !== width * height) throw new Error(`fromArray: ${a.length} values for ${width}×${height}`);
  return { width, height, data: Float32Array.from(a) };
}

/** An image computed from f(x, y). */
export function fromFunction(width: number, height: number, f: (x: number, y: number) => number): Grey {
  const g = createGrey(width, height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) g.data[y * width + x] = f(x, y);
  return g;
}

export function cloneGrey(img: Grey): Grey {
  return { width: img.width, height: img.height, data: img.data.slice() };
}

/** Apply f to every pixel value (new image). */
export function mapGrey(img: Grey, f: (v: number, i: number) => number): Grey {
  const out = createGrey(img.width, img.height);
  for (let i = 0; i < img.data.length; i++) out.data[i] = f(img.data[i], i);
  return out;
}

/** Combine two images of the same size pixel by pixel, e.g. combine(a, b, (p, q) => p - q). */
export function combine(a: Grey, b: Grey, f: (p: number, q: number) => number): Grey {
  sameSize(a, b, "combine");
  const out = createGrey(a.width, a.height);
  for (let i = 0; i < a.data.length; i++) out.data[i] = f(a.data[i], b.data[i]);
  return out;
}

/** The w × h window with top-left corner (x0, y0); outside pixels follow `border`. */
export function cropGrey(img: Grey, x0: number, y0: number, w: number, h: number, border: Border = "zero"): Grey {
  const out = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out.data[y * w + x] = pixel(img, x0 + x, y0 + y, border);
  return out;
}

/** Minimum, maximum, mean and standard deviation of the pixel values. */
export function stats(img: Grey | ArrayLike<number>): { min: number; max: number; mean: number; sd: number } {
  const d = "data" in img ? img.data : img;
  let min = Infinity, max = -Infinity, s = 0, s2 = 0;
  for (let i = 0; i < d.length; i++) {
    const v = d[i];
    if (v < min) min = v;
    if (v > max) max = v;
    s += v;
  }
  const mean = s / d.length;
  for (let i = 0; i < d.length; i++) s2 += (d[i] - mean) ** 2;
  return { min, max, mean, sd: Math.sqrt(s2 / d.length) };
}

/** Linear stretch so the minimum becomes lo and the maximum hi (constant image → lo). */
export function rescale(img: Grey, lo = 0, hi = 255): Grey {
  const { min, max } = stats(img);
  const k = max > min ? (hi - lo) / (max - min) : 0;
  return mapGrey(img, v => lo + (v - min) * k);
}

function sameSize(a: Grey, b: Grey, who: string): void {
  if (a.width !== b.width || a.height !== b.height) throw new Error(`${who}: sizes differ`);
}

// ---------------------------------------------------------------------------------------
// Borders and sampling

/** Index 0..n-1 used for position i under the border rule, or -1 for "zero" outside. */
export function borderIndex(i: number, n: number, border: Border): number {
  if (i >= 0 && i < n) return i;
  switch (border) {
    case "zero": return -1;
    case "replicate": return i < 0 ? 0 : n - 1;
    case "wrap": return ((i % n) + n) % n;
    case "mirror": {
      if (n === 1) return 0;
      const p = 2 * n - 2;
      let m = ((i % p) + p) % p;
      if (m >= n) m = p - m;
      return m;
    }
  }
}

/** Pixel value at integer (x, y), with the border rule outside the image. */
export function pixel(img: Grey, x: number, y: number, border: Border = "replicate"): number {
  const xi = borderIndex(x, img.width, border), yi = borderIndex(y, img.height, border);
  return xi < 0 || yi < 0 ? 0 : img.data[yi * img.width + xi];
}

/** Nearest-neighbour value at real (x, y) (pixel centres at integers). */
export function sampleNearest(img: Grey, x: number, y: number, border: Border = "replicate"): number {
  return pixel(img, Math.round(x), Math.round(y), border);
}

/** Bilinear interpolation at real (x, y) (pixel centres at integers). */
export function sampleBilinear(img: Grey, x: number, y: number, border: Border = "replicate"): number {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const a = pixel(img, x0, y0, border), b = pixel(img, x0 + 1, y0, border);
  const c = pixel(img, x0, y0 + 1, border), d = pixel(img, x0 + 1, y0 + 1, border);
  return (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d);
}

export type Interp = "nearest" | "bilinear";

/**
 * Inverse-mapping warp: output pixel (x, y) takes the input value at map(x, y).
 * The general tool behind resize, rotate and any affine or non-rigid transform.
 */
export function warp(img: Grey, outW: number, outH: number, map: (x: number, y: number) => [number, number],
  interp: Interp = "bilinear", border: Border = "zero"): Grey {
  const out = createGrey(outW, outH);
  const s = interp === "nearest" ? sampleNearest : sampleBilinear;
  for (let y = 0; y < outH; y++) for (let x = 0; x < outW; x++) {
    const [sx, sy] = map(x, y);
    out.data[y * outW + x] = s(img, sx, sy, border);
  }
  return out;
}

/**
 * Resample to w × h (pixel areas aligned, so the image keeps its extent). "area" averages
 * the input pixels under each output pixel (use it to shrink without aliasing; exact for
 * integer factors); "nearest"/"bilinear" sample at the output pixel centres.
 */
export function resize(img: Grey, w: number, h: number, interp: Interp | "area" = "bilinear"): Grey {
  const sx = img.width / w, sy = img.height / h;
  if (interp === "area") {
    const out = createGrey(w, h);
    for (let y = 0; y < h; y++) {
      const ya = Math.floor(y * sy), yb = Math.max(ya + 1, Math.ceil((y + 1) * sy - 1e-9));
      for (let x = 0; x < w; x++) {
        const xa = Math.floor(x * sx), xb = Math.max(xa + 1, Math.ceil((x + 1) * sx - 1e-9));
        let s = 0, n = 0;
        for (let v = ya; v < Math.min(yb, img.height); v++)
          for (let u = xa; u < Math.min(xb, img.width); u++) { s += img.data[v * img.width + u]; n++; }
        out.data[y * w + x] = n ? s / n : 0;
      }
    }
    return out;
  }
  return warp(img, w, h, (x, y) => [(x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5], interp, "replicate");
}

/**
 * Rotate about (cx, cy) (default: the image centre) by `angle` radians; positive turns the
 * content counter-clockwise as seen on screen. Same output size; outside → `border`.
 */
export function rotate(img: Grey, angle: number, opts: { cx?: number; cy?: number; interp?: Interp; border?: Border } = {}): Grey {
  const cx = opts.cx ?? (img.width - 1) / 2, cy = opts.cy ?? (img.height - 1) / 2;
  const c = Math.cos(angle), s = Math.sin(angle);
  return warp(img, img.width, img.height, (x, y) => {
    const dx = x - cx, dy = y - cy;
    return [cx + c * dx - s * dy, cy + s * dx + c * dy];
  }, opts.interp ?? "bilinear", opts.border ?? "zero");
}

/** Shift by (dx, dy) pixels (sub-pixel allowed): out(x, y) = in(x - dx, y - dy). */
export function translate(img: Grey, dx: number, dy: number, interp: Interp = "bilinear", border: Border = "zero"): Grey {
  return warp(img, img.width, img.height, (x, y) => [x - dx, y - dy], interp, border);
}

// ---------------------------------------------------------------------------------------
// Convolution

/**
 * Correlate with a kw × kh kernel (row-major, odd sizes, origin at the centre):
 * out(x, y) = Σ_{i,j} k[j][i] · in(x + i - cx, y + j - cy).
 */
export function correlate2d(img: Grey, k: ArrayLike<number>, kw: number, kh: number, border: Border = "replicate"): Grey {
  const { width: w, height: h, data } = img, out = createGrey(w, h);
  const cx = kw >> 1, cy = kh >> 1;
  for (let y = 0; y < h; y++) {
    const inside = y - cy >= 0 && y - cy + kh <= h;
    for (let x = 0; x < w; x++) {
      let s = 0;
      if (inside && x - cx >= 0 && x - cx + kw <= w) {
        for (let j = 0; j < kh; j++) {
          const row = (y + j - cy) * w + x - cx, kr = j * kw;
          for (let i = 0; i < kw; i++) s += k[kr + i] * data[row + i];
        }
      } else {
        for (let j = 0; j < kh; j++) {
          const yy = borderIndex(y + j - cy, h, border);
          if (yy < 0) continue;
          for (let i = 0; i < kw; i++) {
            const xx = borderIndex(x + i - cx, w, border);
            if (xx >= 0) s += k[j * kw + i] * data[yy * w + xx];
          }
        }
      }
      out.data[y * w + x] = s;
    }
  }
  return out;
}

/** True convolution: the kernel is flipped in both directions, then correlated. */
export function convolve2d(img: Grey, k: ArrayLike<number>, kw: number, kh: number, border: Border = "replicate"): Grey {
  const f = new Float64Array(kw * kh);
  for (let j = 0; j < kh; j++) for (let i = 0; i < kw; i++) f[j * kw + i] = k[(kh - 1 - j) * kw + (kw - 1 - i)];
  return correlate2d(img, f, kw, kh, border);
}

/** Correlate every row with the 1D kernel kx (odd length, centred). */
export function correlateRows(img: Grey, kx: ArrayLike<number>, border: Border = "replicate"): Grey {
  const { width: w, height: h, data } = img, out = createGrey(w, h), r = kx.length >> 1, n = kx.length;
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      if (x - r >= 0 && x + r < w) for (let i = 0; i < n; i++) s += kx[i] * data[o + x - r + i];
      else for (let i = 0; i < n; i++) { const xx = borderIndex(x - r + i, w, border); if (xx >= 0) s += kx[i] * data[o + xx]; }
      out.data[o + x] = s;
    }
  }
  return out;
}

/** Correlate every column with the 1D kernel ky (odd length, centred). */
export function correlateCols(img: Grey, ky: ArrayLike<number>, border: Border = "replicate"): Grey {
  const { width: w, height: h, data } = img, out = createGrey(w, h), r = ky.length >> 1, n = ky.length;
  for (let y = 0; y < h; y++) {
    const inside = y - r >= 0 && y + r < h;
    for (let x = 0; x < w; x++) {
      let s = 0;
      if (inside) for (let j = 0; j < n; j++) s += ky[j] * data[(y - r + j) * w + x];
      else for (let j = 0; j < n; j++) { const yy = borderIndex(y - r + j, h, border); if (yy >= 0) s += ky[j] * data[yy * w + x]; }
      out.data[y * w + x] = s;
    }
  }
  return out;
}

/** Separable filter: rows with kx, then columns with ky (= correlate2d with the outer product ky·kxᵀ). */
export function separable(img: Grey, kx: ArrayLike<number>, ky: ArrayLike<number>, border: Border = "replicate"): Grey {
  return correlateCols(correlateRows(img, kx, border), ky, border);
}

// ---------------------------------------------------------------------------------------
// Gaussian filters

/**
 * Sampled 1D Gaussian kernel (as a CORRELATION kernel) of the given derivative order,
 * on offsets -radius..radius (default radius ceil(4σ), at least 1).
 *  order 0: G, normalised to sum 1;
 *  order 1: -G' = (i/σ²) G, scaled so a ramp f(x) = x gives exactly 1 (d/dx);
 *  order 2: G'' = (i²/σ⁴ - 1/σ²) G, mean removed, scaled so f(x) = x² gives exactly 2.
 */
export function gaussianKernel(sigma: number, order: 0 | 1 | 2 = 0, radius?: number): Float64Array {
  const r = radius ?? Math.max(1, Math.ceil(4 * sigma));
  const k = new Float64Array(2 * r + 1);
  for (let i = -r; i <= r; i++) {
    const g = Math.exp(-i * i / (2 * sigma * sigma));
    k[i + r] = order === 0 ? g : order === 1 ? i * g : (i * i / (sigma * sigma) - 1) * g;
  }
  if (order === 0) {
    let s = 0; for (const v of k) s += v;
    for (let i = 0; i < k.length; i++) k[i] /= s;
  } else if (order === 1) {
    let s = 0; for (let i = -r; i <= r; i++) s += k[i + r] * i;
    for (let i = 0; i < k.length; i++) k[i] /= s;
  } else {
    let m = 0; for (const v of k) m += v;
    m /= k.length;
    let s = 0;
    for (let i = -r; i <= r; i++) { k[i + r] -= m; s += k[i + r] * i * i; }
    for (let i = 0; i < k.length; i++) k[i] *= 2 / s;
  }
  return k;
}

const IDENTITY = new Float64Array([1]);

/** Gaussian blur with standard deviation σ pixels (σ ≤ 0 returns a copy). */
export function gaussianBlur(img: Grey, sigma: number, border: Border = "replicate"): Grey {
  if (!(sigma > 0)) return cloneGrey(img);
  const k = gaussianKernel(sigma, 0);
  return separable(img, k, k, border);
}

/**
 * Gaussian derivative ∂^(ox+oy) (G_σ * I) / ∂x^ox ∂y^oy, with ox, oy ∈ {0,1,2} and
 * ox + oy ≤ 2 (x to the right, y down). E.g. (1,0) = Lx, (0,2) = Lyy, (1,1) = Lxy.
 */
export function gaussianDerivative(img: Grey, sigma: number, ox: 0 | 1 | 2, oy: 0 | 1 | 2, border: Border = "replicate"): Grey {
  const kx = sigma > 0 ? gaussianKernel(sigma, ox) : ox === 0 ? IDENTITY : ox === 1 ? new Float64Array([-0.5, 0, 0.5]) : new Float64Array([1, -2, 1]);
  const ky = sigma > 0 ? gaussianKernel(sigma, oy) : oy === 0 ? IDENTITY : oy === 1 ? new Float64Array([-0.5, 0, 0.5]) : new Float64Array([1, -2, 1]);
  return separable(img, kx, ky, border);
}

/** Gaussian gradient at scale σ: components, magnitude and direction atan2(gy, gx). */
export function gradient(img: Grey, sigma: number, border: Border = "replicate"): { gx: Grey; gy: Grey; mag: Grey; angle: Grey } {
  const gx = gaussianDerivative(img, sigma, 1, 0, border), gy = gaussianDerivative(img, sigma, 0, 1, border);
  return { gx, gy, mag: combine(gx, gy, Math.hypot), angle: combine(gx, gy, (a, b) => Math.atan2(b, a)) };
}

/**
 * Laplacian of Gaussian Lxx + Lyy at scale σ. With `normalised`, multiplied by σ² so
 * responses can be compared across scales (a blob of radius r peaks near σ = r/√2).
 */
export function logFilter(img: Grey, sigma: number, normalised = false, border: Border = "replicate"): Grey {
  const xx = gaussianDerivative(img, sigma, 2, 0, border), yy = gaussianDerivative(img, sigma, 0, 2, border);
  const s = normalised ? sigma * sigma : 1;
  return combine(xx, yy, (a, b) => s * (a + b));
}

/** Difference of Gaussians G_{kσ} * I − G_σ * I  (≈ (k−1)σ² · LoG). */
export function dogFilter(img: Grey, sigma: number, k = 1.6, border: Border = "replicate"): Grey {
  return combine(gaussianBlur(img, k * sigma, border), gaussianBlur(img, sigma, border), (a, b) => a - b);
}

// ---------------------------------------------------------------------------------------
// Small classic kernels

/** Mean over the (2r+1) × (2r+1) square around each pixel (running sums, any r). */
export function boxFilter(img: Grey, r: number, border: Border = "replicate"): Grey {
  if (r <= 0) return cloneGrey(img);
  const pass = (src: Float32Array, w: number, h: number, alongX: boolean): Float32Array => {
    const out = new Float32Array(w * h), n = alongX ? w : h, m = alongX ? h : w, inv = 1 / (2 * r + 1);
    const at = (line: number, i: number) => {
      const j = borderIndex(i, n, border);
      return j < 0 ? 0 : alongX ? src[line * w + j] : src[j * w + line];
    };
    for (let line = 0; line < m; line++) {
      let s = 0;
      for (let i = -r; i <= r; i++) s += at(line, i);
      for (let i = 0; i < n; i++) {
        if (alongX) out[line * w + i] = s * inv; else out[i * w + line] = s * inv;
        s += at(line, i + r + 1) - at(line, i - r);
      }
    }
    return out;
  };
  const a = pass(img.data, img.width, img.height, true);
  return { width: img.width, height: img.height, data: pass(a, img.width, img.height, false) };
}

export const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
export const SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

/** Sobel derivatives (x right, y down), unnormalised as on the slides (divide by 8 for per-pixel slope). */
export function sobel(img: Grey, border: Border = "replicate"): { gx: Grey; gy: Grey; mag: Grey } {
  const gx = correlate2d(img, SOBEL_X, 3, 3, border), gy = correlate2d(img, SOBEL_Y, 3, 3, border);
  return { gx, gy, mag: combine(gx, gy, Math.hypot) };
}

/** 3 × 3 Laplacian: 4-neighbour [0 1 0; 1 −4 1; 0 1 0], or 8-neighbour [1 1 1; 1 −8 1; 1 1 1]. */
export function laplacian(img: Grey, border: Border = "replicate", eight = false): Grey {
  return correlate2d(img, eight ? [1, 1, 1, 1, -8, 1, 1, 1, 1] : [0, 1, 0, 1, -4, 1, 0, 1, 0], 3, 3, border);
}

// ---------------------------------------------------------------------------------------
// Pyramids

/** Blur with σ (default 1) and keep every second pixel: size ceil(w/2) × ceil(h/2). */
export function pyrDown(img: Grey, sigma = 1, border: Border = "mirror"): Grey {
  const b = gaussianBlur(img, sigma, border);
  const w = Math.ceil(img.width / 2), h = Math.ceil(img.height / 2), out = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out.data[y * w + x] = b.data[2 * y * img.width + 2 * x];
  return out;
}

/** Bilinear upsampling to w × h, the inverse geometry of pyrDown (small pixel i sits at 2i). */
export function pyrUp(img: Grey, w: number, h: number): Grey {
  return warp(img, w, h, (x, y) => [x / 2, y / 2], "bilinear", "replicate");
}

/** [original, pyrDown(original), ...]: `levels` images in total, stopping at 1 × 1. */
export function gaussianPyramid(img: Grey, levels: number, sigma = 1): Grey[] {
  const out = [img];
  while (out.length < levels && (out[out.length - 1].width > 1 || out[out.length - 1].height > 1))
    out.push(pyrDown(out[out.length - 1], sigma));
  return out;
}

/** Laplacian pyramid: band i = G_i − pyrUp(G_{i+1}); the last entry is the coarsest Gaussian level. */
export function laplacianPyramid(img: Grey, levels: number, sigma = 1): Grey[] {
  const g = gaussianPyramid(img, levels, sigma);
  return g.map((gi, i) => i === g.length - 1 ? cloneGrey(gi)
    : combine(gi, pyrUp(g[i + 1], gi.width, gi.height), (a, b) => a - b));
}

/** Rebuild the image from a Laplacian pyramid (exact inverse of laplacianPyramid). */
export function collapsePyramid(lap: Grey[]): Grey {
  let cur = lap[lap.length - 1];
  for (let i = lap.length - 2; i >= 0; i--)
    cur = combine(lap[i], pyrUp(cur, lap[i].width, lap[i].height), (a, b) => a + b);
  return cur;
}

/** 2 × 2 mean downsampling (a mipmap level): size ceil(w/2) × ceil(h/2), edge pixels replicated. */
export function boxDown2(img: Grey): Grey {
  const w = Math.ceil(img.width / 2), h = Math.ceil(img.height / 2), out = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    out.data[y * w + x] = (pixel(img, 2 * x, 2 * y) + pixel(img, 2 * x + 1, 2 * y)
      + pixel(img, 2 * x, 2 * y + 1) + pixel(img, 2 * x + 1, 2 * y + 1)) / 4;
  return out;
}

// ---------------------------------------------------------------------------------------
// Display (bytes: no DOM; ImageData/canvas: DOM)

/**
 * How values become grey levels 0..255:
 *  clip: [lo, hi] (default 0..255) linearly to 0..255, values outside clipped;
 *  auto: like clip with lo/hi = the image minimum/maximum (contrast stretch);
 *  abs:  |v| / hi, hi defaulting to max |v| (magnitudes: black = 0);
 *  mid:  0 → mid-grey, ±hi → white/black, hi defaulting to max |v| (signed responses).
 */
export type DisplayMode = "clip" | "auto" | "abs" | "mid";
export interface DisplayOptions { mode?: DisplayMode; lo?: number; hi?: number }

/** The value range actually mapped to black..white for these options (for colour bars and captions). */
export function displayRange(img: Grey, opts: DisplayOptions = {}): { lo: number; hi: number } {
  const mode = opts.mode ?? "clip";
  if (mode === "clip") return { lo: opts.lo ?? 0, hi: opts.hi ?? 255 };
  const st = stats(img);
  if (mode === "auto") return { lo: opts.lo ?? st.min, hi: opts.hi ?? st.max };
  const a = opts.hi ?? Math.max(Math.abs(st.min), Math.abs(st.max));
  return mode === "abs" ? { lo: 0, hi: a } : { lo: -a, hi: a };
}

/** One grey level 0..255 per pixel. */
export function toBytes(img: Grey, opts: DisplayOptions = {}): Uint8ClampedArray {
  const { lo, hi } = displayRange(img, opts), abs = opts.mode === "abs";
  const k = hi > lo ? 255 / (hi - lo) : 0;
  const out = new Uint8ClampedArray(img.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = abs ? Math.abs(img.data[i]) : img.data[i];
    out[i] = Math.round((v - lo) * k);   // Uint8ClampedArray clips to 0..255
  }
  return out;
}

/**
 * RGBA pixels for an ImageData. Optional `lut`: a colour map as 256 RGB triples
 * (768 bytes, e.g. from lib/colour.ts); without it the image is grey.
 */
export function toRGBA(img: Grey, opts: DisplayOptions = {}, lut?: ArrayLike<number>): Uint8ClampedArray<ArrayBuffer> {
  const g = toBytes(img, opts), out = new Uint8ClampedArray(4 * g.length);
  // LUT stride: 256 RGB entries (768 values) or 256 RGBA entries (1024, as lib/colour.ts colormapLUT returns).
  const st = lut && lut.length >= 1024 ? 4 : 3;
  for (let i = 0; i < g.length; i++) {
    const v = g[i];
    if (lut) { out[4 * i] = lut[st * v]; out[4 * i + 1] = lut[st * v + 1]; out[4 * i + 2] = lut[st * v + 2]; }
    else out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = v;
    out[4 * i + 3] = 255;
  }
  return out;
}

/** DOM: an ImageData of the image. */
export function toImageData(img: Grey, opts: DisplayOptions = {}, lut?: ArrayLike<number>): ImageData {
  return new ImageData(toRGBA(img, opts, lut), img.width, img.height);
}

/** DOM: draw into a canvas, resizing it to the image (scale up with CSS; it stays pixelated). */
export function drawGrey(canvas: HTMLCanvasElement, img: Grey, opts: DisplayOptions = {}, lut?: ArrayLike<number>): void {
  if (canvas.width !== img.width) canvas.width = img.width;
  if (canvas.height !== img.height) canvas.height = img.height;
  canvas.style.imageRendering = "pixelated";
  canvas.getContext("2d")!.putImageData(toImageData(img, opts, lut), 0, 0);
}

/** DOM: a new canvas showing the image. */
export function greyCanvas(img: Grey, opts: DisplayOptions = {}, lut?: ArrayLike<number>): HTMLCanvasElement {
  const c = document.createElement("canvas");
  drawGrey(c, img, opts, lut);
  return c;
}

/** Grey from RGBA pixels (ImageData-like): Rec. 709 luma of the encoded values, 0..255. */
export function fromImageData(id: { width: number; height: number; data: ArrayLike<number> }): Grey {
  const g = createGrey(id.width, id.height), d = id.data;
  for (let i = 0; i < g.data.length; i++) g.data[i] = 0.2126 * d[4 * i] + 0.7152 * d[4 * i + 1] + 0.0722 * d[4 * i + 2];
  return g;
}

/**
 * DOM: decode an image URL (usually a data URL inlined by esbuild, so it works from
 * file://) into a Grey 0..255. With `size`, the result is shrunk by area averaging
 * (or enlarged bilinearly) to size × size, or to {width, height}.
 */
export function loadGrey(src: string, size?: number | { width: number; height: number }): Promise<Grey> {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => {
      const c = document.createElement("canvas");
      c.width = im.naturalWidth; c.height = im.naturalHeight;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(im, 0, 0);
      let g = fromImageData(ctx.getImageData(0, 0, c.width, c.height));
      if (size !== undefined) {
        const w = typeof size === "number" ? size : size.width, h = typeof size === "number" ? size : size.height;
        if (w !== g.width || h !== g.height) g = resize(g, w, h, w <= g.width && h <= g.height ? "area" : "bilinear");
      }
      resolve(g);
    };
    im.onerror = reject;
    im.src = src;
  });
}
