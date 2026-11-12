// Texture sampling as a GPU does it, on the CPU (no DOM; tested in graphics3d-test.ts):
// a mipmap pyramid of 2×2 box averages, nearest and bilinear lookups with wrap-around
// (texel k has its centre at (k + 0.5) / n), the level from the pixel footprint, and the four
// filters nearest / bilinear / nearest mip level / trilinear.
export interface Mip { n: number; data: Float32Array }   // RGB, n × n, values 0..255

/** Square checkerboard texture: `checks` × `checks` squares, RGBA bytes. */
export function checkerTexture(n: number, checks: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(4 * n * n), s = n / checks;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const on = (Math.floor(x / s) + Math.floor(y / s)) % 2 === 0, q = 4 * (y * n + x);
    out[q] = on ? 235 : 30; out[q + 1] = on ? 235 : 40; out[q + 2] = on ? 225 : 70; out[q + 3] = 255;
  }
  return out;
}

/** Mip pyramid: level 0 = the texture, each next level the 2×2 box average, down to 1 × 1. */
export function buildMips(rgba: ArrayLike<number>, n: number): Mip[] {
  const l0 = new Float32Array(3 * n * n);
  for (let i = 0; i < n * n; i++) for (let c = 0; c < 3; c++) l0[3 * i + c] = rgba[4 * i + c];
  const out: Mip[] = [{ n, data: l0 }];
  while (out[out.length - 1].n > 1) {
    const a = out[out.length - 1], m = a.n / 2, d = new Float32Array(3 * m * m);
    for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) for (let c = 0; c < 3; c++) {
      const g = (xx: number, yy: number) => a.data[3 * (yy * a.n + xx) + c];
      d[3 * (y * m + x) + c] = (g(2 * x, 2 * y) + g(2 * x + 1, 2 * y) + g(2 * x, 2 * y + 1) + g(2 * x + 1, 2 * y + 1)) / 4;
    }
    out.push({ n: m, data: d });
  }
  return out;
}

/** Total texels of a pyramid (for the "a third more memory" check). */
export const mipTexels = (ms: Mip[]): number => ms.reduce((s, m) => s + m.n * m.n, 0);

const wrap = (k: number, n: number) => ((k % n) + n) % n;

export function nearest(m: Mip, u: number, v: number): [number, number, number] {
  const x = wrap(Math.floor(u * m.n), m.n), y = wrap(Math.floor(v * m.n), m.n), q = 3 * (y * m.n + x);
  return [m.data[q], m.data[q + 1], m.data[q + 2]];
}

export function bilinear(m: Mip, u: number, v: number): [number, number, number] {
  const x = u * m.n - 0.5, y = v * m.n - 0.5, x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const out: [number, number, number] = [0, 0, 0];
  for (const [dx, dy, w] of [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]]) {
    const q = 3 * (wrap(y0 + dy, m.n) * m.n + wrap(x0 + dx, m.n));
    for (let c = 0; c < 3; c++) out[c] += w * m.data[q + c];
  }
  return out;
}

/** Mip level from the footprint: the longer of the two screen-step vectors, in level-0 texels. */
export function footprintLevel(dudx: number, dvdx: number, dudy: number, dvdy: number): number {
  const rho = Math.max(Math.hypot(dudx, dvdx), Math.hypot(dudy, dvdy));
  return Math.log2(Math.max(rho, 1e-9));
}

/** Colours used to show the mip level (level 0 first). */
export const MIP_TINT: [number, number, number][] = [[1, 1, 1], [1, 0.45, 0.45], [1, 0.8, 0.3], [0.5, 0.95, 0.45], [0.35, 0.85, 1], [0.6, 0.55, 1], [1, 0.5, 0.95], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]];

/** Sample with one of the four filters; `lam` = log₂ footprint. Returns RGB bytes. */
export function sampleMip(ms: Mip[], u: number, v: number, filter: "nearest" | "bilinear" | "mipnearest" | "trilinear", lam: number, tint = false): [number, number, number] {
  const top = ms.length - 1, L = Math.max(0, Math.min(top, lam));
  let c: [number, number, number], tc: [number, number, number] = [1, 1, 1];
  if (filter === "nearest") c = nearest(ms[0], u, v);
  else if (filter === "bilinear") c = bilinear(ms[0], u, v);
  else if (filter === "mipnearest") { const k = Math.round(L); c = bilinear(ms[k], u, v); tc = MIP_TINT[k]; }
  else {
    const k = Math.floor(L), k1 = Math.min(top, k + 1), f = L - k;
    const a = bilinear(ms[k], u, v), b = bilinear(ms[k1], u, v);
    c = [0, 1, 2].map(i => (1 - f) * a[i] + f * b[i]) as [number, number, number];
    tc = [0, 1, 2].map(i => (1 - f) * MIP_TINT[k][i] + f * MIP_TINT[k1][i]) as [number, number, number];
  }
  if (!tint) return c;
  if (filter === "nearest" || filter === "bilinear") tc = MIP_TINT[0];
  return [c[0] * (0.35 + 0.65 * tc[0]), c[1] * (0.35 + 0.65 * tc[1]), c[2] * (0.35 + 0.65 * tc[2])];
}
