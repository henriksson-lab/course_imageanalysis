// Synthetic high-bit-depth images and integer arithmetic for the window16 demo.
//  - fluorescence(): a 12-bit microscope image (values 0..4095) of a few cells:
//    offset 100 ADU, background, cells with bright nuclei, two saturated spots,
//    shot noise (seeded) and read noise.
//  - ctPhantom(): a CT slice in Hounsfield units (-1024..3071, stored as int16):
//    air, a body of soft tissue with a fat layer, two lungs with vessels, a liver
//    with a faint lesion (+20 HU), a vertebra with a cortical rim.
// Arithmetic: wrap-around (modular, what uint16 does in C or NumPy), saturating
// (clip to the type's range), and float (exact here).
import { rng } from "../lib/ui";

export const N = 256;

export function fluorescence(seed = 3): Uint16Array {
  const r = rng(seed);
  const out = new Uint16Array(N * N);
  const cells = [
    // cx, cy, rx, ry, angle, cytoplasm level, nucleus level
    [70, 80, 46, 34, 0.4, 700, 2600], [175, 70, 40, 30, -0.3, 500, 1900],
    [120, 175, 55, 38, 0.1, 900, 3300], [210, 190, 30, 26, 0.8, 350, 1200],
  ];
  const spots = [[132, 160], [62, 92]];   // tiny bright spots that saturate the ADC
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    let s = 60 + 25 * Math.sin(j / 37) * Math.cos(i / 53);   // background, photons
    for (const [cx, cy, rx, ry, a, cyto, nuc] of cells) {
      const dx = j - cx, dy = i - cy, c = Math.cos(a), sn = Math.sin(a);
      const u = (c * dx + sn * dy) / rx, v = (-sn * dx + c * dy) / ry;
      const d2 = u * u + v * v;
      s += cyto / (1 + Math.exp((Math.sqrt(d2) - 1) * 12));
      s += (nuc - cyto) * Math.exp(-d2 / (2 * 0.16));
    }
    for (const [x, y] of spots) s += 12000 * Math.exp(-((j - x) ** 2 + (i - y) ** 2) / 4);
    // shot noise (variance = signal, in ADU at gain 1) and read noise 3 ADU, offset 100
    const v = 100 + s + Math.sqrt(s) * r.norm() + 3 * r.norm();
    out[i * N + j] = Math.max(0, Math.min(4095, Math.round(v)));
  }
  return out;
}

export function ctPhantom(seed = 4): Int16Array {
  const r = rng(seed);
  const out = new Int16Array(N * N);
  const inEll = (x: number, y: number, cx: number, cy: number, rx: number, ry: number) =>
    ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1;
  const vessels = [[80, 110, 3], [92, 140, 2.5], [70, 150, 2], [176, 112, 3], [166, 138, 2.5], [186, 152, 2]];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = j + 0.5, y = i + 0.5;
    let hu = -1000;                                   // air
    if (inEll(x, y, 128, 132, 118, 92)) hu = -100;    // fat layer
    if (inEll(x, y, 128, 132, 110, 84)) hu = 40;      // soft tissue (muscle)
    if (inEll(x, y, 82, 125, 36, 50)) hu = -850;      // lungs
    if (inEll(x, y, 174, 125, 36, 50)) hu = -850;
    for (const [vx, vy, vr] of vessels) if (inEll(x, y, vx, vy, vr, vr)) hu = 40;
    if (inEll(x, y, 128, 186, 34, 22)) hu = 60;       // liver
    if (inEll(x, y, 140, 184, 7, 6)) hu = 80;         // faint lesion
    if (inEll(x, y, 128, 82, 17, 15)) hu = 1400;      // vertebra: cortical rim
    if (inEll(x, y, 128, 82, 12, 10)) hu = 300;       // and spongy interior
    const noisy = hu + (hu > -900 ? 12 : 5) * r.norm();
    out[i * N + j] = Math.max(-1024, Math.min(3071, Math.round(noisy)));
  }
  return out;
}

export type Arith = "none" | "wrap" | "saturate" | "float";

/** v - b in the given arithmetic for a uint16 image (`float` returns the exact value). */
export function subtractU16(v: number, b: number, mode: Arith): number {
  if (mode === "none") return v;
  if (mode === "wrap") return ((v - b) % 65536 + 65536) % 65536;
  if (mode === "saturate") return Math.max(0, Math.min(65535, v - b));
  return v - b;
}

/** Linear window: lo -> 0, hi -> 255, clipped, rounded to an 8-bit display level. */
export function windowLevel(v: number, lo: number, hi: number): number {
  const t = (v - lo) / (hi - lo);
  return Math.round(255 * Math.min(1, Math.max(0, t)));
}

/** The p-th percentile (0..100) of the values, by sorting a copy. */
export function percentile(xs: ArrayLike<number>, p: number): number {
  const a = Float64Array.from(xs as ArrayLike<number>).sort();
  const k = Math.min(a.length - 1, Math.max(0, Math.round(p / 100 * (a.length - 1))));
  return a[k];
}
