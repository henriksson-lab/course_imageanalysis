// Synthetic 128 × 128 images for the embedprobe demo and its offline training (no DOM).
// Nuclei of very different brightness (the class to find) among things that are NOT nuclei
// but can be as bright: large soft cytoplasm-like regions, small debris spots and thin fibres,
// on an uneven textured background.
// Ground truth: the nucleus mask. Seeded.
import { rasteriseEllipses, type Ellipse } from "../lib/synth";
import { createGrey, gaussianBlur, type Grey } from "../lib/imageops";
import { rng } from "../lib/ui";

export const W = 128;

export interface ProbeScene { image: Float32Array; truth: Uint8Array; ellipses: Ellipse[] }

export function probeScene(seed: number): ProbeScene {
  const r = rng(seed), n = W * W;
  const ellipses: Ellipse[] = [];
  const count = 8 + Math.floor(r.unif() * 5);
  for (let tries = 0; ellipses.length < count && tries < 500; tries++) {
    const m = 6 + 5 * r.unif(), q = 1 + 0.5 * r.unif();
    const e: Ellipse = { x: 10 + (W - 20) * r.unif(), y: 10 + (W - 20) * r.unif(), a: m * Math.sqrt(q), b: m / Math.sqrt(q), angle: Math.PI * r.unif() };
    if (ellipses.every(o => Math.hypot(o.x - e.x, o.y - e.y) > o.a + e.a + 3)) ellipses.push(e);
  }
  const lab = rasteriseEllipses(W, W, ellipses);
  const truth = Uint8Array.from(lab, v => (v > 0 ? 1 : 0));
  // background: illumination slope + fine texture
  const tex = createGrey(W, W); for (let i = 0; i < n; i++) tex.data[i] = r.norm();
  const t1 = gaussianBlur(tex, 1.5);
  const gx = 30 * (r.unif() - 0.5), gy = 30 * (r.unif() - 0.5), bg = 35 + 15 * r.unif();
  const img = createGrey(W, W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    img.data[i] = bg + gx * (x / W - 0.5) + gy * (y / W - 0.5) + 28 * t1.data[i];
  }
  // cytoplasm-like regions: large, smooth, soft-edged, as bright as nuclei (so brightness alone cannot tell)
  const soft = createGrey(W, W);
  const ns = 2 + Math.floor(r.unif() * 3);
  for (let k = 0; k < ns; k++) {
    const cx = W * r.unif(), cy = W * r.unif(), ra = 16 + 14 * r.unif(), rb = 12 + 10 * r.unif(), ang = Math.PI * r.unif(), lev = 60 + 140 * r.unif();
    const c = Math.cos(ang), s = Math.sin(ang);
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const u = c * (x - cx) + s * (y - cy), v = -s * (x - cx) + c * (y - cy);
      if ((u / ra) ** 2 + (v / rb) ** 2 < 1) soft.data[y * W + x] = Math.max(soft.data[y * W + x], lev);
    }
  }
  const softB = gaussianBlur(soft, 5);
  for (let i = 0; i < n; i++) img.data[i] = Math.max(img.data[i], softB.data[i] + 10 * t1.data[i]);
  // fibres: thin lines, medium to bright
  const nf = 1 + Math.floor(r.unif() * 3);
  for (let f = 0; f < nf; f++) {
    const x0 = W * r.unif(), y0 = W * r.unif(), ang = Math.PI * r.unif(), lev = 90 + 90 * r.unif();
    const c = Math.cos(ang), s = Math.sin(ang);
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const d = Math.abs(-(x - x0) * s + (y - y0) * c);
      if (d < 1.1 && !truth[y * W + x]) img.data[y * W + x] = Math.max(img.data[y * W + x], lev);
    }
  }
  // nuclei: brightness anywhere from dim to bright, smooth inner texture
  const t2 = gaussianBlur(tex, 2.5);
  const level = ellipses.map(() => 75 + 160 * r.unif());
  for (let i = 0; i < n; i++) if (lab[i]) img.data[i] = level[lab[i] - 1] * (1 + 0.6 * t2.data[i]);
  // debris: small bright spots (not nuclei)
  const nd = 6 + Math.floor(r.unif() * 6);
  for (let d = 0; d < nd; d++) {
    const cx = W * r.unif(), cy = W * r.unif(), rad = 1.5 + r.unif(), lev = 140 + 110 * r.unif();
    for (let y = Math.floor(cy - 4); y <= cy + 4; y++) for (let x = Math.floor(cx - 4); x <= cx + 4; x++) {
      if (x < 0 || y < 0 || x >= W || y >= W || truth[y * W + x]) continue;
      if (Math.hypot(x - cx, y - cy) < rad) img.data[y * W + x] = lev;
    }
  }
  const out: Grey = gaussianBlur(img, 1);
  for (let i = 0; i < n; i++) out.data[i] = Math.max(0, Math.min(255, out.data[i] + 5 * r.norm()));
  return { image: Float32Array.from(out.data), truth, ellipses };
}
