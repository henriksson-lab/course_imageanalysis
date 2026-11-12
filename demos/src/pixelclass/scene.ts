// The synthetic test scene of the pixel-classification demo (no DOM), with ground truth.
// Three classes: 1 = background (uneven, noisy), 2 = nucleus (smooth; a few are dim),
// 3 = granular cell (the SAME mean brightness as a bright nucleus, but a fine granular
// texture). Intensity alone separates background from objects, but not nuclei from granular
// cells: that needs the texture / blob channels of the feature stack. Seeded.
import { Grey, createGrey, gaussianBlur } from "../lib/imageops";
import { Ellipse, rasteriseEllipses } from "../lib/synth";
import { rng } from "../lib/ui";

export const SCENE_CLASSES = ["background", "nucleus", "granular cell"];

export interface Scene {
  image: Grey;
  /** 1 = background, 2 = nucleus, 3 = granular cell. */
  truth: Uint8Array;
  ellipses: Ellipse[];
  kinds: ("bright" | "dim" | "granular")[];
}

export function synthScene(size = 256, seed = 1): Scene {
  const s = size / 256, r = rng(seed), w = size, h = size;
  const spec: ("bright" | "dim" | "granular")[] = ["bright", "granular", "bright", "dim", "granular", "bright", "granular", "bright", "dim", "granular", "bright", "granular", "dim", "bright"];
  const ellipses: Ellipse[] = [], kinds: typeof spec = [];
  for (let q = 0, tries = 0; q < spec.length && tries < 5000; tries++) {
    const big = spec[q] === "granular";
    const m = (big ? 17 : 13) + 4 * r.unif(), el = 1 + 0.35 * r.unif();
    const e: Ellipse = { x: 0, y: 0, a: s * m * Math.sqrt(el), b: s * m / Math.sqrt(el), angle: Math.PI * r.unif() };
    e.x = e.a + 4 * s + (w - 2 * e.a - 8 * s) * r.unif();
    e.y = e.a + 4 * s + (h - 2 * e.a - 8 * s) * r.unif();
    if (ellipses.some(p => Math.hypot(p.x - e.x, p.y - e.y) < p.a + e.a + 6 * s)) continue;
    ellipses.push(e); kinds.push(spec[q]); q++;
  }
  const lab = rasteriseEllipses(w, h, ellipses);
  // granular texture: fine noise blurred a little, scaled to s.d. ≈ 38 grey levels
  const tex = createGrey(w, h);
  for (let i = 0; i < tex.data.length; i++) tex.data[i] = r.norm();
  const tb = gaussianBlur(tex, 1.1 * s);
  let sd = 0; for (const v of tb.data) sd += v * v;
  sd = Math.sqrt(sd / tb.data.length);
  const img = createGrey(w, h), truth = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, k = lab[i];
    let v = 55 + 25 * (x / w) + 10 * Math.sin(y / h * Math.PI);
    truth[i] = 1;
    if (k > 0) {
      const kind = kinds[k - 1];
      if (kind === "bright") { v = 150; truth[i] = 2; }
      else if (kind === "dim") { v = 102; truth[i] = 2; }
      else { v = 150 + 38 * tb.data[i] / sd; truth[i] = 3; }
    }
    img.data[i] = v;
  }
  const out = gaussianBlur(img, 0.8 * s);
  const rn = rng(seed + 101);
  for (let i = 0; i < out.data.length; i++) out.data[i] = Math.min(255, Math.max(0, out.data[i] + 7 * rn.norm()));
  return { image: out, truth, ellipses, kinds };
}
