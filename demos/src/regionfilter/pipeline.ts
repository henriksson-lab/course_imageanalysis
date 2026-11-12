// The measurement pipeline of the regionfilter demo (labelling day, unit label-measure/):
// threshold → open → fill holes → label → measure → filter → count. No DOM.
import { Grey, gaussianBlur } from "../lib/imageops";
import { discs, addGaussianNoise, Ellipse } from "../lib/synth";
import { rng } from "../lib/random";
import { label2d, relabel } from "../lib/label";
import { openBinary, seDisk, fillHoles } from "../lib/morph";
import { maskHull, shoelaceArea } from "../lib/convexhull";
import { regionProps, cropLabel, Region } from "../moments/regionprops";

export type Kind = "nucleus" | "rod" | "speck" | "pair";

/** Seeded synthetic "mixed population": round nuclei, rods, specks and touching pairs of
 *  nuclei, some cut by the image border; anti-aliased, blurred, noisy. */
export function mixedPopulation(seed = 3, size = 256): { image: Grey; truth: Int32Array; kinds: Kind[]; ellipses: Ellipse[] } {
  const r = rng(seed), es: Ellipse[] = [], kinds: Kind[] = [], values: number[] = [];
  const fits = (e: Ellipse, gap: number, ignore = -1) => es.every((q, k) => k === ignore || Math.hypot(q.x - e.x, q.y - e.y) > q.a + e.a + gap);
  const place = (kind: Kind, make: () => Ellipse, n: number, v: number, edge = false) => {
    for (let k = 0, t = 0; k < n && t < 3000; t++) {
      const e = make();
      const m = edge ? -e.b * 0.6 : e.a + 3;
      e.x = m + r.unif() * (size - 2 * m); e.y = m + r.unif() * (size - 2 * m);
      if (edge) { if (Math.min(e.x, e.y, size - 1 - e.x, size - 1 - e.y) > e.b * 0.5) continue; }
      else if (e.x < e.a + 2 || e.y < e.a + 2 || e.x > size - e.a - 3 || e.y > size - e.a - 3) continue;
      if (!fits(e, kind === "speck" ? 4 : 6)) continue;
      es.push(e); kinds.push(kind); values.push(v + 20 * (r.unif() - 0.5)); k++;
    }
  };
  const nucleus = (): Ellipse => { const m = 9 + 4 * r.unif(), q = 1 + 0.25 * r.unif(); return { x: 0, y: 0, a: m * Math.sqrt(q), b: m / Math.sqrt(q), angle: Math.PI * r.unif() }; };
  // touching pairs first (they need room)
  for (let k = 0, t = 0; k < 3 && t < 2000; t++) {
    const a = nucleus(), b = nucleus();
    a.x = 30 + r.unif() * (size - 60); a.y = 30 + r.unif() * (size - 60);
    const ang = 2 * Math.PI * r.unif(), d = 0.85 * (a.b + b.b);
    b.x = a.x + d * Math.cos(ang); b.y = a.y + d * Math.sin(ang);
    const pair: Ellipse = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, a: d / 2 + Math.max(a.a, b.a), b: 0, angle: 0 };
    if (!fits(pair, 8) || b.x < b.a + 3 || b.y < b.a + 3 || b.x > size - b.a - 3 || b.y > size - b.a - 3) continue;
    es.push(a, b); kinds.push("pair", "pair"); values.push(175, 175); k++;
  }
  place("nucleus", nucleus, 12, 175);
  place("nucleus", nucleus, 3, 175, true);
  place("rod", () => ({ x: 0, y: 0, a: 13 + 6 * r.unif(), b: 3.2 + 1.2 * r.unif(), angle: Math.PI * r.unif() }), 9, 165);
  place("speck", () => { const s = 1 + 1.2 * r.unif(); return { x: 0, y: 0, a: s, b: s, angle: 0 }; }, 22, 150);
  const d = discs(size, size, es, { background: 40, values });
  const img = addGaussianNoise(gaussianBlur(d.image, 1), 10, seed + 101);
  return { image: img, truth: d.labels, kinds, ellipses: es };
}

/** Keep the 8- (or 4-) connected components with at least `a` pixels: area opening. */
export function areaOpening(mask: ArrayLike<number>, w: number, h: number, a: number, conn: 4 | 8 = 8): Uint8Array {
  const lab = label2d(mask, w, h, conn), size = new Int32Array(lab.count + 1);
  for (const v of lab.data) size[v]++;
  return Uint8Array.from(lab.data, v => (v && size[v] >= a ? 1 : 0));
}

/** Unbiased counting frame (Gundersen) with inclusion edges top and left: the frame is
 *  [m, w−m) × [m, h−m); an object counts if it hits the frame and does not touch the
 *  exclusion line = the right edge (column w−m) extended up to the top of the image and the
 *  bottom edge (row h−m) extended left to the image border. keep[label] = 1 if counted. */
export function countingFrame(lab: ArrayLike<number>, w: number, h: number, count: number, m: number): Uint8Array {
  const hits = new Uint8Array(count + 1), bad = new Uint8Array(count + 1);
  const x1 = w - m, y1 = h - m;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const l = lab[y * w + x];
    if (!l) continue;
    if (x >= m && x < x1 && y >= m && y < y1) hits[l] = 1;
    if ((x === x1 && y <= y1) || (y === y1 && x <= x1)) bad[l] = 1;
  }
  return Uint8Array.from(hits, (v, l) => (l && v && !bad[l] ? 1 : 0));
}

export type BorderRule = "keep" | "remove" | "frame";

export interface PipelineOptions {
  threshold: number;
  /** Objects darker than the background (threshold below). */
  dark?: boolean;
  openRadius: number;
  fill: boolean;
  conn: 4 | 8;
  border: BorderRule;
  frameMargin?: number;
}

export interface Row extends Region {
  solidity: number;
  formFactor: number;
  perimeter: number;
  /** Counted by the border rule. */
  counted: boolean;
}

export const PROPS: { id: keyof Row; label: string; fmt: (v: number) => string }[] = [
  { id: "area", label: "area", fmt: v => v.toFixed(0) },
  { id: "perimeter", label: "perimeter", fmt: v => v.toFixed(1) },
  { id: "eccentricity", label: "eccentricity", fmt: v => v.toFixed(2) },
  { id: "solidity", label: "solidity", fmt: v => v.toFixed(2) },
  { id: "formFactor", label: "form factor", fmt: v => v.toFixed(2) },
  { id: "major", label: "major axis", fmt: v => v.toFixed(1) },
  { id: "minor", label: "minor axis", fmt: v => v.toFixed(1) },
];

/** Threshold → open → fill holes → label → measure (+ border rule). */
export function runPipeline(img: Grey, o: PipelineOptions): { mask: Uint8Array; labels: Int32Array; count: number; rows: Row[] } {
  const w = img.width, h = img.height;
  let mask: Uint8Array = Uint8Array.from(img.data, v => ((o.dark ? v < o.threshold : v >= o.threshold) ? 1 : 0));
  if (o.openRadius > 0) mask = openBinary(mask, w, h, seDisk(o.openRadius));
  if (o.fill) mask = fillHoles(mask, w, h, o.conn === 8 ? 4 : 8);
  let lab = label2d(mask, w, h, o.conn);
  if (o.border === "remove") {
    const props0 = regionProps(lab.data, w, h, lab.count, { perimeters: false });
    const drop = new Set(props0.filter(p => p.touchesBorder).map(p => p.label));
    lab = relabel(lab.data.map(v => (drop.has(v) ? 0 : v)), w, h);
  }
  const keep = o.border === "frame" ? countingFrame(lab.data, w, h, lab.count, o.frameMargin ?? 16) : null;
  const props = regionProps(lab.data, w, h, lab.count);
  const rows: Row[] = props.map(p => {
    const c = cropLabel(lab.data, w, p.x0, p.y0, p.x1 - p.x0 + 1, p.y1 - p.y0 + 1, p.label);
    const ha = Math.abs(shoelaceArea(maskHull(c.mask, c.w, c.h, "corners")));
    const P = p.perimCrofton;
    return { ...p, solidity: ha > 0 ? p.area / ha : 1, perimeter: P, formFactor: P > 0 ? 4 * Math.PI * p.area / (P * P) : 0, counted: keep ? !!keep[p.label] : true };
  });
  return { mask, labels: lab.data, count: lab.count, rows };
}
