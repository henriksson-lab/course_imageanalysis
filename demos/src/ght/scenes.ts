// Templates and seeded scenes for the ght demo (no DOM; also used by demos/test/hough-test.ts).
// A template is an outer polygon plus optional holes, in pixel coordinates around (0, 0)
// (y down). Its oriented edge points are those of a BRIGHT shape on a dark background.
import { Grey, fromArray } from "../lib/imageops";
import { OrientedPt, polygonEdgePoints, placePolygon, fillPolygons, circlePoly, Pt } from "../lib/hough";
import { rng } from "../lib/random";
import { canny, edgeList } from "../lib/edges";

export type Poly = [number, number][];
export interface Template { id: string; label: string; poly: Poly; holes: Poly[] }

export const N = 160;
const FG = 200, BG = 50;

function gear(teeth: number, r0: number, r1: number): Poly {
  const out: Poly = [];
  for (let k = 0; k < teeth; k++) {
    const a = 2 * Math.PI * k / teeth, d = 2 * Math.PI / teeth;
    for (const [f, r] of [[0, r0], [0.15, r1], [0.5, r1], [0.65, r0]] as [number, number][])
      out.push([r * Math.cos(a + f * d), r * Math.sin(a + f * d)]);
  }
  return out;
}

export const TEMPLATES: Template[] = [
  {
    id: "key", label: "key", holes: [[[-17, -4], [-11, -4], [-11, 4], [-17, 4]]],
    poly: [[-24, -10], [-6, -10], [-4, -3], [24, -3], [24, 3], [20, 3], [20, 9], [16, 9], [16, 3], [11, 3], [11, 7], [7, 7], [7, 3], [-4, 3], [-6, 10], [-24, 10]],
  },
  {
    id: "leaf", label: "leaf", holes: [],
    poly: [[-24, 0], [-16, -9], [-6, -14], [6, -13], [15, -8], [22, -2], [26, -1], [26, 1], [22, 2], [15, 8], [6, 12], [-6, 13], [-16, 8]],
  },
  {
    id: "R", label: "letter R", holes: [[[-6, -14], [4, -14], [7, -10], [7, -6], [4, -2], [-6, -2]]],
    poly: [[-14, -22], [6, -22], [14, -16], [15, -8], [11, -1], [18, 22], [9, 22], [3, 3], [-6, 3], [-6, 22], [-14, 22]],
  },
  { id: "gear", label: "gear", holes: [], poly: gear(10, 15, 21) },
  { id: "square", label: "square", holes: [], poly: [[-15, -15], [15, -15], [15, 15], [-15, 15]] },
  { id: "circle", label: "circle", holes: [], poly: circlePoly(0, 0, 18, 72) },
];

/** Oriented edge points of a template (outer boundary and holes), every `step` pixels. */
export function templatePoints(t: { poly: Poly; holes: Poly[] }, step = 1): OrientedPt[] {
  const pts = polygonEdgePoints(t.poly, step);
  // a hole is a dark polygon inside the bright shape: its gradient points away from it
  for (const h of t.holes) for (const p of polygonEdgePoints(h, step)) pts.push({ ...p, theta: p.theta! + Math.PI });
  return pts;
}

/** Canny settings shared by the template and the scene, so both are discretised the same way. */
export const CANNY = { sigma: 1.2, lo: 4, hi: 10 };
const TS = 34;   // template image: (2 TS + 1)^2 pixels, template origin at the centre pixel

/**
 * The template's edge points as the scene will see them: render the template (bright on
 * dark, as in the scene), run Canny with the scene's settings, and shift the points so the
 * template origin is (0, 0). This is how an R-table is built from an example image.
 */
export function templateEdgePoints(t: { poly: Poly; holes: Poly[] }): OrientedPt[] {
  const n = 2 * TS + 1, o: Pt = { x: TS, y: TS };
  const polys = [{ poly: placePolygon(t.poly, { x: 0, y: 0 }, 0, 1, o), value: FG }, ...t.holes.map(h => ({ poly: placePolygon(h, { x: 0, y: 0 }, 0, 1, o), value: BG }))];
  const img = fromArray(n, n, fillPolygons(n, n, polys, BG));
  const c = canny(img, CANNY.sigma, CANNY.lo, CANNY.hi);
  return edgeList(c.hyst.edges, c.gx, c.gy, c.mag).map(p => ({ x: p.x - TS, y: p.y - TS, theta: p.theta, strength: p.strength }));
}

export interface Copy { x: number; y: number; phi: number; s: number; occlude?: number; similar?: boolean }
export interface GhtSceneOptions { copies: Copy[]; clutter: number; noise: number; occlusion: number; seed?: number }
export interface GhtScene { img: Grey; copies: Copy[] }

/** A variant of the template for "similar clutter": stretched by 25 % along x. */
export const similarOf = (t: { poly: Poly; holes: Poly[] }): { poly: Poly; holes: Poly[] } =>
  ({ poly: t.poly.map(([x, y]) => [1.25 * x, 0.85 * y] as [number, number]), holes: t.holes.map(h => h.map(([x, y]) => [1.25 * x, 0.85 * y] as [number, number])) });

/**
 * A scene with the given copies (bright, value 200 on 50), `clutter` random ellipses and
 * bars (value 150), a background-coloured occluding disc over each copy with
 * `occlude` > 0 (or `occlusion` for the first copy), and Gaussian noise. Seeded.
 */
export function makeGhtScene(t: { poly: Poly; holes: Poly[] }, o: GhtSceneOptions): GhtScene {
  const r = rng(o.seed ?? 5), polys: { poly: Poly; value: number }[] = [];
  const origin: Pt = { x: 0, y: 0 };
  for (let k = 0; k < o.clutter; k++) {
    const cx = 10 + (N - 20) * r.unif(), cy = 10 + (N - 20) * r.unif(), a = 4 + 9 * r.unif(), b = 2 + 5 * r.unif(), ang = Math.PI * r.unif();
    const e: Poly = Array.from({ length: 24 }, (_, i) => {
      const u = 2 * Math.PI * i / 24, x = a * Math.cos(u), y = b * Math.sin(u);
      return [cx + x * Math.cos(ang) - y * Math.sin(ang), cy + x * Math.sin(ang) + y * Math.cos(ang)] as [number, number];
    });
    polys.push({ poly: e, value: 150 });
  }
  for (const c of o.copies) {
    const src = c.similar ? similarOf(t) : t;
    polys.push({ poly: placePolygon(src.poly, origin, c.phi, c.s, c), value: FG });
    for (const h of src.holes) polys.push({ poly: placePolygon(h, origin, c.phi, c.s, c), value: BG });
  }
  o.copies.forEach((c, k) => {
    const occ = c.occlude ?? (k === 0 ? o.occlusion : 0);
    if (occ > 0) {
      // a background disc covering one side of the copy; radius grows with occ
      const R = 6 + 30 * occ, ox = c.x + 22 * c.s, oy = c.y - 6 * c.s;
      polys.push({ poly: circlePoly(ox, oy, R, 40), value: BG });
    }
  });
  const data = fillPolygons(N, N, polys, BG);
  let img = fromArray(N, N, data);
  if (o.noise > 0) {
    const rn = rng((o.seed ?? 5) + 99);
    img = fromArray(N, N, img.data.map(v => v + o.noise * rn.norm()));
  }
  return { img, copies: o.copies };
}
