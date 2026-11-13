// 2D ray optics (owner: ray-tracing day, units whitted/ and optional rayoptics/; no DOM).
// Used by the ray2d sandbox.
//
// A scene is a set of convex regions of glass/water/oil (each an intersection of half-planes
// and discs: slab, prism, ball, biconvex lens, half-disc, layers) in air or another ambient
// medium, plus mirror segments. `trace2d` follows a ray: at every boundary it computes the
// incidence angle, Snell's law and the Fresnel reflectance, and (with `fresnel` on) splits into a
// reflected and a refracted ray carrying R and 1 − R of the power; with `fresnel` off only the
// refracted ray continues (reflected on total internal reflection). Depth cap and power cut-off.
// Dispersion: a two-term Cauchy model n(λ) = A + B/λ² through the given index at 550 nm.
import { fresnel } from "./raytrace";

export type P2 = [number, number];
export type Constraint =
  /** inside: nx·x + ny·y ≤ c (outward normal (nx, ny), unit). */
  | { kind: "half"; nx: number; ny: number; c: number }
  /** inside: ‖x − (cx, cy)‖ ≤ r. */
  | { kind: "disc"; cx: number; cy: number; r: number };

export interface Region { name: string; n: number; /** Cauchy B in µm² (0: no dispersion) */ B: number; cons: Constraint[] }
export interface Mirror { a: P2; b: P2 }
export interface Scene2D { regions: Region[]; mirrors: Mirror[]; nAmbient: number; ambientB?: number }

/** Index at a wavelength (nm) from the index at 550 nm and Cauchy's B (µm²). */
export const cauchy = (n550: number, B: number, nm: number): number => n550 + B * (1 / (nm / 1000) ** 2 - 1 / 0.55 ** 2);

/** Typical Cauchy B for a medium of index n (≈ 0.0042 µm² for crown glass, scaled with n − 1). */
export const defaultB = (n: number): number => (n <= 1.0001 ? 0 : 0.0042 * (n - 1) / 0.52);

const EPS = 1e-7;

export function inside(r: Region, x: number, y: number, tol = 0): boolean {
  for (const c of r.cons) {
    if (c.kind === "half") { if (c.nx * x + c.ny * y > c.c + tol) return false; }
    else if ((x - c.cx) ** 2 + (y - c.cy) ** 2 > (c.r + tol) ** 2) return false;
  }
  return true;
}

/** Medium (index at wavelength nm) at a point: the last region that contains it, else ambient. */
export function indexAt(sc: Scene2D, x: number, y: number, nm = 550): number {
  for (let i = sc.regions.length - 1; i >= 0; i--) { const r = sc.regions[i]; if (inside(r, x, y)) return cauchy(r.n, r.B, nm); }
  return cauchy(sc.nAmbient, sc.ambientB ?? 0, nm);
}

/** Nearest boundary crossing along o + t d (t > tmin): t, the boundary normal (unit) and whether it is a mirror. */
export function nearestBoundary(sc: Scene2D, o: P2, d: P2, tmin = 1e-9): { t: number; nx: number; ny: number; mirror: boolean; region: number } | null {
  let best: { t: number; nx: number; ny: number; mirror: boolean; region: number } | null = null;
  const consider = (t: number, nx: number, ny: number, mirror: boolean, region: number) => {
    if (t > tmin && (!best || t < best.t)) best = { t, nx, ny, mirror, region };
  };
  sc.regions.forEach((r, ri) => {
    r.cons.forEach((c, ci) => {
      const ts: [number, number, number][] = [];   // t, nx, ny
      if (c.kind === "half") {
        const den = c.nx * d[0] + c.ny * d[1];
        if (Math.abs(den) < 1e-15) return;
        ts.push([(c.c - c.nx * o[0] - c.ny * o[1]) / den, c.nx, c.ny]);
      } else {
        const ox = o[0] - c.cx, oy = o[1] - c.cy, b = ox * d[0] + oy * d[1], k = ox * ox + oy * oy - c.r * c.r;
        const disc = b * b - k;
        if (disc < 0) return;
        const s = Math.sqrt(disc);
        for (const t of [-b - s, -b + s]) { const x = o[0] + t * d[0], y = o[1] + t * d[1]; ts.push([t, (x - c.cx) / c.r, (y - c.cy) / c.r]); }
      }
      for (const [t, nx, ny] of ts) {
        if (!(t > tmin)) continue;
        const x = o[0] + t * d[0], y = o[1] + t * d[1];
        // on the region's boundary: satisfies every other constraint
        let ok = true;
        r.cons.forEach((c2, cj) => {
          if (cj === ci || !ok) return;
          if (c2.kind === "half") { if (c2.nx * x + c2.ny * y > c2.c + 1e-9) ok = false; }
          else if ((x - c2.cx) ** 2 + (y - c2.cy) ** 2 > (c2.r + 1e-9) ** 2) ok = false;
        });
        if (ok) consider(t, nx, ny, false, ri);
      }
    });
  });
  for (const m of sc.mirrors) {
    const ex = m.b[0] - m.a[0], ey = m.b[1] - m.a[1], den = d[0] * ey - d[1] * ex;
    if (Math.abs(den) < 1e-15) continue;
    const wx = m.a[0] - o[0], wy = m.a[1] - o[1];
    const t = (wx * ey - wy * ex) / den, s = (wx * d[1] - wy * d[0]) / den;
    if (s < 0 || s > 1) continue;
    const L = Math.hypot(ex, ey);
    consider(t, ey / L, -ex / L, true, -1);
  }
  return best;
}

export type SegKind = "source" | "reflect" | "refract";
export interface Seg { x0: number; y0: number; x1: number; y1: number; power: number; kind: SegKind; depth: number; nm: number; escaped: boolean }
export interface HitInfo {
  x: number; y: number;
  /** Incidence and refraction angles in degrees from the normal (θ2 null on total internal reflection). */
  theta1: number; theta2: number | null;
  n1: number; n2: number;
  /** Fresnel reflectance (1 on TIR and at mirrors). */
  R: number;
  power: number;
  mirror: boolean;
  depth: number;
  nm: number;
}

export interface Trace2DOptions { fresnel: boolean; maxDepth?: number; minPower?: number; nm?: number; maxLen?: number }

/** Trace one ray from (o, d) with the given power; returns the drawn segments and the hits. */
export function trace2d(sc: Scene2D, o: P2, d: P2, power: number, opt: Trace2DOptions, out: { segs: Seg[]; hits: HitInfo[] } = { segs: [], hits: [] }, kind: SegKind = "source", depth = 0): { segs: Seg[]; hits: HitInfo[] } {
  const nm = opt.nm ?? 550, maxD = opt.maxDepth ?? 12, minP = opt.minPower ?? 1e-3, maxLen = opt.maxLen ?? 1e4;
  const L = Math.hypot(d[0], d[1]); d = [d[0] / L, d[1] / L];
  const h = nearestBoundary(sc, o, d);
  if (!h || h.t > maxLen) {
    out.segs.push({ x0: o[0], y0: o[1], x1: o[0] + maxLen * d[0], y1: o[1] + maxLen * d[1], power, kind, depth, nm, escaped: true });
    return out;
  }
  const x = o[0] + h.t * d[0], y = o[1] + h.t * d[1];
  out.segs.push({ x0: o[0], y0: o[1], x1: x, y1: y, power, kind, depth, nm, escaped: false });
  // normal facing the incoming ray
  let nx = h.nx, ny = h.ny;
  if (nx * d[0] + ny * d[1] > 0) { nx = -nx; ny = -ny; }
  const c1 = -(nx * d[0] + ny * d[1]);
  const rx = d[0] + 2 * c1 * nx, ry = d[1] + 2 * c1 * ny;
  const th1 = Math.acos(Math.min(1, c1)) * 180 / Math.PI;
  if (h.mirror) {
    out.hits.push({ x, y, theta1: th1, theta2: null, n1: NaN, n2: NaN, R: 1, power, mirror: true, depth, nm });
    if (depth < maxD && power >= minP) trace2d(sc, [x + EPS * nx, y + EPS * ny], [rx, ry], power, opt, out, "reflect", depth + 1);
    return out;
  }
  const n1 = indexAt(sc, x + EPS * nx, y + EPS * ny, nm), n2 = indexAt(sc, x - EPS * nx, y - EPS * ny, nm);
  const eta = n1 / n2, k = 1 - eta * eta * (1 - c1 * c1);
  const tir = k < 0;
  const R = tir ? 1 : fresnel(c1, n1, n2);
  let th2: number | null = null, tx = 0, ty = 0;
  if (!tir) { const f = eta * c1 - Math.sqrt(k); tx = eta * d[0] + f * nx; ty = eta * d[1] + f * ny; th2 = Math.asin(Math.min(1, eta * Math.sqrt(1 - c1 * c1))) * 180 / Math.PI; }
  out.hits.push({ x, y, theta1: th1, theta2: th2, n1, n2, R, power, mirror: false, depth, nm });
  if (depth >= maxD) return out;
  if (Math.abs(n1 - n2) < 1e-12) {   // same medium on both sides (touching regions): go straight on
    trace2d(sc, [x - EPS * nx, y - EPS * ny], d, power, opt, out, kind, depth + 1);
    return out;
  }
  if (opt.fresnel) {
    if (power * R >= minP) trace2d(sc, [x + EPS * nx, y + EPS * ny], [rx, ry], power * R, opt, out, "reflect", depth + 1);
    if (!tir && power * (1 - R) >= minP) trace2d(sc, [x - EPS * nx, y - EPS * ny], [tx, ty], power * (1 - R), opt, out, "refract", depth + 1);
  } else if (tir) trace2d(sc, [x + EPS * nx, y + EPS * ny], [rx, ry], power, opt, out, "reflect", depth + 1);
  else trace2d(sc, [x - EPS * nx, y - EPS * ny], [tx, ty], power, opt, out, "refract", depth + 1);
  return out;
}

// ---- shape constructors (world coordinates; angle in degrees, counter-clockwise with y up) ----

const rot = (x: number, y: number, a: number): P2 => { const c = Math.cos(a), s = Math.sin(a); return [c * x - s * y, s * x + c * y]; };
function halfFrom(px: number, py: number, nx: number, ny: number): Constraint { return { kind: "half", nx, ny, c: nx * px + ny * py }; }

/** A convex polygon (vertices counter-clockwise) as half-planes. */
export function polygonRegion(name: string, n: number, pts: P2[], B = defaultB(n)): Region {
  const cons: Constraint[] = [];
  for (let i = 0; i < pts.length; i++) {
    const [ax, ay] = pts[i], [bx, by] = pts[(i + 1) % pts.length], ex = bx - ax, ey = by - ay, L = Math.hypot(ex, ey);
    cons.push(halfFrom(ax, ay, ey / L, -ex / L));
  }
  return { name, n, B, cons };
}
export function rectRegion(name: string, n: number, cx: number, cy: number, w: number, h: number, angleDeg = 0, B = defaultB(n)): Region {
  const a = angleDeg * Math.PI / 180;
  const pts: P2[] = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]].map(([x, y]) => { const [u, v] = rot(x, y, a); return [cx + u, cy + v] as P2; });
  return polygonRegion(name, n, pts, B);
}
export function discRegion(name: string, n: number, cx: number, cy: number, r: number, B = defaultB(n)): Region {
  return { name, n, B, cons: [{ kind: "disc", cx, cy, r }] };
}
/** Symmetric biconvex lens: two arcs of radius R, centre thickness th, optical axis along angleDeg. */
export function lensRegion(name: string, n: number, cx: number, cy: number, R: number, th: number, angleDeg = 0, B = defaultB(n)): Region {
  const a = angleDeg * Math.PI / 180, ux = Math.cos(a), uy = Math.sin(a), off = R - th / 2;
  return { name, n, B, cons: [{ kind: "disc", cx: cx + off * ux, cy: cy + off * uy, r: R }, { kind: "disc", cx: cx - off * ux, cy: cy - off * uy, r: R }] };
}
/** Equilateral prism of side s, centred, rotated. */
export function prismRegion(name: string, n: number, cx: number, cy: number, s: number, angleDeg = 0, B = defaultB(n)): Region {
  const a = angleDeg * Math.PI / 180, R = s / Math.sqrt(3);
  const pts: P2[] = [90, 210, 330].map(t => { const [u, v] = rot(R * Math.cos(t * Math.PI / 180), R * Math.sin(t * Math.PI / 180), a); return [cx + u, cy + v] as P2; });
  return polygonRegion(name, n, pts, B);
}
/** Half-disc: the flat side through the centre, its outward normal at angleDeg. */
export function halfDiscRegion(name: string, n: number, cx: number, cy: number, r: number, angleDeg = 0, B = defaultB(n)): Region {
  const a = angleDeg * Math.PI / 180;
  return { name, n, B, cons: [{ kind: "disc", cx, cy, r }, halfFrom(cx, cy, Math.cos(a), Math.sin(a))] };
}

/** Paraxial focal length of a thin symmetric biconvex lens (lensmaker): 1/f = (n − 1)(2/R). */
export const thinLensF = (n: number, R: number, nAmb = 1): number => 1 / ((n / nAmb - 1) * (2 / R));
/** Paraxial focus of a ball lens, measured from its centre: f = n R / (2 (n − 1)). */
export const ballLensF = (n: number, R: number): number => n * R / (2 * (n - 1));
