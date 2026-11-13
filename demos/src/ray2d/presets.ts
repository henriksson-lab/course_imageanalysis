// Presets and geometry of the ray2d demo (no DOM; also used by the figure and number scripts).
import * as R2 from "../lib/rays2d";

export type ObjType = "interface" | "slab" | "ball" | "lens" | "prism" | "halfdisc" | "mirror" | "layer";
export interface Obj { type: ObjType; name: string; x: number; y: number; angle: number; n: number; a: number; b: number }
export type SrcType = "ray" | "point" | "beam" | "focus";
export interface Src { type: SrcType; x: number; y: number; angle: number; count: number; spread: number; width: number; focus: number }
export interface State { objs: Obj[]; src: Src; fresnel: boolean; dispersion: boolean; nAmb: number; sel: number; hit: number; axis: boolean }

export const mk = (type: ObjType, name: string, x: number, y: number, n: number, a = 0, b = 0, angle = 0): Obj => ({ type, name, x, y, angle, n, a, b });
export const src = (s: Partial<Src>): Src => ({ type: "ray", x: 4, y: 20, angle: -30, count: 1, spread: 120, width: 6, focus: 20, ...s });

export const PRESETS: { id: string; label: string; s: () => State }[] = [
  { id: "snell", label: "Snell at a flat surface", s: () => st0([mk("interface", "glass", 30, 12, 1.5)], src({ type: "ray", x: 20 - 10 * Math.tan(Math.PI / 6), y: 22, angle: -60 }), { hit: 0 }) },
  { id: "critical", label: "the critical angle", s: () => st0([mk("interface", "glass", 32, 14, 1.5)], src({ type: "point", x: 20, y: 3, angle: 90, count: 25, spread: 150 }), { fresnel: false }) },
  { id: "guide", label: "a light guide", s: () => st0([mk("slab", "fibre core", 21, 12, 1.5, 38, 3)], src({ type: "point", x: 2.2, y: 12, angle: 0, count: 9, spread: 70 }), { fresnel: false }) },
  { id: "grazing", label: "grazing glass: a mirror", s: () => st0([mk("slab", "glass plate", 22, 10, 1.52, 30, 3)], src({ type: "beam", x: 3, y: 15.5, angle: -12, count: 5, width: 2 }), { fresnel: true }) },
  { id: "ball", label: "a ball: aberration", s: () => st0([mk("ball", "glass ball", 13, 12.5, 1.5, 5)], src({ type: "beam", x: 2, y: 12.5, angle: 0, count: 21, width: 9.6 }), { fresnel: false, axis: true }) },
  { id: "lens", label: "a thin lens focuses", s: () => st0([mk("lens", "lens", 10, 12.5, 1.5, 20, 2.4)], src({ type: "beam", x: 2, y: 12.5, angle: 0, count: 11, width: 5 }), { fresnel: false, axis: true }) },
  { id: "prism", label: "prism: dispersion", s: () => st0([mk("prism", "prism", 18, 11, 1.52, 9, 0, 0)], src({ type: "ray", x: 3, y: 9.2, angle: 12 }), { fresnel: false, dispersion: true }) },
  { id: "oil", label: "oil objective, watery sample", s: () => stackState(1.518) },
  { id: "water", label: "water objective", s: () => stackState(1.33) },
];

function stackState(nImm: number): State {
  return st0([
    mk("layer", "immersion", 7, 12.5, nImm, 14, 60),
    mk("layer", "coverslip", 14.75, 12.5, 1.52, 1.5, 60),
    mk("layer", "sample (water)", 115.5, 12.5, 1.33, 200, 60),
  ], src({ type: "focus", x: 6, y: 12.5, angle: 0, count: 15, spread: 80, focus: 16 }), { fresnel: false, axis: true, nAmb: 1 });
}
function st0(objs: Obj[], s: Src, o: Partial<State> = {}): State { return { objs, src: s, fresnel: true, dispersion: false, nAmb: 1, sel: 0, hit: -1, axis: false, ...o }; }

export function regionOf(o: Obj): R2.Region | null {
  const B = R2.defaultB(o.n);
  switch (o.type) {
    case "interface": { const a = o.angle * Math.PI / 180; return R2.rectRegion(o.name, o.n, o.x + 40 * Math.sin(a), o.y - 40 * Math.cos(a), 400, 80, o.angle, B); }
    case "slab": case "layer": return R2.rectRegion(o.name, o.n, o.x, o.y, o.a, o.b, o.angle, B);
    case "ball": return R2.discRegion(o.name, o.n, o.x, o.y, o.a, B);
    case "lens": return R2.lensRegion(o.name, o.n, o.x, o.y, o.a, o.b, o.angle, B);
    case "prism": return R2.prismRegion(o.name, o.n, o.x, o.y, o.a, o.angle, B);
    case "halfdisc": return R2.halfDiscRegion(o.name, o.n, o.x, o.y, o.a, o.angle, B);
    case "mirror": return null;
  }
}

export function sceneOf(st: State): R2.Scene2D {
  const regions: R2.Region[] = [], mirrors: R2.Mirror[] = [];
  for (const o of st.objs) {
    const r = regionOf(o);
    if (r) regions.push(r);
    else { const a = o.angle * Math.PI / 180, c = Math.cos(a) * o.a / 2, s = Math.sin(a) * o.a / 2; mirrors.push({ a: [o.x - c, o.y - s], b: [o.x + c, o.y + s] }); }
  }
  return { regions, mirrors, nAmbient: st.nAmb };
}

export function sourceRays(st: State): { o: R2.P2; d: R2.P2 }[] {
  const s = st.src, a = s.angle * Math.PI / 180, d: R2.P2 = [Math.cos(a), Math.sin(a)], nrm: R2.P2 = [-d[1], d[0]];
  const n = s.type === "ray" ? 1 : s.count, out: { o: R2.P2; d: R2.P2 }[] = [];
  for (let k = 0; k < n; k++) {
    const f = n === 1 ? 0 : k / (n - 1) - 0.5;
    if (s.type === "ray") out.push({ o: [s.x, s.y], d });
    else if (s.type === "point") { const b = a + f * s.spread * Math.PI / 180; out.push({ o: [s.x, s.y], d: [Math.cos(b), Math.sin(b)] }); }
    else if (s.type === "beam") out.push({ o: [s.x + f * s.width * nrm[0], s.y + f * s.width * nrm[1]], d });
    else {
      // converging: rays from a line through the source, aimed at the nominal focus s.focus ahead
      const half = Math.tan(s.spread / 2 * Math.PI / 180) * s.focus, o: R2.P2 = [s.x + 2 * f * half * nrm[0], s.y + 2 * f * half * nrm[1]];
      const F: R2.P2 = [s.x + s.focus * d[0], s.y + s.focus * d[1]];
      out.push({ o, d: [F[0] - o[0], F[1] - o[1]] });
    }
  }
  return out;
}


/** Trace every source ray (one wavelength). */
export function traceAll(st: State, nm = 550): { segs: R2.Seg[]; hits: R2.HitInfo[] } {
  const sc = sceneOf(st), out = { segs: [] as R2.Seg[], hits: [] as R2.HitInfo[] };
  for (const r of sourceRays(st)) R2.trace2d(sc, r.o, r.d, 1, { fresnel: st.fresnel, maxDepth: 12, minPower: 1e-3, nm, maxLen: 200 }, out);
  return out;
}

/** Where the refracted and outgoing rays (power ≥ 5 %) cross the horizontal axis y = y0 inside the window. */
export function axisCrossings(segs: R2.Seg[], y0: number, xmax = 40): number[] {
  const xs: number[] = [];
  for (const g of segs) {
    if ((!g.escaped && g.kind !== "refract") || g.power < 0.05) continue;
    const dy = g.y1 - g.y0;
    if (Math.abs(dy) < 1e-9) continue;
    const t = (y0 - g.y0) / dy;
    const x = g.x0 + t * (g.x1 - g.x0);
    if (t > 0 && t < 1 && x >= 0 && x <= xmax) xs.push(x);
  }
  return xs;
}
