// Seeded synthetic scenes for the houghlines demo (no DOM; also used by demos/test/hough-test.ts).
// Every scene is N x N, grey values 0..255, built from filled polygons so that each straight
// boundary gives one edge. Ground truth: the polygon sides as (theta, rho) lines in the
// world convention of lib/hough.ts (origin at the image centre, y up).
import { Grey, fromArray } from "../lib/imageops";
import { fillPolygons, lineThrough } from "../lib/hough";
import { rng } from "../lib/random";

export const N = 192;
export type SceneId = "shapes" | "grid" | "road" | "page" | "filaments";
export const SCENES: { id: SceneId; label: string }[] = [
  { id: "shapes", label: "shapes" },
  { id: "grid", label: "window grid" },
  { id: "road", label: "road" },
  { id: "page", label: "tilted page" },
  { id: "filaments", label: "filaments" },
];

type Poly = [number, number][];
export interface Scene { img: Grey; polys: { poly: Poly; value: number }[]; tilt?: number }

const C = (N - 1) / 2;
const rot = (p: Poly, ang: number, cx = C, cy = C): Poly => p.map(([x, y]) => {
  // rotate CCW on screen by ang about (cx, cy)
  const dx = x - cx, dy = y - cy, c = Math.cos(ang), s = Math.sin(ang);
  return [cx + c * dx + s * dy, cy - s * dx + c * dy];
});
const rect = (x0: number, y0: number, x1: number, y1: number): Poly => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
/** A thin bar from (x0, y0) to (x1, y1) of width wd. */
function bar(x0: number, y0: number, x1: number, y1: number, wd: number): Poly {
  const L = Math.hypot(x1 - x0, y1 - y0), nx = -(y1 - y0) / L * wd / 2, ny = (x1 - x0) / L * wd / 2;
  return [[x0 + nx, y0 + ny], [x1 + nx, y1 + ny], [x1 - nx, y1 - ny], [x0 - nx, y0 - ny]];
}

export function makeScene(id: SceneId, seed = 3): Scene {
  const r = rng(seed);
  let polys: { poly: Poly; value: number }[] = [], base = 40, tilt: number | undefined;
  if (id === "shapes") {
    base = 50;
    polys = [
      { poly: [[20, 150], [95, 30], [150, 160]], value: 170 },
      { poly: rot(rect(105, 40, 175, 90), 0.35, 140, 65), value: 230 },
      { poly: [[30, 40], [80, 25], [70, 90], [15, 80]], value: 120 },
    ];
  } else if (id === "grid") {
    base = 90;
    for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) {
      if ((i === 1 && j === 2) || (i === 3 && j === 0)) continue;   // two missing windows: gaps in the rows
      polys.push({ poly: rect(18 + i * 33, 22 + j * 40, 18 + i * 33 + 22, 22 + j * 40 + 26), value: 200 });
    }
  } else if (id === "road") {
    base = 150;
    polys = [
      { poly: [[0, 70], [N, 70], [N, N], [0, N]], value: 120 },                  // ground
      { poly: [[86, 70], [106, 70], [176, N], [16, N]], value: 55 },             // road
      { poly: bar(90, 72, 31, N - 1, 4), value: 235 },                          // left lane line
      { poly: bar(102, 72, 161, N - 1, 4), value: 235 },                        // right lane line
      { poly: [[0, 0], [N, 0], [N, 70], [0, 70]], value: 200 },                  // sky
    ];
    // dashed centre line
    for (let t = 0; t < 6; t++) {
      const y0 = 78 + t * 20, y1 = y0 + 10;
      polys.push({ poly: bar(96, y0, 96, y1, 1.5 + 0.03 * (y0 - 70)), value: 235 });
    }
  } else if (id === "page") {
    base = 60;
    tilt = 7 * Math.PI / 180;
    const page: { poly: Poly; value: number }[] = [{ poly: rect(22, 14, 170, 178), value: 235 }];
    for (let l = 0; l < 12; l++) {
      const y = 28 + l * 12.5;
      let x = 34;
      const end = l === 11 ? 100 : 158;
      while (x < end - 8) {
        const wlen = 6 + Math.floor(r.unif() * 22), x1 = Math.min(end, x + wlen);
        page.push({ poly: rect(x, y, x1, y + 5), value: 70 });
        x = x1 + 4;
      }
    }
    polys = page.map(p => ({ poly: rot(p.poly, tilt!), value: p.value }));
  } else {
    base = 45;
    for (let k = 0; k < 6; k++) {
      const a = Math.PI * r.unif(), L = 60 + 70 * r.unif(), cx = 30 + (N - 60) * r.unif(), cy = 30 + (N - 60) * r.unif();
      const dx = Math.cos(a) * L / 2, dy = Math.sin(a) * L / 2;
      polys.push({ poly: bar(cx - dx, cy - dy, cx + dx, cy + dy, 2.5), value: 200 });
    }
  }
  const data = fillPolygons(N, N, polys, base);
  return { img: fromArray(N, N, data), polys, tilt };
}

/** Random round blobs (texture/clutter): `density` blobs per 10 000 pixels, radius 1.5-4. Seeded. */
export function addClutter(img: Grey, density: number, seed = 11): Grey {
  if (density <= 0) return img;
  const r = rng(seed), out = fromArray(img.width, img.height, img.data), n = Math.round(density * img.width * img.height / 10000);
  for (let k = 0; k < n; k++) {
    const cx = img.width * r.unif(), cy = img.height * r.unif(), rad = 1.5 + 2.5 * r.unif(), v = 255 * r.unif();
    const R = Math.ceil(rad + 1);
    for (let y = Math.max(0, Math.floor(cy - R)); y <= Math.min(img.height - 1, Math.ceil(cy + R)); y++)
      for (let x = Math.max(0, Math.floor(cx - R)); x <= Math.min(img.width - 1, Math.ceil(cx + R)); x++) {
        const f = Math.max(0, Math.min(1, rad + 0.5 - Math.hypot(x - cx, y - cy)));
        if (f > 0) { const i = y * img.width + x; out.data[i] = out.data[i] * (1 - f) + v * f; }
      }
  }
  return out;
}

/** The sides of the scene's polygons as normal-form lines (world coordinates of lib/hough.ts), with their lengths. */
export function groundTruthLines(s: Scene): { theta: number; rho: number; length: number }[] {
  const out: { theta: number; rho: number; length: number }[] = [];
  for (const { poly } of s.polys) for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    const L = lineThrough(x0 - C, C - y0, x1 - C, C - y1);
    out.push({ ...L, length: Math.hypot(x1 - x0, y1 - y0) });
  }
  return out;
}
