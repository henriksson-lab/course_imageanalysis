// The dock2d scene (no DOM, used by the demo, its test and the figure script): a synthetic
// 2D receptor with a pocket and a key-shaped ligand, in world units on a 64 × 64 box (y down).
// The shapes stand in for molecules; nothing here is a real protein.
//
// The receptor is an ellipse with the pocket cut out: every point within `CLEARANCE` of the
// ligand at its true pose (without the flexible bump) is removed. A grid of spacing h has
// n = ceil(64 / h) cells across the box; it sits in the middle of an M × M grid (M = the next
// power of two ≥ 2n), so a ligand near the box never wraps around onto the receptor
// (zero-padding, as on the FFT registration day).
import {
  Poly, Pt, centred, inside, distToOutline, rotatePolyCW, layers, receptorGrid, ligandCells,
  Docker, Layers, Ligand, localMaxima, topPoses, Pose, scoreAt,
} from "../lib/docking";
import { nextPow2 } from "../lib/fft";

export const BOX = 64;
export const CLEARANCE = 0.5;

/** The key, before centring: a cap with a stem that has a tooth on its right. */
const KEY: Poly = [[-8, -5], [8, -5], [8, 0], [3, 0], [3, 5], [5.5, 5], [5.5, 8], [3, 8], [3, 11], [-3, 11], [-3, 0], [-8, 0]];
/** The same key with a flexible bump on the left of the stem (the receptor has no room for it). */
const KEY_BUMP: Poly = [[-8, -5], [8, -5], [8, 0], [3, 0], [3, 5], [5.5, 5], [5.5, 8], [3, 8], [3, 11], [-3, 11], [-3, 6.5], [-5.5, 6.5], [-5.5, 3.5], [-3, 3.5], [-3, 0], [-8, 0]];

/** The ligand, centred on its centroid. The bump moves the centroid a little: both versions
 *  use the centroid of the plain key, so that the true pose is the same. */
const KEY_C = centred(KEY);
const SHIFT: Pt = [KEY_C[0][0] - KEY[0][0], KEY_C[0][1] - KEY[0][1]];
export function ligandPoly(bump: boolean): Poly {
  return bump ? KEY_BUMP.map(([x, y]) => [x + SHIFT[0], y + SHIFT[1]] as Pt) : KEY_C;
}

/** The true pose: the centroid of the key in world units and its angle (degrees, clockwise). */
export const TRUE_POSE = { x: 32.5, y: 28.5, angle: 20 };

/** The ellipse of the receptor body. */
export const BODY = { cx: 32, cy: 40, rx: 24, ry: 14 };

/** The key at its true pose, in world units. */
export function trueKey(): Poly {
  return rotatePolyCW(KEY_C, TRUE_POSE.angle).map(([x, y]) => [x + TRUE_POSE.x, y + TRUE_POSE.y] as Pt);
}

/** Is the world point (x, y) part of the receptor? */
export function inReceptor(x: number, y: number, key: Poly = trueKey()): boolean {
  const ex = (x - BODY.cx) / BODY.rx, ey = (y - BODY.cy) / BODY.ry;
  if (ex * ex + ey * ey > 1) return false;
  return !(inside(key, x, y) || distToOutline(key, x, y) < CLEARANCE);
}

export interface Settings {
  rho: number;        // value of the core
  thickness: number;  // surface layer, in cells
  spacing: number;    // grid spacing h, world units per cell
  step: number;       // angle step of the search, degrees
  bump: boolean;      // the flexible bump on the ligand
  angle: number;      // ligand angle by hand, degrees
  x: number;          // ligand centroid by hand, world units
  y: number;
}
export const DEFAULTS: Settings = { rho: -9, thickness: 2, spacing: 1, step: 5, bump: false, angle: 0, x: 14, y: 12 };

export interface Scene {
  h: number;
  n: number;          // cells across the box
  M: number;          // FFT grid size
  o: number;          // first cell of the box in the M grid
  R: Layers;          // receptor layers on the M grid
  a: Float64Array;    // receptor grid
}

/** The receptor on the grid of spacing h. */
export function makeScene(s: Settings): Scene {
  const h = s.spacing, n = Math.ceil(BOX / h), M = nextPow2(2 * n), o = (M - n) >> 1;
  const key = trueKey(), mask = new Uint8Array(M * M);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++)
    if (inReceptor((i + 0.5) * h, (j + 0.5) * h, key)) mask[(j + o) * M + i + o] = 1;
  const R = layers(mask, M, M, s.thickness);
  return { h, n, M, o, R, a: receptorGrid(R, s.rho) };
}

/** Grid cell of the ligand's reference cell for a centroid at world (x, y), and back. */
export const toCell = (sc: Scene, x: number, y: number): [number, number] =>
  [Math.round(x / sc.h - 0.5) + sc.o, Math.round(y / sc.h - 0.5) + sc.o];
export const toWorld = (sc: Scene, gx: number, gy: number): [number, number] =>
  [(gx - sc.o + 0.5) * sc.h, (gy - sc.o + 0.5) * sc.h];

/** The ligand's cells at an angle on the scene's grid. */
export const ligandAt = (sc: Scene, s: Settings, angle: number): Ligand => ligandCells(ligandPoly(s.bump), angle, sc.h);

/** Score of the hand-placed pose. */
export function handScore(sc: Scene, s: Settings) {
  const [gx, gy] = toCell(sc, s.x, s.y);
  return { ...scoreAt(sc.a, sc.R, ligandAt(sc, s, s.angle), gx, gy), gx, gy };
}

/** The angles of a search with step `step`: 0, step, 2·step, … below 360. */
export const anglesOf = (step: number): number[] => Array.from({ length: Math.ceil(360 / step - 1e-9) }, (_, i) => i * step);

export interface AngleResult { angle: number; best: number; x: number; y: number; peaks: Pose[] }

/** One step of a search: the score map for one angle, its best value and its local maxima
 *  (inside the box). */
export function scoreAngle(sc: Scene, s: Settings, dk: Docker, angle: number): { map: Float64Array; res: AngleResult } {
  const map = dk.map(ligandAt(sc, s, angle));
  const pk = localMaxima(map, sc.M, [sc.o, sc.o, sc.o + sc.n, sc.o + sc.n], 8, 2);
  const peaks = pk.map(p => ({ angle, x: p.x, y: p.y, score: p.score }));
  return { map, res: { angle, best: pk.length ? pk[0].score : 0, x: pk.length ? pk[0].x : 0, y: pk.length ? pk[0].y : 0, peaks } };
}

/** A whole search at once (the demo animates the same steps). */
export function search(s: Settings): { sc: Scene; dk: Docker; results: AngleResult[]; top: Pose[] } {
  const sc = makeScene(s), dk = new Docker(sc.a, sc.M);
  const results = anglesOf(s.step).map(t => scoreAngle(sc, s, dk, t).res);
  return { sc, dk, results, top: rankPoses(sc, results) };
}

/** The top five poses over all angles, with non-maximum suppression (3 world units, 20°). */
export function rankPoses(sc: Scene, results: AngleResult[]): Pose[] {
  return topPoses(results.flatMap(r => r.peaks), 5, 3 / sc.h, 20);
}

/** Distance of a pose from the true pose: position (world units) and angle (degrees). */
export function poseError(sc: Scene, p: Pose): { dist: number; dAngle: number } {
  const [x, y] = toWorld(sc, p.x, p.y);
  const d = ((p.angle - TRUE_POSE.angle) % 360 + 540) % 360 - 180;
  return { dist: Math.hypot(x - TRUE_POSE.x, y - TRUE_POSE.y), dAngle: d };
}

/** Is a pose the true fit (within 1.5 world units and 10°)? */
export const isTrueFit = (sc: Scene, p: Pose): boolean => { const e = poseError(sc, p); return e.dist <= 1.5 && Math.abs(e.dAngle) <= 10; };
