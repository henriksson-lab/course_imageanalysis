// Decode mode of the starrays demo: a synthetic 128 × 128 nuclei scene with ground truth and a
// SIMULATED StarDist output (no network): on a grid of one cell per 2 × 2 pixels, the exact
// objectness target (normalised distance to the background) times a brightness factor (a dim
// nucleus gets a lower peak, debris specks a low one) plus smooth noise, and the exact ray
// lengths to the object's border with noise. Then the real decoding: probability threshold →
// one polygon per remaining cell → polygon NMS → labels, scored against the truth with the
// evaluation day's matching (lib/eval.ts) at τ = 0.5. No DOM.
import { rng } from "../lib/ui";
import { createGrey, gaussianBlur } from "../lib/imageops";
import { match, objectScores, type IouTable, type ObjectScores } from "../lib/eval";
import { objectness, raysAt, makeCandidate, polygonNms, paintLabels, type Candidate } from "./geom";
import type { NmsResult } from "../lib/boxes";
import { label2d } from "../lib/label";

export const SN = 128;
export const GRID = 2;

interface El { x: number; y: number; a: number; b: number; angle: number; kind: "nucleus" | "dim" | "debris" }

// Nuclei (true objects 1..10), then three debris specks (not in the ground truth).
const ELLIPSES: El[] = [
  { x: 28, y: 30, a: 11, b: 10, angle: 0.2, kind: "nucleus" },      // 1 \ touching pair
  { x: 48.5, y: 33, a: 11, b: 9.5, angle: -0.3, kind: "nucleus" },  // 2 /
  { x: 95, y: 24, a: 13, b: 10, angle: 0.4, kind: "nucleus" },      // 3
  { x: 103, y: 62, a: 10, b: 9, angle: 0, kind: "dim" },            // 4 dim
  { x: 34, y: 76, a: 17, b: 7.5, angle: 0.55, kind: "nucleus" },    // 5 elongated
  { x: 68, y: 63, a: 9, b: 9, angle: 0, kind: "nucleus" },          // 6
  { x: 66, y: 100, a: 12, b: 9, angle: 0.25, kind: "nucleus" },     // 7 \ touching pair
  { x: 85, y: 108, a: 11, b: 8.5, angle: -0.6, kind: "nucleus" },   // 8 /
  { x: 22, y: 112, a: 9, b: 8, angle: 0, kind: "nucleus" },         // 9
  { x: 112, y: 104, a: 8, b: 7, angle: 1.2, kind: "nucleus" },      // 10
  { x: 63, y: 12, a: 2.6, b: 2.2, angle: 0, kind: "debris" },
  { x: 120, y: 40, a: 2.4, b: 2.4, angle: 0, kind: "debris" },
  { x: 9, y: 54, a: 2.6, b: 2.3, angle: 0.5, kind: "debris" },
];
/** Objectness peak per kind (what the simulated network is sure of). */
const PEAK = { nucleus: 1, dim: 0.62, debris: 0.4 };

export interface Scene {
  /** All objects incl. debris (1..all), used to simulate the network. */
  objects: Int32Array; nAll: number;
  /** Ground truth: nuclei only (1..nTrue). */
  truth: Int32Array; nTrue: number;
  kinds: El["kind"][];
  image: Float32Array;    // 0..255
  target: Float32Array;   // exact objectness target (debris included), 0..1
  /** Groups of touching objects (4-connected foreground), for the contact blur. */
  groups: Int32Array;
}

let cached: Scene | null = null;

export function makeScene(): Scene {
  if (cached) return cached;
  const N = SN, objects = new Int32Array(N * N), best = new Float64Array(N * N).fill(1);
  ELLIPSES.forEach((e, k) => {
    const c = Math.cos(e.angle), s = Math.sin(e.angle);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const dx = x + 0.5 - e.x, dy = y + 0.5 - e.y, u = c * dx + s * dy, v = -s * dx + c * dy;
      const d = (u / e.a) ** 2 + (v / e.b) ** 2, i = y * N + x;
      if (d <= 1 && (objects[i] === 0 || d < best[i])) { objects[i] = k + 1; best[i] = d; }
    }
  });
  const kinds = ELLIPSES.map(e => e.kind);
  const nTrue = kinds.filter(k => k !== "debris").length;
  const truth = Int32Array.from(objects, l => (l && kinds[l - 1] !== "debris" ? l : 0));
  const r = rng(7), g = createGrey(N, N);
  for (let i = 0; i < N * N; i++) {
    const l = objects[i];
    g.data[i] = l ? (kinds[l - 1] === "dim" ? 85 : kinds[l - 1] === "debris" ? 120 : 175) : 30;
  }
  const b = gaussianBlur(g, 1);
  const image = Float32Array.from(b.data, v => Math.max(0, Math.min(255, v + 7 * r.norm())));
  const groups = label2d(Uint8Array.from(objects, l => (l ? 1 : 0)), N, N, 4).data;
  cached = { objects, nAll: ELLIPSES.length, truth, nTrue, kinds, image, target: objectness(objects, N, N), groups };
  return cached;
}

/** A smooth seeded random field with unit standard deviation. */
function smoothField(seed: number, sigma: number): Float32Array {
  const g = createGrey(SN, SN), r = rng(seed);
  for (let i = 0; i < g.data.length; i++) g.data[i] = r.norm();
  const b = gaussianBlur(g, sigma);
  let s = 0; for (const v of b.data) s += v * v;
  const k = 1 / Math.sqrt(s / b.data.length);
  return Float32Array.from(b.data, v => v * k);
}

export interface Sim {
  nRays: number; noise: number; contact: number;
  /** Simulated objectness at full resolution (for display). */
  prob: Float32Array;
  /** Every grid cell with probability ≥ MIN_PROB as a candidate polygon (any order). */
  cands: Candidate[];
}

export const MIN_PROB = 0.05;
const simCache = new Map<string, Sim>();

/**
 * The simulated network output for n rays, a noise level 0..1 and a contact blur 0..1 (cached).
 * Contact blur: the network is unsure where touching nuclei meet, so its rays reach a fraction
 * `contact` of the way from the true border towards the far border of the touching group.
 */
export function simulate(nRays: number, noise: number, contact = 0.2): Sim {
  const key = `${nRays}|${noise.toFixed(3)}|${contact.toFixed(3)}`;
  const hit = simCache.get(key);
  if (hit) return hit;
  const sc = makeScene(), N = SN;
  const f = smoothField(11, 2.5);
  const prob = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const l = sc.objects[i];
    const peak = l ? PEAK[sc.kinds[l - 1]] : 0;
    prob[i] = Math.max(0, Math.min(1, peak * sc.target[i] + 0.12 * noise * f[i]));
  }
  const cands: Candidate[] = [];
  for (let gy = 0; gy < N / GRID; gy++) for (let gx = 0; gx < N / GRID; gx++) {
    const x = gx * GRID, y = gy * GRID, p = y * N + x, cx = x + 0.5, cy = y + 0.5;
    if (prob[p] < MIN_PROB) continue;
    const l = sc.objects[p];
    let rays: Float64Array;
    if (l) {
      rays = raysAt(sc.objects, N, N, cx, cy, nRays, l);
      if (contact > 0) {
        const ru = raysAt(sc.groups, N, N, cx, cy, nRays, sc.groups[p]);
        for (let k = 0; k < nRays; k++) rays[k] += contact * (ru[k] - rays[k]);
      }
    }
    else rays = new Float64Array(nRays).fill(1.5);   // background cell: a tiny polygon
    // multiplicative noise, smooth around the polygon, seeded per grid cell
    const r = rng(1000 + p);
    const g0 = Array.from({ length: nRays }, () => r.norm());
    for (let k = 0; k < nRays; k++) {
      const gk = (g0[(k + nRays - 1) % nRays] + 2 * g0[k] + g0[(k + 1) % nRays]) / Math.sqrt(6);
      rays[k] = Math.max(0.5, rays[k] * (1 + 0.25 * noise * gk));
    }
    cands.push(makeCandidate(cx, cy, prob[p], rays, N, N));
  }
  const sim = { nRays, noise, contact, prob, cands };
  if (simCache.size > 12) simCache.clear();
  simCache.set(key, sim);
  return sim;
}

export interface Decoded {
  /** Indices into sim.cands of the candidates above the probability threshold. */
  above: number[];
  /** NMS on those (indices into `above`). */
  nms: NmsResult & { iouComputed: number };
  /** Kept candidates (indices into sim.cands), in kept order. */
  kept: number[];
  labels: Int32Array;
  scores: ObjectScores;
  /** For each kept polygon: the true nucleus it matches at τ = 0.5 (0 = none, a false positive). */
  matchOf: number[];
  /** True nuclei without a match (false negatives). */
  missed: number[];
}

/** Probability threshold (keep cells with prob > probT), polygon NMS at nmsT, scoring. */
export function decode(sim: Sim, probT: number, nmsT: number): Decoded {
  const sc = makeScene(), N = SN;
  const above: number[] = [];
  sim.cands.forEach((c, i) => { if (c.score > probT) above.push(i); });
  const sub = above.map(i => sim.cands[i]);
  const nms = polygonNms(sub, nmsT, N * N);
  const kept = nms.keep.map(i => above[i]);
  const labels = paintLabels(sim.cands, kept, N * N);
  // objects = the kept polygons themselves (so duplicates count, even where painted over)
  const nt = sc.nTrue, np = kept.length;
  const inter = new Int32Array(nt * np), areaT = new Int32Array(nt), areaP = new Int32Array(np);
  for (const l of sc.truth) if (l) areaT[l - 1]++;
  kept.forEach((ci, j) => {
    const pix = sim.cands[ci].pix;
    areaP[j] = pix.length;
    for (const q of pix) { const t = sc.truth[q]; if (t) inter[(t - 1) * np + j]++; }
  });
  const iou = new Float64Array(nt * np);
  for (let i = 0; i < nt; i++) for (let j = 0; j < np; j++) {
    const c = inter[i * np + j];
    if (c) iou[i * np + j] = c / (areaT[i] + areaP[j] - c);
  }
  const T: IouTable = { nt, np, inter, areaT, areaP, iou };
  const m = match(T, 0.5);
  const matchOf = new Array(np).fill(0), hit = new Uint8Array(nt);
  for (const q of m.pairs) { matchOf[q.p] = q.t + 1; hit[q.t] = 1; }
  const missed: number[] = [];
  for (let t = 0; t < nt; t++) if (!hit[t]) missed.push(t + 1);
  return { above, nms, kept, labels, scores: objectScores(m), matchOf, missed };
}
