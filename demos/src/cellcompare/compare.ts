// The comparison of the cellnets-compare unit without DOM, so the demo (main.ts), the node
// measurement script (tools/figs/cellnets-compare/measure.ts) and the test
// (demos/test/cellnets-compare-test.ts) share it: the seven annotated images and their settings,
// the classical pipeline (smooth, threshold, fill holes, distance transform, h-maxima,
// watershed, minimum size; as in ws-distance/), and the scoring with the evaluation day's
// object matching (lib/eval.ts: IoU table, optimal matching at tau, F1(tau), splits/merges).
import { gaussianBlur, fromArray } from "../lib/imageops";
import { histogram } from "../lib/histogram";
import { otsu } from "../lib/threshold";
import { fillHoles, hMaxima, regionalMaxima } from "../lib/morph";
import { edt } from "../lib/distance";
import { label2d, removeSmall, relabel } from "../lib/label";
import { watershed } from "../lib/watershed";
import { iouTable, match, objectScores, splitsMerges, type IouTable } from "../lib/eval";

export type MethodId = "classical" | "stardist" | "cellpose" | "sam";
export const METHODS: { id: MethodId; label: string; colour: string }[] = [
  { id: "classical", label: "Classical", colour: "#e69f00" },
  { id: "stardist", label: "StarDist", colour: "#56b4e9" },
  { id: "cellpose", label: "Cellpose", colour: "#009e73" },
  { id: "sam", label: "SAM automatic", colour: "#cc79a7" },
];

export interface Sample {
  id: string;
  /** keys in demos/dist/models/index.json */
  file: string;
  labels: string;
  label: string;
  /** Cellpose diameter (pixels): the median nucleus diameter of the annotation, rounded
   *  (a user would measure a few nuclei; same as the cellposenet demo where it has the image). */
  diameter: number;
  /** nuclei darker than the background (bright-field, stained tissue) */
  dark?: boolean;
}

export const SAMPLES: Sample[] = [
  { id: "sparse", file: "images/dsb-fluo-small.png", labels: "images/dsb-fluo-small-labels.png", label: "Sparse small nuclei (256²)", diameter: 13 },
  { id: "dim", file: "images/dsb-fluo-medium.png", labels: "images/dsb-fluo-medium-labels.png", label: "Dim large nuclei (360²)", diameter: 44 },
  { id: "dense", file: "images/dsb-fluo-dense.png", labels: "images/dsb-fluo-dense-labels.png", label: "Dense nuclei (696 × 520)", diameter: 21 },
  { id: "u2os", file: "images/bbbc039-A02_s1.png", labels: "images/bbbc039-A02_s1-labels.png", label: "U2OS nuclei, field 1 (696 × 520)", diameter: 30 },
  { id: "u2os2", file: "images/bbbc039-A06_s6.png", labels: "images/bbbc039-A06_s6-labels.png", label: "U2OS nuclei, field 2 (696 × 520)", diameter: 30 },
  { id: "histology", file: "images/dsb-histology.png", labels: "images/dsb-histology-labels.png", label: "Histology, stained tissue (320 × 256)", diameter: 24, dark: true },
  { id: "brightfield", file: "images/dsb-brightfield.png", labels: "images/dsb-brightfield-labels.png", label: "Bright field (1024²)", diameter: 18, dark: true },
];

/** IoU thresholds of the F1(tau) curves: 0.5, 0.55, ..., 0.95. */
export const TAUS: number[] = Array.from({ length: 10 }, (_, i) => Math.round((0.5 + 0.05 * i) * 100) / 100);

// ---- images ------------------------------------------------------------------------------------

/** Grey image (0..255) from decoded PNG data: mean of the colour channels (16-bit scaled down). */
export function greyFromPixels(data: ArrayLike<number>, w: number, h: number, channels: number, depth = 8): Float32Array {
  const nc = channels >= 3 ? 3 : 1, sc = depth === 16 ? 1 / 257 : 1;
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    let s = 0;
    for (let c = 0; c < nc; c++) s += data[i * channels + c];
    g[i] = (s / nc) * sc;
  }
  return g;
}

/** 255 - v: dark nuclei become bright (what the fluorescence-trained networks expect). */
export function invertGrey(g: Float32Array): Float32Array {
  return g.map((v) => 255 - v);
}

/** Interleaved RGB bytes of a grey image (for the promptable segmenter, which wants colour). */
export function rgbFromGrey(g: Float32Array): Uint8Array {
  const o = new Uint8Array(3 * g.length);
  for (let i = 0; i < g.length; i++) { const v = Math.max(0, Math.min(255, Math.round(g[i]))); o[3 * i] = o[3 * i + 1] = o[3 * i + 2] = v; }
  return o;
}

/** Value at percentile q (0..100) of an array (nearest rank on the sorted copy). */
export function percentileOf(x: ArrayLike<number>, q: number): number {
  const s = Float32Array.from(x as ArrayLike<number>).sort();
  return s[Math.min(s.length - 1, Math.max(0, Math.round((q / 100) * (s.length - 1))))];
}

/** Scale so the 1st percentile is 0 and the 99.8th is 1 (as the StarDist input). */
export function normalise(g: Float32Array): Float32Array {
  const lo = percentileOf(g, 1), hi = percentileOf(g, 99.8);
  const d = hi - lo > 1e-6 ? hi - lo : 1;
  return g.map((v) => (v - lo) / d);
}

// ---- the classical pipeline ---------------------------------------------------------------------

export interface ClassicalParams {
  /** Gaussian smoothing (pixels), 0 = none */
  sigma: number;
  /** "otsu": automatic; "manual": `thr` on the normalised image (0 = 1st, 1 = 99.8th percentile) */
  method: "otsu" | "manual";
  thr: number;
  /** h of the h-maxima of the distance transform (pixels); 0 = all regional maxima */
  h: number;
  /** objects smaller than this (pixels) are removed */
  minSize: number;
}

export const CLASSICAL_DEFAULT: ClassicalParams = { sigma: 1, method: "otsu", thr: 0.3, h: 1, minSize: 15 };

export interface ClassicalResult {
  labels: Int32Array;
  count: number;
  /** threshold used, on the normalised image */
  thr: number;
  mask: Uint8Array;
  ms: number;
}

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** Smooth -> threshold -> fill holes -> distance transform -> h-maxima markers -> watershed
 *  on minus the distance inside the mask -> drop small objects. Input: grey (any scale). */
export function classical(grey: Float32Array, w: number, h: number, p: ClassicalParams): ClassicalResult {
  const t0 = now();
  const n = normalise(grey);
  const sm = p.sigma > 0 ? gaussianBlur(fromArray(w, h, n), p.sigma).data : n;
  let thr = p.thr;
  if (p.method === "otsu") {
    const hist = histogram(sm, { bins: 256, min: 0, max: 1 });
    const r = otsu(hist.counts);
    thr = hist.lo + (r.t + 1) * hist.binWidth;
  }
  const raw = new Uint8Array(w * h);
  for (let i = 0; i < raw.length; i++) raw[i] = sm[i] > thr ? 1 : 0;
  const mask = fillHoles(raw, w, h);
  const dt = edt(mask, w, h);
  const pk = p.h > 0 ? hMaxima(dt, w, h, p.h, 8) : regionalMaxima(dt, w, h, 8);
  for (let i = 0; i < pk.length; i++) if (!mask[i]) pk[i] = 0;
  const markers = label2d(pk, w, h, 8);
  const land = Float32Array.from(dt, (v) => -v);
  const ws = watershed(land, w, h, markers.data, { conn: 8, mask });
  const lab = removeSmall(relabel(ws, w, h), p.minSize);
  return { labels: lab.data, count: lab.count, thr, mask, ms: now() - t0 };
}

// ---- scoring -----------------------------------------------------------------------------------

export interface Score {
  nTrue: number;
  nPred: number;
  /** F1 at each tau of TAUS (optimal matching) */
  f1: number[];
  tp5: number; fp5: number; fn5: number;
  tp7: number; fp7: number; fn7: number;
  splits: number;
  merges: number;
}

/** Labels 1..n without gaps (any label values in, e.g. a network's output). */
export function compact(lab: ArrayLike<number>): Int32Array {
  const map = new Map<number, number>();
  const out = new Int32Array(lab.length);
  for (let i = 0; i < lab.length; i++) {
    const v = lab[i];
    if (v <= 0) continue;
    let k = map.get(v);
    if (k === undefined) { k = map.size + 1; map.set(v, k); }
    out[i] = k;
  }
  return out;
}

export function table(truth: ArrayLike<number>, pred: ArrayLike<number>): IouTable {
  return iouTable(compact(truth), compact(pred));
}

export function score(truth: ArrayLike<number>, pred: ArrayLike<number>): Score {
  const T = table(truth, pred);
  const f1 = TAUS.map((t) => objectScores(match(T, t)).f1);
  const m5 = match(T, 0.5), m7 = match(T, 0.7);
  const sm = splitsMerges(T);
  return {
    nTrue: T.nt, nPred: T.np, f1,
    tp5: m5.tp, fp5: m5.fp, fn5: m5.fn, tp7: m7.tp, fp7: m7.fp, fn7: m7.fn,
    splits: sm.splits.length, merges: sm.merges.length,
  };
}

export type MarkKind = "miss" | "extra" | "merge" | "split";
export interface Mark { kind: MarkKind; x: number; y: number }

/** Error marks at tau = 0.5: missed true objects (not matched and not part of a merge),
 *  extra predictions (not matched and not part of a split), merges (at the merged prediction)
 *  and splits (at the split true object). Positions are centroids. */
export function errorMarks(truth: ArrayLike<number>, pred: ArrayLike<number>, w: number): Mark[] {
  const t = compact(truth), p = compact(pred);
  const T = iouTable(t, p);
  const m = match(T, 0.5), sm = splitsMerges(T);
  const cT = centroidsOf(t, T.nt, w), cP = centroidsOf(p, T.np, w);
  const inMerge = new Set<number>(), inSplit = new Set<number>();
  for (const mg of sm.merges) mg.parts.forEach((i) => inMerge.add(i));
  for (const sp of sm.splits) sp.parts.forEach((j) => inSplit.add(j));
  const marks: Mark[] = [];
  for (const mg of sm.merges) marks.push({ kind: "merge", x: cP[mg.p][0], y: cP[mg.p][1] });
  for (const sp of sm.splits) marks.push({ kind: "split", x: cT[sp.t][0], y: cT[sp.t][1] });
  for (let i = 0; i < T.nt; i++) if (m.matchT[i] < 0 && !inMerge.has(i) && !sm.splits.some((s) => s.t === i)) marks.push({ kind: "miss", x: cT[i][0], y: cT[i][1] });
  for (let j = 0; j < T.np; j++) if (m.matchP[j] < 0 && !inSplit.has(j) && !sm.merges.some((s) => s.p === j)) marks.push({ kind: "extra", x: cP[j][0], y: cP[j][1] });
  return marks;
}

function centroidsOf(lab: Int32Array, n: number, w: number): [number, number][] {
  const sx = new Float64Array(n + 1), sy = new Float64Array(n + 1), c = new Float64Array(n + 1);
  for (let i = 0; i < lab.length; i++) { const l = lab[i]; if (!l) continue; sx[l] += i % w; sy[l] += Math.floor(i / w); c[l]++; }
  const out: [number, number][] = [];
  for (let l = 1; l <= n; l++) out.push([sx[l] / (c[l] || 1), sy[l] / (c[l] || 1)]);
  return out;
}

/** Pixels on the border of a label (4-neighbour differs), for outlines. */
export function outlinePixels(lab: ArrayLike<number>, w: number, h: number): Uint8Array {
  const o = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, v = lab[i];
    if (!v) continue;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || lab[i - 1] !== v || lab[i + 1] !== v || lab[i - w] !== v || lab[i + w] !== v) o[i] = 1;
  }
  return o;
}

/** Images per hour from seconds per image. */
export const perHour = (seconds: number): number => 3600 / seconds;
