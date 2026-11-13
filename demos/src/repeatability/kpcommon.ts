// Shared parts of the keypoints-day demos (repeatability, descriptorlab, featurematch,
// featurealign): the scenes (synthetic, repetitive, micrograph) with views under a known
// transform, the detect → orient → describe pipeline with caching, keypoint drawing on an
// ImgView, and a coalescing scheduler for heavy recomputation ("computing…").
import { Grey, resize } from "../lib/imageops";
import { loadCell } from "../lib/cell";
import * as K from "../lib/keypoints";
import * as D from "../lib/descriptors";
import { featureScene, gridScene, viewPair, viewOf, ViewPairSpec, ViewOptions, Transform2D, invert, mapOf } from "../lib/transforms2d";
import { ImgView } from "../canny/imgview";
import { svgEl } from "../lib/ui";

export type SceneKind = "scene" | "grid" | "cell";
export const SCENE_SIZE = 512;
export const VIEW = 256;

const scenes = new Map<SceneKind, Grey>();
const pending = new Map<SceneKind, Promise<Grey>>();

/** The 512² scene (the micrograph is the 256² image upsampled 2× by bilinear interpolation). */
export function getScene(kind: SceneKind): Promise<Grey> {
  const s = scenes.get(kind);
  if (s) return Promise.resolve(s);
  let p = pending.get(kind);
  if (!p) {
    p = kind === "cell" ? loadCell(256).then(g => resize(g, SCENE_SIZE, SCENE_SIZE, "bilinear"))
      : Promise.resolve(kind === "grid" ? gridScene({ size: SCENE_SIZE }) : featureScene({ size: SCENE_SIZE, seed: 11 }));
    p = p.then(g => { scenes.set(kind, g); return g; });
    pending.set(kind, p);
  }
  return p;
}

export interface ViewPairResult { v1: Grey; v2: Grey; T12: Transform2D; T21: Transform2D; map12: K.PointMap; map21: K.PointMap; zoom: number }

const v1Cache = new Map<SceneKind, Grey>();
/** View 1 (centre crop, cached) and view 2 under the spec and lighting. */
export function makeViews(kind: SceneKind, scene: Grey, spec: ViewPairSpec, light: ViewOptions = {}): ViewPairResult {
  const { V1, V2, T12 } = viewPair(SCENE_SIZE, VIEW, spec);
  let v1 = v1Cache.get(kind);
  if (!v1) { v1 = viewOf(scene, V1, { size: VIEW }); v1Cache.set(kind, v1); }
  const v2 = viewOf(scene, V2, { size: VIEW, seed: 5, ...light });
  const T21 = invert(T12);
  return { v1, v2, T12, T21, map12: mapOf(T12), map21: mapOf(T21), zoom: spec.zoom ?? 1 };
}

export type Detector = "harris" | "harrisms" | "dog" | "fast";
export const DETECTOR_NAMES: Record<Detector, string> = { harris: "Harris (one σ)", harrisms: "Harris multi-scale", dog: "DoG", fast: "FAST" };

export interface Features { pyr: K.DoGPyramid; kps: K.Keypoint[]; ms: number }

const featCache = new WeakMap<Grey, Map<string, Features>>();
/** Detect keypoints (strongest n), cached per image and settings. */
export function detect(img: Grey, det: Detector, n: number, extra = ""): Features {
  const key = `${det}|${n}|${extra}`;
  let m = featCache.get(img);
  if (!m) { m = new Map(); featCache.set(img, m); }
  const hit = m.get(key);
  if (hit) return hit;
  const t = performance.now();
  const pyr = K.dogPyramid(img, { S: 3 });
  let kps: K.Keypoint[];
  if (det === "dog") kps = K.detectDoG(img, { contrast: 3, maxPoints: n }, pyr).kps;
  else if (det === "harris") kps = K.detectHarris(img, 1.5, { maxPoints: n, rel: 0.002 });
  else if (det === "harrisms") kps = K.detectHarrisMultiScale(img, undefined, { maxPoints: n, rel: 0.002 });
  else kps = K.detectFAST(img, 20, 9, { maxPoints: n });
  const f = { pyr, kps, ms: performance.now() - t };
  m.set(key, f);
  return f;
}

export type DescKind = "hist" | "binary" | "patch";
export interface DescOptions { rotate?: boolean; normalise?: boolean; clip?: number; soft?: boolean; root?: boolean }
export const PATTERN = D.binaryPattern(7, 256, 32);

/** Orient (gradient histogram, or intensity centroid for binary) and describe. */
export function describe(f: Features, kind: DescKind, o: DescOptions = {}): { kps: K.Keypoint[]; desc: D.Desc[] } {
  const rot = o.rotate !== false;
  const kps = !rot ? f.kps.map(k => ({ ...k, theta: 0 })) : kind === "binary" ? D.centroidOrientations(f.pyr, f.kps) : K.assignOrientations(f.pyr, f.kps);
  return { kps, desc: describeKps(f, kps, kind, o) };
}

/** Descriptors of keypoints that already carry their θ. */
export function describeKps(f: Features, kps: K.Keypoint[], kind: DescKind, o: DescOptions = {}): D.Desc[] {
  const rot = o.rotate !== false;
  return kps.map(k => kind === "binary" ? D.binaryDescriptor(f.pyr, k, PATTERN, { steer: rot })
    : kind === "patch" ? D.patchDescriptor(f.pyr, k, { rotate: rot })
    : D.gradHistDescriptor(f.pyr, k, { rotate: rot, normalise: o.normalise !== false, clip: o.clip ?? 0.2, soft: o.soft !== false, root: !!o.root }));
}

/** Draw a keypoint as a circle of radius 2σ with a tick along θ. */
export function drawKp(view: ImgView, kp: K.Keypoint, color: string, o: { fill?: string; tick?: boolean; width?: number; r?: number } = {}): void {
  const r = o.r ?? Math.max(2, 2 * kp.sigma), cx = kp.x + 0.5, cy = kp.y + 0.5;
  view.add("circle", { cx, cy, r, fill: o.fill ?? "none", stroke: color, "stroke-width": o.width ?? 1.6 });
  if (o.tick !== false) view.add("line", { x1: cx, y1: cy, x2: cx + r * Math.cos(kp.theta), y2: cy - r * Math.sin(kp.theta), stroke: color, "stroke-width": o.width ?? 1.6 });
}

/** Run `fn` once per animation frame at most; shows `busy` while waiting. */
export function scheduler(busy: (on: boolean) => void): (fn: () => void) => void {
  let next: (() => void) | null = null, queued = false;
  return (fn: () => void) => {
    next = fn;
    if (queued) return;
    queued = true; busy(true);
    requestAnimationFrame(() => setTimeout(() => {
      queued = false;
      const f = next; next = null;
      try { f?.(); } finally { busy(false); }
    }, 0));
  };
}

/** A line between two image panels placed side by side in one SVG (image coordinates of each panel). */
export function matchLine(svg: SVGSVGElement, a: [number, number], b: [number, number], gap: number, color: string, width = 1.4, opacity = 0.85): SVGLineElement {
  const l = svgEl("line", { x1: a[0] + 0.5, y1: a[1] + 0.5, x2: VIEW + gap + b[0] + 0.5, y2: b[1] + 0.5, "stroke-width": width, "vector-effect": "non-scaling-stroke" }, svg);
  l.style.stroke = color; l.style.opacity = String(opacity);
  return l;
}

export const GREEN = "#2ca02c", RED = "#e8402a", YELLOW = "#ffd400", CYAN = "#22d3ee", ORANGE = "#ff9f1c", BLUE = "#5b8ff9", GREY = "#9aa0a6";

export const COMMON_CSS = `
.qb-demo .kp-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.2em; }
.reveal .qb-demo .kp-busy { font-size: 16px; }
`;

/** Two images side by side on one canvas, with an SVG overlay in pixel coordinates (view 2 offset by VIEW + gap). */
export class PairView {
  readonly fig: HTMLElement;
  readonly cap: HTMLElement;
  readonly wrap: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  readonly svg: SVGSVGElement;
  readonly gap = 12;
  constructor(parent: Element, caption = "") {
    this.fig = document.createElement("figure"); this.fig.className = "ev-fig kp-pairview"; parent.appendChild(this.fig);
    this.cap = document.createElement("figcaption"); this.cap.textContent = caption; this.fig.appendChild(this.cap);
    this.wrap = document.createElement("div"); this.wrap.className = "ev-wrap"; this.fig.appendChild(this.wrap);
    const W = 2 * VIEW + this.gap;
    this.wrap.style.aspectRatio = `${W} / ${VIEW}`;
    this.canvas = document.createElement("canvas"); this.canvas.className = "ev-canvas"; this.canvas.width = W; this.canvas.height = VIEW; this.wrap.appendChild(this.canvas);
    this.svg = svgEl("svg", { class: "ev-svg", viewBox: `0 0 ${W} ${VIEW}`, preserveAspectRatio: "none" }, this.wrap);
  }
  /** Draw both images (grey, each scaled to 0..255 by `lo`/`hi`). */
  images(a: Grey, b: Grey, lo = 0, hi = 255): void {
    const W = 2 * VIEW + this.gap, id = new ImageData(W, VIEW);
    id.data.fill(255);
    for (const [img, ox] of [[a, 0], [b, VIEW + this.gap]] as [Grey, number][]) {
      for (let y = 0; y < VIEW; y++) for (let x = 0; x < VIEW; x++) {
        const v = Math.max(0, Math.min(255, 255 * (img.data[y * VIEW + x] - lo) / (hi - lo))), k = 4 * (y * W + ox + x);
        id.data[k] = id.data[k + 1] = id.data[k + 2] = v; id.data[k + 3] = 255;
      }
    }
    this.canvas.getContext("2d")!.putImageData(id, 0, 0);
  }
  clear(): void { this.svg.replaceChildren(); }
  /** A clip path covering one of the two views (for outlines that leave it); returns "url(#id)". */
  clip(view: 1 | 2): string {
    const g = globalThis as unknown as { __kpClip?: number };
    g.__kpClip = (g.__kpClip ?? 0) + 1;
    const id = `kp-clip-${g.__kpClip}`, defs = svgEl("defs", {}, this.svg), cp = svgEl("clipPath", { id }, defs);
    svgEl("rect", { x: view === 2 ? VIEW + this.gap : 0, y: 0, width: VIEW, height: VIEW }, cp);
    return `url(#${id})`;
  }
  /** Image coordinates of a point in view 1 or 2 → overlay coordinates. */
  at(view: 1 | 2, x: number, y: number): [number, number] { return [x + 0.5 + (view === 2 ? VIEW + this.gap : 0), y + 0.5]; }
  add<K2 extends keyof SVGElementTagNameMap>(tag: K2, attrs: Record<string, string | number>): SVGElementTagNameMap[K2] {
    return svgEl(tag, { "vector-effect": "non-scaling-stroke", ...attrs }, this.svg);
  }
  line(a: [number, number], b: [number, number], color: string, width = 1.4, opacity = 0.85): void {
    const l = this.add("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], "stroke-width": width });
    l.style.stroke = color; l.style.opacity = String(opacity);
  }
  dot(p: [number, number], r: number, color: string, fill = "none", width = 1.6): void {
    this.add("circle", { cx: p[0], cy: p[1], r, fill, stroke: color, "stroke-width": width });
  }
  /** Click: which view and the image coordinates. */
  onPick(cb: (view: 1 | 2, x: number, y: number) => void): void {
    this.wrap.style.cursor = "crosshair";
    this.wrap.addEventListener("pointerdown", ev => {
      const r = this.wrap.getBoundingClientRect(), W = 2 * VIEW + this.gap;
      const X = (ev.clientX - r.left) / r.width * W, Y = (ev.clientY - r.top) / r.height * VIEW;
      if (X < VIEW) cb(1, X, Y); else if (X >= VIEW + this.gap) cb(2, X - VIEW - this.gap, Y);
    });
  }
}
