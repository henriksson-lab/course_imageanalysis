// The colour cloud: an image and its pixels as a rotatable 3D point cloud (WebGL2 via
// lib/gl3d.ts), each point in its own colour, inside the outline of the sRGB gamut in the
// chosen space (encoded sRGB, linear RGB, XYZ, Y'CbCr, HSV, CIELAB). Switching space morphs
// the cloud. Brush on the image -> those points light up; pick a colour + a radius -> a
// ball in the current space selects points, and their pixels light up in the image.
// Mode "pca": the principal axes of the (encoded) colour cloud and a decorrelation stretch.
//
// Embeddable: mountColourCloud(host, ext) takes extra spaces and scenes, so another day
// (unmixing: optical density) can plug its own space in without touching this file:
//   import { mountColourCloud } from "../colourcloud/cloud";
//   autoMount("unmixcloud", h => mountColourCloud(h, { spaces: { od: {...} } }));
// data-* options: data-scene (objects|he|chart|isolum), data-space (srgb|linear|xyz|ycbcr|
// hsv|lab), data-spaces (list shown, e.g. "srgb hsv lab"), data-mode (explore|pca),
// data-preset, data-noise (0..0.06).
import { htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { GLView, OrbitCamera, viewButtons, LineData } from "../lib/gl3d";
import { RGB, srgbToLinear, linearToSrgb, linearRgbToXyz, xyzToLab, rgbToHsv } from "../lib/colour";
import { jacobiEigen } from "../lib/linalg";
import { Scene, SceneId, SCENE_LABELS, makeScene, withNoise, withShading, sceneToRGBA } from "./scenes";
import { rgbToYcbcr } from "./colourx";

export type V3 = [number, number, number];

/** A colour space for the cloud: position in a box of about [-0.5, 0.5]^3 (y up) from the
 *  linear and the encoded RGB of a pixel. */
export interface SpaceDef {
  label: string;
  /** Short axis description shown under the view. */
  axes: string;
  pos(lin: RGB, enc: RGB): V3;
}

export const SPACES: Record<string, SpaceDef> = {
  srgb: { label: "sRGB (encoded)", axes: "x = R′, y = G′, z = B′ (display values)", pos: (_l, e) => [e[0] - 0.5, e[1] - 0.5, e[2] - 0.5] },
  linear: { label: "linear RGB", axes: "x = R, y = G, z = B (light)", pos: l => [l[0] - 0.5, l[1] - 0.5, l[2] - 0.5] },
  xyz: { label: "XYZ", axes: "x = X, y = Y (luminance), z = Z", pos: l => { const v = linearRgbToXyz(l); return [v[0] / 0.95 - 0.5, v[1] - 0.5, v[2] / 1.09 - 0.5]; } },
  ycbcr: { label: "Y′CbCr", axes: "up = Y′ (luma), x = Cb, z = Cr", pos: (_l, e) => { const v = rgbToYcbcr(255 * e[0], 255 * e[1], 255 * e[2]); return [(v[1] - 128) / 255, v[0] / 255 - 0.5, -(v[2] - 128) / 255]; } },
  hsv: { label: "HSV", axes: "up = V, angle = hue, distance from axis = S·V", pos: (_l, e) => {
    const [h, s, v] = rgbToHsv(e), a = (Number.isFinite(h) ? h : 0) * Math.PI / 180, r = 0.5 * s * v;
    return [r * Math.cos(a), v - 0.5, -r * Math.sin(a)];
  } },
  lab: { label: "CIELAB", axes: "up = L*, x = a* (green–red), z = b* (blue–yellow)", pos: l => {
    const [L, a, b] = xyzToLab(linearRgbToXyz(l));
    return [a / 200, L / 100 - 0.5, -b / 200];
  } },
};

export interface CloudExtensions {
  spaces?: Record<string, SpaceDef>;
  scenes?: Record<string, { label: string; make: () => Scene }>;
  /** Extra presets (label + settings) appended to the row. */
  presets?: { label: string; settings: Partial<Settings> }[];
}

export interface Settings {
  scene: string; space: string; noise: number; shading: number; radius: number;
  grey: boolean; select: "none" | "class" | "ball" | "brush"; cls: number; ref: number; stretch: number; pcaAxes: boolean;
}

const NAME = "colourcloud";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .cc-img { flex: 0 1 270px; min-width: 200px; margin: 0; }
[data-demo="${NAME}"] .cc-img canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; cursor: crosshair; touch-action: none; }
[data-demo="${NAME}"] .cc-gl { flex: 1 1 380px; min-width: 260px; max-width: 470px; }
[data-demo="${NAME}"] .cc-axes, [data-demo="${NAME}"] .cc-credit { font-size: 14px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 420px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13.5px; white-space: pre; }
[data-demo="${NAME}"] .cc-row { display: flex; flex-wrap: wrap; gap: .3em; }
[data-demo="${NAME}"] .cc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .cc-img { flex-basis: 250px; }
.reveal [data-demo="${NAME}"] .cc-gl { flex-basis: 430px; max-width: 450px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Positions of every pixel of a scene in a space (Float32Array, 3 per pixel). */
export function cloudPositions(sc: { r: ArrayLike<number>; g: ArrayLike<number>; b: ArrayLike<number> }, sp: SpaceDef): Float32Array {
  const n = sc.r.length, out = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) {
    const l: RGB = [sc.r[p], sc.g[p], sc.b[p]], e: RGB = [linearToSrgb(Math.max(0, Math.min(1, l[0]))), linearToSrgb(Math.max(0, Math.min(1, l[1]))), linearToSrgb(Math.max(0, Math.min(1, l[2])))];
    const v = sp.pos(l, e);
    out[3 * p] = v[0]; out[3 * p + 1] = v[1]; out[3 * p + 2] = v[2];
  }
  return out;
}

/** The 12 edges of the RGB cube (encoded), each in 24 pieces, mapped into a space. */
export function gamutLines(sp: SpaceDef): LineData {
  const corners: RGB[] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges: [number, number][] = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    const d = corners[i].reduce((s, v, k) => s + Math.abs(v - corners[j][k]), 0);
    if (d === 1) edges.push([i, j]);
  }
  const pos: number[] = [], col: number[] = [];
  const at = (e: RGB): V3 => sp.pos([srgbToLinear(e[0]), srgbToLinear(e[1]), srgbToLinear(e[2])], e);
  for (const [i, j] of edges) {
    const a = corners[i], b = corners[j], N = 24;
    for (let k = 0; k < N; k++) {
      const t0 = k / N, t1 = (k + 1) / N;
      const e0: RGB = [a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]), a[2] + t0 * (b[2] - a[2])];
      const e1: RGB = [a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1]), a[2] + t1 * (b[2] - a[2])];
      pos.push(...at(e0), ...at(e1));
      const m = 0.6 * (e0[0] + e1[0]) / 2, m2 = 0.6 * (e0[1] + e1[1]) / 2, m3 = 0.6 * (e0[2] + e1[2]) / 2;
      col.push(0.25 + m, 0.25 + m2, 0.25 + m3);
    }
  }
  return { positions: Float32Array.from(pos), colors: Float32Array.from(col) };
}

/** PCA of the encoded colour cloud: mean, eigenvalues (descending) and eigenvectors (columns). */
export function colourPCA(enc: { r: ArrayLike<number>; g: ArrayLike<number>; b: ArrayLike<number> }): { mean: V3; values: number[]; vectors: number[][]; cov: number[][] } {
  const n = enc.r.length, ch = [enc.r, enc.g, enc.b];
  const mean: V3 = [0, 0, 0];
  for (let c = 0; c < 3; c++) { let s = 0; for (let p = 0; p < n; p++) s += ch[c][p]; mean[c] = s / n; }
  const C = new Float64Array(9);
  for (let p = 0; p < n; p++) {
    const d = [ch[0][p] - mean[0], ch[1][p] - mean[1], ch[2][p] - mean[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] += d[i] * d[j];
  }
  for (let i = 0; i < 9; i++) C[i] /= n;
  const { values, vectors } = jacobiEigen(C, 3, true);
  const vecs = [0, 1, 2].map(k => [vectors[k], vectors[3 + k], vectors[6 + k]]);
  return { mean, values: Array.from(values), vectors: vecs, cov: [[C[0], C[1], C[2]], [C[3], C[4], C[5]], [C[6], C[7], C[8]]] };
}

/** Decorrelation stretch of encoded RGB by a fraction t in [0, 1]: rotate to the principal
 *  axes, scale axis k by (σ̄/σ_k)^t (σ̄ = RMS of the σ_k), rotate back, clip. */
export function decorrelationStretch(enc: { r: ArrayLike<number>; g: ArrayLike<number>; b: ArrayLike<number> }, t: number,
  pca = colourPCA(enc)): { r: Float32Array; g: Float32Array; b: Float32Array } {
  const n = enc.r.length, U = pca.vectors, sd = pca.values.map(v => Math.sqrt(Math.max(v, 1e-12)));
  const target = Math.sqrt(pca.values.reduce((s, v) => s + Math.max(v, 0), 0) / 3);
  const f = sd.map(s => (target / s) ** t);
  // T = U diag(f) Uᵀ
  const T = [0, 1, 2].map(i => [0, 1, 2].map(j => U[0][i] * f[0] * U[0][j] + U[1][i] * f[1] * U[1][j] + U[2][i] * f[2] * U[2][j]));
  const out = { r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n) }, m = pca.mean;
  const cl = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
  for (let p = 0; p < n; p++) {
    const d = [enc.r[p] - m[0], enc.g[p] - m[1], enc.b[p] - m[2]];
    out.r[p] = cl(m[0] + T[0][0] * d[0] + T[0][1] * d[1] + T[0][2] * d[2]);
    out.g[p] = cl(m[1] + T[1][0] * d[0] + T[1][1] * d[1] + T[1][2] * d[2]);
    out.b[p] = cl(m[2] + T[2][0] * d[0] + T[2][1] * d[1] + T[2][2] * d[2]);
  }
  return out;
}

export function mountColourCloud(host: HTMLElement, ext: CloudExtensions = {}): void {
  injectStyle();
  const spaces: Record<string, SpaceDef> = { ...SPACES, ...(ext.spaces ?? {}) };
  const sceneDefs: Record<string, { label: string; make: () => Scene }> = {};
  for (const id of Object.keys(SCENE_LABELS) as SceneId[]) sceneDefs[id] = { label: SCENE_LABELS[id], make: () => makeScene(id) };
  Object.assign(sceneDefs, ext.scenes ?? {});
  const pcaMode = host.dataset.mode === "pca";
  const shown = (host.dataset.spaces ?? Object.keys(spaces).join(" ")).split(/[\s,]+/).filter(s => spaces[s]);

  const S: Settings = {
    scene: host.dataset.scene && sceneDefs[host.dataset.scene] ? host.dataset.scene : "objects",
    space: host.dataset.space && spaces[host.dataset.space] ? host.dataset.space : "srgb",
    noise: Number(host.dataset.noise ?? 0) || 0, shading: 0, radius: 0.08, grey: true,
    select: "none", cls: 1, ref: -1, stretch: 0, pcaAxes: pcaMode,
  };
  const cache = new Map<string, Scene>();
  const baseScene = (id: string) => { let s = cache.get(id); if (!s) { s = sceneDefs[id].make(); cache.set(id, s); } return s; };

  // ---- layout ----
  const fig = htmlEl("figure", "cc-img qb-figure", "", host);
  htmlEl("figcaption", "", "Image (brush or click)", fig);
  const canvas = htmlEl("canvas", "", "", fig);
  const credit = htmlEl("div", "cc-credit", "", fig);
  const glBox = htmlEl("div", "cc-gl", "", host);
  const cam = new OrbitCamera({ distance: 2.6, azimuth: 0.75, elevation: 0.42 });
  const view = new GLView(glBox, { aspect: 1.08, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam, label: "" }],
    fallback: "The pixels of the image as a point cloud need WebGL2." });
  const axesNote = htmlEl("div", "cc-axes", "", glBox);
  viewButtons(glBox, view, 0, ["front", "top", "iso"]);
  const panel = htmlEl("div", "qb-panel", "", host);

  // scene + space buttons
  htmlEl("div", "cc-lab", "Image", panel);
  const sceneRow = htmlEl("div", "cc-row", "", panel);
  const sceneBtns = Object.keys(sceneDefs).filter(id => !pcaMode || id !== "isolum").map(id => {
    const b = button(sceneRow, sceneDefs[id].label, () => { S.scene = id; S.select = "none"; S.ref = -1; rebuild(true); });
    b.dataset.id = id; return b;
  });
  let spaceBtns: HTMLButtonElement[] = [];
  if (!pcaMode) {
    htmlEl("div", "cc-lab", "Colour space (the cloud morphs)", panel);
    const spaceRow = htmlEl("div", "cc-row", "", panel);
    spaceBtns = shown.map(id => { const b = button(spaceRow, spaces[id].label, () => setSpace(id)); b.dataset.id = id; return b; });
  }
  const noiseS = slider(panel, "noise", 0, 0.06, 0.005, S.noise, v => { S.noise = v; rebuild(false); });
  const shadeS = pcaMode ? null : slider(panel, "shading", 0, 0.9, 0.05, 0, v => { S.shading = v; rebuild(false); });
  const stretchS = pcaMode ? slider(panel, "stretch", 0, 1, 0.05, 0, v => { S.stretch = v; rebuild(false); }) : null;
  const radiusS = pcaMode ? null : slider(panel, "ball radius", 0.02, 0.3, 0.01, S.radius, v => { S.radius = v; if (S.select === "ball") updateSelection(); });
  const toolRow = htmlEl("div", "cc-row", "", panel);
  let tool: "brush" | "pick" = "pick";
  const pickBtn = pcaMode ? null : button(toolRow, "click: pick colour", () => setTool("pick"));
  const brushBtn = pcaMode ? null : button(toolRow, "brush", () => setTool("brush"));
  if (!pcaMode) button(toolRow, "clear", () => { S.select = "none"; S.ref = -1; brush.fill(0); updateSelection(); });
  const greyC = checkbox(panel, "grey axis", S.grey, v => { S.grey = v; updateLines(); });
  const pcaC = pcaMode ? checkbox(panel, "principal axes", true, v => { S.pcaAxes = v; updateLines(); }) : null;
  const read = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", pcaMode
    ? "Axes: ±2 standard deviations along each principal component of the encoded colours. Stretch 1 equalises their variances."
    : "Drag the cloud to rotate. Click a pixel: its coordinates, and a ball around its colour in the current space selects points.", panel);
  void note;

  const presetDefs: { id: string; label: string; s: Partial<Settings> }[] = pcaMode ? [
    { id: "he", label: "H&E: a flat cloud", s: { scene: "he", stretch: 0 } },
    { id: "stretch", label: "decorrelation stretch", s: { scene: "he", stretch: 1 } },
    { id: "objects", label: "objects: a cigar along grey", s: { scene: "objects", stretch: 0 } },
    { id: "objects-stretch", label: "objects, stretched", s: { scene: "objects", stretch: 1 } },
  ] : [
    { id: "hsv", label: "cube → HSV cone", s: { scene: "chart", space: "hsv", noise: 0, shading: 0, select: "none" } },
    { id: "shadow", label: "shadows are rays through black", s: { scene: "objects", space: "linear", noise: 0, shading: 0, select: "class", cls: 1 } },
    { id: "grey", label: "grey pixels scatter in hue", s: { scene: "objects", space: "hsv", noise: 0.03, shading: 0, select: "class", cls: 5 } },
    { id: "lab", label: "Lab: white at the top", s: { scene: "chart", space: "lab", noise: 0, shading: 0, select: "none" } },
    { id: "he", label: "H&E: two stains, a sheet", s: { scene: "he", space: "srgb", noise: 0, shading: 0, select: "class", cls: 1 } },
  ];
  for (const p of ext.presets ?? []) presetDefs.push({ id: p.label, label: p.label, s: p.settings });
  const presetBtns = presets(fig, [
    ...presetDefs.map(p => ({ label: p.label, apply: () => applyPreset(p.s) })),
    { label: "reset", apply: () => applyPreset({ scene: "objects", space: "srgb", noise: 0, shading: 0, select: "none", stretch: 0 }) },
  ]);

  // ---- state ----
  let sc: Scene = baseScene(S.scene);
  let enc: { r: Float32Array; g: Float32Array; b: Float32Array } = { r: new Float32Array(0), g: new Float32Array(0), b: new Float32Array(0) };
  let shownLin = sc;             // what the cloud and the image show (linear)
  let brush = new Uint8Array(0);
  let sel = new Uint8Array(0);
  let posCur: Float32Array = new Float32Array(0);
  let anim = 0;
  const allPts = view.addPoints({ positions: new Float32Array(0) }, { size: 4, shaded: false });
  const selPts = view.addPoints({ positions: new Float32Array(0) }, { size: 7, shaded: true });
  const gamut = view.addLines({ positions: new Float32Array(0) }, { width: 2 });
  const extra = view.addLines({ positions: new Float32Array(0) }, { width: 4 });
  let pca = pcaMode ? colourPCA({ r: [], g: [], b: [] }) : null;

  function setTool(t: "brush" | "pick"): void {
    tool = t;
    pickBtn?.classList.toggle("active", t === "pick");
    brushBtn?.classList.toggle("active", t === "brush");
  }
  setTool("pick");

  function rebuild(newScene: boolean): void {
    sc = baseScene(S.scene);
    let s = withShading(sc, S.shading);
    s = withNoise(s, S.noise);
    const n = s.w * s.h;
    enc = { r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n) };
    for (let p = 0; p < n; p++) { enc.r[p] = linearToSrgb(Math.min(1, Math.max(0, s.r[p]))); enc.g[p] = linearToSrgb(Math.min(1, Math.max(0, s.g[p]))); enc.b[p] = linearToSrgb(Math.min(1, Math.max(0, s.b[p]))); }
    if (pcaMode) {
      pca = colourPCA(enc);
      const st = decorrelationStretch(enc, S.stretch, pca);
      enc = st;
      const lin = { ...s, r: Float32Array.from(st.r, srgbToLinear), g: Float32Array.from(st.g, srgbToLinear), b: Float32Array.from(st.b, srgbToLinear) };
      s = lin;
    }
    shownLin = s;
    if (newScene || brush.length !== n) { brush = new Uint8Array(n); sel = new Uint8Array(n); }
    canvas.width = s.w; canvas.height = s.h;
    credit.textContent = sc.credit;
    sceneBtns.forEach(b => b.classList.toggle("active", b.dataset.id === S.scene));
    const target = cloudPositions(shownLin, spaces[pcaMode ? "srgb" : S.space]);
    if (newScene || posCur.length !== target.length) { cancelAnimationFrame(anim); posCur = target; }
    else posCur = target;
    const colors = new Float32Array(3 * n);
    for (let p = 0; p < n; p++) { colors[3 * p] = enc.r[p]; colors[3 * p + 1] = enc.g[p]; colors[3 * p + 2] = enc.b[p]; }
    pointColors = colors;
    updateLines();
    updateSelection();
  }
  let pointColors: Float32Array = new Float32Array(0);

  function setSpace(id: string, morph = true): void {
    const from = posCur, to = cloudPositions(shownLin, spaces[id]);
    S.space = id;
    spaceBtns.forEach(b => b.classList.toggle("active", b.dataset.id === id));
    updateLines();
    cancelAnimationFrame(anim);
    if (!morph || from.length !== to.length) { posCur = to; drawPoints(); return; }
    const t0 = performance.now(), dur = 900, cur = new Float32Array(to.length);
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / dur), e = t * t * (3 - 2 * t);
      for (let i = 0; i < to.length; i++) cur[i] = from[i] + e * (to[i] - from[i]);
      posCur = t < 1 ? cur : to;
      drawPoints();
      if (t < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  function updateLines(): void {
    const sp = spaces[pcaMode ? "srgb" : S.space];
    view.viewports[0].label = sp.label;
    view.setViewports(view.viewports);
    axesNote.textContent = sp.axes;
    gamut.setData(gamutLines(sp));
    const pos: number[] = [], col: number[] = [];
    if (S.grey) {
      const N = 16;
      for (let k = 0; k < N; k++) {
        const a = k / N, b = (k + 1) / N;
        pos.push(...sp.pos([srgbToLinear(a), srgbToLinear(a), srgbToLinear(a)], [a, a, a]), ...sp.pos([srgbToLinear(b), srgbToLinear(b), srgbToLinear(b)], [b, b, b]));
        col.push(0.1, 0.1, 0.1);
      }
    }
    if (pcaMode && pca && S.pcaAxes) {
      const cols = [[0.85, 0.37, 0.0], [0.58, 0.40, 0.74], [0.17, 0.63, 0.17]];
      const m = pca.mean;
      // after a stretch the axes keep their directions; lengths follow the stretched spread
      const target = Math.sqrt(pca.values.reduce((s, v) => s + Math.max(v, 0), 0) / 3);
      for (let k = 0; k < 3; k++) {
        const sd = Math.sqrt(Math.max(pca.values[k], 0)), len = 2 * sd * (target / Math.max(sd, 1e-9)) ** S.stretch, u = pca.vectors[k];
        pos.push(m[0] - len * u[0] - 0.5, m[1] - len * u[1] - 0.5, m[2] - len * u[2] - 0.5, m[0] + len * u[0] - 0.5, m[1] + len * u[1] - 0.5, m[2] + len * u[2] - 0.5);
        col.push(...cols[k]);
      }
    }
    extra.setData({ positions: Float32Array.from(pos), colors: Float32Array.from(col) });
  }

  function drawPoints(): void {
    const n = posCur.length / 3, any = sel.some(v => v);
    const c = new Float32Array(3 * n);
    if (any) for (let i = 0; i < 3 * n; i++) c[i] = 0.55 * pointColors[i] + 0.45 * 0.82;
    else c.set(pointColors);
    allPts.setData({ positions: posCur, colors: c });
    allPts.set({ size: any ? 3 : 4 });
    const idx: number[] = [];
    for (let p = 0; p < n; p++) if (sel[p]) idx.push(p);
    const sp = new Float32Array(3 * idx.length), scol = new Float32Array(3 * idx.length);
    idx.forEach((p, i) => { sp.set(posCur.subarray(3 * p, 3 * p + 3), 3 * i); scol.set(pointColors.subarray(3 * p, 3 * p + 3), 3 * i); });
    selPts.setData({ positions: sp, colors: scol });
    drawImage();
  }

  function updateSelection(): void {
    const n = sc.w * sc.h;
    sel = new Uint8Array(n);
    if (S.select === "class" && sc.truth) for (let p = 0; p < n; p++) sel[p] = sc.truth[p] === S.cls ? 1 : 0;
    else if (S.select === "brush") sel.set(brush);
    else if (S.select === "ball" && S.ref >= 0) {
      const P = cloudPositions(shownLin, spaces[pcaMode ? "srgb" : S.space]), q = S.ref, r2 = S.radius ** 2;
      for (let p = 0; p < n; p++) {
        const dx = P[3 * p] - P[3 * q], dy = P[3 * p + 1] - P[3 * q + 1], dz = P[3 * p + 2] - P[3 * q + 2];
        sel[p] = dx * dx + dy * dy + dz * dz <= r2 ? 1 : 0;
      }
    }
    drawPoints();
    updateReadout();
  }

  function drawImage(): void {
    const ctx = canvas.getContext("2d")!, id = ctx.createImageData(sc.w, sc.h);
    sceneToRGBA(shownLin, id.data);
    if (sel.some(v => v)) for (let p = 0; p < sel.length; p++) if (!sel[p]) {
      id.data[4 * p] = 0.3 * id.data[4 * p] + 60; id.data[4 * p + 1] = 0.3 * id.data[4 * p + 1] + 60; id.data[4 * p + 2] = 0.3 * id.data[4 * p + 2] + 60;
    }
    if (S.ref >= 0) {
      const x0 = S.ref % sc.w, y0 = Math.floor(S.ref / sc.w);
      for (let d = -5; d <= 5; d++) for (const [x, y] of [[x0 + d, y0], [x0, y0 + d]]) {
        if (Math.abs(d) < 2 || x < 0 || y < 0 || x >= sc.w || y >= sc.h) continue;
        const p = y * sc.w + x; id.data[4 * p] = 255; id.data[4 * p + 1] = 255; id.data[4 * p + 2] = 255;
      }
    }
    ctx.putImageData(id, 0, 0);
  }

  function updateReadout(): void {
    const nSel = sel.reduce((s, v) => s + v, 0);
    const lines: string[] = [];
    if (pcaMode && pca) {
      const tot = pca.values.reduce((s, v) => s + v, 0), sd = (v: number) => Math.sqrt(Math.max(v, 0));
      const corr = (i: number, j: number) => pca!.cov[i][j] / Math.sqrt(pca!.cov[i][i] * pca!.cov[j][j]);
      lines.push("encoded values 0..1, before the stretch");
      lines.push(`σ along PC1, PC2, PC3: ${pca.values.map(v => sd(v).toFixed(3)).join(", ")}`);
      lines.push(`share of PC1: ${(100 * pca.values[0] / tot).toFixed(1)} %`);
      lines.push(`PC1 direction: (${pca.vectors[0].map(v => v.toFixed(2)).join(", ")})`);
      lines.push(`correlation R–G ${corr(0, 1).toFixed(2)}, R–B ${corr(0, 2).toFixed(2)}, G–B ${corr(1, 2).toFixed(2)}`);
      read.textContent = lines.join("\n");
      return;
    }
    if (S.ref >= 0) {
      const p = S.ref, l: RGB = [shownLin.r[p], shownLin.g[p], shownLin.b[p]], e: RGB = [enc.r[p], enc.g[p], enc.b[p]];
      const xyz = linearRgbToXyz(l), lab = xyzToLab(xyz), hsv = rgbToHsv(e), y = rgbToYcbcr(255 * e[0], 255 * e[1], 255 * e[2]);
      const f = (v: number[], d = 3) => v.map(x => Number.isFinite(x) ? x.toFixed(d) : "undef.").join(", ");
      lines.push(`pixel (${p % sc.w}, ${Math.floor(p / sc.w)})`);
      lines.push(`sRGB 8-bit  ${e.map(v => Math.round(255 * v)).join(", ")}`);
      lines.push(`linear RGB  ${f(l)}`);
      lines.push(`XYZ         ${f(xyz)}`);
      lines.push(`Y′CbCr      ${f(y, 1)}`);
      lines.push(`HSV         ${Number.isFinite(hsv[0]) ? hsv[0].toFixed(1) + "°" : "undef."}, ${hsv[1].toFixed(3)}, ${hsv[2].toFixed(3)}`);
      lines.push(`Lab         ${f(lab, 1)}`);
    } else lines.push("click a pixel to see its coordinates");
    lines.push(`selected: ${nSel} of ${sel.length} pixels`);
    read.textContent = lines.join("\n");
  }

  // pointer on the image: pick or brush
  const toPix = (ev: PointerEvent): number => {
    const r = canvas.getBoundingClientRect();
    const x = Math.floor((ev.clientX - r.left) / r.width * sc.w), y = Math.floor((ev.clientY - r.top) / r.height * sc.h);
    return x >= 0 && y >= 0 && x < sc.w && y < sc.h ? y * sc.w + x : -1;
  };
  const paintAt = (p: number) => {
    const x0 = p % sc.w, y0 = Math.floor(p / sc.w), R = 4;
    for (let y = Math.max(0, y0 - R); y <= Math.min(sc.h - 1, y0 + R); y++) for (let x = Math.max(0, x0 - R); x <= Math.min(sc.w - 1, x0 + R); x++)
      if ((x - x0) ** 2 + (y - y0) ** 2 <= R * R) brush[y * sc.w + x] = 1;
  };
  canvas.addEventListener("pointerdown", ev => {
    if (pcaMode) return;
    ev.preventDefault();
    const p = toPix(ev);
    if (p < 0) return;
    if (tool === "pick") { S.ref = p; S.select = "ball"; updateSelection(); return; }
    canvas.setPointerCapture(ev.pointerId);
    if (S.select !== "brush") { brush.fill(0); S.select = "brush"; }
    paintAt(p); updateSelection();
    const move = (e: PointerEvent) => { const q = toPix(e); if (q >= 0) { paintAt(q); updateSelection(); } };
    const up = () => { canvas.removeEventListener("pointermove", move); canvas.removeEventListener("pointerup", up); };
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
  });

  function applyPreset(p: Partial<Settings>): void {
    const sceneChanged = p.scene !== undefined && p.scene !== S.scene;
    Object.assign(S, p);
    if (p.select === undefined && sceneChanged) S.select = "none";
    if (p.ref === undefined) S.ref = -1;
    noiseS.set(S.noise); shadeS?.set(S.shading); stretchS?.set(S.stretch); radiusS?.set(S.radius);
    greyC.checked = S.grey; if (pcaC) pcaC.checked = S.pcaAxes;
    const target = S.space;
    if (!pcaMode && p.space !== undefined && !sceneChanged) {
      rebuild(false); setSpace(target);
    } else if (!pcaMode && sceneChanged && p.space && p.space !== "srgb") {
      // show the scene in sRGB first, then morph into the preset's space
      S.space = "srgb"; rebuild(true); spaceBtns.forEach(b => b.classList.toggle("active", b.dataset.id === "srgb"));
      setTimeout(() => setSpace(target), 350);
    } else { rebuild(sceneChanged); spaceBtns.forEach(b => b.classList.toggle("active", b.dataset.id === S.space)); }
  }

  rebuild(true);
  spaceBtns.forEach(b => b.classList.toggle("active", b.dataset.id === S.space));
  const pre = host.dataset.preset;
  if (pre) {
    const pd = presetDefs.find(d => d.id === pre);
    if (pd) { applyPreset(pd.s); presetBtns[presetDefs.indexOf(pd)]?.classList.add("active"); }
  }
}
