// The geometry of mixed pixels. A 3-channel synthetic image (the day's cell with three
// components; or the optical densities of a synthetic H&E slide) and its pixels as a
// rotatable point cloud in channel space (WebGL2 via lib/gl3d.ts). The component spectra
// are arrows from the origin; without noise every pixel lies in the cone they span
// (translucent faces). "normalise" divides every pixel by its total: the cone becomes a
// simplex (a triangle). Right: that triangle in 2D, where the endmembers can be dragged.
// Click a pixel in the image: its least-squares, NNLS and fully constrained (≥ 0, sum to one)
// solutions as points in both views and as abundance bars.
// Auto-mounts on <div data-demo="unmixcloud">. Options: data-scene (cell|he), data-preset
// (clean|noise|nnls|normalise|wrong|fourth), data-mode="2d" (no 3D view: SVG only).
import { htmlEl, slider, presets, checkbox, svgEl, draggable, svgPoint, autoMount } from "../lib/ui";
import { GLView, OrbitCamera, viewButtons } from "../lib/gl3d";
import { RGB } from "../lib/colour";
import * as U from "../lib/unmix";
import { poisson, rng } from "../lib/random";
import { injectStyle, canvasFigure, canvasPixel, fmt, COMMON_CSS } from "../lsunmix/common";

const NAME = "unmixcloud";
const CSS = COMMON_CSS + `
[data-demo="${NAME}"] { gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .uc-left { flex: 0 1 190px; min-width: 150px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .uc-gl { flex: 1 1 340px; min-width: 240px; max-width: 420px; }
[data-demo="${NAME}"] .uc-axes { font-size: 14px; color: var(--qb-muted); }
[data-demo="${NAME}"] .uc-tri { flex: 1 1 300px; min-width: 220px; max-width: 380px; }
[data-demo="${NAME}"] .uc-tri svg { touch-action: none; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .3em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .uc-left { flex: 0 0 170px; }
.reveal [data-demo="${NAME}"] .uc-gl { flex: 0 0 330px; }
.reveal [data-demo="${NAME}"] .uc-tri { flex: 0 0 300px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 300px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .35em; }
.reveal [data-demo="${NAME}"] .uc-axes { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-check, .reveal [data-demo="${NAME}"] .qb-slider { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
`;

type V3 = [number, number, number];
const N = 128, P = N * N;
// 3-channel spectra (fractions of the collected photons, columns sum to 1)
const CELL_E: V3[] = [[0.70, 0.25, 0.05], [0.20, 0.65, 0.15], [0.05, 0.25, 0.70]];
const CELL_AUTO: V3 = [0.50, 0.47, 0.03];
const CELL_NAMES = ["nucleus", "mitochondria", "membrane"];
const SHORT = ["nucl.", "mito.", "membr."], HE_SHORT = ["H", "E"];
const HE_NAMES = ["hematoxylin", "eosin"];
const ECOL: RGB[] = [[0.25, 0.45, 1], [0.1, 0.75, 0.25], [0.95, 0.2, 0.75]];
const HECOL: RGB[] = [[0.35, 0.2, 0.75], [0.9, 0.35, 0.55]];
const MCOL = { ls: "#d62728", nnls: "#ff7f0e", fcls: "#2ca02c" };
const css = (c: RGB) => `rgb(${Math.round(255 * c[0])},${Math.round(255 * c[1])},${Math.round(255 * c[2])})`;

interface St { noise: boolean; logPhot: number; normalise: boolean; faces: boolean; fourth: boolean; E: V3[] }

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const he = host.dataset.scene === "he", flat = host.dataset.mode === "2d";
  const K = he ? 2 : 3, names = he ? HE_NAMES : CELL_NAMES, ecol = he ? HECOL : ECOL;
  const trueE: V3[] = he ? [U.unit3(U.STAINS.H), U.unit3(U.STAINS.E)].map(v => { const s = v[0] + v[1] + v[2]; return [v[0] / s, v[1] / s, v[2] / s] as V3; }) : CELL_E.map(e => [...e] as V3);
  const st: St = { noise: true, logPhot: Math.log10(300), normalise: false, faces: true, fourth: false, E: trueE.map(e => [...e] as V3) };
  let pick = -1;

  // ---- abundance maps (truth) ----
  const sc = U.cellScene(N, 3);
  const heSc = he ? U.stainScene(N, 7, "he") : null;
  const abund = (k: number, p: number): number => {
    if (heSc) {
      // concentrations along unit stain vectors -> along sum-normalised vectors
      const v = U.unit3(k === 0 ? U.STAINS.H : U.STAINS.E);
      return heSc.conc[k][p] * (v[0] + v[1] + v[2]);
    }
    return k === 0 ? 0.9 * sc.nucleus[p] : k === 1 ? sc.mito[p] : 0.8 * sc.membrane[p];
  };
  // ---- data: Y (3 × P) in "abundance units" (counts / photons) ----
  let Y = new Float32Array(3 * P);
  function simulate(): void {
    const r = rng(11), ph = 10 ** st.logPhot;
    Y = new Float32Array(3 * P);
    for (let p = 0; p < P; p++) {
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let k = 0; k < K; k++) v += trueE[k][c] * abund(k, p);
        if (!he) v += 0.004 + (st.fourth ? CELL_AUTO[c] * (sc.vesicles[p] + 0.3 * sc.auto[p]) : 0);   // granules: pure 4th component
        if (st.noise) v = he ? v + 0.02 * r.norm() : poisson(r, ph * v) / ph;
        Y[c * P + p] = v;
      }
    }
  }
  let vmax = 1;

  // ---- layout ----
  const left = htmlEl("div", "uc-left", "", host);
  const img = canvasFigure(left, "image (click a pixel)", N, N);
  img.canvas.style.cursor = "crosshair";
  const bars = svgEl("svg", { viewBox: "0 0 240 190", width: "100%", class: "qb-plot-svg" }, left);
  let view: GLView | null = null;
  let pts: ReturnType<GLView["addPoints"]> | null = null, lines: ReturnType<GLView["addLines"]> | null = null, faces: ReturnType<GLView["addMesh"]> | null = null, marks: ReturnType<GLView["addPoints"]> | null = null;
  if (!flat) {
    const glBox = htmlEl("div", "uc-gl", "", host);
    const cam = new OrbitCamera({ distance: 3.3, azimuth: 0.8, elevation: 0.45, target: [-0.1, -0.1, -0.1] });
    view = new GLView(glBox, { aspect: 1, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam, label: "channel space" }], fallback: "The 3D view needs WebGL2; the triangle on the right shows the same data, normalised." });
    htmlEl("div", "uc-axes", he ? "axes: OD red (x), green (y, up), blue (z)" : "axes: channel 1 (x), 2 (y, up), 3 (z)", glBox);
    viewButtons(glBox, view, 0, ["front", "top", "iso"]);
    faces = view.addMesh({ positions: new Float32Array(0) }, { color: [0.6, 0.6, 0.65], opacity: 0.22, lighting: "unlit", shading: "flat" });
    pts = view.addPoints({ positions: new Float32Array(0) }, { size: 3, shaded: false });
    lines = view.addLines({ positions: new Float32Array(0) }, { width: 4 });
    marks = view.addPoints({ positions: new Float32Array(0) }, { size: 12, shaded: true });
  }
  const triBox = htmlEl("div", "uc-tri", "", host);
  const TW = 380, TH = 370, V: [number, number][] = [[30, 330], [190, 60], [350, 330]];   // ch1, ch2, ch3 corners
  const tri = svgEl("svg", { viewBox: `0 0 ${TW} ${TH}`, width: "100%", class: "qb-plot-svg", role: "img", "aria-label": "Normalised pixels in the triangle of channel fractions" }, triBox);
  svgEl("polygon", { points: V.map(v => v.join(",")).join(" "), fill: "none", stroke: "#999", "stroke-width": 1.5 }, tri);
  const cornerLab = (he ? ["OD R", "OD G", "OD B"] : ["ch 1", "ch 2", "ch 3"]);
  V.forEach((v, i) => { const t = svgEl("text", { x: v[0] + (i === 0 ? -4 : i === 2 ? 4 : 0), y: v[1] + (i === 1 ? -12 : 26), "text-anchor": i === 0 ? "start" : i === 2 ? "end" : "middle", "font-size": 17 }, tri); t.textContent = cornerLab[i]; });
  const triPts = svgEl("g", {}, tri), triSimplex = svgEl("g", {}, tri), triMarks = svgEl("g", {}, tri), triHandles = svgEl("g", {}, tri);
  const triTitle = svgEl("text", { x: TW / 2, y: 20, "text-anchor": "middle", "font-size": 17, "font-weight": 600 }, tri);
  triTitle.textContent = "normalised: channel fractions";
  const toTri = (f: ArrayLike<number>): [number, number] => {
    const s = f[0] + f[1] + f[2] || 1, a = f[0] / s, b = f[1] / s, c = f[2] / s;
    return [a * V[0][0] + b * V[1][0] + c * V[2][0], a * V[0][1] + b * V[1][1] + c * V[2][1]];
  };
  const fromTri = (x: number, y: number): V3 => {
    // barycentric coordinates of (x, y) in the triangle V
    const [x1, y1] = V[0], [x2, y2] = V[1], [x3, y3] = V[2], d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    let a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / d, b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / d;
    a = Math.max(0.01, a); b = Math.max(0.01, b);
    let c = 1 - a - b;
    if (c < 0.01) { const s = a + b; a = a / s * 0.99; b = b / s * 0.99; c = 0.01; }
    return [a, b, c];
  };

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const presetDefs: { id: string; label: string; s: Partial<St>; pick?: "dim" | "membrane" | "auto" }[] = he ? [
    { id: "clean", label: "OD: a flat wedge", s: { noise: false, normalise: false, faces: true } },
    { id: "noise", label: "with noise", s: { noise: true, normalise: false } },
    { id: "normalise", label: "normalise: a line segment", s: { noise: true, normalise: true } },
  ] : [
    { id: "clean", label: "noise-free: everything inside the cone", s: { noise: false, normalise: false, faces: true, fourth: false } },
    { id: "noise", label: "noise: LS goes negative", s: { noise: true, logPhot: Math.log10(300), normalise: false, fourth: false }, pick: "dim" },
    { id: "nnls", label: "NNLS snaps to the face", s: { noise: true, logPhot: Math.log10(300), normalise: true, fourth: false }, pick: "dim" },
    { id: "normalise", label: "normalise: the cone becomes a triangle", s: { noise: true, logPhot: Math.log10(300), normalise: true, fourth: false } },
    { id: "wrong", label: "a wrong endmember", s: { noise: true, logPhot: Math.log10(1000), normalise: true, fourth: false, E: [CELL_E[0], CELL_E[1], [0.15, 0.32, 0.53]] }, pick: "membrane" },
    { id: "fourth", label: "an unmodelled component: a fourth corner", s: { noise: true, logPhot: Math.log10(1000), normalise: true, fourth: true }, pick: "auto" },
  ];
  const pbtn = presets(panel, [
    ...presetDefs.map(p => ({ label: p.label, apply: () => applyPreset(p) })),
    { label: "reset", apply: () => applyPreset({ s: { noise: true, logPhot: Math.log10(300), normalise: false, faces: true, fourth: false } }) },
  ]);
  const noiseC = checkbox(panel, "noise", st.noise, v => { st.noise = v; recompute(); });
  const phS = he ? null : slider(panel, "photons", 1, 3.5, 0.05, st.logPhot, v => { st.logPhot = v; recompute(); }, v => String(Math.round(10 ** v)));
  const normC = checkbox(panel, "normalise (divide by the total)", st.normalise, v => { st.normalise = v; drawAll(); });
  const faceC = checkbox(panel, "cone faces", st.faces, v => { st.faces = v; drawAll(); });
  const fourthC = he ? null : checkbox(panel, "unmodelled 4th component", st.fourth, v => { st.fourth = v; recompute(); });
  const read = htmlEl("div", "qb-readout um-readout", "", triBox);
  const legend = htmlEl("div", "um-row", "", triBox);
  for (const [k, c] of Object.entries(MCOL)) { const s = htmlEl("span", "", "● " + (k === "ls" ? "LS" : k === "nnls" ? "NNLS" : "fully constr."), legend); s.style.color = c; s.style.fontSize = "15px"; }

  // ---- computation helpers ----
  const Mof = (): Float64Array => { const M = new Float64Array(3 * K); for (let c = 0; c < 3; c++) for (let k = 0; k < K; k++) M[c * K + k] = st.E[k][c]; return M; };
  const pixel = (p: number): V3 => [Y[p], Y[P + p], Y[2 * P + p]];
  function solve(p: number): { ls: Float64Array; nn: Float64Array; fc: Float64Array; y: V3 } | null {
    const M = Mof(), y = pixel(p), ls = U.lsPixel(M, 3, K, y);
    if (!ls) return null;
    const s = y[0] + y[1] + y[2];
    const fc = s > 0 ? U.fcls(M, 3, K, y.map(v => v / s)).map(v => v * s) : new Float64Array(K);
    return { ls, nn: U.nnls(M, 3, K, y), fc, y };
  }
  const recon = (a: ArrayLike<number>): V3 => { const out: V3 = [0, 0, 0]; for (let c = 0; c < 3; c++) for (let k = 0; k < K; k++) out[c] += st.E[k][c] * a[k]; return out; };

  // ---- drawing ----
  const world = (v: ArrayLike<number>): V3 => {
    if (st.normalise) { const s = v[0] + v[1] + v[2]; return s > 0 ? [v[0] / s * 1.3 - 0.65, v[1] / s * 1.3 - 0.65, v[2] / s * 1.3 - 0.65] : [-0.65, -0.65, -0.65]; }
    return [v[0] / vmax * 1.3 - 0.65, v[1] / vmax * 1.3 - 0.65, v[2] / vmax * 1.3 - 0.65];
  };
  function drawImage(): void {
    const ctx = img.canvas.getContext("2d")!, id = ctx.createImageData(N, N), d = id.data;
    for (let p = 0; p < P; p++) {
      if (he && heSc) {
        for (let c = 0; c < 3; c++) d[4 * p + c] = 240 * 10 ** -Y[c * P + p];   // Beer–Lambert back to RGB
      } else {
        // channel 1 → blue, 2 → green, 3 → red
        d[4 * p] = 255 * Math.min(1, Y[2 * P + p] / (0.6 * vmax)); d[4 * p + 1] = 255 * Math.min(1, Y[P + p] / (0.6 * vmax)); d[4 * p + 2] = 255 * Math.min(1, Y[p] / (0.6 * vmax));
      }
      d[4 * p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    if (pick >= 0) {
      const x = pick % N, y = Math.floor(pick / N);
      ctx.strokeStyle = he ? "#000" : "#fff"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x + 0.5, y + 0.5, 4, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  function drawAll(): void {
    // which pixels to show: all with some signal
    const show: number[] = [];
    for (let p = 0; p < P; p++) { const s = Y[p] + Y[P + p] + Y[2 * P + p]; if (s > (he ? 0.08 : 0.03)) show.push(p); }
    if (view && pts && lines && faces && marks) {
      const pos = new Float32Array(3 * show.length), col = new Float32Array(3 * show.length);
      show.forEach((p, i) => {
        const w = world(pixel(p));
        pos[3 * i] = w[0]; pos[3 * i + 1] = w[1]; pos[3 * i + 2] = w[2];
        const y = pixel(p), m = Math.max(y[0], y[1], y[2], 1e-6);
        if (he) { col[3 * i] = 10 ** -y[0]; col[3 * i + 1] = 10 ** -y[1]; col[3 * i + 2] = 10 ** -y[2]; }
        else { col[3 * i] = Math.max(0, y[2] / m) * 0.9; col[3 * i + 1] = Math.max(0, y[1] / m) * 0.9; col[3 * i + 2] = Math.max(0, y[0] / m) * 0.9; }
      });
      pts.setData({ positions: pos, colors: col });
      // axes and endmember arrows
      const lp: number[] = [], lc: number[] = [];
      const o = world([0, 0, 0]);
      const axisLen = st.normalise ? [1, 1, 1] : [vmax * 1.1, vmax * 1.1, vmax * 1.1];
      for (let c = 0; c < 3; c++) { const e: V3 = [0, 0, 0]; e[c] = st.normalise ? 1 : axisLen[c]; const w = world(e); lp.push(...o, ...w); lc.push(0.45, 0.45, 0.45); }
      if (st.normalise) { const c3: V3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; for (let i = 0; i < 3; i++) { lp.push(...world(c3[i]), ...world(c3[(i + 1) % 3])); lc.push(0.6, 0.6, 0.6); } }
      st.E.slice(0, K).forEach((e, k) => {
        const amax = he ? 1.3 : 1.0, tip = st.normalise ? world(e) : world([e[0] * amax, e[1] * amax, e[2] * amax]);
        lp.push(...o, ...tip); lc.push(...ecol[k]);
      });
      if (st.normalise) for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) { lp.push(...world(st.E[i]), ...world(st.E[j])); lc.push(0.2, 0.2, 0.2); }
      lines.setData({ positions: Float32Array.from(lp), colors: Float32Array.from(lc) });
      // faces
      const fp: number[] = [];
      if (st.faces) {
        if (st.normalise) { if (K === 3) fp.push(...world(st.E[0]), ...world(st.E[1]), ...world(st.E[2])); }
        else for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
          const L = 0.95 * vmax;
          fp.push(...o, ...world(st.E[i].map(v => v * L / Math.max(...st.E[i])) as V3), ...world(st.E[j].map(v => v * L / Math.max(...st.E[j])) as V3));
        }
      }
      faces.setData({ positions: Float32Array.from(fp) });
    }
    // triangle view
    triPts.replaceChildren();
    let d = "";
    for (let i = 0; i < show.length; i += Math.max(1, Math.floor(show.length / 3500))) { const t = toTri(pixel(show[i])); d += `M${t[0].toFixed(1)} ${t[1].toFixed(1)}h0`; }
    svgEl("path", { d, stroke: "#666", "stroke-width": 3, "stroke-linecap": "round", "stroke-opacity": 0.35, fill: "none" }, triPts);
    triSimplex.replaceChildren();
    const tp = st.E.slice(0, K).map(e => toTri(e));
    svgEl(K === 3 ? "polygon" : "polyline", { points: tp.map(t => t.join(",")).join(" "), fill: K === 3 ? "#888" : "none", "fill-opacity": 0.12, stroke: "#222", "stroke-width": 2 }, triSimplex);
    tp.forEach((t, k) => { const h = handles[k]; if (h) { h.setAttribute("cx", t[0].toFixed(1)); h.setAttribute("cy", t[1].toFixed(1)); } });
    drawImage();
    drawPick();
  }
  const handles: SVGCircleElement[] = [];
  for (let k = 0; k < K; k++) {
    const h = svgEl("circle", { r: 11, fill: css(ecol[k]), stroke: "#fff", "stroke-width": 3 }, triHandles);
    const t = svgEl("title", {}, h); t.textContent = names[k];
    handles.push(h);
    draggable(h, ev => svgPoint(tri, ev), (x, y) => { st.E[k] = fromTri(x, y); pbtn.forEach(b => b.classList.remove("active")); drawAll(); });
  }

  function drawPick(): void {
    triMarks.replaceChildren();
    bars.replaceChildren();
    const markPos: number[] = [], markCol: number[] = [];
    const lines2: string[] = [];
    let negLS = 0, cnt = 0;
    const M = Mof();
    const Pi = U.pseudoInverse(M, 3, K);
    if (Pi) for (let p = 0; p < P; p++) { const s = Y[p] + Y[P + p] + Y[2 * P + p]; if (s < (he ? 0.15 : 0.1)) continue; cnt++; for (let k = 0; k < K; k++) { const a = Pi[k * 3] * Y[p] + Pi[k * 3 + 1] * Y[P + p] + Pi[k * 3 + 2] * Y[2 * P + p]; if (a < -1e-9) { negLS++; break; } } }
    lines2.push(`bright pixels with an LS abundance < 0: ${fmt(100 * negLS / Math.max(cnt, 1), 1)} %`);
    if (pick >= 0) {
      const s = solve(pick);
      if (s) {
        const res = (a: ArrayLike<number>) => { const r = recon(a); return Math.hypot(r[0] - s.y[0], r[1] - s.y[1], r[2] - s.y[2]); };
        const sols: [keyof typeof MCOL, Float64Array][] = [["ls", s.ls], ["nnls", s.nn], ["fcls", s.fc]];
        const pt = toTri(s.y);
        svgEl("circle", { cx: pt[0], cy: pt[1], r: 7, fill: "none", stroke: "#000", "stroke-width": 2.5 }, triMarks);
        for (const [m, a] of sols) {
          const r = recon(a), t = toTri(r);
          svgEl("line", { x1: pt[0], y1: pt[1], x2: t[0], y2: t[1], stroke: MCOL[m], "stroke-width": 2 }, triMarks);
          svgEl("circle", { cx: t[0], cy: t[1], r: 6, fill: MCOL[m], stroke: "#fff", "stroke-width": 1.5 }, triMarks);
          const w = world(r); markPos.push(...w); const c = MCOL[m]; markCol.push(parseInt(c.slice(1, 3), 16) / 255, parseInt(c.slice(3, 5), 16) / 255, parseInt(c.slice(5, 7), 16) / 255);
        }
        markPos.push(...world(s.y)); markCol.push(0, 0, 0);
        lines2.push(`pixel (${pick % N}, ${Math.floor(pick / N)}): y = ${s.y.map(v => fmt(v, 2)).join(", ")}`);
        for (const [m, a] of sols) lines2.push(`${m === "ls" ? "LS  " : m === "nnls" ? "NNLS" : "FCLS"} a = ${Array.from(a).map(v => fmt(v, 2)).join(", ")}  |r| ${fmt(res(a), 3)}`);
        // bars: groups per component, three methods each
        const amax = Math.max(0.05, ...sols.flatMap(([, a]) => Array.from(a).map(Math.abs)));
        const y0 = 120, hsc = 95 / amax, gw = 220 / K;
        svgEl("line", { x1: 10, x2: 235, y1: y0, y2: y0, stroke: "#555", "stroke-width": 1.5 }, bars);
        for (let k = 0; k < K; k++) {
          sols.forEach(([m, a], j) => {
            const v = a[k], x = 14 + k * gw + j * (gw - 10) / 3, h = Math.abs(v) * hsc;
            svgEl("rect", { x, y: v >= 0 ? y0 - h : y0, width: (gw - 14) / 3, height: Math.max(1, h), fill: MCOL[m] }, bars);
          });
          const t = svgEl("text", { x: 14 + k * gw + (gw - 12) / 2, y: 182, "text-anchor": "middle", "font-size": 16, fill: css(ecol[k]) }, bars);
          t.textContent = (he ? HE_SHORT : SHORT)[k];
        }
        const t0 = svgEl("text", { x: 6, y: 16, "font-size": 15 }, bars); t0.textContent = "abundances of the pixel";
      }
    } else lines2.push("click a pixel in the image");
    if (marks) marks.setData({ positions: Float32Array.from(markPos), colors: Float32Array.from(markCol) });
    read.textContent = lines2.join("\n");
  }

  img.canvas.addEventListener("pointerdown", ev => { const p = canvasPixel(img.canvas, ev, N, N); if (p >= 0) { pick = p; drawImage(); drawPick(); } });

  function recompute(): void {
    simulate();
    let m = 0;
    for (let i = 0; i < 3 * P; i++) m = Math.max(m, Y[i]);
    vmax = Math.max(1e-6, m * 0.9);
    drawAll();
  }
  function choosePixel(kind: "dim" | "membrane" | "auto"): number {
    const M = Mof(), Pi = U.pseudoInverse(M, 3, K)!;
    let best = -1, score = -Infinity;
    for (let p = 0; p < P; p++) {
      if (!sc.cell[p]) continue;
      const a = [0, 1, 2].slice(0, K).map(k => Pi[k * 3] * Y[p] + Pi[k * 3 + 1] * Y[P + p] + Pi[k * 3 + 2] * Y[2 * P + p]);
      const s = Y[p] + Y[P + p] + Y[2 * P + p];
      let v = -Infinity;
      if (kind === "dim") v = s > 0.08 && s < 0.25 ? -Math.min(...a) : -Infinity;
      else if (kind === "membrane") v = sc.membrane[p] > 0.9 ? -Math.min(...a) : -Infinity;
      else v = sc.auto[p] + 3 * sc.vesicles[p];
      if (v > score) { score = v; best = p; }
    }
    return best;
  }
  function applyPreset(p: { s: Partial<St>; pick?: "dim" | "membrane" | "auto" }): void {
    Object.assign(st, { E: trueE.map(e => [...e] as V3) }, p.s);
    if (p.s.E) st.E = p.s.E.map(e => [...e] as V3);
    noiseC.checked = st.noise; normC.checked = st.normalise; faceC.checked = st.faces; phS?.set(st.logPhot); if (fourthC) fourthC.checked = st.fourth;
    simulate();
    let m = 0;
    for (let i = 0; i < 3 * P; i++) m = Math.max(m, Y[i]);
    vmax = Math.max(1e-6, m * 0.9);
    pick = p.pick ? choosePixel(p.pick) : -1;
    drawAll();
  }
  const pre = presetDefs.findIndex(d => d.id === host.dataset.preset);
  if (pre >= 0) { applyPreset(presetDefs[pre]); pbtn[pre].classList.add("active"); }
  else applyPreset({ s: {} });
}

autoMount(NAME, mount);
