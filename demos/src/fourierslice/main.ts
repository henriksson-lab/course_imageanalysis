// The Fourier slice theorem. Slice tab: drag the angle handle on the image; the middle
// panel is the 2D (log) spectrum with the central line at that angle; the plot compares the
// 1D spectrum of the projection at that angle with the 2D spectrum sampled along the line
// (they coincide: exactly at 0 and 90 degrees, up to interpolation otherwise).
// Direct-reconstruction tab: the projections' 1D spectra are placed on their spokes,
// gridded onto the Cartesian frequency grid (nearest or bilinear) and inverted with one 2D FFT.
// Auto-mounts on <div data-demo="fourierslice">. Options: data-preset="stripes|match|sparse|nearest|real", data-mode="slice|direct".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { diverging, grey } from "../lib/colour";
import * as R from "../lib/radon";
import { cellObject, objectImage, imagePanel, injectStyle, baseCss, layout, onPointer, drawHeat, lineSeg, CELL_CREDIT, uniqueClip } from "../radonproj/view";
import * as FS from "./model";

const NAME = "fourierslice";
const N = 128, P = 2 * N;

type Obj = "stripes" | "tilted" | "disc" | "shepp" | "cell";
const OBJ_LABEL: Record<Obj, string> = { stripes: "stripes", tilted: "tilted stripes", disc: "disc", shepp: "Shepp–Logan", cell: "micrograph" };
type Comp = "log" | "re" | "im";
const COMP_LABEL: Record<Comp, string> = { log: "log magnitude", re: "real part", im: "imaginary part" };

interface Settings { mode: "slice" | "direct"; obj: Obj; theta: number; comp: Comp; spokes: number; grid: FS.Gridding }
interface Preset { id: string; label: string; s: Settings }
const base: Settings = { mode: "slice", obj: "shepp", theta: 30, comp: "log", spokes: 90, grid: "bilinear" };
const PRESETS: Preset[] = [
  { id: "stripes", label: "stripes: one angle sees them, the other doesn't", s: { ...base, obj: "stripes", theta: 0 } },
  { id: "match", label: "projection ↔ slice match", s: { ...base } },
  { id: "real", label: "a real image", s: { ...base, obj: "cell", theta: 60 } },
  { id: "sparse", label: "spokes are sparse at the edge", s: { ...base, mode: "direct", spokes: 24 } },
  { id: "nearest", label: "nearest gridding: errors everywhere", s: { ...base, mode: "direct", spokes: 90, grid: "nearest" } },
];

const STYLE = baseCss(NAME);

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const pAttr = PRESETS.find(p => p.id === host.dataset.preset);
  const start: Settings = pAttr ? { ...pAttr.s } : { ...base, mode: host.dataset.mode === "direct" ? "direct" : "slice" };
  const st: Settings = { ...start };

  const L = layout(host, 3);
  const imgP = imagePanel(L.figs, "image: drag to turn", N, 6, "rd-fig");
  const credit = htmlEl("div", "rd-credit", CELL_CREDIT, imgP.fig);
  const specFig = plotFigure(L.figs, "2D spectrum", "rd-fig");
  const spec = uniqueClip(new Plot(specFig, { font: 22, width: 320, height: 320, x: [-0.5, 0.5], y: [-0.5, 0.5], equal: true, grid: false, xticks: [-0.5, 0, 0.5], yticks: [-0.5, 0, 0.5], margin: { l: 62, b: 34, t: 6, r: 12 } }));
  const plotFig = plotFigure(L.figs, "projection spectrum vs slice", "rd-fig wide");
  const cmp = uniqueClip(new Plot(plotFig, { font: 22, width: 400, height: 300, x: [-0.5, 0.5], y: [0, 1], xlabel: "frequency f (cycles per pixel)", xticks: [-0.5, -0.25, 0, 0.25, 0.5] }));
  const errP = imagePanel(L.figs, "error", N, 0, "rd-fig");
  const [colA, colB, colC] = L.cols;

  htmlEl("div", "rd-lab", "presets", colA);
  const presetBtns = presets(colA, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s, p) })));
  htmlEl("div", "rd-lab", "mode", colB);
  const modeRow = htmlEl("div", "qb-presets", "", colB);
  const modeBtns = (["slice", "direct"] as const).map(m => button(modeRow, m === "slice" ? "slice theorem" : "direct reconstruction", () => { st.mode = m; draw(); }));
  htmlEl("div", "rd-lab", "image", colB);
  const objRow = htmlEl("div", "qb-presets", "", colB);
  const objs: Obj[] = ["stripes", "tilted", "disc", "shepp", "cell"];
  const objBtns = objs.map(o => button(objRow, OBJ_LABEL[o], () => { st.obj = o; makeImage(); }));
  const sliceBox = htmlEl("div", "rd-sub", "", colC);
  const thS = slider(sliceBox, "θ", 0, 179, 1, st.theta, v => { st.theta = v; draw(); }, v => v + "°");
  const compRow = htmlEl("div", "qb-presets", "", sliceBox);
  const compBtns = (["log", "re", "im"] as Comp[]).map(c => button(compRow, COMP_LABEL[c], () => { st.comp = c; draw(); }));
  const directBox = htmlEl("div", "rd-sub", "", colC);
  const spS = slider(directBox, "spokes (angles)", 2, 360, 1, st.spokes, v => { st.spokes = v; draw(); });
  const gRow = htmlEl("div", "qb-presets", "", directBox);
  const gBtns = (["nearest", "bilinear"] as FS.Gridding[]).map(g => button(gRow, g === "nearest" ? "nearest cell" : "bilinear", () => { st.grid = g; draw(); }));
  button(htmlEl("div", "rd-row", "", colC), "reset", () => load(start));
  const ro = htmlEl("div", "qb-readout", "", colC);

  let img = new Float64Array(N * N) as Float64Array, G: FS.Spectrum = FS.centredSpectrum(img, N);
  let logSpec = new Float64Array(P * P) as Float64Array;
  const mask = R.fovMask(N);

  function makeImage(): void {
    if (st.obj === "cell") { cellObject().then(() => { img = objectImage("cell", N); spectrum(); }); return; }
    img = st.obj === "stripes" ? FS.stripes(N, 0, 8) : st.obj === "tilted" ? FS.stripes(N, 30, 10) : objectImage(st.obj, N);
    spectrum();
  }
  function spectrum(): void {
    G = FS.centredSpectrum(img, N, P);
    logSpec = toLog(G);
    draw();
  }
  function toLog(S: FS.Spectrum): Float64Array {
    const out = new Float64Array(P * P);
    for (let k = 0; k < P; k++) for (let i = 0; i < P; i++) out[(P - 1 - k) * P + i] = Math.log10(1 + Math.hypot(S.re[k * P + i], S.im[k * P + i]));
    return out;
  }

  // spectrum heat + overlays
  let specVals: ArrayLike<number> = logSpec;
  spec.mark(g => drawHeat(spec, g, specVals, P, P, -0.5, 0.5, -0.5, 0.5, 0, undefined, grey), []);
  let spokes: { remove(): void }[] = [];
  const sliceLine = spec.line([0], [0], { color: "var(--qb-c2)", width: 2.5 });
  const axisLine = imgP.plot.line([0], [0], { color: "var(--qb-c2)", width: 3 });
  const rayLines = [-40, 0, 40].map(() => imgP.plot.line([0], [0], { color: "#f2c200", width: 1.5, opacity: 0.8 }));
  const handle = imgP.plot.scatter([0], [0], { color: "var(--qb-c2)", r: 10 });
  const projLine = cmp.line([0], [0], { color: "var(--qb-c1)", width: 4, label: "1D spectrum of the projection" });
  const sliceCurve = cmp.line([0], [0], { color: "var(--qb-c2)", width: 2, dash: "6 4", label: "2D spectrum along the line" });

  onPointer(imgP.plot, {
    down: (x, y) => setTheta(x, y),
    move: (x, y) => setTheta(x, y),
  });
  function setTheta(x: number, y: number): void {
    if (st.mode !== "slice" || Math.hypot(x, y) < 4) return;
    let t = Math.round(Math.atan2(y, x) * 180 / Math.PI);
    t = ((t % 180) + 180) % 180;
    st.theta = t;
    presetBtns.forEach(b => b.classList.remove("active"));
    draw();
  }

  function draw(): void {
    modeBtns.forEach((b, i) => b.classList.toggle("active", ["slice", "direct"][i] === st.mode));
    objBtns.forEach((b, i) => b.classList.toggle("active", objs[i] === st.obj));
    compBtns.forEach((b, i) => b.classList.toggle("active", ["log", "re", "im"][i] === st.comp));
    gBtns.forEach((b, i) => b.classList.toggle("active", ["nearest", "bilinear"][i] === st.grid));
    thS.set(st.theta); spS.set(st.spokes);
    credit.style.display = st.obj === "cell" ? "" : "none";
    const slice = st.mode === "slice";
    sliceBox.style.display = slice ? "" : "none";
    directBox.style.display = slice ? "none" : "";
    plotFig.style.display = slice ? "" : "none";
    errP.fig.style.display = slice ? "none" : "";
    spokes.forEach(m => m.remove()); spokes = [];
    let imax = 0; for (const v of img) imax = Math.max(imax, v);
    if (slice) drawSlice(imax); else drawDirect(imax);
  }

  function drawSlice(imax: number): void {
    const th = st.theta * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
    imgP.show(img, 0, imax || 1);
    imgP.cap.textContent = "image: drag to turn the angle";
    axisLine.set([-62 * c, 62 * c], [-62 * s, 62 * s]);
    [-40, 0, 40].forEach((r, k) => { const [xs, ys] = lineSeg(th, r, 62, -62); rayLines[k].set(xs, ys); });
    handle.set([58 * c], [58 * s]);
    specVals = logSpec; spec.renderAll();
    sliceLine.set([-0.5 * c, 0.5 * c], [-0.5 * s, 0.5 * s]);
    (specFig.querySelector("figcaption") as HTMLElement).textContent = "2D spectrum (log), red: the slice";
    // the comparison
    const p = R.projectAngle(img, N, th);
    const Ps = FS.projectionSpectrum(p, N, P), Sl = FS.sliceOf(G, th);
    const pick = (re: Float64Array, im: Float64Array) => Array.from(re, (v, j) => st.comp === "log" ? Math.log10(1 + Math.hypot(v, im[j])) : st.comp === "re" ? v : im[j]);
    const a = pick(Ps.re, Ps.im), b = pick(Sl.re, Sl.im);
    let lo = Math.min(...a, ...b), hi = Math.max(...a, ...b);
    if (st.comp !== "log") { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; } else lo = 0;
    cmp.setDomain(undefined, [lo, hi * 1.05 || 1]);
    cmp.setLabels({ ylabel: COMP_LABEL[st.comp] });
    projLine.set(Ps.f, a); sliceCurve.set(Ps.f, b);
    (plotFig.querySelector("figcaption") as HTMLElement).textContent = `θ = ${st.theta}°: blue = projection spectrum, red dashed = slice`;
    const d = FS.relDiff(Ps.re, Ps.im, Sl.re, Sl.im);
    ro.textContent = `θ = ${st.theta}°\nrelative difference between the curves: ${(100 * d).toFixed(2)} %\n(0 at 0° and 90°: the line runs along grid points)`;
  }

  function drawDirect(imax: number): void {
    const g = R.geometry(N, R.angleSet(st.spokes)), sino = R.project(img, g);
    const res = FS.directRecon(sino, N, g.angles, st.grid, P);
    imgP.show(res.img, 0, imax || 1);
    imgP.cap.textContent = `direct reconstruction (${st.spokes} spokes)`;
    axisLine.set([0], [0]); rayLines.forEach(l => l.set([0], [0])); handle.set([], []);
    specVals = toLog(res.grid); spec.renderAll();
    sliceLine.set([0], [0]);
    (specFig.querySelector("figcaption") as HTMLElement).textContent = `gridded spectrum (${st.grid})`;
    const err = res.img.map((v, i) => (v - img[i]) * mask[i]);
    // colour scale: the 99th percentile of |error| (a few edge pixels would hide the rest)
    const sorted = Array.from(err, Math.abs).sort((a, b) => a - b), m = Math.max(1e-6, sorted[Math.floor(0.99 * sorted.length)]);
    errP.show(err, -m, m, diverging);
    errP.cap.textContent = "error (reconstruction − truth)";
    const fb = R.fbp(sino, g);
    ro.textContent = `${st.spokes} spokes, ${st.grid} gridding\nRMSE direct: ${R.rmseOf(res.img, img, mask).toFixed(4)}\nRMSE filtered back-projection: ${R.rmseOf(fb, img, mask).toFixed(4)}\ngrid cells filled: ${(100 * res.filled / (P * P)).toFixed(1)} %\nerror colour scale: ±${m.toFixed(3)}`;
  }

  function load(s: Settings, p?: Preset): void {
    Object.assign(st, s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }
  load(start, pAttr);
}

autoMount(NAME, mount);
