// Region-based segmentation with level sets: Chan–Vese (piecewise-constant Mumford–Shah) on a
// 128 × 128 image with grey levels 0..1 (lib/levelset.ts, cvStep). The energy
//   E = Σ_in (I − c1)² + Σ_out (I − c2)² + λ·Length + ν·Area_in
// is lowered by alternating: c1, c2 = means inside and outside; one semi-implicit step of
// φ_t = δ_ε(φ)[λκ + ν + (I − c1)² − (I − c2)²]. Side panels: thresholding at (c1 + c2)/2, φ,
// the histogram with c1, c2 and the threshold, and E against the iteration. A toggle runs the
// edge-based flow of the gac demo on the same image instead, for comparison.
// Auto-mounts on <div data-demo="chanvese">. Optional: data-preset=<id>, data-steps=<n>,
// data-math="on" (show the data-weight slider of the math layer).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { diverging, divergingT } from "../lib/colour";
import { Plot } from "../lib/plot";
import { histogram } from "../lib/histogram";
import * as LS from "../lib/levelset";
import { Contour } from "../lib/marching";
import { Animator } from "../curveflow/anim";
import { N, SceneId, scene, addNoise, InitId, initPhi } from "./model";
import { edgeMap, gacTerms } from "../gac/model";

const NAME = "chanvese";
const S = 512;
const DT = 10;
const GAC = { sigma: 2, k: 8, c: 0.3, eps: 0.3, a: 2 };

type ImageId = SceneId | "cell";
interface Settings { image: ImageId; noise: number; init: InitId; lambda: number; nu: number; eps: number; ratio: number; mode: "cv" | "edge" }
interface Preset { id: string; label: string; s: Settings }
const D: Settings = { image: "shapes", noise: 0.15, init: "grid", lambda: 0.1, nu: 0, eps: 1, ratio: 1, mode: "cv" };
const PRESETS: Preset[] = [
  { id: "noise", label: "noise: regions work", s: { ...D, noise: 0.3, lambda: 0.2 } },
  { id: "lambda0", label: "λ = 0 is thresholding", s: { ...D, lambda: 0 } },
  { id: "biglambda", label: "large λ: loses detail", s: { ...D, lambda: 1 } },
  { id: "hole", label: "a hole appears", s: { ...D, init: "ring" } },
  { id: "gradient", label: "gradient background", s: { ...D, image: "gradient", noise: 0.1 } },
  { id: "badstart", label: "bad start", s: { ...D, init: "hole", eps: 0.1 } },
  { id: "cell", label: "micrograph", s: { ...D, image: "cell", noise: 0, lambda: 0.05 } },
];
const IMAGES: { id: ImageId; label: string }[] = [{ id: "shapes", label: "shapes" }, { id: "gradient", label: "gradient" }, { id: "cell", label: "micrograph" }];
const INITS: { id: InitId; label: string }[] = [{ id: "grid", label: "grid" }, { id: "circle", label: "circle" }, { id: "ring", label: "around the ring" }, { id: "hole", label: "in the hole" }];

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .cv-main { flex: 0 1 400px; min-width: 240px; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
[data-demo="${NAME}"] .cv-cap { font-size: 16px; text-align: center; margin: 0 0 .15em; line-height: 1.2; }
[data-demo="${NAME}"] .cv-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .cv-mid { flex: 0 1 270px; min-width: 200px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .cv-pair { display: flex; gap: .4em; }
[data-demo="${NAME}"] .cv-pair > div { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 360px; gap: .3em; }
[data-demo="${NAME}"] .cv-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .cv-row { display: flex; gap: .3em; flex-wrap: wrap; align-items: center; }
[data-demo="${NAME}"] .cv-sliders { display: flex; flex-direction: column; gap: .3em; }
.reveal [data-demo="${NAME}"] { font-size: 16px; gap: .6em; }
.reveal [data-demo="${NAME}"] .cv-main { flex-basis: 380px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; max-width: 380px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .45em; }
.reveal [data-demo="${NAME}"] .cv-sliders { gap: .05em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: Settings = { ...p0.s };
  let showMath = host.dataset.math === "on";

  // ---- layout ----
  const main = htmlEl("div", "cv-main", "", host);
  const cap = htmlEl("div", "cv-cap", "", main);
  const cv = htmlEl("canvas", "", "", main);
  cv.width = S; cv.height = S;
  const credit = htmlEl("div", "cv-credit", CELL_CREDIT, main);
  const img = document.createElement("canvas"); img.width = N; img.height = N;

  const mid = htmlEl("div", "cv-mid", "", host);
  const pair = htmlEl("div", "cv-pair", "", mid);
  const thrBox = htmlEl("div", "", "", pair), phiBox = htmlEl("div", "", "", pair);
  htmlEl("div", "cv-cap", "threshold (c1+c2)/2", thrBox);
  const cvT = htmlEl("canvas", "", "", thrBox); cvT.width = N; cvT.height = N;
  htmlEl("div", "cv-cap", "φ (blue < 0)", phiBox);
  const cvP = htmlEl("canvas", "", "", phiBox); cvP.width = N; cvP.height = N;
  const H = new Plot(mid, { width: 300, height: 150, x: [-0.2, 1.2], y: [0, 1], xlabel: "grey level", yticks: [], font: 16, margin: { l: 10, r: 10, t: 8 } });
  const histMark = H.histogram([0, 1], [0], { color: "#999", fill: "#bbb" });
  const c1Mark = H.vline(0, { color: "#2166ac", width: 3, text: "c1" });
  const c2Mark = H.vline(1, { color: "#b2182b", width: 3, text: "c2" });
  const tMark = H.vline(0.5, { color: "#111", width: 2, dash: "6 4" });
  const EP = new Plot(mid, { width: 300, height: 150, x: [0, 1], y: [0, 1], xlabel: "iteration", ylabel: "E", font: 16 });
  const eLine = EP.line([0], [0], { color: "#d62728", width: 2.5 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "cv-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "cv-lab", "image", panel);
  const r1 = htmlEl("div", "cv-row", "", panel);
  const imgBtns = IMAGES.map(m => button(r1, m.label, () => { st.image = m.id; changed(); makeImage(); }));
  htmlEl("div", "cv-lab", "start", panel);
  const r2 = htmlEl("div", "cv-row", "", panel);
  const initBtns = INITS.map(m => button(r2, m.label, () => { st.init = m.id; changed(); restart(); }));
  htmlEl("div", "cv-lab", "speed", panel);
  const r3 = htmlEl("div", "cv-row", "", panel);
  const modeBtns = [button(r3, "Chan–Vese (regions)", () => setMode("cv")), button(r3, "edge-based", () => setMode("edge"))];
  const sl = htmlEl("div", "cv-sliders", "", panel);
  const nS = slider(sl, "noise", 0, 0.4, 0.01, st.noise, v => { st.noise = v; changed(); makeImage(); });
  const lS = slider(sl, "length weight λ", 0, 2, 0.01, st.lambda, v => { st.lambda = v; changed(); });
  const vS = slider(sl, "area weight ν", -0.01, 0.01, 0.0005, st.nu, v => { st.nu = v; changed(); }, v => v.toFixed(4));
  const eS = slider(sl, "δ width ε", 0.1, 3, 0.1, st.eps, v => { st.eps = v; changed(); });
  const mathBox = htmlEl("div", "", "", sl);
  const rS = slider(mathBox, "data weights λ1/λ2", -2, 2, 0.25, Math.log2(st.ratio), v => { st.ratio = 2 ** v; changed(); }, v => (2 ** v).toFixed(2));
  const r4 = htmlEl("div", "cv-row", "", panel);
  checkbox(r4, "math layer", showMath, v => { showMath = v; mathBox.style.display = v ? "block" : "none"; });
  mathBox.style.display = showMath ? "block" : "none";
  const playBtn = button(r4, "play", () => anim.toggle());
  button(r4, "step", () => { anim.pause(); advance(1); draw(); });
  button(r4, "reset", () => { anim.pause(); restart(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Yellow: the zero contour, with the inside (φ < 0) tinted. Grey levels run from 0 to 1. 'wrong px' compares the brighter region with the true objects (test images only).", panel);

  const anim = new Animator(host, () => frame(), p => { playBtn.textContent = p ? "pause" : "play"; });

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); }
  function setMode(m: "cv" | "edge"): void {
    if (m === st.mode) return;
    st.mode = m;
    if (m === "edge") { LS.reinitExact(phi, N, N); makeEdge(); }
    changed();
  }
  function load(p: Preset): void {
    anim.pause();
    Object.assign(st, p.s);
    nS.set(st.noise); lS.set(st.lambda); vS.set(st.nu); eS.set(st.eps); rS.set(Math.log2(st.ratio));
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }
  function sync(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMAGES[i].id === st.image));
    initBtns.forEach((b, i) => b.classList.toggle("active", INITS[i].id === st.init));
    modeBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.mode === "cv")));
  }

  // ---- image ----
  let cellImg: Float64Array | null = null, mask: Uint8Array | null = null;
  let I: Float64Array = new Float64Array(N * N), ready = false;
  let edgeT: LS.SpeedTerms = {}, edgeDt = 0;
  function makeImage(): void {
    if (st.image === "cell" && !cellImg) { loadCell(N).then(g => { cellImg = Float64Array.from(g.data, v => v / 255); makeImage(); }); return; }
    if (st.image === "cell") { I = addNoise(cellImg!, st.noise); mask = null; }
    else { const sc = scene(st.image); I = addNoise(sc.img, st.noise); mask = sc.mask; }
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    const h = histogram(I, { bins: 70, min: -0.2, max: 1.2 });
    let mx = 0; for (const c of h.counts) mx = Math.max(mx, c);
    H.setDomain(undefined, [0, mx * 1.05]);
    histMark.set(Float64Array.from({ length: h.counts.length + 1 }, (_, i) => h.lo + i * h.binWidth), h.counts);
    ready = true;
    if (st.mode === "edge") makeEdge();
    restart();
  }
  function makeEdge(): void {
    const E = edgeMap(Float64Array.from(I, v => 255 * v), GAC.sigma, GAC.k);
    edgeT = gacTerms(E, GAC);
    edgeDt = 0.8 * LS.cflLimit(edgeT, N * N);
  }

  // ---- evolution ----
  let phi: Float64Array = new Float64Array(N * N), steps = 0, energies: number[] = [], eSteps: number[] = [];
  let c1 = 0, c2 = 1;
  function params(): LS.CVParams { return { lambda: st.lambda, nu: st.nu, eps: st.eps, w1: st.ratio, w2: 1 }; }
  function restart(): void {
    phi = initPhi(st.init);
    if (st.mode === "edge") LS.reinitExact(phi, N, N);
    steps = 0; energies = []; eSteps = [];
    record();
    sync(); draw();
  }
  function record(): void {
    const m = LS.cvMeans(I, phi); c1 = m.c1; c2 = m.c2;
    energies.push(LS.cvEnergy(I, phi, N, N, params(), m).total); eSteps.push(steps);
  }
  function advance(k: number): void {
    if (!ready) return;
    for (let i = 0; i < k; i++) {
      if (st.mode === "cv") LS.cvStep(I, phi, N, N, params(), DT);
      else {
        LS.lsStep(phi, N, N, edgeT, edgeDt);
        if ((steps + 1) % 20 === 0) LS.reinitExact(phi, N, N);
      }
      steps++;
    }
    record();
  }
  function frame(): boolean {
    advance(5);
    draw();
    return steps < 3000;
  }

  // ---- drawing ----
  function strokeContour(ctx: CanvasRenderingContext2D, C: Contour, k: number): void {
    ctx.beginPath();
    for (let s = 0; s < C.segments.length; s += 2) {
      const a = C.segments[s], b = C.segments[s + 1];
      ctx.moveTo((C.points[2 * a] + 0.5) * k, (C.points[2 * a + 1] + 0.5) * k);
      ctx.lineTo((C.points[2 * b] + 0.5) * k, (C.points[2 * b + 1] + 0.5) * k);
    }
    ctx.stroke();
  }
  function grey(v: number): number { return Math.max(0, Math.min(255, 255 * v)); }
  function draw(): void {
    sync();
    if (!ready) return;
    // main: image, inside tinted, contour
    const ictx = img.getContext("2d")!, id = ictx.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const g = grey(I[p]), ins = phi[p] < 0;
      id.data[4 * p] = ins ? 0.7 * g + 0.3 * 255 : g;
      id.data[4 * p + 1] = ins ? 0.7 * g + 0.3 * 200 : g;
      id.data[4 * p + 2] = ins ? 0.7 * g : g;
      id.data[4 * p + 3] = 255;
    }
    ictx.putImageData(id, 0, 0);
    const ctx = cv.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, S, S);
    const C = LS.zeroContour(phi, N, N);
    ctx.strokeStyle = "#111"; ctx.lineWidth = 5; strokeContour(ctx, C, S / N);
    ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2.5; strokeContour(ctx, C, S / N);
    // threshold at (c1 + c2)/2, the brighter side as "object"
    const th = (c1 + c2) / 2, tctx = cvT.getContext("2d")!, td = tctx.createImageData(N, N);
    let dif = 0;
    for (let p = 0; p < N * N; p++) {
      const lab = c1 >= c2 ? I[p] > th : I[p] <= th;  // same side as "inside"
      if (lab !== phi[p] < 0) dif++;
      td.data[4 * p] = lab ? 255 : 40; td.data[4 * p + 1] = lab ? 212 : 40; td.data[4 * p + 2] = lab ? 0 : 40; td.data[4 * p + 3] = 255;
    }
    tctx.putImageData(td, 0, 0);
    // φ map
    let mxp = 0; for (let p = 0; p < N * N; p++) mxp = Math.max(mxp, Math.abs(phi[p]));
    const pctx = cvP.getContext("2d")!, pd = pctx.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const c = diverging(divergingT(phi[p], mxp || 1));
      pd.data[4 * p] = 255 * c[0]; pd.data[4 * p + 1] = 255 * c[1]; pd.data[4 * p + 2] = 255 * c[2]; pd.data[4 * p + 3] = 255;
    }
    pctx.putImageData(pd, 0, 0);
    // histogram marks and energy curve
    c1Mark.set(c1); c2Mark.set(c2); tMark.set(th);
    const xs = eSteps;
    const lo = Math.min(...energies), hi = Math.max(...energies);
    EP.setDomain([0, Math.max(50, xs[xs.length - 1] || 0)], [lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1)]);
    eLine.set(xs, energies);
    cap.textContent = st.mode === "cv" ? "Chan–Vese: inside tinted, contour yellow" : "edge-based speed on the same image";
    // readout
    const E = LS.cvEnergy(I, phi, N, N, params(), { c1, c2 });
    let wrong = "";
    if (mask) {
      let k = 0;
      for (let p = 0; p < N * N; p++) { const obj = c1 >= c2 ? phi[p] < 0 : phi[p] >= 0; if ((obj ? 1 : 0) !== mask[p]) k++; }
      let kt = 0;
      for (let p = 0; p < N * N; p++) { const obj = I[p] > th; if ((obj ? 1 : 0) !== mask[p]) kt++; }
      wrong = `wrong px        ${k}  (threshold ${kt})`;
    }
    const r = (a: string, b: string) => a.padEnd(16) + b;
    ro.textContent = [
      r("iteration", `${steps}`),
      r("c1 in, c2 out", `${c1.toFixed(3)}, ${c2.toFixed(3)}`),
      r("data in + out", `${(st.ratio * E.dataIn).toFixed(1)} + ${E.dataOut.toFixed(1)}`),
      r("length, area", `${E.length.toFixed(0)} px, ${E.area} px`),
      r("energy E", E.total.toFixed(1)),
      r("≠ threshold", `${dif} px`),
      wrong,
    ].filter(Boolean).join("\n");
  }

  sync();
  load(p0);
  const n0 = Number(host.dataset.steps ?? 0);
  if (n0 > 0) {
    const go = () => { if (!ready) { setTimeout(go, 50); return; } for (let i = 0; i < n0; i += 5) advance(5); draw(); };
    go();
  }
}

autoMount(NAME, mount);
