// Edge-based segmentation with level sets on a 128 × 128 image: geodesic active contours.
// The image becomes an edge-stopping function g = 1/(1 + |∇(G_σ * I)|²/k²) (≈ 1 in flat
// regions, ≈ 0 on edges), and the front moves with φ_t = −g (c − εκ)|∇φ| + a ∇g·∇φ:
// a balloon c (sign = direction), curvature ε, and advection a down the valleys of g.
// Click the image to add start circles, or use the big box. View the image or g, with the
// arrows −∇g. Reinitialisation (exact) every 20 steps. Presets: shrink-wrap, leaking through
// a weak edge, a C shape that needs the balloon, noise without smoothing, the micrograph.
// Auto-mounts on <div data-demo="gac">. Optional: data-preset=<id>, data-steps=<n> (run n steps at load).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import * as LS from "../lib/levelset";
import { Contour } from "../lib/marching";
import { Animator } from "../curveflow/anim";
import { N, synthClean, addNoise, edgeMap, EdgeMap, gacTerms, GacParams, circlesPhi, boxPhi } from "./model";

const NAME = "gac";
const S = 512;
const REINIT = 20;

type Circle = { x: number; y: number; r: number };
interface Settings extends GacParams { sigma: number; k: number; noise: number; image: "synth" | "cell" }
interface Preset { id: string; label: string; s: Settings; init: "box" | Circle[]; spf?: number }
const D: Settings = { image: "synth", noise: 12, sigma: 1.5, k: 8, c: -0.3, eps: 0.3, a: 3 };
const PRESETS: Preset[] = [
  { id: "wrap", label: "shrink-wrap", s: { ...D, a: 5 }, init: "box", spf: 12 },
  { id: "leak", label: "leak through the weak edge", s: { ...D, c: 1, a: 1 }, init: [{ x: 26, y: 100, r: 6 }] },
  { id: "cshape", label: "C shape: no balloon", s: { ...D, c: 0, eps: 1, a: 2 }, init: [{ x: 100, y: 64, r: 30 }] },
  { id: "noise", label: "noise, no smoothing", s: { ...D, noise: 25, sigma: 0 }, init: "box" },
  { id: "cell", label: "micrograph: nucleus", s: { ...D, image: "cell", noise: 0, sigma: 1.5, k: 4, c: 0.3, eps: 0.5, a: 3 }, init: [{ x: 74, y: 62, r: 6 }], spf: 12 },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .ga-main { flex: 0 1 460px; min-width: 260px; }
[data-demo="${NAME}"] .ga-main canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; cursor: crosshair; touch-action: none; }
[data-demo="${NAME}"] .ga-cap { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .ga-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .35em; }
[data-demo="${NAME}"] .ga-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .ga-row { display: flex; gap: .35em; flex-wrap: wrap; align-items: center; }
[data-demo="${NAME}"] .ga-sliders { display: flex; flex-direction: column; gap: .35em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .ga-main { flex-basis: 440px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; max-width: 560px; flex-basis: 520px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .5em; }
.reveal [data-demo="${NAME}"] .ga-sliders { display: grid; grid-template-columns: 1fr 1fr; column-gap: .8em; row-gap: .1em; }
.reveal [data-demo="${NAME}"] .ga-sliders .qb-slider { grid-template-columns: auto 1fr 2.6em; gap: .3em; }
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
  let spf = p0.spf ?? 8;
  let initShape: "box" | Circle[] = p0.init;
  let viewG = false, arrows = false;

  // ---- layout ----
  const main = htmlEl("div", "ga-main", "", host);
  const cap = htmlEl("div", "ga-cap", "", main);
  const cv = htmlEl("canvas", "", "", main);
  cv.width = S; cv.height = S;
  const img = document.createElement("canvas"); img.width = N; img.height = N;
  const credit = htmlEl("div", "ga-credit", CELL_CREDIT, main);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ga-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ga-lab", "image · view · start (or click the image to add circles)", panel);
  const r1 = htmlEl("div", "ga-row", "", panel);
  const imgBtns = (["synth", "cell"] as const).map(m => button(r1, m === "synth" ? "test image" : "micrograph", () => { st.image = m; clearPresets(); makeImage(); }));
  const viewBtns = [button(r1, "image", () => { viewG = false; draw(); }), button(r1, "g", () => { viewG = true; draw(); })];
  checkbox(r1, "−∇g arrows", arrows, v => { arrows = v; draw(); });
  const r2 = htmlEl("div", "ga-row", "", panel);
  button(r2, "big box", () => { initShape = "box"; clearPresets(); restart(); });
  button(r2, "clear start", () => { initShape = []; clearPresets(); restart(); });
  const sl = htmlEl("div", "ga-sliders", "", panel);
  const nS = slider(sl, "noise", 0, 40, 1, st.noise, v => { st.noise = v; clearPresets(); makeImage(); });
  const sS = slider(sl, "smoothing σ", 0, 4, 0.25, st.sigma, v => { st.sigma = v; clearPresets(); makeEdges(); });
  const kS = slider(sl, "edge scale k", 1, 30, 1, st.k, v => { st.k = v; clearPresets(); makeEdges(); });
  const cS = slider(sl, "balloon c", -1, 1, 0.05, st.c, v => { st.c = v; clearPresets(); makeTerms(); });
  const eS = slider(sl, "curvature ε", 0, 2, 0.05, st.eps, v => { st.eps = v; clearPresets(); makeTerms(); });
  const aS = slider(sl, "advection a", 0, 5, 0.25, st.a, v => { st.a = v; clearPresets(); makeTerms(); });
  const r3 = htmlEl("div", "ga-row", "", panel);
  const playBtn = button(r3, "play", () => anim.toggle());
  button(r3, "step", () => { anim.pause(); advance(1); draw(); });
  button(r3, "reset", () => { anim.pause(); restart(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Yellow: the zero contour. Click to add a start circle (radius 8). The test image has one weak edge (right side of the lower-left blob) and a C shape; 'wrong px' counts pixels that differ from the true objects.", panel);

  const anim = new Animator(host, () => frame(), p => { playBtn.textContent = p ? "pause" : "play"; });

  function clearPresets(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function load(p: Preset): void {
    anim.pause();
    Object.assign(st, p.s); spf = p.spf ?? 8; initShape = Array.isArray(p.init) ? p.init.map(c => ({ ...c })) : p.init;
    nS.set(st.noise); sS.set(st.sigma); kS.set(st.k); cS.set(st.c); eS.set(st.eps); aS.set(st.a);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }

  // ---- image → g → terms ----
  const synth = synthClean();
  let cellImg: Float64Array | null = null;
  let I: Float64Array = new Float64Array(N * N), E: EdgeMap | null = null, T: LS.SpeedTerms = {}, dt = 0;
  function makeImage(): void {
    if (st.image === "cell" && !cellImg) { loadCell(N).then(g => { cellImg = Float64Array.from(g.data); makeImage(); }); return; }
    I = addNoise(st.image === "cell" ? cellImg! : synth.img, st.noise);
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    makeEdges();
    restart();
  }
  function makeEdges(): void { E = edgeMap(I, st.sigma, st.k); makeTerms(); }
  function makeTerms(): void {
    T = gacTerms(E!, st);
    const lim = LS.cflLimit(T, N * N);
    dt = Number.isFinite(lim) ? 0.8 * lim : 0;
    sync(); draw();
  }

  // ---- evolution ----
  let phi: Float64Array = new Float64Array(N * N), steps = 0, t = 0, still = 0;
  const work = new Float64Array(N * N);
  function restart(): void {
    phi = initShape === "box" ? boxPhi() : circlesPhi(initShape);
    steps = 0; t = 0; still = 0;
    sync(); draw();
  }
  function advance(k: number): boolean {
    if (dt <= 0) return false;
    let moved = 0;
    for (let i = 0; i < k; i++) {
      const before = LS.insideCount(phi);
      LS.lsStep(phi, N, N, T, dt, "upwind", work);
      steps++; t += dt;
      if (steps % REINIT === 0) LS.reinitExact(phi, N, N);
      moved += Math.abs(LS.insideCount(phi) - before);
    }
    still = moved === 0 ? still + k : 0;
    return still < 300;
  }
  function frame(): boolean {
    const ok = advance(spf);
    draw();
    const n = LS.insideCount(phi);
    return ok && n > 0 && n < N * N;
  }

  cv.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    const r = cv.getBoundingClientRect();
    const c = { x: (ev.clientX - r.left) / r.width * N - 0.5, y: (ev.clientY - r.top) / r.height * N - 0.5, r: 8 };
    if (initShape === "box") initShape = [];
    initShape.push(c);
    clearPresets();
    phi = LS.union(phi, circlesPhi([c]));
    draw();
  });

  // ---- drawing ----
  function sync(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", ["synth", "cell"][i] === st.image));
    viewBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === viewG));
  }
  function strokeContour(ctx: CanvasRenderingContext2D, C: Contour): void {
    const k = S / N;
    ctx.beginPath();
    for (let s = 0; s < C.segments.length; s += 2) {
      const a = C.segments[s], b = C.segments[s + 1];
      ctx.moveTo((C.points[2 * a] + 0.5) * k, (C.points[2 * a + 1] + 0.5) * k);
      ctx.lineTo((C.points[2 * b] + 0.5) * k, (C.points[2 * b + 1] + 0.5) * k);
    }
    ctx.stroke();
  }
  function draw(): void {
    sync();
    if (!E) return;
    const ictx = img.getContext("2d")!, id = ictx.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const v = viewG ? 255 * E.g[p] : Math.max(0, Math.min(255, I[p]));
      id.data[4 * p] = v; id.data[4 * p + 1] = v; id.data[4 * p + 2] = v; id.data[4 * p + 3] = 255;
    }
    ictx.putImageData(id, 0, 0);
    const ctx = cv.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, S, S);
    if (arrows) {
      ctx.strokeStyle = "#1f77b4"; ctx.fillStyle = "#1f77b4"; ctx.lineWidth = 2;
      let mx = 0; for (let p = 0; p < N * N; p++) mx = Math.max(mx, Math.hypot(E.gx[p], E.gy[p]));
      for (let y = 2; y < N; y += 6) for (let x = 2; x < N; x += 6) {
        const p = y * N + x, vx = -E.gx[p] / (mx || 1), vy = -E.gy[p] / (mx || 1), L = Math.hypot(vx, vy);
        if (L < 0.08) continue;
        const k = S / N, x0 = (x + 0.5) * k, y0 = (y + 0.5) * k, x1 = x0 + vx * 22 / Math.max(L, 0.5), y1 = y0 + vy * 22 / Math.max(L, 0.5);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        const ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 7 * Math.cos(ang - 0.45), y1 - 7 * Math.sin(ang - 0.45)); ctx.lineTo(x1 - 7 * Math.cos(ang + 0.45), y1 - 7 * Math.sin(ang + 0.45)); ctx.fill();
      }
    }
    ctx.strokeStyle = "#111"; ctx.lineWidth = 6;
    const C = LS.zeroContour(phi, N, N);
    strokeContour(ctx, C);
    ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 3;
    strokeContour(ctx, C);
    cap.textContent = viewG ? "edge-stopping function g (white = 1, black = 0)" : "image and the front: click to add a start circle";
    const n = LS.insideCount(phi);
    let wrong = 0;
    if (st.image === "synth") for (let p = 0; p < N * N; p++) if ((phi[p] < 0 ? 1 : 0) !== synth.mask[p]) wrong++;
    const r = (a: string, b: string) => a.padEnd(16) + b;
    ro.textContent = [
      r("speed", `g·(${st.c.toFixed(2)} − ${st.eps.toFixed(2)} κ) + advection ${st.a.toFixed(2)}`),
      r("Δt", dt.toFixed(3)),
      r("steps, time t", `${steps}, ${t.toFixed(1)}`),
      r("pieces", `${C.lines.length}`),
      r("inside", `${n} px`),
      st.image === "synth" ? r("wrong", `${wrong} px`) : r("", ""),
    ].join("\n");
  }

  load(p0);
  const n0 = Number(host.dataset.steps ?? 0);
  if (n0 > 0) {
    const go = () => { if (!E) { setTimeout(go, 50); return; } advance(n0); draw(); };
    go();
  }
}

autoMount(NAME, mount);
