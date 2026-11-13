// Drift correction (FFT registration day).
// Frames mode (default): 40 frames of a micrograph drifting by a seeded random walk, with
// noise, bleaching and an optional change of the sample. Register each frame to the first,
// to the previous one, or to key frames; compare the estimated and true trajectories and the
// average image before and after correction.
// data-mode="smlm": localisations of a blinking structure that drifts; render them per time
// block, correlate all block pairs, solve for the drift by least squares, and compare the
// rendered image before and after correction.
// Options: data-mode, data-preset.
import { htmlEl, slider, presets, autoMount, button } from "../lib/ui";
import { Plot, seriesColor, plotFigure } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { drawScalarImage, inferno } from "../lib/colour";
import {
  makeFrames, registerFrames, undrift, DEFAULTS, DriftSettings, DriftMode, Frames, S,
  makeSmlm, smlmDrift, render, interpDrift, SMLM_DEFAULTS, SmlmSettings, FIELD,
} from "./model";

const NAME = "drift";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .dr-imgs { flex: 0 1 300px; min-width: 200px; display: grid; grid-template-columns: 1fr 1fr; gap: .4em; }
.qb-demo[data-demo="${NAME}"].dr-smlm .dr-imgs { flex-basis: 380px; }
.qb-demo[data-demo="${NAME}"].dr-smlm .dr-plot { flex-basis: 320px; max-width: 380px; }
.qb-demo[data-demo="${NAME}"] .dr-plot { flex: 1 1 360px; min-width: 260px; max-width: 470px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 240px; max-width: 310px; gap: .3em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { padding: .12em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre; line-height: 1.3; }
.qb-demo[data-demo="${NAME}"] .dr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .dr-credit { font-size: 12px; color: var(--qb-muted); grid-column: 1 / 3; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"].dr-smlm .dr-imgs { flex-basis: 420px; }
.reveal .qb-demo[data-demo="${NAME}"].dr-smlm .dr-plot { flex-basis: 360px; max-width: 470px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.reveal .slides .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; line-height: 1.15; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

function fig(parent: Element, caption: string): { cap: HTMLElement; cv: HTMLCanvasElement } {
  const f = htmlEl("figure", "", "", parent);
  const cap = htmlEl("figcaption", "", caption, f);
  return { cap, cv: htmlEl("canvas", "", "", f) };
}

interface FPreset { id: string; label: string; s: Partial<DriftSettings>; mode: DriftMode }
const FPRESETS: FPreset[] = [
  { id: "bleach", label: "bleaching: no problem", s: { bleach: 0.8, noise: 10 }, mode: "first" },
  { id: "previous", label: "to the previous: errors add up", s: { bleach: 0.8, noise: 30 }, mode: "previous" },
  { id: "change", label: "sample changes: to the first fails", s: { change: 0.9, noise: 30 }, mode: "first" },
  { id: "key", label: "key frames: a compromise", s: { change: 0.9, noise: 30 }, mode: "key" },
];

function mount(host: HTMLElement): void {
  injectStyle();
  if (host.dataset.mode === "smlm") { mountSmlm(host); return; }
  const p0 = FPRESETS.find(p => p.id === host.dataset.preset) ?? FPRESETS[0];
  const st = { s: { ...DEFAULTS } as DriftSettings, mode: "first" as DriftMode, k: 39 };
  let scene: Float64Array | null = null;

  const imgs = htmlEl("div", "dr-imgs", "", host);
  const fk = fig(imgs, "frame"), fr = fig(imgs, "frame 0");
  const fa = fig(imgs, "average, uncorrected"), fc = fig(imgs, "average, corrected");
  htmlEl("div", "dr-credit", CELL_CREDIT, imgs);
  const pf = plotFigure(htmlEl("div", "dr-plot", "", host), "drift: true (lines) and estimated (dots)");
  const P = uniqueClip(new Plot(pf, { width: 460, height: 360, x: [0, 39], y: [-5, 5], xlabel: "frame", ylabel: "drift (px)" }));
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dr-lab", "presets", panel);
  const presetBtns = presets(panel, FPRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "dr-lab", "register each frame to", panel);
  const modes: [DriftMode, string][] = [["first", "the first"], ["previous", "the previous"], ["key", "key frames (every 10)"]];
  const modeBtns = presets(panel, modes.map(([m, l]) => ({ label: l, apply: () => { st.mode = m; schedule(false); } })));
  const kS = slider(panel, "show frame", 0, 39, 1, st.k, v => { st.k = v; drawFrame(); });
  const nS = slider(panel, "noise σ", 0, 60, 2, 10, v => { st.s.noise = v; schedule(true, false); });
  const bS = slider(panel, "bleaching", 0, 0.95, 0.05, 0, v => { st.s.bleach = v; schedule(true, false); });
  const cS = slider(panel, "sample change", 0, 1, 0.05, 0, v => { st.s.change = v; schedule(true, false); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drift = how far the content has moved since frame 0 (x: blue, y: red). Phase correlation with a Hann taper and the upsampled DFT. Sample change: the frames slowly fade into a mirrored copy of the field.", panel);

  function load(p: FPreset): void {
    st.s = { ...DEFAULTS, ...p.s }; st.mode = p.mode;
    presetBtns.forEach((b, i) => b.classList.toggle("active", FPRESETS[i] === p));
    nS.set(st.s.noise); bS.set(st.s.bleach); cS.set(st.s.change);
    schedule(true);
  }

  let frames: Frames | null = null, est: { x: number[]; y: number[] } | null = null, timer = 0;
  function schedule(regen: boolean, now = true): void {
    modeBtns.forEach((b, i) => b.classList.toggle("active", modes[i][0] === st.mode));
    if (!scene) { loadCell().then(g => { scene = Float64Array.from(g.data); schedule(true); }); return; }
    clearTimeout(timer);
    const go = () => {
      if (regen || !frames) frames = makeFrames(st.s, scene);
      const t0 = performance.now();
      est = registerFrames(frames, st.mode);
      draw(performance.now() - t0);
    };
    if (now) go();
    else { ro.textContent = "registering 40 frames…"; timer = window.setTimeout(go, 150); }
  }

  const marks: { remove(): void }[] = [];
  function draw(ms: number): void {
    const f = frames!, e = est!, ks = Array.from({ length: f.n }, (_, k) => k);
    const all = [...f.dx, ...f.dy, ...e.x, ...e.y].filter(v => Math.abs(v) < 40);
    const lo = Math.min(...all), hi = Math.max(...all), pd = 0.1 * (hi - lo || 1);
    marks.forEach(m => m.remove()); marks.length = 0;
    P.setDomain([0, f.n - 1], [lo - pd, hi + pd]);
    marks.push(P.line(ks, f.dx, { color: seriesColor(0), width: 3, label: "x" }));
    marks.push(P.line(ks, f.dy, { color: seriesColor(1), width: 3, label: "y" }));
    marks.push(P.scatter(ks, e.x, { color: seriesColor(0), r: 5 }));
    marks.push(P.scatter(ks, e.y, { color: seriesColor(1), r: 5 }));
    P.legend(undefined, "tl");
    // averages
    const avg = new Float64Array(S * S), cor = new Float64Array(S * S);
    for (let k = 0; k < f.n; k++) {
      const u = undrift(f.frames[k], e.x[k], e.y[k]);
      for (let i = 0; i < S * S; i++) { avg[i] += f.frames[k][i] / f.n; cor[i] += u[i] / f.n; }
    }
    const crop = (a: Float64Array) => a;   // full frame; borders of the corrected average are blurred by the shifts
    drawScalarImage(fa.cv, crop(avg), S, S, { lo: 0, hi: 255 });
    drawScalarImage(fc.cv, crop(cor), S, S, { lo: 0, hi: 255 });
    drawScalarImage(fr.cv, f.frames[0], S, S, { lo: 0, hi: 255 });
    drawFrame();
    const err = ks.map(k => Math.hypot(e.x[k] - f.dx[k], e.y[k] - f.dy[k]));
    const worst = err.indexOf(Math.max(...err));
    ro.textContent = [
      `registered to ${modes.find(m => m[0] === st.mode)![1]}`,
      `error: max ${err[worst].toFixed(2)} px (frame ${worst})`,
      `       last frame ${err[f.n - 1].toFixed(2)} px`,
      `       rms ${Math.sqrt(err.reduce((a, b) => a + b * b, 0) / f.n).toFixed(2)} px`,
      `true drift at the end (${f.dx[f.n - 1].toFixed(1)}, ${f.dy[f.n - 1].toFixed(1)})`,
      `${f.n - 1} registrations, ${ms.toFixed(0)} ms`,
    ].join("\n");
  }
  function drawFrame(): void {
    if (!frames) return;
    fk.cap.textContent = `frame ${st.k}`;
    drawScalarImage(fk.cv, frames.frames[st.k], S, S, { lo: 0, hi: 255 });
  }
  void kS;
  load(p0);
}

// ---------------------------------------------------------------------------------------

interface SPreset { id: string; label: string; s: Partial<SmlmSettings> }
const SPRESETS: SPreset[] = [
  { id: "smlm", label: "corrected localisations sharpen", s: {} },
  { id: "few", label: "3 blocks: too coarse", s: { blocks: 3 } },
  { id: "many", label: "40 blocks: finer in time", s: { blocks: 40 } },
  { id: "strong", label: "strong drift", s: { step: 0.12 } },
];

function mountSmlm(host: HTMLElement): void {
  host.classList.add("dr-smlm");
  const p0 = SPRESETS.find(p => p.id === host.dataset.preset) ?? SPRESETS[0];
  const st: SmlmSettings = { ...SMLM_DEFAULTS };
  const imgs = htmlEl("div", "dr-imgs", "", host);
  const fu = fig(imgs, "localisations, uncorrected"), fcr = fig(imgs, "corrected");
  const pf = plotFigure(htmlEl("div", "dr-plot", "", host), "drift: true (lines), per block (dots)");
  const P = uniqueClip(new Plot(pf, { width: 460, height: 360, x: [0, 4000], y: [-5, 5], xlabel: "frame", ylabel: "drift (camera px)", xticks: [0, 1000, 2000, 3000, 4000] }));
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dr-lab", "presets", panel);
  const presetBtns = presets(panel, SPRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const bS = slider(panel, "time blocks", 2, 40, 1, st.blocks, v => { st.blocks = v; run(false); });
  const pS = slider(panel, "precision σ", 0.05, 0.6, 0.05, st.precision, v => { st.precision = v; run(false); });
  const dS = slider(panel, "drift strength", 0, 0.15, 0.01, st.step, v => { st.step = v; run(false); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "4000 camera frames of a 64 × 64 px field; each frame localises one to three blinking molecules. Each time block is rendered as an image; every pair of blocks is cross-correlated, and least squares on the graph of pairs gives one drift per block.", panel);

  function load(p: SPreset): void {
    Object.assign(st, SMLM_DEFAULTS, p.s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", SPRESETS[i] === p));
    bS.set(st.blocks); pS.set(st.precision); dS.set(st.step);
    run();
  }
  let timer = 0;
  const marks: { remove(): void }[] = [];
  function run(now = true): void {
    clearTimeout(timer);
    const go = () => {
      const t0 = performance.now();
      const m = makeSmlm(st), d = smlmDrift(m), ms = performance.now() - t0;
      // estimated drift per frame, anchored like the truth (frame 0 = 0)
      const ex = (t: number) => interpDrift(d.centres, d.x, t) - interpDrift(d.centres, d.x, 0);
      const ey = (t: number) => interpDrift(d.centres, d.y, t) - interpDrift(d.centres, d.y, 0);
      const scale = 4, w = FIELD * scale;
      const u = render(m, 0, st.frames, scale), c = render(m, 0, st.frames, scale, ex, ey);
      let hi = 0;
      for (let i = 0; i < u.length; i++) hi = Math.max(hi, u[i], c[i]);
      hi *= 0.6;
      drawScalarImage(fu.cv, u, w, w, { map: inferno, lo: 0, hi });
      drawScalarImage(fcr.cv, c, w, w, { map: inferno, lo: 0, hi });
      marks.forEach(mk => mk.remove()); marks.length = 0;
      const ts = Array.from({ length: st.frames / 20 }, (_, k) => 20 * k);
      const tx = ts.map(t => m.dx[t]), ty = ts.map(t => m.dy[t]);
      const ox = interpDrift(d.centres, d.x, 0), oy = interpDrift(d.centres, d.y, 0);
      const all = [...tx, ...ty, ...d.x.map(v => v - ox), ...d.y.map(v => v - oy)];
      const lo = Math.min(...all), hi2 = Math.max(...all), pd = 0.1 * (hi2 - lo || 1);
      P.setDomain([0, st.frames], [lo - pd, hi2 + pd]);
      marks.push(P.line(ts, tx, { color: seriesColor(0), width: 3, label: "x" }));
      marks.push(P.line(ts, ty, { color: seriesColor(1), width: 3, label: "y" }));
      marks.push(P.line(ts, ts.map(ex), { color: seriesColor(0), width: 1.5, dash: "5 4" }));
      marks.push(P.line(ts, ts.map(ey), { color: seriesColor(1), width: 1.5, dash: "5 4" }));
      marks.push(P.scatter(d.centres, d.x.map(v => v - ox), { color: seriesColor(0), r: 6 }));
      marks.push(P.scatter(d.centres, d.y.map(v => v - oy), { color: seriesColor(1), r: 6 }));
      P.legend(undefined, "tl");
      const err = ts.map(t => Math.hypot(ex(t) - m.dx[t], ey(t) - m.dy[t]));
      ro.textContent = [
        `${m.t.length} localisations`,
        `${st.blocks} blocks of ${Math.round(st.frames / st.blocks)} frames`,
        `${d.pairs} block pairs correlated`,
        `drift error: max ${Math.max(...err).toFixed(2)}, rms ${Math.sqrt(err.reduce((a, b) => a + b * b, 0) / err.length).toFixed(2)} px`,
        `localisation precision ${st.precision} px`,
        `${ms.toFixed(0)} ms`,
      ].join("\n");
    };
    if (now) go();
    else { ro.textContent = "correlating blocks…"; timer = window.setTimeout(go, 150); }
  }
  load(p0);
}


/** lib/plot.ts numbers its clip paths per bundle, so two demos on one page can share an id;
 *  give this plot's clip path a page-unique id (local workaround). */
function uniqueClip(p: Plot): Plot {
  const cp = p.svg.querySelector("clipPath");
  if (cp) {
    const id = "drift-clip-" + Math.random().toString(36).slice(2, 10);
    cp.id = id;
    p.layer.setAttribute("clip-path", `url(#${id})`);
  }
  return p;
}

autoMount(NAME, mount);
