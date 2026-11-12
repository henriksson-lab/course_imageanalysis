// Sparsity in two bases: a 1D signal of length 256 and its "keep the K largest" approximation
// in the real orthonormal Fourier basis (cosines and sines) and in an orthonormal wavelet
// basis (periodic boundary, 5 levels). One slider K drives both reconstructions. Lower left:
// the sorted coefficient magnitudes of both bases (log scale), or the wavelet coefficients
// laid out by level and position; lower right: relative error ‖x − x̂‖/‖x‖ against K for
// both (log axes; exact from the dropped coefficients, Parseval).
// Auto-mounts on <div data-demo="wavesparse">. Optional: data-wavelets="haar db2 db4 db8"
// (menu; default haar), data-signal="step|spike|…", data-k="16", data-preset="…".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import {
  WAVELETS, WaveletName, Wavelet, dwt, idwt, realFourier, realFourierInverse, keepLargest, tailErrors,
  testSignal, SignalName, SIGNAL_LABELS,
} from "../lib/wavelet";
import { inferno } from "../lib/colour";
import { svgEl } from "../lib/ui";

const NAME = "wavesparse";
const N = 256, LEVELS = 5;
const KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256];
const SIGNALS: SignalName[] = ["spike", "step", "twosteps", "sine", "chirp", "blocks", "bumps", "heavisine", "doppler"];
const COL_F = seriesColor(1), COL_W = seriesColor(0);

interface Preset { id: string; label: string; signal: SignalName; k: number; wavelet?: WaveletName }
const PRESETS: Preset[] = [
  { id: "spike", label: "spike: Fourier needs all", signal: "spike", k: 8 },
  { id: "sine", label: "sine: Fourier needs one", signal: "sine", k: 4 },
  { id: "step", label: "step: Haar wins", signal: "step", k: 8, wavelet: "haar" },
  { id: "doppler", label: "Doppler: both struggle", signal: "doppler", k: 32 },
  { id: "heavisine", label: "HeaviSine: db4 vs Haar", signal: "heavisine", k: 16, wavelet: "db4" },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .ws-left { flex: 1 1 560px; min-width: 0; max-width: 800px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .ws-row { display: flex; gap: .6em; }
[data-demo="${NAME}"] .ws-row > div { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .4em; }
[data-demo="${NAME}"] .ws-legend { font-size: 17px; display: flex; gap: 1.2em; padding-left: 60px; font-weight: 600; }
[data-demo="${NAME}"] .ws-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .ws-left { flex-basis: 720px; max-width: 760px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 300px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const pct = (v: number) => (v >= 0.1 ? (100 * v).toFixed(1) : (100 * v).toFixed(2)) + " %";

function mount(host: HTMLElement): void {
  injectStyle();
  const menu = (host.dataset.wavelets ?? "haar").split(/[\s,]+/).filter(n => n in WAVELETS) as WaveletName[];
  if (!menu.length) menu.push("haar");
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init = {
    signal: (p0?.signal ?? (SIGNALS.includes(host.dataset.signal as SignalName) ? host.dataset.signal : "step")) as SignalName,
    k: p0?.k ?? Number(host.dataset.k ?? 8),
    wavelet: (p0?.wavelet && menu.includes(p0.wavelet) ? p0.wavelet : menu[0]) as WaveletName,
  };
  const st = { ...init, view: "sorted" as "sorted" | "levels" };

  const left = htmlEl("div", "ws-left", "", host);
  const legend = htmlEl("div", "ws-legend", "", left);
  const top = htmlEl("div", "", "", left);
  const pSig = new Plot(top, { width: 800, height: 230, x: [0, N], y: [-1, 1], ylabel: "x", xticks: [0, 64, 128, 192, 256], margin: { b: 30 } });
  const row = htmlEl("div", "ws-row", "", left);
  const boxC = htmlEl("div", "", "", row), boxE = htmlEl("div", "", "", row);
  const pErr = new Plot(boxE, { width: 400, height: 300, x: [1, N], y: [1e-4, 1], xlog: true, ylog: true, xlabel: "K kept", ylabel: "relative error" });
  let pCoef: Plot | null = null;

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ws-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.filter(p => !p.wavelet || menu.includes(p.wavelet)).map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ws-lab", "signal", panel);
  const sigRow = htmlEl("div", "qb-presets", "", panel);
  const sigBtns = SIGNALS.map(s => button(sigRow, SIGNAL_LABELS[s], () => { st.signal = s; clearPresets(); compute(); }));
  let wBtns: HTMLButtonElement[] = [];
  if (menu.length > 1) {
    htmlEl("div", "ws-lab", "wavelet", panel);
    const wRow = htmlEl("div", "qb-presets", "", panel);
    wBtns = menu.map(w => button(wRow, WAVELETS[w].label, () => { st.wavelet = w; clearPresets(); compute(); }));
  }
  const kIdx = (k: number) => { let b = 0; KS.forEach((v, i) => { if (Math.abs(v - k) < Math.abs(KS[b] - k)) b = i; }); return b; };
  const kS = slider(panel, "keep K", 0, KS.length - 1, 1, kIdx(st.k), v => { st.k = KS[v]; clearPresets(); update(); }, v => String(KS[v]));
  htmlEl("div", "ws-lab", "lower left panel", panel);
  const vRow = htmlEl("div", "qb-presets", "", panel);
  const vBtns = [
    button(vRow, "sorted |c|", () => { st.view = "sorted"; drawCoef(); syncButtons(); }),
    button(vRow, "wavelet coefficients by level", () => { st.view = "levels"; drawCoef(); syncButtons(); }),
  ];
  button(panel, "reset", () => { Object.assign(st, init); kS.set(kIdx(st.k)); clearPresets(); if (p0) load(p0); else compute(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Both bases are orthonormal, so the squared error is exactly the sum of the dropped squared coefficients.", panel);

  function clearPresets(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function load(p: Preset): void {
    st.signal = p.signal; st.k = p.k;
    if (p.wavelet && menu.includes(p.wavelet)) st.wavelet = p.wavelet;
    kS.set(kIdx(st.k));
    compute();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS.filter(q => !q.wavelet || menu.includes(q.wavelet))[i] === p));
  }
  function syncButtons(): void {
    sigBtns.forEach((b, i) => b.classList.toggle("active", SIGNALS[i] === st.signal));
    wBtns.forEach((b, i) => b.classList.toggle("active", menu[i] === st.wavelet));
    vBtns.forEach((b, i) => b.classList.toggle("active", ["sorted", "levels"][i] === st.view));
  }

  // ---- computation ----
  type F = Float64Array<ArrayBufferLike>;
  let x: F = new Float64Array(N), cF: F = new Float64Array(N), cW: F = new Float64Array(N), eF: F = new Float64Array(N + 1), eW: F = new Float64Array(N + 1);
  let w: Wavelet = WAVELETS.haar;
  function compute(): void {
    w = WAVELETS[st.wavelet];
    x = testSignal(st.signal, N);
    cF = realFourier(x);
    cW = dwt(x, w, LEVELS);
    eF = tailErrors(cF); eW = tailErrors(cW);
    const lo = Math.min(...x), hi = Math.max(...x), pad = Math.max(0.12 * (hi - lo), 0.05);
    pSig.setDomain([0, N], [lo - pad, hi + pad]);
    pSig.clear();
    marks = null;
    drawErr();
    drawCoef();
    syncButtons();
    update();
  }

  let marks: { x: ReturnType<Plot["line"]>; f: ReturnType<Plot["line"]>; w: ReturnType<Plot["line"]> } | null = null;
  const idx = Float64Array.from({ length: N }, (_, i) => i);
  let kLineErr: ReturnType<Plot["vline"]> | null = null, kLineCoef: ReturnType<Plot["vline"]> | null = null;

  function update(): void {
    const k = Math.min(st.k, N);
    const rF = realFourierInverse(keepLargest(cF, k)), rW = idwt(keepLargest(cW, k), w, LEVELS);
    if (!marks) {
      marks = {
        x: pSig.line(idx, x, { color: "var(--qb-fg)", width: 2, label: "signal" }),
        f: pSig.line(idx, rF, { color: COL_F, width: 2.5, label: `Fourier, K = ${k}` }),
        w: pSig.line(idx, rW, { color: COL_W, width: 2.5, label: `${w.label}, K = ${k}` }),
      };
    } else { marks.f.set(idx, rF); marks.w.set(idx, rW); }
    legend.innerHTML = `<span style="color:var(--qb-fg)">━ signal</span> <span style="color:${COL_F}">━ Fourier, K = ${k}</span> <span style="color:${COL_W}">━ ${w.label}, K = ${k}</span>`;
    kLineErr?.remove();
    kLineErr = pErr.vline(k, { color: "var(--qb-fg)", width: 2 });
    if (st.view === "sorted" && pCoef) { kLineCoef?.remove(); kLineCoef = pCoef.vline(k + 0.5, { color: "var(--qb-fg)", width: 2 }); }
    if (st.view === "levels") drawCoef();
    const need = (e: ArrayLike<number>, tol: number) => { for (let i = 0; i <= N; i++) if (e[i] <= tol) return i; return N; };
    ro.textContent = [
      `K = ${k} of ${N} coefficients`,
      `relative error`,
      `  Fourier     ${pct(eF[k])}`,
      `  ${w.label.padEnd(10)}  ${pct(eW[k])}`,
      `K for 1 % error`,
      `  Fourier     ${need(eF, 0.01)}`,
      `  ${w.label.padEnd(10)}  ${need(eW, 0.01)}`,
    ].join("\n");
  }

  function drawErr(): void {
    pErr.clear();
    const ks = Float64Array.from({ length: N }, (_, i) => i + 1);
    const clip = (e: ArrayLike<number>) => Float64Array.from(ks, k => Math.max(e[k], 1e-4));
    pErr.line(ks, clip(eF), { color: COL_F, label: "Fourier" });
    pErr.line(ks, clip(eW), { color: COL_W, label: w.label });
    pErr.legend(undefined, "bl");
    kLineErr = null;
  }

  function drawCoef(): void {
    boxC.replaceChildren();
    kLineCoef = null;
    if (st.view === "sorted") {
      const sF = Float64Array.from(cF, Math.abs).sort().reverse(), sW = Float64Array.from(cW, Math.abs).sort().reverse();
      const mx = Math.max(sF[0], sW[0]);
      pCoef = new Plot(boxC, { width: 400, height: 300, x: [1, N], y: [mx * 1e-4, mx * 1.5], xlog: true, ylog: true, xlabel: "rank", ylabel: "|c|, sorted" });
      const ks = Float64Array.from({ length: N }, (_, i) => i + 1);
      const fl = (a: Float64Array) => Float64Array.from(a, v => Math.max(v, mx * 1e-4));
      pCoef.line(ks, fl(sF), { color: COL_F, label: "Fourier" });
      pCoef.line(ks, fl(sW), { color: COL_W, label: w.label });
      pCoef.legend(undefined, "bl");
      kLineCoef = pCoef.vline(st.k + 0.5, { color: "var(--qb-fg)", width: 2 });
    } else {
      pCoef = null;
      drawLevels();
    }
  }

  function drawLevels(): void {
    // rows: approximation (coarsest), then details coarse → fine; each coefficient spans its support
    const VW = 400, VH = 300, l = 70, r = 390, t = 30, b = 270;
    const svg = svgEl("svg", { viewBox: `0 0 ${VW} ${VH}`, width: "100%", class: "qb-plot-svg" }, boxC);
    const rows = LEVELS + 1, rh = (b - t) / rows;
    const kept = keepLargest(cW, st.k);
    let mx = 0;
    for (let i = N >> LEVELS; i < N; i++) mx = Math.max(mx, Math.abs(cW[i]));
    const title = svgEl("text", { x: (l + r) / 2, y: 20, "text-anchor": "middle", "font-size": 18 }, svg);
    title.textContent = `${w.label} coefficients (bright = large)`;
    for (let row = 0; row < rows; row++) {
      const lev = row === 0 ? LEVELS : LEVELS - row + 1;   // row 0: approximation a5; then d5 … d1
      const len = N >> lev, start = row === 0 ? 0 : len;
      const y = t + row * rh;
      for (let k = 0; k < len; k++) {
        const v = cW[start + k], tt = Math.min(1, Math.abs(v) / (row === 0 ? Math.max(...Array.from(cW.subarray(0, len), Math.abs)) : mx || 1));
        const [cr, cg, cb] = inferno(Math.sqrt(tt));
        const rect = svgEl("rect", { x: l + (r - l) * k / len, y, width: (r - l) / len + 0.3, height: rh - 3, fill: `rgb(${255 * cr},${255 * cg},${255 * cb})` }, svg);
        if (kept[start + k] === 0) rect.setAttribute("opacity", "0.35");
      }
      const lab = svgEl("text", { x: l - 6, y: y + rh / 2 + 6, "text-anchor": "end", "font-size": 17 }, svg);
      lab.textContent = row === 0 ? `a${LEVELS}` : `d${lev}`;
    }
    const xl = svgEl("text", { x: (l + r) / 2, y: VH - 6, "text-anchor": "middle", "font-size": 17 }, svg);
    xl.textContent = "position (faded: dropped)";
  }

  if (p0) load(p0); else compute();
}

autoMount(NAME, mount);
