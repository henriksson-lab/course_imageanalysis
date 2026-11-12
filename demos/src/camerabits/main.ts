// How many bits does a camera need? A simulated camera (full-well capacity, read noise,
// ADC bits, exposure) images the micrograph; next to it a photon-transfer plot: noise
// against signal (electrons, log-log) with the shot-noise, read-noise and quantisation-noise
// lines, their total, and points measured on simulated flat patches (same code as the
// image). Readout: gain, quantisation noise, dynamic range in dB and bits, max SNR.
// Auto-mounts on <div data-demo="camerabits">. Optional: data-fwc="30000",
// data-read="1.5", data-bits="16", data-exposure="0.5" (brightest pixel / full well).
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import { Camera, gain, offsetADU, noiseModel, summary, expose, measureFlat, ptcLevels } from "./sensor";
// @ts-ignore -- esbuild inlines the image as a data URL
import cellUrl from "../fourier2d/cell256.png";

const NAME = "camerabits";
const N = 256;
const SEED = 5;
const PATCH = 4096;    // pixels per simulated flat patch
const CREDIT = "Scene: micrograph of a mesenchymal stem cell (TEM), Robert M. Hunt, CC BY 3.0, via Wikimedia Commons.";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .cb-left { flex: 1 1 640px; min-width: 0; max-width: 820px; display: flex; gap: 12px; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 270px; min-width: 180px; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; text-align: center; line-height: 1.2; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .cb-plot { flex: 1 1 340px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 270px; max-width: 340px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-slider { grid-template-columns: 5.6em 1fr 5.8em; }
.qb-demo[data-demo="${NAME}"] .cb-credit { font-size: 13px; color: var(--qb-muted); margin-top: .3em; }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 19px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] svg text { font-size: 18px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function loadCell(): Promise<Float64Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = N; c.height = N;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0, N, N);
      const d = ctx.getImageData(0, 0, N, N).data;
      const out = new Float64Array(N * N);
      for (let k = 0; k < N * N; k++) out[k] = d[4 * k] / 255;
      resolve(out);
    };
    img.onerror = reject;
    img.src = cellUrl;
  });
}

const fmtE = (v: number) => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);

function mount(host: HTMLElement): void {
  injectStyle();
  const left = htmlEl("div", "cb-left", "", host);
  const fig = htmlEl("figure", "", "", left);
  const cap = htmlEl("figcaption", "", "simulated image", fig);
  const cv = htmlEl("canvas", "", "", fig);
  cv.width = N; cv.height = N;
  htmlEl("div", "cb-credit", CREDIT, fig);
  const plotBox = htmlEl("div", "cb-plot", "", left);
  const W = 460, H = 380, PL = 62, PR = 12, PT = 12, PB = 46;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, plotBox);

  const panel = htmlEl("div", "qb-panel", "", host);
  const d0: Camera & { exposure: number } = {
    fwc: Number(host.dataset.fwc ?? 30000), read: Number(host.dataset.read ?? 1.5),
    bits: Number(host.dataset.bits ?? 16), exposure: Number(host.dataset.exposure ?? 0.5),
  };
  const st = { ...d0 };
  const fwcS = slider(panel, "full well", 3, 5.3, 0.05, Math.log10(st.fwc), v => { st.fwc = Math.round(10 ** v / 100) * 100; schedule(); },
    v => (Math.round(10 ** v / 100) * 100).toLocaleString("en-US") + " e⁻");
  const readS = slider(panel, "read noise", 0.5, 30, 0.5, st.read, v => { st.read = v; schedule(); }, v => v.toFixed(1) + " e⁻");
  const bitsS = slider(panel, "ADC bits", 6, 16, 1, st.bits, v => { st.bits = v; schedule(); });
  const expS = slider(panel, "exposure", -3, 0, 0.1, Math.log10(st.exposure), v => { st.exposure = 10 ** v; schedule(); },
    v => (100 * 10 ** v).toPrecision(2) + "%");
  const row = htmlEl("div", "qb-presets", "", panel);
  const set = (fwc: number, read: number, bits: number, exposure: number) => {
    Object.assign(st, { fwc, read, bits, exposure });
    fwcS.set(Math.log10(fwc)); readS.set(read); bitsS.set(bits); expS.set(Math.log10(exposure));
    schedule();
  };
  presets(row, [
    { label: "sCMOS 16 bit", apply: () => set(30000, 1.5, 16, 0.5) },
    { label: "same, 8-bit ADC", apply: () => set(30000, 1.5, 8, 0.5) },
    { label: "noisy, 16-bit ADC", apply: () => set(10000, 10, 16, 0.5) },
    { label: "low light, 8 bit", apply: () => set(30000, 1.5, 8, 0.01) },
    { label: "low light, 12 bit", apply: () => set(30000, 1.5, 12, 0.01) },
  ]);
  button(row, "reset", () => set(d0.fwc, d0.read, d0.bits, d0.exposure));
  const ro = htmlEl("div", "qb-readout", "", panel);

  let cell: Float64Array | null = null;
  loadCell().then(v => { cell = v; schedule(); });

  let pending = false;
  function schedule(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }

  function draw(): void {
    const c: Camera = { fwc: st.fwc, read: st.read, bits: st.bits };
    const g = gain(c), off = offsetADU(c), sm = summary(c);
    const top = st.exposure * c.fwc;   // electrons in the brightest pixel

    // ---- image ----
    let used = 0;
    if (cell) {
      const lam = new Float64Array(N * N);
      for (let k = 0; k < N * N; k++) lam[k] = top * (0.03 + 0.97 * cell[k]);
      const adu = expose(c, lam, SEED);
      const hi = off + top / g;
      const ctx = cv.getContext("2d")!;
      const id = ctx.createImageData(N, N);
      const seen = new Set<number>();
      for (let k = 0; k < N * N; k++) {
        seen.add(adu[k]);
        const v = Math.max(0, Math.min(255, Math.round((adu[k] - off) / Math.max(1e-9, hi - off) * 255)));
        id.data[4 * k] = id.data[4 * k + 1] = id.data[4 * k + 2] = v; id.data[4 * k + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      used = seen.size;
    }
    cap.textContent = `simulated image, ${used} ADU values used`;

    // ---- photon-transfer plot ----
    svg.innerHTML = "";
    const x0 = 0, x1 = 5.5, y0 = -1, y1 = 3;   // log10 ranges
    const X = (s: number) => PL + (Math.log10(Math.max(s, 1e-9)) - x0) / (x1 - x0) * (W - PL - PR);
    const Y = (n: number) => H - PB - (Math.log10(Math.max(n, 1e-9)) - y0) / (y1 - y0) * (H - PT - PB);
    for (let e = x0; e <= x1; e++) {
      svgEl("line", { x1: X(10 ** e), x2: X(10 ** e), y1: PT, y2: H - PB, class: "grid" }, svg);
      svgEl("text", { x: X(10 ** e), y: H - PB + 20, "text-anchor": "middle" }, svg).textContent = e === 0 ? "1" : `10${sup(e)}`;
    }
    for (let e = y0; e <= y1; e++) {
      svgEl("line", { x1: PL, x2: W - PR, y1: Y(10 ** e), y2: Y(10 ** e), class: "grid" }, svg);
      svgEl("text", { x: PL - 6, y: Y(10 ** e) + 5, "text-anchor": "end" }, svg).textContent = e === 0 ? "1" : e === -1 ? "0.1" : `10${sup(e)}`;
    }
    svgEl("text", { x: (PL + W - PR) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = "signal (electrons)";
    const yl = svgEl("text", { x: 16, y: (PT + H - PB) / 2, "text-anchor": "middle", transform: `rotate(-90 16 ${(PT + H - PB) / 2})` }, svg);
    yl.textContent = "noise (electrons RMS)";
    // full well and brightest pixel
    svgEl("line", { x1: X(c.fwc), x2: X(c.fwc), y1: PT, y2: H - PB, stroke: "var(--qb-muted)", "stroke-dasharray": "6 4" }, svg);
    svgEl("text", { x: X(c.fwc) - 4, y: PT + 14, "text-anchor": "end", fill: "var(--qb-muted)" }, svg).textContent = "full well";
    svgEl("line", { x1: X(top), x2: X(top), y1: PT + 20, y2: H - PB, stroke: "var(--qb-c4)", "stroke-width": 2 }, svg);
    const ss: number[] = [];
    for (let e = x0; e <= Math.log10(c.fwc) + 1e-9; e += 0.05) ss.push(10 ** e);
    ss.push(c.fwc);
    const line = (f: (s: number) => number, attrs: Record<string, string | number>) =>
      svgEl("polyline", { points: ss.map(s => `${X(s).toFixed(1)},${Y(f(s)).toFixed(1)}`).join(" "), fill: "none", ...attrs }, svg);
    line(s => noiseModel(c, s).shot, { stroke: "var(--qb-c3)", "stroke-width": 2 });
    line(s => noiseModel(c, s).read, { stroke: "var(--qb-c1)", "stroke-width": 2 });
    line(s => noiseModel(c, s).quant, { stroke: "var(--qb-c2)", "stroke-width": 2, "stroke-dasharray": "7 4" });
    line(s => noiseModel(c, s).total, { stroke: "var(--qb-fg)", "stroke-width": 3, opacity: 0.55 });
    let zeros = 0;
    ptcLevels(c).forEach((S, i) => {
      const m = measureFlat(c, S, PATCH, 100 + i);
      if (m.sd < 10 ** y0) {
        // every pixel of the patch gave the same ADU value: measured noise 0, off the log axis
        zeros++;
        const x = X(S), y = H - PB - 2;
        svgEl("polygon", { points: `${x - 6},${y - 10} ${x + 6},${y - 10} ${x},${y}`, fill: "var(--qb-c2)" }, svg);
      } else svgEl("circle", { cx: X(S), cy: Y(m.sd), r: 5, fill: "#fff", stroke: "var(--qb-fg)", "stroke-width": 2 }, svg);
    });
    if (zeros) svgEl("text", { x: W - PR - 4, y: H - PB - 16, "text-anchor": "end", fill: "var(--qb-c2)" }, svg).textContent = "▼ measured 0: all pixels the same ADU";
    const leg: [string, string, string][] = [["shot √S", "var(--qb-c3)", ""], ["read", "var(--qb-c1)", ""], ["quantisation g/√12", "var(--qb-c2)", "7 4"], ["total", "var(--qb-fg)", ""]];
    leg.forEach(([t, col, dash], i) => {
      const y = PT + 16 + 20 * i;
      svgEl("line", { x1: PL + 10, x2: PL + 36, y1: y - 5, y2: y - 5, stroke: col, "stroke-width": 3, "stroke-dasharray": dash }, svg);
      svgEl("text", { x: PL + 42, y }, svg).textContent = t;
    });
    svgEl("circle", { cx: PL + 23, cy: PT + 16 + 80 - 5, r: 5, fill: "#fff", stroke: "var(--qb-fg)", "stroke-width": 2 }, svg);
    svgEl("text", { x: PL + 42, y: PT + 16 + 80 }, svg).textContent = "measured (flat patch)";

    ro.textContent = [
      `gain g = ${fmtE(g)} e⁻/ADU`,
      `quantisation noise g/√12 = ${fmtE(sm.quantNoise)} e⁻`,
      `read noise ${fmtE(c.read)} e⁻ → ${sm.quantNoise > c.read ? "quantisation dominates!" : "quantisation " + (sm.quantNoise < c.read / 3 ? "negligible" : "comparable")}`,
      `dynamic range ${Math.round(sm.dynRange).toLocaleString("en-US")} : 1`,
      `  = ${sm.dynRangeDb.toFixed(1)} dB = ${sm.dynRangeBits.toFixed(1)} bits`,
      `max SNR √FWC = ${sm.maxSnr.toFixed(0)} = ${sm.maxSnrDb.toFixed(1)} dB`,
      `bits for g ≤ read noise: ${sm.bitsNeeded}`,
      `brightest pixel ${Math.round(top).toLocaleString("en-US")} e⁻ ≈ ${Math.round(top / g).toLocaleString("en-US")} ADU`,
      `offset ${off} ADU`,
    ].join("\n");
  }
}

function sup(e: number): string {
  const m: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶" };
  return String(e).split("").map(ch => m[ch] ?? ch).join("");
}

autoMount(NAME, mount);
