// Local (adaptive) thresholds under uneven illumination. A synthetic bright-field-like
// image (dark discs of several sizes on a bright background, ground truth known) is lit
// by a left-to-right ramp and a vignette, plus noise; or the course micrograph with an
// artificial shading (no ground truth). Methods: global Otsu, Otsu after dividing by a
// smooth background estimate, mean − C, Niblack (T = m + k s), Sauvola
// (T = m (1 + k (s/R − 1))), all for dark objects (object where I < T). Panels: the image
// with the mask, the threshold surface T(x) as an image, and the profile along a row
// (click the image to move it) with I and T. Readout: Dice against the ground truth.
// Auto-mounts on <div data-demo="localthresh">. Options: data-preset="global|divide|small|large|niblack|sauvola|cell", data-method.
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey, createGrey, drawGrey } from "../lib/imageops";
import { confusion, dice } from "../lib/metrics";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { drawOverlay, injectStyle } from "../histthresh/scenes";
import { N, Method, discTruth, makeImage as makeImg, segment } from "./model";

const NAME = "localthresh";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .lt-img { flex: 0 1 300px; min-width: 220px; }
[data-demo="${NAME}"] .lt-img canvas { cursor: row-resize; touch-action: none; }
[data-demo="${NAME}"] .lt-mid { flex: 1 1 360px; min-width: 260px; max-width: 450px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .lt-row { display: flex; gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .lt-row figure { flex: 0 0 46%; }
[data-demo="${NAME}"] .lt-row .qb-readout { flex: 1 1 auto; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 290px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
[data-demo="${NAME}"] .lt-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .lt-credit { font-size: 12px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
`;

const METHODS: { id: Method; label: string }[] = [
  { id: "global", label: "global Otsu" }, { id: "divide", label: "÷ background, Otsu" },
  { id: "meanC", label: "mean − C" }, { id: "niblack", label: "Niblack" }, { id: "sauvola", label: "Sauvola" },
];

interface State { source: "discs" | "cell"; method: Method; r: number; k: number; C: number; bg: number; light: number; vignette: number; noise: number; row: number }
const BASE: State = { source: "discs", method: "global", r: 25, k: 0.3, C: 10, bg: 30, light: 1.0, vignette: 0.3, noise: 6, row: 128 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "global", label: "global Otsu fails", s: {} },
  { id: "divide", label: "divide by the background", s: { method: "divide" } },
  { id: "small", label: "window too small", s: { method: "meanC", r: 5, C: 10 } },
  { id: "large", label: "window too large", s: { method: "meanC", r: 120, C: 10 } },
  { id: "niblack", label: "Niblack in empty background", s: { method: "niblack", r: 25, k: -0.2 } },
  { id: "sauvola", label: "Sauvola fixes it", s: { method: "sauvola", r: 25, k: 0.3 } },
  { id: "cell", label: "micrograph, shaded", s: { source: "cell", method: "sauvola", r: 25, k: 0.2, noise: 0 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE };

  const fig = htmlEl("figure", "lt-img", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  const credit = htmlEl("div", "lt-credit", CELL_CREDIT, fig);
  const mid = htmlEl("div", "lt-mid", "", host);
  const prof = new Plot(mid, { width: 450, height: 250, x: [0, 255], y: [0, 255], xlabel: "x along the row", ylabel: "grey level", xticks: [0, 50, 100, 150, 200, 250] });
  const row = htmlEl("div", "lt-row", "", mid);
  const tfig = htmlEl("figure", "", "", row);
  htmlEl("figcaption", "", "threshold T(x)", tfig);
  const tcv = htmlEl("canvas", "", "", tfig);
  const ro = htmlEl("div", "qb-readout", "", row);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "lt-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "lt-lab", "method", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = METHODS.map(m => button(mRow, m.label, () => { st.method = m.id; if (m.id === "niblack" && st.k > 0) st.k = -0.2; if (m.id === "sauvola" && st.k < 0) st.k = 0.3; sync(); compute(); }));
  const rS = slider(panel, "window r", 1, 128, 1, st.r, v => { st.r = v; compute(); }, v => `${2 * v + 1}`);
  const kS = slider(panel, "k", -0.6, 0.6, 0.05, st.k, v => { st.k = v; compute(); });
  const cS = slider(panel, "C", 0, 60, 1, st.C, v => { st.C = v; compute(); });
  const bS = slider(panel, "bg. scale", 4, 60, 1, st.bg, v => { st.bg = v; compute(); });
  htmlEl("div", "lt-lab", "image", panel);
  const lS = slider(panel, "light ramp", 0, 1.4, 0.05, st.light, v => { st.light = v; makeImage(); });
  const vS = slider(panel, "vignette", 0, 0.6, 0.05, st.vignette, v => { st.vignette = v; makeImage(); });
  const nS = slider(panel, "noise σ", 0, 30, 1, st.noise, v => { st.noise = v; makeImage(); });
  button(panel, "reset", () => load(p0));

  const show = (s: { input: HTMLInputElement }, on: boolean) => { (s.input.parentElement as HTMLElement).style.display = on ? "" : "none"; };
  function sync(): void {
    rS.set(st.r); kS.set(st.k); cS.set(st.C); bS.set(st.bg); lS.set(st.light); vS.set(st.vignette); nS.set(st.noise);
    mBtns.forEach((b, i) => b.classList.toggle("active", METHODS[i].id === st.method));
    const local = st.method === "meanC" || st.method === "niblack" || st.method === "sauvola";
    show(rS, local); show(kS, st.method === "niblack" || st.method === "sauvola"); show(cS, st.method === "meanC"); show(bS, st.method === "divide");
    credit.style.display = st.source === "cell" ? "" : "none";
  }

  let BASE_METHOD: Method | null = null;
  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, BASE, p.s);
    if (BASE_METHOD) { st.method = BASE_METHOD; BASE_METHOD = null; }
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    sync();
    makeImage();
  }

  const truth = discTruth();
  let cell: Grey | null = null, img: Grey = createGrey(N, N);
  function makeImage(): void {
    if (st.source === "cell" && !cell) { loadCell().then(c => { cell = c; makeImage(); }); return; }
    img = makeImg(st, truth, cell);
    compute();
  }

  let T: Grey = createGrey(N, N), mask: Uint8Array = new Uint8Array(N * N), info = "";
  function compute(): void {
    ({ T, mask, info } = segment(img, st));
    draw();
  }

  function draw(): void {
    const rowMask = new Uint8Array(N * N);
    for (let x = 0; x < N; x++) rowMask[st.row * N + x] = 1;
    drawOverlay(cv, img, [{ mask, rgb: [255, 127, 14], alpha: 0.55 }, { mask: rowMask, rgb: [0, 160, 255], alpha: 0.9 }]);
    cap.textContent = "mask: I < T(x) (click to move the row)";
    drawGrey(tcv, T, { mode: "clip", lo: 0, hi: 255 });
    // profile
    const xs = Array.from({ length: N }, (_, i) => i);
    const I = xs.map(x => img.data[st.row * N + x]), Tr = xs.map(x => T.data[st.row * N + x]);
    prof.clear();
    if (st.source === "discs") {
      const tr = xs.map(x => (truth[st.row * N + x] ? 255 : 0));
      prof.band(xs.map(x => x - 0.5), xs.map(() => 0), tr, { color: "var(--qb-muted, #999)", fillOpacity: 0.18 });
    }
    prof.step(xs.map(x => x - 0.5), I, { color: "var(--qb-fg, #333)", width: 1.3, label: "I" });
    prof.line(xs, Tr, { color: seriesColor(3), width: 3.5, label: "T" });
    prof.legend(undefined, "tl");
    const lines = [info];
    if (st.source === "discs") {
      const c = confusion(mask, truth);
      lines.push(`Dice = ${dice(c).toFixed(3)}`, `FP = ${c.fp}`, `FN = ${c.fn}`);
    } else lines.push("(no ground truth)");
    lines.push(`row y = ${st.row}`);
    ro.textContent = lines.join("\n");
  }

  const setRow = (ev: PointerEvent) => {
    const r = cv.getBoundingClientRect();
    st.row = Math.max(0, Math.min(N - 1, Math.floor((ev.clientY - r.top) / r.height * N)));
    draw();
  };
  let down = false;
  cv.addEventListener("pointerdown", ev => { ev.preventDefault(); down = true; cv.setPointerCapture(ev.pointerId); setRow(ev); });
  cv.addEventListener("pointermove", ev => { if (down) setRow(ev); });
  cv.addEventListener("pointerup", () => { down = false; });
  cv.addEventListener("pointercancel", () => { down = false; });

  if (host.dataset.method && METHODS.some(m => m.id === host.dataset.method)) { BASE_METHOD = host.dataset.method as Method; }
  load(p0);
}

autoMount(NAME, mount);
