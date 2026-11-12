// unetseg: a tiny U-Net (two levels, 8-16-32 channels, about 30,000 weights, trained offline by
// train_unet.py on synthetic nuclei images) segments a 64 × 64 image. Panels: input, nucleus
// probability per pixel, the mask at a threshold, the errors against the true mask (TP white,
// FP red, FN blue) with Dice and IoU, and Otsu's threshold as the baseline. Click a level of the
// architecture to see its feature maps. Weight sets: U-Net (cross-entropy), no skip connections,
// Dice loss. The sliders move the input away from the training images (domain shift).
//
// <div data-demo="unetseg" data-preset="otsu"></div>
import { autoMount, button, htmlEl, presets as presetRow, slider, svgEl } from "../lib/ui";
import { drawGrey } from "../lib/imageops";
import { sample, N, NO_SHIFT, type Shift } from "./data";
import { runUnet, countWeights, type WeightSet, type UnetRun } from "./net";
import { maskAt, otsuMask, scores, losses } from "./model";
import { UNET_INFO } from "./unet_weights";

const NAME = "unetseg";
const STYLE = `
[data-demo="${NAME}"] { flex-direction: column; gap: .5em; }
[data-demo="${NAME}"] .us-panels { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .6em; width: 100%; max-width: 1000px; }
[data-demo="${NAME}"] .us-panel { display: flex; flex-direction: column; align-items: center; gap: 2px; min-width: 0; }
[data-demo="${NAME}"] .us-panel canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .us-cap { font-size: 16px; text-align: center; line-height: 1.2; }
[data-demo="${NAME}"] .us-low { display: flex; flex-wrap: wrap; gap: 1em; align-items: flex-start; width: 100%; }
[data-demo="${NAME}"] .us-arch { flex: 1 1 420px; min-width: 0; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .us-maps { display: grid; grid-template-columns: repeat(16, minmax(0, 1fr)); gap: 2px; }
[data-demo="${NAME}"] .us-maps canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; max-width: 560px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .qb-slider { font-size: 16px; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .us-panels { max-width: 820px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const SETS: { id: WeightSet; label: string }[] = [
  { id: "unet", label: "U-Net" }, { id: "noskip", label: "no skip connections" }, { id: "dice", label: "trained with Dice loss" },
];
const LEVELS: { id: keyof Omit<UnetRun, "prob">; label: string; size: string }[] = [
  { id: "e1", label: "encoder 1", size: "8 × 64²" }, { id: "e2", label: "encoder 2", size: "16 × 32²" }, { id: "b", label: "bottleneck", size: "32 × 16²" },
  { id: "d2", label: "decoder 2", size: "16 × 32²" }, { id: "d1", label: "decoder 1", size: "8 × 64²" },
];
interface Preset { id: string; label: string; k: number; set: WeightSet; shift: Partial<Shift>; t?: number }
const PRESETS: Preset[] = [
  { id: "typical", label: "a typical image", k: 0, set: "unet", shift: {} },
  { id: "otsu", label: "U-Net vs Otsu on uneven illumination", k: 0, set: "unet", shift: { gradient: 100 } },
  { id: "noskip", label: "no skips: small nuclei", k: 2, set: "noskip", shift: { size: 0.5 } },
  { id: "touching", label: "touching nuclei merge", k: 4, set: "unet", shift: {} },
  { id: "noisy", label: "too noisy: outside the training range", k: 0, set: "unet", shift: { noise: 30 } },
  { id: "dim", label: "dimmer microscope: domain shift", k: 5, set: "unet", shift: { brightness: 0.5 } },
  { id: "dice", label: "Dice loss: probabilities near 0 or 1", k: 7, set: "dice", shift: {} },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let k = start.k, set: WeightSet = start.set, shift: Shift = { ...NO_SHIFT, ...start.shift }, t = start.t ?? 0.5, level: (typeof LEVELS)[number] | null = null;
  let run: UnetRun, s: ReturnType<typeof sample>, pending = 0;

  const panels = htmlEl("div", "us-panels", "", host);
  const panel = (cap: string) => { const p = htmlEl("div", "us-panel", "", panels); const c = htmlEl("canvas", "", "", p); const t = htmlEl("div", "us-cap", cap, p); return { c, t }; };
  const pIn = panel("input"), pProb = panel("probability (nucleus)"), pMask = panel("mask"), pErr = panel("errors"), pOtsu = panel("Otsu threshold");
  const low = htmlEl("div", "us-low", "", host);
  const arch = htmlEl("div", "us-arch", "", low);
  const svg = svgEl("svg", { viewBox: "0 0 520 215", width: "100%", style: "max-width:520px", role: "img", "aria-label": "U-Net architecture" }, arch);
  const mapsHead = htmlEl("div", "us-cap", "", arch);
  const maps = htmlEl("div", "us-maps", "", arch);
  const ctl = htmlEl("div", "qb-panel", "", low);
  const setRow = htmlEl("div", "qb-presets", "", ctl);
  const setBtns = SETS.map(q => button(setRow, q.label, () => { set = q.id; recompute(); }));
  const r1 = htmlEl("div", "qb-presets", "", ctl);
  button(r1, "new image", () => { k++; recompute(); });
  button(r1, "back to training conditions", () => { shift = { ...NO_SHIFT }; syncSliders(); recompute(); });
  const sT = slider(ctl, "threshold", 0.05, 0.95, 0.05, t, v => { t = v; drawResults(); });
  const sNoise = slider(ctl, "extra noise", 0, 40, 1, shift.noise, v => { shift.noise = v; schedule(); });
  const sGrad = slider(ctl, "illumination gradient", -1, 200, 1, shift.gradient, v => { shift.gradient = v; schedule(); }, v => (v < 0 ? "random" : String(v)));
  const sBright = slider(ctl, "brightness ×", 0.3, 2.5, 0.05, shift.brightness, v => { shift.brightness = v; schedule(); });
  const sSize = slider(ctl, "nucleus size ×", 0.4, 2, 0.05, shift.size, v => { shift.size = v; schedule(); });
  const readout = htmlEl("div", "qb-readout", "", ctl);
  const presetBtns = presetRow(ctl, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));

  function syncSliders(): void { sT.set(t); sNoise.set(shift.noise); sGrad.set(shift.gradient); sBright.set(shift.brightness); sSize.set(shift.size); }
  function schedule(): void { if (!pending) pending = requestAnimationFrame(() => { pending = 0; recompute(); }); }

  function drawArch(): void {
    svg.replaceChildren();
    // boxes: e1 (top left), e2, b (bottom), d2, d1 (top right); skips as dashed arrows
    const pos: Record<string, [number, number, number]> = { e1: [10, 28, 70], e2: [105, 88, 70], b: [210, 145, 80], d2: [315, 88, 70], d1: [420, 28, 70] };
    const skip = set !== "noskip";
    const line = (x1: number, y1: number, x2: number, y2: number, dash = "") => svgEl("line", { x1, y1, x2, y2, stroke: "#777", "stroke-width": 2.5, "stroke-dasharray": dash }, svg);
    line(60, 74, 120, 88); line(160, 134, 230, 145); line(290, 145, 355, 134); line(395, 88, 450, 74);
    if (skip) { line(100, 51, 420, 51, "7 5"); line(195, 111, 315, 111, "7 5"); }
    const t1 = svgEl("text", { x: 260, y: 16, "text-anchor": "middle", style: "font-size:15px", fill: "#666" }, svg); t1.textContent = skip ? "skip connections (copy and concatenate)" : "no skip connections";
    for (const L of LEVELS) {
      const [x, y, w] = pos[L.id], on = level?.id === L.id;
      const r = svgEl("rect", { x, y, width: w + 20, height: 46, rx: 6, fill: on ? "#ffd8a8" : "#e8eef6", stroke: on ? "#d62728" : "#557", "stroke-width": on ? 3 : 1.5 }, svg);
      r.style.cursor = "pointer";
      r.addEventListener("click", () => { level = level?.id === L.id ? null : L; drawArch(); drawMaps(); });
      const a = svgEl("text", { x: x + (w + 20) / 2, y: y + 20, "text-anchor": "middle", style: "font-size:15px;pointer-events:none" }, svg); a.textContent = L.label;
      const b = svgEl("text", { x: x + (w + 20) / 2, y: y + 39, "text-anchor": "middle", style: "font-size:14px;pointer-events:none", fill: "#555" }, svg); b.textContent = L.size;
    }
    const lg = svgEl("text", { x: 10, y: 210, style: "font-size:14px", fill: "#666" }, svg); lg.textContent = "click a level to see its feature maps (after ReLU)";
  }
  function drawMaps(): void {
    maps.replaceChildren();
    if (!level) { mapsHead.textContent = ""; return; }
    const T = run[level.id];
    mapsHead.textContent = `${level.label}: ${T.c} feature maps of ${T.h} × ${T.w}`;
    maps.style.gridTemplateColumns = `repeat(${Math.min(16, T.c)}, minmax(0, 1fr))`;
    for (let c = 0; c < T.c; c++) {
      const d = T.data.subarray(c * T.h * T.w, (c + 1) * T.h * T.w);
      let mx = 1e-6; for (const v of d) mx = Math.max(mx, v);
      const cv = htmlEl("canvas", "", "", maps);
      drawGrey(cv, { width: T.w, height: T.h, data: Float32Array.from(d) }, { mode: "clip", lo: 0, hi: mx });
    }
  }

  function recompute(): void {
    s = sample(200001 + k, shift);
    run = runUnet(s.image, set);
    setBtns.forEach((b, i) => b.classList.toggle("active", SETS[i].id === set));
    drawGrey(pIn.c, { width: N, height: N, data: s.image }, { mode: "clip", lo: 0, hi: 255 });
    drawGrey(pProb.c, { width: N, height: N, data: run.prob }, { mode: "clip", lo: 0, hi: 1 });
    drawArch(); drawMaps(); drawResults();
  }

  function errImage(c: HTMLCanvasElement, pred: Uint8Array): void {
    c.width = N; c.height = N;
    const ctx = c.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const p = pred[i], g = s.mask[i];
      const [r, gg, b] = p && g ? [235, 235, 235] : p ? [214, 39, 40] : g ? [31, 119, 220] : [25, 25, 25];
      id.data[4 * i] = r; id.data[4 * i + 1] = gg; id.data[4 * i + 2] = b; id.data[4 * i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }

  function drawResults(): void {
    const m = maskAt(run.prob, t), sc = scores(m, s.mask), o = otsuMask(s.image), so = scores(o.mask, s.mask), L = losses(run.prob, s.mask);
    drawGrey(pMask.c, { width: N, height: N, data: Float32Array.from(m) }, { mode: "clip", lo: 0, hi: 1 });
    pMask.t.textContent = `mask: probability > ${t.toFixed(2)}`;
    errImage(pErr.c, m);
    pErr.t.textContent = `errors: Dice ${sc.dice.toFixed(3)}, IoU ${sc.iou.toFixed(3)}`;
    errImage(pOtsu.c, o.mask);
    pOtsu.t.textContent = `Otsu (t = ${o.t}): Dice ${so.dice.toFixed(3)}`;
    let mid = 0; for (const p of run.prob) if (p > 0.1 && p < 0.9) mid++;
    readout.textContent = `image ${k + 1}: ${s.params.count} nuclei (${(100 * s.mask.reduce((a, b) => a + b, 0) / (N * N)).toFixed(1)} % of the pixels)\n` +
      `TP ${sc.c.tp}  FP ${sc.c.fp} (red)  FN ${sc.c.fn} (blue)\n` +
      `cross-entropy ${L.ce.toFixed(3)}, soft Dice ${L.softDice.toFixed(3)}, pixels with 0.1 < p < 0.9: ${mid}\n` +
      `${SETS.find(q => q.id === set)!.label}: ${countWeights(set)} weights, validation Dice ${UNET_INFO[set + " val Dice"]}`;
  }

  function applyPreset(p: Preset): void {
    k = p.k; set = p.set; shift = { ...NO_SHIFT, ...p.shift }; t = p.t ?? 0.5; level = null;
    syncSliders(); recompute();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }
  applyPreset(start);
}

autoMount(NAME, mount);
