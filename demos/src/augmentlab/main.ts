// augmentlab: two views (data-view).
//  "explore" (default): the course micrograph with augmentations applied live: flips, zoom,
//     rotation, elastic deformation (strength and smoothness, with a deformed grid overlay),
//     gain / offset / gamma, blur, noise; and a strip of random copies within chosen ranges.
//  "train": the crop experiment of the unit. 20 training crops (the elongated cells all lying)
//     and 200 validation crops; the training and validation curves of the network with and
//     without each augmentation setting, recorded with model.ts by
//     tools/figs/augmentation/dump.ts (curves.json), played back. The augmented crops are made live.
//
// <div data-demo="augmentlab" data-view="explore" data-preset="elastic"></div>
// <div data-demo="augmentlab" data-view="train" data-setting="geom"></div>
import { autoMount, button, checkbox, htmlEl, presets as presetRow, slider, rng } from "../lib/ui";
import { Plot } from "../lib/plot";
import { drawGrey, createGrey, type Grey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { applyExplore, randomExplore, augmentCrop, experiment, NO_CHANGE, type Explore, type Aug, type Task } from "./model";
import curves from "./curves.json";

const NAME = "augmentlab";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .al-left { flex: 0 1 420px; min-width: 0; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .al-main { width: 100%; max-width: 400px; aspect-ratio: 1; image-rendering: auto; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .al-strip { display: flex; flex-wrap: wrap; gap: 3px; }
[data-demo="${NAME}"] .al-strip canvas { width: 64px; height: 64px; image-rendering: pixelated; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .al-crops canvas { width: 34px; height: 34px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; max-width: 460px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] .al-trainleft { flex: 1 1 520px; min-width: 0; max-width: 600px; display: flex; flex-direction: column; gap: .3em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .al-main { max-width: 340px; }
.reveal [data-demo="${NAME}"] .al-strip canvas { width: 40px; height: 40px; }
.reveal [data-demo="${NAME}"] .al-crops canvas { width: 28px; height: 28px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
.reveal [data-demo="${NAME}"] .qb-slider { font-size: 15px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const unit = { mode: "clip" as const, lo: 0, hi: 1 };

// ---------------------------------------------------------------------------------------
// explore view

interface ExPreset { id: string; label: string; e: Partial<Explore> }
const EX_PRESETS: ExPreset[] = [
  { id: "none", label: "original", e: {} },
  { id: "flip", label: "flipped", e: { flipH: true } },
  { id: "rotate", label: "rotated 30°", e: { angle: 30 } },
  { id: "zoom", label: "zoom 1.5 (other pixel size)", e: { zoom: 1.5 } },
  { id: "elastic", label: "elastic", e: { alpha: 3, sigma: 10 } },
  { id: "toomuch", label: "elastic, too strong", e: { alpha: 10, sigma: 3 } },
  { id: "dim", label: "dimmer, gamma 0.6", e: { gain: 0.8, offset: 0.05, gamma: 0.6 } },
  { id: "noisy", label: "blur + noise", e: { blur: 1.5, noise: 0.08 } },
];

function mountExplore(host: HTMLElement): void {
  const startP = EX_PRESETS.find(p => p.id === host.dataset.preset) ?? EX_PRESETS[0];
  let e: Explore = { ...NO_CHANGE, ...startP.e };
  let grid = true;
  let img: Grey | null = null, gridImg: Grey | null = null;

  const left = htmlEl("div", "al-left", "", host);
  const canvas = htmlEl("canvas", "al-main", "", left);
  const credit = htmlEl("div", "qb-note", CELL_CREDIT, left);
  void credit;
  const presetBtns = presetRow(left, EX_PRESETS.map(p => ({ label: p.label, apply: () => { e = { ...NO_CHANGE, ...p.e, seed: e.seed }; syncSliders(); draw(); } })));

  const panel = htmlEl("div", "qb-panel", "", host);
  const row1 = htmlEl("div", "qb-presets", "", panel);
  const cH = checkbox(row1, "flip left–right", e.flipH, v => { e.flipH = v; draw(); });
  const cV = checkbox(row1, "flip up–down", e.flipV, v => { e.flipV = v; draw(); });
  const sA = slider(panel, "rotation °", -180, 180, 1, e.angle, v => { e.angle = v; draw(); });
  const sZ = slider(panel, "zoom", -0.3, 0.3, 0.01, Math.log10(e.zoom), v => { e.zoom = 10 ** v; draw(); }, v => (10 ** v).toFixed(2));
  const sAl = slider(panel, "elastic: shift px", 0, 15, 0.5, e.alpha, v => { e.alpha = v; draw(); });
  const sSg = slider(panel, "elastic: smoothness", 2, 20, 1, e.sigma, v => { e.sigma = v; draw(); });
  const sG = slider(panel, "gain", 0.5, 1.5, 0.01, e.gain, v => { e.gain = v; draw(); });
  const sO = slider(panel, "offset", -0.3, 0.3, 0.01, e.offset, v => { e.offset = v; draw(); });
  const sGm = slider(panel, "gamma", 0.4, 2.5, 0.05, e.gamma, v => { e.gamma = v; draw(); });
  const sB = slider(panel, "blur σ", 0, 4, 0.1, e.blur, v => { e.blur = v; draw(); });
  const sN = slider(panel, "noise sd", 0, 0.2, 0.005, e.noise, v => { e.noise = v; draw(); });
  const row2 = htmlEl("div", "qb-presets", "", panel);
  checkbox(row2, "show the grid", grid, v => { grid = v; draw(); });
  button(row2, "new random field", () => { e.seed++; draw(); });
  const row3 = htmlEl("div", "qb-presets", "", panel);
  button(row3, "8 random copies", () => copies());
  htmlEl("div", "qb-note", "flips, any angle, zoom ±20 %, elastic 3 px, gain ±30 %, noise 0.03", panel);
  const strip = htmlEl("div", "al-strip", "", panel);
  let copySeed = 1;

  function syncSliders(): void {
    cH.checked = e.flipH; cV.checked = e.flipV; sA.set(e.angle); sZ.set(Math.log10(e.zoom)); sAl.set(e.alpha); sSg.set(e.sigma);
    sG.set(e.gain); sO.set(e.offset); sGm.set(e.gamma); sB.set(e.blur); sN.set(e.noise);
  }
  function draw(): void {
    if (!img || !gridImg) return;
    const out = applyExplore(img, e);
    const rgba = new ImageData(out.width, out.height);
    let gOut: Grey | null = null;
    if (grid) gOut = applyExplore(gridImg, { ...e, gain: 1, offset: 0, gamma: 1, blur: 0, noise: 0 });
    for (let i = 0; i < out.data.length; i++) {
      const v = Math.max(0, Math.min(255, Math.round(255 * out.data[i])));
      const gl = gOut ? Math.max(0, Math.min(1, gOut.data[i])) : 0;
      rgba.data[4 * i] = Math.round(v * (1 - gl) + 230 * gl); rgba.data[4 * i + 1] = Math.round(v * (1 - gl) + 40 * gl);
      rgba.data[4 * i + 2] = Math.round(v * (1 - gl) + 40 * gl); rgba.data[4 * i + 3] = 255;
    }
    canvas.width = out.width; canvas.height = out.height;
    canvas.getContext("2d")!.putImageData(rgba, 0, 0);
  }
  function copies(): void {
    if (!img) return;
    strip.replaceChildren();
    const r = rng(copySeed++);
    for (let k = 0; k < 8; k++) {
      const x = randomExplore({ flips: true, maxAngle: 180, zoomRange: 0.2, alpha: 3, sigma: 10, gainRange: 0.3, noise: 0.03 }, r);
      const c = htmlEl("canvas", "", "", strip);
      drawGrey(c, applyExplore(img, x), unit);
      c.title = `angle ${x.angle.toFixed(0)}°, zoom ${x.zoom.toFixed(2)}, gain ${x.gain.toFixed(2)}${x.flipH ? ", flipped" : ""}`;
    }
  }

  loadCell(192).then(g => {
    img = { width: g.width, height: g.height, data: g.data.map(v => v / 255) };
    gridImg = createGrey(g.width, g.height);
    for (let y = 0; y < g.height; y++) for (let x = 0; x < g.width; x++) if (x % 16 === 8 || y % 16 === 8) gridImg.data[y * g.width + x] = 1;
    presetBtns.forEach((b, i) => b.classList.toggle("active", EX_PRESETS[i] === startP));
    draw(); copies();
  });
}

// ---------------------------------------------------------------------------------------
// train view

interface Recorded { id: string; label: string; task: Task; aug: Aug; decay: number; train: number[]; val: number[]; trainAcc: number[]; valAcc: number[] }
const RUNS = curves as unknown as Recorded[];

function mountTrain(host: HTMLElement): void {
  let cur = RUNS.find(r => r.id === host.dataset.setting) ?? RUNS[2];
  let shown = 0, timer = 0, drawSeed = 1;

  const left = htmlEl("div", "al-trainleft", "", host);
  const plot = new Plot(left, { width: 600, height: 270, x: [0, 300], y: [0, 2.5], xlabel: "epoch", ylabel: "loss", margin: { l: 64, r: 12, t: 10, b: 50 } });
  const bT = plot.line([], [], { color: "#1f77b4", width: 2.5, label: "no augmentation: training" });
  const bV = plot.line([], [], { color: "#d62728", width: 2.5, label: "no augmentation: validation" });
  const sT = plot.line([], [], { color: "#2ca02c", width: 3, label: "this setting: training" });
  const sV = plot.line([], [], { color: "#ff7f0e", width: 3, label: "this setting: validation" });
  plot.legend(undefined, "tr");
  const readout = htmlEl("div", "qb-readout", "", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  const setBtns = presetRow(left, RUNS.filter(r => r.id !== "none" && r.id !== "o-none").map(r => ({ label: r.label, apply: () => { cur = r; play(); } })));
  const ctl = htmlEl("div", "qb-presets", "", left);
  button(ctl, "play", () => play());
  button(ctl, "new augmented copies", () => { drawSeed++; crops(); });
  const lab1 = htmlEl("div", "qb-note", "", panel);
  const s1 = htmlEl("div", "al-strip al-crops", "", panel);
  const lab2 = htmlEl("div", "qb-note", "", panel);
  const s2 = htmlEl("div", "al-strip al-crops", "", panel);
  const lab3 = htmlEl("div", "qb-note", "", panel);
  const s3 = htmlEl("div", "al-strip al-crops", "", panel);

  function baseline(): Recorded { return RUNS.find(r => r.id === (cur.task === "shape" ? "none" : "o-none"))!; }
  function crops(): void {
    const { tr, va } = experiment(cur.task), r = rng(drawSeed);
    const show = (box: HTMLElement, imgs: Grey[]) => { box.replaceChildren(); imgs.forEach(g => drawGrey(htmlEl("canvas", "", "", box), g, unit)); };
    lab1.textContent = cur.task === "shape" ? "training crops (10 of 20): round or elongated; the elongated ones all lie" : "training crops (10 of 20): lying or standing";
    show(s1, tr.imgs.slice(0, 10));
    lab2.textContent = `as the network sees them: ${cur.label}`;
    show(s2, tr.imgs.slice(0, 10).map(g => augmentCrop(g, cur.aug, r)));
    lab3.textContent = "validation crops (10 of 200)";
    show(s3, va.imgs.slice(0, 10));
  }
  function draw(): void {
    const b = baseline(), n = b.train.length - 1, k = Math.min(n, shown), ep = Array.from({ length: k + 1 }, (_, i) => i);
    const clip = (v: number) => Math.min(2.5, v);
    bT.set(ep, b.train.slice(0, k + 1).map(clip)); bV.set(ep, b.val.slice(0, k + 1).map(clip));
    sT.set(ep, cur.train.slice(0, k + 1).map(clip)); sV.set(ep, cur.val.slice(0, k + 1).map(clip));
    const best = (r: Recorded) => { let i0 = 0; for (let i = 0; i <= k; i++) if (r.val[i] < r.val[i0]) i0 = i; return i0; };
    const line = (name: string, r: Recorded) => { const i = best(r); return `${name}: ${(100 * r.valAcc[k]).toFixed(0)} % correct; best val. loss ${r.val[i].toFixed(3)} (epoch ${i})`; };
    readout.textContent = `epoch ${k}, validation set:\n` + line("without", b) + "\n" + line("with   ", cur);
    setBtns.forEach((btn, i) => btn.classList.toggle("active", RUNS.filter(r => r.id !== "none" && r.id !== "o-none")[i] === cur));
  }
  function play(): void {
    clearTimeout(timer); shown = 0; crops();
    const tick = () => { if (!host.isConnected) return; shown += 4; draw(); if (shown < 300) timer = window.setTimeout(tick, 30); };
    tick();
  }
  play();
}

function mount(host: HTMLElement): void {
  injectStyle();
  if (host.dataset.view === "train") mountTrain(host); else mountExplore(host);
}

autoMount(NAME, mount);
