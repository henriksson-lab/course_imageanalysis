// Interactive graph-cut segmentation (Boykov & Jolly 2001) of a 128 x 128 image. Paint
// object and background seeds; the intensity model (a Gaussian or a histogram per label)
// is estimated from the seed pixels and gives the data term -log P(I | label); n-links
// carry lambda * w_pq with w_pq = exp(-(I_p - I_q)^2 / 2 sigma^2) (sigma = infinity: w = 1),
// over a 4- or 8-neighbourhood. One max-flow (lib/maxflow.ts) gives the exact minimum.
// Beside it: the thresholded result (lambda = 0) and the energy of both. Presets show the
// typical failure modes: shrinking bias, blocky boundaries, a bad intensity model.
// Auto-mounts on <div data-demo="graphcut">. Optional: data-preset="noise|shrink|blocky|nucleus|badmodel".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { N, PRESETS, Preset, Settings, synthImage, synthTruth, downsample, addNoise, rasterize, paintLine, segment, Segmentation } from "./model";
// @ts-ignore -- esbuild inlines the image as a data URL
import cellUrl from "../fourier2d/cell256.png";

const NAME = "graphcut";
const CREDIT = "Image: transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons";
const SIGMAS = [3, 6, 12, 25, 50, 100, Infinity];
const BRUSH = 2;

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; line-height: 1.2; margin: 0 0 .2em; }
[data-demo="${NAME}"] .gc-main { flex: 0 1 410px; min-width: 240px; }
[data-demo="${NAME}"] .gc-main canvas { cursor: crosshair; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; touch-action: none; image-rendering: pixelated; }
[data-demo="${NAME}"] .gc-mid { flex: 0 1 290px; min-width: 200px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .gc-mid figure { max-width: 210px; align-self: center; width: 100%; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .4em; }
[data-demo="${NAME}"] .gc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .gc-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-btn.obj.active { background: #ff7f0e; border-color: #ff7f0e; }
[data-demo="${NAME}"] .qb-btn.bg.active { background: #1f77b4; border-color: #1f77b4; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .gc-main { flex-basis: 400px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .5em; }
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
      c.width = 256; c.height = 256;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0, 256, 256);
      const d = ctx.getImageData(0, 0, 256, 256).data;
      const big = new Float64Array(256 * 256);
      for (let k = 0; k < big.length; k++) big[k] = d[4 * k];
      resolve(downsample(big));
    };
    img.onerror = reject;
    img.src = cellUrl;
  });
}

/** Draw the image with the labelling (tint + boundary) and the seeds on top. */
function paint(c: HTMLCanvasElement, I: ArrayLike<number>, L: Uint8Array | null, seeds: Int8Array | null): void {
  const ctx = c.getContext("2d")!;
  const id = ctx.createImageData(N, N);
  for (let p = 0; p < N * N; p++) {
    const g = I[p];
    let r = g, gg = g, b = g;
    if (L) {
      const x = p % N, y = (p - x) / N;
      const edge = L[p] === 1 && ((x > 0 && !L[p - 1]) || (x < N - 1 && !L[p + 1]) || (y > 0 && !L[p - N]) || (y < N - 1 && !L[p + N]));
      if (edge) { r = 255; gg = 230; b = 0; }
      else if (L[p]) { r = 0.55 * g + 0.45 * 255; gg = 0.55 * g + 0.45 * 127; b = 0.55 * g + 0.45 * 14; }
    }
    if (seeds && seeds[p] === 1) { r = 230; gg = 40; b = 40; }
    if (seeds && seeds[p] === -1) { r = 30; gg = 90; b = 230; }
    id.data[4 * p] = r; id.data[4 * p + 1] = gg; id.data[4 * p + 2] = b; id.data[4 * p + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

const f1 = (v: number) => v.toFixed(1);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    image: p0.image, noise: p0.noise, settings: { ...p0.settings } as Settings,
    seeds: rasterize(p0.strokes), brush: 1 as 1 | -1 | 0, show: true,
  };

  // ---- layout ----
  const main = htmlEl("figure", "gc-main", "", host);
  const mainCap = htmlEl("figcaption", "", "graph cut: paint seeds here", main);
  const cv = htmlEl("canvas", "", "", main);
  cv.width = N; cv.height = N;
  const credit = htmlEl("div", "gc-credit", CREDIT, main);
  const mid = htmlEl("div", "gc-mid", "", host);
  const thrFig = htmlEl("figure", "", "", mid);
  htmlEl("figcaption", "", "threshold (λ = 0)", thrFig);
  const cvT = htmlEl("canvas", "", "", thrFig);
  cvT.width = N; cvT.height = N;
  const ro = htmlEl("div", "qb-readout", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "gc-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "gc-lab", "image", panel);
  const imgBtns = presets(panel, [
    { label: "test image", apply: () => { st.image = "synth"; makeImage(); } },
    { label: "micrograph", apply: () => { st.image = "cell"; makeImage(); } },
  ]);
  const noiseS = slider(panel, "noise", 0, 100, 5, st.noise, v => { st.noise = v; makeImage(); });
  htmlEl("div", "gc-lab", "paint", panel);
  const brushRow = htmlEl("div", "qb-presets", "", panel);
  const brushBtns = [
    button(brushRow, "object", () => setBrush(1), "obj"),
    button(brushRow, "background", () => setBrush(-1), "bg"),
    button(brushRow, "erase", () => setBrush(0)),
    button(brushRow, "clear", () => { st.seeds = new Int8Array(N * N); run(); }),
  ];
  // lambda slider is quadratic: lambda = 20 (u / 100)^2
  const lamOf = (u: number) => Math.round(20 * (u / 100) ** 2 * 20) / 20;
  const uOf = (l: number) => 100 * Math.sqrt(l / 20);
  const lamS = slider(panel, "λ", 0, 100, 0.5, uOf(st.settings.lambda), u => { st.settings.lambda = lamOf(u); run(); }, u => lamOf(u).toFixed(2));
  const sigS = slider(panel, "σ", 0, SIGMAS.length - 1, 1, SIGMAS.indexOf(st.settings.sigma),
    v => { st.settings.sigma = SIGMAS[v]; run(); }, v => (Number.isFinite(SIGMAS[v]) ? String(SIGMAS[v]) : "∞"));
  htmlEl("div", "gc-lab", "neighbours · intensity model", panel);
  const optRow = htmlEl("div", "qb-presets", "", panel);
  const nbBtns = [4, 8].map(k => button(optRow, String(k), () => { st.settings.nbhd = k as 4 | 8; run(); }));
  const modelBtns = (["gauss", "hist"] as const).map(m => button(optRow, m === "gauss" ? "Gaussian" : "histogram", () => { st.settings.model = m; run(); }));
  checkbox(panel, "show the result", st.show, v => { st.show = v; draw(); });
  button(panel, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", panel);

  function setBrush(b: 1 | -1 | 0): void { st.brush = b; syncButtons(); }

  function load(p: Preset): void {
    st.image = p.image; st.noise = p.noise; st.settings = { ...p.settings }; st.seeds = rasterize(p.strokes);
    noiseS.set(st.noise); lamS.set(uOf(st.settings.lambda)); sigS.set(SIGMAS.indexOf(st.settings.sigma));
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }

  function syncButtons(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", i === (st.image === "synth" ? 0 : 1)));
    brushBtns.forEach((b, i) => b.classList.toggle("active", i < 3 && [1, -1, 0][i] === st.brush));
    nbBtns.forEach((b, i) => b.classList.toggle("active", [4, 8][i] === st.settings.nbhd));
    modelBtns.forEach((b, i) => b.classList.toggle("active", ["gauss", "hist"][i] === st.settings.model));
  }

  // ---- images ----
  let cellClean: Float64Array | null = null;
  const synthClean = synthImage(), truth = synthTruth();
  let I: Float64Array = new Float64Array(N * N);
  function makeImage(): void {
    if (st.image === "cell" && !cellClean) { loadCell().then(v => { cellClean = v; makeImage(); }); return; }
    I = addNoise(st.image === "cell" ? cellClean! : synthClean, st.noise);
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    run();
  }

  // ---- painting ----
  let last: [number, number] | null = null;
  const toPix = (ev: PointerEvent): [number, number] => {
    const r = cv.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * N, (ev.clientY - r.top) / r.height * N];
  };
  cv.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    cv.setPointerCapture(ev.pointerId);
    last = toPix(ev);
    paintLine(st.seeds, ...last, ...last, BRUSH, st.brush);
    presetBtns.forEach(b => b.classList.remove("active"));
    draw();
  });
  cv.addEventListener("pointermove", ev => {
    if (!last) return;
    const q = toPix(ev);
    paintLine(st.seeds, ...last, ...q, BRUSH, st.brush);
    last = q;
    draw();
  });
  const up = () => { if (last) { last = null; run(); } };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);

  // ---- segmentation ----
  let seg: Segmentation | null = null;
  let pending = false;
  function run(): void {
    syncButtons();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; seg = segment(I, st.seeds, st.settings); draw(); });
  }

  function draw(): void {
    const ok = seg !== null && seg.ok;
    paint(cv, I, st.show && ok ? seg!.cut.L : null, st.seeds);
    paint(cvT, I, ok ? seg!.thr : null, null);
    mainCap.textContent = `graph cut (λ = ${st.settings.lambda}): paint seeds here`;
    if (!seg || !ok) { ro.textContent = seg ? seg.info : ""; note.textContent = "Paint at least one object and one background seed."; return; }
    const c = seg.cut.energy, t = seg.thrEnergy;
    const r = (nm: string, a: string, b: string) => nm.padEnd(10) + a.padStart(10) + b.padStart(10);
    const cnt = (L: Uint8Array) => L.reduce((a, b) => a + b, 0);
    const wrong = (L: Uint8Array) => L.reduce((a, l, p) => a + (l !== truth[p] ? 1 : 0), 0);
    const lines = [
      r("", "graph cut", "threshold"),
      r("data", f1(c.data), f1(t.data)),
      r("Σ w (cut)", f1(c.boundary), f1(t.boundary)),
      r("λ·Σ w", f1(c.smooth), f1(t.smooth)),
      r("E", f1(c.total), f1(t.total)),
      r("object px", String(cnt(seg.cut.L)), String(cnt(seg.thr))),
    ];
    if (st.image === "synth") lines.push(r("wrong px", String(wrong(seg.cut.L)), String(wrong(seg.thr))));
    lines.push(`max flow = ${f1(seg.cut.flow)}`, `${seg.ms.toFixed(0)} ms`);
    ro.textContent = lines.join("\n");
    note.textContent = `Intensity model from the seeds: ${seg.info}. Red: object seeds, blue: background seeds; yellow line: the boundary of the graph-cut object.`;
  }

  syncButtons();
  load(p0);
}

autoMount(NAME, mount);
