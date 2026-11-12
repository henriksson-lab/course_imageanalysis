// Blob detection over scales. A 256 × 256 image: synthetic discs (radii 2–20 px) and Gaussian
// spots, a touching pair, an elongated blob, on an uneven background with seeded noise
// (ground truth known), or the micrograph (dark granules). The response −σ²∇²L (normalised
// LoG, or the difference of Gaussians with ratio k) is computed for 16 scales σ = 1 … 16
// once per image; the σ slider only indexes the stack. Detections: local maxima at the
// current σ, or in (x, y, σ) over all scales (26 neighbours; not at the first and last
// scale), above a threshold, drawn as
// circles of radius √2·σ coloured by scale. Click a pixel: its response against σ, raw and
// normalised. "Pyramid" shows the Gaussian and Laplacian pyramids and the reconstruction error.
//
// <div data-demo="blobscale" data-preset="one|raw|all|edges|radius|touch|long|dog|background|cell|pyramid"
//      data-presets="one,raw,all"  (optional: show only these preset buttons, e.g. on a slide)></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey, gaussianPyramid, laplacianPyramid, collapsePyramid, mapGrey } from "../lib/imageops";
import { responseStack } from "./stack";
import { geomScales, extrema3D, localMaxima2D } from "../lib/edges";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { addGaussianNoise } from "../lib/synth";
import { hsvToRgb } from "../lib/colour";
import { makeBlobImage, Blob } from "./scene";
import { ImgView, injectStyle } from "../canny/imgview";

const NAME = "blobscale";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .bs-main { flex: 0 1 430px; min-width: 240px; }
[data-demo="${NAME}"] .bs-mid { flex: 0 1 260px; min-width: 200px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .bs-pyr canvas { width: 100%; image-rendering: pixelated; display: block; }
[data-demo="${NAME}"] .bs-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.2em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .bs-main { flex: 0 0 440px; }
.reveal [data-demo="${NAME}"] .bs-mid { flex: 0 0 240px; gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 330px; gap: .18em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .25em .45em; }
.reveal [data-demo="${NAME}"] .ev-credit { display: none; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

const N = 256, LEVELS = 16, SIGMAS = geomScales(1, 16, LEVELS);
interface State {
  image: "synthetic" | "cell"; noise: number; background: boolean; level: number; norm: boolean; dog: boolean; k: number;
  all: boolean; thr: number; truth: boolean; dark: boolean; pyramid: boolean; px: number; py: number;
}
interface Preset { id: string; label: string; s: Partial<State> }
const lev = (s: number) => SIGMAS.reduce((b, v, i) => (Math.abs(v - s) < Math.abs(SIGMAS[b] - s) ? i : b), 0);
const DEFAULT: State = { image: "synthetic", noise: 6, background: true, level: lev(4), norm: true, dog: false, k: 1.6, all: false, thr: 26, truth: false, dark: false, pyramid: false, px: 146, py: 30 };
const PRESETS: Preset[] = [
  { id: "one", label: "one σ finds one size", s: { level: lev(4.2), all: false } },
  { id: "raw", label: "raw LoG: only the small spots", s: { norm: false, all: true, thr: 12, px: 120, py: 92 } },
  { id: "all", label: "all scales, normalised", s: { all: true } },
  { id: "edges", label: "low threshold: edges respond too", s: { all: true, thr: 18, px: 40, py: 71, level: lev(1.45) } },
  { id: "radius", label: "radius = √2·σ (truth overlay)", s: { all: true, truth: true, px: 200, py: 110, level: lev(14) } },
  { id: "touch", label: "touching blobs", s: { all: true, truth: true, px: 54, py: 215, level: lev(7) } },
  { id: "long", label: "elongated blob", s: { all: true, truth: true, px: 140, py: 225, level: lev(3.5) } },
  { id: "dog", label: "DoG matches LoG", s: { all: true, dog: true, k: 1.2 } },
  { id: "background", label: "uneven background does not matter", s: { all: true, level: lev(16), px: 180, py: 192 } },
  { id: "cell", label: "dark granules in the micrograph", s: { image: "cell", noise: 0, all: true, dark: true, thr: 50, px: 106, py: 68, level: lev(5) } },
  { id: "pyramid", label: "the pyramid", s: { pyramid: true } },
];

const scaleColour = (k: number): string => { const c = hsvToRgb([280 * k / (LEVELS - 1), 0.9, 1]); return `rgb(${c.map(v => Math.round(255 * v)).join(",")})`; };

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const main = htmlEl("div", "bs-main", "", host);
  const view = new ImgView(main, "");
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);
  const mid = htmlEl("div", "bs-mid", "", host);
  const resp = new ImgView(mid, "");
  const pyrDiv = htmlEl("div", "bs-pyr", "", mid);
  const pyrCap = htmlEl("figcaption", "", "", pyrDiv);
  pyrCap.style.fontSize = "16px";
  const pyrCanvas = htmlEl("canvas", "", "", pyrDiv);
  const busy = htmlEl("div", "bs-busy", "", mid);
  const sp = new Plot(mid, { width: 270, height: 200, x: [1, 16], xlog: true, y: [0, 1], font: 16, margin: { l: 44, r: 8, t: 26, b: 44 }, xlabel: "σ", xticks: [1, 2, 4, 8, 16], xfmt: v => String(v) });
  const cRaw = sp.line([], [], { color: "#999", width: 2.5 });
  const cNorm = sp.line([], [], { color: seriesColor(1), width: 3 });
  const vSig = sp.vline(1, { color: "var(--qb-c4)", width: 2 });
  const tSp = sp.text(1.02, 0, "", { anchor: "start", size: 15 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "presets", panel);
  const only = host.dataset.presets ? host.dataset.presets.split(/[\s,]+/) : null;
  const shownPresets = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shownPresets.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ev-lab", "image", panel);
  const imgBtns = presets(panel, [
    { label: "spots (synthetic)", apply: () => { st.image = "synthetic"; st.dark = false; darkC.checked = false; makeImage(); } },
    { label: "micrograph", apply: () => { st.image = "cell"; st.dark = true; darkC.checked = true; makeImage(); } },
  ]);
  const sigS = slider(panel, "σ", 0, LEVELS - 1, 1, st.level, v => { st.level = v; draw(); }, v => SIGMAS[v].toFixed(2));
  const thrS = slider(panel, "threshold", 0, 100, 0.5, st.thr, v => { st.thr = v; draw(); });
  const noiseS = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; makeImage(); });
  const kDiv = htmlEl("div", "", "", panel);
  const kS = slider(kDiv, "DoG ratio k", 1.05, 2, 0.05, st.k, v => { st.k = v; if (st.dog) computeStack(); });
  const checks = htmlEl("div", "", "", panel);
  checks.style.display = "flex"; checks.style.flexWrap = "wrap"; checks.style.gap = "0 1em";
  const normC = checkbox(checks, "normalise (× σ²)", st.norm, v => { st.norm = v; draw(); });
  const dogC = checkbox(checks, "DoG", st.dog, v => { st.dog = v; computeStack(); });
  const allC = checkbox(checks, "detect over all scales", st.all, v => { st.all = v; draw(); });
  const truthC = checkbox(checks, "truth", st.truth, v => { st.truth = v; draw(); });
  const darkC = checkbox(checks, "dark blobs", st.dark, v => { st.dark = v; computeStack(); });
  const pyrC = checkbox(checks, "pyramid", st.pyramid, v => { st.pyramid = v; draw(); });
  button(panel, "reset", () => load({ id: "", label: "", s: p0?.s ?? {} }));
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: Preset): void {
    Object.assign(st, DEFAULT, p.s);
    if (!p.id) presetBtns.forEach(b => b.classList.remove("active"));
    sigS.set(st.level); thrS.set(st.thr); noiseS.set(st.noise); kS.set(st.k);
    normC.checked = st.norm; dogC.checked = st.dog; allC.checked = st.all; truthC.checked = st.truth; darkC.checked = st.dark; pyrC.checked = st.pyramid;
    makeImage();
  }
  const pick = (x: number, y: number) => { st.px = x; st.py = y; draw(); };
  view.onPick(pick); resp.onPick(pick);

  let cell: Grey | null = null, img: Grey | null = null, blobs: Blob[] = [];
  let stack: Grey[] = [];          // normalised response −σ²∇²L (or DoG equivalent), sign for the chosen polarity
  let pyrKey = "";
  function makeImage(): void {
    if (st.image === "cell" && !cell) { loadCell(N).then(g => { cell = g; makeImage(); }); return; }
    if (st.image === "cell") { img = st.noise > 0 ? addGaussianNoise(cell!, st.noise, 1) : cell!; blobs = []; }
    else { const s = makeBlobImage("synthetic", N, st.noise, 1, st.background); img = s.image; blobs = s.blobs; }
    pyrKey = "";
    computeStack();
  }
  function computeStack(): void {
    busy.textContent = "computing the scale space …";
    setTimeout(() => {
      const t0 = performance.now(), sign = st.dark ? 1 : -1;
      stack = responseStack(img!, SIGMAS, sign as 1 | -1, st.dog ? st.k : 0);
      busy.textContent = `scale space: ${LEVELS} levels in ${Math.round(performance.now() - t0)} ms`;
      draw();
    }, 20);
  }

  function detections(): { x: number; y: number; s: number; value: number }[] {
    if (!stack.length) return [];
    const S = st.norm ? stack : stack.map((g, k) => mapGrey(g, v => v / (SIGMAS[k] * SIGMAS[k])));
    if (st.all) return extrema3D(S, st.thr, 3, st.norm);   // raw: the maxima sit at the smallest scale, so keep the end scales
    return localMaxima2D(S[st.level], st.thr, 2, 3).map(m => ({ ...m, s: st.level }));
  }

  function score(det: { x: number; y: number; s: number }[]): { hits: number; misses: number; fa: number } {
    const used = new Set<number>(); let hits = 0;
    for (const b of blobs) {
      let best = -1, bd = Infinity;
      det.forEach((d, j) => { const dd = Math.hypot(d.x - b.x, d.y - b.y); if (!used.has(j) && dd < bd) { bd = dd; best = j; } });
      if (best >= 0 && bd <= Math.max(3, 0.5 * b.r)) { used.add(best); hits++; }
    }
    return { hits, misses: blobs.length - hits, fa: det.length - hits };
  }

  function draw(): void {
    if (!img || !stack.length) return;
    imgBtns.forEach((b, i) => b.classList.toggle("active", i === (st.image === "synthetic" ? 0 : 1)));
    kDiv.style.display = st.dog ? "" : "none";
    credit.style.display = st.image === "cell" ? "" : "none";
    view.grey(img, { mode: "auto" });
    const s = SIGMAS[st.level], cur = stack[st.level], f = st.norm ? 1 : 1 / (s * s);
    const shown = st.norm ? cur : mapGrey(cur, v => v * f);
    resp.grey(shown, { mode: "mid" });
    resp.setCaption(`${st.dog ? "DoG" : "LoG"} response${st.norm ? " ×σ²" : " (raw)"}, σ = ${s.toFixed(2)}`);
    const det = detections();
    view.setCaption(st.all ? `detections over all scales: ${det.length}` : `maxima at σ = ${s.toFixed(2)}: ${det.length}`);
    view.clear(); resp.clear();
    if (st.truth) for (const b of blobs) {
      if (b.kind === "ellipse") view.add("ellipse", { cx: b.x + 0.5, cy: b.y + 0.5, rx: b.b!, ry: b.r, fill: "none", stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "3 3" });
      else view.add("circle", { cx: b.x + 0.5, cy: b.y + 0.5, r: b.r, fill: "none", stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "3 3" });
    }
    for (const d of det.slice(0, 400)) {
      const c = scaleColour(d.s);
      view.add("circle", { cx: d.x + 0.5, cy: d.y + 0.5, r: Math.SQRT2 * SIGMAS[d.s], fill: "none", stroke: c, "stroke-width": 2 });
      if (d.s === st.level || !st.all) resp.add("circle", { cx: d.x + 0.5, cy: d.y + 0.5, r: Math.SQRT2 * SIGMAS[d.s], fill: "none", stroke: c, "stroke-width": 1.5 });
    }
    view.box(st.px, st.py, 0, "#ffd400", 2.5); resp.box(st.px, st.py, 0, "#ffd400", 2.5);
    // σ-plot at the clicked pixel
    const i = st.py * N + st.px;
    const vn = stack.map(g => g.data[i]), vr = stack.map((g, k) => g.data[i] / (SIGMAS[k] * SIGMAS[k]));
    let lo = 0, hi = 0; for (const v of [...vn, ...vr]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = 0.1 * (hi - lo || 1);
    sp.setDomain(undefined, [lo - pad, hi + 2 * pad]);
    cNorm.set(SIGMAS, vn); cRaw.set(SIGMAS, vr); vSig.set(s);
    let kb = 0; vn.forEach((v, k) => { if (v > vn[kb]) kb = k; });
    tSp.set(1.02, hi + 1.4 * pad, `(${st.px}, ${st.py}): ×σ² (red), raw (grey)`);
    // pyramid
    resp.fig.style.display = st.pyramid ? "none" : "";
    pyrDiv.style.display = st.pyramid ? "" : "none";
    let pyrErr = "";
    if (st.pyramid) pyrErr = drawPyramid();
    const sc = blobs.length ? score(det) : null;
    ro.textContent =
      `detections: ${det.length}` + (sc ? `\nhits ${sc.hits}, misses ${sc.misses}, false alarms ${sc.fa} (of ${blobs.length} true)` : "") +
      `\nat (${st.px}, ${st.py}): best σ = ${SIGMAS[kb].toFixed(2)} → radius √2·σ = ${(Math.SQRT2 * SIGMAS[kb]).toFixed(1)}` + pyrErr;
  }

  function drawPyramid(): string {
    const G = gaussianPyramid(img!, 5), L = laplacianPyramid(img!, 5), back = collapsePyramid(L);
    let err = 0; for (let i = 0; i < N * N; i++) err = Math.max(err, Math.abs(back.data[i] - img!.data[i]));
    const key = pyrKey;
    if (key !== "done") {
      const W = N + N / 2 + 4, H = 2 * N + 6;
      pyrCanvas.width = W; pyrCanvas.height = H;
      const ctx = pyrCanvas.getContext("2d")!;
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
      const put = (g: Grey, x: number, y: number, signed: boolean) => {
        let lo = Infinity, hi = -Infinity, m = 0;
        for (const v of g.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); m = Math.max(m, Math.abs(v)); }
        const id = ctx.createImageData(g.width, g.height);
        for (let i = 0; i < g.data.length; i++) {
          const v = signed ? 128 + 127 * g.data[i] / (m || 1) : 255 * (g.data[i] - lo) / (hi - lo || 1);
          id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255;
        }
        ctx.putImageData(id, x, y);
      };
      const place = (arr: Grey[], y0: number, signed: boolean) => {
        put(arr[0], 0, y0, signed);
        let y = y0;
        for (let k = 1; k < arr.length; k++) { put(arr[k], N + 4, y, signed && k < arr.length - 1); y += arr[k].height + 2; }
      };
      place(G, 0, false); place(L, N + 6, true);
      pyrKey = "done";
    }
    pyrCap.textContent = "Gaussian pyramid (top) and Laplacian pyramid (bottom)";
    return `\npyramid: 256² + 128² + … = ${G.reduce((a, g) => a + g.width * g.height, 0)} pixels\nreconstruction max error: ${err.toExponential(1)}`;
  }

  if (p0 && shownPresets.includes(p0)) presetBtns[shownPresets.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
