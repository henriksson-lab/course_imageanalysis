// Quantisation and dithering. An image with (nearly) continuous values in [0, 1] (a ramp,
// smooth blobs, or the 8-bit micrograph), quantised to b = 1..8 bits with no dither,
// random dither or Floyd-Steinberg error diffusion. Shows the original, the quantised
// image, the error image (0 = grey, +-Delta = white/black) and the error histogram with the
// uniform (or triangular, for random dither) density; readout: levels, Delta, RMS error
// against Delta/sqrt(12), SNR in dB.
// Auto-mounts on <div data-demo="quantise">. Optional: data-image="ramp|blobs|cell",
// data-bits="4", data-dither="none|random|diffuse".
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import { quantise, stats, errorHist, ramp, blobs, step, Dither } from "./quant";
// @ts-ignore -- esbuild inlines the image as a data URL
import cellUrl from "../fourier2d/cell256.png";

const NAME = "quantise";
const N = 256;
const CREDIT = "Micrograph: mesenchymal stem cell (TEM), Robert M. Hunt, CC BY 3.0, via Wikimedia Commons. Stored with 8 bits.";
const BINS = 60, RANGE = 1.5;   // error histogram over [-1.5, 1.5] Delta

type ImgId = "ramp" | "blobs" | "cell";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .qz-left { flex: 1 1 640px; min-width: 0; max-width: 840px; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qz-imgs { display: flex; gap: 10px; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; text-align: center; line-height: 1.2; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .45em; }
.qb-demo[data-demo="${NAME}"] .qz-lab { font-size: 16px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .qz-credit { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 20px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qz-lab { font-size: 18px; }
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

function paintGrey(c: HTMLCanvasElement, g: (k: number) => number): void {
  const ctx = c.getContext("2d")!;
  const id = ctx.createImageData(N, N);
  for (let k = 0; k < N * N; k++) {
    const v = Math.max(0, Math.min(255, Math.round(g(k))));
    id.data[4 * k] = id.data[4 * k + 1] = id.data[4 * k + 2] = v; id.data[4 * k + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const left = htmlEl("div", "qz-left", "", host);
  const imgs = htmlEl("div", "qz-imgs", "", left);
  const fig = (t: string) => {
    const f = htmlEl("figure", "", "", imgs);
    const cap = htmlEl("figcaption", "", t, f);
    const c = htmlEl("canvas", "", "", f);
    c.width = N; c.height = N;
    return { c, cap };
  };
  const fIn = fig("original"), fQ = fig("quantised"), fE = fig("error (0 = grey)");
  const W = 820, H = 150, PL = 50, PR = 10, PT = 8, PB = 30;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, left);
  const credit = htmlEl("div", "qz-credit", CREDIT, left);

  const panel = htmlEl("div", "qb-panel", "", host);
  const d0 = {
    image: (["ramp", "blobs", "cell"].includes(host.dataset.image ?? "") ? host.dataset.image : "ramp") as ImgId,
    bits: Math.max(1, Math.min(8, Number(host.dataset.bits ?? 4))),
    dither: (["none", "random", "diffuse"].includes(host.dataset.dither ?? "") ? host.dataset.dither : "none") as Dither,
  };
  const st = { ...d0 };

  htmlEl("div", "qz-lab", "image", panel);
  const imgItems: { label: string; id: ImgId }[] = [{ label: "ramp", id: "ramp" }, { label: "blobs", id: "blobs" }, { label: "micrograph", id: "cell" }];
  const imgBtns = presets(panel, imgItems.map(it => ({ label: it.label, apply: () => { st.image = it.id; load(); } })));
  const bitsS = slider(panel, "bits b", 1, 8, 1, st.bits, v => { st.bits = v; schedule(); });
  htmlEl("div", "qz-lab", "dither", panel);
  const dItems: { label: string; id: Dither }[] = [{ label: "none", id: "none" }, { label: "random", id: "random" }, { label: "error diffusion", id: "diffuse" }];
  const dBtns = presets(panel, dItems.map(it => ({ label: it.label, apply: () => { st.dither = it.id; schedule(); } })));
  htmlEl("div", "qz-lab", "presets", panel);
  const set = (image: ImgId, bits: number, dither: Dither) => { st.image = image; st.bits = bits; st.dither = dither; load(); };
  const row = htmlEl("div", "qb-presets", "", panel);
  presets(row, [
    { label: "8 bits", apply: () => set(st.image, 8, "none") },
    { label: "4 bits: banding", apply: () => set("blobs", 4, "none") },
    { label: "4 bits + dither", apply: () => set("blobs", 4, "random") },
    { label: "2 bits, diffused", apply: () => set("cell", 2, "diffuse") },
    { label: "1 bit", apply: () => set("cell", 1, "diffuse") },
  ]);
  button(row, "reset", () => set(d0.image, d0.bits, d0.dither));
  const ro = htmlEl("div", "qb-readout", "", panel);

  const sources: Partial<Record<ImgId, Float64Array>> = {};
  function load(): void {
    if (st.image === "cell" && !sources.cell) { loadCell().then(v => { sources.cell = v; schedule(); }); return; }
    if (!sources.ramp) sources.ramp = ramp(N);
    if (!sources.blobs) sources.blobs = blobs(N);
    schedule();
  }

  let pending = false;
  function schedule(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }

  function draw(): void {
    const x = sources[st.image];
    if (!x) { load(); return; }
    const b = st.bits, L = 2 ** b - 1, D = step(b);
    const q = quantise(x, N, N, b, st.dither);
    const s = stats(x, q, b);
    paintGrey(fIn.c, k => x[k] * 255);
    paintGrey(fQ.c, k => q[k] * 255);
    paintGrey(fE.c, k => 128 + 127 * (q[k] - x[k]) * L);
    fQ.cap.textContent = `quantised: ${b} bit${b > 1 ? "s" : ""}`;
    bitsS.set(b);
    imgBtns.forEach((bt, i) => bt.classList.toggle("active", imgItems[i].id === st.image));
    dBtns.forEach((bt, i) => bt.classList.toggle("active", dItems[i].id === st.dither));
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";

    // error histogram (density in units of 1/Delta) and the theoretical density
    const hist = errorHist(x, q, b, BINS, RANGE);
    const n = x.length, bw = 2 * RANGE / BINS;
    const dens = hist.map(c => c / n / bw);
    const ymax = Math.max(1.2, Math.min(4, Math.max(...dens) * 1.1));
    svg.innerHTML = "";
    const X = (e: number) => PL + (e + RANGE) / (2 * RANGE) * (W - PL - PR);
    const Y = (d: number) => H - PB - d / ymax * (H - PT - PB);
    svgEl("line", { x1: PL, x2: W - PR, y1: Y(0), y2: Y(0), class: "axis" }, svg);
    svgEl("line", { x1: PL, x2: PL, y1: PT, y2: Y(0), class: "axis" }, svg);
    for (const e of [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]) {
      svgEl("line", { x1: X(e), x2: X(e), y1: Y(0), y2: Y(0) + 5, class: "axis" }, svg);
      svgEl("text", { x: X(e), y: H - 8, "text-anchor": "middle" }, svg).textContent = (e === 0 ? "0" : `${e}Δ`).replace("-", "−");
    }
    svgEl("text", { x: PL - 6, y: Y(1) + 5, "text-anchor": "end" }, svg).textContent = "1/Δ";
    svgEl("line", { x1: PL - 3, x2: PL, y1: Y(1), y2: Y(1), class: "axis" }, svg);
    dens.forEach((d, i) => {
      const x0 = X(-RANGE + i * bw), x1 = X(-RANGE + (i + 1) * bw);
      const hh = Math.min(d, ymax);
      svgEl("rect", { x: x0, y: Y(hh), width: Math.max(0, x1 - x0 - 1), height: Y(0) - Y(hh), fill: "var(--qb-c1)", opacity: 0.7 }, svg);
    });
    let pts = "";
    if (st.dither === "none") pts = `${X(-RANGE)},${Y(0)} ${X(-0.5)},${Y(0)} ${X(-0.5)},${Y(1)} ${X(0.5)},${Y(1)} ${X(0.5)},${Y(0)} ${X(RANGE)},${Y(0)}`;
    else if (st.dither === "random") pts = `${X(-RANGE)},${Y(0)} ${X(-1)},${Y(0)} ${X(0)},${Y(1)} ${X(1)},${Y(0)} ${X(RANGE)},${Y(0)}`;
    if (pts) svgEl("polyline", { points: pts, fill: "none", stroke: "var(--qb-c2)", "stroke-width": 2.5 }, svg);
    svgEl("text", { x: W - PR, y: PT + 14, "text-anchor": "end" }, svg).textContent = "histogram of the error" +
      (st.dither === "none" ? "; red: uniform on ±Δ/2" : st.dither === "random" ? "; red: triangle on ±Δ" : "");

    const f4 = (v: number) => v.toFixed(4);
    ro.textContent = [
      `${2 ** b} levels, Δ = 1/${L} = ${f4(D)}`,
      `  (= ${(255 * D).toFixed(1)} of 255 grey levels)`,
      `RMS error  ${f4(s.rms)}`,
      `Δ/√12      ${f4(s.theory)}`,
      `mean error ${s.meanErr >= 0 ? " " : ""}${f4(s.meanErr)}`.replace("-", "−"),
      `SNR ${Number.isFinite(s.snrDb) ? s.snrDb.toFixed(1) : "∞"} dB   (6.02 b = ${(6.02 * b).toFixed(1)})`,
      `levels used: ${s.distinct}`,
    ].join("\n");
  }

  load();
}

autoMount(NAME, mount);
