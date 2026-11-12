// Predictive (lossless) coding. An image, the residual e = x − x̂ of a predictor (left, up,
// average, planar, Paeth, MED, or none), the histograms of pixels and residuals on one axis
// with the entropy of each, and a scatter of each pixel against its left neighbour. Readouts:
// entropy in bits per pixel, the ideal size H·N, and the real DEFLATE size of the raw and of
// the residual bytes from the browser's own compressor (CompressionStream; no network, no
// library). A noise slider adds seeded Gaussian noise: the residual can never be narrower than
// the noise. Click a pixel to see its neighbours, the prediction and the residual.
// Auto-mounts on <div data-demo="predict">. Options: data-image="cell|smooth|shapes|flat|
// mask|fluor12", data-predictor="none|left|up|average|planar|paeth|med", data-noise="0".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Predictor, PREDICTORS, residuals, predictAt, toBytes } from "../lib/predict";
import { entropy } from "../lib/entropy";
import { intHistogram } from "../lib/histogram";
import { Plot } from "../lib/plot";
import { drawScalarImage, diverging } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { spots } from "../lib/synth";
import { smooth, shapes, flat, withNoise, SIZE } from "../jpeg/testimages";

const NAME = "predict";
const N = SIZE;
type ImgId = "cell" | "smooth" | "shapes" | "flat" | "mask" | "fluor12";
const IMAGES: { id: ImgId; label: string; credit: string }[] = [
  { id: "cell", label: "cell", credit: CELL_CREDIT },
  { id: "smooth", label: "smooth", credit: "Synthetic: soft blobs on a gradient, noise sd 1.5 (fixed seed)." },
  { id: "shapes", label: "shapes", credit: "Synthetic: flat shapes with straight and diagonal edges, noise sd 1.5." },
  { id: "flat", label: "flat field", credit: "Synthetic: constant grey 128 (add noise with the slider)." },
  { id: "mask", label: "binary mask", credit: "Synthetic: the bright shapes as a binary mask (0 / 255)." },
  { id: "fluor12", label: "12-bit spots", credit: "Synthetic 12-bit fluorescence: 60 spots on an uneven background, Poisson noise (fixed seed)." },
];
const PRED_LABEL: Record<Predictor, string> = { none: "none", left: "left", up: "up", average: "average", planar: "planar a+b−c", paeth: "Paeth", med: "MED" };

const STYLE = `
[data-demo="${NAME}"] .pr-main { display: flex; flex-direction: column; gap: .5em; flex: 1 1 640px; min-width: 0; max-width: 880px; }
[data-demo="${NAME}"] .pr-row { display: flex; gap: 10px; align-items: flex-start; }
[data-demo="${NAME}"] .pr-row > figure { margin: 0; flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; cursor: crosshair; touch-action: none; }
[data-demo="${NAME}"] .pr-hist { flex: 1.9 1 0 !important; }
[data-demo="${NAME}"] .pr-read { flex: 1 1 0; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 230px; max-width: 280px; }
[data-demo="${NAME}"] .pr-btns { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
[data-demo="${NAME}"] .qb-btn.on { background: var(--qb-accent); color: #fff; border-color: var(--qb-accent); }
[data-demo="${NAME}"] .pr-credit { font-size: 13px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 14px; }
.reveal [data-demo="${NAME}"] .pr-main { max-width: 860px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** The 12-bit synthetic fluorescence image (integers 0..4095). */
function fluor12(): Float64Array {
  const sp = spots({ width: N, height: N, count: 60, sigma: 2, amplitude: [300, 3000], background: 300, slope: 150, bump: 250, photons: 1, seed: 5 });
  return Float64Array.from(sp.image.data, v => Math.min(4095, Math.max(0, Math.round(v))));
}

/** DEFLATE size (zlib stream) of some bytes with the browser's CompressionStream; NaN if unavailable. */
async function deflateSize(bytes: Uint8Array): Promise<number> {
  const CS = (globalThis as unknown as { CompressionStream?: new (f: string) => TransformStream<Uint8Array, Uint8Array> }).CompressionStream;
  if (!CS) return NaN;
  const stream = new Blob([bytes as unknown as BlobPart]).stream().pipeThrough(new CS("deflate"));
  const buf = await new Response(stream).arrayBuffer();
  return buf.byteLength;
}

const fmtB = (b: number) => (Number.isFinite(b) ? Math.round(b).toLocaleString("en-US").replace(/,/g, " ") + " B" : "n/a");

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const init = {
    img: (IMAGES.some(i => i.id === d.image) ? d.image : "cell") as ImgId,
    pred: (PREDICTORS.includes(d.predictor as Predictor) ? d.predictor : "left") as Predictor,
    noise: Number(d.noise ?? 0),
  };
  const st = { ...init, px: -1, py: -1 };

  const main = htmlEl("div", "pr-main", "", host);
  const row1 = htmlEl("div", "pr-row", "", main);
  const fig = (parent: Element, title: string, cls = "") => {
    const f = htmlEl("figure", cls, "", parent);
    htmlEl("figcaption", "", title, f);
    return f;
  };
  const cImg = htmlEl("canvas", "", "", fig(row1, "image x"));
  const resCap = fig(row1, "residual e = x − x̂");
  const cRes = htmlEl("canvas", "", "", resCap);
  const scFig = fig(row1, "pixel vs left neighbour");
  const scatter = new Plot(scFig, { width: 300, height: 300, x: [0, 255], y: [0, 255], xlabel: "left neighbour", ylabel: "pixel", font: 17, margin: { l: 70, b: 50, r: 12, t: 8 } });
  const row2 = htmlEl("div", "pr-row", "", main);
  const hFig = fig(row2, "histograms: pixels (blue) and residuals (orange)", "pr-hist");
  const hist = new Plot(hFig, { width: 560, height: 230, x: [-128, 255], y: [0, 0.1], xlabel: "value", ylabel: "fraction", font: 17, margin: { l: 70, b: 52, r: 12, t: 8 } });
  const read = htmlEl("div", "pr-read", "", row2);
  const ro = htmlEl("div", "qb-readout", "", read);
  const roPix = htmlEl("div", "qb-readout", "Click a pixel to see its prediction.", read);
  const credit = htmlEl("div", "pr-credit", "", main);

  const panel = htmlEl("div", "qb-panel", "", host);
  const presetList = [
    { label: "left neighbour on the cell", s: { img: "cell", pred: "left", noise: 0 } },
    { label: "smooth image", s: { img: "smooth", pred: "left", noise: 0 } },
    { label: "noise kills prediction", s: { img: "smooth", pred: "left", noise: 10 } },
    { label: "edges are what remains", s: { img: "shapes", pred: "planar", noise: 0 } },
    { label: "MED at an edge", s: { img: "shapes", pred: "med", noise: 0 } },
    { label: "binary mask", s: { img: "mask", pred: "left", noise: 0 } },
    { label: "12-bit spots", s: { img: "fluor12", pred: "med", noise: 0 } },
  ];
  presets(panel, presetList.map(p => ({ label: p.label, apply: () => { Object.assign(st, p.s); syncUI(); load(); } })));
  htmlEl("div", "", "image:", panel);
  const imgRow = htmlEl("div", "pr-btns", "", panel);
  const imgBtns = IMAGES.map(im => button(imgRow, im.label, () => { st.img = im.id; syncUI(); load(); }));
  htmlEl("div", "", "predictor x̂:", panel);
  const predRow = htmlEl("div", "pr-btns", "", panel);
  const predBtns = PREDICTORS.map(p => button(predRow, PRED_LABEL[p], () => { st.pred = p; syncUI(); compute(); }));
  const noiseS = slider(panel, "added noise sd", 0, 20, 1, st.noise, v => { st.noise = v; noisy(); compute(); });
  button(panel, "reset", () => { Object.assign(st, init); st.px = st.py = -1; noiseS.set(st.noise); syncUI(); load(); });
  htmlEl("div", "qb-note", "Neighbours: a = left, b = up, c = up-left. Residual map: white = 0, red = positive, blue = negative (±32 grey levels; ±512 for 12 bit). DEFLATE: the browser's own compressor, on 1 byte per pixel (2 bytes for 12 bit).", panel);

  function syncUI(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("on", IMAGES[i].id === st.img));
    predBtns.forEach((b, i) => b.classList.toggle("on", PREDICTORS[i] === st.pred));
    noiseS.set(st.noise);
  }

  // ---- data ----
  let base: Float64Array = new Float64Array(N * N), img: Float64Array = base, maxV = 255;
  let cellImg: Float64Array | null = null;
  let version = 0;

  function load(): void {
    const id = st.img;
    credit.textContent = IMAGES.find(i => i.id === id)!.credit;
    maxV = id === "fluor12" ? 4095 : 255;
    if (id === "cell") {
      if (!cellImg) { loadCell().then(g => { cellImg = Float64Array.from(g.data, v => Math.round(v)); if (st.img === "cell") load(); }); return; }
      base = cellImg;
    } else if (id === "smooth") base = smooth();
    else if (id === "shapes") base = shapes();
    else if (id === "flat") base = flat(128);
    else if (id === "mask") base = shapes().map(v => (v > 150 ? 255 : 0));
    else base = fluor12();
    noisy();
    compute();
  }
  function noisy(): void {
    img = st.noise > 0 ? withNoise(base, st.noise * (maxV > 255 ? 16 : 1), 99, maxV) : base;
  }

  function compute(): void {
    const e = residuals(img, N, N, st.pred);
    const Hx = entropy(img), He = entropy(e);
    const bpv = maxV > 255 ? 2 : 1;
    // images
    drawScalarImage(cImg, img, N, N, { lo: 0, hi: maxV > 255 ? maxV : 255 });
    const R = maxV > 255 ? 512 : 32;
    drawScalarImage(cRes, e, N, N, { map: diverging, lo: -R, hi: R });
    marker();
    // histograms on one axis
    const bin = maxV > 255 ? 16 : 1;
    const lo = maxV > 255 ? -1024 : -128, hi = maxV;
    const binned = (a: ArrayLike<number>) => {
      const h = intHistogram(Float64Array.from(a, v => Math.floor(v / bin)), Math.floor(lo / bin), Math.floor(hi / bin));
      const xs = Float64Array.from({ length: h.counts.length + 1 }, (_, i) => (Math.floor(lo / bin) + i) * bin);
      const ys = Float64Array.from({ length: h.counts.length + 1 }, (_, i) => (i < h.counts.length ? h.counts[i] / a.length : 0));
      return { xs, ys };
    };
    const hp = binned(img), hr = binned(e);
    let ymax = 0;
    for (const v of hp.ys) ymax = Math.max(ymax, v);
    for (const v of hr.ys) ymax = Math.max(ymax, v);
    hist.clear();
    hist.setDomain([lo, hi], [0, ymax * 1.08 || 1]);
    hist.step(hp.xs, hp.ys, { color: "#1f77b4", width: 2 });
    hist.step(hr.xs, hr.ys, { color: "#ff7f0e", width: 2 });
    hist.vline(0, { color: "var(--qb-axis)", width: 1, dash: "4 3" });
    // scatter: 2500 pixels (fixed sample)
    const xs: number[] = [], ys: number[] = [];
    for (let k = 0; k < 2500; k++) {
      const i = (k * 7919 + 13) % (N * N);
      if (i % N === 0) continue;
      xs.push(img[i - 1]); ys.push(img[i]);
    }
    scatter.clear();
    scatter.setDomain([0, maxV], [0, maxV]);
    scatter.line([0, maxV], [0, maxV], { color: "var(--qb-grid)", width: 1.5 });
    scatter.scatter(xs, ys, { r: 1.6, color: "#1f77b4", opacity: 0.45 });
    const ideal = (H: number) => H * N * N / 8;
    const v = ++version;
    const text = (dx: number, de: number) => {
      const f = (b: number) => (Number.isFinite(b) ? Math.round(b).toLocaleString("en-US").replace(/,/g, " ") : "n/a").padStart(7);
      ro.textContent =
        `             pixels  resid.\n` +
        `H bits/px   ${Hx.toFixed(2).padStart(7)} ${He.toFixed(2).padStart(7)}\n` +
        `ideal B     ${f(ideal(Hx))} ${f(ideal(He))}\n` +
        `DEFLATE B   ${f(dx)} ${f(de)}\n` +
        `raw: ${fmtB(N * N * bpv)} (${8 * bpv} bits/px)`;
    };
    text(NaN, NaN);
    Promise.all([deflateSize(toBytes(img, bpv)), deflateSize(toBytes(e, bpv))])
      .then(([dx, de]) => { if (v === version) text(dx, de); })
      .catch(() => { /* leave n/a */ });
    pixelInfo();
  }

  // ---- click a pixel ----
  function marker(): void {
    if (st.px < 0) return;
    for (const c of [cImg, cRes]) {
      const ctx = c.getContext("2d")!;
      ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(st.px + 0.5, st.py + 0.5, 6, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  function pixelInfo(): void {
    if (st.px < 0) return;
    const x = st.px, y = st.py, i = y * N + x;
    const g = (xx: number, yy: number) => (xx >= 0 && yy >= 0 ? String(img[yy * N + xx]) : "–");
    const p = predictAt(st.pred, img, N, x, y);
    roPix.textContent =
      `pixel (${x}, ${y}) = ${img[i]}\n` +
      `c = ${g(x - 1, y - 1)}   b = ${g(x, y - 1)}\n` +
      `a = ${g(x - 1, y)}   x = ${img[i]}\n` +
      `${PRED_LABEL[st.pred]}: x̂ = ${p}, e = ${img[i] - p}`;
  }
  for (const c of [cImg, cRes]) {
    c.addEventListener("pointerdown", ev => {
      const r = c.getBoundingClientRect();
      st.px = Math.max(0, Math.min(N - 1, Math.floor((ev.clientX - r.left) / r.width * N)));
      st.py = Math.max(0, Math.min(N - 1, Math.floor((ev.clientY - r.top) / r.height * N)));
      compute();
    });
  }

  syncUI();
  load();
}

autoMount(NAME, mount);
