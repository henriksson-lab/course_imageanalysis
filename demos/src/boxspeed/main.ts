// Box filters of any size: what they cost, and repeated boxes look Gaussian.
//  Tab "speed": a 256 × 256 image blurred with a k × k box (k = 1..101); the additions per
//    pixel of four methods (direct k² − 1, separable 2(k − 1), running sum 4, integral
//    image 6) against k, and the measured time of each (median of 3 runs, this machine;
//    direct is skipped above k = 51).
//  Tab "repeat": n = 1..4 passes of a box of width k; the 1D impulse response (bars)
//    against the Gaussian of the same variance n(k² − 1)/12 (curve), the 2D impulse response
//    with a circle (a contour of that Gaussian), and the image after n passes.
// Options: data-tab="speed|repeat", data-image="cell|dots|star", data-k, data-n, data-preset.
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot } from "../lib/plot";
import { drawScalarImage, viridis, colormapLUT } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { siemensStar } from "../lib/synth";
import { BoxMethod, boxFilterBy, boxOpsPerPixel, repeatedBox1d, repeatedBoxVariance, boxMean } from "../lib/integral";

const NAME = "boxspeed";
const N = 256;
const METHODS: { id: BoxMethod; label: string; color: string }[] = [
  { id: "direct", label: "direct", color: "#d62728" },
  { id: "separable", label: "separable", color: "#ff7f0e" },
  { id: "running", label: "running sum", color: "#2ca02c" },
  { id: "integral", label: "integral image", color: "#1f77b4" },
];
const DIRECT_MAX = 51;

const STYLE = `
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .bs-img { flex: 0 1 280px; min-width: 180px; }
[data-demo="${NAME}"] .bs-plot { flex: 1 1 380px; min-width: 0; max-width: 520px; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 280px; gap: .4em; }
[data-demo="${NAME}"] .bs-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .bs-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
[data-demo="${NAME}"] .bs-tabs .qb-btn { font-weight: 600; }
.reveal [data-demo="${NAME}"] { font-size: 18px; gap: .8em; }
.reveal [data-demo="${NAME}"] .bs-img { flex-basis: 270px; }
.reveal [data-demo="${NAME}"] .bs-plot { flex-basis: 470px; max-width: 480px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 300px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type ImageId = "cell" | "dots" | "star";

function dotsImage(): Float64Array {
  const a = new Float64Array(N * N).fill(20);
  const pts = [[48, 48], [128, 64], [208, 52], [70, 140], [180, 150], [128, 210], [40, 220], [220, 220]];
  for (const [x, y] of pts) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) a[(y + dy) * N + x + dx] = 255;
  return a;
}

interface Preset { id: string; label: string; tab: "speed" | "repeat"; image: ImageId; k: number; n: number; method: BoxMethod }
const PRESETS: Preset[] = [
  { id: "slow", label: "direct at k = 51 is slow", tab: "speed", image: "cell", k: 51, n: 1, method: "direct" },
  { id: "flat", label: "integral image: flat line", tab: "speed", image: "cell", k: 101, n: 1, method: "integral" },
  { id: "square", label: "one box: square artefacts", tab: "repeat", image: "dots", k: 15, n: 1, method: "integral" },
  { id: "gauss", label: "three boxes ≈ Gaussian", tab: "repeat", image: "dots", k: 9, n: 3, method: "integral" },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const p0: Preset = PRESETS.find(p => p.id === d.preset) ?? {
    id: "", label: "", tab: d.tab === "repeat" ? "repeat" : "speed", image: (d.image as ImageId) ?? "cell",
    k: Number(d.k ?? 15), n: Number(d.n ?? 3), method: "integral",
  };
  const st = { ...p0 };
  const images: Partial<Record<ImageId, Float64Array>> = { dots: dotsImage(), star: Float64Array.from(siemensStar(N, 18, 30, 225).data) };

  // ---- layout ----
  const imgFig = htmlEl("figure", "bs-img", "", host);
  const imgCap = htmlEl("figcaption", "", "", imgFig);
  const cv = htmlEl("canvas", "", "", imgFig);
  const credit = htmlEl("div", "bs-credit", CELL_CREDIT, imgFig);
  const imgFig2 = htmlEl("figure", "", "", imgFig);
  const cap2 = htmlEl("figcaption", "", "", imgFig2);
  const cv2 = htmlEl("canvas", "", "", imgFig2);
  const plotBox = htmlEl("div", "bs-plot", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  const tabRow = htmlEl("div", "qb-presets bs-tabs", "", panel);
  const tabBtns = [button(tabRow, "speed", () => setTab("speed")), button(tabRow, "repeated boxes", () => setTab("repeat"))];
  htmlEl("div", "bs-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "bs-lab", "image", panel);
  const imRow = htmlEl("div", "qb-presets", "", panel);
  const imBtns = (["cell", "dots", "star"] as ImageId[]).map(id => button(imRow, id === "cell" ? "micrograph" : id === "dots" ? "points" : "star", () => { st.image = id; update(); }));
  const kS = slider(panel, "box width k", 1, 101, 2, st.k, v => { st.k = v; update(); });
  const nS = slider(panel, "passes n", 1, 4, 1, st.n, v => { st.n = v; update(); });
  const methodLab = htmlEl("div", "bs-lab", "time this method", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = METHODS.map(m => button(mRow, m.label, () => { st.method = m.id; update(); }));
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  function load(p: Preset): void {
    Object.assign(st, p);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    update();
  }
  function setTab(t: "speed" | "repeat"): void { st.tab = t; update(); }

  // ---- plots (rebuilt per tab) ----
  let plot: Plot | null = null, plotTab = "";
  function ensurePlot(): Plot {
    if (plot && plotTab === st.tab) return plot;
    plotBox.replaceChildren();
    plotTab = st.tab;
    plot = st.tab === "speed"
      ? new Plot(plotBox, { width: 480, height: 400, x: [1, 101], y: [1, 20000], ylog: true, xlabel: "box width k (pixels)", ylabel: "additions per pixel", xticks: [1, 21, 41, 61, 81, 101] })
      : new Plot(plotBox, { width: 480, height: 400, x: [-20, 20], y: [0, 0.3], xlabel: "offset (pixels)", ylabel: "weight" });
    return plot;
  }

  // ---- timing (debounced; the slow methods are timed only when the slider rests) ----
  const times: Partial<Record<BoxMethod, number | null>> = {};
  let timer = 0, timedKey = "";
  function scheduleTiming(img: Float64Array): void {
    const key = st.image + ":" + st.k;
    if (key === timedKey) return;
    for (const m of METHODS) times[m.id] = undefined;
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      for (const m of METHODS) {
        if (m.id === "direct" && st.k > DIRECT_MAX) { times[m.id] = null; continue; }
        const runs: number[] = [];
        const reps = m.id === "direct" && st.k > 21 ? 1 : 3;
        for (let r = 0; r < reps; r++) {
          const t0 = performance.now();
          boxFilterBy(m.id, img, N, N, st.k);
          runs.push(performance.now() - t0);
        }
        runs.sort((a, b) => a - b);
        times[m.id] = runs[runs.length >> 1];
      }
      timedKey = key;
      if (st.tab === "speed") drawSpeed(img);
    }, 350);
  }

  function getImage(): Float64Array | null {
    if (st.image === "cell" && !images.cell) {
      loadCell(N).then(g => { images.cell = Float64Array.from(g.data); update(); });
      return null;
    }
    return images[st.image]!;
  }

  function sync(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === (st.tab === "repeat")));
    imBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "dots", "star"][i] === st.image));
    mBtns.forEach((b, i) => b.classList.toggle("active", METHODS[i].id === st.method));
    const speed = st.tab === "speed";
    (nS.input.parentElement as HTMLElement).style.display = speed ? "none" : "";
    methodLab.style.display = mRow.style.display = speed ? "" : "none";
    imgFig2.style.display = speed ? "none" : "";
    credit.style.display = st.image === "cell" ? "" : "none";
    kS.input.max = speed ? "101" : "21";
    if (!speed && st.k > 21) st.k = 21;
    kS.set(st.k); nS.set(st.n);
  }

  function update(): void {
    sync();
    const img = getImage();
    if (!img) return;
    if (st.tab === "speed") { drawSpeed(img); scheduleTiming(img); } else drawRepeat(img);
  }

  function drawSpeed(img: Float64Array): void {
    const out = boxMean(img, N, N, (st.k - 1) >> 1);
    imgCap.textContent = `${st.k} × ${st.k} box mean`;
    drawScalarImage(cv, out, N, N, { lo: 0, hi: 255 });
    const P = ensurePlot();
    P.clear();
    const ks = Array.from({ length: 51 }, (_, i) => 2 * i + 1);
    for (const m of METHODS) P.line(ks, ks.map(k => Math.max(1, boxOpsPerPixel(m.id, k))), { color: m.color, width: m.id === st.method ? 5 : 2.5, label: m.label });
    P.vline(st.k, { color: "#555" });
    for (const m of METHODS) P.scatter([st.k], [Math.max(1, boxOpsPerPixel(m.id, st.k))], { color: m.color, r: 7 });
    P.legend(undefined, "tl");
    const fmt = (m: BoxMethod) => {
      const t = times[m];
      return t === undefined ? "  …" : t === null ? "  skipped (too slow)" : `${t.toFixed(1).padStart(6)} ms`;
    };
    ro.textContent = `additions per pixel at k = ${st.k}\n` +
      METHODS.map(m => `${m.label.padEnd(15)}${String(boxOpsPerPixel(m.id, st.k)).padStart(6)}`).join("\n") +
      `\n\nmeasured, ${N} × ${N} image\n(this machine, median of 3)\n` +
      METHODS.map(m => `${m.label.padEnd(15)}${fmt(m.id)}`).join("\n");
    note.textContent = "The time depends on the computer and the browser; the counts do not. All four methods give the same image (windows clipped at the border).";
  }

  function drawRepeat(img: Float64Array): void {
    const k = st.k, n = st.n;
    const h = repeatedBox1d(k, n), L = h.length, c = (L - 1) / 2;
    const v = repeatedBoxVariance(k, n), sigma = Math.sqrt(v);
    const g = (x: number) => v > 0 ? Math.exp(-x * x / (2 * v)) / Math.sqrt(2 * Math.PI * v) : (x === 0 ? 1 : 0);
    let maxDiff = 0, peak = 0;
    for (let i = 0; i < L; i++) { maxDiff = Math.max(maxDiff, Math.abs(h[i] - g(i - c))); peak = Math.max(peak, g(i - c)); }
    // 1D plot
    const P = ensurePlot();
    P.clear();
    const half = Math.max(8, Math.ceil(c + 2));
    const ymax = Math.max(...Array.from(h)) * 1.45;
    P.setDomain([-half, half], [0, ymax]);
    P.bars(Array.from({ length: L }, (_, i) => i - c), h, { color: "#1f77b4", label: `${n} × box, width ${k}` });
    P.fn(g, { color: "#d62728", width: 3, label: `Gaussian, σ = ${sigma.toFixed(2)}` }, 300);
    P.legend(undefined, "tl");
    // 2D impulse response with a contour circle of the Gaussian at 1/4 of its peak
    const S = 4, R = 2 * half + 1, size = R * S;
    cv2.width = size; cv2.height = size;
    const ctx = cv2.getContext("2d")!;
    const lut = colormapLUT(viridis, 256);
    const id = ctx.createImageData(R, R);
    const h2max = h[Math.round(c)] ** 2;
    for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
      const ix = x - half + c, iy = y - half + c;
      const val = Number.isInteger(ix) && Number.isInteger(iy) && ix >= 0 && iy >= 0 && ix < L && iy < L ? h[ix] * h[iy] / h2max : 0;
      const band = Math.floor(val * 8) / 8; // 8 bands make the contours visible
      const li = Math.round(band * 255), p = 4 * (y * R + x);
      id.data[p] = lut[4 * li]; id.data[p + 1] = lut[4 * li + 1]; id.data[p + 2] = lut[4 * li + 2]; id.data[p + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = R; tmp.height = R;
    tmp.getContext("2d")!.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, size, size);
    if (v > 0) {
      const rr = sigma * Math.sqrt(2 * Math.log(4)) * S;
      ctx.strokeStyle = "#ff4040"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(size / 2, size / 2, rr, 0, 2 * Math.PI); ctx.stroke();
    }
    cap2.textContent = "2D impulse response (bands), Gaussian contour (red)";
    // image after n passes
    let out = img;
    for (let p = 0; p < n; p++) out = boxMean(out, N, N, (k - 1) >> 1);
    imgCap.textContent = `${n} pass${n > 1 ? "es" : ""} of a ${k} × ${k} box`;
    drawScalarImage(cv, out, N, N, st.image === "dots" ? {} : { lo: 0, hi: 255 });
    ro.textContent = `variance n(k² − 1)/12 = ${v.toFixed(2)}\n` +
      `matching Gaussian: σ = ${sigma.toFixed(2)}\n` +
      `largest difference:\n  ${(100 * maxDiff / peak).toFixed(1)} % of its peak\n` +
      `cost: ${n} × 6 = ${6 * n} additions/pixel`;
    note.textContent = "The points image shows each bright point as the 2D impulse response: a square for one pass, rounder with more passes.";
  }

  load(p0);
}

autoMount(NAME, mount);
