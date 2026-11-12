// Local mean, standard deviation and contrast normalisation from two integral images
// (of I and I²), and what goes wrong with the numbers.
//  Tab "stats": input, local mean m, local SD s, and (I − m)/(s + ε); window and ε sliders;
//    click the input to choose a row: its profile shows I, m and m ± s.
//  Tab "precision": the tables in float32, float64 or 32-bit integers (wrapping), with or
//    without subtracting an offset first; a constant can be added to the image to make it
//    bright, and the image tiled to 1024 × 1024. Shows the SD computed this way, its error
//    against an exact computation, the largest error and the number of negative variances.
// Options: data-tab="stats|precision", data-image="cell|synth", data-preset.
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot } from "../lib/plot";
import { drawScalarImage, inferno } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { N, synthImage } from "./model";
import { localStats, contrastNormalise, integralImage, TableKind } from "../lib/integral";

const NAME = "localstats";

const STYLE = `
[data-demo="${NAME}"] figure { margin: 0; position: relative; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; white-space: nowrap; }
[data-demo="${NAME}"] canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .ls-grid { flex: 0 1 420px; min-width: 260px; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: .4em .5em; }
[data-demo="${NAME}"] .ls-grid canvas.ls-click { cursor: crosshair; }
[data-demo="${NAME}"] .ls-plot { flex: 1 1 330px; min-width: 0; max-width: 460px; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 270px; gap: .4em; }
[data-demo="${NAME}"] .ls-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .ls-credit { font-size: 12px; color: var(--qb-muted); grid-column: 1 / 3; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
[data-demo="${NAME}"] .ls-tabs .qb-btn { font-weight: 600; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .7em; }
.reveal [data-demo="${NAME}"] .ls-grid { flex-basis: 400px; }
.reveal [data-demo="${NAME}"] .ls-plot { flex-basis: 380px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 300px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note, .reveal [data-demo="${NAME}"] .ls-credit { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset {
  id: string; label: string; tab: "stats" | "precision"; image: "cell" | "synth";
  r: number; eps: number; kind: TableKind; subtract: boolean; add: number; size: 256 | 1024;
}
const BASE = { r: 8, eps: 4, kind: "f64" as TableKind, subtract: false, add: 0, size: 256 as const };
const PRESETS: Preset[] = [
  { id: "equalise", label: "normalisation equalises the cells", tab: "stats", image: "synth", ...BASE },
  { id: "eps0", label: "ε = 0: background noise blows up", tab: "stats", image: "synth", ...BASE, eps: 0 },
  { id: "cell", label: "micrograph, SD = texture", tab: "stats", image: "cell", ...BASE, r: 4 },
  { id: "negative", label: "float32, bright: negative variances", tab: "precision", image: "synth", ...BASE, r: 3, kind: "f32", add: 200 },
  { id: "offset", label: "subtract the offset: much better", tab: "precision", image: "synth", ...BASE, r: 3, kind: "f32", add: 200, subtract: true },
  { id: "wrap", label: "32-bit integers wrap, sums are right", tab: "precision", image: "synth", ...BASE, r: 3, kind: "u32", add: 200, size: 1024 },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const p0: Preset = PRESETS.find(p => p.id === d.preset) ??
    { id: "", label: "", tab: d.tab === "precision" ? "precision" : "stats", image: d.image === "cell" ? "cell" : "synth", ...BASE };
  const st = { ...p0, row: 64 };
  const images: { cell?: Float64Array; synth: Float64Array } = { synth: synthImage() };

  // ---- layout ----
  const grid = htmlEl("div", "ls-grid", "", host);
  const figs = [0, 1, 2, 3].map(() => {
    const f = htmlEl("figure", "", "", grid);
    const cap = htmlEl("figcaption", "", "", f);
    const cv = htmlEl("canvas", "", "", f);
    return { cap, cv };
  });
  figs[0].cv.classList.add("ls-click");
  const credit = htmlEl("div", "ls-credit", CELL_CREDIT, grid);
  const plotBox = htmlEl("div", "ls-plot", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  const tabRow = htmlEl("div", "qb-presets ls-tabs", "", panel);
  const tabBtns = [button(tabRow, "local statistics", () => { st.tab = "stats"; update(); }), button(tabRow, "precision", () => { st.tab = "precision"; update(); })];
  htmlEl("div", "ls-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ls-lab", "image", panel);
  const imRow = htmlEl("div", "qb-presets", "", panel);
  const imBtns = [button(imRow, "synthetic cells", () => { st.image = "synth"; update(); }), button(imRow, "micrograph", () => { st.image = "cell"; update(); })];
  const rS = slider(panel, "window r", 1, 30, 1, st.r, v => { st.r = v; update(); }, v => `${2 * v + 1}²`);
  const eS = slider(panel, "ε", 0, 30, 1, st.eps, v => { st.eps = v; update(); });
  const precBox = htmlEl("div", "", "", panel);
  precBox.style.cssText = "display:flex;flex-direction:column;gap:.35em";
  htmlEl("div", "ls-lab", "number type of the two tables", precBox);
  const kRow = htmlEl("div", "qb-presets", "", precBox);
  const KINDS: { id: TableKind; label: string }[] = [{ id: "f32", label: "float32" }, { id: "f64", label: "float64" }, { id: "u32", label: "32-bit int" }];
  const kBtns = KINDS.map(k => button(kRow, k.label, () => { st.kind = k.id; update(); }));
  const subC = checkbox(precBox, "subtract an offset (the mean) first", st.subtract, v => { st.subtract = v; update(); });
  const aS = slider(precBox, "add to image", 0, 200, 10, st.add, v => { st.add = v; update(); });
  const sRow = htmlEl("div", "qb-presets", "", precBox);
  const sBtns = [button(sRow, "256 × 256", () => { st.size = 256; update(); }), button(sRow, "1024 × 1024 (tiled)", () => { st.size = 1024; update(); })];
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  function load(p: Preset): void {
    Object.assign(st, p);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    update();
  }

  figs[0].cv.addEventListener("pointerdown", ev => {
    if (st.tab !== "stats") return;
    const rc = figs[0].cv.getBoundingClientRect();
    st.row = Math.max(0, Math.min(N - 1, Math.floor((ev.clientY - rc.top) / rc.height * N)));
    update();
  });

  let plot: Plot | null = null;
  function ensurePlot(): Plot {
    if (!plot) plot = new Plot(plotBox, { width: 440, height: 360, x: [0, N - 1], y: [0, 255], xlabel: "x along the row", ylabel: "grey level" });
    return plot;
  }

  function sync(): void {
    const stats = st.tab === "stats";
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === !stats));
    imBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === (st.image === "cell")));
    kBtns.forEach((b, i) => b.classList.toggle("active", KINDS[i].id === st.kind));
    sBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === (st.size === 1024)));
    precBox.style.display = stats ? "none" : "flex";
    plotBox.style.display = stats ? "" : "none";
    (eS.input.parentElement as HTMLElement).style.display = stats ? "" : "none";
    rS.set(st.r); eS.set(st.eps); aS.set(st.add); subC.checked = st.subtract;
    credit.style.display = st.image === "cell" ? "" : "none";
  }

  function update(): void {
    sync();
    if (st.image === "cell" && !images.cell) {
      loadCell(N).then(g => { images.cell = Float64Array.from(g.data, v => Math.round(v)); update(); });
      return;
    }
    const img = images[st.image]!;
    if (st.tab === "stats") drawStats(img); else drawPrecision(img);
  }

  function drawStats(img: Float64Array): void {
    const ls = localStats(img, N, N, st.r);
    const cn = contrastNormalise(img, ls, st.eps);
    figs[0].cap.textContent = "input I (click: row)";
    figs[1].cap.textContent = "local mean m";
    figs[2].cap.textContent = "local SD s";
    figs[3].cap.textContent = `(I − m) / (s + ${st.eps})`;
    drawScalarImage(figs[0].cv, img, N, N, { lo: 0, hi: 255 });
    const ctx = figs[0].cv.getContext("2d")!;
    ctx.fillStyle = "rgba(255,127,14,0.9)"; ctx.fillRect(0, st.row, N, 2);
    drawScalarImage(figs[1].cv, ls.mean, N, N, { lo: 0, hi: 255 });
    let smax = 0;
    for (const v of ls.sd) smax = Math.max(smax, v);
    drawScalarImage(figs[2].cv, ls.sd, N, N, { lo: 0, hi: smax });
    drawScalarImage(figs[3].cv, cn, N, N, { lo: -3, hi: 3 });
    // profile
    const P = ensurePlot();
    P.clear();
    const xs = Array.from({ length: N }, (_, i) => i), row = st.row * N;
    const I = xs.map(x => img[row + x]), m = xs.map(x => ls.mean[row + x]), s = xs.map(x => ls.sd[row + x]);
    P.band(xs, m.map((v, i) => v - s[i]), m.map((v, i) => v + s[i]), { color: "#1f77b4", fillOpacity: 0.25, label: "m ± s" });
    P.line(xs, I, { color: "#999", width: 1.5, label: "I" });
    P.line(xs, m, { color: "#1f77b4", width: 3, label: "m" });
    P.legend(undefined, "tl");
    // readout: mean |normalised| inside the cells vs background on this image
    let sAbs = 0;
    for (let i = 0; i < cn.length; i++) sAbs += Math.abs(cn[i]);
    ro.textContent = `window ${2 * st.r + 1} × ${2 * st.r + 1}, row y = ${st.row}\n` +
      `largest local SD: ${smax.toFixed(1)}\n` +
      `mean |normalised|: ${(sAbs / cn.length).toFixed(2)}`;
    note.textContent = "Two tables (of I and of I²), four lookups each per pixel, give m and s for any window size. The SD map is bright at edges and in textured regions.";
  }

  function drawPrecision(base: Float64Array): void {
    const n = st.size, reps = n / N;
    const img = new Float64Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) img[y * n + x] = base[(y % N) * N + (x % N)] + st.add;
    let sum = 0;
    for (const v of img) sum += v;
    const offset = Math.round(sum / img.length);
    const exact = localStats(img, n, n, st.r, { offset });        // integers in float64: exact
    const got = localStats(img, n, n, st.r, { kind: st.kind, offset: st.subtract ? offset : 0 });
    const err = new Float64Array(n * n);
    let emax = 0, esum = 0;
    for (let i = 0; i < err.length; i++) {
      const e = Math.abs(got.sd[i] - exact.sd[i]);
      err[i] = got.variance[i] < 0 ? NaN : e;
      if (e > emax) emax = e;
      esum += e;
    }
    // largest table entries (of the image as it went into the tables)
    const shift = st.subtract ? offset : 0;
    let total2 = 0;
    for (const v of img) total2 += (v - shift) ** 2;
    const T2 = integralImage(img, n, n, st.kind, v => (v - shift) ** 2);
    const last2 = T2.data[T2.data.length - 1];
    figs[0].cap.textContent = `input${st.add ? " + " + st.add : ""}, ${n}²`;
    figs[1].cap.textContent = `SD, ${KINDS.find(k => k.id === st.kind)!.label} tables`;
    figs[2].cap.textContent = "SD, exact";
    figs[3].cap.textContent = "SD error (red: var. < 0)";
    drawScalarImage(figs[0].cv, img, n, n, { lo: 0, hi: 455 });
    let smax = 0;
    for (const v of exact.sd) smax = Math.max(smax, v);
    drawScalarImage(figs[1].cv, got.sd, n, n, { lo: 0, hi: smax });
    drawScalarImage(figs[2].cv, exact.sd, n, n, { lo: 0, hi: smax });
    drawScalarImage(figs[3].cv, err, n, n, { map: inferno, lo: 0, hi: Math.max(emax, 1e-9), nan: [255, 40, 40, 255] });
    const fmtBig = (v: number) => v.toExponential(2);
    ro.textContent = `offset subtracted: ${st.subtract ? offset : "none"}\n` +
      `sum of (I − c)²:    ${fmtBig(total2)}\n` +
      `  float32 exact to  2^24 ≈ 1.7e7\n` +
      `  32-bit int to     2^32 ≈ 4.3e9\n` +
      `stored last entry:  ${fmtBig(last2)}\n` +
      `largest SD error:   ${emax.toPrecision(3)}\n` +
      `mean SD error:      ${(esum / err.length).toPrecision(3)}\n` +
      `negative variances: ${got.negatives}`;
    note.textContent = `Exact: the same formula with the mean (${offset}) subtracted, in float64 (all sums are integers below 2^53). ${reps > 1 ? "The image is tiled " + reps + " × " + reps + "." : ""}`;
  }

  load(p0);
}

autoMount(NAME, mount);
