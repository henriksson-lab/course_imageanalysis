// The 2D wavelet transform of a 256 × 256 image: the subband pyramid (Mallat layout: LL
// top-left, then V, H, D bands per level; detail bands in a diverging map, blue −, red +),
// the reconstruction from the kept coefficients and the error. "Keep K largest" keeps a
// percentage of all coefficients; "zero bands" removes whole subbands (click them). With
// "compare", the same number of coefficients is kept in the global Fourier basis and in the
// 8 × 8 block DCT. The plot shows PSNR against the percentage kept for the three bases.
// "Shift by 1 px" moves the image one pixel to the right (circularly): the decimated
// coefficients change a lot. Orthogonal wavelets use the periodic boundary; CDF 9/7 the
// symmetric one (lifting).
// Auto-mounts on <div data-demo="dwt2d">. Optional: data-image="cell|disc|square|bars|checker|gradient",
// data-wavelet="haar|db2|db4|db8|cdf97", data-levels="1..6", data-keep="3", data-preset="…".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { WAVELETS, WaveletName, dwt2, idwt2, bandAt, Band, subbandRect, tailErrors } from "../lib/wavelet";
import { fft2d } from "../lib/fft";
import { psnr, psnrFromMse } from "../lib/metrics";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { N, ImageName, IMAGE_LABELS, synthImage, blockDct, drawImage, drawSigned, drawPyramid, PyramidDisplay } from "./images";

const NAME = "dwt2d";
const WNAMES: WaveletName[] = ["haar", "db2", "db4", "db8", "cdf97"];
const IMAGES: ImageName[] = ["cell", "disc", "square", "bars", "checker", "gradient"];
const KEEPS = [0.5, 1, 2, 3, 5, 10, 20, 50, 100];
const COL = { w: seriesColor(0), f: seriesColor(1), d: seriesColor(2) };

interface Settings {
  image: ImageName; wavelet: WaveletName; levels: number; keep: number; mode: "keep" | "zero";
  compare: boolean; shift: boolean; display: PyramidDisplay; dropped: string[];
}
const DEFAULTS: Settings = { image: "cell", wavelet: "haar", levels: 3, keep: 100, mode: "keep", compare: false, shift: false, display: "band", dropped: [] };
interface Preset { id: string; label: string; s: Partial<Settings> }
const PRESETS: Preset[] = [
  { id: "orient", label: "one level: three orientations", s: { image: "bars", wavelet: "haar", levels: 1 } },
  { id: "pyramid", label: "pyramid of the cell", s: { image: "cell", wavelet: "db2", levels: 4 } },
  { id: "keep3", label: "keep 3 %: Haar blocks", s: { image: "cell", wavelet: "haar", levels: 4, keep: 3 } },
  { id: "compare", label: "wavelets vs Fourier vs DCT at 5 %", s: { image: "cell", wavelet: "db4", levels: 4, keep: 5, compare: true } },
  { id: "dropfine", label: "drop the finest level = blur", s: { image: "cell", wavelet: "db4", levels: 3, mode: "zero", dropped: ["1:V", "1:H", "1:D"] } },
  { id: "shift", label: "shift variance", s: { image: "square", wavelet: "haar", levels: 2, shift: true } },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .d2-grid { flex: 0 1 540px; min-width: 260px; display: grid; grid-template-columns: 1fr 1fr; gap: .3em .5em; }
[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.15; margin: 0 0 .15em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 2px; }
[data-demo="${NAME}"] canvas.pick { cursor: pointer; }
[data-demo="${NAME}"] .d2-side { flex: 1 1 250px; max-width: 330px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 240px; max-width: 320px; gap: .38em; }
[data-demo="${NAME}"] .d2-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .d2-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-slider.off { opacity: .35; pointer-events: none; }
.reveal [data-demo="${NAME}"] { font-size: 16px; gap: .5em; }
.reveal [data-demo="${NAME}"] .d2-grid { flex-basis: 470px; gap: .1em .4em; }
.reveal [data-demo="${NAME}"] figcaption { font-size: 15px; }
.reveal [data-demo="${NAME}"] .d2-side { flex-basis: 270px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 290px; gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 13px; padding: .1em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const f1 = (v: number) => (Number.isFinite(v) && v < 100 ? v.toFixed(1) : "∞ (exact)");

/** Keep the k largest |c| (indices sorted once per transform). */
function keepTop(c: ArrayLike<number>, order: ArrayLike<number>, k: number): Float64Array {
  const out = new Float64Array(c.length);
  for (let r = 0; r < k && r < order.length; r++) out[order[r]] = c[order[r]];
  return out;
}
function orderOf(mag: Float64Array): Uint32Array {
  return Uint32Array.from({ length: mag.length }, (_, i) => i).sort((a, b) => mag[b] - mag[a]);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: Settings = { ...DEFAULTS };
  if (IMAGES.includes(host.dataset.image as ImageName)) init.image = host.dataset.image as ImageName;
  if (WNAMES.includes(host.dataset.wavelet as WaveletName)) init.wavelet = host.dataset.wavelet as WaveletName;
  if (host.dataset.levels) init.levels = Math.max(1, Math.min(6, Number(host.dataset.levels) || 3));
  if (host.dataset.keep && KEEPS.includes(Number(host.dataset.keep))) init.keep = Number(host.dataset.keep);
  if (p0) Object.assign(init, p0.s);
  const st: Settings = { ...init, dropped: [...init.dropped] };

  // ---- layout ----
  const grid = htmlEl("div", "d2-grid", "", host);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", grid); const c = htmlEl("figcaption", "", cap, f); const cv = htmlEl("canvas", "", "", f); cv.width = N; cv.height = N; return { c, cv }; };
  const F = [fig("image"), fig("subbands"), fig("reconstruction"), fig("error")];
  const side = htmlEl("div", "d2-side", "", host);
  const pBox = htmlEl("div", "", "", side);
  const plot = new Plot(pBox, { width: 360, height: 300, x: [0.5, 100], y: [10, 60], xlog: true, xlabel: "% kept", ylabel: "PSNR (dB)", xticks: [1, 10, 100] });
  const ro = htmlEl("div", "qb-readout", "", side);
  const credit = htmlEl("div", "d2-credit", CELL_CREDIT, side);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "d2-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULTS, p.s); st.dropped = [...(p.s.dropped ?? [])]; newImage(); } })));
  htmlEl("div", "d2-lab", "image", panel);
  const iRow = htmlEl("div", "qb-presets", "", panel);
  const iBtns = IMAGES.map(im => button(iRow, IMAGE_LABELS[im], () => { st.image = im; changed(); newImage(); }));
  htmlEl("div", "d2-lab", "wavelet", panel);
  const wRow = htmlEl("div", "qb-presets", "", panel);
  const wBtns = WNAMES.map(w => button(wRow, WAVELETS[w].label, () => { st.wavelet = w; changed(); transform(); }));
  const levS = slider(panel, "levels", 1, 6, 1, st.levels, v => { st.levels = v; st.dropped = []; changed(); transform(); });
  htmlEl("div", "d2-lab", "mode", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = [
    button(mRow, "keep K largest", () => { st.mode = "keep"; changed(); rebuild(); }),
    button(mRow, "zero bands (click)", () => { st.mode = "zero"; changed(); rebuild(); }),
  ];
  const kS = slider(panel, "keep %", 0, KEEPS.length - 1, 1, KEEPS.indexOf(st.keep), v => { st.keep = KEEPS[v]; st.mode = "keep"; changed(); rebuild(); }, v => String(KEEPS[v]));
  const cmpBox = checkbox(panel, "compare: Fourier and 8×8 DCT", st.compare, v => { st.compare = v; changed(); rebuild(); });
  const shBox = checkbox(panel, "shift by 1 px", st.shift, v => { st.shift = v; changed(); newImage(); });
  const dRow = htmlEl("div", "qb-presets", "", panel);
  const dBtns = [
    button(dRow, "per-band colours", () => { st.display = "band"; changed(); rebuild(); }),
    button(dRow, "log |w|", () => { st.display = "log"; changed(); rebuild(); }),
  ];
  button(panel, "reset", () => { Object.assign(st, init); st.dropped = [...init.dropped]; newImage(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  htmlEl("div", "qb-note", "Subbands: grey = LL (approximation); colours = detail bands (blue negative, red positive), each scaled to its own largest value. Faded = dropped.", panel);

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    iBtns.forEach((b, i) => b.classList.toggle("active", IMAGES[i] === st.image));
    wBtns.forEach((b, i) => b.classList.toggle("active", WNAMES[i] === st.wavelet));
    mBtns.forEach((b, i) => b.classList.toggle("active", ["keep", "zero"][i] === st.mode));
    dBtns.forEach((b, i) => b.classList.toggle("active", ["band", "log"][i] === st.display));
    levS.set(st.levels); kS.set(KEEPS.indexOf(st.keep)); cmpBox.checked = st.compare; shBox.checked = st.shift;
    kS.input.parentElement!.classList.toggle("off", st.mode !== "keep");
    F[1].cv.classList.toggle("pick", st.mode === "zero");
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
  }

  // ---- data ----
  type F64 = Float64Array<ArrayBufferLike>;
  type U32 = Uint32Array<ArrayBufferLike>;
  let cell: F64 | null = null;
  let img: F64 = new Float64Array(N * N), base: F64 = new Float64Array(N * N);
  let coef: F64 = new Float64Array(N * N), wOrder: U32 = new Uint32Array(0), coefUnshifted: F64 | null = null;
  let fRe: F64 = new Float64Array(N * N), fIm: F64 = new Float64Array(N * N), fOrder: U32 = new Uint32Array(0);
  let dct: F64 = new Float64Array(N * N), dOrder: U32 = new Uint32Array(0);
  let curves: { w: Float64Array; f: Float64Array; d: Float64Array } | null = null;

  function newImage(): void {
    sync();
    if (st.image === "cell" && !cell) { loadCell().then(g => { cell = Float64Array.from(g.data); newImage(); }); return; }
    base = st.image === "cell" ? Float64Array.from(cell!) : synthImage(st.image);
    img = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) img[y * N + x] = base[y * N + ((x - (st.shift ? 1 : 0) + N) % N)];
    // Fourier and DCT (independent of the wavelet)
    fRe = Float64Array.from(img); fIm = new Float64Array(N * N);
    fft2d(fRe, fIm, N);
    const fMag = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) fMag[i] = Math.hypot(fRe[i], fIm[i]);
    fOrder = orderOf(fMag);
    dct = blockDct(img, N);
    dOrder = orderOf(Float64Array.from(dct, Math.abs));
    transform();
  }

  function transform(): void {
    sync();
    const w = WAVELETS[st.wavelet];
    coef = dwt2(img, N, N, w, st.levels);
    wOrder = orderOf(Float64Array.from(coef, Math.abs));
    coefUnshifted = st.shift ? dwt2(base, N, N, w, st.levels) : null;
    // PSNR curves at the KEEPS percentages
    const mseOfTail = (vals: ArrayLike<number>, scale: number) => {
      const te = tailErrors(vals);   // relative error sqrt(tail/total)
      let tot = 0;
      for (let i = 0; i < vals.length; i++) tot += vals[i] * vals[i];
      return KEEPS.map(p => { const k = Math.round(p / 100 * N * N); return te[k] ** 2 * tot * scale / (N * N); });
    };
    const fAbs = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) fAbs[i] = Math.hypot(fRe[i], fIm[i]);
    const mseF = mseOfTail(fAbs, 1 / (N * N));
    const mseD = mseOfTail(dct, 1);
    let mseW: number[];
    if (w.kind === "orth") mseW = mseOfTail(coef, 1);
    else mseW = KEEPS.map(p => { const r = idwt2(keepTop(coef, wOrder, Math.round(p / 100 * N * N)), N, N, w, st.levels); let s = 0; for (let i = 0; i < N * N; i++) s += (r[i] - img[i]) ** 2; return s / (N * N); });
    const toP = (m: number[]) => Float64Array.from(m, v => Math.min(psnrFromMse(v, 255), 60));
    curves = { w: toP(mseW), f: toP(mseF), d: toP(mseD) };
    plot.clear();
    const xs = Float64Array.from(KEEPS);
    plot.line(xs, curves.f, { color: COL.f, label: "Fourier" });
    plot.line(xs, curves.d, { color: COL.d, label: "8×8 DCT" });
    plot.line(xs, curves.w, { color: COL.w, label: w.label });
    plot.scatter(xs, curves.w, { color: COL.w, r: 4 });
    plot.legend(undefined, "tl");
    keepMark = null;
    rebuild();
  }

  let keepMark: ReturnType<Plot["vline"]> | null = null;
  let pending = false;
  function rebuild(): void {
    sync();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; render(); });
  }

  function render(): void {
    const w = WAVELETS[st.wavelet], L = st.levels;
    const dropped = new Set(st.dropped);
    let kept: Float64Array, keptMask: Uint8Array | undefined, nKept: number;
    if (st.mode === "keep") {
      const k = Math.round(st.keep / 100 * N * N);
      kept = keepTop(coef, wOrder, k);
      nKept = k;
      keptMask = st.keep < 100 ? Uint8Array.from(kept, v => (v !== 0 ? 1 : 0)) : undefined;
    } else {
      kept = Float64Array.from(coef);
      nKept = N * N;
      for (const key of dropped) {
        const [lv, b] = key.split(":"), r = subbandRect(Number(lv), b as Band, N, N);
        for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) kept[y * N + x] = 0;
        nKept -= r.w * r.h;
      }
    }
    const rec = idwt2(kept, N, N, w, L);
    const pW = psnr(img, rec, 255);
    drawImage(F[0].cv, img, N);
    F[0].c.textContent = st.shift ? "image, shifted by 1 px" : "image";
    drawPyramid(F[1].cv, coef, N, L, st.display, st.mode === "zero" ? dropped : new Set(), st.mode === "keep" ? keptMask : undefined);
    F[1].c.textContent = st.mode === "zero" ? `${w.label}, ${L} level${L > 1 ? "s" : ""}: click a band` : `${w.label} subbands, ${L} level${L > 1 ? "s" : ""}`;
    const k = nKept;
    const lines: string[] = [`kept ${k} of ${N * N} (${(100 * k / (N * N)).toFixed(1)} %)`, `PSNR ${w.label.padEnd(8)} ${f1(pW)} dB`];
    if (st.compare) {
      const kk = Math.min(k, N * N);
      const re = new Float64Array(N * N), im = new Float64Array(N * N);
      for (let r = 0; r < kk; r++) { const i = fOrder[r]; re[i] = fRe[i]; im[i] = fIm[i]; }
      fft2d(re, im, N, true);
      const recD = blockDct(keepTop(dct, dOrder, kk), N, true);
      const pF = psnr(img, re, 255), pD = psnr(img, recD, 255);
      drawImage(F[0].cv, rec, N); F[0].c.textContent = `${w.label}: ${f1(pW)} dB`;
      drawImage(F[2].cv, re, N); F[2].c.textContent = `Fourier: ${f1(pF)} dB`;
      drawImage(F[3].cv, recD, N); F[3].c.textContent = `8×8 DCT: ${f1(pD)} dB`;
      lines.push(`PSNR Fourier  ${f1(pF)} dB`, `PSNR 8×8 DCT  ${f1(pD)} dB`);
    } else {
      drawImage(F[2].cv, rec, N);
      F[2].c.textContent = "reconstruction";
      const err = Float64Array.from(rec, (v, i) => img[i] - v);
      drawSigned(F[3].cv, err, N, 40);
      F[3].c.textContent = "error (±40 grey levels)";
    }
    // shift variance: change of the level-1 detail coefficients
    const lev1 = (c: ArrayLike<number>) => {
      const e: Record<string, number> = {};
      for (const b of ["V", "H", "D"] as Band[]) {
        const r = subbandRect(1, b, N, N);
        let s = 0;
        for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) s += c[y * N + x] ** 2;
        e[b] = s;
      }
      return e;
    };
    if (coefUnshifted) {
      const e1 = lev1(coef), e0 = lev1(coefUnshifted);
      lines.push("level-1 energy  before → after");
      for (const b of ["V", "H", "D"]) lines.push(`  ${b}  ${e0[b].toExponential(2)} → ${e1[b].toExponential(2)}`);
      let d = 0, t = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (x >= N >> L || y >= N >> L) { const i = y * N + x; d += (coef[i] - coefUnshifted[i]) ** 2; t += coefUnshifted[i] ** 2; }
      lines.push(`details changed by ${(100 * Math.sqrt(d / t)).toFixed(0)} %`);
    }
    ro.textContent = lines.join("\n");
    keepMark?.remove();
    keepMark = st.mode === "keep" ? plot.vline(Math.max(0.5, st.keep), { color: "var(--qb-fg)", width: 2 }) : null;
  }

  // click on the subband image: toggle a band (zero mode)
  F[1].cv.addEventListener("click", ev => {
    if (st.mode !== "zero") return;
    const r = F[1].cv.getBoundingClientRect();
    const x = Math.floor((ev.clientX - r.left) / r.width * N), y = Math.floor((ev.clientY - r.top) / r.height * N);
    const b = bandAt(x, y, N, N, st.levels), key = `${b.level}:${b.band}`;
    st.dropped = st.dropped.includes(key) ? st.dropped.filter(k => k !== key) : [...st.dropped, key];
    changed();
    rebuild();
  });

  newImage();
  if (p0) presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
}

autoMount(NAME, mount);
