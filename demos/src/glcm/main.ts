// Grey-level co-occurrence matrix of a texture patch. Left: the patch (64 × 64, zoomed)
// with a few pixel pairs at the current offset drawn as arrows; middle: the GLCM as a heat
// map (rows: level at x, columns: level at x + d; the diagonal dashed); right: the main
// Haralick features and a polar plot of the contrast against the angle at the current
// distance. Compare mode adds a second patch B with its own GLCM and feature column.
// Controls: distance d, angle, number of levels L, quantisation range (patch min–max or
// fixed 0–255), symmetric counting, a disc mask (only pairs with both ends inside count).
// Auto-mounts on <div data-demo="glcm">. Options: data-preset="<id>".
import { htmlEl, slider, presets, checkbox, button, autoMount } from "../lib/ui";
import { Plot, PolarPlot, plotFigure } from "../lib/plot";
import { inferno } from "../lib/colour";
import { textureFeatures, offsetFor, Haralick } from "../lib/glcm";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { P, PATCHES, PatchId, PatchParams, makePatch, discMask } from "./model";

const NAME = "glcm";
const LEVELS = [2, 4, 8, 16, 32, 64];
const ANGLES = [0, 45, 90, 135];

/** Page-only selector (not inside reveal.js slides). */
const PG = `.qb-demo[data-demo="${NAME}"]:not(.reveal *)`;
const STYLE = `
[data-demo="${NAME}"] { gap: .7em; }
[data-demo="${NAME}"] .gl-main { flex: 1 1 520px; display: flex; flex-direction: column; gap: .3em; min-width: 0; }
[data-demo="${NAME}"] .gl-row { display: flex; gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .gl-row figure { margin: 0; flex: 1 1 240px; min-width: 0; max-width: 270px; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .1em; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .gl-side { flex: 0 1 260px; min-width: 210px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] table { border-collapse: collapse; font-size: 16px; width: 100%; }
[data-demo="${NAME}"] td, [data-demo="${NAME}"] th { padding: 1px 5px; text-align: right; border-bottom: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] td:first-child, [data-demo="${NAME}"] th:first-child { text-align: left; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 250px; gap: .3em; }
[data-demo="${NAME}"] .gl-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .gl-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] select { font: inherit; font-size: 15px; }
/* Pages: patches and matrices (two rows), features and polar plot, controls: three columns
   that fit a 1366x768 window (each figure at most half the window height minus captions). */
@media (min-width: 641px) {
  ${PG} { flex-wrap: nowrap; gap: .7em; }
  ${PG} .gl-main { flex: 0 1 calc(2 * min(270px, (100vh - 250px) / 2) + .6em); }
  ${PG} .gl-row figure { max-width: min(270px, calc((100vh - 250px) / 2)); }
  ${PG} figcaption { font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  ${PG} .gl-side { flex: 0 1 250px; min-width: 0; }
  ${PG} .qb-panel { flex: 1 1 280px; min-width: 0; max-width: 360px; gap: .25em; }
  ${PG} .qb-btn { padding: .15em .5em; }
  ${PG} table { font-size: 15px; }
}
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 15px; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal .qb-demo[data-demo="${NAME}"] .gl-row figure { max-width: 215px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State {
  a: PatchId; b: PatchId | "off";
  d: number; angle: number; L: number; minmax: boolean; symmetric: boolean; disc: number; log: boolean;
  period: number; stripeAngle: number; corr: number; bContrast: number; bBright: number;
}
const DEFAULT: State = { a: "stripes", b: "off", d: 1, angle: 0, L: 8, minmax: false, symmetric: true, disc: 0, log: false, period: 8, stripeAngle: 0, corr: 3, bContrast: 1, bBright: 0 };

const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "samehist", label: "same histogram, different texture", s: { a: "pairA", b: "pairB", d: 1, L: 8 } },
  { id: "angle", label: "stripes: contrast depends on angle", s: { a: "stripes", stripeAngle: 0, period: 8, d: 2, angle: 0 } },
  { id: "period", label: "period = d: contrast vanishes", s: { a: "stripes", stripeAngle: 0, period: 8, d: 8, angle: 0 } },
  { id: "scale", label: "fine vs coarse: contrast vs d", s: { a: "noise", b: "smooth", corr: 4, d: 1, L: 8 } },
  { id: "levels", label: "too many levels on a small patch", s: { a: "smooth", b: "smooth", corr: 2, disc: 6, L: 64, d: 1 } },
  { id: "minmax", label: "min–max rescaling hides a brightness difference", s: { a: "smooth", b: "smooth", corr: 3, bContrast: 0.5, bBright: -40, minmax: true, L: 8 } },
];

const FEATS: { key: keyof Haralick; label: string; dec: number }[] = [
  { key: "contrast", label: "contrast", dec: 2 },
  { key: "homogeneity", label: "homogeneity", dec: 3 },
  { key: "asm", label: "ASM", dec: 3 },
  { key: "energy", label: "energy", dec: 3 },
  { key: "entropy", label: "entropy (bits)", dec: 2 },
  { key: "correlation", label: "correlation", dec: 3 },
];

const fmt = (v: number, d: number) => (Number.isFinite(v) ? v.toFixed(d) : "undef.");

function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 28 : 18;

  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };
  let cellImg: Float32Array | null = null;

  const main = htmlEl("div", "gl-main", "", host);
  const mkRow = (tag: string) => {
    const row = htmlEl("div", "gl-row", "", main);
    const f1 = htmlEl("figure", "", "", row), c1 = htmlEl("figcaption", "", "patch " + tag, f1), cv = htmlEl("canvas", "", "", f1);
    cv.width = 256; cv.height = 256;
    const f2 = htmlEl("figure", "", "", row), c2 = htmlEl("figcaption", "", "co-occurrence matrix " + tag, f2);
    const pl = new Plot(f2, { width: 300, height: 300, x: [0, 8], y: [0, 8], font: FS, margin: { l: 1.6 * FS + 14, b: 1.6 * FS + 14, t: 6, r: 6 }, grid: false, equal: true, xfmt: v => String(v), yfmt: v => String(v) });
    return { row, cv, c1, c2, pl };
  };
  const A = mkRow("A"), B = mkRow("B");
  const credit = htmlEl("div", "gl-credit", CELL_CREDIT, main);
  const side = htmlEl("div", "gl-side", "", host);
  const tbl = htmlEl("div", "", "", side);
  const polFig = plotFigure(side, "contrast vs angle (at this d)", "qb-figure");
  const polar = new PolarPlot(polFig, { size: 260, rmax: 1, spokeStep: 45, font: onSlide ? 22 : 16 });
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "gl-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); run(); } })));
  const selRow = htmlEl("div", "", "", panel);
  const selA = htmlEl("select", "", "", htmlEl("label", "", "A: ", selRow));
  const selB = htmlEl("select", "", "", htmlEl("label", "", " B: ", selRow));
  for (const p of PATCHES) { htmlEl("option", "", p.label, selA).value = p.id; }
  htmlEl("option", "", "off", selB).value = "off";
  for (const p of PATCHES) { htmlEl("option", "", p.label, selB).value = p.id; }
  selA.addEventListener("change", () => { st.a = selA.value as PatchId; selA.blur(); unpreset(); run(); });
  selB.addEventListener("change", () => { st.b = selB.value as PatchId | "off"; selB.blur(); unpreset(); run(); });
  const dS = slider(panel, "distance d", 1, 16, 1, st.d, v => { st.d = v; unpreset(); run(); });
  const aS = slider(panel, "angle", 0, 3, 1, ANGLES.indexOf(st.angle), v => { st.angle = ANGLES[v]; unpreset(); run(); }, v => ANGLES[v] + "°");
  const lS = slider(panel, "levels L", 0, LEVELS.length - 1, 1, LEVELS.indexOf(st.L), v => { st.L = LEVELS[v]; unpreset(); run(); }, v => String(LEVELS[v]));
  const perS = slider(panel, "stripe period", 2, 32, 1, st.period, v => { st.period = v; unpreset(); run(); });
  const saS = slider(panel, "stripe angle", 0, 180, 15, st.stripeAngle, v => { st.stripeAngle = v; unpreset(); run(); }, v => v + "°");
  const cS = slider(panel, "noise σ", 1, 8, 0.5, st.corr, v => { st.corr = v; unpreset(); run(); });
  const bcS = slider(panel, "B contrast", 0.25, 1, 0.05, st.bContrast, v => { st.bContrast = v; unpreset(); run(); });
  const discS = slider(panel, "disc mask r", 0, 32, 1, st.disc, v => { st.disc = v; unpreset(); run(); }, v => (v ? String(v) : "off"));
  const mmC = checkbox(panel, "quantise each patch min–max (else 0–255)", st.minmax, v => { st.minmax = v; unpreset(); run(); });
  const symC = checkbox(panel, "symmetric (count both directions)", st.symmetric, v => { st.symmetric = v; unpreset(); run(); });
  const logC = checkbox(panel, "log colour scale", st.log, v => { st.log = v; run(); });
  button(panel, "reset", () => { Object.assign(st, DEFAULT); sync(); pb.forEach(b => b.classList.remove("active")); run(); });

  function unpreset(): void { pb.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    selA.value = st.a; selB.value = st.b;
    dS.set(st.d); aS.set(ANGLES.indexOf(st.angle)); lS.set(LEVELS.indexOf(st.L)); perS.set(st.period); saS.set(st.stripeAngle);
    cS.set(st.corr); bcS.set(st.bContrast); discS.set(st.disc); mmC.checked = st.minmax; symC.checked = st.symmetric; logC.checked = st.log;
  }

  function patchOf(id: PatchId, isB: boolean): Float64Array {
    const prm: PatchParams = { period: st.period, angle: st.stripeAngle, corr: st.corr, seed: isB && id === st.a ? 2 : 1 };
    if (isB) { prm.contrast = st.bContrast; prm.brightness = st.bBright; }
    let crop: Float64Array | undefined;
    if (id === "cell" && cellImg) {
      const [ox, oy] = isB ? [150, 170] : [90, 70];
      crop = Float64Array.from({ length: P * P }, (_, i) => cellImg![(oy + Math.floor(i / P)) * 256 + ox + (i % P)]);
    }
    return makePatch(id, prm, crop);
  }

  function analyse(img: Float64Array, angle = st.angle) {
    let lo = 0, hi = 256;
    const mask = st.disc ? discMask(st.disc) : undefined;
    if (st.minmax) {
      lo = Infinity; hi = -Infinity;
      for (let i = 0; i < img.length; i++) if (!mask || mask[i]) { lo = Math.min(lo, img[i]); hi = Math.max(hi, img[i]); }
      hi += 1e-9;
    }
    return textureFeatures(img, P, P, { L: st.L, lo, hi, d: st.d, angle, mask, symmetric: st.symmetric });
  }

  function drawPatch(cv: HTMLCanvasElement, img: Float64Array, q: Int32Array, off: [number, number]): void {
    const ctx = cv.getContext("2d")!;
    const id = ctx.createImageData(P, P);
    const mask = st.disc ? discMask(st.disc) : null;
    for (let i = 0; i < P * P; i++) {
      const g = st.L <= 64 ? Math.round(255 * (q[i] + 0.5) / st.L) : img[i];
      const dim = mask && !mask[i] ? 0.35 : 1;
      id.data[4 * i] = g * dim; id.data[4 * i + 1] = g * dim; id.data[4 * i + 2] = g * dim + (dim < 1 ? 40 : 0); id.data[4 * i + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = P; tmp.height = P;
    tmp.getContext("2d")!.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, 256, 256);
    const s = 4;
    const starts: [number, number][] = [[22, 26], [34, 34], [26, 40]];
    for (const [x, y] of starts) {
      const x2 = x + off[0], y2 = y + off[1];
      const ax = (x + 0.5) * s, ay = (y + 0.5) * s, bx = (x2 + 0.5) * s, by = (y2 + 0.5) * s;
      ctx.strokeStyle = "#ffd400"; ctx.fillStyle = "#ffd400"; ctx.lineWidth = 3;
      ctx.strokeRect(x * s, y * s, s, s); ctx.strokeRect(x2 * s, y2 * s, s, s);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      const len = Math.hypot(bx - ax, by - ay);
      if (len > 8) {
        const ux = (bx - ax) / len, uy = (by - ay) / len;
        ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - 9 * ux - 5 * uy, by - 9 * uy + 5 * ux); ctx.lineTo(bx - 9 * ux + 5 * uy, by - 9 * uy - 5 * ux); ctx.fill();
      }
    }
  }

  function drawGlcm(pl: Plot, p: Float64Array, L: number): void {
    pl.clear();
    pl.opts.xticks = L <= 8 ? Array.from({ length: L }, (_, k) => k + 0.5) : [0.5, L / 2 - 0.5, L - 0.5];
    pl.opts.yticks = pl.opts.xticks;
    pl.opts.xfmt = v => String(Math.floor(v)); pl.opts.yfmt = v => String(L - 1 - Math.floor(v));
    pl.setDomain([0, L], [0, L]);
    const vals = st.log ? Array.from(p, v => (v > 0 ? Math.log10(v) : NaN)) : p;
    let mx = 0;
    for (const v of p) mx = Math.max(mx, v);
    pl.heatmap(vals, L, L, { map: inferno, lo: st.log ? Math.log10(mx) - 4 : 0, hi: st.log ? Math.log10(mx) : mx, numbers: L <= 4, fmt: v => (st.log ? (10 ** v).toFixed(2) : v.toFixed(2)) });
    pl.line([0, L], [L, 0], { color: "#7fd3ff", dash: "5 5", width: 2 });
  }

  function run(): void {
    if (st.a === "cell" || st.b === "cell") {
      if (!cellImg) { loadCell().then(g => { cellImg = g.data; run(); }); }
    }
    credit.style.display = st.a === "cell" || st.b === "cell" ? "" : "none";
    const imgA = patchOf(st.a, false);
    const rA = analyse(imgA);
    drawPatch(A.cv, imgA, rA.q, rA.offset);
    drawGlcm(A.pl, rA.p, st.L);
    A.c1.textContent = `patch A: ${PATCHES.find(p => p.id === st.a)!.label}`;
    A.c2.textContent = `A: co-occurrence, offset (${rA.offset[0]}, ${rA.offset[1]})`;
    let rB: ReturnType<typeof analyse> | null = null;
    B.row.style.display = st.b === "off" ? "none" : "";
    if (st.b !== "off") {
      const imgB = patchOf(st.b, true);
      rB = analyse(imgB);
      drawPatch(B.cv, imgB, rB.q, rB.offset);
      drawGlcm(B.pl, rB.p, st.L);
      B.c1.textContent = `patch B: ${PATCHES.find(p => p.id === st.b)!.label}`;
      B.c2.textContent = `B: co-occurrence (rows: level at x)`;
    }
    // table
    let html = `<table><tr><th>feature</th><th>A</th>${rB ? "<th>B</th>" : ""}</tr>`;
    for (const f of FEATS) html += `<tr><td>${f.label}</td><td>${fmt(rA.f[f.key], f.dec)}</td>${rB ? `<td>${fmt(rB.f[f.key], f.dec)}</td>` : ""}</tr>`;
    tbl.innerHTML = html + "</table>";
    // polar: contrast vs angle (0..345 in steps of 15, offsets rounded)
    const th: number[] = [], cA: number[] = [], cB: number[] = [];
    for (let a = 0; a <= 360; a += 15) {
      th.push(a * Math.PI / 180);
      cA.push(analyse(imgA, a).f.contrast);
      if (rB) cB.push(analyse(patchOf(st.b as PatchId, true), a).f.contrast);
    }
    const rmax = Math.max(1e-9, ...cA, ...cB);
    polar.clear();
    polar.setRmax(rmax * 1.05);
    polar.line(th, cA, { color: "var(--qb-c1)", width: 3 });
    if (rB) polar.line(th, cB, { color: "var(--qb-c2)", width: 3, dash: "6 4" });
    polar.ray(st.angle * Math.PI / 180, rA.f.contrast, { color: "var(--qb-c4)", width: 3 });
    const off = offsetFor(st.d, st.angle);
    let nz = 0;
    for (const v of rA.p) if (v > 0) nz++;
    ro.textContent = `offset (dx, dy) = (${off[0]}, ${off[1]}) px\nA: ${rA.pairs} pairs counted, ${nz} of ${st.L * st.L} cells non-zero`
      + (rB ? "\nblue: A, red dashed: B (polar plot)" : "");
  }

  sync();
  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  run();
}

autoMount(NAME, mount);
