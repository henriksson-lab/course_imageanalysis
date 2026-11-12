// Resolution, sampling and measuring the PSF. Three modes (data-mode):
//  - "twopoints" (default): two point sources (or a sinusoidal grating) imaged by an ideal
//    lens and recorded on a pixel grid with photon noise. Separation/period, NA, λ, pixel
//    size (Nyquist λ/4NA marked), photons. Panels: the pixel image and the line profile
//    (continuous and per pixel), dip in %, Rayleigh and Sparrow distances; aliasing of the
//    grating when the pixels are larger than half its period.
//  - "axial": xz slice of the 3D PSF (widefield, or confocal = squared), the same slice
//    sampled with a chosen z-step, lateral and axial profiles with FWHMs.
//  - "beads": a simulated bead slide (30 beads, seeded), and the PSF measured by
//    background subtraction, sub-pixel centring and averaging of 1–30 beads, against the
//    true one.
// Auto-mounts on <div data-demo="resolve">. Options: data-mode, data-preset (ids below),
// data-presets="id,id,..." limits the preset buttons.
import { htmlEl, presets as presetRow, button, checkbox, autoMount, slider, Slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { inferno, grey } from "../lib/colour";
import { injectStyle, baseStyle, select, imagePanel, legendRow, logSlider, fmtSci } from "../deconvlab/widgets";
import * as M from "./model";

const NAME = "resolve";
const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .rs-left { flex: 1 1 400px; max-width: 460px; display: flex; flex-direction: column; gap: .3em; min-width: 0; }
[data-demo="${NAME}"] .rs-mid { flex: 1 1 330px; max-width: 420px; display: flex; flex-direction: column; gap: .3em; min-width: 0; }
[data-demo="${NAME}"] .rs-pix canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 2px; }
[data-demo="${NAME}"] .rs-xz { display: grid; grid-template-columns: 1fr 1fr; gap: .5em; }
[data-demo="${NAME}"] .rs-xz canvas { width: 100%; aspect-ratio: 1 / 2; display: block; image-rendering: pixelated; }
[data-demo="${NAME}"] .rs-beads { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .4em .5em; }
[data-demo="${NAME}"] .rs-beads .rs-field { grid-column: 1 / 4; max-width: 300px; justify-self: center; width: 100%; }
[data-demo="${NAME}"] .rs-beads .rs-field canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; }
.reveal [data-demo="${NAME}"] .rs-left { flex: 0 1 440px; max-width: 440px; }
.reveal [data-demo="${NAME}"] .rs-mid { flex: 0 1 400px; max-width: 400px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 1 280px; }
.reveal [data-demo="${NAME}"] .rs-xz { max-width: 280px; }
.reveal [data-demo="${NAME}"] .rs-beads .rs-field { grid-column: 1 / 4; max-width: 290px; justify-self: center; width: 100%; }
`;

function showSlider(s: Slider, on: boolean): void { s.input.closest("label")!.style.display = on ? "" : "none"; }
function setLabel(s: Slider, text: string): void { (s.input.closest("label")!.querySelector(".qb-slider-name") as HTMLElement).textContent = text; }

interface Preset<S> { id: string; label: string; s: Partial<S> }
function setup<S extends object>(host: HTMLElement, panel: HTMLElement, list: Preset<S>[], dflt: S, st: S, apply: () => void): { load(p: Preset<S>): void; p0: Preset<S> } {
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? list.filter(p => only.includes(p.id)) : list;
  const p0 = list.find(p => p.id === host.dataset.preset) ?? list[0];
  htmlEl("div", "dc-lab", "presets", panel);
  const btns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  function load(p: Preset<S>): void {
    Object.assign(st, dflt, p.s);
    btns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    apply();
  }
  return { load, p0 };
}

function drawCanvas(c: HTMLCanvasElement, v: ArrayLike<number>, w: number, h: number, lo: number, hi: number, map = grey): void {
  const id = new ImageData(w, h), n = 255;
  for (let k = 0; k < w * h; k++) {
    const t = hi > lo ? Math.min(1, Math.max(0, (v[k] - lo) / (hi - lo))) : 0;
    const [r, g, b] = map(t);
    id.data[4 * k] = Math.round(r * n); id.data[4 * k + 1] = Math.round(g * n); id.data[4 * k + 2] = Math.round(b * n); id.data[4 * k + 3] = 255;
  }
  c.width = w; c.height = h;
  c.getContext("2d")!.putImageData(id, 0, 0);
}

// ---------------------------------------------------------------------------------------
// Mode "twopoints"

interface TP { object: "points" | "grating"; d: number; NA: number; wavelength: number; pixel: number; photons: number; noise: boolean }
const TP_DEFAULT: TP = { object: "points", d: 227, NA: 1.4, wavelength: 520, pixel: 40, photons: 10000, noise: false };
const TP_PRESETS: Preset<TP>[] = [
  { id: "rayleigh", label: "Rayleigh", s: { d: 227 } },
  { id: "sparrow", label: "Sparrow", s: { d: 176 } },
  { id: "photons", label: "Rayleigh, 20 photons", s: { d: 227, pixel: 65, photons: 20, noise: true } },
  { id: "nyquist", label: "Nyquist sampled", s: { d: 227, pixel: 93 } },
  { id: "aliasing", label: "undersampled: aliasing", s: { object: "grating", d: 300, pixel: 200, photons: 10000 } },
  { id: "oversampled", label: "oversampled: noisier pixels", s: { d: 300, pixel: 20, photons: 500, noise: true } },
];
const FIELD_W = 2000, FIELD_H = 1000;

function mountTwoPoints(host: HTMLElement): void {
  const st: TP = { ...TP_DEFAULT };
  const left = htmlEl("div", "rs-left", "", host);
  const fig = htmlEl("figure", "rs-pix", "", left);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  const plot = new Plot(left, { width: 460, height: 250, x: [-FIELD_W / 2, FIELD_W / 2], y: [0, 1], xlabel: "position along the line (nm)", ylabel: "photons / pixel", font: 23, margin: { l: 98 } });
  const lCont = plot.line([], [], { color: seriesColor(0), width: 3 });
  const lPix = plot.step([], [], { color: seriesColor(1), width: 2.5 }, "mid");
  const vA = plot.vline(0, { color: "var(--qb-muted, #777)", width: 1.5 }), vB = plot.vline(0, { color: "var(--qb-muted, #777)", width: 1.5 });
  legendRow(left, [{ label: "optical image (no pixels, no noise)", color: seriesColor(0) }, { label: "recorded pixels", color: seriesColor(1) }]);

  const panel = htmlEl("div", "qb-panel", "", host);
  const { load, p0 } = setup(host, panel, TP_PRESETS, TP_DEFAULT, st, () => { sync(); update(); });
  const objS = select<TP["object"]>(panel, "object", [["points", "two points"], ["grating", "grating (sinusoidal)"]], st.object, v => { st.object = v; if (v === "grating" && st.d < 120) st.d = 250; sync(); update(); });
  const dS = slider(panel, "distance d (nm)", 50, 1000, 1, st.d, v => { st.d = v; update(); }, v => v.toFixed(0));
  const naS = slider(panel, "NA", 0.3, 1.45, 0.05, st.NA, v => { st.NA = v; update(); }, v => v.toFixed(2));
  const lamS = slider(panel, "λ (nm)", 400, 700, 10, st.wavelength, v => { st.wavelength = v; update(); }, v => v.toFixed(0));
  const pxS = slider(panel, "pixel (nm)", 10, 400, 1, st.pixel, v => { st.pixel = v; update(); }, v => v.toFixed(0));
  const phS = logSlider(panel, "photons", 10, 1e5, st.photons, v => { st.photons = Math.round(v); update(); }, v => fmtSci(Math.round(v)));
  const nzC = checkbox(panel, "photon noise (seeded)", st.noise, v => { st.noise = v; update(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);

  function sync(): void {
    objS.set(st.object); dS.set(st.d); naS.set(st.NA); lamS.set(st.wavelength); pxS.set(st.pixel); phS.setValue(st.photons); nzC.checked = st.noise;
    setLabel(dS, st.object === "points" ? "distance d (nm)" : "period p (nm)");
    setLabel(phS, st.object === "points" ? "photons / point" : "photons / µm²");
  }
  let pending = false;
  function update(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }
  function draw(): void {
    const o = { NA: st.NA, wavelength: st.wavelength }, pts = st.object === "points";
    const grat = M.gratingDensity(st.d, o, st.photons);
    const dens = pts ? M.twoPointDensity(st.d, o, st.photons) : (x: number, _y: number) => grat(x);
    const s = M.samplePixels(dens, FIELD_W, FIELD_H, st.pixel, st.noise);
    let hi = 0, lo = Infinity;
    for (const v of s.counts) { hi = Math.max(hi, v); lo = Math.min(lo, v); }
    drawCanvas(cv, s.counts, s.nx, s.ny, pts ? 0 : lo, hi || 1);   // the grating: contrast stretched
    cv.style.aspectRatio = `${s.nx} / ${s.ny}`;
    cap.textContent = `${s.nx} × ${s.ny} pixels of ${st.pixel} nm (field ${FIELD_W / 1000} × ${FIELD_H / 1000} µm${pts ? "" : ", stretched"})`;
    // profile along y = 0
    const xs = Float64Array.from({ length: 601 }, (_, i) => -FIELD_W / 2 + i * FIELD_W / 600);
    const a2 = st.pixel * st.pixel;
    const cont = xs.map(x => dens(x, 0) * a2);
    const row = (s.ny - 1) / 2, pix = s.counts.subarray(row * s.nx, (row + 1) * s.nx);
    lCont.set(xs, cont); lPix.set(s.xs, pix);
    const ymax = Math.max(...cont, ...pix) * 1.12 || 1;
    plot.setDomain(undefined, [0, ymax]);
    vA.set(pts ? -st.d / 2 : -1e9); vB.set(pts ? st.d / 2 : -1e9);
    // readouts
    const lna = st.wavelength / st.NA, nyq = st.wavelength / (4 * st.NA);
    const lines: string[] = [];
    if (pts) {
      const dp = M.dip(st.d, o);
      const mid = pix[(s.nx - 1) / 2], iA = Math.round(-st.d / 2 / st.pixel + (s.nx - 1) / 2), iB = Math.round(st.d / 2 / st.pixel + (s.nx - 1) / 2);
      const pk = (pix[iA] + pix[iB]) / 2, pdip = pk > 0 ? 1 - mid / pk : 0;
      lines.push(`d = ${st.d.toFixed(0)} nm = ${(st.d / lna).toFixed(2)} λ/NA`);
      lines.push(`Rayleigh 0.61λ/NA = ${(0.61 * lna).toFixed(0)} nm`);
      lines.push(`Sparrow 0.47λ/NA = ${(0.47 * lna).toFixed(0)} nm`);
      lines.push(`dip, optical image: ${(100 * dp).toFixed(1)} %`);
      lines.push(`dip in the pixels: ${(100 * pdip).toFixed(1)} %`);
      lines.push(`peak pixel ≈ ${pk.toFixed(pk < 100 ? 1 : 0)} photons` + (st.noise && pk > 0 ? ` (noise ±${Math.sqrt(pk).toFixed(1)} = ${(100 / Math.sqrt(pk)).toFixed(0)} %)` : ""));
    } else {
      const m = M.gratingContrast(st.d, o), ap = M.apparentPeriod(st.d, st.pixel);
      lines.push(`period p = ${st.d.toFixed(0)} nm (${(1000 / st.d).toFixed(2)} cycles/µm)`);
      lines.push(`cut-off 2NA/λ = ${(2000 * st.NA / st.wavelength).toFixed(2)} cycles/µm`);
      lines.push(`contrast after the optics: ${(100 * m).toFixed(1)} %`);
      lines.push(st.pixel <= st.d / 2 ? `pixels sample it: ${(st.d / st.pixel).toFixed(1)} pixels per period` : `ALIASING: pixels > p/2; looks like period ${Number.isFinite(ap) ? ap.toFixed(0) + " nm" : "∞ (flat)"}`);
    }
    lines.push(`Nyquist pixel λ/4NA = ${nyq.toFixed(0)} nm; this pixel: ${st.pixel <= nyq ? "fine" : `${(st.pixel / nyq).toFixed(1)}× too large`}`);
    ro.textContent = lines.join("\n");
  }
  sync();
  load(p0);
}

// ---------------------------------------------------------------------------------------
// Mode "axial"

interface AX { NA: number; wavelength: number; zstep: number; confocal: boolean; log: boolean }
const AX_DEFAULT: AX = { NA: 1.4, wavelength: 520, zstep: 200, confocal: false, log: false };
const AX_PRESETS: Preset<AX>[] = [
  { id: "na14", label: "NA 1.4 (oil)", s: {} },
  { id: "na05", label: "NA 0.5 (air)", s: { NA: 0.5, zstep: 1000 } },
  { id: "coarse", label: "z-step too large", s: { zstep: 1000 } },
  { id: "cone", label: "out-of-focus light (log)", s: { log: true } },
  { id: "confocal", label: "confocal", s: { confocal: true, log: true } },
];

function mountAxial(host: HTMLElement): void {
  const st: AX = { ...AX_DEFAULT };
  const left = htmlEl("div", "rs-left", "", host);
  const xz = htmlEl("div", "rs-xz", "", left);
  const pA = imagePanel(xz, "xz slice"), pB = imagePanel(xz, "sampled every Δz");
  const capAx = htmlEl("div", "qb-note", "", left);
  const mid = htmlEl("div", "rs-mid", "", host);
  const plot = new Plot(mid, { width: 420, height: 290, x: [-3, 3], y: [0, 1.05], xlabel: "distance from the focus (µm)", ylabel: "intensity", font: 23, margin: { l: 98 } });
  const lLat = plot.line([], [], { color: seriesColor(0), width: 3 });
  const lAx = plot.line([], [], { color: seriesColor(1), width: 3 });
  const dots = plot.scatter([], [], { color: seriesColor(1), r: 6 });
  legendRow(mid, [{ label: "across (x)", color: seriesColor(0) }, { label: "along the axis (z)", color: seriesColor(1) }, { label: "dots: planes every Δz", color: seriesColor(1), dash: true }]);
  const ro = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  const { load, p0 } = setup(host, panel, AX_PRESETS, AX_DEFAULT, st, () => { sync(); recompute(); });
  const naS = slider(panel, "NA", 0.3, 1.45, 0.05, st.NA, v => { st.NA = v; recompute(); }, v => v.toFixed(2));
  const lamS = slider(panel, "λ (nm)", 400, 700, 10, st.wavelength, v => { st.wavelength = v; recompute(); }, v => v.toFixed(0));
  const zS = slider(panel, "z-step (nm)", 50, 1500, 10, st.zstep, v => { st.zstep = v; draw(); }, v => v.toFixed(0));
  const modeS = select<"wf" | "cf">(panel, "microscope", [["wf", "widefield"], ["cf", "confocal (point pinhole)"]], "wf", v => { st.confocal = v === "cf"; recompute(); });
  const logC = checkbox(panel, "log scale (see the faint cone)", st.log, v => { st.log = v; draw(); });
  button(panel, "reset", () => load(p0));
  htmlEl("div", "qb-note", "Scalar model of an ideal lens; oil (n = 1.518) for NA ≥ 1, air below. Confocal: excitation × detection PSF, same wavelength.", panel);

  function sync(): void { naS.set(st.NA); lamS.set(st.wavelength); zS.set(st.zstep); modeS.set(st.confocal ? "cf" : "wf"); logC.checked = st.log; }
  let psf: M.AxialPSF | null = null, opt: M.AxialOptions = { NA: 1.4, wavelength: 520, ri: 1.518 };
  let pending = false;
  function recompute(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      opt = { NA: st.NA, wavelength: st.wavelength, ri: st.NA >= 1 ? 1.518 : 1.0 };
      const Z = Math.max(3000, 0.75 * M.axialResolution(opt));
      psf = M.axialPSF(opt, st.confocal, Z / 2, Z);
      draw();
    });
  }
  function draw(): void {
    if (!psf) return;
    const p = psf, show = (v: number) => (st.log ? Math.log10(Math.max(v, 1e-4)) : v);
    const lo = st.log ? -4 : 0, hi = st.log ? 0 : 1;
    pA.draw(p.img.map(show), p.nx, p.nz, { lo, hi, map: inferno });
    // nearest z plane k·Δz for every row
    const dz = 2 * p.Z / (p.nz - 1), samp = new Float64Array(p.nx * p.nz);
    for (let r = 0; r < p.nz; r++) {
      const z = (p.nz - 1 - r - (p.nz - 1) / 2) * dz, zk = Math.round(z / st.zstep) * st.zstep;
      const rr = Math.round(p.nz - 1 - (zk / dz + (p.nz - 1) / 2));
      for (let i = 0; i < p.nx; i++) samp[r * p.nx + i] = rr >= 0 && rr < p.nz ? show(p.img[rr * p.nx + i]) : lo;
    }
    pB.draw(samp, p.nx, p.nz, { lo, hi, map: inferno });
    pA.cap.textContent = st.confocal ? "confocal xz" : "widefield xz";
    pB.cap.textContent = `planes every ${st.zstep} nm`;
    capAx.textContent = `Each panel: ${(2 * p.X / 1000).toFixed(1)} µm wide (x), ${(2 * p.Z / 1000).toFixed(1)} µm tall (z, optical axis vertical).`;
    const Zum = p.Z / 1000;
    plot.setDomain([-Zum, Zum], [0, 1.05]);
    lLat.set(p.xs.map(x => x / 1000), p.lateral);
    lAx.set(p.zs.map(z => z / 1000), p.axial);
    const zk: number[] = [], vk: number[] = [];
    for (let k = -Math.floor(p.Z / st.zstep); k <= Math.floor(p.Z / st.zstep); k++) {
      const z = k * st.zstep, j = Math.round(z / dz + (p.nz - 1) / 2);
      zk.push(z / 1000); vk.push(p.axial[j]);
    }
    dots.set(zk, vk);
    const nyq = M.axialNyquist(opt), zero = st.wavelength / (opt.ri * (1 - Math.sqrt(1 - (st.NA / opt.ri) ** 2)));
    ro.textContent = [
      `lateral FWHM ${p.fwhmLat.toFixed(0)} nm   axial FWHM ${p.fwhmAx.toFixed(0)} nm`,
      `axial / lateral = ${(p.fwhmAx / p.fwhmLat).toFixed(1)}`,
      `first axial minimum ≈ λ/n(1−cos α) = ${zero.toFixed(0)} nm`,
      `  (small-NA rule 2nλ/NA² = ${M.axialResolution(opt).toFixed(0)} nm)`,
      `axial Nyquist step λ/2n(1−cos α) = ${nyq.toFixed(0)} nm`,
      `z-step ${st.zstep} nm: ${st.zstep <= nyq ? "fine" : `${(st.zstep / nyq).toFixed(1)}× too large`}`,
    ].join("\n");
  }
  sync();
  load(p0);
}

// ---------------------------------------------------------------------------------------
// Mode "beads"

interface BD { count: number; diameter: number; photons: number; background: number; subtract: boolean }
const BD_DEFAULT: BD = { count: 1, diameter: 40, photons: 500, background: 10, subtract: true };
const BD_PRESETS: Preset<BD>[] = [
  { id: "one", label: "one bead, noisy", s: {} },
  { id: "thirty", label: "30 beads", s: { count: 30 } },
  { id: "large", label: "beads too large", s: { count: 30, diameter: 200, photons: 2000 } },
  { id: "background", label: "forgot the background", s: { count: 30, subtract: false } },
];

function mountBeads(host: HTMLElement): void {
  const st: BD = { ...BD_DEFAULT };
  const left = htmlEl("div", "rs-left", "", host);
  const grid = htmlEl("div", "rs-beads", "", left);
  const pField = imagePanel(grid, "bead slide (10.4 µm, 65 nm pixels)", "rs-field");
  const pRaw = imagePanel(grid, "one bead"), pAvg = imagePanel(grid, "averaged"), pTrue = imagePanel(grid, "true PSF");
  const mid = htmlEl("div", "rs-mid", "", host);
  const R = M.PATCH / 2 * M.BEAD_PIXEL;
  const plot = new Plot(mid, { width: 420, height: 290, x: [-R, R], y: [-0.05, 1.1], xticks: [-400, 0, 400], xlabel: "x (nm)", ylabel: "PSF / true peak", font: 23, margin: { l: 98 } });
  const lTrue = plot.line([], [], { color: "var(--qb-muted, #777)", width: 3, dash: "7 5" });
  const lAvg = plot.line([], [], { color: seriesColor(0), width: 3 });
  legendRow(mid, [{ label: "measured (averaged beads)", color: seriesColor(0) }, { label: "true PSF", color: "var(--qb-muted, #777)", dash: true }]);
  const ro = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  const { load, p0 } = setup(host, panel, BD_PRESETS, BD_DEFAULT, st, () => { sync(); rebuild(); });
  const nS = slider(panel, "beads averaged", 1, 30, 1, st.count, v => { st.count = v; measure(); }, v => v.toFixed(0));
  const dS = slider(panel, "bead diameter (nm)", 0, 400, 10, st.diameter, v => { st.diameter = v; rebuild(); }, v => v.toFixed(0));
  const phS = logSlider(panel, "photons / bead", 100, 1e5, st.photons, v => { st.photons = Math.round(v); refield(); }, v => fmtSci(Math.round(v)));
  const bgS = slider(panel, "background / px", 0, 100, 1, st.background, v => { st.background = v; refield(); }, v => v.toFixed(0));
  const subC = checkbox(panel, "subtract the background", st.subtract, v => { st.subtract = v; measure(); });
  button(panel, "reset", () => load(p0));
  htmlEl("div", "qb-note", "NA 1.4, λ 520 nm, 100× objective with 6.5 µm camera pixels (65 nm). Simulated, seeded photon noise.", panel);

  function sync(): void { nS.set(st.count); dS.set(st.diameter); phS.setValue(st.photons); bgS.set(st.background); subC.checked = st.subtract; }
  const beads = M.beadPositions(), truth = M.truePSF();
  let fine = M.beadImageFine(st.diameter), field = M.beadField(fine, beads, st.photons, st.background);
  let pend = 0;
  const later = (lvl: number) => {
    if (pend) { pend = Math.max(pend, lvl); return; }
    pend = lvl;
    requestAnimationFrame(() => {
      const l = pend; pend = 0;
      if (l >= 3) fine = M.beadImageFine(st.diameter);
      if (l >= 2) field = M.beadField(fine, beads, st.photons, st.background);
      draw();
    });
  };
  const rebuild = () => later(3), refield = () => later(2), measure = () => later(1);

  function draw(): void {
    const c = field.counts;
    let hi = 0;
    for (const v of c) hi = Math.max(hi, v);
    drawCanvas(pField.canvas, c, M.FIELD, M.FIELD, 0, hi, grey);
    const ctx = pField.canvas.getContext("2d")!;
    ctx.strokeStyle = "#ff7f0e"; ctx.lineWidth = 1;
    for (const b of beads.slice(0, st.count)) ctx.strokeRect(b.px - 6.5, b.py - 6.5, 13, 13);
    const raw = M.patchAt(c, beads[0].px, beads[0].py);
    pRaw.draw(raw, M.PATCH, M.PATCH, { map: inferno });
    const avg = M.averageBeads(c, beads, st.count, st.subtract);
    const m = M.PATCH * M.UP, k = (M.PATCH / 2) * M.UP, tpk = truth.up[k * m + k];
    pAvg.draw(avg.up, m, m, { lo: 0, hi: Math.max(...avg.up), map: inferno });
    pTrue.draw(truth.up, m, m, { lo: 0, hi: tpk, map: inferno });
    pAvg.cap.textContent = `averaged (${st.count})`;
    const xs = Float64Array.from({ length: m }, (_, i) => (i - k) * M.BEAD_PIXEL / M.UP);
    lTrue.set(xs, Float64Array.from({ length: m }, (_, i) => truth.up[k * m + i] / tpk));
    lAvg.set(xs, Float64Array.from({ length: m }, (_, i) => avg.up[k * m + i] / tpk));
    const err = 100 * (avg.fwhm / truth.fwhm - 1);
    ro.textContent = [
      `measured FWHM ${avg.fwhm.toFixed(0)} nm (${err >= 0 ? "+" : ""}${err.toFixed(1)} %)`,
      `true FWHM ${truth.fwhm.toFixed(0)} nm (point, on these pixels)`,
      `bead ${st.diameter} nm, ${st.photons} photons, background ${st.background}/px`,
      `peak pixel of one bead ≈ ${Math.max(...raw).toFixed(0)} photons`,
    ].join("\n");
  }
  sync();
  load(p0);
}

autoMount(NAME, host => {
  injectStyle(NAME, STYLE);
  const mode = host.dataset.mode ?? "twopoints";
  if (mode === "axial") mountAxial(host);
  else if (mode === "beads") mountBeads(host);
  else mountTwoPoints(host);
});
