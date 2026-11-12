// From emission spectra and filters to the mixing matrix. Top: emission spectra of 2–4
// dyes (parametric split Gaussians with a red tail) on 400–750 nm, each drawn in the colour
// of its peak wavelength; drag a dye's peak handle sideways. Shaded rectangles are the
// channels' filter bands (drag the band to move it, its edges to resize it), or, with
// "spectral detector", 8–32 contiguous bins. Right: the mixing matrix M (entry = fraction of
// the dye's photons collected by the channel), with each column scaled to its own channel
// (the spillover matrix of the crosstalk unit) as a heat map with numbers; bleed-through
// percentages, det (square) and the condition number κ, and the photons each dye delivers
// to its own channel.
// Auto-mounts on <div data-demo="spectramix">. Options: data-preset (classic|separated|
// narrow|close|detector), data-compact="1" (slides: shorter panel).
import { htmlEl, slider, presets, checkbox, button, draggable, autoMount, svgEl } from "../lib/ui";
import { Plot, heatmapGrid } from "../lib/plot";
import { mapFromAnchors } from "../lib/colour";
import * as U from "../lib/unmix";
import { det } from "../lib/linalg";
import { injectStyle, uniqueClip, wlCss, fmt, COMMON_CSS } from "../lsunmix/common";

const NAME = "spectramix";
const CSS = COMMON_CSS + `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .sm-left { flex: 1 1 560px; min-width: 300px; max-width: 700px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .35em; }
[data-demo="${NAME}"] .sm-bottom { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; margin-top: .3em; }
[data-demo="${NAME}"] .sm-mat { flex: 0 1 300px; max-width: 460px; }
[data-demo="${NAME}"] .sm-bottom .qb-readout { flex: 1 1 260px; }
[data-demo="${NAME}"] svg .sm-band { cursor: grab; }
[data-demo="${NAME}"] svg .sm-edge { cursor: ew-resize; }
.reveal [data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .sm-left { flex: 0 0 640px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 460px; max-width: 460px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .sm-bottom .qb-readout { font-size: 16px; }
`;

interface State { dyes: U.Dye[]; bands: U.Band[]; detector: boolean; bins: number; sel: number }

const DYE_NAMES = ["dye 1", "dye 2", "dye 3", "dye 4"];
const dye = (i: number, peak: number, width = 14, tail = 2): U.Dye => ({ name: DYE_NAMES[i], peak, width, tail });

const PRESETS: Record<string, { label: string; s: () => State }> = {
  classic: { label: "green into orange: the classic tail", s: () => ({ dyes: [dye(0, 520, 13, 2.6), dye(1, 578, 11, 2.2)], bands: [{ lo: 510, hi: 550 }, { lo: 565, hi: 605 }], detector: false, bins: 8, sel: 0 }) },
  separated: { label: "well separated", s: () => ({ dyes: [dye(0, 460, 12, 2), dye(1, 650, 12, 1.6)], bands: [{ lo: 445, hi: 490 }, { lo: 640, hi: 690 }], detector: false, bins: 8, sel: 0 }) },
  narrow: { label: "narrow filters: less bleed-through, fewer photons", s: () => ({ dyes: [dye(0, 520, 13, 2.6), dye(1, 578, 11, 2.2)], bands: [{ lo: 515, hi: 527 }, { lo: 580, hi: 592 }], detector: false, bins: 8, sel: 0 }) },
  close: { label: "two dyes 20 nm apart", s: () => ({ dyes: [dye(0, 520, 13, 2.6), dye(1, 540, 13, 2.6)], bands: [{ lo: 505, hi: 525 }, { lo: 540, hi: 570 }], detector: false, bins: 8, sel: 0 }) },
  detector: { label: "8-bin spectral detector", s: () => ({ dyes: [dye(0, 510, 12, 2.2), dye(1, 560, 12, 2.2), dye(2, 610, 12, 1.8), dye(3, 665, 12, 1.5)], bands: [{ lo: 500, hi: 540 }, { lo: 550, hi: 590 }, { lo: 600, hi: 640 }, { lo: 655, hi: 700 }], detector: true, bins: 8, sel: 0 }) },
};
const BAND_DEFAULT = [{ lo: 500, hi: 540 }, { lo: 560, hi: 600 }, { lo: 620, hi: 660 }, { lo: 670, hi: 710 }, { lo: 440, hi: 470 }, { lo: 715, hi: 745 }, { lo: 405, hi: 435 }, { lo: 600, hi: 615 }];
const DYE_DEFAULT = [520, 578, 630, 680];
const heat = mapFromAnchors(["#ffffff", "#fde0c5", "#f59e72", "#d6604d", "#8c1d2c"]);

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const pre = PRESETS[host.dataset.preset ?? ""] ? host.dataset.preset! : "classic";
  let S: State = PRESETS[pre].s();

  // ---- spectra plot ----
  const left = htmlEl("div", "sm-left", "", host);
  const plot = new Plot(left, { width: 680, height: 300, x: [U.L_MIN, U.L_MAX], y: [0, 1.25], xlabel: "wavelength (nm)", ylabel: "emission", xticks: [400, 450, 500, 550, 600, 650, 700, 750], yticks: [0, 0.5, 1], ariaLabel: "Emission spectra of the dyes and the filter bands of the channels" });
  uniqueClip(plot.svg, NAME);
  const bandLayer = svgEl("g", {}, plot.layer), curveLayer = svgEl("g", {}, plot.layer), handleLayer = svgEl("g", {}, plot.over);
  plot.layer.insertBefore(bandLayer, plot.layer.firstChild);
  const note = htmlEl("div", "qb-note", "Drag a dye's peak (circle) sideways; drag a filter band to move it, its edges to resize it.", left);
  void note;

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtn = presets(panel, [
    ...Object.entries(PRESETS).map(([id, p]) => ({ label: p.label, apply: () => { S = p.s(); syncControls(); rebuildHandles(); draw(); void id; } })),
    { label: "reset", apply: () => { S = PRESETS.classic.s(); syncControls(); rebuildHandles(); draw(); } },
  ]);
  pbtn[Object.keys(PRESETS).indexOf(pre)]?.classList.add("active");
  const nDyeS = slider(panel, "dyes", 2, 4, 1, S.dyes.length, v => { setDyes(v); manual(); });
  const nChS = slider(panel, "channels", 2, 8, 1, S.bands.length, v => { setBands(v); manual(); });
  const detC = checkbox(panel, "spectral detector (contiguous bins)", S.detector, v => { S.detector = v; rebuildHandles(); manual(); });
  const binS = slider(panel, "bins", 8, 32, 1, S.bins, v => { S.bins = v; manual(); });
  const selRow = htmlEl("div", "um-row", "", panel);
  htmlEl("span", "um-lab", "dye: ", selRow);
  const selBtns: HTMLButtonElement[] = [0, 1, 2, 3].map(i => button(selRow, String(i + 1), () => { S.sel = i; syncControls(); }));
  const wS = slider(panel, "width (nm)", 6, 40, 1, S.dyes[0].width, v => { S.dyes[S.sel].width = v; manual(); });
  const tS = slider(panel, "red tail", 1, 4, 0.1, S.dyes[0].tail, v => { S.dyes[S.sel].tail = v; manual(); });
  const bottom = htmlEl("div", "sm-bottom", "", left);
  const matBox = htmlEl("div", "sm-mat", "", bottom);
  const read = htmlEl("div", "qb-readout um-readout", "", bottom);

  function manual(): void { pbtn.forEach(b => b.classList.remove("active")); draw(); }
  function setDyes(k: number): void {
    while (S.dyes.length < k) S.dyes.push(dye(S.dyes.length, DYE_DEFAULT[S.dyes.length]));
    S.dyes.length = k; S.sel = Math.min(S.sel, k - 1); syncControls(); rebuildHandles();
  }
  function setBands(n: number): void {
    while (S.bands.length < n) S.bands.push({ ...BAND_DEFAULT[S.bands.length] });
    S.bands.length = n; rebuildHandles();
  }
  function syncControls(): void {
    nDyeS.set(S.dyes.length); nChS.set(S.bands.length); detC.checked = S.detector; binS.set(S.bins);
    selBtns.forEach((b, i) => { b.style.display = i < S.dyes.length ? "" : "none"; b.classList.toggle("active", i === S.sel); b.style.borderColor = i < S.dyes.length ? wlCss(S.dyes[i].peak) : ""; });
    wS.set(S.dyes[S.sel].width); tS.set(S.dyes[S.sel].tail);
  }
  const bandsNow = (): U.Band[] => S.detector ? U.contiguousBands(470, 710, S.bins) : S.bands;

  // ---- handles ----
  function rebuildHandles(): void {
    handleLayer.replaceChildren();
    bandLayer.replaceChildren();
    if (!S.detector) S.bands.forEach((b, i) => {
      const rect = svgEl("rect", { class: "sm-band", fill: "#888", "fill-opacity": 0.16, stroke: "#666", "stroke-width": 1.5 }, bandLayer);
      const e0 = svgEl("rect", { class: "sm-edge", fill: "#555", "fill-opacity": 0.5, width: 10 }, handleLayer);
      const e1 = svgEl("rect", { class: "sm-edge", fill: "#555", "fill-opacity": 0.5, width: 10 }, handleLayer);
      const lab = svgEl("text", { "text-anchor": "middle", "font-size": 17, fill: "#333" }, handleLayer);
      let grab = 0;
      rect.addEventListener("pointerdown", ev => { grab = plot.toData(ev)[0] - b.lo; });
      draggable(rect, ev => plot.toData(ev), x => { const w = b.hi - b.lo; b.lo = Math.max(U.L_MIN, Math.min(U.L_MAX - w, Math.round(x - grab))); b.hi = b.lo + w; manual(); });
      draggable(e0, ev => plot.toData(ev), x => { b.lo = Math.max(U.L_MIN, Math.min(b.hi - 4, Math.round(x))); manual(); });
      draggable(e1, ev => plot.toData(ev), x => { b.hi = Math.min(U.L_MAX, Math.max(b.lo + 4, Math.round(x))); manual(); });
      (b as U.Band & { els?: SVGElement[] }).els = [rect, e0, e1, lab];
      void i;
    });
    S.dyes.forEach(d => {
      const h = svgEl("circle", { r: 13, "stroke-width": 3, fill: "#fff" }, handleLayer);
      const lt = svgEl("text", { "text-anchor": "middle", "font-size": 17, "font-weight": 700, "pointer-events": "none" }, handleLayer);
      draggable(h, ev => plot.toData(ev), x => { d.peak = Math.max(420, Math.min(720, Math.round(x))); manual(); });
      (d as U.Dye & { el?: SVGCircleElement; lab?: SVGTextElement }).el = h;
      (d as U.Dye & { lab?: SVGTextElement }).lab = lt;
    });
  }

  function draw(): void {
    const bands = bandsNow(), n = bands.length, k = S.dyes.length;
    // bands
    if (S.detector) {
      bandLayer.replaceChildren();
      bands.forEach((b, i) => svgEl("rect", { x: plot.x(b.lo), y: plot.box.t, width: plot.x(b.hi) - plot.x(b.lo), height: plot.box.b - plot.box.t, fill: i % 2 ? "#888" : "#bbb", "fill-opacity": 0.14 }, bandLayer));
    } else S.bands.forEach((b, i) => {
      const [rect, e0, e1, lab] = (b as U.Band & { els?: SVGElement[] }).els ?? [];
      if (!rect) return;
      const x0 = plot.x(b.lo), x1 = plot.x(b.hi), t = plot.box.t, h = plot.box.b - plot.box.t;
      rect.setAttribute("x", String(x0)); rect.setAttribute("y", String(t)); rect.setAttribute("width", String(x1 - x0)); rect.setAttribute("height", String(h));
      for (const [e, x] of [[e0, x0], [e1, x1]] as [SVGElement, number][]) { e.setAttribute("x", String(x - 5)); e.setAttribute("y", String(t + h * 0.82)); e.setAttribute("height", String(h * 0.18)); }
      lab.setAttribute("x", String((x0 + x1) / 2)); lab.setAttribute("y", String(t + 20)); lab.textContent = "ch " + (i + 1);
    });
    // curves
    curveLayer.replaceChildren();
    S.dyes.forEach((d, j) => {
      const xs: number[] = [], ys: number[] = [];
      for (let l = U.L_MIN; l <= U.L_MAX; l += 2) { xs.push(l); ys.push(U.emission(d, l)); }
      const pts = xs.map((x, i) => `${plot.x(x).toFixed(1)},${plot.y(ys[i]).toFixed(1)}`).join(" ");
      const col = wlCss(d.peak);
      svgEl("polygon", { points: `${plot.x(U.L_MIN)},${plot.y(0)} ${pts} ${plot.x(U.L_MAX)},${plot.y(0)}`, fill: col, "fill-opacity": 0.22, stroke: "none" }, curveLayer);
      svgEl("polyline", { points: pts, fill: "none", stroke: col, "stroke-width": 3.5 }, curveLayer);
      const h = (d as U.Dye & { el?: SVGCircleElement }).el;
      if (h) { h.setAttribute("cx", String(plot.x(d.peak))); h.setAttribute("cy", String(plot.y(1))); h.setAttribute("stroke", col); }
      const lt = (d as U.Dye & { lab?: SVGTextElement }).lab;
      if (lt) { lt.setAttribute("x", String(plot.x(d.peak))); lt.setAttribute("y", String(plot.y(1) + 6)); lt.textContent = String(j + 1); }
    });
    // matrix
    const M = U.channelMatrix(S.dyes, bands);
    const own = Array.from({ length: k }, (_, j) => { let best = 0; for (let c = 1; c < n; c++) if (M[c * k + j] > M[best * k + j]) best = c; return best; });
    const Sn = new Float64Array(n * k);
    for (let j = 0; j < k; j++) { const o = n === k ? M[j * k + j] : M[own[j] * k + j]; for (let c = 0; c < n; c++) Sn[c * k + j] = o > 0 ? M[c * k + j] / o : 0; }
    matBox.replaceChildren();
    let g;
    if (n <= 4) g = heatmapGrid(matBox, Sn, n, k, { map: heat, lo: 0, hi: 1, numbers: true, fmt: v => v.toFixed(2), rowLabels: bands.map((_, i) => "ch " + (i + 1)), colLabels: S.dyes.map((_, j) => "dye " + (j + 1)), font: 17, cell: 60 });
    else {
      // more channels than fit as rows: transposed (one row per dye, one column per channel)
      const T = new Float64Array(k * n);
      for (let c = 0; c < n; c++) for (let j = 0; j < k; j++) T[j * n + c] = Sn[c * k + j];
      g = heatmapGrid(matBox, T, k, n, { map: heat, lo: 0, hi: 1, numbers: n <= 8, fmt: v => v.toFixed(2), rowLabels: S.dyes.map((_, j) => "dye " + (j + 1)), colLabels: n <= 8 ? bands.map((_, i) => "ch " + (i + 1)) : undefined, font: 15, cell: n <= 8 ? 44 : Math.max(10, Math.floor(360 / n)), title: n <= 8 ? undefined : `channels 1 … ${n} →` });
    }
    uniqueClip(g.svg, NAME);
    matBox.style.flexBasis = Math.round(g.svg.viewBox.baseVal.width * 1.05) + "px";
    // readouts
    const unitCols = new Float64Array(n * k);
    for (let j = 0; j < k; j++) { let s = 0; for (let c = 0; c < n; c++) s += M[c * k + j] ** 2; s = Math.sqrt(s); for (let c = 0; c < n; c++) unitCols[c * k + j] = s > 0 ? M[c * k + j] / s : 0; }
    const kappa = n >= k ? U.condition(unitCols, n, k) : Infinity;
    const lines: string[] = [];
    lines.push(n === k ? "M, each column ÷ its own channel (diagonal = 1)" : "M, each column ÷ its largest entry");
    if (S.detector) lines.push("each dye spreads over several bins: least squares\nuses them all (next unit)");
    else if (n === k && k === 2) lines.push(`bleed-through: dye 1 → ch 2 ${fmt(100 * Sn[2], 1)} %, dye 2 → ch 1 ${fmt(100 * Sn[1], 1)} %`);
    else {
      let worst = 0, wi = 0, wj = 0;
      for (let c = 0; c < n; c++) for (let j = 0; j < k; j++) if (c !== own[j] && Sn[c * k + j] > worst) { worst = Sn[c * k + j]; wi = c; wj = j; }
      lines.push(`largest bleed-through: dye ${wj + 1} → ch ${wi + 1}: ${fmt(100 * worst, 1)} %`);
    }
    if (n === k) lines.push(`det (diagonal = 1): ${fmt(det(Sn, n), 3)}`);
    lines.push(`condition number κ: ${n < k ? "∞ (fewer channels than dyes)" : fmt(kappa, kappa < 100 ? 2 : 0)}`);
    lines.push("photons in the dye's own channel:");
    lines.push("  " + S.dyes.map((_, j) => `${j + 1}: ${fmt(100 * M[own[j] * k + j], 0)} %`).join("   "));
    read.textContent = lines.join("\n");
  }

  syncControls(); rebuildHandles(); draw();
}

autoMount(NAME, mount);
