// flood1d: watershed by flooding on a 1D profile (watershed day, unit ws-landscape).
// The profile is drawn as a landscape (grey); a water-level slider floods it from below,
// with one colour per basin; dams (dark bars) appear where two basins meet, labelled with
// their height. Options: flood |f'| instead of f (the derivative is drawn below the input
// and flooded; its basins are the flat parts of f, its ridges the edges), h (shallow minima
// are filled first, hatched, and their basins merge), markers (click samples: they become
// the only sources; other minima are drowned). Drag on the profile to redraw it.
// The basins come from lib/watershed.ts (Meyer flooding on a 1 × n image).
// Options: data-preset (cells | noisy | hmin | edges | plateau | twelve), data-level, data-h.
import { autoMount, htmlEl, slider, button, presets, checkbox, svgEl } from "../lib/ui";
import { watershed, hMinimaMarkers, minimaMarkers } from "../lib/watershed";
import { hMinTransform } from "../lib/morph";
import { labelColor, toCss } from "../lib/colour";
import { basinsAtLevel, dams, derivative1d, twoCells, steps3, plateau, TWELVE } from "./f1d";

const NAME = "flood1d";
const W = 1100, ML = 58, MR = 14;

interface Preset { id: string; label: string; f: number[]; level: number; h: number; deriv: boolean; marks?: number[] }
const PRESETS: Preset[] = [
  { id: "cells", label: "two cells", f: twoCells(), level: 5, h: 0, deriv: false },
  { id: "noisy", label: "noisy: one basin per wiggle", f: twoCells(0.35), level: 5, h: 0, deriv: false },
  { id: "hmin", label: "h removes the noise", f: twoCells(0.35), level: 5, h: 1, deriv: false },
  { id: "edges", label: "edges: flood the derivative", f: steps3(), level: 1, h: 0.3, deriv: true },
  { id: "plateau", label: "plateau", f: plateau(), level: 6, h: 0, deriv: false },
  { id: "twelve", label: "12 samples (exercise)", f: TWELVE, level: 4, h: 0, deriv: false },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { flex-direction: column; gap: .45em; }
.qb-demo[data-demo="${NAME}"] .f1-plot { width: 100%; max-width: 1100px; }
.qb-demo[data-demo="${NAME}"] .f1-plot svg { touch-action: none; cursor: pointer; }
.qb-demo[data-demo="${NAME}"] .f1-row { display: flex; flex-wrap: wrap; gap: .4em 1.2em; align-items: center; width: 100%; }
.qb-demo[data-demo="${NAME}"] .f1-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-slider { min-width: 230px; }
.qb-demo[data-demo="${NAME}"] svg .num { font-size: 18px; font-family: ui-monospace, monospace; }
.qb-demo[data-demo="${NAME}"] svg .lab { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const TERRAIN = "#a39a8c", WATERLINE = "#1f5fa8", DAM = "#3b2f2f";
const letter = (k: number) => String.fromCharCode(64 + k);
const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { f: [] as number[], level: 5, h: 0, deriv: false, useMarks: false, marks: new Set<number>(), edit: "profile" as "profile" | "marker" };

  const plot = htmlEl("div", "f1-plot", "", host);
  const svg = svgEl("svg", { width: "100%", role: "img", "aria-label": "a 1D landscape flooded with water" }, plot);
  const defs = svgEl("defs", {}, svg);
  const hatchId = "f1-hatch-" + Math.random().toString(36).slice(2);
  const pat = svgEl("pattern", { id: hatchId, width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
  svgEl("rect", { width: 7, height: 7, fill: "#7f7f7f", "fill-opacity": 0.15 }, pat);
  svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 7, stroke: "#555", "stroke-width": 2.2 }, pat);
  const g = svgEl("g", {}, svg);

  const row1 = htmlEl("div", "f1-row", "", host);
  const levelS = slider(row1, "water level", 0, 10, 0.1, 5, v => { st.level = v; draw(); });
  const hS = slider(row1, "h", 0, 4, 0.1, 0, v => { st.h = v; recompute(); });
  const derivC = checkbox(row1, "flood |f′| instead of f", false, v => { st.deriv = v; recompute(); });
  const row2 = htmlEl("div", "f1-row", "", host);
  const markC = checkbox(row2, "markers", false, v => { st.useMarks = v; st.edit = v ? "marker" : "profile"; recompute(); });
  const editBox = htmlEl("span", "qb-presets", "", row2);
  htmlEl("span", "f1-lab", "a click", editBox);
  const editBtns = presets(editBox, [
    { label: "draws the profile", apply: () => { st.edit = "profile"; draw(); } },
    { label: "sets a marker", apply: () => { st.edit = "marker"; draw(); } },
  ]);
  button(row2, "clear markers", () => { st.marks.clear(); recompute(); });
  const ro = htmlEl("div", "qb-readout", "", row2);
  const row3 = htmlEl("div", "f1-row", "", host);
  htmlEl("span", "f1-lab", "examples", row3);
  const presetBtns = presets(row3, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(row3, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", host);

  let gsig: number[] = [], markers: Int32Array = new Int32Array(0), lab: Int32Array = new Int32Array(0), filled: Float32Array | null = null, nMin = 0, vmax = 10;

  function load(p: Preset): void {
    const first = p === p0;
    st.f = p.f.slice();
    st.deriv = p.deriv; derivC.checked = p.deriv;
    st.h = first && host.dataset.h ? Number(host.dataset.h) : p.h;
    st.level = first && host.dataset.level ? Number(host.dataset.level) : p.level;
    st.marks = new Set(p.marks ?? []);
    st.useMarks = st.marks.size > 0; markC.checked = st.useMarks;
    st.edit = "profile";
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    recompute();
  }

  function recompute(): void {
    const n = st.f.length;
    gsig = st.deriv ? derivative1d(st.f) : st.f.slice();
    vmax = st.deriv ? Math.max(1, Math.ceil(Math.max(...gsig) * 1.15 * 2) / 2) : Math.max(10, Math.ceil(Math.max(...st.f)));
    levelS.input.max = String(vmax);
    levelS.input.step = st.deriv ? "0.05" : "0.1";
    hS.input.max = st.deriv ? "1.5" : "4";
    hS.input.step = st.deriv ? "0.05" : "0.1";
    st.level = Math.min(st.level, vmax); levelS.set(st.level);
    st.h = Math.min(st.h, Number(hS.input.max)); hS.set(st.h);
    nMin = minimaMarkers(gsig, n, 1, 4).count;
    filled = null;
    if (st.useMarks) {
      markers = new Int32Array(n);
      [...st.marks].sort((a, b) => a - b).forEach((i, k) => { if (i < n) markers[i] = k + 1; });
    } else {
      markers = hMinimaMarkers(gsig, n, 1, st.h, 4).data;
      if (st.h > 0) filled = hMinTransform(gsig, n, 1, st.h, 4);
    }
    lab = watershed(gsig, n, 1, markers, { conn: 4 });
    draw();
  }

  function draw(): void {
    editBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.edit === "profile")));
    editBox.style.display = st.useMarks ? "" : "none";
    (hS.input.closest("label") as HTMLElement).style.opacity = st.useMarks ? "0.4" : "1";
    const n = st.f.length, few = n <= 20, deriv = st.deriv;
    // vertical layout: [input plot (deriv only)] [flooded plot] [strip] [number rows]
    const topH = deriv ? 150 : 0, floodH = deriv ? 190 : 250, gap = deriv ? 34 : 0;
    const MT = 16, stripH = 14, rowsH = few ? 52 : 26;
    const y0f = MT, y0g = MT + topH + gap, yStrip = y0g + floodH + 6;
    const H = yStrip + stripH + rowsH + 6;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    g.replaceChildren();
    const bw = (W - ML - MR) / n, X = (i: number) => ML + i * bw;
    const fmax = Math.max(10, Math.ceil(Math.max(...st.f)));
    const Yf = (v: number) => y0f + topH * (1 - v / fmax);
    const Yg = (v: number) => y0g + floodH * (1 - Math.max(0, Math.min(vmax, v)) / vmax);
    const col = (k: number) => toCss(labelColor(k, [1, 1, 1]));

    // axes and grid of the flooded plot
    const tickStep = vmax <= 2 ? 0.5 : vmax <= 5 ? 1 : 2;
    for (let v = 0; v <= vmax + 1e-9; v += tickStep) {
      svgEl("line", { x1: ML, x2: W - MR, y1: Yg(v), y2: Yg(v), stroke: "var(--qb-grid, #e3e3e3)" }, g);
      const t = svgEl("text", { x: ML - 8, y: Yg(v) + 6, "text-anchor": "end", class: "num" }, g); t.textContent = fmt(v);
    }
    const name = svgEl("text", { x: ML + 6, y: y0g + 18, class: "lab" }, g); name.textContent = deriv ? "|f′| (flooded)" : "f";
    // terrain
    for (let i = 0; i < n; i++) svgEl("rect", { x: X(i), y: Yg(gsig[i]), width: bw + 0.4, height: Yg(0) - Yg(gsig[i]), fill: TERRAIN }, g);
    // h-filling (hatched)
    if (filled) for (let i = 0; i < n; i++) if (filled[i] > gsig[i] + 1e-6)
      svgEl("rect", { x: X(i), y: Yg(filled[i]), width: bw + 0.4, height: Yg(gsig[i]) - Yg(filled[i]), fill: `url(#${hatchId})` }, g);
    // water
    const water = basinsAtLevel(gsig, lab, markers, st.level);
    for (let i = 0; i < n; i++) if (water[i] && st.level > gsig[i])
      svgEl("rect", { x: X(i), y: Yg(st.level), width: bw + 0.4, height: Yg(gsig[i]) - Yg(st.level), fill: col(water[i]), "fill-opacity": 0.85 }, g);
    for (let i = 0; i < n; i++) if (water[i] && st.level > gsig[i])
      svgEl("line", { x1: X(i), x2: X(i + 1), y1: Yg(st.level), y2: Yg(st.level), stroke: WATERLINE, "stroke-width": 2 }, g);
    // dams
    const dm = dams(gsig, lab), built: string[] = [];
    for (const d of dm) {
      const x = X(d.at + 1);
      if (st.level >= d.height) {
        svgEl("rect", { x: x - 4, y: Yg(st.level) - 6, width: 8, height: Yg(d.height) - Yg(st.level) + 6, fill: DAM }, g);
        const t = svgEl("text", { x, y: Yg(st.level) - 12, "text-anchor": "middle", class: "num" }, g); t.textContent = fmt(d.height);
        built.push(fmt(d.height));
      } else {
        svgEl("line", { x1: x, x2: x, y1: Yg(d.height), y2: Yg(0), stroke: DAM, "stroke-width": 1.5, "stroke-dasharray": "4 4" }, g);
      }
    }
    // level line
    svgEl("line", { x1: ML, x2: W - MR, y1: Yg(st.level), y2: Yg(st.level), stroke: WATERLINE, "stroke-width": 1, "stroke-dasharray": "6 5" }, g);
    svgEl("line", { x1: ML, x2: W - MR, y1: Yg(0), y2: Yg(0), stroke: "var(--qb-axis, #444)", "stroke-width": 1.5 }, g);
    // markers (sources)
    // one triangle per marker (at the lowest sample of its run, middle of ties)
    const runs = new Map<number, number[]>();
    for (let i = 0; i < n; i++) if (markers[i]) { const r = runs.get(markers[i]) ?? []; r.push(i); runs.set(markers[i], r); }
    for (const r of runs.values()) {
      const lo = Math.min(...r.map(i => gsig[i])), at = r.filter(i => gsig[i] === lo), i = at[Math.floor((at.length - 1) / 2)];
      const cx = X(i) + bw / 2, cy = Yg(gsig[i]) - 2;
      svgEl("polygon", { points: `${cx},${cy} ${cx - 7},${cy + 12} ${cx + 7},${cy + 12}`, fill: "#fff", stroke: "#222", "stroke-width": 2 }, g);
    }
    // basin strip (the segmentation)
    for (let i = 0; i < n; i++) svgEl("rect", { x: X(i), y: yStrip, width: bw + 0.4, height: stripH, fill: lab[i] ? col(lab[i]) : "#fff" }, g);
    // number rows
    if (few) {
      for (let i = 0; i < n; i++) {
        const cx = X(i) + bw / 2;
        const t1 = svgEl("text", { x: cx, y: yStrip + stripH + 22, "text-anchor": "middle", class: "num" }, g); t1.textContent = fmt(gsig[i]);
        const t2 = svgEl("text", { x: cx, y: yStrip + stripH + 46, "text-anchor": "middle", class: "num" }, g); t2.textContent = lab[i] ? letter(lab[i]) : "·";
      }
      const a = svgEl("text", { x: ML - 8, y: yStrip + stripH + 22, "text-anchor": "end", class: "lab" }, g); a.textContent = deriv ? "|f′|" : "f";
      const b = svgEl("text", { x: ML - 8, y: yStrip + stripH + 46, "text-anchor": "end", class: "lab" }, g); b.textContent = "basin";
    } else {
      for (let i = 0; i < n; i += 5) { const t = svgEl("text", { x: X(i) + bw / 2, y: yStrip + stripH + 20, "text-anchor": "middle", class: "num" }, g); t.textContent = String(i); t.style.fill = "var(--qb-muted, #777)"; }
    }
    // input profile above (derivative mode)
    if (deriv) {
      for (let v = 0; v <= fmax; v += 5) {
        svgEl("line", { x1: ML, x2: W - MR, y1: Yf(v), y2: Yf(v), stroke: "var(--qb-grid, #e3e3e3)" }, g);
        const t = svgEl("text", { x: ML - 8, y: Yf(v) + 6, "text-anchor": "end", class: "num" }, g); t.textContent = String(v);
      }
      for (let i = 0; i < n; i++) svgEl("rect", { x: X(i), y: Yf(st.f[i]), width: bw + 0.4, height: Yf(0) - Yf(st.f[i]), fill: lab[i] ? col(lab[i]) : TERRAIN, "fill-opacity": 0.75 }, g);
      let d = "";
      for (let i = 0; i < n; i++) d += `${i ? "L" : "M"}${X(i).toFixed(1)},${Yf(st.f[i]).toFixed(1)}H${X(i + 1).toFixed(1)}`;
      svgEl("path", { d, fill: "none", stroke: "#333", "stroke-width": 2 }, g);
      const t = svgEl("text", { x: ML + 6, y: y0f + 18, class: "lab" }, g); t.textContent = "f (input, coloured by basin of |f′|)";
      for (const dd of dm) svgEl("line", { x1: X(dd.at + 1), x2: X(dd.at + 1), y1: Yf(fmax), y2: Yf(0), stroke: DAM, "stroke-width": 2.5 }, g);
    }
    // readout and note
    const k = new Set(Array.from(lab).filter(v => v > 0)).size;
    const srcs = st.useMarks ? `markers: ${st.marks.size}` : st.h > 0 ? `minima: ${nMin}, after filling by h: ${k}` : `regional minima: ${nMin}`;
    ro.textContent = `${srcs} · basins: ${k} · dams built: ${built.length ? `${built.length} (height ${built.join(", ")})` : "none"}`;
    note.textContent = st.useMarks
      ? (st.marks.size ? "▲ = markers, the only sources. Water from a marker drowns every unmarked minimum it reaches; dams are built only between markers." : "Choose \"sets a marker\" and click samples: they become the only sources.")
      : deriv ? "Flooding |f′|: the flat parts of f are valleys, the edges are ridges, so the dams (dark lines, also drawn on f) sit on the edges."
      : st.h > 0 ? `Every basin is first filled by h = ${fmt(st.h)} (hatched); a basin that overflows merges with its neighbour, so only minima deeper than h remain sources (▲).`
      : "Every regional minimum (▲) is a source. Raise the water level: where water from two basins would meet, a dam is built; its height is the pass between them.";
  }

  // pointer: draw the profile, or toggle markers
  let dragging = false;
  const sampleAt = (ev: PointerEvent): [number, number] => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return [-1, 0];
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(ctm.inverse());
    const n = st.f.length, i = Math.floor((p.x - ML) / ((W - MR - ML) / n));
    const deriv = st.deriv, topH = deriv ? 150 : 250, y0 = 16;
    const fmax = deriv ? Math.max(10, Math.ceil(Math.max(...st.f))) : vmax;
    let v = fmax * (1 - (p.y - y0) / topH);
    const ints = st.f.every(Number.isInteger);
    v = ints ? Math.round(v) : Math.round(v * 10) / 10;
    return [i, Math.max(0, Math.min(fmax, v))];
  };
  svg.addEventListener("pointerdown", ev => {
    const [i, v] = sampleAt(ev);
    if (i < 0 || i >= st.f.length) return;
    ev.preventDefault();
    if (st.useMarks && st.edit === "marker") {
      if (st.marks.has(i)) st.marks.delete(i); else st.marks.add(i);
      recompute(); return;
    }
    svg.setPointerCapture(ev.pointerId);
    dragging = true;
    if (st.f[i] !== v) { st.f[i] = v; recompute(); }
  });
  svg.addEventListener("pointermove", ev => {
    if (!dragging) return;
    const [i, v] = sampleAt(ev);
    if (i < 0 || i >= st.f.length || st.f[i] === v) return;
    st.f[i] = v; recompute();
  });
  const end = () => { dragging = false; };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);

  load(p0);
}

autoMount(NAME, mount);
