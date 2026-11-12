// recon1d: geodesic dilation and reconstruction on a 1D integer signal, one iteration at a
// time (morphology day, unit morph-reconstruct; embedded on the watershed day with
// data-mode="hmin"). Grey bars: the signal I (drag a bar to change it). Modes:
//   reconstruct  marker M = clicked samples (value I there, 0 elsewhere); R_I(M) by dilation
//   hmax         marker I − h; R_I(I − h) = HMAX_h(I): peaks lowered by h, low ones vanish
//   hmin         marker I + h; R^ε_I(I + h) = HMIN_h(I): basins shallower than h fill up
//   impose       click samples to make markers; M = 0 there, max I + 1 elsewhere;
//                R^ε_G(M) with G = min(I + 1, M): minima only at the markers
// "step" applies one geodesic dilation (erosion): the dashed line is the marker dilated
// (eroded) before clipping, the hatched part is what the mask clips away.
// Options: data-mode, data-preset (seed | hmax | hmin | impose | twelve), data-signal
// ("7 1 4 ..." integers 0..9), data-h.
import { autoMount, htmlEl, slider, button, presets, checkbox, svgEl, numList } from "../lib/ui";
import { Sig, flat3, iterates, regionalMax, runs, imposeParts, TWELVE, HILLS, VMAX } from "./r1d";

const NAME = "recon1d";
type Mode = "reconstruct" | "hmax" | "hmin" | "impose";
const MODES: { id: Mode; label: string }[] = [
  { id: "reconstruct", label: "reconstruct" }, { id: "hmax", label: "h-maxima" },
  { id: "hmin", label: "h-minima" }, { id: "impose", label: "impose minima" },
];
const W = 1100, H = 362, ML = 64, MR = 12, MT = 26, MB = 92;



interface Preset { id: string; label: string; mode: Mode | null; f: number[]; h: number; marks?: number[] }
const argmaxIn = (f: number[], a: number, b: number) => { let k = a; for (let i = a; i <= b; i++) if (f[i] > f[k]) k = i; return k; };
const argminIn = (f: number[], a: number, b: number) => { let k = a; for (let i = a; i <= b; i++) if (f[i] < f[k]) k = i; return k; };
const PRESETS: Preset[] = [
  { id: "seed", label: "a single seed floods its hill", mode: "reconstruct", f: HILLS, h: 2, marks: [argmaxIn(HILLS, 14, 26)] },
  { id: "hmax", label: "h removes the shallow peaks", mode: "hmax", f: HILLS, h: 2 },
  { id: "hmin", label: "h-minima: basins fill", mode: "hmin", f: HILLS, h: 2 },
  { id: "impose", label: "two markers: imposition", mode: "impose", f: HILLS, h: 2, marks: [argminIn(HILLS, 0, 14), argminIn(HILLS, 26, 39)] },
  { id: "twelve", label: "12 samples (exercise)", mode: null, f: TWELVE, h: 2 },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { flex-direction: column; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .r1-plot { width: 100%; max-width: 1100px; }
.qb-demo[data-demo="${NAME}"] .r1-plot svg { touch-action: none; cursor: pointer; }
.qb-demo[data-demo="${NAME}"] .r1-row { display: flex; flex-wrap: wrap; gap: .5em 1.2em; align-items: center; width: 100%; }
.qb-demo[data-demo="${NAME}"] .r1-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-slider { min-width: 200px; }
.qb-demo[data-demo="${NAME}"] svg .num { font-size: 18px; font-family: ui-monospace, monospace; }
.qb-demo[data-demo="${NAME}"] svg .rowlab { font-size: 18px; font-weight: 600; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const custom = host.dataset.signal ? numList(host.dataset.signal, TWELVE).map(v => Math.max(0, Math.min(VMAX, Math.round(v)))) : null;
  const modeOf = (s?: string): Mode | null => (MODES.some(m => m.id === s) ? (s as Mode) : null);
  const dataMode = modeOf(host.dataset.mode);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS.find(p => p.mode === dataMode) ?? PRESETS[0];
  const st = { mode: (dataMode ?? p0.mode ?? "reconstruct") as Mode, f: [] as number[], h: 2, marks: new Set<number>(), k: 0, edit: "value" as "value" | "marker", preview: true };
  let its: Sig[] = [], mask: Sig = new Int32Array(0), timer: number | null = null;

  const plot = htmlEl("div", "r1-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "1D signal and marker" }, plot);
  const defs = svgEl("defs", {}, svg);
  const pat = svgEl("pattern", { id: "r1-hatch-" + Math.random().toString(36).slice(2), width: 8, height: 8, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
  svgEl("rect", { width: 8, height: 8, fill: "#d62728", "fill-opacity": 0.12 }, pat);
  svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 8, stroke: "#d62728", "stroke-width": 3 }, pat);
  const gMain = svgEl("g", {}, svg);

  const row1 = htmlEl("div", "r1-row", "", host);
  const modeBtns = presets(row1, MODES.map(m => ({ label: m.label, apply: () => { st.mode = m.id; recompute(); } })));
  const hS = slider(row1, "h", 1, 5, 1, 2, v => { st.h = v; recompute(); });
  const editBox = htmlEl("span", "qb-presets", "", row1);
  htmlEl("span", "r1-lab", "a click", editBox);
  const editBtns = presets(editBox, [
    { label: "drags a bar", apply: () => { st.edit = "value"; draw(); } },
    { label: "sets a marker", apply: () => { st.edit = "marker"; draw(); } },
  ]);
  const row2 = htmlEl("div", "r1-row", "", host);
  const ctl = htmlEl("span", "qb-presets", "", row2);
  button(ctl, "⏮ start", () => { play(false); st.k = 0; draw(); });
  button(ctl, "◀ back", () => { play(false); st.k = Math.max(0, st.k - 1); draw(); });
  const stepBtn = button(ctl, "step ▶", () => { play(false); st.k = Math.min(its.length - 1, st.k + 1); draw(); });
  const runBtn = button(ctl, "run to stability", () => play());
  checkbox(row2, "show next step", true, v => { st.preview = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", row2);
  const row3 = htmlEl("div", "r1-row", "", host);
  htmlEl("span", "r1-lab", "examples", row3);
  const presetBtns = presets(row3, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(row3, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", host);

  function play(on?: boolean): void {
    const want = on ?? timer === null;
    if (timer !== null) { clearInterval(timer); timer = null; }
    if (want) {
      if (st.k >= its.length - 1) st.k = 0;
      draw();
      timer = window.setInterval(() => {
        if (!host.isConnected || st.k >= its.length - 1) { play(false); return; }
        st.k++; draw();
      }, 550);
    }
    runBtn.classList.toggle("active", timer !== null);
  }

  function load(p: Preset): void {
    play(false);
    const first = p === p0;
    st.f = (first && custom ? custom : p.f).slice();
    if (first && dataMode) st.mode = dataMode; else if (p.mode) st.mode = p.mode;
    st.h = first && Number(host.dataset.h) ? Number(host.dataset.h) : p.h;
    hS.set(st.h);
    st.marks = new Set(first && custom ? [] : p.marks ?? []);
    st.edit = "value";
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    recompute();
  }

  const byDil = () => st.mode === "reconstruct" || st.mode === "hmax";

  function recompute(): void {
    play(false);
    const f = st.f, n = f.length;
    let marker: Int32Array;
    mask = Int32Array.from(f);
    if (st.mode === "reconstruct") marker = Int32Array.from(f, (v, i) => (st.marks.has(i) ? v : 0));
    else if (st.mode === "hmax") marker = Int32Array.from(f, v => v - st.h);
    else if (st.mode === "hmin") marker = Int32Array.from(f, v => v + st.h);
    else {
      const mk = Uint8Array.from({ length: n }, (_, i) => (st.marks.has(i) ? 1 : 0));
      const { fm, g } = imposeParts(f, mk);
      marker = fm; mask = g;
    }
    its = iterates(marker, mask, byDil());
    st.k = 0;
    draw();
  }

  // ---- drawing ----
  const n = () => st.f.length;
  const bw = () => (W - ML - MR) / n();
  const X = (i: number) => ML + i * bw();
  const top = () => Math.max(VMAX + 1, ...st.f, ...(st.mode === "impose" ? [] : its[0] ?? []));
  const bot = () => Math.min(0, ...(its[0] ?? []));
  const Y = (v: number) => MT + (H - MT - MB) * (1 - (Math.max(bot(), Math.min(top(), v)) - bot()) / (top() - bot()));

  function stepPath(vals: ArrayLike<number>): string {
    let d = "";
    for (let i = 0; i < vals.length; i++) d += `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(vals[i]).toFixed(1)}H${X(i + 1).toFixed(1)}`;
    return d;
  }

  function draw(): void {
    modeBtns.forEach((b, i) => b.classList.toggle("active", MODES[i].id === st.mode));
    editBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.edit === "value")));
    const markerModes = st.mode === "reconstruct" || st.mode === "impose";
    editBox.style.display = markerModes ? "" : "none";
    if (!markerModes) st.edit = "value";
    hS.input.closest("label")!.style.display = st.mode === "hmax" || st.mode === "hmin" ? "" : "none";
    const f = st.f, N = n(), m = its[st.k], last = its[its.length - 1], dil = byDil();
    const g = gMain;
    g.replaceChildren();
    // axis, grid
    for (let v = bot(); v <= top(); v += 1) {
      svgEl("line", { x1: ML, x2: W - MR, y1: Y(v), y2: Y(v), stroke: "var(--qb-grid, #e3e3e3)", "stroke-width": 1 }, g);
      if (v % 2 === 0 || top() <= 10) { const t = svgEl("text", { x: ML - 8, y: Y(v) + 6, "text-anchor": "end", class: "num" }, g); t.textContent = String(v); }
    }
    const w = bw(), pad = Math.min(3, w * 0.12);
    for (let i = 0; i < N; i++) {
      svgEl("rect", { x: X(i) + pad, y: Y(f[i]), width: w - 2 * pad, height: Y(0) - Y(f[i]), fill: "#9aa0a8" }, g);
    }
    // the current iterate
    for (let i = 0; i < N; i++) {
      const v = m[i];
      if (dil) {
        svgEl("rect", { x: X(i) + pad, y: Y(Math.max(0, v)), width: w - 2 * pad, height: Math.abs(Y(0) - Y(v)), fill: "#ff7f0e", "fill-opacity": 0.9 }, g);
      } else if (v > f[i]) {
        svgEl("rect", { x: X(i) + pad, y: Y(v), width: w - 2 * pad, height: Y(f[i]) - Y(v), fill: "#4c9be8", "fill-opacity": 0.75 }, g);
      } else if (v < f[i]) {
        svgEl("rect", { x: X(i) + pad, y: Y(f[i]), width: w - 2 * pad, height: Y(v) - Y(f[i]), fill: "#9467bd", "fill-opacity": 0.8 }, g);
      }
    }
    svgEl("path", { d: stepPath(m), fill: "none", stroke: dil ? "#b35300" : "#1f5fa8", "stroke-width": 3 }, g);
    // preview of the next step: dilated (eroded) marker before clipping, clipped part hatched
    if (st.preview && st.k < its.length - 1) {
      const p = flat3(m, dil);
      for (let i = 0; i < N; i++) {
        const lim = mask[i];
        if (dil && p[i] > lim) svgEl("rect", { x: X(i) + pad, y: Y(p[i]), width: w - 2 * pad, height: Y(lim) - Y(p[i]), fill: `url(#${pat.id})` }, g);
        if (!dil && p[i] < lim) svgEl("rect", { x: X(i) + pad, y: Y(lim), width: w - 2 * pad, height: Y(p[i]) - Y(lim), fill: `url(#${pat.id})` }, g);
      }
      svgEl("path", { d: stepPath(p), fill: "none", stroke: "#d62728", "stroke-width": 2.5, "stroke-dasharray": "7 5" }, g);
    }
    if (st.mode === "impose") svgEl("path", { d: stepPath(mask), fill: "none", stroke: "#2ca02c", "stroke-width": 2, "stroke-dasharray": "2 4" }, g);
    // extrema of the result (at stability)
    if (st.k === its.length - 1) {
      const ext = regionalMax(last, !dil);
      for (let i = 0; i < N; i++) if (ext[i]) svgEl("circle", { cx: X(i) + w / 2, cy: Y(last[i]) - 9, r: 6, fill: dil ? "#d62728" : "#1f5fa8" }, g);
    }
    // markers
    svgEl("line", { x1: ML, x2: W - MR, y1: Y(0), y2: Y(0), stroke: "var(--qb-axis, #444)", "stroke-width": 1.5 }, g);
    // number rows
    const rowY = [H - MB + 42, H - MB + 66, H - MB + 88];
    const lab = (y: number, s: string) => { const t = svgEl("text", { x: ML - 8, y, "text-anchor": "end", class: "rowlab" }, g); t.textContent = s; };
    lab(rowY[0], "I"); lab(rowY[1], st.mode === "impose" ? "G" : "M"); if (st.mode === "impose") lab(rowY[2], "M");
    for (let i = 0; i < N; i++) {
      const cx = X(i) + w / 2;
      const t1 = svgEl("text", { x: cx, y: rowY[0], "text-anchor": "middle", class: "num" }, g); t1.textContent = String(f[i]);
      const second = st.mode === "impose" ? mask[i] : m[i];
      const t2 = svgEl("text", { x: cx, y: rowY[1], "text-anchor": "middle", class: "num" }, g); t2.textContent = String(second);
      if (st.mode === "impose") { const t3 = svgEl("text", { x: cx, y: rowY[2], "text-anchor": "middle", class: "num" }, g); t3.textContent = String(m[i]); }
      if (st.marks.has(i) && markerModes) { const t = svgEl("text", { x: cx, y: Y(bot()) + 22, "text-anchor": "middle", class: "num" }, g); t.textContent = "▲"; t.style.fill = "#9467bd"; }
      else if (i % 5 === 0 && N > 14) { const t = svgEl("text", { x: cx, y: Y(bot()) + 22, "text-anchor": "middle", class: "num" }, g); t.textContent = String(i); t.style.fill = "var(--qb-muted, #777)"; }
      else if (N <= 14) { const t = svgEl("text", { x: cx, y: Y(bot()) + 22, "text-anchor": "middle", class: "num" }, g); t.textContent = String(i); t.style.fill = "var(--qb-muted, #777)"; }
    }
    // readout
    const K = its.length - 1;
    const ext0 = runs(regionalMax(f, !dil)), ext1 = runs(regionalMax(last, !dil));
    const what = dil ? "regional maxima" : "regional minima";
    stepBtn.disabled = st.k >= K;
    ro.textContent = `iteration ${st.k} of ${K}${st.k === K ? " (stable)" : ""} · ${what}: I has ${ext0}, result ${ext1}`;
    const txt: Record<Mode, string> = {
      reconstruct: st.marks.size ? "Orange: the marker M, grown by one sample per step and clipped under I; at the end it covers everything the seed can reach without climbing above its own height." : "Choose \"sets a marker\" and click samples to place seeds.",
      hmax: `Marker M = I − ${st.h}. At the end every peak is lowered by ${st.h}; peaks that rise ${st.h} or less above the pass to a higher peak are gone. The grey left above the orange = the domes I − HMAX.`,
      hmin: `Marker M = I + ${st.h} (blue = water). At the end every basin is filled by ${st.h}; basins ${st.h} deep or less are full and no longer minima.`,
      impose: st.marks.size ? "▲: the markers (value 0). Dotted green: the mask G = min(I + 1, M). Blue: the imposed landscape above I; its only minima are the markers." : "Choose \"sets a marker\" and click samples to place markers.",
    };
    note.textContent = txt[st.mode];
  }

  // ---- pointer ----
  let dragging = false;
  const colAt = (ev: PointerEvent): [number, number] => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return [-1, 0];
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(ctm.inverse());
    const i = Math.floor((p.x - ML) / bw());
    const v = Math.round(bot() + (top() - bot()) * (1 - (p.y - MT) / (H - MT - MB)));
    return [i, Math.max(0, Math.min(VMAX, v))];
  };
  svg.addEventListener("pointerdown", ev => {
    const [i, v] = colAt(ev);
    if (i < 0 || i >= n()) return;
    ev.preventDefault();
    if (st.edit === "marker") {
      if (st.marks.has(i)) st.marks.delete(i); else st.marks.add(i);
      recompute(); return;
    }
    svg.setPointerCapture(ev.pointerId);
    dragging = true;
    if (st.f[i] !== v) { st.f[i] = v; recompute(); }
  });
  svg.addEventListener("pointermove", ev => {
    if (!dragging) return;
    const [i, v] = colAt(ev);
    if (i < 0 || i >= n() || st.f[i] === v) return;
    st.f[i] = v; recompute();
  });
  const end = () => { dragging = false; };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);

  load(p0);
  if (host.dataset.frame === "end") { st.k = its.length - 1; draw(); }
}

autoMount(NAME, mount);
