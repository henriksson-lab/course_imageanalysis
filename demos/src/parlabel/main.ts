// Irregular algorithms on a (simulated) GPU (lib/parallel.ts):
//  - mode "label": connected components of a 24 x 24 binary image (click to edit). Sequential
//    two-pass union–find (reference, lib/label.ts) vs label propagation (every pixel takes the
//    smallest label among its neighbours, all at once, pass after pass) and propagation with
//    pointer jumping. Passes, pixel updates, and a model time (passes x launch + work / lanes)
//    against the sequential loop, here and extrapolated to 1024 x 1024.
//  - mode "wavefront": a DP table filled anti-diagonal by anti-diagonal; busy lanes per step.
//  - mode "flood": watershed flooding with a priority queue (one pixel per step) vs
//    level-synchronous flooding (all pixels of one level that touch a basin, per step).
//
// <div data-demo="parlabel" data-mode="label|wavefront|flood" data-preset="..."></div>
import { autoMount, htmlEl, presets, button, svgEl } from "../lib/ui";
import { label2d } from "../lib/label";
import { labelColor, toCss, viridis } from "../lib/colour";
import * as P from "../lib/parallel";

const NAME = "parlabel";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { flex-wrap: nowrap; gap: 1em; }
${S} .pl-main { flex: 1 1 640px; min-width: 0; }
${S} .pl-main svg { width: 100%; display: block; touch-action: none; }
${S} .qb-panel { flex: 0 1 330px; gap: .4em; }
${S} .pl-lab { font-size: 14px; color: var(--qb-muted); margin-top: .2em; }
${S} .qb-readout { font-size: 15px; }
${S} .pl-hidden { display: none !important; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .pl-main svg { max-height: 540px; }
.reveal ${S} .qb-panel { flex: 0 0 340px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .4em; }
.reveal ${S} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${S} .pl-lab { font-size: 15px; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

const N = 24;
type Mode = "label" | "wavefront" | "flood";
type Algo = "seq" | "prop" | "jump";
type Shape = "blobs" | "spiral" | "comb" | "bar";
interface State { mode: Mode; algo: Algo; shape: Shape; step: number }
const DEFAULT: State = { mode: "label", algo: "prop", shape: "blobs", step: 9999 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "blobs", label: "blobs: 12 passes", s: { mode: "label", shape: "blobs", algo: "prop" } },
  { id: "spiral", label: "spiral: 262 passes", s: { mode: "label", shape: "spiral", algo: "prop" } },
  { id: "jump", label: "pointer jumping: 41 passes, not 262", s: { mode: "label", shape: "spiral", algo: "jump" } },
  { id: "wavefront", label: "DP wavefront: starts and ends with one lane", s: { mode: "wavefront" } },
  { id: "flood", label: "flooding: one pixel at a time", s: { mode: "flood" } },
];

function shapeMask(s: Shape): Uint8Array {
  const m = new Uint8Array(N * N), set = (x: number, y: number) => { if (x >= 0 && y >= 0 && x < N && y < N) m[y * N + x] = 1; };
  if (s === "blobs") {
    const discs: [number, number, number][] = [[6, 6, 4], [16, 5, 3], [17, 16, 5], [5, 17, 3.5], [11, 11, 1.5]];
    for (const [cx, cy, r] of discs) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) set(x, y);
  } else if (s === "spiral") {
    let x = 1, y = 1, dir = 0, len = 21, k = 0;
    const D = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    set(x, y);
    while (len > 0) {
      for (let i = 0; i < len; i++) { x += D[dir][0]; y += D[dir][1]; set(x, y); }
      dir = (dir + 1) % 4; k++;
      if (k >= 3 && k % 2 === 1) len -= 2;
    }
  } else if (s === "comb") {
    for (let x = 1; x < 23; x++) set(x, 2);
    for (let x = 1; x < 23; x += 3) for (let y = 3; y < 22; y++) set(x, y);
  } else {
    for (let x = 1; x < 23; x++) { set(x, 11); set(x, 12); }
  }
  return m;
}

/** A grey-level landscape with three basins (levels 0..15) and markers at their minima. */
function landscape(): { f: Int32Array; markers: Int32Array } {
  const f = new Int32Array(N * N), mk = new Int32Array(N * N);
  const c: [number, number][] = [[5, 6], [18, 5], [12, 18]];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const d = Math.min(...c.map(([cx, cy]) => Math.hypot(x - cx, y - cy)));
    f[y * N + x] = Math.min(15, Math.floor(d * 1.3 + 0.5 * Math.sin(x * 0.9) + 0.5));
  }
  c.forEach(([cx, cy], i) => { mk[cy * N + cx] = i + 1; f[cy * N + cx] = 0; });
  return { f, markers: mk };
}

const DP_R = 10, DP_C = 16;

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const m0 = host.dataset.mode as Mode | undefined;
  const start = (): State => ({ ...DEFAULT, ...(m0 && ["label", "wavefront", "flood"].includes(m0) ? { mode: m0 } : {}), ...(p0?.s ?? {}) });
  const st = start();
  let mask = shapeMask(st.shape);

  const main = htmlEl("div", "pl-main", "", host);
  const svg = svgEl("svg", { viewBox: "0 0 660 485", role: "img", "aria-label": "parallel labelling and flooding" }, main);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pl-lab", "presets", panel);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); mask = shapeMask(st.shape); draw(); } })));
  const labBox = htmlEl("div", "", "", panel);
  htmlEl("div", "pl-lab", "algorithm", labBox);
  const ALGOS: [Algo, string][] = [["seq", "sequential union–find"], ["prop", "label propagation"], ["jump", "propagation + pointer jumping"]];
  const algoBtns = presets(labBox, ALGOS.map(([a, l]) => ({ label: l, apply: () => { st.algo = a; st.step = 9999; draw(); } })));
  htmlEl("div", "pl-lab", "image (click pixels to edit)", labBox);
  const SHAPES: Shape[] = ["blobs", "spiral", "comb", "bar"];
  const shapeBtns = presets(labBox, SHAPES.map(s => ({ label: s, apply: () => { st.shape = s; mask = shapeMask(s); st.step = 9999; draw(); } })));
  const stepRow = htmlEl("div", "qb-presets", "", panel);
  button(stepRow, "first", () => { st.step = 0; draw(); });
  button(stepRow, "back", () => { st.step = Math.max(0, Math.min(st.step, nFrames - 1) - 1); draw(); });
  button(stepRow, "step", () => { st.step = Math.min(nFrames - 1, st.step + 1); draw(); });
  button(stepRow, "+10", () => { st.step = Math.min(nFrames - 1, Math.min(st.step, nFrames - 1) + 10); draw(); });
  button(stepRow, "all", () => { st.step = 9999; draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); mask = shapeMask(st.shape); pBtns.forEach(b => b.classList.remove("active")); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  let nFrames = 1;
  function txt(x: number, y: number, s: string, size = 18, anchor = "start", fill = "var(--qb-fg)"): SVGTextElement {
    const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": size }, svg);
    t.style.fill = fill; t.textContent = s; return t;
  }

  function draw(): void {
    svg.replaceChildren();
    labBox.classList.toggle("pl-hidden", st.mode !== "label");
    if (st.mode === "label") drawLabel(); else if (st.mode === "wavefront") drawWavefront(); else drawFlood();
  }

  function grid(values: (i: number) => string, ox: number, oy: number, cs: number, mark?: (i: number) => boolean, onClick?: (i: number) => void): void {
    for (let i = 0; i < N * N; i++) {
      const x = i % N, y = Math.floor(i / N);
      const r = svgEl("rect", { x: ox + x * cs, y: oy + y * cs, width: cs - 0.6, height: cs - 0.6 }, svg);
      r.style.fill = values(i);
      if (mark?.(i)) { r.style.stroke = "#d62728"; r.style.strokeWidth = "2"; }
      if (onClick) { r.style.cursor = "pointer"; r.addEventListener("pointerdown", () => onClick(i)); }
    }
  }

  function drawLabel(): void {
    algoBtns.forEach((b, i) => b.classList.toggle("active", ALGOS[i][0] === st.algo));
    shapeBtns.forEach((b, i) => b.classList.toggle("active", SHAPES[i] === st.shape));
    const cs = 18, ox = 10, oy = 30;
    const fgN = mask.reduce((a, b) => a + b, 0);
    const toggle = (i: number) => { mask[i] = mask[i] ? 0 : 1; st.step = 9999; draw(); };
    const col = (l: number) => l ? toCss(labelColor(l)) : "var(--qb-panel)";
    const lanes = 1e4, rate = 1e9, launch = 10e-6, cpuStep = 5e-9, sc = 1024 / N;
    if (st.algo === "seq") {
      nFrames = 1;
      const L = label2d(mask, N, N, 4);
      grid(i => col(L.data[i]), ox, oy, cs, undefined, toggle);
      txt(ox, 22, `sequential two-pass labelling: ${L.count} components`, 18);
      const t = 2 * N * N * cpuStep;
      ro.textContent = `two raster passes over all ${N * N} pixels,\none pixel per step: ${2 * N * N} steps\n` +
        `model CPU time   ${(t * 1e6).toFixed(1)} µs\n1024×1024:       ${(t * sc * sc * 1e3).toFixed(1)} ms`;
      return;
    }
    const r = P.labelPropagation(mask, N, N, 4, st.algo === "jump");
    nFrames = r.frames.length;
    const k = Math.min(st.step, nFrames - 1), cur = r.frames[k], prev = k > 0 ? r.frames[k - 1] : null;
    grid(i => col(cur[i]), ox, oy, cs, i => !!prev && prev[i] !== cur[i], toggle);
    txt(ox, 22, `pass ${k} of ${r.passes}${k === 0 ? " (start: every pixel its own label)" : ": changed pixels outlined"}`, 18);
    const gpu = (passes: number, px: number) => (passes + 1) * (launch + px / (lanes * rate));
    const tg = gpu(r.passes, N * N), tc = 2 * N * N * cpuStep;
    ro.textContent =
      `${st.algo === "jump" ? "propagation + pointer jumping" : "label propagation"}\n` +
      `passes (depth)   ${r.passes} (+1 to see nothing changed)\n` +
      `pixel updates    ${r.work} (${fgN} pixels per pass)\n` +
      `model GPU        ${(tg * 1e6).toFixed(0)} µs (launch ${launch * 1e6} µs per pass)\n` +
      `sequential CPU   ${(tc * 1e6).toFixed(1)} µs\n` +
      (st.algo === "jump" ? `1024×1024: CPU ≈ ${(tc * sc * sc * 1e3).toFixed(1)} ms; the passes grow\n  much more slowly than the path length`
        : `1024×1024, same shape scaled up:\n  passes ≈ ${Math.round(r.passes * sc)}, GPU ≈ ${(gpu(r.passes * sc, 1024 * 1024) * 1e3).toFixed(1)} ms, CPU ≈ ${(tc * sc * sc * 1e3).toFixed(1)} ms`);
  }

  function drawWavefront(): void {
    const steps = P.wavefront(DP_R, DP_C);
    nFrames = steps.length + 1;
    const k = Math.min(st.step, nFrames - 1);
    const cs = 32, ox = 20, oy = 34;
    const when = new Int32Array(DP_R * DP_C);
    steps.forEach((c, d) => c.forEach(i => { when[i] = d; }));
    txt(ox, 24, `DP table ${DP_R}×${DP_C}: cell (i, j) needs (i−1, j), (i, j−1), (i−1, j−1)`, 18);
    for (let i = 0; i < DP_R * DP_C; i++) {
      const r = Math.floor(i / DP_C), c = i % DP_C, d = when[i];
      const rect = svgEl("rect", { x: ox + c * cs, y: oy + r * cs, width: cs - 2, height: cs - 2, rx: 3 }, svg);
      rect.style.fill = d < k - 1 ? toCss(viridis(0.25 + 0.6 * d / steps.length)) : d === k - 1 ? "#ff9f1c" : "var(--qb-panel)";
      if (d <= k - 1) txt(ox + c * cs + cs / 2 - 1, oy + r * cs + cs / 2 + 6, String(d), 15, "middle", d === k - 1 ? "#000" : "#fff");
    }
    // busy lanes per step
    const by = oy + DP_R * cs + 20, bh = 70, bw = (DP_C * cs) / steps.length;
    const mx = Math.max(...steps.map(s => s.length));
    steps.forEach((s, d) => {
      const h = bh * s.length / mx;
      const r = svgEl("rect", { x: ox + d * bw, y: by + bh - h, width: bw - 2, height: h }, svg);
      r.style.fill = d === k - 1 ? "#ff9f1c" : "#5b8ff9";
    });
    txt(ox, by + bh + 24, `busy lanes per step (max ${mx})`, 18);
    const cells = DP_R * DP_C;
    ro.textContent =
      (k === 0 ? "start: nothing computed\n" : `step ${k}: anti-diagonal i + j = ${k - 1}, ${steps[k - 1].length} cells in parallel\n`) +
      `steps (depth)  ${steps.length} = rows + cols − 1\n` +
      `cells (work)   ${cells}\n` +
      `average parallelism ${(cells / steps.length).toFixed(2)}\n` +
      `a GPU with 10 000 lanes uses at most ${mx}`;
  }

  function drawFlood(): void {
    const { f, markers } = landscape();
    const lev = P.levelFlood(f, N, N, markers), q = P.queueFlood(f, N, N, markers);
    nFrames = lev.steps.length + 1;
    const k = Math.min(st.step, nFrames - 1);
    const lab = Int32Array.from(markers), newest = new Set<number>();
    const ref = lev.labels;
    for (let s = 0; s < k; s++) for (const p of lev.steps[s]) { lab[p] = ref[p]; if (s === k - 1) newest.add(p); }
    const cs = 16, ox = 10, oy = 30;
    txt(ox, 22, `level-synchronous flooding, step ${k} of ${lev.steps.length}`, 18);
    grid(i => {
      if (lab[i]) return toCss(labelColor(lab[i]));
      const g = Math.round(235 - f[i] * 11); return `rgb(${g},${g},${g})`;
    }, ox, oy, cs, i => newest.has(i));
    // pixels per step
    const bx = 410, by = 40, bh = 300, mx = Math.max(...lev.steps.map(s => s.length));
    txt(bx, by - 12, "pixels per step", 18);
    const bw = 220 / lev.steps.length;
    lev.steps.forEach((s, d) => {
      const h = bh * s.length / mx;
      const r = svgEl("rect", { x: bx + d * bw, y: by + bh - h, width: Math.max(1, bw - 1), height: h }, svg);
      r.style.fill = d === k - 1 ? "#ff9f1c" : "#5b8ff9";
    });
    txt(bx, by + bh + 24, `${lev.steps.length} steps, max ${mx}`, 18);
    txt(bx, by + bh + 50, `priority queue: ${q.steps.length} steps of 1`, 18);
    let same = 0; for (let i = 0; i < N * N; i++) if (q.labels[i] === lev.labels[i]) same++;
    ro.textContent =
      `priority queue: ${q.steps.length} steps (one pixel each)\n` +
      `level-synchronous: ${lev.steps.length} steps, ${(q.steps.length / lev.steps.length).toFixed(1)} pixels per step\n` +
      `same label for ${same} of ${N * N} pixels\n(differences sit on ties and plateaus)`;
  }

  draw();
}

autoMount(NAME, mount);
