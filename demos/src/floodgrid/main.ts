// floodgrid: Meyer's priority-queue flooding step by step on a small grid (watershed day,
// unit ws-flooding). Cells show their height (grey: higher = darker); click a cell to raise
// (upper half) or lower (lower half) it, or to place a marker A–D. "step" pops one pixel:
// it is outlined in red, its labelled neighbours dashed, and the decision is written out.
// The queue is listed next to the grid in pop order (height, arrival number, cell), so a
// hand trace can be checked line by line. Options: 4/8 neighbours, watershed line on/off,
// FIFO/LIFO among equal heights. The trace comes from lib/watershed.ts (watershedTrace).
// Options: data-preset (two | plateau | minima | slope | hand), data-conn (4 | 8), data-lines (0 | 1),
// data-step (start after this many pops).
import { autoMount, htmlEl, button, presets, checkbox, svgEl } from "../lib/ui";
import { watershedTrace, minimaMarkers, Trace, QItem } from "../lib/watershed";
import { labelColor, toCss, RGB } from "../lib/colour";

const NAME = "floodgrid";
const CELL = 52;

const G = (s: string) => s.trim().split("\n").map(r => r.trim().split(/\s+/).map(Number));
const TWO = G(`
9 8 7 6 6 7 8 9
8 5 4 5 6 5 6 8
7 4 1 3 7 3 4 7
7 4 2 4 8 2 3 6
8 5 4 5 7 1 2 6
8 6 5 6 6 3 3 7
9 7 6 7 7 5 5 8
9 9 8 8 8 7 8 9`);
const PLATEAU = G(`
6 6 6 6 6 6 6 6 6
6 3 3 3 3 3 3 3 6
1 3 3 3 3 3 3 3 1
6 3 3 3 3 3 3 3 6
6 6 6 6 6 6 6 6 6`);
const NOISY = G(`
5 6 4 5 7 6 5 6
4 3 5 6 6 4 3 5
5 4 6 7 5 5 4 6
6 5 7 8 6 3 5 7
5 2 6 7 6 4 6 6
4 3 5 6 5 5 4 5
5 4 3 5 6 4 2 4
6 5 4 6 7 5 4 5`);
/** The 5 × 5 grid of the by-hand exercise (markers A at row 1, column 1 and B at row 4, column 4). */
export const HAND = G(`
3 2 3 5 6
2 1 2 4 5
3 2 4 3 2
5 4 3 2 1
6 5 3 1 0`);

interface Preset { id: string; label: string; grid: number[][]; marks: [number, number, number][] | "minima"; conn: 4 | 8; lines: boolean; lifo?: boolean }
const PRESETS: Preset[] = [
  { id: "two", label: "two basins", grid: TWO, marks: [[2, 2, 1], [4, 5, 2]], conn: 4, lines: true },
  { id: "plateau", label: "plateau between two markers", grid: PLATEAU, marks: [[2, 0, 1], [2, 8, 2]], conn: 4, lines: true },
  { id: "minima", label: "every minimum is a marker", grid: NOISY, marks: "minima", conn: 8, lines: true },
  { id: "slope", label: "a marker that is not a minimum", grid: TWO, marks: [[2, 2, 1], [1, 6, 2]], conn: 4, lines: true },
  { id: "hand", label: "5 × 5 (exercise)", grid: HAND, marks: [[1, 1, 1], [4, 4, 2]], conn: 4, lines: true },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em 1.1em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .fg-grid { flex: 0 1 470px; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .fg-grid svg { width: 100%; touch-action: none; cursor: pointer; }
.qb-demo[data-demo="${NAME}"] .fg-queue { flex: 0 1 190px; min-width: 150px; }
.qb-demo[data-demo="${NAME}"] .fg-queue table { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 16px; width: 100%; }
.qb-demo[data-demo="${NAME}"] .fg-queue td, .qb-demo[data-demo="${NAME}"] .fg-queue th { padding: 1px 6px; text-align: right; }
.qb-demo[data-demo="${NAME}"] .fg-queue th { font-weight: 600; color: var(--qb-muted); border-bottom: 1px solid var(--qb-grid, #ccc); }
.qb-demo[data-demo="${NAME}"] .fg-queue tr.next td { font-weight: 700; color: var(--qb-c2, #d62728); }
.qb-demo[data-demo="${NAME}"] .fg-queue tr.new td { background: rgba(255, 200, 60, .25); }
.qb-demo[data-demo="${NAME}"] .fg-qtitle { font-weight: 600; margin-bottom: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 340px; max-width: 460px; gap: .45em; }
.qb-demo[data-demo="${NAME}"] .fg-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .fg-decision { font-size: 17px; min-height: 2.6em; padding: .3em .5em; border-left: 4px solid var(--qb-accent); background: var(--qb-panel, #f6f7f9); }
.qb-demo[data-demo="${NAME}"] .qb-btn:disabled { opacity: .45; cursor: default; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .fg-grid { flex-basis: 430px; }
.reveal .qb-demo[data-demo="${NAME}"] .fg-queue table { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const letter = (k: number) => (k > 0 ? String.fromCharCode(64 + k) : "line");
const mix = (c: RGB, t: number): RGB => [c[0] * (1 - t) + t, c[1] * (1 - t) + t, c[2] * (1 - t) + t];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    rows: 8, cols: 8, f: [] as number[], marks: new Int32Array(0), conn: 4 as 4 | 8, lines: true, lifo: false,
    edit: 0, k: 0,   // edit: 0 = height ±, 1..4 = marker A..D, 5 = remove marker
  };
  let tr: Trace | null = null, timer: number | null = null;

  const gridBox = htmlEl("div", "fg-grid", "", host);
  const svg = svgEl("svg", { role: "img", "aria-label": "grid of heights with markers" }, gridBox);
  const qBox = htmlEl("div", "fg-queue", "", host);
  htmlEl("div", "fg-qtitle", "queue (pop order)", qBox);
  const qTable = htmlEl("table", "", "", qBox);
  const panel = htmlEl("div", "qb-panel", "", host);

  const ctl = htmlEl("div", "qb-presets", "", panel);
  const startBtn = button(ctl, "⏮ start", () => { play(false); st.k = 0; draw(); });
  const backBtn = button(ctl, "◀ back", () => { play(false); st.k = Math.max(0, st.k - 1); draw(); });
  const stepBtn = button(ctl, "step ▶", () => { play(false); st.k = Math.min(nSteps(), st.k + 1); draw(); });
  const levelBtn = button(ctl, "next level ▶▶", () => { play(false); nextLevel(); });
  const playBtn = button(ctl, "play", () => play());
  const endBtn = button(ctl, "run to end", () => { play(false); st.k = nSteps(); draw(); });
  const decision = htmlEl("div", "fg-decision", "", panel);
  const ro = htmlEl("div", "qb-readout", "", panel);

  const opts = htmlEl("div", "qb-presets", "", panel);
  const c8 = checkbox(opts, "8 neighbours", false, v => { st.conn = v ? 8 : 4; retrace(); });
  const cl = checkbox(opts, "watershed line", true, v => { st.lines = v; retrace(); });
  const cf = checkbox(opts, "LIFO ties", false, v => { st.lifo = v; retrace(); });

  htmlEl("div", "fg-lab", "a click on a cell", panel);
  const editBtns = presets(panel, ["height + / −", "marker A", "B", "C", "D", "no marker"].map((label, i) => ({ label, apply: () => { st.edit = i; draw(); } })));
  editBtns.forEach((b, i) => { if (i >= 1 && i <= 4) b.style.boxShadow = `inset 4px 0 0 ${toCss(labelColor(i, [1, 1, 1]))}`; });
  const mrow = htmlEl("div", "qb-presets", "", panel);
  button(mrow, "all regional minima as markers", () => { setMinima(); retrace(); });
  button(mrow, "clear markers", () => { st.marks.fill(0); retrace(); });
  const srow = htmlEl("div", "qb-presets", "", panel);
  htmlEl("span", "fg-lab", "size", srow);
  for (const n of [5, 8, 10, 12]) button(srow, `${n}×${n}`, () => resize(n, n));

  htmlEl("div", "fg-lab", "examples", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(htmlEl("div", "qb-presets", "", panel), "reset", () => load(p0));

  const nSteps = () => tr?.steps.length ?? 0;

  function setMinima(): void {
    const mk = minimaMarkers(st.f, st.cols, st.rows, st.conn);
    st.marks = Int32Array.from(mk.data, v => (v > 0 ? ((v - 1) % 26) + 1 : 0));
  }

  function load(p: Preset): void {
    play(false);
    const first = p === p0;
    st.rows = p.grid.length; st.cols = p.grid[0].length; st.f = p.grid.flat();
    st.conn = first && host.dataset.conn ? (Number(host.dataset.conn) === 8 ? 8 : 4) : p.conn;
    st.lines = first && host.dataset.lines ? host.dataset.lines !== "0" : p.lines;
    st.lifo = !!p.lifo;
    c8.checked = st.conn === 8; cl.checked = st.lines; cf.checked = st.lifo;
    st.marks = new Int32Array(st.rows * st.cols);
    if (p.marks === "minima") setMinima();
    else for (const [r, c, l] of p.marks) st.marks[r * st.cols + c] = l;
    st.edit = 0;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    retrace();
  }

  function resize(rows: number, cols: number): void {
    const f = new Array(rows * cols).fill(5), m = new Int32Array(rows * cols);
    for (let r = 0; r < Math.min(rows, st.rows); r++) for (let c = 0; c < Math.min(cols, st.cols); c++) {
      f[r * cols + c] = st.f[r * st.cols + c]; m[r * cols + c] = st.marks[r * st.cols + c];
    }
    st.rows = rows; st.cols = cols; st.f = f; st.marks = m;
    retrace();
  }

  function retrace(): void {
    play(false);
    tr = watershedTrace(st.f, st.cols, st.rows, st.marks, { conn: st.conn, lines: st.lines, lifo: st.lifo });
    st.k = 0;
    draw();
  }

  function nextLevel(): void {
    if (!tr || st.k >= nSteps()) return;
    const target = tr.steps[st.k].level;
    let k = st.k + 1;
    while (k < nSteps() && tr.steps[k].level <= target) k++;
    st.k = k; draw();
  }

  function play(on?: boolean): void {
    const want = on ?? timer === null;
    if (timer !== null) { clearInterval(timer); timer = null; }
    if (want) {
      if (st.k >= nSteps()) st.k = 0;
      timer = window.setInterval(() => {
        if (!host.isConnected || st.k >= nSteps()) { play(false); return; }
        st.k++; draw();
      }, 650);
    }
    playBtn.classList.toggle("active", timer !== null);
  }

  const rc = (p: number) => `(${Math.floor(p / st.cols)}, ${p % st.cols})`;

  function draw(): void {
    if (!tr) return;
    editBtns.forEach((b, i) => b.classList.toggle("active", i === st.edit));
    const R = st.rows, C = st.cols, c = CELL, ox = 30, oy = 26;
    const Wv = ox + C * c + 6, Hv = oy + R * c + 6;
    svg.setAttribute("viewBox", `0 0 ${Wv} ${Hv}`);
    svg.replaceChildren();
    const step = st.k > 0 ? tr.steps[st.k - 1] : null;
    const labels = step ? step.labels : tr.init.labels;
    const queue: QItem[] = step ? step.queue : tr.init.queue;
    const queued = new Map<number, QItem>();
    for (const q of queue) queued.set(q.p, q);
    let fmax = 1;
    for (const v of st.f) fmax = Math.max(fmax, v);
    for (let k = 0; k < C; k++) { const t = svgEl("text", { x: ox + (k + 0.5) * c, y: oy - 8, "text-anchor": "middle" }, svg); t.textContent = String(k); t.style.cssText = "font-size:16px;fill:var(--qb-muted,#777)"; }
    for (let k = 0; k < R; k++) { const t = svgEl("text", { x: ox - 8, y: oy + (k + 0.5) * c + 6, "text-anchor": "end" }, svg); t.textContent = String(k); t.style.cssText = "font-size:16px;fill:var(--qb-muted,#777)"; }
    for (let i = 0; i < R * C; i++) {
      const r = Math.floor(i / C), q = i % C, x = ox + q * c, y = oy + r * c, L = labels[i], v = st.f[i];
      let fill: string, txt = "#111";
      if (L > 0) fill = toCss(mix(labelColor(L, [1, 1, 1]), 0.35));
      else if (L < 0) { fill = "#111"; txt = "#fff"; }
      else { const g = 1 - 0.48 * v / fmax; fill = toCss([g, g, g]); }
      svgEl("rect", { x, y, width: c, height: c, fill, stroke: "var(--qb-grid, #bbb)", "stroke-width": 1 }, svg);
      const t = svgEl("text", { x: x + c / 2, y: y + c / 2 + 8, "text-anchor": "middle" }, svg);
      t.textContent = String(v); t.style.cssText = `font-size:23px;font-weight:600;fill:${txt}`;
      if (st.marks[i]) {
        svgEl("rect", { x: x + 2.5, y: y + 2.5, width: c - 5, height: c - 5, fill: "none", stroke: toCss(labelColor(st.marks[i], [1, 1, 1])), "stroke-width": 5 }, svg);
        const m = svgEl("text", { x: x + 6, y: y + 17 }, svg); m.textContent = letter(st.marks[i]); m.style.cssText = "font-size:15px;font-weight:700;fill:#111";
      }
      const qi = queued.get(i);
      if (qi) { const a = svgEl("text", { x: x + c - 4, y: y + c - 5, "text-anchor": "end" }, svg); a.textContent = "#" + qi.arrival; a.style.cssText = `font-size:13px;fill:${txt === "#fff" ? "#ffd" : "#7a4b00"};font-weight:600`; }
    }
    svgEl("rect", { x: ox, y: oy, width: C * c, height: R * c, fill: "none", stroke: "var(--qb-axis, #444)", "stroke-width": 1.5 }, svg);
    if (step) {
      for (const p of step.nbPixels) {
        const x = ox + (p % C) * c, y = oy + Math.floor(p / C) * c;
        svgEl("rect", { x: x + 6, y: y + 6, width: c - 12, height: c - 12, fill: "none", stroke: "#1f5fa8", "stroke-width": 3, "stroke-dasharray": "5 4", rx: 3 }, svg);
      }
      const p = step.item.p, x = ox + (p % C) * c, y = oy + Math.floor(p / C) * c;
      svgEl("rect", { x: x + 1.5, y: y + 1.5, width: c - 3, height: c - 3, fill: "none", stroke: "#d62728", "stroke-width": 4.5 }, svg);
    }
    // queue table
    qTable.replaceChildren();
    const head = htmlEl("tr", "", "", qTable);
    for (const s of ["height", "#", "cell"]) htmlEl("th", "", s, head);
    const pushed = new Set(step?.pushed ?? []);
    const MAXQ = 14;
    queue.slice(0, MAXQ).forEach((q, j) => {
      const tr_ = htmlEl("tr", j === 0 ? "next" : pushed.has(q.p) ? "new" : "", "", qTable);
      htmlEl("td", "", String(q.value), tr_); htmlEl("td", "", "#" + q.arrival, tr_); htmlEl("td", "", rc(q.p), tr_);
    });
    if (queue.length > MAXQ) { const t = htmlEl("tr", "", "", qTable); const d = htmlEl("td", "", `+ ${queue.length - MAXQ} more`, t); d.colSpan = 3; }
    if (!queue.length) { const t = htmlEl("tr", "", "", qTable); const d = htmlEl("td", "", "(empty)", t); d.colSpan = 3; }
    // decision text
    const nMark = new Set(Array.from(st.marks).filter(v => v > 0)).size;
    if (!nMark) decision.textContent = "No markers: choose \"marker A\" and click a cell, or \"all regional minima as markers\".";
    else if (!step) decision.textContent = `Start: the unlabelled neighbours of the markers are queued with their heights (#: arrival order). Step pops the lowest; ties: ${st.lifo ? "last in, first out" : "first in, first out"}.`;
    else {
      const it = step.item, nb = step.nbLabels.map(letter).join(" and ");
      const what = step.label === 0 ? `${nb} meet → watershed line` : st.lines ? `all labelled neighbours are ${letter(step.label)} → ${letter(step.label)}` : `queued by ${letter(it.from)} → ${letter(step.label)}`;
      const pu = step.pushed.length ? ` Queued: ${step.pushed.map(p => `${rc(p)} #${queued.get(p)?.arrival ?? ""}`).join(", ")}.` : "";
      decision.textContent = `Pop ${st.k}: ${rc(it.p)}, height ${it.value}, #${it.arrival}. ${what}.${pu}`;
    }
    const nLine = Array.from(labels).filter(v => v < 0).length;
    const left = R * C - Array.from(labels).filter(v => v !== 0).length;
    ro.textContent = `pop ${st.k} of ${nSteps()} · water level ${step ? step.level : "–"} · queue ${queue.length} · line pixels ${nLine} · unlabelled ${left}`;
    startBtn.disabled = backBtn.disabled = st.k === 0;
    stepBtn.disabled = levelBtn.disabled = endBtn.disabled = st.k >= nSteps();
  }

  // editing
  svg.addEventListener("pointerdown", ev => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(ctm.inverse());
    const q = Math.floor((p.x - 30) / CELL), r = Math.floor((p.y - 26) / CELL);
    if (q < 0 || r < 0 || q >= st.cols || r >= st.rows) return;
    ev.preventDefault();
    const i = r * st.cols + q;
    if (st.edit === 0) {
      const up = (p.y - 26) / CELL - r < 0.5;
      st.f[i] = Math.max(0, Math.min(9, st.f[i] + (up ? 1 : -1)));
    } else if (st.edit === 5) st.marks[i] = 0;
    else st.marks[i] = st.marks[i] === st.edit ? 0 : st.edit;
    retrace();
  });

  load(p0);
  if (host.dataset.step) { st.k = Math.max(0, Math.min(nSteps(), Number(host.dataset.step) || 0)); draw(); }
}

autoMount(NAME, mount);
