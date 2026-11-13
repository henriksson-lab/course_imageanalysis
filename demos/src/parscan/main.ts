// Data-parallel patterns, step by step (lib/parallel.ts): tree reduction, Hillis–Steele and
// Blelloch scans, stream compaction, and a histogram with atomic additions (contention and
// privatisation). Simulated: every row is one parallel step; arrows are the additions done
// in that step. Counters: steps (depth) and additions (work) against the sequential loop.
//
// <div data-demo="parscan" data-mode="reduce|scan-hs|scan-blelloch|compact|histogram"
//      data-preset="..." data-modes="reduce,scan-hs,..."></div>
// (data-mode="scan" is accepted as "scan-hs"; data-modes limits the mode buttons.)
import { autoMount, htmlEl, presets, checkbox, button, svgEl } from "../lib/ui";
import * as P from "../lib/parallel";
import { roundF16 } from "../floatbits/fp";

const NAME = "parscan";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { flex-wrap: nowrap; gap: 1em; }
${S} .ps-main { flex: 1 1 680px; min-width: 0; }
${S} .ps-main svg { width: 100%; display: block; }
${S} .qb-panel { flex: 0 1 320px; gap: .4em; }
${S} .ps-lab { font-size: 14px; color: var(--qb-muted); margin-top: .2em; }
${S} .qb-readout { font-size: 15px; }
${S} .ps-hidden { display: none !important; }
${S} .ps-cell { cursor: pointer; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .ps-main svg { max-height: 540px; }
.reveal ${S} .qb-panel { flex: 0 0 330px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .4em; }
.reveal ${S} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${S} .ps-lab { font-size: 15px; }
.reveal ${S} .ps-note { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

type Mode = "reduce" | "scan-hs" | "scan-blelloch" | "compact" | "histogram";
const MODES: [Mode, string][] = [["reduce", "reduction"], ["scan-hs", "scan: Hillis–Steele"], ["scan-blelloch", "scan: Blelloch"], ["compact", "compaction"], ["histogram", "histogram"]];
type Pix = "uniform" | "dark" | "bimodal";
interface State { mode: Mode; values: number[]; mask: number[]; f16: boolean; order: number; step: number; pix: Pix; priv: boolean }
const V0 = [3, 1, 7, 0, 4, 1, 6, 3, 5, 2, 2, 8, 1, 4, 0, 6];
const M0 = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1];
const VF = [2000, 1.5, 3, 0.5, 2.5, 1, 3.5, 0.5, 1.5, 2, 1, 3, 0.5, 2.5, 1.5, 1];
const DEFAULT: State = { mode: "reduce", values: V0, mask: M0, f16: false, order: 0, step: 99, pix: "uniform", priv: false };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "reduce", label: "16 numbers in 4 steps", s: { mode: "reduce" } },
  { id: "hs", label: "Hillis–Steele does more work", s: { mode: "scan-hs" } },
  { id: "blelloch", label: "Blelloch: twice the steps, less work", s: { mode: "scan-blelloch" } },
  { id: "compact", label: "keep the foreground", s: { mode: "compact" } },
  { id: "dark", label: "dark image: one bin, all wait", s: { mode: "histogram", pix: "dark", priv: false } },
  { id: "private", label: "private histograms fix it", s: { mode: "histogram", pix: "dark", priv: true } },
  { id: "order", label: "the sum changes with the order", s: { mode: "reduce", values: VF, f16: true } },
];

function pixels(kind: Pix): number[] {
  let s = 4242;
  const u = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  return Array.from({ length: 64 }, () => {
    if (kind === "uniform") return Math.floor(u() * 8);
    if (kind === "dark") return u() < 0.94 ? 0 : 1;
    return u() < 0.5 ? (u() < 0.7 ? 1 : 2) : (u() < 0.7 ? 6 : 5);
  });
}

function permutation(n: number, seed: number): number[] | undefined {
  if (!seed) return undefined;
  const p = Array.from({ length: n }, (_, i) => i);
  let s = seed * 7919 + 17;
  for (let i = n - 1; i > 0; i--) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; const j = (s >>> 8) % (i + 1); [p[i], p[j]] = [p[j], p[i]]; }
  return p;
}

const fmt = (v: number) => Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(5)));

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  let m0 = host.dataset.mode as Mode | "scan" | undefined;
  if (m0 === "scan") m0 = "scan-hs";
  const allowed = (host.dataset.modes ?? "").split(",").map(s => s.trim()).filter(Boolean);
  const modes = MODES.filter(([m]) => !allowed.length || allowed.includes(m));
  const start = (): State => ({ ...DEFAULT, values: [...V0], mask: [...M0], ...(m0 && MODES.some(([m]) => m === m0) ? { mode: m0 as Mode } : {}), ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "ps-main", "", host);
  const svg = svgEl("svg", { viewBox: "0 0 690 470", role: "img", "aria-label": "parallel pattern, step by step" }, main);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "pattern", panel);
  const modeBtns = presets(panel, modes.map(([m, l]) => ({ label: l, apply: () => { st.mode = m; st.step = 99; draw(); } })));
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBoxes = new Map<Mode, HTMLElement>();
  const pBtns: HTMLButtonElement[] = [];
  for (const [m] of modes) {
    const box = htmlEl("div", "", "", panel);
    presetBoxes.set(m, box);
    pBtns.push(...presets(box, PRESETS.filter(p => p.s.mode === m).map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, { values: [...V0], mask: [...M0] }, p.s); if (p.s.values) st.values = [...p.s.values]; sync(); draw(); } }))));
  }
  const stepRow = htmlEl("div", "qb-presets", "", panel);
  button(stepRow, "first", () => { st.step = 0; draw(); });
  button(stepRow, "back", () => { st.step = Math.max(0, Math.min(st.step, nSteps - 1) - 1); draw(); });
  button(stepRow, "step", () => { st.step = Math.min(nSteps - 1, st.step + 1); draw(); });
  button(stepRow, "all", () => { st.step = 99; draw(); });
  const redBox = htmlEl("div", "", "", panel), histBox = htmlEl("div", "", "", panel);
  const f16C = checkbox(redBox, "float16 values (round after every addition)", st.f16, v => { st.f16 = v; draw(); });
  button(redBox, "shuffle the split", () => { st.order++; draw(); });
  button(redBox, "random values", () => { let s = 31 + st.order++; st.values = st.values.map(() => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) % 10; }); draw(); });
  htmlEl("div", "ps-lab", "pixels (64, values = bins 0–7)", histBox);
  const pixBtns = presets(histBox, (["uniform", "dark", "bimodal"] as Pix[]).map(k => ({ label: k, apply: () => { st.pix = k; draw(); } })));
  const privC = checkbox(histBox, "privatised: 8 blocks, each its own histogram", st.priv, v => { st.priv = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); pBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "ps-lab ps-note", "click a value: upper half +1, lower half −1 (compaction: click the mask)", panel);

  let nSteps = 1;
  function sync(): void { f16C.checked = st.f16; privC.checked = st.priv; }

  function txt(x: number, y: number, s: string, size = 18, anchor = "middle", fill = "var(--qb-fg)", weight = 400): SVGTextElement {
    const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": size, "font-weight": weight }, svg);
    t.style.fill = fill; t.textContent = s; return t;
  }
  const BW = 38, OX = 40, CX = (i: number) => OX + i * BW + BW / 2;

  function box(i: number, y: number, v: string, fill: string, h = 28, edit?: (delta: number) => void): void {
    const r = svgEl("rect", { x: OX + i * BW + 2, y, width: BW - 4, height: h, rx: 4 }, svg);
    r.style.fill = fill; r.style.stroke = "#999";
    const t = txt(CX(i), y + h / 2 + 6, v, v.length > 3 ? 15 : 18);
    if (edit) {
      for (const el of [r, t]) {
        el.classList.add("ps-cell");
        el.addEventListener("pointerdown", (ev: Event) => {
          const b = r.getBoundingClientRect();
          edit((ev as PointerEvent).clientY < b.top + b.height / 2 ? 1 : -1);
        });
      }
    }
  }

  function drawSteps(steps: P.Step[], extraLabel: (k: number) => string, highlight?: (k: number, i: number) => boolean): void {
    nSteps = steps.length;
    const show = Math.min(st.step, nSteps - 1);
    const rowH = Math.min(46, 420 / nSteps), bh = Math.min(28, rowH - 12);
    for (let k = 0; k <= show; k++) {
      const y = 30 + k * rowH, s = steps[k];
      txt(OX - 8, y + bh / 2 + 6, extraLabel(k), 15, "end", "var(--qb-muted)");
      // arrows from the previous row
      if (k > 0) for (const [tgt, src] of s.ops) {
        const yp = 30 + (k - 1) * rowH + bh;
        const l = svgEl("line", { x1: CX(src), y1: yp, x2: CX(tgt), y2: y }, svg);
        l.style.stroke = k === show ? "#d62728" : "#bbb"; l.style.strokeWidth = k === show ? "2.2" : "1.4";
      }
      s.values.forEach((v, i) => {
        const touched = s.ops.some(([t]) => t === i);
        const fill = k === 0 ? "var(--qb-panel)" : touched ? (k === show ? "#ffd8a8" : "#fdebd3") : highlight?.(k, i) ? "#d6e6f5" : "var(--qb-bg)";
        box(i, y, fmt(v), fill, bh, k === 0 ? d => { st.values[i] = Math.max(-99, Math.min(999, st.values[i] + d)); draw(); } : undefined);
      });
    }
    ro.textContent = steps[show].label + "\n";
  }

  function draw(): void {
    svg.replaceChildren();
    modeBtns.forEach((b, i) => b.classList.toggle("active", modes[i][0] === st.mode));
    presetBoxes.forEach((box, m) => box.classList.toggle("ps-hidden", m !== st.mode));
    redBox.classList.toggle("ps-hidden", st.mode !== "reduce");
    histBox.classList.toggle("ps-hidden", st.mode !== "histogram");
    const n = st.values.length, lg = Math.log2(n);
    if (st.mode === "reduce") {
      const round = st.f16 ? roundF16 : (x: number) => x;
      const order = permutation(n, st.order);
      const r = P.reduceTree(st.values, round, order);
      drawSteps(r.steps, k => k === 0 ? "in" : `${k}`);
      const seq = P.sumSequential(st.values, round), exact = st.values.reduce((a, b) => a + b, 0);
      ro.textContent += `tree:       ${r.depth} steps, ${r.work} additions\n` +
        `sequential: ${n - 1} steps, ${n - 1} additions\n` +
        `tree sum ${fmt(r.sum)}, sequential sum ${fmt(seq)}` + (st.f16 ? `\nexact ${fmt(exact)} (float16 rounds${order ? ", split shuffled " + st.order + "×" : ""})` : "");
    } else if (st.mode === "scan-hs") {
      const r = P.scanHillisSteele(st.values);
      drawSteps(r.steps, k => k === 0 ? "in" : `${k}`);
      ro.textContent += `Hillis–Steele: ${r.depth} steps, ${r.work} additions\n= n·log₂n − (n−1) = ${n * lg - (n - 1)}\nsequential: ${n - 1} steps, ${n - 1} additions\nresult: every running total (inclusive)`;
    } else if (st.mode === "scan-blelloch") {
      const r = P.scanBlelloch(st.values);
      drawSteps(r.steps, k => k === 0 ? "in" : k <= r.upSteps ? `up ${k}` : k === r.upSteps + 1 ? "0" : `dn ${k - r.upSteps - 1}`);
      ro.textContent += `Blelloch: ${r.depth} steps, ${r.work} additions = 2(n−1)\nsequential: ${n - 1} steps, ${n - 1} additions\nresult: running totals before each element (exclusive)`;
    } else if (st.mode === "compact") drawCompact();
    else drawHistogram();
  }

  function drawCompact(): void {
    nSteps = 3;
    const show = Math.min(st.step, 2);
    const c = P.compact(st.values, st.mask);
    const rows: [string, number[] | null][] = [["value", st.values], ["keep", st.mask], ["scan", c.positions]];
    rows.forEach(([lab, vals], k) => {
      if (k > show && k > 1) return;
      const y = 40 + k * 60;
      txt(OX - 8, y + 20, lab, 15, "end", "var(--qb-muted)");
      vals!.forEach((v, i) => {
        const fill = st.mask[i] ? (k === 1 ? "#ffd8a8" : "#d6e6f5") : "var(--qb-bg)";
        box(i, y, k === 2 && !st.mask[i] ? "·" : String(v), fill, 28,
          k === 1 ? () => { st.mask[i] = st.mask[i] ? 0 : 1; draw(); } : k === 0 ? d => { st.values[i] = Math.max(-99, Math.min(999, st.values[i] + d)); draw(); } : undefined);
      });
    });
    if (show >= 2) {
      const y = 40 + 4 * 60;
      txt(OX - 8, y + 20, "out", 15, "end", "var(--qb-muted)");
      for (let i = 0; i < st.values.length; i++) if (st.mask[i]) {
        const l = svgEl("line", { x1: CX(i), y1: 40 + 2 * 60 + 28, x2: CX(c.positions[i]), y2: y }, svg);
        l.style.stroke = "#d62728"; l.style.strokeWidth = "1.8";
      }
      c.out.forEach((v, j) => box(j, y, String(v), "#d6e6f5"));
    }
    const lab = ["the values and the 0/1 mask (click to change)", "the mask: which elements to keep", "exclusive scan of the mask = output position of each kept element"][show];
    ro.textContent = `${lab}\nkept ${c.count} of ${st.values.length}; scan: ${Math.log2(16)} parallel steps\nevery kept element writes to its own slot: no conflicts`;
  }

  function drawHistogram(): void {
    pixBtns.forEach((b, i) => b.classList.toggle("active", ["uniform", "dark", "bimodal"][i] === st.pix));
    nSteps = 1;
    const px = pixels(st.pix), blocks = st.priv ? 8 : 1;
    const h = P.atomicHistogram(px, 8, blocks);
    // the 64 pixels as 8 x 8, blocks as rows
    txt(20, 24, "64 pixels (one thread each)" + (st.priv ? "; each row = one block" : ""), 18, "start");
    const cs = 34;
    px.forEach((v, i) => {
      const x = 20 + (i % 8) * cs, y = 36 + Math.floor(i / 8) * cs, g = Math.round(30 + v * 30);
      const r = svgEl("rect", { x, y, width: cs - 2, height: cs - 2 }, svg); r.style.fill = `rgb(${g},${g},${g})`;
      txt(x + cs / 2 - 1, y + cs / 2 + 6, String(v), 18, "middle", v > 3 ? "#000" : "#fff");
    });
    // queues: one column per bin, one cell per cycle
    const qx = 330, qy = 36, avail = 330;
    const q = st.priv ? Array.from({ length: 8 }, (_, b) => Math.max(...h.queues.map(qq => qq[b]))) : h.queues[0];
    const mx = Math.max(1, ...q) + h.mergeCycles, ch = Math.min(22, avail / mx);
    txt(qx, 24, st.priv ? "cycles per bin: busiest block, then merge" : "cycles per bin: updates wait in line", 18, "start");
    for (let b = 0; b < 8; b++) {
      const x = qx + b * 42;
      for (let c = 0; c < q[b]; c++) { const r = svgEl("rect", { x, y: qy + c * ch, width: 36, height: Math.max(1, ch - (ch > 4 ? 2 : 0)) }, svg); r.style.fill = "#ff9f1c"; }
      for (let c = 0; c < h.mergeCycles; c++) { const r = svgEl("rect", { x, y: qy + (q[b] + c) * ch, width: 36, height: Math.max(1, ch - (ch > 4 ? 2 : 0)) }, svg); r.style.fill = "#9467bd"; }
      txt(x + 18, qy + avail + 34, String(b), 18);
      txt(x + 18, qy + avail + 58, String(h.counts[b]), 18, "middle", "var(--qb-muted)");
    }
    txt(qx - 6, qy + avail + 34, "bin", 15, "end", "var(--qb-muted)");
    txt(qx - 6, qy + avail + 58, "count", 15, "end", "var(--qb-muted)");
    ro.textContent =
      `atomic additions: one per cycle per bin;\ndifferent bins in parallel\n` +
      (st.priv ? `private: busiest block bin ${h.cycles - h.mergeCycles} + merge ${h.mergeCycles}\n` : `global: busiest bin ${h.cycles}\n`) +
      `time: ${h.cycles} cycles (without contention: 1)`;
  }

  sync();
  draw();
}

autoMount(NAME, mount);
