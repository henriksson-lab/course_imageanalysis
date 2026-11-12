// Huffman coding step by step. An editable histogram of up to 12 symbols (drag a bar to
// change its count), the Huffman tree built one merge at a time (the two lightest roots are
// merged; the pair just merged is orange, the next pair dashed), the code table
// (count, p, −log2 p, code word, length), the entropy H against the average code length L and
// the fixed-length code, and a short message encoded with the code and decoded by walking the
// tree. The tie rule (older or newer node first among equal counts) gives different trees with
// the same average length.
// Auto-mounts on <div data-demo="huffman">. Options: data-preset="example|flat|dominant|
// dyadic|mask|residuals|cell" (default example), data-tie="older|newer", data-step="all|0"
// (start with the finished tree or with no merges; default all), data-table="0" hides the table.
import { htmlEl, svgEl, button, presets, autoMount, svgPoint } from "../lib/ui";
import { huffman, averageLength, encode, decode, TieRule, HuffmanCode } from "../lib/huffman";
import { entropyFromCounts, fixedLength, information } from "../lib/entropy";
import { rng } from "../lib/random";

const NAME = "huffman";
const YMAX = 50;
const ORANGE = "#ff7f0e", BLUE = "#1f77b4", GREEN = "#2ca02c", GREY = "#888";
const LETTERS = "ABCDEFGHIJKL".split("");

interface Preset { id: string; label: string; counts: number[]; labels?: string[]; note: string }
const PRESETS: Preset[] = [
  { id: "example", label: "worked example", counts: [8, 4, 4, 2, 2], note: "p = 0.4, 0.2, 0.2, 0.1, 0.1 (the lecture's example)." },
  { id: "flat", label: "flat: no gain", counts: [5, 5, 5, 5, 5, 5, 5, 5], note: "8 equally likely symbols: H = 3 bits, nothing to gain over a fixed 3-bit code." },
  { id: "dominant", label: "one dominant symbol", counts: [36, 2, 1, 1], note: "p(A) = 0.9: H is far below 1 bit, but every Huffman code word has at least 1 bit." },
  { id: "dyadic", label: "dyadic: L = H", counts: [16, 8, 4, 2, 2], note: "Every p is a power of 1/2: code lengths −log2 p are whole numbers and L = H exactly." },
  { id: "mask", label: "binary mask", counts: [47, 3], labels: ["0", "1"], note: "A mostly-black mask: H = 0.33 bits, Huffman 1 bit per pixel. Run lengths (or arithmetic coding) can beat 1 bit per pixel." },
  { id: "residuals", label: "prediction residuals", counts: [4, 10, 20, 32, 44, 50, 49, 38, 26, 14, 6, 6],
    labels: ["−5", "−4", "−3", "−2", "−1", "0", "+1", "+2", "+3", "+4", "+5", "big"],
    note: "Residuals of the left-neighbour predictor on the smooth synthetic image (lossless unit), −5..+5 and 'big' (|e| > 5), scaled to a maximum of 50." },
  { id: "cell", label: "cell grey levels", counts: [1, 3, 9, 19, 27, 33, 35, 28, 29, 42, 46, 50],
    labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
    note: "Grey levels of the cell micrograph in 12 bins (bin 0 = darkest), scaled to a maximum of 50." },
];

const STYLE = `
[data-demo="${NAME}"] { display: grid; grid-template-columns: minmax(280px, 400px) minmax(340px, 1fr) minmax(200px, 250px);
  grid-template-areas: "hist tree panel" "table info panel"; gap: .6em 1em; align-items: start; }
[data-demo="${NAME}"] .hf-hist { grid-area: hist; }
[data-demo="${NAME}"] .hf-tree { grid-area: tree; }
[data-demo="${NAME}"] .hf-table { grid-area: table; }
[data-demo="${NAME}"] .hf-info { grid-area: info; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .qb-panel { grid-area: panel; max-width: none; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; color: var(--qb-fg); }
[data-demo="${NAME}"] table { border-collapse: collapse; font-size: 16px; width: 100%; }
[data-demo="${NAME}"] th, [data-demo="${NAME}"] td { padding: 1px 6px; text-align: right; border-bottom: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] td.code { font-family: ui-monospace, monospace; text-align: left; }
[data-demo="${NAME}"] th { font-weight: 600; }
[data-demo="${NAME}"] .hf-msg { font-family: ui-monospace, monospace; font-size: 16px; line-height: 1.5; word-break: break-all; }
[data-demo="${NAME}"] .hf-msg .w0 { color: ${BLUE}; } [data-demo="${NAME}"] .hf-msg .w1 { color: ${ORANGE}; }
[data-demo="${NAME}"] .hf-row { display: flex; gap: .35em; flex-wrap: wrap; align-items: center; }
[data-demo="${NAME}"] .qb-btn.on { background: var(--qb-accent); color: #fff; border-color: var(--qb-accent); }
.reveal [data-demo="${NAME}"] { font-size: 18px; grid-template-columns: 360px 1fr 230px; }
.reveal [data-demo="${NAME}"] table { font-size: 15px; }
.reveal [data-demo="${NAME}"] .hf-msg { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .3em .5em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .35em; }
@media (max-width: 900px) {
  [data-demo="${NAME}"] { grid-template-columns: 1fr; grid-template-areas: "panel" "hist" "tree" "table" "info"; }
}
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const txt = (parent: Element, x: number, y: number, s: string, attrs: Record<string, string | number> = {}): SVGTextElement => {
  const t = svgEl("text", { x, y, "text-anchor": "middle", ...attrs }, parent);
  t.textContent = s;
  return t;
};

function mount(host: HTMLElement): void {
  injectStyle();
  const preset0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const tie0: TieRule = host.dataset.tie === "newer" ? "newer" : "older";
  const startAll = host.dataset.step !== "0";
  const st = {
    counts: [...preset0.counts], labels: preset0.labels ? [...preset0.labels] : LETTERS.slice(0, preset0.counts.length),
    tie: tie0, step: 0, msgSeed: 1, note: preset0.note,
  };
  let code: HuffmanCode = huffman(st.counts, st.tie);

  // ---- layout ----
  const histFig = htmlEl("figure", "hf-hist", "", host);
  htmlEl("figcaption", "", "histogram: drag a bar to change a count", histFig);
  const HW = 400, HH = 250, hl = 44, hr = 8, ht = 24, hb = 40;
  const hsvg = svgEl("svg", { viewBox: `0 0 ${HW} ${HH}`, width: "100%" }, histFig);
  const hg = svgEl("g", {}, hsvg), hitG = svgEl("g", {}, hsvg);
  let hitK = -1;
  const treeFig = htmlEl("figure", "hf-tree", "", host);
  const treeCap = htmlEl("figcaption", "", "", treeFig);
  const TW = 560, TH = 300;
  const tsvg = svgEl("svg", { viewBox: `0 0 ${TW} ${TH}`, width: "100%" }, treeFig);
  const tableBox = htmlEl("div", "hf-table", "", host);
  if (host.dataset.table === "0") tableBox.style.display = "none";
  const info = htmlEl("div", "hf-info", "", host);
  const BW = 560, BH = 104;
  const bsvg = svgEl("svg", { viewBox: `0 0 ${BW} ${BH}`, width: "100%" }, info);
  const msgBox = htmlEl("div", "hf-msg", "", info);

  const panel = htmlEl("div", "qb-panel", "", host);
  presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => loadPreset(p) })))
    .forEach((b, i) => b.classList.toggle("active", PRESETS[i] === preset0));
  const stepRow = htmlEl("div", "hf-row", "", panel);
  button(stepRow, "next merge", () => { if (st.step < code.merges.length) st.step++; draw(); });
  button(stepRow, "build all", () => { st.step = code.merges.length; draw(); });
  button(stepRow, "restart", () => { st.step = 0; draw(); });
  const tieRow = htmlEl("div", "hf-row", "", panel);
  htmlEl("span", "", "ties:", tieRow);
  const tieBtns = [
    button(tieRow, "older first", () => setTie("older")),
    button(tieRow, "newer first", () => setTie("newer")),
  ];
  const symRow = htmlEl("div", "hf-row", "", panel);
  htmlEl("span", "", "symbols:", symRow);
  button(symRow, "+", () => {
    if (st.counts.length >= 12) return;
    st.counts.push(1); st.labels.push(LETTERS[st.counts.length - 1]); rebuild(true);
  });
  button(symRow, "−", () => {
    if (st.counts.length <= 2) return;
    st.counts.pop(); st.labels.pop(); rebuild(true);
  });
  button(symRow, "new message", () => { st.msgSeed++; draw(); });
  button(panel, "reset", () => { st.tie = tie0; st.msgSeed = 1; loadPreset(preset0); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  function setTie(t: TieRule): void { st.tie = t; rebuild(false); }
  function loadPreset(p: Preset): void {
    st.counts = [...p.counts];
    st.labels = p.labels ? [...p.labels] : LETTERS.slice(0, p.counts.length);
    st.note = p.note;
    rebuild(startAll);
  }
  /** Recompute the code; keep the current step if possible (all = jump to the finished tree). */
  function rebuild(all: boolean): void {
    const wasComplete = st.step >= code.merges.length;
    code = huffman(st.counts, st.tie);
    if (all || wasComplete) st.step = code.merges.length;
    else st.step = Math.min(st.step, code.merges.length);
    draw();
  }

  // ---- histogram with draggable bars ----
  const K = () => st.counts.length;
  const bx = (i: number) => hl + (i + 0.5) * (HW - hl - hr) / K();
  const by = (c: number) => HH - hb - c / YMAX * (HH - hb - ht);
  function drawHist(): void {
    hg.replaceChildren();
    for (const v of [0, 10, 20, 30, 40, 50]) {
      svgEl("line", { x1: hl, x2: HW - hr, y1: by(v), y2: by(v), stroke: "var(--qb-grid)" }, hg);
      txt(hg, hl - 6, by(v) + 6, String(v), { "text-anchor": "end", "font-size": 16 });
    }
    const bw = 0.7 * (HW - hl - hr) / K();
    const last = st.step > 0 ? code.merges[st.step - 1] : null;
    st.counts.forEach((c, i) => {
      const merged = last && (last.a === i || last.b === i);
      svgEl("rect", { x: bx(i) - bw / 2, y: by(c), width: bw, height: Math.max(0, by(0) - by(c)), fill: merged ? ORANGE : BLUE, rx: 2 }, hg);
      txt(hg, bx(i), by(c) - 4, String(c), { "font-size": 15 });
      txt(hg, bx(i), HH - hb + 22, st.labels[i], { "font-size": K() > 8 ? 15 : 18, "font-weight": 600 });
    });
    txt(hg, 14, (ht + HH - hb) / 2, "count", { "font-size": 16, transform: `rotate(-90 14 ${(ht + HH - hb) / 2})` });
    // invisible hit areas, one per bar column (rebuilt only when the number of symbols changes,
    // so a drag keeps its pointer capture)
    if (hitK === K()) return;
    hitK = K();
    hitG.replaceChildren();
    st.counts.forEach((_, i) => {
      const hit = svgEl("rect", { x: bx(i) - (HW - hl - hr) / K() / 2, y: ht - 6, width: (HW - hl - hr) / K(), height: HH - hb - ht + 12, fill: "transparent" }, hitG);
      hit.style.cursor = "ns-resize"; hit.style.touchAction = "none";
      hit.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        try { hit.setPointerCapture(ev.pointerId); } catch { /* synthetic */ }
        const set = (e: PointerEvent) => {
          const [, y] = svgPoint(hsvg, e);
          const c = Math.max(0, Math.min(YMAX, Math.round((HH - hb - y) / (HH - hb - ht) * YMAX)));
          if (c !== st.counts[i]) { st.counts[i] = c; rebuild(false); }
        };
        set(ev);
        const up = () => { hit.removeEventListener("pointermove", set); hit.removeEventListener("pointerup", up); hit.removeEventListener("pointercancel", up); };
        hit.addEventListener("pointermove", set);
        hit.addEventListener("pointerup", up);
        hit.addEventListener("pointercancel", up);
      });
    });
  }

  // ---- tree ----
  function drawTree(): void {
    tsvg.replaceChildren();
    const M = code.merges.length;
    treeCap.textContent = code.root < 0 ? "no symbols" : `Huffman tree: ${st.step} of ${M} merges` + (st.step === M ? " (done)" : "");
    if (code.root < 0) return;
    const nodes = code.nodes;
    const height = new Map<number, number>();
    const hOf = (n: number): number => {
      const nd = nodes[n];
      const v = nd.symbol >= 0 ? 0 : 1 + Math.max(hOf(nd.left), hOf(nd.right));
      height.set(n, v);
      return v;
    };
    const H = hOf(code.root);
    // leaves in the left-to-right order of the finished tree
    const order: number[] = [];
    const walk = (n: number) => { const nd = nodes[n]; if (nd.symbol >= 0) order.push(n); else { walk(nd.left); walk(nd.right); } };
    walk(code.root);
    const xs = new Map<number, number>();
    const gap = Math.min(70, (TW - 40) / Math.max(1, order.length));
    const x0 = TW / 2 - gap * (order.length - 1) / 2;
    order.forEach((n, i) => xs.set(n, x0 + i * gap));
    const xOf = (n: number): number => {
      if (xs.has(n)) return xs.get(n)!;
      const v = (xOf(nodes[n].left) + xOf(nodes[n].right)) / 2;
      xs.set(n, v);
      return v;
    };
    const yTop = 26, yBot = TH - 40, dy = H > 0 ? (yBot - yTop) / H : 0;
    const yOf = (n: number) => yBot - height.get(n)! * dy;
    const shown = new Set<number>(order);
    for (let s = 0; s < st.step; s++) shown.add(code.merges[s].node);
    const last = st.step > 0 ? code.merges[st.step - 1] : null;
    const next = st.step < M ? code.merges[st.step] : null;
    // edges
    for (let s = 0; s < st.step; s++) {
      const m = code.merges[s];
      [m.a, m.b].forEach((c, bit) => {
        const x1 = xOf(m.node), y1 = yOf(m.node), x2 = xOf(c), y2 = yOf(c);
        svgEl("line", { x1, y1, x2, y2, stroke: last === m ? ORANGE : "var(--qb-axis)", "stroke-width": last === m ? 3 : 2 }, tsvg);
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        txt(tsvg, mx + (bit ? 9 : -9), my - 3, String(bit), { "font-size": 16, "font-weight": 700, fill: bit ? ORANGE : BLUE, class: "qb-halo" });
      });
    }
    // nodes
    for (const n of shown) {
      const nd = nodes[n], x = xOf(n), y = yOf(n);
      const hiLast = last && (last.a === n || last.b === n), hiNext = next && (next.a === n || next.b === n), isNew = last && last.node === n;
      const stroke = hiLast ? ORANGE : hiNext ? GREEN : "var(--qb-axis)";
      const attrs = { stroke, "stroke-width": hiLast || hiNext ? 3 : 1.5, fill: isNew ? "#ffe2c6" : "var(--qb-bg)", ...(hiNext ? { "stroke-dasharray": "5 3" } : {}) };
      if (nd.symbol >= 0) {
        svgEl("rect", { x: x - 17, y: y - 15, width: 34, height: 30, rx: 5, ...attrs }, tsvg);
        txt(tsvg, x, y + 6, st.labels[nd.symbol], { "font-size": st.labels[nd.symbol].length > 2 ? 14 : 18, "font-weight": 600 });
        txt(tsvg, x, y + 33, String(nd.weight), { "font-size": 15, fill: GREY });
      } else {
        svgEl("circle", { cx: x, cy: y, r: 16, ...attrs }, tsvg);
        txt(tsvg, x, y + 5, String(nd.weight), { "font-size": 15 });
      }
    }
    // counts of zero: no code
    const unused = st.counts.map((c, i) => (c === 0 ? st.labels[i] : "")).filter(Boolean);
    if (unused.length) txt(tsvg, 6, TH - 6, "count 0, no code: " + unused.join(", "), { "text-anchor": "start", "font-size": 14, fill: GREY });
  }

  // ---- table, bars, message ----
  function drawTable(done: boolean): void {
    const tot = st.counts.reduce((a, b) => a + b, 0);
    let h = "<table><tr><th>symbol</th><th>count</th><th>p</th><th>−log₂ p</th><th style='text-align:left'>code word</th><th>L</th></tr>";
    st.counts.forEach((c, i) => {
      const p = tot ? c / tot : 0;
      h += `<tr><td style="text-align:center"><b>${st.labels[i]}</b></td><td>${c}</td><td>${p.toFixed(3)}</td><td>${c ? information(p).toFixed(2) : "–"}</td>` +
        `<td class="code">${done ? code.codes[i] || "–" : "…"}</td><td>${done && code.lengths[i] ? code.lengths[i] : done ? "–" : "…"}</td></tr>`;
    });
    tableBox.innerHTML = h + "</table>";
  }

  function drawBars(done: boolean): void {
    bsvg.replaceChildren();
    const used = st.counts.filter(c => c > 0).length;
    const H = entropyFromCounts(st.counts), L = done ? averageLength(st.counts, code.lengths) : NaN, F = fixedLength(used);
    const rows: [string, number, string][] = [["entropy H", H, GREEN], ["Huffman L", L, ORANGE], [`fixed ⌈log₂ ${used}⌉`, F, BLUE]];
    const vmax = Math.max(1, F, Number.isFinite(L) ? L : 0), l = 150, r = 110;
    rows.forEach(([name, v, col], k) => {
      const y = 8 + k * 32;
      txt(bsvg, l - 8, y + 19, name, { "text-anchor": "end", "font-size": 17 });
      if (Number.isFinite(v)) svgEl("rect", { x: l, y, width: Math.max(1, (BW - l - r) * v / vmax), height: 24, fill: col, rx: 3 }, bsvg);
      txt(bsvg, l + (Number.isFinite(v) ? (BW - l - r) * v / vmax : 0) + 6, y + 19, Number.isFinite(v) ? v.toFixed(3) + " bits" : "(build the tree)", { "text-anchor": "start", "font-size": 16 });
    });
    ro.textContent = `H = ${H.toFixed(3)} bits/symbol\n` + (done ? `L = ${L.toFixed(3)} bits/symbol\nL − H = ${(L - H).toFixed(3)}` : `L: build the tree`) + `\nfixed: ${F} bits/symbol`;
  }

  function drawMessage(done: boolean): void {
    const tot = st.counts.reduce((a, b) => a + b, 0);
    msgBox.replaceChildren();
    if (!done || tot === 0) { msgBox.textContent = "Message: finish the tree (build all) to get the code words."; return; }
    // a 16-symbol message drawn from the histogram (seeded)
    const r = rng(st.msgSeed * 7919);
    const msg: number[] = [];
    for (let k = 0; k < 16; k++) {
      let u = r.unif() * tot, s = 0;
      while (u >= st.counts[s]) { u -= st.counts[s]; s++; }
      msg.push(s);
    }
    const bits = encode(msg, code.codes), dec = decode(bits, code);
    const used = st.counts.filter(c => c > 0).length;
    const sep = st.labels.some(l => l.length > 1) ? " " : "";
    htmlEl("div", "", "message: " + msg.map(s => st.labels[s]).join(sep), msgBox);
    const line = htmlEl("div", "", "", msgBox);
    line.append("bits: ");
    msg.forEach((s, k) => htmlEl("span", "w" + (k % 2), code.codes[s], line));
    htmlEl("div", "", `${bits.length} bits (fixed code: ${16 * fixedLength(used)})  ·  decoded by walking the tree: ${dec.map(s => st.labels[s]).join(sep)}`, msgBox);
  }

  function draw(): void {
    const done = st.step >= code.merges.length;
    tieBtns.forEach((b, i) => b.classList.toggle("on", (i === 0) === (st.tie === "older")));
    drawHist(); drawTree(); drawTable(done); drawBars(done); drawMessage(done);
    note.textContent = st.note + (code.merges.length && st.step < code.merges.length ? " Green dashed: the two lightest roots, merged next." : "");
  }

  if (!startAll) st.step = 0; else st.step = code.merges.length;
  draw();
}

autoMount(NAME, mount);
