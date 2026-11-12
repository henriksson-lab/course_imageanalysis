// The Haar transform one level at a time. An editable 8- or 16-sample signal (drag the bar
// tops); "step" replaces neighbouring pairs by their average s and difference d, keeps the
// differences and repeats on the averages, until one overall average is left. The lower
// panel shows the coefficient vector in Mallat order [s | d_J | … | d_1], coloured by level.
// Normalisation 1/2 (averages, as by hand) or 1/√2 (orthonormal: energy is preserved).
// Click a coefficient to drop it (or "keep K largest"): the reconstruction is drawn over the
// signal, with the squared error and the sum of the dropped squared coefficients (equal in
// the orthonormal case). "Show basis" draws the transform matrix (rows = basis vectors).
// Auto-mounts on <div data-demo="haarstep">. Optional: data-n="8|16", data-preset="board|spike|step|ramp|const",
// data-norm="half|orth", data-levels="0..4" (levels already done at start).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { haarStates, haarInverse, HaarNorm, maxLevels } from "../lib/wavelet";
import { diverging, divergingT } from "../lib/colour";

const NAME = "haarstep";
const VW = 760, VH = 560;
const SIG = { l: 70, r: 740, t: 34, b: 250 };
const COEF = { l: 70, r: 740, t: 330, b: 500 };
const LEVEL_COL = ["#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b"];
const REC = "#ff7f0e";

interface Preset { id: string; label: string; x8: number[]; x16: number[] }
const PRESETS: Preset[] = [
  { id: "board", label: "board example", x8: [4, 6, 10, 12, 8, 6, 5, 5], x16: [4, 6, 10, 12, 8, 6, 5, 5, 3, 3, 7, 9, 12, 12, 6, 2] },
  { id: "spike", label: "spike", x8: [0, 0, 0, 0, 8, 0, 0, 0], x16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0] },
  { id: "step", label: "step", x8: [2, 2, 2, 10, 10, 10, 10, 10], x16: [2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] },
  { id: "ramp", label: "ramp", x8: [1, 2, 3, 4, 5, 6, 7, 8], x16: [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5] },
  { id: "const", label: "constant", x8: [5, 5, 5, 5, 5, 5, 5, 5], x16: Array(16).fill(5) },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .hs-plot { flex: 1 1 520px; min-width: 0; max-width: 760px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .45em; }
[data-demo="${NAME}"] .hs-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] svg text { font-size: 19px; }
[data-demo="${NAME}"] svg .num { font-size: 16px; font-family: ui-monospace, monospace; }
[data-demo="${NAME}"] svg .small { font-size: 16px; }
[data-demo="${NAME}"] svg .coef { cursor: pointer; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .3em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const fmt = (v: number) => {
  if (Math.abs(v) < 5e-10) return "0";
  if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
  return v.toFixed(2).replace(/0$/, "").replace("-", "−");
};

/** Level of entry i after k levels on n samples: 0 = current averages, j = details of level j. */
function levelAt(i: number, n: number, k: number): number {
  if (i < n >> k) return 0;
  for (let j = k; j >= 1; j--) if (i < n >> (j - 1)) return j;
  return 1;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const N0 = host.dataset.n === "16" ? 16 : 8;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const norm0: HaarNorm = host.dataset.norm === "orth" ? "orth" : "half";
  const st = {
    n: N0, x: [...(N0 === 8 ? p0.x8 : p0.x16)], norm: norm0, k: 0,
    dropped: new Set<number>(), basis: false,
  };
  st.k = Math.min(Number(host.dataset.levels ?? 0) || 0, maxLevels(st.n) + 1);

  const plot = htmlEl("div", "hs-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${VW} ${VH}`, width: "100%", role: "img", "aria-label": "Haar transform step by step" }, plot);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "hs-lab", "signal", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "hs-lab", "transform", panel);
  const ctl = htmlEl("div", "qb-presets", "", panel);
  const bBack = button(ctl, "◀ back", () => { if (st.k > 0) { st.k--; st.dropped.clear(); draw(); } });
  const bStep = button(ctl, "step ▶", () => { if (st.k < levelsMax()) { st.k++; st.dropped.clear(); draw(); } });
  button(ctl, "all ▶▶", () => { st.k = levelsMax(); st.dropped.clear(); draw(); });
  htmlEl("div", "hs-lab", "normalisation · length", panel);
  const normRow = htmlEl("div", "qb-presets", "", panel);
  const normBtns = (["half", "orth"] as HaarNorm[]).map(nm => button(normRow, nm === "half" ? "s = (a+b)/2" : "s = (a+b)/√2", () => { st.norm = nm; draw(); }));
  const nBtns = [8, 16].map(n => button(normRow, `N = ${n}`, () => { if (st.n !== n) { st.n = n; load(PRESETS.find(p => p.id === curPreset) ?? PRESETS[0]); } }));
  const kS = slider(panel, "keep K largest", 0, 16, 1, 16, v => keepLargest(v), v => String(Math.min(v, st.n)));
  const basisBox = checkbox(panel, "show basis (transform matrix)", false, v => { st.basis = v; draw(); });
  button(panel, "reset", () => { st.n = N0; st.norm = norm0; load(p0); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag the bar tops to change the signal. Click a coefficient to drop it or bring it back.", panel);

  let curPreset = p0.id;
  const levelsMax = () => maxLevels(st.n) + 1;

  function load(p: Preset): void {
    curPreset = p.id;
    st.x = [...(st.n === 8 ? p.x8 : p.x16)];
    st.k = p === p0 ? Math.min(Number(host.dataset.levels ?? 0) || 0, levelsMax()) : 0;
    st.dropped.clear();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    kS.input.max = String(st.n); kS.set(st.n);
    draw();
  }

  function keepLargest(K: number): void {
    const c = states()[st.k];
    const idx = Array.from({ length: st.n }, (_, i) => i).sort((i, j) => Math.abs(c[j]) - Math.abs(c[i]) || i - j);
    st.dropped = new Set(idx.slice(Math.min(K, st.n)));
    draw(false);
  }

  const states = () => haarStates(st.x, st.norm, levelsMax());

  // ---- drawing ----
  function draw(syncK = true): void {
    svg.replaceChildren();
    const n = st.n, S = states(), c = S[st.k];
    bStep.disabled = st.k >= levelsMax(); bBack.disabled = st.k <= 0;
    normBtns.forEach((b, i) => b.classList.toggle("active", ["half", "orth"][i] === st.norm));
    nBtns.forEach((b, i) => b.classList.toggle("active", [8, 16][i] === n));
    basisBox.checked = st.basis;
    if (syncK) kS.set(st.n - st.dropped.size);

    // reconstruction from the kept coefficients of the current state
    const kept = Float64Array.from(c, (v, i) => (st.dropped.has(i) ? 0 : v));
    const rec = haarInverse(kept, st.norm, st.k);
    const anyDrop = st.dropped.size > 0;

    // signal panel
    const ylo = Math.min(-2, ...st.x, ...(anyDrop ? rec : [])) - 0.5, yhi = Math.max(14, ...st.x, ...(anyDrop ? rec : [])) + 0.5;
    const bw = (SIG.r - SIG.l) / n;
    const X = (i: number) => SIG.l + i * bw;
    const Y = (v: number) => SIG.b - (v - ylo) / (yhi - ylo) * (SIG.b - SIG.t);
    for (let v = Math.ceil(ylo / 2) * 2; v <= yhi; v += 2) {
      svgEl("line", { x1: SIG.l, x2: SIG.r, y1: Y(v), y2: Y(v), stroke: v === 0 ? "var(--qb-axis)" : "var(--qb-grid)", "stroke-width": v === 0 ? 1.5 : 1 }, svg);
      const t = svgEl("text", { x: SIG.l - 8, y: Y(v) + 6, "text-anchor": "end", class: "small" }, svg); t.textContent = String(v);
    }
    const title = svgEl("text", { x: SIG.l, y: 22, "font-weight": 600 }, svg);
    title.textContent = st.k === 0 ? "signal x (drag the bars)" : `signal x, with the ${n >> st.k} average${n >> st.k > 1 ? "s" : ""} after level ${st.k}`;
    // averages of the current level as wide blocks (scaled back to signal units)
    if (st.k > 0) {
      const m = n >> st.k, w = 2 ** st.k, sc = st.norm === "half" ? 1 : 2 ** (-st.k / 2);
      for (let j = 0; j < m; j++) {
        const v = c[j] * sc;
        svgEl("rect", { x: X(j * w) + 2, width: w * bw - 4, y: Math.min(Y(v), Y(0)), height: Math.abs(Y(v) - Y(0)), fill: LEVEL_COL[0], "fill-opacity": 0.18, stroke: LEVEL_COL[0], "stroke-width": 2 }, svg);
      }
    }
    // pairs to be merged next
    if (st.k < levelsMax()) {
      const w = 2 ** (st.k + 1), m = n / w;
      for (let j = 0; j < m; j++) {
        const x0 = X(j * w) + 6, x1 = X((j + 1) * w) - 6, yb = SIG.b + 12;
        svgEl("path", { d: `M${x0},${yb - 6}V${yb}H${x1}V${yb - 6}`, fill: "none", stroke: LEVEL_COL[(st.k % 5) + 1], "stroke-width": 2.5 }, svg);
      }
    }
    for (let i = 0; i < n; i++) {
      const v = st.x[i];
      svgEl("rect", { x: X(i) + bw * 0.18, width: bw * 0.64, y: Math.min(Y(v), Y(0)), height: Math.max(1, Math.abs(Y(v) - Y(0))), fill: "var(--qb-c1)", "fill-opacity": 0.75 }, svg);
      const lab = svgEl("text", { x: X(i) + bw / 2, y: Y(v) + (v >= 0 ? -6 : 20), "text-anchor": "middle", class: "num" }, svg);
      lab.textContent = fmt(v);
      const handle = svgEl("rect", { x: X(i) + bw * 0.1, width: bw * 0.8, y: Y(v) - 10, height: 20, fill: "transparent", style: "cursor: ns-resize; touch-action: none" }, svg);
      handle.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        handle.setPointerCapture(ev.pointerId);
        const move = (e: PointerEvent) => {
          const [, py] = svgPoint(svg, e);
          const val = Math.round(Math.max(-2, Math.min(14, ylo + (SIG.b - py) / (SIG.b - SIG.t) * (yhi - ylo))));
          if (val !== st.x[i]) { st.x[i] = val; presetBtns.forEach(b => b.classList.remove("active")); st.dropped.clear(); draw(); }
        };
        const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
        handle.addEventListener("pointermove", move);
        handle.addEventListener("pointerup", up);
      });
    }
    if (anyDrop) {
      let d = "";
      for (let i = 0; i < n; i++) d += `${i ? "L" : "M"}${X(i)},${Y(rec[i])}H${X(i + 1)}`;
      svgEl("path", { d, fill: "none", stroke: REC, "stroke-width": 4, "stroke-dasharray": "8 4" }, svg);
      const t = svgEl("text", { x: SIG.r, y: 22, "text-anchor": "end", fill: REC, "font-weight": 600 }, svg);
      t.textContent = "reconstruction from the kept coefficients";
      (t as SVGTextElement).style.fill = REC;
    }

    if (st.basis) drawBasis(); else drawCoefs(c);

    // readout
    const ex = st.x.reduce((a, v) => a + v * v, 0), ec = Array.from(c).reduce((a, v) => a + v * v, 0);
    let dropSq = 0;
    st.dropped.forEach(i => { dropSq += c[i] * c[i]; });
    let err = 0;
    for (let i = 0; i < n; i++) err += (rec[i] - st.x[i]) ** 2;
    const lines = [
      `levels done   ${st.k} of ${levelsMax()}`,
      `Σ x²          ${fmt(ex)}`,
      `Σ c²          ${fmt(ec)}${Math.abs(ex - ec) < 1e-9 ? "  (= Σ x²)" : ""}`,
      `kept          ${n - st.dropped.size} of ${n}`,
      `Σ dropped c²  ${fmt(dropSq)}`,
      `Σ (x − x̂)²    ${fmt(err)}`,
    ];
    ro.textContent = lines.join("\n");
  }

  function drawCoefs(c: Float64Array): void {
    const n = st.n, k = st.k;
    const lo = Math.min(0, ...c), hi = Math.max(0, ...c), span = Math.max(hi - lo, 1);
    const ylo = lo - 0.12 * span, yhi = hi + 0.18 * span;
    const bw = (COEF.r - COEF.l) / n;
    const X = (i: number) => COEF.l + i * bw;
    const Y = (v: number) => COEF.b - (v - ylo) / (yhi - ylo) * (COEF.b - COEF.t);
    const title = svgEl("text", { x: COEF.l, y: COEF.t - 18, "font-weight": 600 }, svg);
    title.textContent = k === 0 ? "coefficients: none yet (press step)" : "coefficients [ s | d (coarse) … d (level 1) ]: click to drop";
    svgEl("line", { x1: COEF.l, x2: COEF.r, y1: Y(0), y2: Y(0), stroke: "var(--qb-axis)", "stroke-width": 1.5 }, svg);
    for (let i = 0; i < n; i++) {
      const lev = levelAt(i, n, k), col = LEVEL_COL[lev === 0 ? 0 : ((lev - 1) % 5) + 1];
      const dropped = st.dropped.has(i), v = c[i];
      const g = svgEl("g", { class: "coef" }, svg);
      svgEl("rect", { x: X(i), width: bw, y: COEF.t - 4, height: COEF.b - COEF.t + 30, fill: "transparent" }, g);
      svgEl("rect", {
        x: X(i) + bw * 0.15, width: bw * 0.7, y: Math.min(Y(v), Y(0)), height: Math.max(1.5, Math.abs(Y(v) - Y(0))),
        fill: dropped ? "none" : col, "fill-opacity": 0.85, stroke: col, "stroke-width": 2, "stroke-dasharray": dropped ? "5 4" : "",
      }, g);
      const t = svgEl("text", { x: X(i) + bw / 2, y: v >= 0 ? Y(v) - 6 : Y(v) + 18, "text-anchor": "middle", class: "num" }, g);
      t.textContent = fmt(v);
      if (dropped) (t as SVGTextElement).style.opacity = "0.4";
      g.addEventListener("click", () => {
        if (st.dropped.has(i)) st.dropped.delete(i); else st.dropped.add(i);
        presetBtns.forEach(b => b.classList.remove("active"));
        draw();
      });
    }
    // group labels under the bars
    const groups: { from: number; to: number; lab: string; lev: number }[] = [];
    const m = n >> k;
    groups.push({ from: 0, to: m, lab: k === 0 ? "x" : "s", lev: 0 });
    for (let j = k; j >= 1; j--) groups.push({ from: n >> j, to: n >> (j - 1), lab: `d level ${j}`, lev: j });
    for (const gr of groups) {
      const x0 = X(gr.from) + 3, x1 = X(gr.to) - 3, yb = COEF.b + 14, col = LEVEL_COL[gr.lev === 0 ? 0 : ((gr.lev - 1) % 5) + 1];
      svgEl("path", { d: `M${x0},${yb - 6}V${yb}H${x1}V${yb - 6}`, fill: "none", stroke: col, "stroke-width": 2.5 }, svg);
      const lab = gr.to - gr.from === 1 && gr.lab.startsWith("d") ? `d${gr.lev}` : gr.lab;
      const t = svgEl("text", { x: (x0 + x1) / 2, y: yb + 24, "text-anchor": "middle", class: "small" }, svg);
      t.textContent = lab;
      (t as SVGTextElement).style.fill = col;
    }
  }

  function drawBasis(): void {
    // rows of the transform matrix of the current state: c = M x
    const n = st.n, k = st.k, M: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      const e = new Array(n).fill(0); e[i] = 1;
      M.push(haarStates(e, st.norm, levelsMax())[k]);
    }
    let mx = 0;
    for (const col of M) for (const v of col) mx = Math.max(mx, Math.abs(v));
    const title = svgEl("text", { x: COEF.l, y: COEF.t - 18, "font-weight": 600 }, svg);
    title.textContent = "transform matrix: row r gives coefficient r (blue −, red +)";
    const cw = (COEF.r - COEF.l) / n, ch = (COEF.b + 30 - COEF.t) / n;
    for (let r = 0; r < n; r++) for (let i = 0; i < n; i++) {
      const v = M[i][r];   // column i of the matrix = transform of e_i
      const [cr, cg, cb] = diverging(divergingT(v, mx));
      svgEl("rect", { x: COEF.l + i * cw, y: COEF.t + r * ch, width: cw - 1, height: ch - 1, fill: `rgb(${255 * cr},${255 * cg},${255 * cb})` }, svg);
      if (n === 8 && Math.abs(v) > 1e-12) {
        const t = svgEl("text", { x: COEF.l + (i + 0.5) * cw, y: COEF.t + (r + 0.5) * ch + 6, "text-anchor": "middle", class: "num" }, svg);
        t.textContent = fmt(v);
      }
    }
    for (let r = 0; r < n; r++) {
      const lev = levelAt(r, n, k), t = svgEl("text", { x: COEF.l - 8, y: COEF.t + (r + 0.5) * ch + 6, "text-anchor": "end", class: "small" }, svg);
      t.textContent = lev === 0 ? (k === 0 ? "x" : "s") : `d${lev}`;
      (t as SVGTextElement).style.fill = LEVEL_COL[lev === 0 ? 0 : ((lev - 1) % 5) + 1];
    }
  }

  load(p0);
  st.norm = norm0;
  draw();
}

autoMount(NAME, mount);
