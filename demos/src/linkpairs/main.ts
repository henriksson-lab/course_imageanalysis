// Linking two frames by hand. Frame t: filled discs (A, B, C, …); frame t+1: open rings
// (P, Q, R, …); all draggable. Methods: nearest neighbour (conflicts in red), greedy (cheapest
// pair first; "step" shows the order), global assignment (lib/linking.ts, Hungarian on the
// augmented matrix). Cost = squared distance or distance; a gate (circles) forbids long links;
// b = the cost of ending a track (and of starting one). The cost matrix (or the augmented
// matrix) is shown with the chosen cells boxed.
//
// <div data-demo="linkpairs" data-preset="conflict|greedy|global|gate|end|newcomer|augmented"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, draggable } from "../lib/ui";
import { Plot, heatmapGrid } from "../lib/plot";
import { mapFromAnchors } from "../lib/colour";
import * as LK from "../lib/linking";

const NAME = "linkpairs";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .8em; }
${S} .lp-field { flex: 1 1 540px; min-width: 260px; max-width: 640px; }
${S} .qb-presets { flex-basis: 100%; }
${S} .lp-field svg { touch-action: none; }
${S} .qb-panel { flex: 1 1 330px; max-width: 420px; gap: .35em; }
${S} .lp-matrix { max-width: 400px; }
${S} .lp-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-size: 18px; white-space: pre-wrap; }
${S} .qb-btn.lp-on { background: var(--qb-accent); border-color: var(--qb-accent); color: #fff; }
.reveal ${S} { font-size: 19px; }
.reveal ${S} .lp-field { flex: 0 0 560px; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .lp-matrix { max-width: 330px; }
`;

type Method = "nn" | "greedy" | "global";
interface State { A: LK.Pt[]; B: LK.Pt[]; method: Method; sq: boolean; gate: number; b: number; aug: boolean; step: number }
const GATE_OFF = 10, B_INF = 20;
const ABPQ = (): Pick<State, "A" | "B"> => ({ A: [{ x: 0, y: 0 }, { x: 3, y: 0 }], B: [{ x: 2, y: 0 }, { x: 5, y: 0 }] });
const DEFAULT: State = { ...ABPQ(), method: "global", sq: true, gate: GATE_OFF, b: B_INF, aug: false, step: -1 };
const PRESETS: { id: string; label: string; s: () => Partial<State> }[] = [
  { id: "conflict", label: "NN conflict", s: () => ({ ...ABPQ(), method: "nn" }) },
  { id: "greedy", label: "greedy loses", s: () => ({ ...ABPQ(), method: "greedy", step: 0 }) },
  { id: "global", label: "global: 8 < 26", s: () => ({ ...ABPQ(), method: "global" }) },
  { id: "gate", label: "gate too small: a track breaks", s: () => ({
    A: [{ x: 0, y: 2 }, { x: 0, y: 0 }, { x: 0, y: -2 }], B: [{ x: 1.5, y: 2.2 }, { x: 4.5, y: 0.2 }, { x: 1.4, y: -1.8 }], method: "global", gate: 3 }) },
  { id: "end", label: "ending is cheaper (b = 2)", s: () => ({ ...ABPQ(), method: "global", b: 2 }) },
  { id: "newcomer", label: "a newcomer steals a link", s: () => ({
    A: [{ x: 0, y: 1.5 }, { x: 4, y: -1 }], B: [{ x: 2.2, y: 1.5 }, { x: 0.9, y: 0.4 }, { x: 5.8, y: -1 }], method: "global", b: 4 }) },
  { id: "augmented", label: "augmented matrix (b = 3)", s: () => ({ ...ABPQ(), method: "global", b: 3, aug: true }) },
];
const ROWN = "ABCDEFGH", COLN = "PQRSTUVW";
const COL_T = "var(--qb-c1, #1f77b4)", COL_T1 = "var(--qb-c4, #ff7f0e)", RED = "#d62728", GREEN = "#2a9d4b";
const cmap = mapFromAnchors(["#f7fbff", "#c6dbef", "#6baed6", "#2171b5"]);
const fmtNum = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) >= 10 ? 1 : 2));

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

autoMount(NAME, host => {
  injectStyle();
  let st: State = { ...DEFAULT, A: DEFAULT.A.map(p => ({ ...p })), B: DEFAULT.B.map(p => ({ ...p })) };
  const fieldBox = htmlEl("div", "lp-field", "", host);
  const plot = new Plot(fieldBox, { width: 600, height: 370, x: [-1.5, 8.5], y: [-3.2, 3.2], equal: true, xticks: 10, yticks: 7, font: 24, ariaLabel: "two frames of detections" });
  const gGate = svgEl("g", {}, plot.layer), gLinks = svgEl("g", {}, plot.layer), gPts = svgEl("g", {}, plot.over);
  const panel = htmlEl("div", "qb-panel", "", host);
  const mBox = htmlEl("div", "lp-matrix", "", panel);
  const read = htmlEl("div", "qb-readout", "", panel);
  const methRow = htmlEl("div", "lp-row", "", panel);
  const methBtns: Record<Method, HTMLButtonElement> = {
    nn: button(methRow, "nearest", () => { st.method = "nn"; render(); }),
    greedy: button(methRow, "greedy", () => { st.method = "greedy"; render(); }),
    global: button(methRow, "global", () => { st.method = "global"; render(); }),
  };
  let nGreedy = 0;
  const stepBtn = button(methRow, "step", () => {
    st.step = st.method !== "greedy" || st.step < 0 || st.step >= nGreedy ? 0 : st.step + 1;
    st.method = "greedy"; render();
  });
  const gateS = slider(panel, "gate", 0.5, GATE_OFF, 0.25, st.gate, v => { st.gate = v; render(); }, v => (v >= GATE_OFF ? "off" : v.toFixed(2)));
  const bS = slider(panel, "end/start b", 0.25, B_INF, 0.25, st.b, v => { st.b = v; render(); }, v => (v >= B_INF ? "∞" : v.toFixed(2)));
  const chkRow = htmlEl("div", "lp-row", "", panel);
  const sqC = checkbox(chkRow, "squared distance", st.sq, v => { st.sq = v; render(); });
  const augC = checkbox(chkRow, "augmented matrix", st.aug, v => { st.aug = v; render(); });
  const addRow = htmlEl("div", "lp-row", "", panel);
  button(addRow, "+ disc", () => { if (st.A.length < 8) st.A.push({ x: 1 + 0.7 * st.A.length, y: -2.4 }); render(true); });
  button(addRow, "+ ring", () => { if (st.B.length < 8) st.B.push({ x: 2 + 0.7 * st.B.length, y: 2.4 }); render(true); });
  button(addRow, "− disc", () => { if (st.A.length > 1) st.A.pop(); render(true); });
  button(addRow, "− ring", () => { if (st.B.length > 1) st.B.pop(); render(true); });
  const presetBtns = presets(host, [
    ...PRESETS.map(p => ({ label: p.label, apply: () => apply(p.s()) })),
    { label: "reset", apply: () => apply({}) },
  ]);

  function apply(p: Partial<State>): void {
    st = { ...DEFAULT, ...ABPQ(), ...p };
    st.A = st.A.map(q => ({ ...q })); st.B = st.B.map(q => ({ ...q }));
    gateS.set(st.gate); bS.set(st.b); sqC.checked = st.sq; augC.checked = st.aug;
    render(true);
  }

  const handles: SVGCircleElement[] = [], labels: SVGTextElement[] = [];
  function buildHandles(): void {
    gPts.replaceChildren(); handles.length = 0; labels.length = 0;
    const all = [...st.A.map((p, i) => ({ p, i, f: 0 })), ...st.B.map((p, i) => ({ p, i, f: 1 }))];
    for (const h of all) {
      const c = svgEl("circle", { r: 13 }, gPts);
      c.style.fill = h.f === 0 ? COL_T : "var(--qb-bg, #fff)";
      c.style.stroke = h.f === 0 ? COL_T : COL_T1;
      c.style.strokeWidth = "4";
      const t = svgEl("text", { "text-anchor": "middle", class: "qb-halo" }, gPts);
      t.textContent = h.f === 0 ? ROWN[h.i] : COLN[h.i];
      t.style.fontSize = "20px"; t.style.fontWeight = "600"; t.style.pointerEvents = "none";
      handles.push(c); labels.push(t);
      draggable(c, ev => plot.toData(ev), (x, y) => {
        const q = h.f === 0 ? st.A[h.i] : st.B[h.i];
        q.x = Math.round(Math.max(-1.4, Math.min(8.4, x)) * 10) / 10; q.y = Math.round(Math.max(-3.1, Math.min(3.1, y)) * 10) / 10;
        render();
      });
    }
  }

  function render(rebuild = false): void {
    if (rebuild || handles.length !== st.A.length + st.B.length) buildHandles();
    const m = st.A.length, n = st.B.length, kind: LK.CostKind = st.sq ? "sq" : "dist";
    const gate = st.gate >= GATE_OFF ? Infinity : st.gate, b = st.b >= B_INF ? Infinity : st.b;
    const C = LK.costMatrix(st.A, st.B, kind, gate);
    // positions of handles
    const all = [...st.A, ...st.B];
    all.forEach((p, k) => {
      handles[k].setAttribute("cx", String(plot.x(p.x))); handles[k].setAttribute("cy", String(plot.y(p.y)));
      labels[k].setAttribute("x", String(plot.x(p.x) + (k < m ? -16 : 16))); labels[k].setAttribute("y", String(plot.y(p.y) + (k < m ? -18 : 34)));
    });
    // gates
    gGate.replaceChildren();
    if (Number.isFinite(gate)) for (const p of st.A) {
      const c = svgEl("circle", { cx: plot.x(p.x), cy: plot.y(p.y), r: Math.abs(plot.x(gate) - plot.x(0)) }, gGate);
      c.style.fill = COL_T; c.style.fillOpacity = "0.07"; c.style.stroke = COL_T; c.style.strokeDasharray = "5 4"; c.style.strokeWidth = "1.5";
    }
    // candidate pairs: thin grey lines (allowed), none for gated
    gLinks.replaceChildren();
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) if (Number.isFinite(C[i * n + j])) {
      const l = svgEl("line", { x1: plot.x(st.A[i].x), y1: plot.y(st.A[i].y), x2: plot.x(st.B[j].x), y2: plot.y(st.B[j].y) }, gLinks);
      l.style.stroke = "#bbb"; l.style.strokeWidth = "1.5"; l.style.strokeDasharray = "3 4";
    }
    // the chosen links
    let links: LK.Link[] = [], conflictCols = new Set<number>(), note = "";
    const ab = Number.isFinite(b) ? b : 1e9;
    if (st.method === "nn") {
      const succ = LK.nearestNeighbour(C, m, n);
      succ.forEach((j, i) => { if (j >= 0) links.push({ i, j, c: C[i * n + j] }); });
      conflictCols = LK.conflicts(succ);
    } else if (st.method === "greedy") {
      const all = LK.greedy(C, m, n, 2 * ab);
      const k = st.step < 0 ? all.length : Math.min(st.step, all.length);
      links = all.slice(0, k);
      nGreedy = all.length;
      note = st.step < 0 ? "" : `greedy step ${k}/${all.length}` + (k === 0 ? " (press step)" : "");
    } else {
      links = LK.assignGlobal(C, m, n, ab, ab).links;
    }
    links.forEach((l, k) => {
      const p = st.A[l.i], q = st.B[l.j], bad = conflictCols.has(l.j);
      const x0 = plot.x(p.x), y0 = plot.y(p.y), x1 = plot.x(q.x), y1 = plot.y(q.y);
      const L = Math.hypot(x1 - x0, y1 - y0) || 1, ux = (x1 - x0) / L, uy = (y1 - y0) / L;
      const ax = x1 - ux * 15, ay = y1 - uy * 15;
      const ln = svgEl("line", { x1: x0 + ux * 13, y1: y0 + uy * 13, x2: ax - ux * 10, y2: ay - uy * 10 }, gLinks);
      ln.style.stroke = bad ? RED : GREEN; ln.style.strokeWidth = "4";
      const hd = svgEl("polygon", { points: `${ax},${ay} ${ax - ux * 14 - uy * 7},${ay - uy * 14 + ux * 7} ${ax - ux * 14 + uy * 7},${ay - uy * 14 - ux * 7}` }, gLinks);
      hd.style.fill = bad ? RED : GREEN;
      const lab = svgEl("text", { x: (x0 + x1) / 2, y: (y0 + y1) / 2 - 8, "text-anchor": "middle", class: "qb-halo" }, gLinks);
      lab.textContent = (st.method === "greedy" && st.step >= 0 ? `${k + 1}: ` : "") + fmtNum(l.c);
      lab.style.fontSize = "18px"; lab.style.fill = bad ? RED : "var(--qb-fg)";
    });
    // matrix
    mBox.replaceChildren();
    const chosen = new Set(links.map(l => l.i * 1000 + l.j));
    const linkedR = new Set(links.map(l => l.i)), linkedC = new Set(links.map(l => l.j));
    let vals: Float64Array, R = m, Cn = n, rowL = ROWN.slice(0, m).split(""), colL = COLN.slice(0, n).split("");
    const boxes: [number, number][] = [];
    if (st.aug) {
      vals = LK.augmentedMatrix(C, m, n, b, b);
      R = Cn = m + n;
      rowL = [...rowL, ...colL.map(c => "+" + c)];
      colL = [...colL, ...ROWN.slice(0, m).split("").map(c => c + "✕")];
      if (st.method === "global" || st.method === "greedy") {
        for (const l of links) boxes.push([l.i, l.j]);
        for (let i = 0; i < m; i++) if (!linkedR.has(i)) boxes.push([i, n + i]);
        for (let j = 0; j < n; j++) if (!linkedC.has(j)) boxes.push([m + j, j]);
        // the leftover dummies pair up in the zero block
        const freeRows = [...linkedC].sort((a, b) => a - b), freeCols = [...linkedR].sort((a, b) => a - b);
        freeRows.forEach((j, k) => boxes.push([m + j, n + freeCols[k]]));
      }
    } else {
      vals = C;
      for (const l of links) boxes.push([l.i, l.j]);
    }
    const fin = [...vals].filter(Number.isFinite);
    const hm = heatmapGrid(mBox, vals, R, Cn, {
      rowLabels: rowL, colLabels: colL, cell: R > 6 ? 36 : 46, map: cmap, lo: Math.min(0, ...fin), hi: Math.max(1, ...fin),
      title: st.aug ? "augmented matrix" : "costs (rows t, columns t+1)", font: 18,
      fmt: fmtNum, text: v => (Number.isFinite(v) ? undefined : st.aug ? "∞" : "–"),
    });
    // boxes over the chosen cells: same geometry as heatmapGrid
    const f = 18, cell = R > 6 ? 36 : 46;
    const lw = Math.max(...rowL.map(s => s.length)) * f * 0.6 + 10, th = f * 1.5 + f * 1.4;
    for (const [r, c] of boxes) {
      const bx = svgEl("rect", { x: lw + c * cell + 1.5, y: th + r * cell + 1.5, width: cell - 3, height: cell - 3, rx: 3 }, hm.svg);
      bx.style.fill = "none"; bx.style.stroke = chosen.has(r * 1000 + c) && conflictCols.has(c) ? RED : "#111"; bx.style.strokeWidth = "3.5";
    }
    // readout: totals of all three methods
    // totals include b per end and start; with b = ∞ only the link costs are added up
    const total = (ls: LK.Link[]) => (Number.isFinite(b) ? LK.totalCost(ls, m, n, b, b) : ls.reduce((s, l) => s + l.c, 0));
    const g = LK.greedy(C, m, n, 2 * ab), gl = LK.assignGlobal(C, m, n, ab, ab);
    const nnSucc = LK.nearestNeighbour(C, m, n), nnConf = LK.conflicts(nnSucc);
    const lines = [
      `nearest: ${nnConf.size ? "conflict on " + [...nnConf].map(j => COLN[j]).join(", ") : "no conflict"}`,
      `greedy total ${fmtNum(total(g))}, global total ${fmtNum(total(gl.links))}`,
      `shown: ${links.length} links, ${m - linkedR.size} ends, ${n - linkedC.size} starts`,
    ];
    if (note) lines.push(note);
    read.textContent = lines.join("\n");
    (Object.keys(methBtns) as Method[]).forEach(k => methBtns[k].classList.toggle("lp-on", st.method === k));
    stepBtn.classList.toggle("lp-on", st.method === "greedy" && st.step >= 0);
  }

  const pre = host.dataset.preset ? PRESETS.find(p => p.id === host.dataset.preset) : undefined;
  apply(pre ? pre.s() : {});
});
