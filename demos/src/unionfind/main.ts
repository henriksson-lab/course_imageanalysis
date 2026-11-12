// unionfind: the disjoint-set forest drawn as trees (labelling day, unit label-connectivity/).
// A sequence of union and find operations (presets, or click two nodes to unite them, or
// one node to find its root) is played step by step. Toggles: union by rank, and path
// compression none / full / halving. Readout: pointer steps followed by all finds so far,
// the depth of the deepest node, the number of sets; below the trees the parent array.
//
// Auto-mounts on <div data-demo="unionfind">. Options: data-preset (id below),
// data-step (operation to show first: a number or "end").
import { autoMount, htmlEl, button, svgEl, checkbox } from "../lib/ui";
import { Forest, Compression } from "./forest";

const NAME = "unionfind";
const W = 720, H = 470, TABLE = 78;

type Op = { kind: "union"; a: number; b: number } | { kind: "find"; a: number };
interface Preset { id: string; label: string; n: number; byRank: boolean; comp: Compression; ops: Op[]; step?: number | "end" }

const U = (a: number, b: number): Op => ({ kind: "union", a, b });
const F = (a: number): Op => ({ kind: "find", a });
const chain = (n: number) => Array.from({ length: n - 1 }, (_, i) => U(i + 1, i + 2));
function randomOps(n: number, k: number, seed: number): Op[] {
  let s = seed >>> 0;
  const r = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  return Array.from({ length: k }, () => U(1 + Math.floor(r() * n), 1 + Math.floor(r() * n)));
}

const PRESETS: Preset[] = [
  { id: "chain", label: "naive unions build a chain", n: 8, byRank: false, comp: "none", ops: [...chain(8), F(1), F(1)] },
  { id: "rank", label: "union by rank keeps trees flat", n: 8, byRank: true, comp: "none", ops: [...chain(8), F(1), F(1)] },
  { id: "compress", label: "compression flattens on the way", n: 8, byRank: false, comp: "full", ops: [...chain(8), F(1), F(1)] },
  { id: "halving", label: "halving: every other pointer", n: 8, byRank: false, comp: "halving", ops: [...chain(8), F(1), F(1)] },
  { id: "comb", label: "two-pass equivalences of the comb", n: 5, byRank: true, comp: "full", ops: [U(2, 1), U(3, 1), U(4, 1), U(5, 1)] },
  { id: "hand", label: "6 unions (exercise)", n: 8, byRank: false, comp: "none", ops: [U(1, 2), U(3, 4), U(2, 4), U(5, 6), U(7, 8), U(6, 8), U(4, 8), F(1)] },
  { id: "random", label: "random: 16 nodes", n: 16, byRank: true, comp: "full", ops: randomOps(16, 14, 5) },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .uf-plot { flex: 1 1 560px; min-width: 0; max-width: 760px; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .45em; }
[data-demo="${NAME}"] .uf-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .uf-ops { font-family: ui-monospace, monospace; font-size: 15px; line-height: 1.5; }
[data-demo="${NAME}"] .uf-ops span { padding: 0 .2em; border-radius: 4px; white-space: nowrap; cursor: pointer; }
[data-demo="${NAME}"] .uf-ops span.done { color: var(--qb-muted); }
[data-demo="${NAME}"] .uf-ops span.cur { background: var(--qb-c4); color: #fff; }
[data-demo="${NAME}"] svg text { font-size: 19px; }
[data-demo="${NAME}"] svg .node { cursor: pointer; }
.reveal [data-demo="${NAME}"] { font-size: 19px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .45em; }
.reveal [data-demo="${NAME}"] .qb-note, .reveal [data-demo="${NAME}"] .uf-lab { font-size: 15px; }
@media (max-width: 900px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const opText = (o: Op) => (o.kind === "union" ? `union(${o.a}, ${o.b})` : `find(${o.a})`);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let n = p0.n, ops = p0.ops.slice(), k = 0, byRank = p0.byRank, comp: Compression = p0.comp;
  let picked: number | null = null, clickMode: "union" | "find" = "union";

  const box = htmlEl("div", "uf-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "union–find forest" }, box);
  const panel = htmlEl("div", "qb-panel", "", host);

  const rankBox = checkbox(panel, "union by rank", byRank, v => { byRank = v; render(); });
  htmlEl("div", "uf-lab", "path compression", panel);
  const crow = htmlEl("div", "qb-presets", "", panel);
  const COMPS: Compression[] = ["none", "full", "halving"];
  const cbtn = COMPS.map(c => button(crow, c, () => { comp = c; render(); }));
  htmlEl("div", "uf-lab", "operations", panel);
  const opsEl = htmlEl("div", "uf-ops", "", panel);
  const ctl = htmlEl("div", "qb-presets", "", panel);
  button(ctl, "⏮", () => { k = 0; render(); });
  button(ctl, "◀ back", () => { k = Math.max(0, k - 1); render(); });
  button(ctl, "step ▶", () => { k = Math.min(ops.length, k + 1); render(); });
  button(ctl, "end ⏭", () => { k = ops.length; render(); });
  const read = htmlEl("div", "qb-readout", "", panel);
  const mrow = htmlEl("div", "qb-presets", "", panel);
  const mU = button(mrow, "click: union", () => { clickMode = "union"; picked = null; render(); });
  const mF = button(mrow, "click: find", () => { clickMode = "find"; picked = null; render(); });
  const note = htmlEl("div", "qb-note", "", panel);
  htmlEl("div", "uf-lab", "examples", panel);
  const prow = htmlEl("div", "qb-presets", "", panel);
  const pbtn = PRESETS.map(p => button(prow, p.label, () => load(p)));
  button(prow, "⟲ reset", () => load(p0));

  function load(p: Preset): void {
    n = p.n; ops = p.ops.slice(); byRank = p.byRank; comp = p.comp; rankBox.checked = byRank; picked = null;
    k = p.step === "end" ? ops.length : p.step ?? 0;
    pbtn.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    render();
  }

  function addOp(o: Op): void { ops = ops.slice(0, k); ops.push(o); k = ops.length; render(); }

  function render(): void {
    cbtn.forEach((b, i) => b.classList.toggle("active", COMPS[i] === comp));
    mU.classList.toggle("active", clickMode === "union"); mF.classList.toggle("active", clickMode === "find");
    // replay the first k operations
    const f = new Forest(n, { byRank, compression: comp });
    let pathA: number[] = [], pathB: number[] = [], linked: [number, number] | null = null, lastMsg = "", lastSteps = 0;
    for (let i = 0; i < k; i++) {
      const o = ops[i], s0 = f.steps;
      if (o.kind === "union") {
        const r = f.union(o.a - 1, o.b - 1);
        pathA = r.pathA; pathB = r.pathB; linked = r.child === null ? null : [r.child, r.root];
        lastMsg = r.child === null ? `${opText(o)}: already in one set (root ${r.root + 1})` : `${opText(o)}: roots ${r.ra + 1} and ${r.rb + 1}; ${r.child + 1} now points to ${r.root + 1}`;
      } else {
        const r = f.find(o.a - 1);
        pathA = r.path; pathB = []; linked = null;
        lastMsg = `${opText(o)}: path ${r.path.map(v => v + 1).join(" → ")}, root ${r.root + 1}`;
      }
      lastSteps = f.steps - s0;
    }
    // ops list
    opsEl.replaceChildren();
    ops.forEach((o, i) => {
      const s = htmlEl("span", i < k - 1 ? "done" : i === k - 1 ? "cur" : "", opText(o), opsEl);
      s.addEventListener("click", () => { k = i + 1; render(); });
      opsEl.append(" ");
    });
    if (!ops.length) opsEl.textContent = "(none yet: click the nodes)";
    const sets = f.classes().length;
    read.textContent = `${k ? lastMsg : "start: every node is its own set"}\n` +
      `pointer steps, last op: ${lastSteps}   all: ${f.steps}\ndeepest node: depth ${f.maxDepth()}   sets: ${sets}`;
    note.textContent = clickMode === "union" ? (picked === null ? "Click a node, then another: union(first, second)." : `union(${picked + 1}, …): click the second node.`) : "Click a node to find its root.";
    draw(f, pathA, pathB, linked);
  }

  function draw(f: Forest, pathA: number[], pathB: number[], linked: [number, number] | null): void {
    svg.replaceChildren();
    const kids: number[][] = Array.from({ length: n }, () => []);
    const roots: number[] = [];
    for (let i = 0; i < n; i++) { if (f.parent[i] === i) roots.push(i); else kids[f.parent[i]].push(i); }
    const depth = new Int32Array(n);
    const pos = new Float64Array(2 * n);
    let leaf = 0, maxD = 0;
    const place = (v: number, d: number): number => {
      depth[v] = d; maxD = Math.max(maxD, d);
      if (!kids[v].length) { pos[2 * v] = leaf++; return pos[2 * v]; }
      const xs = kids[v].sort((a, b) => a - b).map(c => place(c, d + 1));
      pos[2 * v] = (xs[0] + xs[xs.length - 1]) / 2;
      return pos[2 * v];
    };
    for (const r of roots.sort((a, b) => a - b)) { place(r, 0); leaf += 0.4; }
    const top = 36, bottom = H - TABLE - 30, dy = Math.min(62, (bottom - top) / Math.max(1, maxD));
    const span = Math.max(1, leaf - 0.4 - 1), dx = Math.min(70, (W - 80) / Math.max(1, span)), x0 = (W - span * dx) / 2;
    const X = (v: number) => x0 + pos[2 * v] * dx, Y = (v: number) => top + depth[v] * dy;
    const R = Math.min(19, dx * 0.42);
    const onPath = new Set([...pathA, ...pathB]);
    // edges child → parent
    for (let v = 0; v < n; v++) {
      const p = f.parent[v];
      if (p === v) continue;
      const isNew = !!linked && linked[0] === v && linked[1] === p;
      const ux = X(p) - X(v), uy = Y(p) - Y(v), L = Math.hypot(ux, uy) || 1;
      const ax = X(v) + ux / L * R, ay = Y(v) + uy / L * R, bx = X(p) - ux / L * (R + 2), by = Y(p) - uy / L * (R + 2);
      const col = isNew ? "var(--qb-c2, #d62728)" : "var(--qb-axis, #444)";
      const ln = svgEl("line", { x1: ax, y1: ay, x2: bx - ux / L * 8, y2: by - uy / L * 8 }, svg);
      ln.style.setProperty("stroke", col); ln.style.setProperty("stroke-width", isNew ? "4" : "2.2");
      const hx = -uy / L * 6, hy = ux / L * 6;
      const head = svgEl("polygon", { points: `${bx},${by} ${bx - ux / L * 11 + hx},${by - uy / L * 11 + hy} ${bx - ux / L * 11 - hx},${by - uy / L * 11 - hy}` }, svg);
      head.style.setProperty("fill", col);
    }
    for (let v = 0; v < n; v++) {
      const g = svgEl("g", { class: "node" }, svg);
      const c = svgEl("circle", { cx: X(v), cy: Y(v), r: R }, g);
      const root = f.parent[v] === v;
      c.style.setProperty("fill", onPath.has(v) ? "var(--qb-c4, #ff7f0e)" : picked === v ? "var(--qb-c1, #1f77b4)" : "var(--qb-bg, #fff)");
      c.style.setProperty("stroke", root ? "var(--qb-c1, #1f77b4)" : "var(--qb-axis, #444)");
      c.style.setProperty("stroke-width", root ? "4" : "2");
      const t = svgEl("text", { x: X(v), y: Y(v) + 6.5, "text-anchor": "middle" }, g);
      t.textContent = String(v + 1);
      if (onPath.has(v) || picked === v) t.style.setProperty("fill", "#fff");
      if (byRank && root) {
        const rt = svgEl("text", { x: X(v) + R + 3, y: Y(v) - R + 4, "text-anchor": "start" }, g);
        rt.textContent = `r${f.rank[v]}`;
        rt.style.setProperty("font-size", "14px"); rt.style.setProperty("fill", "var(--qb-muted, #777)");
      }
      g.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        if (clickMode === "find") { addOp(F(v + 1)); return; }
        if (picked === null) { picked = v; render(); }
        else { const a = picked; picked = null; if (a !== v) addOp(U(a + 1, v + 1)); else render(); }
      });
    }
    // parent array
    const ty = H - TABLE + 8, cw = Math.min(42, (W - 90) / n), tx = 90;
    const rows: [string, (i: number) => string][] = [["node", i => String(i + 1)], ["parent", i => String(f.parent[i] + 1)]];
    if (byRank) rows.push(["rank", i => String(f.rank[i])]);
    rows.forEach(([name, val], r) => {
      const y = ty + r * 24;
      const lt = svgEl("text", { x: tx - 10, y, "text-anchor": "end" }, svg);
      lt.textContent = name; lt.style.setProperty("font-size", "16px"); lt.style.setProperty("fill", "var(--qb-muted, #777)");
      for (let i = 0; i < n; i++) {
        const t = svgEl("text", { x: tx + (i + 0.5) * cw, y, "text-anchor": "middle" }, svg);
        t.textContent = val(i);
        t.style.setProperty("font-size", "16px");
        if (r === 1 && f.parent[i] === i) t.style.setProperty("font-weight", "700");
      }
    });
  }

  load(p0);
  const st = host.dataset.step;
  if (st === "end") { k = ops.length; render(); } else if (st && Number.isFinite(Number(st))) { k = Math.min(ops.length, Number(st)); render(); }
}

autoMount(NAME, mount);
