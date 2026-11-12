// Maximum flow step by step. A small directed graph with capacities; Ford-Fulkerson with
// breadth-first search (Edmonds-Karp) finds one shortest augmenting path per step in the
// residual graph. Edges show flow / capacity, or the residual graph (forward: capacity
// left; backward, red: flow that can be undone). When no path is left, the nodes still
// reachable from s are shaded and the cut edges marked: flow = cut. Click an edge to change
// its capacity. Presets: an example, a graph where a back-edge is needed, and a graph
// with two minimum cuts (the toggle shows the cut nearest t as well).
// Auto-mounts on <div data-demo="maxflow">. Optional: data-preset="textbook|backedge|twocuts".
import { htmlEl, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { edmondsKarpSteps, EKResult, Edge, cutCost, EPS } from "../lib/maxflow";
import { PRESETS, Preset } from "./graphs";

const NAME = "maxflow";
const W = 800, H = 450, R = 25;
const C = { edge: "#8a8a8a", full: "#333", path: "#ff7f0e", cut: "#d62728", back: "#c2185b", sel: "#9467bd", S: "#cfe3f6", T: "#fde3cc" };

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .mf-plot { flex: 1 1 560px; min-width: 0; max-width: 800px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .45em; }
[data-demo="${NAME}"] .mf-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .mf-edit { display: flex; align-items: center; gap: .4em; min-height: 1.9em; }
[data-demo="${NAME}"] .mf-edit .qb-btn { font-size: 18px; padding: 0 .6em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; min-height: 7.5em; }
[data-demo="${NAME}"] svg text { font-size: 20px; }
[data-demo="${NAME}"] svg .lbl { font-size: 18px; font-family: ui-monospace, monospace; }
[data-demo="${NAME}"] svg .hit { cursor: pointer; }
.reveal [data-demo="${NAME}"] { font-size: 19px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

let uid = 0;

/** Nodes from which t can be reached in the residual graph. */
function reachesT(n: number, edges: Edge[], flow: number[], t: number): boolean[] {
  const seen = new Array<boolean>(n).fill(false);
  const q = [t]; seen[t] = true;
  while (q.length) {
    const v = q.shift()!;
    edges.forEach((e, i) => {
      // residual arc u -> v exists if e = u->v with room, or e = v->u with flow
      if (e.v === v && e.c - flow[i] > EPS && !seen[e.u]) { seen[e.u] = true; q.push(e.u); }
      if (e.u === v && flow[i] > EPS && !seen[e.v]) { seen[e.v] = true; q.push(e.v); }
    });
  }
  return seen;
}

const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(1));

function mount(host: HTMLElement): void {
  injectStyle();
  const id = `mf${uid++}`;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let preset: Preset = p0;
  let edges: Edge[] = p0.edges.map(e => ({ ...e }));
  let ek: EKResult = edmondsKarpSteps(p0.nodes.length, edges, 0, 1);
  let k = 0;                  // steps done; ek.steps.length + 1 = "finished, show the cut"
  let residual = false, nearT = false;
  let sel = -1;

  const plot = htmlEl("div", "mf-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, plot);
  const defs = svgEl("defs", {}, svg);
  for (const [name, col] of Object.entries(C)) {
    const m = svgEl("marker", { id: `${id}-${name}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 4.2, markerHeight: 4.2, orient: "auto-start-reverse" }, defs);
    svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: col }, m);
  }
  const gEdges = svgEl("g", {}, svg);
  const gNodes = svgEl("g", {}, svg);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "mf-lab", "graph", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "mf-lab", "Ford–Fulkerson, shortest paths first", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  const nextBtn = button(row, "next augmenting path", () => { if (k <= ek.steps.length) k++; draw(); });
  button(row, "run to end", () => { k = ek.steps.length + 1; draw(); });
  button(row, "flow = 0", () => { k = 0; draw(); });
  const resBox = checkbox(panel, "show the residual graph", residual, v => { residual = v; draw(); });
  const tBox = checkbox(panel, "cut nearest t (instead of s)", nearT, v => { nearT = v; draw(); });
  const edit = htmlEl("div", "mf-edit", "", panel);
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click an edge to change its capacity (the flow starts again from 0).", panel);
  button(panel, "reset", () => { load(p0); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));

  function load(p: Preset): void {
    preset = p; edges = p.edges.map(e => ({ ...e })); sel = -1; recompute();
  }
  function recompute(): void {
    ek = edmondsKarpSteps(preset.nodes.length, edges, 0, 1); k = 0; draw();
  }
  function changeCap(d: number): void {
    if (sel < 0) return;
    edges[sel].c = Math.max(0, Math.min(30, edges[sel].c + d));
    recompute();
  }

  const name = (i: number) => preset.nodes[i].name;

  /** A straight or bent arrow from node u to node v; bend > 0 curves to the left. Returns the label point. */
  function arrow(u: number, v: number, bend: number, attrs: Record<string, string | number>, parent: Element): [number, number] {
    const a = preset.nodes[u], b = preset.nodes[v];
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy);
    const nx = -dy / L, ny = dx / L;
    const mx = (a.x + b.x) / 2 + nx * bend, my = (a.y + b.y) / 2 + ny * bend;
    // shorten both ends by the node radius, along the direction to the control point
    const d1 = Math.hypot(mx - a.x, my - a.y), d2 = Math.hypot(b.x - mx, b.y - my);
    const sx = a.x + (mx - a.x) / d1 * R, sy = a.y + (my - a.y) / d1 * R;
    const ex = b.x - (b.x - mx) / d2 * (R + 2), ey = b.y - (b.y - my) / d2 * (R + 2);
    svgEl("path", { d: `M${sx},${sy} Q${mx},${my} ${ex},${ey}`, fill: "none", ...attrs }, parent);
    return [(sx + 2 * mx + ex) / 4, (sy + 2 * my + ey) / 4];
  }

  function label(x: number, y: number, text: string, color: string, parent: Element): void {
    const w = 11 * text.length + 10;
    svgEl("rect", { x: x - w / 2, y: y - 14, width: w, height: 27, rx: 6, fill: "#fff", stroke: color, "stroke-width": 1.5 }, parent);
    const t = svgEl("text", { x, y: y + 6, "text-anchor": "middle", class: "lbl", fill: color }, parent);
    t.textContent = text;
    t.style.fill = color;
  }

  function draw(): void {
    const n = preset.nodes.length, L = ek.steps.length;
    const done = k > L;
    const flow = k === 0 ? edges.map(() => 0) : ek.steps[Math.min(k, L) - 1].flow;
    const step = k >= 1 && k <= L ? ek.steps[k - 1] : null;
    const S = nearT ? reachesT(n, edges, ek.flow, 1).map(x => !x) : ek.S;
    const onPath = new Map<number, boolean>();   // edge -> used forwards?
    if (step) for (const a of step.path) onPath.set(a.i, a.fwd);

    gEdges.replaceChildren();
    gNodes.replaceChildren();
    const labels = svgEl("g", {}, svg);
    labels.remove();
    labels.style.pointerEvents = "none";
    edges.forEach((e, i) => {
      const isCut = done && S[e.u] && !S[e.v];
      const used = onPath.get(i);
      const hit = svgEl("g", { class: "hit" }, gEdges);
      hit.addEventListener("pointerdown", ev => { ev.preventDefault(); sel = sel === i ? -1 : i; draw(); });
      if (!residual) {
        if (used !== undefined) arrow(e.u, e.v, 0, { stroke: C.path, "stroke-width": 13, opacity: 0.55 }, hit);
        if (i === sel) arrow(e.u, e.v, 0, { stroke: C.sel, "stroke-width": 11, opacity: 0.45 }, hit);
        const full = flow[i] >= e.c - EPS && e.c > 0;
        const col = isCut ? C.cut : full ? C.full : C.edge;
        const [lx, ly] = arrow(e.u, e.v, 0, {
          stroke: col, "stroke-width": isCut ? 5 : 3, "marker-end": `url(#${id}-${isCut ? "cut" : full ? "full" : "edge"})`,
          ...(isCut ? { "stroke-dasharray": "10 6" } : {}),
        }, hit);
        arrow(e.u, e.v, 0, { stroke: "transparent", "stroke-width": 22 }, hit);
        label(lx, ly, `${fmt(flow[i])}/${fmt(e.c)}`, isCut ? C.cut : flow[i] > EPS ? "#1f4e8c" : "#555", labels);
      } else {
        const fw = e.c - flow[i], bw = flow[i];
        const both = fw > EPS && bw > EPS;
        // the augmenting path, in the direction it was used (forward residual may now be 0)
        if (used !== undefined) {
          const a = used ? [e.u, e.v] : [e.v, e.u];
          arrow(a[0], a[1], 0, { stroke: C.path, "stroke-width": 13, opacity: 0.55 }, hit);
        }
        if (fw > EPS) {
          const [lx, ly] = arrow(e.u, e.v, both ? 24 : 0, { stroke: C.edge, "stroke-width": 3, "marker-end": `url(#${id}-edge)` }, hit);
          label(lx, ly, fmt(fw), "#555", labels);
        }
        if (bw > EPS) {
          const [lx, ly] = arrow(e.v, e.u, both ? 24 : 0, { stroke: C.back, "stroke-width": 3, "marker-end": `url(#${id}-back)` }, hit);
          label(lx, ly, fmt(bw), C.back, labels);
        }
        if (i === sel) arrow(e.u, e.v, 0, { stroke: C.sel, "stroke-width": 11, opacity: 0.35 }, hit);
        arrow(e.u, e.v, 0, { stroke: "transparent", "stroke-width": 22 }, hit);
      }
    });
    gEdges.appendChild(labels);
    preset.nodes.forEach((nd, i) => {
      const onP = step ? step.nodes.includes(i) : false;
      svgEl("circle", {
        cx: nd.x, cy: nd.y, r: R, fill: done ? (S[i] ? C.S : C.T) : "#fff",
        stroke: onP ? C.path : "#444", "stroke-width": onP ? 4 : 2,
      }, gNodes);
      svgEl("text", { x: nd.x, y: nd.y + 7, "text-anchor": "middle", "font-weight": i < 2 ? 700 : 400 }, gNodes).textContent = nd.name;
    });
    if (done) {
      svgEl("text", { x: 14, y: H - 12, fill: "#1f4e8c" }, gNodes).textContent = "blue: S (source side)";
      svgEl("text", { x: W - 14, y: H - 12, "text-anchor": "end", fill: "#a0522d" }, gNodes).textContent = "orange: T (sink side)";
    }

    // edit box
    edit.replaceChildren();
    if (sel >= 0) {
      htmlEl("span", "", `capacity ${name(edges[sel].u)} → ${name(edges[sel].v)}:`, edit);
      button(edit, "−", () => changeCap(-1));
      htmlEl("b", "", fmt(edges[sel].c), edit);
      button(edit, "+", () => changeCap(1));
    } else htmlEl("span", "qb-note", "(no edge selected)", edit);

    // readout
    const total = k === 0 ? 0 : ek.steps[Math.min(k, L) - 1].total;
    let txt: string;
    if (k === 0) txt = `Flow 0. ${L} augmenting path${L === 1 ? "" : "s"} to find.`;
    else if (step) {
      const back = step.path.filter(a => !a.fwd).map(a => `${name(edges[a.i].v)}→${name(edges[a.i].u)}`);
      txt = `Step ${k} of ${L}: path ${step.nodes.map(name).join("→")}\nbottleneck ${fmt(step.bottleneck)}, total flow ${fmt(total)}`
        + (back.length ? `\nback-edge ${back.join(", ")}: undoes flow` : "");
    } else {
      const cut = edges.filter(e => S[e.u] && !S[e.v]);
      txt = `No path from s to t is left.\nS = {${preset.nodes.map((_, i) => i).filter(i => S[i]).map(name).join(", ")}}\n`
        + `cut: ${cut.map(e => fmt(e.c)).join(" + ") || "0"} = ${fmt(cutCost(edges, S))}\nflow = ${fmt(ek.total)} = cut`;
    }
    ro.textContent = txt;
    nextBtn.disabled = done;
    nextBtn.style.opacity = done ? "0.45" : "1";
    tBox.parentElement!.style.opacity = done ? "1" : "0.45";
    resBox.checked = residual;
  }

  draw();
}

autoMount(NAME, mount);
