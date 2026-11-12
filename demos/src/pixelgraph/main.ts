// A small image drawn as its graph, the bridge between the by-hand construction and real
// images. Pixels (grey levels 0..3) are the nodes; n-links between 4-neighbours have
// thickness lambda * w_pq; t-links carry the data costs D_p(0) = I_p^2 (to s, cut when p
// is background) and D_p(1) = (3 - I_p)^2 (to t, cut when p is object), i.e. two Gaussians
// with means 0 and 3. Small images (up to 4 pixels) show the full graph with s and t and
// every capacity; 6 x 6 images show the t-links as bars. The minimum cut is computed by
// max-flow (lib/maxflow.ts via lib/graph.ts). Click a pixel to change its grey level, or
// switch to "labels" and choose your own cut to compare its cost with the minimum.
// Auto-mounts on <div data-demo="pixelgraph">. Optional: data-preset="row|square2|blob|line|noisy".
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import { gridEdges, contrastWeights, solveSeg, segEnergy, SegProblem, EnergySplit } from "../lib/graph";

const NAME = "pixelgraph";
const W = 720, H = 540;
const COL = { obj: "#ff7f0e", bg: "#1f77b4", cut: "#d62728", n: "#666" };
const SIGMAS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, Infinity];

interface Preset { id: string; label: string; rows: number; cols: number; I: number[]; lambda: number; sigma: number }
const PRESETS: Preset[] = [
  { id: "row", label: "1 × 3", rows: 1, cols: 3, I: [3, 1, 3], lambda: 1, sigma: Infinity },
  { id: "square2", label: "2 × 2", rows: 2, cols: 2, I: [3, 2, 1, 0], lambda: 2, sigma: Infinity },
  {
    id: "blob", label: "6 × 6 blob", rows: 6, cols: 6, lambda: 1.5, sigma: Infinity, I: [
      0, 0, 1, 0, 0, 0,
      0, 3, 3, 3, 0, 0,
      0, 3, 1, 3, 2, 0,
      0, 3, 3, 3, 0, 0,
      0, 2, 3, 3, 0, 2,
      0, 0, 0, 0, 0, 0],
  },
  {
    id: "line", label: "thin vs thick", rows: 6, cols: 6, lambda: 1, sigma: Infinity, I: [
      0, 2, 0, 2, 2, 2,
      0, 2, 0, 2, 2, 2,
      0, 2, 0, 2, 2, 2,
      0, 2, 0, 2, 2, 2,
      0, 2, 0, 2, 2, 2,
      0, 2, 0, 2, 2, 2],
  },
  {
    id: "noisy", label: "noisy edge", rows: 6, cols: 6, lambda: 1, sigma: 1, I: [
      0, 1, 0, 2, 3, 3,
      1, 0, 0, 3, 3, 2,
      0, 0, 1, 3, 2, 3,
      0, 2, 0, 3, 3, 3,
      1, 0, 0, 2, 3, 1,
      0, 0, 1, 3, 3, 3],
  },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .pg-plot { flex: 1 1 480px; min-width: 0; max-width: 680px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .45em; }
[data-demo="${NAME}"] .pg-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] svg text { font-size: 20px; }
[data-demo="${NAME}"] svg .cap { font-size: 17px; font-family: ui-monospace, monospace; }
[data-demo="${NAME}"] svg .px { cursor: pointer; }
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

const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2).replace(/0$/, ""));
const greyOf = (v: number) => { const g = Math.round(30 + 205 * v / 3); return `rgb(${g},${g},${g})`; };

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[2];
  const st = { rows: 0, cols: 0, I: [] as number[], lambda: 1, sigma: Infinity, mode: "value" as "value" | "label", user: new Uint8Array(0) };

  const plot = htmlEl("div", "pg-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, plot);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pg-lab", "image", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const lamS = slider(panel, "λ", 0, 5, 0.1, 1, v => { st.lambda = v; draw(); });
  const sigS = slider(panel, "σ", 0, SIGMAS.length - 1, 1, SIGMAS.length - 1,
    v => { st.sigma = SIGMAS[v]; draw(); }, v => (Number.isFinite(SIGMAS[v]) ? String(SIGMAS[v]) : "∞"));
  htmlEl("div", "pg-lab", "a click on a pixel changes its", panel);
  const modeBtns = presets(panel, [
    { label: "grey level", apply: () => { st.mode = "value"; draw(); } },
    { label: "label (your cut)", apply: () => { st.mode = "label"; draw(); } },
  ]);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);
  button(panel, "reset", () => load(p0));

  function load(p: Preset): void {
    st.rows = p.rows; st.cols = p.cols; st.I = p.I.slice(); st.lambda = p.lambda; st.sigma = p.sigma;
    lamS.set(p.lambda); sigS.set(SIGMAS.indexOf(p.sigma));
    st.mode = "value";
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    st.user = new Uint8Array(0);
    draw();
  }

  function problem(): SegProblem {
    const n = st.rows * st.cols;
    const E = gridEdges(st.rows, st.cols, 4);
    return {
      n, E, w: contrastWeights(st.I, E, st.sigma), lambda: st.lambda,
      D0: st.I.map(v => v * v), D1: st.I.map(v => (3 - v) ** 2),
    };
  }

  function draw(): void {
    modeBtns.forEach((b, i) => b.classList.toggle("active", i === (st.mode === "value" ? 0 : 1)));
    const P = problem();
    const res = solveSeg(P);
    if (st.user.length !== P.n || st.mode === "value") st.user = res.L.slice();
    const L = st.mode === "label" ? st.user : res.L;
    const small = P.n <= 4;
    svg.replaceChildren();

    // layout
    const pitch = small ? 190 : 84, sq = small ? 84 : 56;
    const pitchX = small && st.rows > 1 ? 280 : pitch;
    const gw = (st.cols - 1) * pitchX, gh = (st.rows - 1) * pitch;
    const cx = W / 2, cy = small ? H / 2 : H / 2 - 4;
    const pos = (p: number): [number, number] => [cx - gw / 2 + (p % st.cols) * pitchX, cy - gh / 2 + Math.floor(p / st.cols) * pitch];
    const sPos: [number, number] = [cx, 38], tPos: [number, number] = [cx, H - 38];

    const gLinks = svgEl("g", {}, svg), gPix = svgEl("g", {}, svg), gCap = svgEl("g", {}, svg);
    const capLabel = (x: number, y: number, text: string, color: string) => {
      const w = 10.5 * text.length + 10;
      svgEl("rect", { x: x - w / 2, y: y - 13, width: w, height: 25, rx: 5, fill: "#fff", stroke: color, "stroke-width": 1.3 }, gCap);
      const t = svgEl("text", { x, y: y + 6, "text-anchor": "middle", class: "cap" }, gCap);
      t.textContent = text; t.style.fill = color;
    };

    // t-links (full graph only)
    if (small) {
      for (let p = 0; p < P.n; p++) {
        const [x, y] = pos(p);
        for (const [term, cap, isCut, color] of [
          [sPos, P.D0[p], L[p] === 0, COL.obj], [tPos, P.D1[p], L[p] === 1, COL.bg],
        ] as [[number, number], number, boolean, string][]) {
          // bend links that would pass behind another pixel outwards
          const far = Math.abs(term[1] - y) > pitch;
          const bx = far ? x + Math.sign(x - cx) * 190 : (x + term[0]) / 2;
          const by = far ? y + (term[1] - y) * 0.35 : (y + term[1]) / 2;
          const d = `M${term[0]},${term[1]} Q${bx},${by} ${x},${y}`;
          svgEl("path", {
            d, fill: "none", stroke: isCut ? COL.cut : color, "stroke-width": isCut ? 4 : 2.5,
            "stroke-dasharray": isCut ? "9 6" : "", opacity: cap === 0 ? 0.25 : 1,
          }, gLinks);
          const lx = (term[0] + 2 * bx + x) / 4, ly = (term[1] + 2 * by + y) / 4;
          capLabel(lx, ly, fmt(cap), isCut ? COL.cut : color);
        }
      }
      for (const [tp, nm, color] of [[sPos, "s", COL.obj], [tPos, "t", COL.bg]] as [[number, number], string, string][]) {
        svgEl("circle", { cx: tp[0], cy: tp[1], r: 26, fill: color }, gPix);
        const t = svgEl("text", { x: tp[0], y: tp[1] + 8, "text-anchor": "middle", "font-weight": 700 }, gPix);
        t.textContent = nm; t.style.fill = "#fff";
      }
      svgEl("text", { x: sPos[0] + 36, y: sPos[1] + 7 }, gPix).textContent = "object";
      svgEl("text", { x: tPos[0] + 36, y: tPos[1] + 7 }, gPix).textContent = "background";
    }

    // n-links
    for (let k = 0; k < P.E.p.length; k++) {
      const a = P.E.p[k], b = P.E.q[k];
      const [x1, y1] = pos(a), [x2, y2] = pos(b);
      const c = P.lambda * P.w[k];
      const isCut = L[a] !== L[b];
      const horiz = y1 === y2;
      const off = sq / 2 + 3;
      const sx = horiz ? x1 + off : x1, sy = horiz ? y1 : y1 + off, ex = horiz ? x2 - off : x2, ey = horiz ? y2 : y2 - off;
      svgEl("line", {
        x1: sx, y1: sy, x2: ex, y2: ey, stroke: isCut ? COL.cut : COL.n,
        "stroke-width": Math.max(1, Math.min(16, 1 + 3 * c)), "stroke-dasharray": isCut ? "7 5" : "",
        opacity: c < 0.05 ? 0.3 : 1,
      }, gLinks);
      if (small) capLabel((sx + ex) / 2, (sy + ey) / 2, fmt(c), isCut ? COL.cut : "#444");
    }

    // pixels
    for (let p = 0; p < P.n; p++) {
      const [x, y] = pos(p);
      const g = svgEl("g", { class: "px" }, gPix);
      svgEl("rect", {
        x: x - sq / 2, y: y - sq / 2, width: sq, height: sq, rx: 6, fill: greyOf(st.I[p]),
        stroke: L[p] ? COL.obj : COL.bg, "stroke-width": small ? 7 : 6,
      }, g);
      const t = svgEl("text", { x, y: y + 8, "text-anchor": "middle", "font-weight": 700 }, g);
      t.textContent = String(st.I[p]);
      t.style.fill = st.I[p] >= 2 ? "#000" : "#fff";
      if (!small) {
        // t-link bars: left = s -> p (D_p(0), orange), right = p -> t (D_p(1), blue); the cut one is solid
        const hmax = sq - 10;
        for (const [k, cap, color, isCut] of [[0, P.D0[p], COL.obj, L[p] === 0], [1, P.D1[p], COL.bg, L[p] === 1]] as [number, number, string, boolean][]) {
          const h = hmax * Math.min(1, cap / 9);
          const bx = k === 0 ? x - sq / 2 + 4 : x + sq / 2 - 11;
          svgEl("rect", {
            x: bx, y: y + sq / 2 - 5 - h, width: 7, height: h, fill: isCut ? color : "#fff",
            stroke: color, "stroke-width": 1.5, opacity: isCut ? 1 : 0.8,
          }, g);
        }
      }
      g.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        if (st.mode === "value") st.I[p] = (st.I[p] + 1) % 4;
        else st.user[p] = st.user[p] ? 0 : 1;
        presetBtns.forEach(b => b.classList.remove("active"));
        draw();
      });
    }

    // readout
    const eMin = res.energy, eUser = segEnergy(P, st.user);
    const r = (name: string, a: string, b: string) => name.padEnd(12) + a.padStart(9) + b.padStart(8);
    const rows = (e: EnergySplit) => [fmt(e.data), fmt(e.boundary), fmt(e.smooth), fmt(e.total)];
    const m = rows(eMin), u = rows(eUser);
    const lines = [r("", "min cut", st.mode === "label" ? "yours" : "")];
    ["data", "Σ w (cut)", "λ·Σ w", "E = cut"].forEach((nm, i) => lines.push(r(nm, m[i], st.mode === "label" ? u[i] : "")));
    lines.push(`max flow = ${fmt(res.flow)}`);
    ro.textContent = lines.join("\n");
    note.textContent = (small
      ? "Capacities: s → p is D(0) = I², p → t is D(1) = (3 − I)², n-links λ·w. Dashed red: the cut. "
      : "Bars in each pixel: left (orange) s → p = D(0) = I², right (blue) p → t = D(1) = (3 − I)²; the solid bar is cut. Line width: λ·w. ")
      + "Orange frame = object, blue = background. w = exp(−ΔI²/2σ²).";
  }

  load(p0);
}

autoMount(NAME, mount);
