// Stitching a 3 × 3 mosaic (FFT registration day). Nine 128² tiles of a micrograph (enlarged
// to 384², honest caption) at jittered stage positions. Pairwise shifts of neighbours by
// phase correlation and the four candidates; tile positions from the stage, by chaining
// pairwise shifts along a path or a tree, or by global least squares on the tile graph.
// Click an edge of the graph to see that pair's correlation surface and four candidates.
// Options: data-preset="<id>".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { drawScalarImage, inferno } from "../lib/colour";
import { fftshift } from "../lib/fft";
import { solveGlobal, maxSpanningTree, chainPositions, TileEdge } from "../lib/registration";
import { makeTiles, registerPairs, DEFAULTS, StitchSettings, Tiles, PairResult, EDGES, T, BIG, NT, nominal, positionErrors, weighted, SNAKE } from "./model";

const NAME = "stitch";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
[data-demo="${NAME}"] .st-stack { position: relative; width: 100%; }
[data-demo="${NAME}"] .st-stack canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .st-stack svg { position: absolute; inset: 0; width: 100%; height: 100%; }
[data-demo="${NAME}"] .st-main { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .st-pair { flex: 0 1 230px; min-width: 190px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .st-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .st-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .st-edge { cursor: pointer; }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Mode = "stage" | "chain" | "tree" | "global";
interface Preset { id: string; label: string; s: Partial<StitchSettings>; mode: Mode; blend?: boolean; edge?: number }
const PRESETS: Preset[] = [
  { id: "stage", label: "stage positions: seams", s: {}, mode: "stage" },
  { id: "chain", label: "chained pairs: errors add up", s: { noise: 12 }, mode: "chain" },
  { id: "global", label: "global: errors spread", s: { noise: 12 }, mode: "global" },
  { id: "empty", label: "an empty tile", s: { empty: 4 }, mode: "chain", edge: 2 },
  { id: "wrong", label: "a wrong pair", s: { badPair: 3 }, mode: "global", edge: 3 },
  { id: "four", label: "the four candidates", s: {}, mode: "global", edge: 1 },
  { id: "blend", label: "global + blending", s: {}, mode: "global", blend: true },
];

const DEF: StitchSettings = { ...DEFAULTS, vignette: 0.15 };

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { s: { ...DEF }, mode: "stage" as Mode, blend: false, thr: 0.7, sel: 0, dropped: new Set<number>() };
  let base: Float64Array | null = null;

  const main = htmlEl("div", "st-main", "", host);
  const cap = htmlEl("figcaption", "", "", main);
  const stk = htmlEl("div", "st-stack", "", main);
  const cv = htmlEl("canvas", "", "", stk);
  cv.width = BIG; cv.height = BIG;
  const svg = svgEl("svg", { viewBox: `0 0 ${BIG} ${BIG}` }, stk);
  htmlEl("div", "st-credit", "Tiles cut from a micrograph enlarged 1.5× to 384 × 384. " + CELL_CREDIT, main);
  const pair = htmlEl("div", "st-pair", "", host);
  const pcap = htmlEl("figcaption", "", "", pair);
  const pstk = htmlEl("div", "st-stack", "", pair);
  const pcv = htmlEl("canvas", "", "", pstk);
  const psvg = svgEl("svg", { viewBox: `${-T / 2 - 0.5} ${-T / 2 - 0.5} ${T} ${T}`, preserveAspectRatio: "none" }, pstk);
  const pro = htmlEl("div", "qb-readout", "", pair);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "st-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "st-lab", "tile positions from", panel);
  const modes: [Mode, string][] = [["stage", "stage"], ["chain", "chain 0-1-2-5-4-3-6-7-8"], ["tree", "tree (most confident)"], ["global", "global least squares"]];
  const modeBtns = presets(panel, modes.map(([m, l]) => ({ label: l, apply: () => { st.mode = m; draw(); } })));
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "drop worst edge", () => dropWorst());
  button(row, "restore edges", () => { st.dropped.clear(); draw(); });
  const blendC = checkbox(panel, "blend the seams (feathering)", false, v => { st.blend = v; draw(); });
  const noiseS = slider(panel, "noise σ", 0, 30, 1, st.s.noise, v => { st.s.noise = v; rebuild(); });
  const vigS = slider(panel, "vignetting", 0, 0.5, 0.05, st.s.vignette, v => { st.s.vignette = v; rebuild(); });
  const thrS = slider(panel, "trust NCC ≥", 0, 1, 0.05, st.thr, v => { st.thr = v; draw(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Graph: one node per tile, one edge per overlapping pair, green if its overlap NCC is trusted, red dashed if not. Click an edge. Errors are distances (px) from the true positions, after matching tile 0.", panel);

  function load(p: Preset): void {
    st.s = { ...DEF, ...p.s }; st.mode = p.mode; st.blend = !!p.blend; st.sel = p.edge ?? 0; st.thr = 0.7; st.dropped.clear();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    noiseS.set(st.s.noise); vigS.set(st.s.vignette); thrS.set(st.thr); blendC.checked = st.blend;
    rebuild();
  }

  let tiles: Tiles | null = null, pairs: PairResult[] = [], ms = 0;
  let pending = false;
  function rebuild(): void {
    if (!base) { loadCell(BIG).then(g => { base = Float64Array.from(g.data); rebuild(); }); return; }
    if (pending) return;
    pending = true;
    cap.textContent = "registering 12 pairs…";
    setTimeout(() => {
      pending = false;
      tiles = makeTiles(st.s, base);
      const r = registerPairs(tiles);
      pairs = r.edges; ms = r.ms;
      draw();
    }, 20);
  }

  function edgesNow(): TileEdge[] { return weighted(pairs, st.thr, st.dropped); }

  function positions(): { x: Float64Array; y: Float64Array; res: number[] | null } {
    const E = edgesNow();
    if (st.mode === "stage") {
      const x = new Float64Array(NT), y = new Float64Array(NT);
      for (let k = 0; k < NT; k++) { const [a, b] = nominal(k); x[k] = a; y[k] = b; }
      return { x, y, res: null };
    }
    if (st.mode === "chain" || st.mode === "tree") {
      const tree = st.mode === "chain" ? SNAKE : maxSpanningTree(NT, E.map((e, k) => ({ ...e, w: st.dropped.has(k) ? -1 : e.w })));
      const c = chainPositions(NT, E, tree);
      return { ...c, res: null };
    }
    const g = solveGlobal(NT, E);
    if (!g) return { x: new Float64Array(NT), y: new Float64Array(NT), res: null };
    return { x: g.x, y: g.y, res: g.rx.map((r, k) => Math.hypot(r, g.ry[k])) };
  }

  function dropWorst(): void {
    if (st.mode !== "global") { st.mode = "global"; }
    const p = positions();
    if (!p.res) return;
    let worst = -1;
    p.res.forEach((r, k) => { if (!st.dropped.has(k) && pairs[k].ncc >= st.thr && (worst < 0 || r > p.res![worst])) worst = k; });
    if (worst >= 0) { st.dropped.add(worst); st.sel = worst; }
    draw();
  }

  function draw(): void {
    if (!tiles) return;
    modeBtns.forEach((b, i) => b.classList.toggle("active", modes[i][0] === st.mode));
    const P = positions(), t = tiles;
    // absolute placement: estimated positions relative to tile 0, anchored at its true position
    const ax = st.mode === "stage" ? 0 : t.pos[0][0] - P.x[0], ay = st.mode === "stage" ? 0 : t.pos[0][1] - P.y[0];
    const X = Array.from(P.x, v => v + ax), Y = Array.from(P.y, v => v + ay);
    // composite
    const acc = new Float64Array(BIG * BIG), wsum = new Float64Array(BIG * BIG);
    const feather = (u: number) => Math.min(1, (Math.min(u, T - 1 - u) + 1) / 24);
    for (let k = 0; k < NT; k++) {
      const ox = Math.round(X[k]), oy = Math.round(Y[k]), tile = t.tiles[k];
      if (!Number.isFinite(ox) || !Number.isFinite(oy)) continue;
      for (let y = 0; y < T; y++) {
        const Yy = oy + y;
        if (Yy < 0 || Yy >= BIG) continue;
        for (let x = 0; x < T; x++) {
          const Xx = ox + x;
          if (Xx < 0 || Xx >= BIG) continue;
          const i = Yy * BIG + Xx;
          if (st.blend) { const w = feather(x) * feather(y); acc[i] += w * tile[y * T + x]; wsum[i] += w; }
          else { acc[i] = tile[y * T + x]; wsum[i] = 1; }
        }
      }
    }
    for (let i = 0; i < acc.length; i++) acc[i] = wsum[i] > 0 ? acc[i] / wsum[i] : NaN;
    drawScalarImage(cv, acc, BIG, BIG, { lo: 0, hi: 255, nan: [235, 235, 235, 255] });
    // graph overlay
    svg.replaceChildren();
    for (let k = 0; k < NT; k++) {
      const r = svgEl("rect", { x: X[k], y: Y[k], width: T, height: T, fill: "none" }, svg);
      r.style.stroke = "rgba(255,255,255,0.45)"; r.style.strokeWidth = "1";
    }
    const cx = (k: number) => X[k] + T / 2, cy = (k: number) => Y[k] + T / 2;
    EDGES.forEach(([i, j], k) => {
      const e = pairs[k], trusted = e.ncc >= st.thr && !st.dropped.has(k);
      const g = svgEl("g", { class: "st-edge" }, svg);
      const hit = svgEl("line", { x1: cx(i), y1: cy(i), x2: cx(j), y2: cy(j) }, g);
      hit.style.stroke = "transparent"; hit.style.strokeWidth = "16";
      const ln = svgEl("line", { x1: cx(i), y1: cy(i), x2: cx(j), y2: cy(j) }, g);
      ln.style.stroke = st.dropped.has(k) ? "#888" : trusted ? "#2ca02c" : "#d62728";
      ln.style.strokeWidth = k === st.sel ? "7" : "4";
      if (!trusted) ln.style.strokeDasharray = "8 6";
      if (k === st.sel) { const halo = ln.cloneNode() as SVGLineElement; halo.style.stroke = "#ffd400"; halo.style.strokeWidth = "11"; halo.style.strokeDasharray = ""; g.insertBefore(halo, ln); }
      g.addEventListener("click", () => { st.sel = k; draw(); });
    });
    for (let k = 0; k < NT; k++) {
      const c = svgEl("circle", { cx: cx(k), cy: cy(k), r: 11 }, svg);
      c.style.fill = "#fff"; c.style.stroke = "#333"; c.style.strokeWidth = "2";
      const tx = svgEl("text", { x: cx(k), y: cy(k) + 5.5, "text-anchor": "middle" }, svg);
      tx.textContent = String(k); tx.style.fontSize = "16px"; tx.style.fill = "#000"; tx.style.fontWeight = "700";
    }
    const modeName = { stage: "stage positions", chain: "chained along 0-1-2-5-4-3-6-7-8", tree: "chained along the most confident tree", global: "global least squares" }[st.mode];
    cap.textContent = `mosaic: ${modeName}`;
    // errors
    const err = positionErrors(t, P.x, P.y);
    const grid = [0, 1, 2].map(r => [0, 1, 2].map(c => err[3 * r + c]).map(v => (Number.isFinite(v) ? v.toFixed(1) : "–").padStart(6)).join("")).join("\n");
    const finite = err.filter(Number.isFinite);
    const rms = Math.sqrt(finite.reduce((a, b) => a + b * b, 0) / finite.length);
    const lines = [`position error per tile (px)`, grid, `max ${Math.max(...finite).toFixed(2)}   rms ${rms.toFixed(2)}`];
    if (P.res) {
      let w = 0;
      P.res.forEach((r, k) => { if (r > P.res![w]) w = k; });
      lines.push(`largest residual: edge ${EDGES[w][0]}-${EDGES[w][1]}, ${P.res[w].toFixed(1)} px`);
    }
    if (st.dropped.size) lines.push(`dropped: ${[...st.dropped].map(k => EDGES[k].join("-")).join(", ")}`);
    lines.push(`12 pairs registered in ${ms.toFixed(0)} ms`);
    ro.textContent = lines.join("\n");
    drawPair(P.res);
  }

  function drawPair(res: number[] | null): void {
    const k = st.sel, e = pairs[k], [i, j] = EDGES[k], t = tiles!;
    pcap.textContent = `pair ${i}-${j}: correlation surface`;
    const S = e.surface;
    drawScalarImage(pcv, fftshift(S.data, S.w, S.h), S.w, S.h, { map: inferno });
    psvg.replaceChildren();
    const pk = { x: ((e.candidates[0]?.tx ?? 0) % T + T + T / 2) % T - T / 2, y: ((e.candidates[0]?.ty ?? 0) % T + T + T / 2) % T - T / 2 };
    const c = svgEl("circle", { cx: pk.x, cy: pk.y, r: 5, fill: "none" }, psvg);
    c.style.stroke = "#00e5ff"; c.style.strokeWidth = "2.5"; c.setAttribute("vector-effect", "non-scaling-stroke");
    const tx = t.pos[j][0] - t.pos[i][0], ty = t.pos[j][1] - t.pos[i][1];
    const lines = [
      `true shift   (${tx.toFixed(1)}, ${ty.toFixed(1)})`,
      `measured     (${e.edge.dx.toFixed(1)}, ${e.edge.dy.toFixed(1)})`,
      `peak used: #${e.peakRank + 1}, height ${e.height.toFixed(3)}`,
      `candidates of that peak:`,
      `  shift       overlap   NCC`,
      ...e.candidates.map(c => `  ${`(${c.tx}, ${c.ty})`.padEnd(12)}${String(c.overlap).padStart(6)}  ${Number.isFinite(c.ncc) ? c.ncc.toFixed(2).padStart(5) : "    –"}`),
      `confidence (NCC) ${e.ncc.toFixed(2)}${e.ncc >= st.thr ? "" : "  not trusted"}`,
    ];
    if (res) lines.push(`residual after global: ${res[k].toFixed(2)} px`);
    pro.textContent = lines.join("\n");
  }

  load(p0);
}

autoMount(NAME, mount);
