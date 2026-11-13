// Memory coalescing and tiling, simulated (lib/gpusim.ts).
// Tab "access": the 32 lanes of a warp each read one float32; memory is served in 128-byte
// lines of four 32-byte sectors. Patterns: row (consecutive), column of a W-wide row-major
// image, stride s, the R of interleaved RGB, the R plane of planar RGB, random, and a
// transpose through a shared-memory tile. Touched sectors light up; readouts: lines,
// sectors, bytes moved and used, efficiency.
// Tab "tile": a k x k stencil on 32 x 32 outputs; direct global reads vs T x T tiles with a
// halo loaded once into shared memory. Heat map: how often each input pixel is read.
//
// <div data-demo="coalesce" data-tab="access|tile" data-preset="..." data-layout="full"></div>
import { autoMount, htmlEl, slider, presets, button, svgEl, Slider } from "../lib/ui";
import { viridis, toCss } from "../lib/colour";
import * as G from "../lib/gpusim";

const NAME = "coalesce";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { flex-wrap: nowrap; gap: 1em; }
${S} .co-main { flex: 1 1 660px; min-width: 0; }
${S} .co-main svg { width: 100%; display: block; }
${S} .qb-panel { flex: 0 1 320px; gap: .4em; }
${S} .co-tabs { display: flex; gap: .3em; margin-bottom: .3em; }
${S} .co-lab { font-size: 14px; color: var(--qb-muted); margin-top: .2em; }
${S} .qb-readout { font-size: 15px; }
${S} .co-hidden { display: none !important; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .qb-panel { flex: 0 0 330px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .4em; }
.reveal ${S} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${S} .co-lab { font-size: 15px; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

type Pattern = G.AccessPattern | "transpose";
const PATTERNS: [Pattern, string][] = [
  ["row", "row (consecutive)"], ["column", "column of a W-wide image"], ["stride", "stride s"],
  ["rgb", "interleaved RGB, read R"], ["planar", "planar, read R"], ["random", "random"], ["transpose", "transpose via tile"],
];
interface State { tab: "access" | "tile"; pattern: Pattern; W: number; s: number; k: number; T: number }
const DEFAULT: State = { tab: "access", pattern: "row", W: 1024, s: 2, k: 3, T: 16 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "rows", label: "rows: 100 %", s: { tab: "access", pattern: "row" } },
  { id: "columns", label: "columns: 12.5 %", s: { tab: "access", pattern: "column", W: 1024 } },
  { id: "rgb", label: "interleaved RGB: a third", s: { tab: "access", pattern: "rgb" } },
  { id: "tile16", label: "tile 16 with a 3×3 halo", s: { tab: "tile", k: 3, T: 16 } },
  { id: "k31", label: "31×31 kernel: the halo dominates small tiles", s: { tab: "tile", k: 31, T: 8 } },
];
const TILES = [0, 4, 8, 16, 32]; // 0 = direct

function mount(host: HTMLElement): void {
  injectStyle();
  const full = host.dataset.layout === "full";
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.tab === "tile" ? { tab: "tile" as const } : {}), ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "co-main", "", host);
  const tabs = htmlEl("div", "co-tabs", "", main);
  const tabBtns = [button(tabs, "warp access", () => { st.tab = "access"; draw(); }), button(tabs, "stencil tiles", () => { st.tab = "tile"; draw(); })];
  const svg = svgEl("svg", { viewBox: "0 0 680 420", role: "img", "aria-label": "memory access of a warp" }, main);

  const panel = htmlEl("div", "qb-panel", "", host);
  const accBox = htmlEl("div", "", "", panel), tileBox = htmlEl("div", "", "", panel);
  const presetRow = (box: HTMLElement, tab: State["tab"]) => {
    htmlEl("div", "co-lab", "presets", box);
    return presets(box, PRESETS.filter(p => p.s.tab === tab).map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); draw(); } })));
  };
  const pBtns = [...presetRow(accBox, "access"), ...presetRow(tileBox, "tile")];
  htmlEl("div", "co-lab", "access pattern", accBox);
  const patBtns = presets(accBox, PATTERNS.map(([p, l]) => ({ label: l, apply: () => { st.pattern = p; draw(); } })));
  const wS = slider(accBox, "image width W", 5, 12, 1, Math.log2(st.W), v => { st.W = 2 ** v; draw(); }, v => String(2 ** v));
  const sS = slider(accBox, "stride s", 1, 32, 1, st.s, v => { st.s = v; draw(); });
  const kS = slider(tileBox, "kernel k", 1, 31, 2, st.k, v => { st.k = v; draw(); });
  htmlEl("div", "co-lab", "output tile T × T", tileBox);
  const tBtns = presets(tileBox, TILES.map(T => ({ label: T ? String(T) : "none (direct)", apply: () => { st.T = T; draw(); } })));
  button(panel, "reset", () => { Object.assign(st, start()); pBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function sync(): void {
    const ss: [Slider, number][] = [[wS, Math.log2(st.W)], [sS, st.s], [kS, st.k]];
    for (const [s, v] of ss) s.set(v);
  }
  function txt(x: number, y: number, s: string, size = 18, anchor = "start", weight = 400): SVGTextElement {
    const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": size, "font-weight": weight }, svg);
    t.textContent = s; return t;
  }

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.tab === "access")));
    accBox.classList.toggle("co-hidden", st.tab !== "access");
    tileBox.classList.toggle("co-hidden", st.tab !== "tile");
    svg.replaceChildren();
    if (st.tab === "access") drawAccess(); else drawTile();
  }

  function drawAccess(): void {
    patBtns.forEach((b, i) => b.classList.toggle("active", PATTERNS[i][0] === st.pattern));
    wS.input.closest("label")!.classList.toggle("co-hidden", st.pattern !== "column");
    sS.input.closest("label")!.classList.toggle("co-hidden", st.pattern !== "stride");
    const pat: G.AccessPattern = st.pattern === "transpose" ? "row" : st.pattern;
    const addrs = G.accessAddresses(pat, { W: st.W, s: st.s, seed: 7 });
    const tr = G.transactions(addrs);
    // lanes
    const ox = 90, cw = 17.5, oy = 28;
    txt(10, oy + 18, "lanes", 18);
    for (let i = 0; i < 32; i++) {
      const r = svgEl("rect", { x: ox + i * cw, y: oy, width: cw - 3, height: 24, rx: 3 }, svg);
      r.style.fill = "#ff9f1c";
      if (full && i % 4 === 0) { const t = txt(ox + i * cw + (cw - 3) / 2, oy - 6, String(i), 13, "middle"); t.style.fill = "var(--qb-muted)"; }
    }
    // touched lines, sorted; each drawn as 32 float cells
    const lines = [...new Set(addrs.map(a => Math.floor(a / 128)))].sort((a, b) => a - b);
    const used = new Set(addrs.map(a => a / 4));
    const sectors = new Set(tr.sectorList);
    const my = oy + 96, avail = 250, lh = Math.min(26, avail / lines.length), gap = lines.length > 12 ? 1 : 4;
    txt(10, my - 12, "memory: touched 128-byte lines (4 sectors of 32 bytes each)", 18);
    lines.forEach((L, li) => {
      const y = my + li * lh;
      if (lines.length <= 12 || li % 4 === 0) { const t = txt(ox - 8, y + Math.min(lh, 20) - 4, `${L * 128}`, Math.min(16, lh + 6), "end"); t.style.fill = "var(--qb-muted)"; }
      for (let j = 0; j < 32; j++) {
        const e = L * 32 + j, sec = Math.floor(e * 4 / 32);
        const r = svgEl("rect", { x: ox + j * cw, y, width: cw - 1, height: Math.max(2, lh - gap) }, svg);
        r.style.fill = used.has(e) ? "#1f77b4" : sectors.has(sec) ? "#a9c8e8" : "var(--qb-panel)";
      }
      for (let j = 8; j < 32; j += 8) {
        const l = svgEl("line", { x1: ox + j * cw - 0.5, x2: ox + j * cw - 0.5, y1: y - 1, y2: y + Math.max(2, lh - gap) + 1 }, svg);
        l.style.stroke = "#333"; l.style.strokeWidth = "2";
      }
    });
    // arrows lane -> element
    addrs.forEach((a, i) => {
      const e = a / 4, li = lines.indexOf(Math.floor(e / 32)), j = e % 32;
      const ln = svgEl("line", { x1: ox + i * cw + (cw - 3) / 2, y1: oy + 24, x2: ox + j * cw + cw / 2, y2: my + li * lh }, svg);
      ln.style.stroke = "#ff9f1c"; ln.style.strokeWidth = "1.2"; ln.style.opacity = "0.6";
    });
    const ly = my + avail + 30;
    ([["#1f77b4", "requested"], ["#a9c8e8", "moved, not used"], ["var(--qb-panel)", "not moved"]] as [string, string][]).forEach(([c, l], i) => {
      const r = svgEl("rect", { x: ox + i * 200, y: ly - 14, width: 16, height: 16 }, svg); r.style.fill = c; r.style.stroke = "#999";
      txt(ox + 22 + i * 200, ly, l, 18);
    });
    const note = st.pattern === "transpose" ? "transpose via a shared-memory tile: read rows of a 32×32 tile (coalesced), write rows of the transposed tile (coalesced): both directions look like this.\n"
      : st.pattern === "column" ? `lane i reads pixel (0, i) of a ${st.W}-wide image: ${st.W * 4} bytes apart.\n` : "";
    ro.textContent = note +
      `lines touched   ${tr.lines}\n` +
      `sectors moved   ${tr.sectors}  = ${tr.moved} bytes\n` +
      `bytes used      ${tr.used}\n` +
      `efficiency      ${(100 * tr.efficiency).toFixed(1)} %`;
  }

  function drawTile(): void {
    tBtns.forEach((b, i) => b.classList.toggle("active", TILES[i] === st.T));
    const n = 32, T = st.T || null;
    const rc = G.readCounts(n, st.k, T);
    const m = rc.size, cell = Math.min(11, 360 / m), ox = 20, oy = 40;
    let mx = 0; for (const v of rc.counts) mx = Math.max(mx, v);
    txt(ox, 24, `reads from global memory per input pixel (32×32 outputs, halo ${(st.k - 1) / 2})`, 18);
    for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
      const v = rc.counts[y * m + x];
      const r = svgEl("rect", { x: ox + x * cell, y: oy + y * cell, width: cell + 0.3, height: cell + 0.3 }, svg);
      r.style.fill = v ? toCss(viridis(mx > 1 ? (v - 1) / (mx - 1) : 0.5)) : "var(--qb-panel)";
    }
    // the image border and tile borders
    const r0 = (st.k - 1) / 2;
    const b = svgEl("rect", { x: ox + r0 * cell, y: oy + r0 * cell, width: n * cell, height: n * cell }, svg);
    b.style.fill = "none"; b.style.stroke = "#fff"; b.style.strokeWidth = "2"; b.style.strokeDasharray = "5 3";
    if (T) for (let t = T; t < n; t += T) {
      for (const [x1, y1, x2, y2] of [[t, 0, t, n], [0, t, n, t]]) {
        const l = svgEl("line", { x1: ox + (r0 + x1) * cell, y1: oy + (r0 + y1) * cell, x2: ox + (r0 + x2) * cell, y2: oy + (r0 + y2) * cell }, svg);
        l.style.stroke = "#fff"; l.style.strokeWidth = "1";
      }
    }
    // colour bar
    const cx = ox + m * cell + 30, ch = 240;
    for (let i = 0; i < 40; i++) { const r = svgEl("rect", { x: cx, y: oy + ch - (i + 1) * ch / 40, width: 20, height: ch / 40 + 0.5 }, svg); r.style.fill = toCss(viridis((i + 0.5) / 40)); }
    txt(cx + 28, oy + 14, `${mx}`, 18); txt(cx + 28, oy + ch, `${mx > 1 ? 1 : 0}`, 18);
    txt(cx, oy + ch + 30, "reads", 18);
    const perOut = rc.total / (n * n);
    ro.textContent =
      `kernel ${st.k}×${st.k}, ${T ? `tiles ${T}×${T} (+ halo: ${T + st.k - 1}×${T + st.k - 1} loaded)` : "no tiles: every tap read from global memory"}\n` +
      `global reads per output pixel: ${perOut.toFixed(3)}\n` +
      (T ? `formula (T+k−1)²/T² = ${G.readsTiled(st.k, T).toFixed(3)}\n` : `formula k² = ${st.k * st.k}\n`) +
      `direct would be ${st.k * st.k}: ${(st.k * st.k / perOut).toFixed(1)}× more`;
  }

  sync();
  draw();
}

autoMount(NAME, mount);
