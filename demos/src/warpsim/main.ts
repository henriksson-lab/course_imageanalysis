// A simulated GPU multiprocessor: warps of 32 lanes and branch divergence (tab "divergence"),
// and latency hiding with many resident warps (tab "latency"). A model with the rules of a
// real GPU (lib/gpusim.ts), not a measurement.
//
// Divergence: a 64 x 32 image tiled into warps (rows of 32 or 8 x 4 tiles); a kernel
// "if pixel > t: A (expensive) else B (cheap)". Each warp is coloured by its state (all A,
// all B, mixed); click a warp to see its active-lane masks. Loop mode: a per-pixel trip
// count; the warp waits for its slowest lane.
// Latency: N resident warps alternate k compute cycles and a load of latency L; the issue
// slots of one multiprocessor as a Gantt chart; utilisation vs Little's law.
//
// <div data-demo="warpsim" data-tab="divergence|latency" data-preset="..."></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, Slider } from "../lib/ui";
import { nuclei, checkerboard } from "../lib/synth";
import { Grey, createGrey } from "../lib/imageops";
import * as G from "../lib/gpusim";

const NAME = "warpsim";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { flex-wrap: nowrap; gap: 1em; }
${S} .wp-main { flex: 1 1 640px; min-width: 0; }
${S} .wp-main svg { width: 100%; display: block; }
${S} .qb-panel { flex: 0 1 330px; gap: .4em; }
${S} .wp-tabs { display: flex; gap: .3em; margin-bottom: .3em; }
${S} .wp-lab { font-size: 14px; color: var(--qb-muted); margin-top: .2em; }
${S} .qb-readout { font-size: 15px; }
${S} .wp-hidden { display: none !important; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .qb-panel { flex: 0 0 340px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .4em; }
.reveal ${S} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${S} .wp-lab { font-size: 15px; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

const W = 64, H = 32;
const COL_A = "#ff9f1c", COL_B = "#5b8ff9", COL_MIX = "#d62728";

type ImageKind = "nuclei" | "noise" | "uniform" | "blocks";
interface State {
  tab: "divergence" | "latency";
  image: ImageKind; t: number; cA: number; cB: number; tiles: boolean; loop: boolean; slow: boolean; warp: number;
  N: number; k: number; L: number; regs: number;
}
const DEFAULT: State = { tab: "divergence", image: "nuclei", t: 115, cA: 100, cB: 10, tiles: false, loop: false, slow: false, warp: 20, N: 4, k: 4, L: 200, regs: 32 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "uniform", label: "uniform image: no divergence", s: { tab: "divergence", image: "uniform" } },
  { id: "noise", label: "noise: every warp diverges", s: { tab: "divergence", image: "noise", t: 128 } },
  { id: "rows", label: "nuclei, warps = rows", s: { tab: "divergence", image: "nuclei", tiles: false } },
  { id: "tiles", label: "nuclei, warps = 8×4 tiles", s: { tab: "divergence", image: "nuclei", tiles: true } },
  { id: "slow", label: "one slow lane holds 31 hostage", s: { tab: "divergence", image: "uniform", loop: true, slow: true } },
  { id: "one", label: "one warp cannot hide memory", s: { tab: "latency", N: 1, k: 4, L: 200 } },
  { id: "enough", label: "enough warps hide it", s: { tab: "latency", N: 32, k: 8, L: 200 } },
  { id: "regs", label: "too many registers: occupancy 25 %", s: { tab: "latency", N: 32, k: 8, L: 200, regs: 255 } },
];

function makeImage(kind: ImageKind): Grey {
  if (kind === "nuclei") return nuclei({ width: W, height: H, count: 5, radius: [4.5, 7], noise: 5, seed: 4, touching: 0 }).image;
  if (kind === "blocks") return checkerboard(W, H, 8, 60, 180);
  const g = createGrey(W, H, 60);
  if (kind === "noise") { let s = 12345; for (let i = 0; i < W * H; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; g.data[i] = (s >>> 8) % 256; } }
  return g;
}

/** Trip counts per pixel for the loop mode: inside pixels iterate more (seeded), or (slow)
 *  every pixel 1 trip except one lane in every warp-row band with 32 trips. */
function makeTrips(inside: Uint8Array, slow: boolean): Int32Array {
  const t = new Int32Array(W * H);
  let s = 99;
  for (let i = 0; i < W * H; i++) {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    t[i] = slow ? 1 : inside[i] ? 4 + ((s >>> 8) % 8) : 1 + ((s >>> 8) % 2);
  }
  if (slow) for (let y = 0; y < H; y += 4) for (let x = 5; x < W; x += 16) t[y * W + x + (y % 3)] = 32;
  return t;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.tab === "latency" ? { tab: "latency" as const } : {}), ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "wp-main", "", host);
  const tabs = htmlEl("div", "wp-tabs", "", main);
  const tabBtns = [button(tabs, "divergence", () => { st.tab = "divergence"; draw(); }), button(tabs, "latency hiding", () => { st.tab = "latency"; draw(); })];
  const svg = svgEl("svg", { viewBox: "0 0 660 405", role: "img", "aria-label": "simulated warps" }, main);

  const panel = htmlEl("div", "qb-panel", "", host);
  const divBox = htmlEl("div", "", "", panel), latBox = htmlEl("div", "", "", panel);
  const presetRow = (box: HTMLElement, tab: State["tab"]) => {
    htmlEl("div", "wp-lab", "presets", box);
    return presets(box, PRESETS.filter(p => p.s.tab === tab).map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); draw(); } })));
  };
  const pBtns = [...presetRow(divBox, "divergence"), ...presetRow(latBox, "latency")];
  // divergence controls
  htmlEl("div", "wp-lab", "image", divBox);
  const imgBtns = presets(divBox, (["nuclei", "noise", "uniform", "blocks"] as ImageKind[]).map(k => ({ label: k, apply: () => { st.image = k; draw(); } })));
  const tS = slider(divBox, "threshold t", 0, 255, 1, st.t, v => { st.t = v; draw(); });
  const aS = slider(divBox, "cost A (inside)", 1, 200, 1, st.cA, v => { st.cA = v; draw(); });
  const bS = slider(divBox, "cost B (outside)", 1, 200, 1, st.cB, v => { st.cB = v; draw(); });
  const tileC = checkbox(divBox, "warps as 8×4 tiles (else rows of 32)", st.tiles, v => { st.tiles = v; draw(); });
  const loopC = checkbox(divBox, "loop until converged (trip count per pixel)", st.loop, v => { st.loop = v; draw(); });
  // latency controls
  const nS = slider(latBox, "warps N", 1, 32, 1, st.N, v => { st.N = v; draw(); });
  const kS = slider(latBox, "compute k", 1, 40, 1, st.k, v => { st.k = v; draw(); });
  const lS = slider(latBox, "latency L", 20, 600, 10, st.L, v => { st.L = v; draw(); });
  const rS = slider(latBox, "registers/thread", 32, 255, 1, st.regs, v => { st.regs = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); pBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function sync(): void {
    const ss: [Slider, number][] = [[tS, st.t], [aS, st.cA], [bS, st.cB], [nS, st.N], [kS, st.k], [lS, st.L], [rS, st.regs]];
    for (const [s, v] of ss) s.set(v);
    tileC.checked = st.tiles; loopC.checked = st.loop;
  }

  let img: Grey | null = null, imgKind: ImageKind | null = null;

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.tab === "divergence")));
    divBox.classList.toggle("wp-hidden", st.tab !== "divergence");
    latBox.classList.toggle("wp-hidden", st.tab !== "latency");
    svg.replaceChildren();
    if (st.tab === "divergence") drawDivergence(); else drawLatency();
  }

  function txt(x: number, y: number, s: string, size = 18, anchor = "start", fill = "var(--qb-fg)", weight = 400): SVGTextElement {
    const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": size, "font-weight": weight }, svg);
    t.style.fill = fill; t.textContent = s; return t;
  }

  function drawDivergence(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", ["nuclei", "noise", "uniform", "blocks"][i] === st.image));
    if (imgKind !== st.image) { img = makeImage(st.image); imgKind = st.image; }
    const im = img!;
    const inside = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) inside[i] = im.data[i] > st.t ? 1 : 0;
    const [tw, th] = st.tiles ? [8, 4] : [32, 1];
    const warps = G.warpsOf(W, H, tw, th);
    const c = 8, ox = 10, oy = 30;
    txt(ox, 20, `image ${W}×${H}: ${warps.length} warps of 32 pixels`, 18);
    // pixels
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const v = Math.max(0, Math.min(255, im.data[y * W + x]));
      const r = svgEl("rect", { x: ox + x * c, y: oy + y * c, width: c, height: c }, svg);
      r.style.fill = `rgb(${v},${v},${v})`;
    }
    // per-warp result
    let trips: Int32Array | null = null;
    if (st.loop) trips = makeTrips(inside, st.slow);
    let cyc = 0, ideal = 0, useful = 0, issued = 0, mixed = 0;
    const res = warps.map(lanes => {
      if (trips) {
        const r = G.warpLoop(lanes.map(p => trips![p]), 10);
        cyc += r.cycles; ideal += r.useful / 32; useful += r.useful; issued += r.issued;
        const st2 = r.maxTrips === Math.min(...lanes.map(p => trips![p])) ? "A" : "mixed";
        if (st2 === "mixed") mixed++;
        return { state: st2 as "A" | "B" | "mixed", r };
      }
      const r = G.warpBranch(lanes.map(p => inside[p]), st.cA, st.cB);
      cyc += r.cycles; ideal += r.ideal; useful += r.useful; issued += r.issued; if (r.state === "mixed") mixed++;
      return { state: r.state, r };
    });
    if (st.warp >= warps.length) st.warp = 0;
    warps.forEach((lanes, wi) => {
      const x0 = lanes[0] % W, y0 = Math.floor(lanes[0] / W);
      const s = res[wi].state, col = trips ? (s === "mixed" ? COL_MIX : "#888") : s === "A" ? COL_A : s === "B" ? COL_B : COL_MIX;
      const r = svgEl("rect", { x: ox + x0 * c + 1, y: oy + y0 * c + 1, width: tw * c - 2, height: th * c - 2, rx: 2 }, svg);
      r.style.fill = s === "mixed" ? COL_MIX : "none"; r.style.fillOpacity = "0.18";
      r.style.stroke = col; r.style.strokeWidth = wi === st.warp ? "3.5" : "1.4"; r.style.cursor = "pointer";
      if (wi === st.warp) { r.style.stroke = "#ffd400"; }
      r.addEventListener("pointerdown", () => { st.warp = wi; draw(); });
    });
    // legend
    const ly = oy + H * c + 24;
    const leg: [string, string][] = trips ? [["#888", "all lanes finish together"], [COL_MIX, "lanes wait for the slowest"]] : [[COL_A, "all A"], [COL_B, "all B"], [COL_MIX, "mixed: runs A and B"]];
    let lx = ox;
    for (const [col, lab] of leg) {
      const r = svgEl("rect", { x: lx, y: ly - 13, width: 16, height: 16 }, svg); r.style.fill = col;
      txt(lx + 22, ly, lab, 18); lx += 40 + lab.length * 9.5;
    }
    // the selected warp: lane masks
    const lanes = warps[st.warp], my = ly + 12, sq = 18;
    txt(ox, my + 14, `warp ${st.warp} (yellow): active lanes per path`, 18, "start", "var(--qb-fg)", 600);
    if (trips) {
      const tr = lanes.map(p => trips![p]), mx = Math.max(...tr);
      txt(ox, my + 40, `trips per lane (max ${mx}, mean ${(tr.reduce((a, b) => a + b, 0) / 32).toFixed(1)})`, 18);
      tr.forEach((v, i) => {
        const h = 46 * v / 32;
        const r = svgEl("rect", { x: ox + 350 + i * 9, y: my + 50 - h, width: 8, height: Math.max(1, h) }, svg);
        r.style.fill = v === mx ? COL_MIX : "#888";
      });
    } else {
      const rows: [string, string, number, (p: number) => boolean][] = [["A", COL_A, st.cA, p => !!inside[p]], ["B", COL_B, st.cB, p => !inside[p]]];
      rows.forEach(([nm, col, cost, on], ri) => {
        const y = my + 24 + ri * (sq + 6);
        const any = lanes.some(on);
        txt(ox, y + 15, `${nm}: ${any ? cost + " cycles" : "skipped"}`, 18);
        lanes.forEach((p, i) => {
          const r = svgEl("rect", { x: ox + 160 + i * 15, y, width: 12, height: sq }, svg);
          r.style.fill = on(p) ? col : "var(--qb-panel)"; r.style.stroke = "#999"; r.style.strokeWidth = "0.6";
        });
      });
    }
    const wr = res[st.warp].r as G.WarpBranchResult & { maxTrips?: number };
    const eff = issued ? useful / issued : 1;
    ro.textContent =
      `all ${warps.length} warps, one after another:\n` +
      `  cycles issued      ${cyc}\n` +
      `  without divergence ${Math.round(ideal)}\n` +
      `  lane efficiency    ${(100 * eff).toFixed(1)} %\n` +
      `  divergent warps    ${mixed} of ${warps.length}\n` +
      `selected warp: ${wr.cycles} cycles` + (trips ? "" : ` (${wr.nA} lanes A, ${wr.nB} lanes B)`);
  }

  function drawLatency(): void {
    const maxW = G.residentWarps(st.regs), N = Math.min(st.N, maxW);
    const iters = 6;
    const r = G.schedule(N, st.k, st.L, iters);
    const ox = 120, oy = 34, gw = 520, gh = 270;
    const T = r.total, sx = gw / T, rh = Math.min(48, gh / Math.max(N, 1));
    txt(10, 22, `one multiprocessor, ${N} resident warp${N > 1 ? "s" : ""}` + (N < st.N ? ` (registers allow only ${maxW})` : ""), 18, "start", "var(--qb-fg)", 600);
    for (let w = 0; w < N; w++) {
      if (N <= 12 || w % Math.ceil(N / 8) === 0) txt(ox - 8, oy + (w + 0.5) * rh + 6, `warp ${w}`, 18, "end");
      for (const [a, b] of r.waits[w]) {
        const rr = svgEl("rect", { x: ox + a * sx, y: oy + w * rh + rh * 0.15, width: Math.max(0.5, (b - a) * sx), height: rh * 0.7 }, svg);
        rr.style.fill = "#ccd"; rr.style.opacity = "0.7";
      }
    }
    // issue slots: merge runs
    let t0 = 0;
    for (let t = 1; t <= T; t++) {
      if (t === T || r.issue[t] !== r.issue[t0]) {
        const w = r.issue[t0];
        if (w >= 0) { const rr = svgEl("rect", { x: ox + t0 * sx, y: oy + w * rh + rh * 0.1, width: Math.max(0.6, (t - t0) * sx), height: rh * 0.8 }, svg); rr.style.fill = COL_A; }
        t0 = t;
      }
    }
    // utilisation strip
    const uy = oy + Math.min(gh, N * rh) + 14;
    txt(ox - 8, uy + 16, "busy", 18, "end");
    t0 = 0;
    for (let t = 1; t <= T; t++) {
      if (t === T || (r.issue[t] >= 0) !== (r.issue[t0] >= 0)) {
        const rr = svgEl("rect", { x: ox + t0 * sx, y: uy, width: Math.max(0.5, (t - t0) * sx), height: 22 }, svg);
        rr.style.fill = r.issue[t0] >= 0 ? "#2ca02c" : "#e0e0e0";
        t0 = t;
      }
    }
    const ax = svgEl("line", { x1: ox, x2: ox + gw, y1: uy + 34, y2: uy + 34 }, svg); ax.style.stroke = "var(--qb-axis)";
    txt(ox, uy + 56, "0", 18, "middle"); txt(ox + gw, uy + 56, `${T} cycles`, 18, "end");
    const leg: [string, string][] = [[COL_A, "issuing"], ["#ccd", "waiting for memory"], ["#e0e0e0", "idle issue slot"]];
    const lx = [120, 250, 470];
    leg.forEach(([c, l], i) => { const rr = svgEl("rect", { x: lx[i], y: uy + 64, width: 16, height: 16 }, svg); rr.style.fill = c; txt(lx[i] + 22, uy + 78, l, 18); });
    const need = Math.ceil((st.k + st.L) / st.k);
    ro.textContent =
      `${N} warps × ${iters} × (k = ${st.k} compute, then a load of L = ${st.L})\n` +
      `  utilisation       ${(100 * r.utilisation).toFixed(1)} %\n` +
      `  Little's law      ${(100 * r.predicted).toFixed(1)} %  = min(1, N·k/(k+L))\n` +
      `  warps to hide L   ${need}  = (k+L)/k\n` +
      `  registers allow   ${maxW} warps (occupancy ${(100 * maxW / 32).toFixed(0)} %)`;
  }

  sync();
  draw();
}

autoMount(NAME, mount);
