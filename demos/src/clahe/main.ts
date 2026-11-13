// CLAHE: contrast-limited adaptive histogram equalisation. Left: the input with the tile grid
// and the output. Click a pixel: its four nearest tile centres are marked, and the inspector
// shows their four curves and the blended curve (with the bilinear weights), and the clicked
// tile's histogram with the clip line and the redistributed excess. Controls: tiles per side,
// clip limit β (or no clip), interpolation between tiles on/off, image. Readouts: largest
// contrast gain of the clicked tile's curve (≤ β), noise SD in a background box before and
// after, and two pixels with the same raw value that get different outputs.
// Auto-mounts on <div data-demo="clahe">. Options: data-preset (global|noclip|clahe|seams|
// small|large|beta1|same), data-source (fluo12|cell|foggy|faint).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import * as C from "../lib/contrast";

const NAME = "clahe";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${P} { gap: .7em; }
${P} figure { margin: 0; }
${P} figcaption { font-size: 16px; text-align: center; margin: 0 0 .1em; }
${P} .cl-left { flex: 1 1 430px; min-width: 300px; max-width: 470px; display: flex; flex-direction: column; gap: .25em; }
${P} .cl-imgs { display: flex; gap: .5em; }
${P} .cl-imgs figure { flex: 1 1 0; min-width: 0; }
${P} .cl-imgs canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; cursor: crosshair; touch-action: none; }
${P} .cl-mid { flex: 1 1 340px; min-width: 260px; max-width: 400px; display: flex; flex-direction: column; gap: .1em; }
${P} .qb-panel { flex: 0 1 260px; gap: .25em; }
${P} .qb-readout { font-size: 14px; }
${P} .cl-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${P} .cl-credit { font-size: 12px; color: var(--qb-muted); }
.reveal ${P} { font-size: 18px; }
.reveal ${P} figcaption, .reveal ${P} .qb-figure figcaption { font-size: 17px; }
.reveal ${P} .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal ${P} .cl-lab { font-size: 15px; }
.reveal ${P} .qb-panel { gap: .15em; }
.reveal ${P} .qb-readout { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Source = "fluo12" | "cell" | "foggy" | "faint";
interface State { source: Source; tiles: number; beta: number; noClip: boolean; interp: boolean; px: number; py: number }
const SOURCES: { id: Source; label: string }[] = [
  { id: "fluo12", label: "fluorescence 12 bit" }, { id: "faint", label: "faint cell" },
  { id: "foggy", label: "foggy scene" }, { id: "cell", label: "micrograph" },
];
const BASE: State = { source: "fluo12", tiles: 8, beta: 3, noClip: false, interp: true, px: 120, py: 175 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "global", label: "global (1 tile)", s: { source: "faint", tiles: 1, noClip: true, px: 180, py: 170 } },
  { id: "noclip", label: "no clip: the noise explodes", s: { source: "faint", noClip: true, px: 180, py: 170 } },
  { id: "clahe", label: "CLAHE β = 3", s: { source: "faint", px: 180, py: 170 } },
  { id: "seams", label: "no interpolation: seams", s: { interp: false } },
  { id: "small", label: "tiles too small", s: { tiles: 16 } },
  { id: "large", label: "tiles too large = global", s: { tiles: 2 } },
  { id: "beta1", label: "β = 1: nothing happens", s: { beta: 1 } },
  { id: "same", label: "same raw value, two outputs", s: { source: "foggy" } },
];
/** A background box per scene (x0, y0, size), away from all objects. */
const BOX: Record<Source, [number, number, number]> = { fluo12: [8, 216, 32], faint: [204, 16, 36], foggy: [8, 216, 32], cell: [8, 8, 24] };

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE };
  const N = C.SCENE_N;

  const left = htmlEl("div", "cl-left", "", host);
  const imgs = htmlEl("div", "cl-imgs", "", left);
  const f1 = htmlEl("figure", "", "", imgs), f2 = htmlEl("figure", "", "", imgs);
  htmlEl("figcaption", "", "input, tiles", f1);
  const cap2 = htmlEl("figcaption", "", "output", f2);
  const cv1 = htmlEl("canvas", "", "", f1), cv2 = htmlEl("canvas", "", "", f2);
  cv1.width = cv1.height = cv2.width = cv2.height = N;
  const ro = htmlEl("div", "qb-readout", "", left);
  const credit = htmlEl("div", "cl-credit", CELL_CREDIT, left);

  const mid = htmlEl("div", "cl-mid", "", host);
  const curves = new Plot(mid, { width: 400, height: 250, x: [0, 255], y: [0, 255], xlabel: "input value", ylabel: "output" });
  const th = new Plot(mid, { width: 400, height: 215, x: [0, 255], y: [0, 1], xlabel: "clicked tile: histogram", yticks: [], margin: { l: 20 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "cl-lab", "presets", panel);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "cl-lab", "settings", panel);
  const tS = slider(panel, "tiles per side", 1, 16, 1, st.tiles, v => { st.tiles = v; compute(); });
  const bS = slider(panel, "clip β", 1, 16, 0.5, st.beta, v => { st.beta = v; st.noClip = false; ncCb.checked = false; compute(); });
  const ncCb = checkbox(panel, "no clip (β = ∞)", st.noClip, v => { st.noClip = v; compute(); });
  const inCb = checkbox(panel, "interpolate between tiles", st.interp, v => { st.interp = v; compute(); });
  htmlEl("div", "cl-lab", "image", panel);
  const sRow = htmlEl("div", "qb-presets", "", panel);
  const sBtns = SOURCES.map(s => button(sRow, s.label, () => { st.source = s.id; prepare(); }));
  button(panel, "reset", () => load(p0));

  let cell: Float32Array | null = null;
  const scenes: Record<Exclude<Source, "cell">, Float32Array> = {
    fluo12: Float32Array.from(C.fluorescence12()), foggy: Float32Array.from(C.foggyScene()), faint: Float32Array.from(C.faintCellScene()),
  };
  loadCell().then(g => { cell = Float32Array.from(g.data, v => Math.round(v)); if (st.source === "cell") prepare(); });

  let img: { width: number; height: number; data: Float32Array } = { width: N, height: N, data: new Float32Array(N * N) };
  let lo = 0, hi = 1;
  function prepare(): void {
    const d = st.source === "cell" ? (cell ?? new Float32Array(N * N)) : scenes[st.source];
    img = { width: N, height: N, data: d };
    lo = Infinity; hi = -Infinity;
    for (const v of d) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi <= lo) hi = lo + 1;
    sync(); compute();
  }
  function sync(): void {
    tS.set(st.tiles); bS.set(st.beta); ncCb.checked = st.noClip; inCb.checked = st.interp;
    sBtns.forEach((b, i) => b.classList.toggle("active", SOURCES[i].id === st.source));
    credit.style.display = st.source === "cell" ? "" : "none";
  }
  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, BASE, p.s);
    pBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    prepare();
  }

  let res: C.ClaheResult | null = null;
  function compute(): void {
    res = C.clahe(img, { tilesX: st.tiles, tilesY: st.tiles, clip: st.noClip ? Infinity : st.beta, bins: 256, range: [lo, hi + 1e-6], interpolate: st.interp });
    draw();
  }

  const put = (cv: HTMLCanvasElement, g: (i: number) => number) => {
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) { const v = Math.max(0, Math.min(255, Math.round(g(i)))); id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
    ctx.putImageData(id, 0, 0);
    return ctx;
  };

  function draw(): void {
    if (!res) return;
    const r = res, d = img.data, T = st.tiles;
    const lin = (v: number) => 255 * (v - lo) / (hi - lo);
    const c1 = put(cv1, i => lin(d[i])), c2 = put(cv2, i => r.out.data[i]);
    // tile grid
    c1.strokeStyle = "rgba(31,119,180,0.85)"; c1.lineWidth = 1;
    for (let k = 1; k < T; k++) {
      const p = Math.round(k * N / T) + 0.5;
      c1.beginPath(); c1.moveTo(p, 0); c1.lineTo(p, N); c1.moveTo(0, p); c1.lineTo(N, p); c1.stroke();
    }
    // background box
    const [bx, by, bsz] = BOX[st.source];
    for (const c of [c1, c2]) { c.strokeStyle = "#2ca02c"; c.lineWidth = 2; c.strokeRect(bx + 0.5, by + 0.5, bsz, bsz); }
    // clicked pixel, its tile and the four nearest tile centres
    const ws = C.claheWeights(st.px, st.py, N, N, T, T);
    const own = C.tileOf(st.px, st.py, N, N, T, T);
    const tx = own % T, ty = Math.floor(own / T);
    c1.strokeStyle = "#d62728"; c1.lineWidth = 2;
    c1.strokeRect(tx * N / T + 1, ty * N / T + 1, N / T - 2, N / T - 2);
    const cols = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"];
    ws.forEach((w, k) => {
      const [cx, cy] = r.centres[w.tile];
      c1.fillStyle = cols[k % 4]; c1.beginPath(); c1.arc(cx + 0.5, cy + 0.5, 4, 0, 2 * Math.PI); c1.fill();
    });
    for (const c of [c1, c2]) { c.strokeStyle = "#ff7f0e"; c.lineWidth = 2; c.beginPath(); c.arc(st.px + 0.5, st.py + 0.5, 6, 0, 2 * Math.PI); c.stroke(); }
    // two pixels with the same raw bin and the most different outputs
    const mn = new Int32Array(r.bins).fill(-1), mx = new Int32Array(r.bins).fill(-1);
    for (let i = 0; i < N * N; i++) {
      const b = r.binOfPixel[i];
      if (mn[b] < 0 || r.out.data[i] < r.out.data[mn[b]]) mn[b] = i;
      if (mx[b] < 0 || r.out.data[i] > r.out.data[mx[b]]) mx[b] = i;
    }
    let bb = 0, bd = -1;
    for (let b = 0; b < r.bins; b++) if (mn[b] >= 0) { const dd = r.out.data[mx[b]] - r.out.data[mn[b]]; if (dd > bd) { bd = dd; bb = b; } }
    const pa = mn[bb], pb = mx[bb];
    if (bd >= 0.5) for (const c of [c1, c2]) for (const [p, col] of [[pa, "#e377c2"], [pb, "#17becf"]] as [number, string][]) {
      c.strokeStyle = col; c.lineWidth = 2; c.strokeRect((p % N) - 3.5, Math.floor(p / N) - 3.5, 8, 8);
    }
    cap2.textContent = st.tiles === 1 ? "output (one tile)" : "output";

    // curves of the four nearest tiles and the blend, against the input value
    const bc = Array.from({ length: r.bins }, (_, b) => lo + (b + 0.5) * (hi - lo) / r.bins);
    curves.setDomain([lo, hi], [0, 255]);
    curves.clear();
    let blend = new Float64Array(r.bins);
    ws.forEach((w, k) => {
      const lut = r.tileLuts[w.tile];
      curves.line(bc, Array.from(lut), { color: cols[k % 4], width: 2, opacity: 0.8, label: `w = ${w.w.toFixed(2)}` });
      for (let b = 0; b < r.bins; b++) blend[b] += w.w * lut[b];
    });
    if (!st.interp) blend = Float64Array.from(r.tileLuts[own]);
    curves.line(bc, Array.from(blend), { color: seriesColor(3), width: 4, label: st.interp ? "blend" : "own tile" });
    curves.line([lo, hi], [0, 255], { color: "var(--qb-muted, #999)", width: 1.5, dash: "4 4" });
    const v = d[st.py * N + st.px], b0 = r.binOfPixel[st.py * N + st.px];
    curves.line([v, v], [0, blend[b0]], { color: seriesColor(3), width: 2, dash: "5 4" });
    curves.legend(undefined, "br");

    // the clicked tile's histogram: raw, clip line, clipped + redistributed
    const raw = r.rawHists[own], clp = r.tileHists[own], Nt = r.tileN[own];
    const limit = st.noClip ? Infinity : st.beta * Nt / r.bins;
    const top = Math.max(...raw, Number.isFinite(limit) ? limit : 0) * 1.08 || 1;
    th.setDomain([lo, hi], [0, 1]);
    th.clear();
    const edges = Array.from({ length: r.bins + 1 }, (_, b) => lo + b * (hi - lo) / r.bins);
    th.histogram(edges, Array.from(raw, c => c / top), { color: "var(--qb-muted, #999)", fillOpacity: 0.35, width: 0.5, label: "tile histogram" });
    th.histogram(edges, Array.from(clp, c => c / top), { color: seriesColor(0), fillOpacity: 0.7, width: 0.5, label: "clipped + spread" });
    if (Number.isFinite(limit)) th.hline(limit / top, { color: seriesColor(1), width: 2.5, dash: "7 4", label: "clip limit" });
    th.legend(undefined, "tr");

    // readouts
    const gain = C.maxGain(clp, Nt);
    const box = new Uint8Array(N * N);
    for (let y = by; y < by + bsz; y++) for (let x = bx; x < bx + bsz; x++) box[y * N + x] = 1;
    const sdIn = C.maskedSd(Float32Array.from(d, lin), box), sdOut = C.maskedSd(r.out.data, box);
    const rawv = (p: number) => Math.round(d[p]);
    ro.textContent = [
      `clicked tile: largest gain ×${gain.toFixed(2)}${st.noClip ? "" : ` (β = ${st.beta})`}`,
      `noise SD in the green box: ${sdIn.toFixed(1)} → ${sdOut.toFixed(1)} grey levels`,
      bd < 0.5 ? "one curve for all pixels: same raw value → same output"
        : `same raw value ≈ ${rawv(pa)}: → ${Math.round(r.out.data[pa])} (pink) and ${Math.round(r.out.data[pb])} (cyan)`,
    ].join("\n");
  }

  const pick = (cv: HTMLCanvasElement, ev: PointerEvent) => {
    const rc = cv.getBoundingClientRect();
    st.px = Math.max(0, Math.min(N - 1, Math.floor((ev.clientX - rc.left) / rc.width * N)));
    st.py = Math.max(0, Math.min(N - 1, Math.floor((ev.clientY - rc.top) / rc.height * N)));
    draw();
  };
  for (const cv of [cv1, cv2]) {
    let down = false;
    cv.addEventListener("pointerdown", ev => { ev.preventDefault(); down = true; cv.setPointerCapture(ev.pointerId); pick(cv, ev); });
    cv.addEventListener("pointermove", ev => { if (down) pick(cv, ev); });
    cv.addEventListener("pointerup", () => { down = false; });
    cv.addEventListener("pointercancel", () => { down = false; });
  }

  load(p0);
  if (host.dataset.source && SOURCES.some(s => s.id === host.dataset.source)) { st.source = host.dataset.source as Source; prepare(); }
}

autoMount(NAME, mount);
