// Canny, step by step. A synthetic scene (rectangle, disc, a 4-px bar, a T-junction, a faint
// fading rectangle; seeded noise) or the micrograph. Stage tabs: input, smoothed, gradient
// magnitude, direction (colour wheel), after non-maximum suppression, after hysteresis, and
// the edges over the input. Click a pixel: its 3 × 3 magnitude neighbourhood, the gradient
// arrow, the two neighbours NMS compares, and the verdict; in the hysteresis tab: seed /
// connected weak / isolated weak. "Grow" replays hysteresis one ring per click (or play).
// A small histogram of the post-NMS magnitudes shows both thresholds.
//
// <div data-demo="canny" data-preset="small|large|nms|one|hyst|cell" data-size="128|256"
//      data-image="synthetic|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Grey, combine } from "../lib/imageops";
import { canny, CannyResult, screenAngle, quantiseDirection, NMS_OFFSETS } from "../lib/edges";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { addGaussianNoise } from "../lib/synth";
import { Plot } from "../lib/plot";
import { histogram } from "../lib/histogram";
import { makeScene } from "./scene";
import { ImgView, hueRGBA, colourWheel, injectStyle } from "./imgview";

const NAME = "canny";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .cy-main { flex: 0 1 430px; min-width: 240px; }
[data-demo="${NAME}"] .cy-mid { flex: 0 1 250px; min-width: 200px; display: flex; flex-direction: column; gap: .35em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .cy-inset svg { width: 100%; display: block; }
[data-demo="${NAME}"] .cy-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
[data-demo="${NAME}"] .cy-tabs .qb-btn { padding: .12em .45em; }
[data-demo="${NAME}"] .cy-wheel { display: flex; align-items: center; gap: .4em; font-size: 14px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .cy-main { flex: 0 0 400px; }
.reveal [data-demo="${NAME}"] .cy-mid { flex: 0 0 230px; gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 330px; gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .25em .45em; }
.reveal [data-demo="${NAME}"] .ev-credit { display: none; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

type Stage = "input" | "smoothed" | "mag" | "dir" | "nms" | "hyst" | "overlay";
const STAGES: [Stage, string][] = [["input", "input"], ["smoothed", "smoothed"], ["mag", "magnitude"], ["dir", "direction"], ["nms", "after NMS"], ["hyst", "hysteresis"], ["overlay", "edges on input"]];
interface State { image: "synthetic" | "cell"; sigma: number; noise: number; tLow: number; tHigh: number; sub: boolean; stage: Stage; px: number; py: number; ring: number }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { image: "synthetic", sigma: 1.5, noise: 8, tLow: 3, tHigh: 8, sub: false, stage: "overlay", px: 47, py: 30, ring: -1 };
const PRESETS: Preset[] = [
  { id: "small", label: "σ too small: noise edges", s: { sigma: 0.7, stage: "overlay" } },
  { id: "large", label: "σ too large: edges merge", s: { sigma: 4, tLow: 1.5, tHigh: 4, stage: "overlay", px: 14, py: 80 } },
  { id: "nms", label: "NMS thins the ridge", s: { stage: "mag", px: 46, py: 30 } },
  { id: "one", label: "one threshold: gaps or noise", s: { tLow: 8, tHigh: 8, stage: "hyst", px: 96, py: 100 } },
  { id: "hyst", label: "hysteresis closes the gap", s: { stage: "hyst", ring: 0, px: 96, py: 100 } },
  { id: "cell", label: "micrograph", s: { image: "cell", noise: 0, sigma: 2, tLow: 8, tHigh: 16, stage: "overlay", px: 64, py: 64 } },
];
const SIG_MIN = 0.5, SIG_MAX = 8;
const sigOf = (u: number) => Math.round(SIG_MIN * Math.pow(SIG_MAX / SIG_MIN, u / 100) * 100) / 100;
const uOf = (s: number) => 100 * Math.log(s / SIG_MIN) / Math.log(SIG_MAX / SIG_MIN);
const RED = "#e8402a", ORANGE = "#ff9f1c", BLUE = "#5b8ff9";

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const N = host.dataset.size === "256" ? 256 : 128;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.image === "cell" ? { image: "cell" as const, noise: 0 } : {}), ...(p0?.s ?? {}) });
  const st: State = start();
  const scl = N / 128;
  st.px = Math.round(st.px * scl); st.py = Math.round(st.py * scl);

  // ---- layout ----
  const main = htmlEl("div", "cy-main", "", host);
  const tabs = htmlEl("div", "cy-tabs", "", main);
  const tabBtns = STAGES.map(([s, l]) => button(tabs, l, () => { st.stage = s; draw(); }));
  const view = new ImgView(main, "");
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);
  const mid = htmlEl("div", "cy-mid", "", host);
  const insetDiv = htmlEl("div", "cy-inset", "", mid);
  const inset = svgEl("svg", { viewBox: "0 0 240 240" }, insetDiv);
  const verdict = htmlEl("div", "qb-readout", "", mid);
  const wheelRow = htmlEl("div", "cy-wheel", "", mid);
  colourWheel(wheelRow, 70, false);
  htmlEl("span", "", "hue = gradient direction (counter-clockwise on screen), brightness = magnitude", wheelRow);
  const histPlot = new Plot(mid, { width: 260, height: 150, x: [0, 30], y: [0, 1], font: 16, xticks: 4, yticks: [], margin: { l: 10, r: 8, t: 6, b: 46 }, xlabel: "magnitude after NMS" });
  const hBars = histPlot.mark<[number[], number[], number]>((g, edges, counts, cap) => {
    for (let i = 0; i < counts.length; i++) {
      const x0 = histPlot.x(edges[i]), x1 = histPlot.x(edges[i + 1]), y0 = histPlot.y(0), y1 = histPlot.y(Math.min(counts[i], cap));
      const r = svgEl("rect", { x: x0, y: y1, width: Math.max(0.5, x1 - x0 - 0.5), height: y0 - y1 }, g);
      r.style.fill = "var(--qb-c1)"; r.style.opacity = "0.7";
    }
  }, [[], [], 1]);
  void hBars;
  const vLow = histPlot.vline(0, { color: BLUE, width: 3, text: "low" }), vHigh = histPlot.vline(0, { color: RED, width: 3, text: "high" });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ev-lab", "image", panel);
  const imgBtns = presets(panel, [
    { label: "test scene (synthetic)", apply: () => { st.image = "synthetic"; makeImage(); } },
    { label: "micrograph", apply: () => { st.image = "cell"; makeImage(); } },
  ]);
  const sigS = slider(panel, "σ", 0, 100, 0.5, uOf(st.sigma), u => { st.sigma = sigOf(u); run(); }, u => sigOf(u).toFixed(2));
  const noiseS = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; makeImage(); });
  const lowS = slider(panel, "t low", 0, 30, 0.5, st.tLow, v => { st.tLow = v; if (st.tHigh < v) { st.tHigh = v; highS.set(v); } runHyst(); });
  const highS = slider(panel, "t high", 0, 30, 0.5, st.tHigh, v => { st.tHigh = v; if (st.tLow > v) { st.tLow = v; lowS.set(v); } runHyst(); });
  const subC = checkbox(panel, "sub-pixel NMS (interpolated neighbours)", st.sub, v => { st.sub = v; run(); });
  htmlEl("div", "ev-lab", "hysteresis, step by step", panel);
  const growRow = htmlEl("div", "qb-presets", "", panel);
  button(growRow, "seeds only", () => { stop(); st.ring = 0; st.stage = "hyst"; draw(); });
  button(growRow, "grow 1 ring", () => { stop(); if (st.ring >= 0) st.ring = Math.min(st.ring + 1, res ? res.hyst.rings : 0); st.stage = "hyst"; draw(); });
  const playB = button(growRow, "play", () => { if (timer) stop(); else play(); });
  button(growRow, "all", () => { stop(); st.ring = -1; draw(); });
  button(panel, "reset", () => { stop(); Object.assign(st, start()); st.px = Math.round(st.px * scl); st.py = Math.round(st.py * scl); presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImage(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  let timer = 0;
  function stop(): void { if (timer) { clearInterval(timer); timer = 0; } playB.textContent = "play"; }
  function play(): void {
    st.stage = "hyst"; if (st.ring < 0 || (res && st.ring >= res.hyst.rings)) st.ring = 0;
    playB.textContent = "pause";
    timer = window.setInterval(() => {
      if (!res || st.ring < 0 || st.ring >= res.hyst.rings) { stop(); return; }
      st.ring++; draw();
    }, 350);
    draw();
  }

  function load(p: Preset): void {
    stop();
    Object.assign(st, DEFAULT, p.s);
    st.px = Math.round(st.px * scl); st.py = Math.round(st.py * scl);
    sync(); makeImage();
  }
  function sync(): void { sigS.set(uOf(st.sigma)); noiseS.set(st.noise); lowS.set(st.tLow); highS.set(st.tHigh); subC.checked = st.sub; }

  view.onPick((x, y) => { st.px = x; st.py = y; draw(); });

  // ---- images and computation ----
  let cell: Grey | null = null, synth: Grey | null = null, img: Grey | null = null, res: CannyResult | null = null;
  function makeImage(): void {
    if (st.image === "cell" && !cell) { loadCell(N).then(g => { cell = g; makeImage(); }); return; }
    if (st.image === "cell") img = st.noise > 0 ? addGaussianNoise(cell!, st.noise, 1) : cell!;
    else { if (!synth || synthNoise !== st.noise) { synth = makeScene("synthetic", N, st.noise, 1).image; synthNoise = st.noise; } img = synth; }
    run();
  }
  let synthNoise = -1;
  function run(): void {
    if (!img) return;
    res = canny(img, st.sigma, st.tLow, st.tHigh, st.sub);
    draw();
  }
  function runHyst(): void {
    if (!img || !res) return;
    res = { ...res, hyst: canny(img, st.sigma, st.tLow, st.tHigh, st.sub).hyst };
    draw();
  }

  // ---- drawing ----
  function draw(): void {
    if (!img || !res) return;
    tabBtns.forEach((b, i) => b.classList.toggle("active", STAGES[i][0] === st.stage));
    imgBtns.forEach((b, i) => b.classList.toggle("active", i === (st.image === "synthetic" ? 0 : 1)));
    credit.style.display = st.image === "cell" ? "" : "none";
    const r = res, n = N * N;
    let magMax = 0; for (const v of r.mag.data) magMax = Math.max(magMax, v);
    switch (st.stage) {
      case "input": view.grey(img, { mode: "clip", lo: 0, hi: 255 }); view.setCaption("input image"); break;
      case "smoothed": view.grey(r.smoothed, { mode: "clip", lo: 0, hi: 255 }); view.setCaption(`smoothed, σ = ${st.sigma}`); break;
      case "mag": view.grey(r.mag, { mode: "clip", lo: 0, hi: magMax }); view.setCaption("gradient magnitude ‖∇L‖"); break;
      case "dir": {
        const ang = combine(r.gx, r.gy, (a, b) => screenAngle(a, b));
        view.rgba(hueRGBA(ang.data, r.mag.data, 0.6 * magMax), N, N); view.setCaption("gradient direction (hue) and magnitude"); break;
      }
      case "nms": view.grey(r.nms, { mode: "clip", lo: 0, hi: magMax }); view.setCaption(st.sub ? "after NMS (sub-pixel)" : "after NMS: 1 pixel wide"); break;
      case "hyst": case "overlay": {
        const px = new Uint8ClampedArray(4 * n), H = r.hyst, show = st.ring < 0 ? Infinity : st.ring;
        for (let i = 0; i < n; i++) {
          let c: [number, number, number];
          const v = r.nms.data[i], g = st.stage === "overlay" ? img.data[i] * 0.8 : 0;
          if (H.ring[i] === 0) c = [232, 64, 42];
          else if (H.ring[i] > 0 && H.ring[i] <= show) c = [255, 159, 28];
          else if (st.stage === "hyst" && v >= st.tLow) c = H.ring[i] > 0 ? [60, 90, 150] : [91, 143, 249];
          else c = [g, g, g];
          if (st.stage === "hyst" && c[0] === 0 && v > 0) { const t = Math.min(1, v / st.tLow) * 60; c = [t, t, t]; }
          px[4 * i] = c[0]; px[4 * i + 1] = c[1]; px[4 * i + 2] = c[2]; px[4 * i + 3] = 255;
        }
        view.rgba(px, N, N);
        view.setCaption(st.stage === "overlay" ? "Canny edges on the input (red: strong, orange: weak kept)"
          : st.ring < 0 ? "hysteresis: red seeds, orange kept, blue weak dropped" : `hysteresis: ring ${st.ring} of ${H.rings - 1}`);
        break;
      }
    }
    // overlay: selected pixel, NMS neighbours, gradient arrow
    view.clear();
    const i = st.py * N + st.px, gx = r.gx.data[i], gy = r.gy.data[i], m = r.mag.data[i];
    const nb = NMS_OFFSETS[quantiseDirection(gx, gy)];
    view.box(st.px, st.py, 0, "#ffd400", 2.5);
    if (st.stage !== "input" && st.stage !== "smoothed") {
      view.box(st.px + nb[0], st.py + nb[1], 0, "#22d3ee", 2); view.box(st.px - nb[0], st.py - nb[1], 0, "#22d3ee", 2);
    }
    if (m > 1e-9) { const L = 6 * scl; view.arrow(st.px + 0.5, st.py + 0.5, st.px + 0.5 + L * gx / m, st.py + 0.5 + L * gy / m, "#ffd400", 2.5); }
    drawInset();
    drawHist();
    wheelRow.style.display = st.stage === "dir" ? "" : "none";
    histPlot.svg.style.display = st.stage === "dir" ? "none" : "";
    ro.textContent = `seeds (≥ t high): ${r.hyst.strong}\nweak (t low … t high): ${r.hyst.weak}, kept ${r.hyst.weakKept}\nedge pixels: ${r.hyst.strong + r.hyst.weakKept}`;
  }

  function drawInset(): void {
    const r = res!, x = st.px, y = st.py, i = y * N + x;
    inset.replaceChildren();
    const C = 76, O = 6;
    const nb = NMS_OFFSETS[quantiseDirection(r.gx.data[i], r.gy.data[i])];
    const inside = (u: number, v: number) => u >= 0 && v >= 0 && u < N && v < N;
    let mx = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (inside(x + dx, y + dy)) mx = Math.max(mx, r.mag.data[(y + dy) * N + x + dx]);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const u = x + dx, v = y + dy, X = O + (dx + 1) * C, Y = O + (dy + 1) * C;
      const val = inside(u, v) ? r.mag.data[v * N + u] : NaN;
      const g = Number.isFinite(val) ? Math.round(255 * Math.min(1, val / (mx || 1)) * 0.85) : 128;
      const rect = svgEl("rect", { x: X, y: Y, width: C - 2, height: C - 2, rx: 4 }, inset);
      rect.style.fill = `rgb(${g},${g},${g})`;
      const isNb = (dx === nb[0] && dy === nb[1]) || (dx === -nb[0] && dy === -nb[1]);
      if (isNb) { rect.style.stroke = "#22d3ee"; rect.style.strokeWidth = "5"; }
      if (!dx && !dy) { rect.style.stroke = "#ffd400"; rect.style.strokeWidth = "5"; }
    }
    const gx = r.gx.data[i], gy = r.gy.data[i], m = Math.hypot(gx, gy);
    if (m > 1e-9) {
      const cx = O + 1.5 * C - 1, cy = O + 1.5 * C - 1, L = 1.05 * C, ex = cx + L * gx / m, ey = cy + L * gy / m;
      const l = svgEl("line", { x1: cx, y1: cy, x2: ex, y2: ey, "stroke-width": 4, "stroke-linecap": "round" }, inset);
      l.style.stroke = "#ffd400"; l.style.opacity = "0.75";
      const ux = gx / m, uy = gy / m, hb = 16;
      svgEl("polygon", { points: `${ex + ux * 6},${ey + uy * 6} ${ex - ux * hb - uy * 9},${ey - uy * hb + ux * 9} ${ex - ux * hb + uy * 9},${ey - uy * hb - ux * 9}`, fill: "#ffd400" }, inset);
    }
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const u = x + dx, v = y + dy, X = O + (dx + 1) * C, Y = O + (dy + 1) * C;
      const val = inside(u, v) ? r.mag.data[v * N + u] : NaN;
      const g = Number.isFinite(val) ? Math.round(255 * Math.min(1, val / (mx || 1)) * 0.85) : 128;
      const isNb = (dx === nb[0] && dy === nb[1]) || (dx === -nb[0] && dy === -nb[1]);
      const t = svgEl("text", { x: X + C / 2 - 1, y: Y + C / 2 + 7, "text-anchor": "middle", "font-size": 21, "font-weight": isNb || (!dx && !dy) ? 700 : 400, class: "qb-halo" }, inset);
      t.textContent = Number.isFinite(val) ? val.toFixed(1) : "–";
      t.style.fill = g > 140 ? "#111" : "#fff";
      t.style.stroke = g > 140 ? "rgba(255,255,255,.6)" : "rgba(0,0,0,.5)";
    }
    const a = screenAngle(gx, gy) * 180 / Math.PI, q = quantiseDirection(gx, gy) * 45;
    const kept = r.nms.data[i] > 0, H = r.hyst;
    let hv = "";
    if (st.stage === "hyst" || st.stage === "overlay") {
      const nv = r.nms.data[i];
      hv = H.ring[i] === 0 ? "seed (≥ t high)" : H.ring[i] > 0 ? `weak, connected (ring ${H.ring[i]})` : nv >= st.tLow ? "weak, isolated: dropped" : kept ? "below t low: dropped" : "removed by NMS";
    }
    verdict.textContent =
      `pixel (${x}, ${y})\nLx = ${gx.toFixed(2)}, Ly = ${gy.toFixed(2)}\n‖∇L‖ = ${m.toFixed(2)}, θ = ${a.toFixed(0)}° → ${q}°\n` +
      `NMS: ${kept ? "kept (a maximum along θ)" : "removed (a neighbour is larger)"}` + (hv ? `\n${hv}` : "");
  }

  function drawHist(): void {
    const r = res!, vals: number[] = [];
    for (const v of r.nms.data) if (v > 0) vals.push(v);
    const top = Math.max(st.tHigh * 1.4, 12);
    const h = histogram(vals, { bins: 40, min: 0, max: top });
    const counts = Array.from(h.counts), edges = Array.from({ length: counts.length + 1 }, (_, k) => h.lo + k * h.binWidth);
    const sorted = counts.slice().sort((a, b) => b - a), cap = Math.max(1, sorted[Math.min(3, sorted.length - 1)]);
    histPlot.setDomain([0, top], [0, cap * 1.05]);
    hBars.set(edges, counts, cap);
    vLow.set(st.tLow); vHigh.set(st.tHigh);
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
