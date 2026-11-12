// Build a wave from five frequencies. Five sliders set the amplitudes a_1..a_5 of
// sin(2π·f·t), f = 1..5 cycles per window. The plot shows the five components (thin,
// faint, coloured like their slider) and their sum (bold), over one window t ∈ [0, 1].
// A bar chart shows the amplitude spectrum |a_f|: what a Fourier transform of the
// sum would give back. Presets: zero, pure tone, square wave, sawtooth, random.
// Auto-mounts on <div data-demo="fourierwave">. Optional: data-a="1 0 0 0 0".
import { svgEl, htmlEl, slider, presets, button, scale, rng, numList, autoMount, Slider } from "../lib/ui";

const NAME = "fourierwave";
const NF = 5;
const COLS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"];
const STYLE = `
.qb-demo[data-demo="${NAME}"] .qb-plot { flex: 1 1 420px; max-width: 640px; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; font-size: 20px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-plot { flex: 0 0 640px; width: 640px; max-width: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; max-width: 470px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .fw-formula { font-size: 1em; font-style: italic; }
.qb-demo[data-demo="${NAME}"] .qb-slider-name { font-weight: 600; }
.qb-demo[data-demo="${NAME}"] .fw-spec { width: 100%; max-width: 440px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const comp = (f: number, t: number) => Math.sin(2 * Math.PI * f * t);
const r2 = (v: number) => Math.round(v * 100) / 100;

function mount(host: HTMLElement): void {
  injectStyle();
  const a0 = numList(host.dataset.a, [1, 0, 0, 0, 0]).slice(0, NF);
  while (a0.length < NF) a0.push(0);
  const a = [...a0];

  // ---- main plot: components and their sum over one window ----
  const W = 640, H = 560, L = 62, R = 16, T = 16, B = 56;
  const sx = scale(0, 1, L, W - R);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "qb-plot", role: "img",
    "aria-label": "Five sine waves and their sum over one window" }, host);
  const grid = svgEl("g", {}, svg);
  const lines = svgEl("g", { fill: "none" }, svg);
  const compPaths = COLS.map(c => svgEl("path", { stroke: c, "stroke-width": 2.5, opacity: 0.55 }, lines));
  const sumPath = svgEl("path", { stroke: "#222", "stroke-width": 4.5, "stroke-linejoin": "round" }, lines);
  svgEl("text", { x: sx(1), y: H - 10, "text-anchor": "end", "font-size": 20, "font-style": "italic" }, svg).textContent = "t";
  const legend = svgEl("text", { x: L + 8, y: T + 20, "font-size": 19 }, svg);
  legend.innerHTML = `<tspan font-weight="700">bold: the sum</tspan><tspan fill="#777">   thin: the five waves</tspan>`;
  let yMax = 0;

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fw-formula", "x(t) = a₁ sin(2π·1·t) + … + a₅ sin(2π·5·t)", panel);
  const sliders: Slider[] = [];
  for (let i = 0; i < NF; i++) {
    const s = slider(panel, `a${"₁₂₃₄₅"[i]}  (f = ${i + 1})`, -1, 1, 0.01, a[i], v => { a[i] = v; clear(); draw(); },
      v => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(2).replace("-", "−"));
    s.input.style.accentColor = COLS[i];
    (s.input.previousElementSibling as HTMLElement).style.color = COLS[i];
    sliders.push(s);
  }
  const set = (v: number[]) => { v.forEach((x, i) => { a[i] = x; sliders[i].set(x); }); draw(); };
  const btns = presets(panel, [
    { label: "all zero", apply: () => set([0, 0, 0, 0, 0]) },
    { label: "pure tone", apply: () => set([0, 1, 0, 0, 0]) },
    { label: "square", apply: () => set([1, 0, 1 / 3, 0, 1 / 5].map(r2)) },
    { label: "sawtooth", apply: () => set([1, -1 / 2, 1 / 3, -1 / 4, 1 / 5].map(r2)) },
    { label: "random", apply: () => { const g = rng(7); set(a.map(() => Math.round((2 * g.unif() - 1) * 20) / 20)); } },
  ]);
  const clear = () => btns.forEach(b => b.classList.remove("active"));
  button(btns[0].parentElement!, "reset", () => { clear(); set(a0); });

  // ---- amplitude spectrum ----
  const SW = 440, SH = 215, SL = 44, SB = 34, ST = 26;
  const spec = svgEl("svg", { viewBox: `0 0 ${SW} ${SH}`, class: "fw-spec", role: "img",
    "aria-label": "Amplitude spectrum: one bar per frequency" }, panel);
  const bx = (i: number) => SL + (i + 0.5) * (SW - SL - 8) / NF, bw = 40;
  const by = scale(0, 1, SH - SB, ST);
  for (const v of [0, 0.5, 1]) {
    svgEl("line", { x1: SL, x2: SW - 8, y1: by(v), y2: by(v), class: v === 0 ? "axis" : "grid", "stroke-width": v === 0 ? 2 : 1 }, spec);
    svgEl("text", { x: SL - 6, y: by(v) + 6, "text-anchor": "end", "font-size": 18, fill: "#666" }, spec).textContent = String(v);
  }
  svgEl("text", { x: SL, y: 18, "font-size": 18 }, spec).textContent = "amplitude spectrum: |a| per frequency";
  const bars = COLS.map((c, i) => {
    svgEl("text", { x: bx(i), y: SH - 8, "text-anchor": "middle", "font-size": 18 }, spec).textContent = `f=${i + 1}`;
    return svgEl("rect", { x: bx(i) - bw / 2, width: bw, fill: c }, spec);
  });

  function drawGrid(): void {
    grid.innerHTML = "";
    const sy = scale(-yMax, yMax, H - B, T + 34);
    const step = yMax <= 2 ? 0.5 : 1;
    for (let v = -yMax; v <= yMax + 1e-9; v += step) {
      const z = Math.abs(v) < 1e-9;
      svgEl("line", { x1: L, x2: W - R, y1: sy(v), y2: sy(v), class: z ? "axis" : "grid", "stroke-width": z ? 2 : 1 }, grid);
      svgEl("text", { x: L - 8, y: sy(v) + 6, "text-anchor": "end", "font-size": 18, fill: "#666" }, grid)
        .textContent = (z ? 0 : v).toString().replace("-", "−");
    }
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svgEl("line", { x1: sx(t), x2: sx(t), y1: T + 34, y2: H - B, class: "grid" }, grid);
      svgEl("text", { x: sx(t), y: H - B + 24, "text-anchor": "middle", "font-size": 18, fill: "#666" }, grid).textContent = String(t);
    }
  }

  function draw(): void {
    const N = 400, ts = Array.from({ length: N + 1 }, (_, k) => k / N);
    const sum = ts.map(t => a.reduce((s, ai, i) => s + ai * comp(i + 1, t), 0));
    // y range: ±1.5 normally; widen in whole steps only when the sum needs it
    const need = Math.max(1.5, ...sum.map(Math.abs)) * 1.05;
    const ym = need <= 1.5 ? 1.5 : need <= 2 ? 2 : Math.ceil(need);
    if (ym !== yMax) { yMax = ym; drawGrid(); }
    const sy = scale(-yMax, yMax, H - B, T + 34);
    const path = (ys: number[]) => "M" + ys.map((y, k) => `${sx(ts[k]).toFixed(1)},${sy(y).toFixed(1)}`).join("L");
    compPaths.forEach((p, i) => {
      p.style.display = Math.abs(a[i]) > 1e-9 ? "" : "none";
      p.setAttribute("d", path(ts.map(t => a[i] * comp(i + 1, t))));
    });
    sumPath.setAttribute("d", path(sum));
    bars.forEach((b, i) => {
      const h = Math.abs(a[i]);
      b.setAttribute("y", String(by(h))); b.setAttribute("height", String(by(0) - by(h)));
    });
  }

  draw();
}

autoMount(NAME, mount);
