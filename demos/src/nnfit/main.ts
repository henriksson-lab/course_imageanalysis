// A tiny neural network (1 input → H ReLU units → 1 output) fitted live to a 1D intensity
// profile: a synthetic line scan across two nuclei of different brightness, with noise.
// Shows the data, the network's curve, every hidden unit's contribution (a hinge), the
// hinge positions, and the loss (mean squared error) against the training step.
// Training: full-batch gradient descent with Adam (lib/nn.ts), fixed seeds.
//
// <div data-demo="nnfit" data-h="3" data-lr="0.02"></div>
import { autoMount, button, checkbox, htmlEl, presets, slider, svgEl, scale } from "../lib/ui";
import { Adam, mlp, predict, trainStep, forward, type Mlp } from "../lib/nn";
import { profileData, initHinges } from "./data";

const NAME = "nnfit";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .nf-plot { flex: 1 1 560px; min-width: 0; max-width: 780px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .45em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; }
[data-demo="${NAME}"] svg text { font-size: 18px; }
.reveal [data-demo="${NAME}"] { font-size: 19px; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 16px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const W = 780, H = 470, X0 = 70, X1 = 765, Y0 = 14, Y1 = 290, LY0 = 340, LY1 = 440;
const MAXSTEP = 3000;

function mount(host: HTMLElement): void {
  injectStyle();
  const num = (k: string, d: number) => (host.dataset[k] !== undefined && Number.isFinite(Number(host.dataset[k])) ? Number(host.dataset[k]) : d);
  const init = { h: Math.round(num("h", 3)), lr: num("lr", 0.02), bend: true };
  const st = { ...init };
  const { X, Y } = profileData();

  let net: Mlp, opt: Adam, step = 0, losses: number[] = [], timer = 0;
  function reset(): void {
    cancelAnimationFrame(timer); timer = 0;
    net = mlp([1, st.h, 1], [st.bend ? "relu" : "id", "id"], 11);
    initHinges(net, Y, 11);
    opt = new Adam(net, st.lr);
    step = 0; losses = [lossNow()];
    draw();
  }
  const lossNow = () => { let s = 0; X.forEach((x, i) => { s += (predict(net, x)[0] - Y[i][0]) ** 2; }); return s / X.length; };

  const plotDiv = htmlEl("div", "nf-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Network fit to an intensity profile" }, plotDiv);
  const sx = scale(0, 1, X0, X1), sy = scale(-0.1, 1.1, Y1, Y0);
  for (const v of [0, 0.5, 1]) {
    svgEl("line", { x1: X0, x2: X1, y1: sy(v), y2: sy(v), class: "grid" }, svg);
    svgEl("text", { x: X0 - 8, y: sy(v) + 6, "text-anchor": "end" }, svg).textContent = String(v);
  }
  svgEl("text", { x: (X0 + X1) / 2, y: Y1 + 30, "text-anchor": "middle" }, svg).textContent = "position x along the line scan";
  svgEl("text", { x: 18, y: (Y0 + Y1) / 2, transform: `rotate(-90 18 ${(Y0 + Y1) / 2})`, "text-anchor": "middle" }, svg).textContent = "intensity";
  const gHinge = svgEl("g", {}, svg);
  const gData = svgEl("g", {}, svg);
  X.forEach((x, i) => svgEl("circle", { cx: sx(x[0]).toFixed(1), cy: sy(Y[i][0]).toFixed(1), r: 3.2, fill: "#777", "fill-opacity": 0.7 }, gData));
  const fitPath = svgEl("path", { fill: "none", stroke: "#d62728", "stroke-width": 4 }, svg);
  // loss panel
  const lsx = scale(0, MAXSTEP, X0, X1), lsy = scale(-4, 0, LY1, LY0);
  svgEl("rect", { x: X0, y: LY0, width: X1 - X0, height: LY1 - LY0, fill: "none", class: "grid" }, svg);
  for (const e of [-4, -3, -2, -1, 0]) svgEl("text", { x: X0 - 8, y: lsy(e) + 6, "text-anchor": "end", style: "font-size:15px" }, svg).textContent = e === 0 ? "1" : `1e${e}`;
  svgEl("text", { x: X0 + 6, y: LY0 - 6 }, svg).textContent = "loss (mean squared error, log scale) vs training step";
  const lossPath = svgEl("path", { fill: "none", stroke: "#1f77b4", "stroke-width": 2.5 }, svg);
  svgEl("text", { x: X1, y: LY1 + 22, "text-anchor": "end", style: "font-size:15px" }, svg).textContent = String(MAXSTEP);

  const panel = htmlEl("div", "qb-panel", "", host);
  const sH = slider(panel, "hidden units H", 1, 12, 1, st.h, v => { st.h = v; reset(); });
  const sLr = slider(panel, "learning rate", -3, 0, 0.1, Math.log10(st.lr), v => { st.lr = 10 ** v; opt.lr = st.lr; }, v => (10 ** v).toPrecision(2));
  const cb = checkbox(panel, "ReLU bend (untick: no bend)", st.bend, v => { st.bend = v; reset(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "1 step", () => { doSteps(1); draw(); });
  button(row, "train 500 steps", () => run(500));
  button(row, "new start", () => reset());
  const out = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Grey dots: the data. Red: the network. Faint: each hidden unit's hinge, times its output weight. Ticks: hinge positions.", panel);
  htmlEl("div", "qb-note", "Presets:", panel);
  const apply = (h: number, lr: number, bend: boolean, steps: number) => {
    st.h = h; st.lr = lr; st.bend = bend; sH.set(h); sLr.set(Math.log10(lr)); cb.checked = bend; reset(); run(steps);
  };
  presets(panel, [
    { label: "one hinge", apply: () => apply(1, 0.02, true, 3000) },
    { label: "3 hinges", apply: () => apply(3, 0.02, true, 3000) },
    { label: "12 hinges", apply: () => apply(12, 0.02, true, 3000) },
    { label: "no bend", apply: () => apply(12, 0.02, false, 3000) },
    { label: "learning rate too large", apply: () => apply(12, 0.5, true, 500) },
    { label: "Reset", apply: () => { Object.assign(st, init); sH.set(st.h); sLr.set(Math.log10(st.lr)); cb.checked = st.bend; reset(); } },
  ]);

  function doSteps(n: number): void {
    for (let k = 0; k < n && step < MAXSTEP; k++) { trainStep(net, X, Y, opt); step++; losses.push(lossNow()); }
  }
  function run(n: number): void {
    cancelAnimationFrame(timer);
    const target = Math.min(MAXSTEP, step + n), per = Math.max(5, Math.ceil(n / 40));
    const tick = () => { doSteps(Math.min(per, target - step)); draw(); if (step < target) timer = requestAnimationFrame(tick); else timer = 0; };
    timer = requestAnimationFrame(tick);
  }

  function draw(): void {
    const xs = Array.from({ length: 301 }, (_, i) => i / 300);
    fitPath.setAttribute("d", xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(predict(net, [x])[0]).toFixed(1)}`).join(""));
    gHinge.replaceChildren();
    const c = net.W[1], c0 = net.b[1][0];
    for (let k = 0; k < st.h; k++) {
      const pts = xs.map(x => { const f = forward(net, [x]); return c[k] * f.a[1][k] + c0; });
      svgEl("path", { d: xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(Math.max(-0.1, Math.min(1.1, pts[i]))).toFixed(1)}`).join(""),
        fill: "none", stroke: "#ff7f0e", "stroke-width": 2, "stroke-opacity": 0.45 }, gHinge);
      const w = net.W[0][k], b = net.b[0][k];
      const xh = w !== 0 ? -b / w : NaN;
      if (st.bend && xh > 0 && xh < 1) svgEl("path", { d: `M${sx(xh).toFixed(1)},${Y1}l-7,12h14z`, fill: "#ff7f0e" }, gHinge);
    }
    lossPath.setAttribute("d", losses.map((l, i) => `${i ? "L" : "M"}${lsx(i).toFixed(1)},${lsy(Math.max(-4, Math.min(0, Math.log10(l)))).toFixed(1)}`).join(""));
    const nW = 1 * st.h + st.h + st.h * 1 + 1;
    out.textContent = `step ${step}\nloss (MSE) ${losses[losses.length - 1].toFixed(5)}\nweights + shifts: ${nW}`;
  }
  reset();
}

autoMount(NAME, mount);
