// Segmentation of a 1D intensity profile as energy minimisation. A noisy profile (a cell
// and a thin structure on a dark background), two intensity levels mu0 (background) and
// mu1 (object), and the smoothness weight lambda. Shows the thresholded labels (lambda = 0),
// the labelling with the smallest energy (exact, by dynamic programming), the true labels,
// and the energy split into data + lambda * boundaries for both.
// Data term: D_p(l) = (I_p - mu_l)^2 / (2 s^2) with s = 30, i.e. -log of a Gaussian.
// Auto-mounts on <div data-demo="segchain">. Optional: data-lambda="3", data-noise="30",
// data-mu0="60", data-mu1="160".
import { htmlEl, slider, presets, button, autoMount, rng, svgEl } from "../lib/ui";
import { chainDP, chainEnergy, chainThreshold, ChainEnergy, makeProfile, chainCosts, PROFILE_N, PROFILE_SEED, PROFILE_TRUTH, S_MODEL } from "./dp";

const NAME = "segchain";
const N = PROFILE_N;
const TRUTH = PROFILE_TRUTH;
const COL_OBJ = "#ff7f0e", COL_BG = "#1f77b4";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; }
.qb-demo[data-demo="${NAME}"] .sc-plot { flex: 1 1 560px; min-width: 0; max-width: 820px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .45em; }
.qb-demo[data-demo="${NAME}"] .sc-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; overflow-x: auto; }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 19px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const W = 820, H = 410, X0 = 122, X1 = 806, Y0 = 18, Y1 = 238;
const xOf = (p: number) => X0 + (p + 0.5) * (X1 - X0) / N;
const yOf = (v: number) => Y1 - v / 255 * (Y1 - Y0);
const STRIPS = [{ y: 262, label: "true" }, { y: 304, label: "threshold" }, { y: 346, label: "optimal" }];

function mount(host: HTMLElement): void {
  injectStyle();
  const num = (k: string, d: number) => (host.dataset[k] !== undefined && Number.isFinite(Number(host.dataset[k])) ? Number(host.dataset[k]) : d);
  const init = { lambda: num("lambda", 3), noise: num("noise", 30), mu0: num("mu0", 60), mu1: num("mu1", 160) };
  const st = { ...init };

  const plotDiv = htmlEl("div", "sc-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, plotDiv);
  const panel = htmlEl("div", "qb-panel", "", host);

  // static axes
  for (const v of [0, 64, 128, 192, 255]) {
    svgEl("line", { x1: X0, x2: X1, y1: yOf(v), y2: yOf(v), class: "grid" }, svg);
    svgEl("text", { x: X0 - 8, y: yOf(v) + 6, "text-anchor": "end" }, svg).textContent = String(v);
  }
  svgEl("text", { x: 14, y: (Y0 + Y1) / 2, transform: `rotate(-90 14 ${(Y0 + Y1) / 2})`, "text-anchor": "middle" }, svg).textContent = "intensity";
  const gShade = svgEl("g", {}, svg);
  const gLines = svgEl("g", {}, svg);
  const path = svgEl("polyline", { fill: "none", stroke: "#999", "stroke-width": 1 }, svg);
  const gDots = svgEl("g", {}, svg);
  const dots = Array.from({ length: N }, (_, p) => svgEl("circle", { cx: xOf(p), r: 3.6 }, gDots));
  const strips = STRIPS.map(s => {
    svgEl("text", { x: X0 - 8, y: s.y + 21, "text-anchor": "end" }, svg).textContent = s.label;
    return Array.from({ length: N }, (_, p) => svgEl("rect", {
      x: X0 + p * (X1 - X0) / N, y: s.y, width: (X1 - X0) / N - 0.6, height: 28,
    }, svg));
  });
  const wrongMarks = [1, 2].map(k => Array.from({ length: N }, (_, p) =>
    svgEl("circle", { cx: xOf(p), cy: STRIPS[k].y + 14, r: 3.2, fill: "#000" }, svg)));
  const legend = svgEl("text", { x: X0, y: H - 14 }, svg);
  const mk = (cls: string, color: string, dash: string) =>
    svgEl("line", { x1: X0, x2: X1, stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash, class: cls }, gLines);
  const lMu0 = mk("", COL_BG, "8 5"), lMu1 = mk("", COL_OBJ, "8 5"), lThr = mk("", "#555", "2 4");

  // panel
  // the lambda slider is quadratic, so small values are easy to set: lambda = 120 (u / 100)^2
  const lamOf = (u: number) => Math.round(120 * (u / 100) ** 2 * 4) / 4;
  const uOf = (l: number) => 100 * Math.sqrt(l / 120);
  const lamS0 = slider(panel, "λ", 0, 100, 0.5, uOf(st.lambda), u => { st.lambda = lamOf(u); draw(); }, u => lamOf(u).toFixed(2));
  const lamS = { set: (l: number) => lamS0.set(uOf(l)) };
  const mu0S = slider(panel, "μ₀ (dark)", 0, 255, 1, st.mu0, v => { st.mu0 = v; draw(); });
  const mu1S = slider(panel, "μ₁ (bright)", 0, 255, 1, st.mu1, v => { st.mu1 = v; draw(); });
  const noiseS = slider(panel, "noise", 0, 60, 5, st.noise, v => { st.noise = v; newProfile(); draw(); });
  htmlEl("div", "sc-lab", "presets", panel);
  const presetBtns = presets(panel, [
    { label: "λ = 0: threshold", apply: () => setLam(0) },
    { label: "λ = 3", apply: () => setLam(3) },
    { label: "thin structure lost", apply: () => setLam(10) },
    { label: "too smooth", apply: () => setLam(100) },
  ]);
  button(panel, "reset", () => {
    Object.assign(st, init);
    lamS.set(st.lambda); mu0S.set(st.mu0); mu1S.set(st.mu1); noiseS.set(st.noise);
    presetBtns.forEach(b => b.classList.remove("active"));
    newProfile(); draw();
  });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", `Data cost D(ℓ) = (I − μℓ)² / (2·${S_MODEL}²). Black dots in a strip: wrong compared with the truth.`, panel);

  function setLam(v: number): void { st.lambda = v; lamS.set(v); st.mu0 = init.mu0; st.mu1 = init.mu1; mu0S.set(st.mu0); mu1S.set(st.mu1); draw(); }

  let I: number[] = [];
  function newProfile(): void { I = makeProfile(st.noise, rng(PROFILE_SEED).norm); }

  function table(et: ChainEnergy, eo: ChainEnergy, wt: number, wo: number): string {
    const r = (name: string, a: string, b: string) => name.padEnd(13) + a.padStart(10) + b.padStart(9);
    return [
      r("", "threshold", "optimal"),
      r("data", et.data.toFixed(1), eo.data.toFixed(1)),
      r("boundaries", String(et.boundaries), String(eo.boundaries)),
      r("λ·boundaries", (st.lambda * et.boundaries).toFixed(1), (st.lambda * eo.boundaries).toFixed(1)),
      r("total E", et.total.toFixed(1), eo.total.toFixed(1)),
      r("wrong", String(wt), String(wo)),
    ].join("\n");
  }

  function draw(): void {
    const { D0, D1 } = chainCosts(I, st.mu0, st.mu1);
    const Lt = chainThreshold(D0, D1), Lo = chainDP(D0, D1, st.lambda);
    const et = chainEnergy(D0, D1, st.lambda, Lt), eo = chainEnergy(D0, D1, st.lambda, Lo);

    // shading of the optimal object runs
    gShade.replaceChildren();
    for (let p = 0; p < N;) {
      if (!Lo[p]) { p++; continue; }
      let q = p; while (q < N && Lo[q]) q++;
      svgEl("rect", { x: X0 + p * (X1 - X0) / N, y: Y0, width: (q - p) * (X1 - X0) / N, height: Y1 - Y0, fill: COL_OBJ, opacity: 0.13 }, gShade);
      p = q;
    }
    const thr = (st.mu0 + st.mu1) / 2;
    for (const [l, v] of [[lMu0, st.mu0], [lMu1, st.mu1], [lThr, thr]] as [SVGLineElement, number][]) {
      l.setAttribute("y1", String(yOf(v))); l.setAttribute("y2", String(yOf(v)));
    }
    path.setAttribute("points", I.map((v, p) => `${xOf(p)},${yOf(v)}`).join(" "));
    dots.forEach((d, p) => { d.setAttribute("cy", String(yOf(I[p]))); d.setAttribute("fill", Lo[p] ? COL_OBJ : COL_BG); });
    const labs = [TRUTH, Lt, Lo];
    strips.forEach((cells, k) => cells.forEach((c, p) => c.setAttribute("fill", labs[k][p] ? COL_OBJ : COL_BG)));
    let wt = 0, wo = 0;
    wrongMarks[0].forEach((m, p) => { const w = Lt[p] !== TRUTH[p]; wt += +w; m.style.display = w ? "" : "none"; });
    wrongMarks[1].forEach((m, p) => { const w = Lo[p] !== TRUTH[p]; wo += +w; m.style.display = w ? "" : "none"; });
    legend.textContent = `orange = object, blue = background; dashed: μ₀, μ₁; dotted: threshold ${thr.toFixed(1)}`;

    ro.textContent = table(et, eo, wt, wo);
  }

  newProfile();
  draw();
}

autoMount(NAME, mount);
