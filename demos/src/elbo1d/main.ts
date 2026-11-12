// The VAE loss for one image, in one dimension, where everything can be computed exactly.
// Top: the prior N(0, 1) (dashed) and the encoder's cloud q = N(mu, s²) (orange), with
// 20 reparameterised draws z = mu + s ε (fixed ε). Middle: those draws decoded, w z, next
// to the observed grey value x. Bottom: bars for the reconstruction term, β × KL and the
// total, with the best possible total marked. See model.ts for the formulas.
//
// <div data-demo="elbo1d" data-mu="0" data-s="1" data-beta="1" data-x="1.5"></div>
import { autoMount, button, htmlEl, presets, slider, svgEl, scale, type Slider } from "../lib/ui";
import { DEFAULT, EPS, best, kl, loss, recon, reconMC, type Params } from "./model";

const NAME = "elbo1d";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .eb-plot { flex: 1 1 560px; min-width: 0; max-width: 760px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .45em; }
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

const W = 760, H = 530, X0 = 40, X1 = 740;
const ORANGE = "#e07b1a";
const npdf = (z: number, m: number, s: number) => Math.exp(-0.5 * ((z - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

function mount(host: HTMLElement): void {
  injectStyle();
  const num = (k: string, d: number) => (host.dataset[k] !== undefined && Number.isFinite(Number(host.dataset[k])) ? Number(host.dataset[k]) : d);
  const p: Params = { ...DEFAULT, x: num("x", DEFAULT.x), beta: num("beta", DEFAULT.beta) };
  const init = { mu: num("mu", 0), s: num("s", 1), x: p.x, beta: p.beta };
  let mu = init.mu, s = init.s;

  const plotDiv = htmlEl("div", "eb-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Prior, encoder cloud and loss terms" }, plotDiv);
  const sz = scale(-3.2, 3.2, X0, X1);
  // panel 1: densities over z (y 20..190)
  const D0 = 30, D1 = 190, sd = scale(0, 1.0, D1, D0);
  svgEl("line", { x1: X0, x2: X1, y1: D1, y2: D1, class: "axis" }, svg);
  for (const t of [-3, -2, -1, 0, 1, 2, 3]) svgEl("text", { x: sz(t), y: D1 + 22, "text-anchor": "middle", style: "font-size:15px" }, svg).textContent = String(t);
  svgEl("text", { x: X1, y: D0 - 8, "text-anchor": "end" }, svg).textContent = "latent z";
  const qArea = svgEl("path", { fill: ORANGE, "fill-opacity": 0.25, stroke: ORANGE, "stroke-width": 3 }, svg);
  const prior = svgEl("path", { fill: "none", stroke: "#333", "stroke-width": 2.5, "stroke-dasharray": "7 5" }, svg);
  const gDraws = svgEl("g", {}, svg);
  const lab1 = svgEl("text", { x: X0 + 4, y: D0 - 8 }, svg);
  lab1.innerHTML = `<tspan fill="#555">prior N(0, 1)</tspan>   <tspan fill="${ORANGE}">encoder cloud q = N(μ, s²)</tspan>`;
  // panel 2: decoded values on the x axis (y 240..290)
  const sxv = scale(-3.2, 3.2, X0, X1), R0 = 300;
  svgEl("line", { x1: X0, x2: X1, y1: R0, y2: R0, class: "axis" }, svg);
  for (const t of [-3, -2, -1, 0, 1, 2, 3]) svgEl("text", { x: sxv(t), y: R0 + 40, "text-anchor": "middle", style: "font-size:15px" }, svg).textContent = String(t);
  svgEl("text", { x: X0 + 4, y: R0 - 40 }, svg).textContent = "grey value: draws decoded (w z) and the image x";
  const gDec = svgEl("g", {}, svg);
  const xMark = svgEl("path", { fill: "#d62728" }, svg);
  // panel 3: bars (y 320..490)
  const B0 = 370, BH = 32, bx = scale(0, 8, 200, X1);
  const barNames = ["reconstruction", "β × KL", "total loss"], barCols = ["#1f77b4", ORANGE, "#444"];
  const bars = barNames.map((n, i) => {
    svgEl("text", { x: 190, y: B0 + i * (BH + 14) + BH / 2 + 6, "text-anchor": "end" }, svg).textContent = n;
    const r = svgEl("rect", { x: bx(0), y: B0 + i * (BH + 14), height: BH, fill: barCols[i], "fill-opacity": 0.8 }, svg);
    const t = svgEl("text", { y: B0 + i * (BH + 14) + BH / 2 + 6 }, svg);
    return { r, t };
  });
  const bestLine = svgEl("line", { y1: B0 + 2 * (BH + 14) - 6, y2: B0 + 2 * (BH + 14) + BH + 6, stroke: "#2ca02c", "stroke-width": 3.5 }, svg);
  const bestLab = svgEl("text", { y: B0 + 2 * (BH + 14) + BH + 24, fill: "#2ca02c", "text-anchor": "middle", style: "font-size:15px" }, svg);
  bestLab.textContent = "best possible";

  const panel = htmlEl("div", "qb-panel", "", host);
  const sMu: Slider = slider(panel, "cloud centre μ", -3, 3, 0.01, mu, v => { mu = v; draw(); }, v => v.toFixed(2));
  const sS: Slider = slider(panel, "cloud spread s", 0.02, 1.5, 0.01, s, v => { s = v; draw(); }, v => v.toFixed(2));
  const sB: Slider = slider(panel, "weight β of KL", 0, 5, 0.1, p.beta, v => { p.beta = v; draw(); }, v => v.toFixed(1));
  const sX: Slider = slider(panel, "image value x", -3, 3, 0.1, p.x, v => { p.x = v; draw(); }, v => v.toFixed(1));
  button(htmlEl("div", "qb-presets", "", panel), "best cloud for this x and β", () => setQ(best(p).mu, best(p).s));
  const out = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", `Decoder: x̂ = w z with w = ${p.w}, pixel noise sₓ = ${p.sx}. Dots: ${EPS.length} draws z = μ + s·ε with fixed ε.`, panel);
  presets(panel, [
    { label: "ignore the image (q = prior)", apply: () => setAll(0, 1, 1, init.x) },
    { label: "too confident", apply: () => setAll(init.x, 0.05, 1, init.x) },
    { label: "best (β = 1)", apply: () => { setAll(0, 1, 1, init.x); setQ(best(p).mu, best(p).s); } },
    { label: "best (β = 5)", apply: () => { setAll(0, 1, 5, init.x); setQ(best(p).mu, best(p).s); } },
    { label: "Reset", apply: () => setAll(init.mu, init.s, init.beta, init.x) },
  ]);
  function setQ(m: number, sp: number): void { mu = m; s = sp; sMu.set(m); sS.set(sp); draw(); }
  function setAll(m: number, sp: number, b: number, x: number): void { p.beta = b; p.x = x; sB.set(b); sX.set(x); setQ(m, sp); }

  function draw(): void {
    const zs = Array.from({ length: 321 }, (_, i) => -3.2 + 6.4 * i / 320);
    const yq = (z: number) => Math.max(D0 - 10, sd(npdf(z, mu, s)));
    prior.setAttribute("d", zs.map((z, i) => `${i ? "L" : "M"}${sz(z).toFixed(1)},${sd(npdf(z, 0, 1)).toFixed(1)}`).join(""));
    qArea.setAttribute("d", `M${sz(-3.2)},${D1}` + zs.map(z => `L${sz(z).toFixed(1)},${yq(z).toFixed(1)}`).join("") + `L${sz(3.2)},${D1}Z`);
    gDraws.replaceChildren(); gDec.replaceChildren();
    EPS.forEach(e => {
      const z = mu + s * e;
      if (z > -3.2 && z < 3.2) svgEl("circle", { cx: sz(z).toFixed(1), cy: D1 - 7, r: 5, fill: ORANGE, stroke: "#fff", "stroke-width": 1 }, gDraws);
      const xd = p.w * z;
      if (xd > -3.2 && xd < 3.2) svgEl("circle", { cx: sxv(xd).toFixed(1), cy: R0, r: 5, fill: "#1f77b4", "fill-opacity": 0.7 }, gDec);
    });
    xMark.setAttribute("d", `M${sxv(p.x).toFixed(1)},${R0 - 16}l-9,-14h18z M${sxv(p.x).toFixed(1)},${R0 + 16}l-9,14h18z`);
    const R = recon(p, mu, s), K = kl(mu, s), T = loss(p, mu, s), b = best(p), Tb = loss(p, b.mu, b.s);
    [R, p.beta * K, T].forEach((v, i) => {
      const wpx = Math.max(0, Math.min(bx(8), bx(v)) - bx(0));
      bars[i].r.setAttribute("width", wpx.toFixed(1));
      bars[i].t.setAttribute("x", (bx(0) + wpx + 8).toFixed(1));
      bars[i].t.textContent = v.toFixed(3) + (v > 8 ? " →" : "");
    });
    const xb = Math.min(bx(8), bx(Tb));
    bestLine.setAttribute("x1", xb.toFixed(1)); bestLine.setAttribute("x2", xb.toFixed(1));
    bestLab.setAttribute("x", xb.toFixed(1));
    out.textContent =
      `reconstruction ${R.toFixed(3)}\n  (from the ${EPS.length} draws: ${reconMC(p, mu, s).toFixed(3)})\n` +
      `KL ${K.toFixed(3)}   β × KL ${(p.beta * K).toFixed(3)}\n` +
      `total ${T.toFixed(3)}\n` +
      `best: μ = ${b.mu.toFixed(3)}, s = ${b.s.toFixed(3)}, total ${Tb.toFixed(3)}`;
  }
  draw();
}

autoMount(NAME, mount);
