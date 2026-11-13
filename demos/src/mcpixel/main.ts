// mcpixel: Monte Carlo for one pixel (ray-tracing day, unit pathtrace/). A surface point and the
// half-circle of directions light can come from; the incoming light L_i(θ) (overcast sky, a
// window, a tiny bright lamp, two lamps) times cos θ is the integrand. Draw N samples from a pdf
// (uniform, cosine, towards the lamp, a pdf that misses the lamp): the samples are rays whose
// length shows the weight f/p; the estimate is their mean. "Repeat 500×": the histogram of 500
// estimates and the RMS error against N on log–log axes with the σ/√N line.
// Options: data-preset (uniform16, uniform64, cosine, fireflies, lamp, bad).
import { htmlEl, svgEl, slider, button, presets, autoMount } from "../lib/ui";
import { rng } from "../lib/random";
import { Plot, seriesColor } from "../lib/plot";
import { histogram } from "../lib/histogram";
import { PROFILES, ProfileId, PdfId, integrand, exactIntegral, makeSampler, singleSigma, estimateN, radiance } from "./model";

const NAME = "mcpixel";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .6em; }
${S} .mc-grid { flex: 0 1 700px; min-width: 0; display: grid; grid-template-columns: 1fr 1fr; gap: .3em .5em; }
${S} .qb-panel { flex: 1 1 270px; max-width: 340px; gap: .3em; font-size: 15px; }
${S} .qb-btn { font-size: 13.5px; padding: .12em .5em; }
${S} .qb-presets { gap: .25em; }
${S} svg.mc-half { width: 100%; display: block; }
${S} .mc-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-family: system-ui, sans-serif; white-space: pre; line-height: 1.3; }
.reveal ${S} { flex-wrap: nowrap; font-size: 18px; gap: .4em; }
.reveal ${S} .mc-grid { flex: 0 0 780px; }
.reveal ${S} .qb-panel { flex: 0 0 340px; gap: .2em; padding: .3em .45em; font-size: 18px; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .qb-presets { gap: .2em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

interface State { profile: ProfileId; pdf: PdfId; logN: number; seed: number }
const DEF: State = { profile: "sky", pdf: "uniform", logN: 4, seed: 1 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "uniform16", label: "uniform, 16 samples", s: { profile: "sky", pdf: "uniform", logN: 4 } },
  { id: "uniform64", label: "4× samples halves the error", s: { profile: "sky", pdf: "uniform", logN: 6 } },
  { id: "cosine", label: "cosine sampling", s: { profile: "sky", pdf: "cosine", logN: 4 } },
  { id: "fireflies", label: "tiny lamp: fireflies", s: { profile: "lamp", pdf: "cosine", logN: 6 } },
  { id: "lamp", label: "sample the lamp", s: { profile: "lamp", pdf: "lamp", logN: 6 } },
  { id: "bad", label: "a pdf that misses the lamp", s: { profile: "lamp", pdf: "bad", logN: 8 } },
];
const NS = Array.from({ length: 13 }, (_, k) => 2 ** k);
const REPS = 500;
const D = Math.PI / 180;

function mount(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  let st: State = { ...DEF, ...(p0?.s ?? {}) };

  const grid = htmlEl("div", "mc-grid", "", host);
  // (1) half circle
  const half = htmlEl("div", "", "", grid);
  const HW = 370, HH = 240, cx = HW / 2, cy = HH - 34, R = 160;
  const svg = svgEl("svg", { viewBox: `0 0 ${HW} ${HH}`, class: "mc-half" }, half);
  const gBase = svgEl("g", {}, svg), gS = svgEl("g", {}, svg);
  // (2) integrand plot
  const pf = new Plot(grid, { width: 370, height: 240, x: [-90, 90], y: [0, 1.2], xlabel: "direction θ (°)", ylabel: "L(θ) cos θ", font: 21, xticks: [-90, -45, 0, 45, 90], margin: { l: 74, b: 58, t: 8, r: 14 } });
  const fLine = pf.line([], [], { color: "var(--qb-fg)", width: 2.5, label: "f(θ)" });
  const pLine = pf.line([], [], { color: seriesColor(2), width: 2, dash: "6 4", label: "pdf (scaled)" });
  const sDots = pf.scatter([], [], { r: 4, color: seriesColor(3) });
  pf.legend(undefined, "tl");
  // (3) histogram of estimates
  const ph = new Plot(grid, { width: 370, height: 230, x: [0, 2], y: [0, 1], xlabel: "estimate (500 repeats)", ylabel: "count", font: 21, margin: { l: 74, b: 58, t: 8, r: 14 }, yticks: 3, xticks: 4 });
  const hBars = ph.histogram([0, 1], [0], { color: seriesColor(0) });
  const hTrue = ph.vline(1, { color: seriesColor(1), width: 3, text: "true" });
  // (4) RMS vs N
  const pr = new Plot(grid, { width: 370, height: 230, x: [1, 4096], y: [1e-3, 10], xlog: true, ylog: true, xlabel: "samples N", ylabel: "RMS error", font: 21, margin: { l: 74, b: 58, t: 8, r: 14 } });
  const rLine = pr.line([], [], { color: seriesColor(0), label: "measured" });
  const rRef = pr.line([], [], { color: "var(--qb-muted)", dash: "6 5", width: 2, label: "σ/√N" });
  const rDot = pr.scatter([], [], { r: 7, color: "#e0a800" });
  pr.legend(undefined, "bl");

  const panel = htmlEl("div", "qb-panel", "", host);
  const pb = presets(panel, [...PRESETS.map(p => ({ label: p.label, apply: () => { st = { ...DEF, ...p.s }; sync(); } })), { label: "reset", apply: () => { st = { ...DEF }; sync(); pb.forEach(b => b.classList.remove("active")); } }]);
  htmlEl("div", "", "Incoming light:", panel);
  const r1 = htmlEl("div", "mc-row", "", panel);
  const profBtns = PROFILES.map(p => ({ id: p.id, b: button(r1, p.label, () => { st.profile = p.id; repeat(); draw(); }) }));
  htmlEl("div", "", "Sample directions from:", panel);
  const r2 = htmlEl("div", "mc-row", "", panel);
  const pdfBtns = (["uniform", "cosine", "lamp", "bad"] as PdfId[]).map(id => ({ id, b: button(r2, id === "lamp" ? "towards the lamp" : id === "bad" ? "a pdf that misses" : id, () => { st.pdf = id; repeat(); draw(); }) }));
  const nS = slider(panel, "samples N", 0, 12, 1, st.logN, v => { st.logN = v; draw(); }, v => String(2 ** v));
  const r3 = htmlEl("div", "mc-row", "", panel);
  button(r3, "new draw", () => { st.seed++; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  let rms: number[] = [], ests: Float64Array[] = [], I = 1, sigma = 1;
  function repeat(): void {
    const p = PROFILES.find(q => q.id === st.profile)!, s = makeSampler(st.pdf, p);
    I = exactIntegral(p); sigma = singleSigma(p, s, I);
    const r = rng(500 + st.profile.length * 7 + st.pdf.length);
    rms = []; ests = [];
    for (const N of NS) {
      const e = new Float64Array(REPS);
      let se = 0;
      for (let k = 0; k < REPS; k++) { e[k] = estimateN(p, s, N, r.unif).est; se += (e[k] - I) ** 2; }
      rms.push(Math.sqrt(se / REPS)); ests.push(e);
    }
  }

  function sync(): void { nS.set(st.logN); repeat(); draw(); }

  function draw(): void {
    profBtns.forEach(({ id, b }) => b.classList.toggle("active", id === st.profile));
    pdfBtns.forEach(({ id, b }) => b.classList.toggle("active", id === st.pdf));
    const p = PROFILES.find(q => q.id === st.profile)!, s = makeSampler(st.pdf, p), N = 2 ** st.logN;
    // integrand and pdf curves
    const ths: number[] = [], fs: number[] = [], ps: number[] = [];
    for (let i = 0; i <= 1800; i++) { const th = -Math.PI / 2 + i * Math.PI / 1800; ths.push(th / D); fs.push(integrand(p, th)); ps.push(s.pdf(th)); }
    const fmax = Math.max(...fs), ymax = Math.min(fmax, p.lamps.length ? Math.max(3, 1.2 * Math.max(...p.lamps.map(l => l.L * Math.cos((l.a + l.b) / 2)))) : 1.2 * fmax);
    const pmax = Math.max(...ps.filter(Number.isFinite)) || 1;
    pf.setDomain([-90, 90], [0, ymax * 1.05]);
    fLine.set(ths, fs); pLine.set(ths, ps.map(v => v / pmax * ymax * 0.9));
    const res = estimateN(p, s, N, rng(st.seed * 7919 + N).unif, true);
    const shown = res.samples.slice(0, 256);
    sDots.set(shown.map(q => q[0] / D), shown.map(q => Math.min(ymax, integrand(p, q[0]))));
    // half circle
    gBase.replaceChildren(); gS.replaceChildren();
    svgEl("line", { x1: 10, x2: HW - 10, y1: cy, y2: cy, stroke: "#555", "stroke-width": 4 }, gBase);
    svgEl("path", { d: `M${cx - R},${cy} A${R},${R} 0 0 1 ${cx + R},${cy}`, fill: "none", stroke: "var(--qb-grid)", "stroke-width": 2 }, gBase);
    // light around the half circle: brightness of the arc
    for (let k = 0; k < 180; k++) {
      const a0 = (-90 + k) * D, a1 = (-89 + k) * D, L = radiance(p, (a0 + a1) / 2) + Math.max(0, ...p.lamps.map(l => (l.a < a1 && l.b > a0 ? l.L : 0)));
      const v = Math.min(1, L / 1.2);
      const pt = (a: number, r: number) => `${cx + r * Math.sin(a)},${cy - r * Math.cos(a)}`;
      const fc = `rgb(${Math.round(60 + 195 * v)},${Math.round(60 + 185 * v)},${Math.round(70 + 100 * v)})`;
      svgEl("path", { d: `M${pt(a0, R)} A${R},${R} 0 0 1 ${pt(a1, R)} L${pt(a1, R + 14)} A${R + 14},${R + 14} 0 0 0 ${pt(a0, R + 14)}Z`, fill: fc, stroke: fc, "stroke-width": 0.8 }, gBase);
    }
    const lab = svgEl("text", { x: cx, y: cy + 26, "text-anchor": "middle", "font-size": 21 }, gBase); lab.textContent = "surface point";
    const wmax = Math.max(1e-9, ...shown.map(q => q[1]));
    const wref = Math.max(I * 2, 1e-9);
    for (const [th, w] of shown) {
      const len = R * Math.min(1, w / wref), cut = w > wref;
      svgEl("line", { x1: cx, y1: cy, x2: cx + len * Math.sin(th), y2: cy - len * Math.cos(th), stroke: cut ? "#d62728" : "#e08a00", "stroke-width": cut ? 3.5 : 2, "stroke-opacity": 0.75 }, gS);
      svgEl("line", { x1: cx + (R - 9) * Math.sin(th), y1: cy - (R - 9) * Math.cos(th), x2: cx + R * Math.sin(th), y2: cy - R * Math.cos(th), stroke: "#555", "stroke-width": 1.5 }, gS);
    }
    void wmax;
    svgEl("circle", { cx, cy, r: 6, fill: "#000" }, gS);
    // histogram for this N
    const e = ests[st.logN];
    let lo = Math.min(...e), hi = Math.max(...e);
    lo = Math.min(lo, I); hi = Math.max(hi, I);
    if (hi - lo < 1e-9) { lo -= 0.05; hi += 0.05; }
    const pad = 0.05 * (hi - lo); lo -= pad; hi += pad;
    const hg = histogram(e, { bins: 40, min: lo, max: hi });
    const edges = Array.from({ length: 41 }, (_, k) => lo + k * hg.binWidth);
    ph.setDomain([lo, hi], [0, Math.max(...hg.counts) * 1.1 || 1]);
    hBars.set(edges, hg.counts); hTrue.set(I);
    // RMS plot
    const ylo = Math.max(1e-4, Math.min(...rms.filter(v => v > 0), sigma / 64) / 2), yhi = Math.max(...rms, sigma) * 2;
    pr.setDomain([1, 4096], [ylo, yhi]);
    rLine.set(NS, rms.map(v => Math.max(v, ylo)));
    rRef.set(NS, NS.map(n => Math.max(ylo, sigma / Math.sqrt(n))));
    rDot.set([N], [Math.max(ylo, rms[st.logN])]);
    // readout
    const meanE = e.reduce((a, b) => a + b, 0) / e.length;
    ro.textContent = [
      `this draw: ${res.est.toFixed(4)} (N = ${N})`,
      `true value: ${I.toFixed(4)}`,
      `500 draws: mean ${meanE.toFixed(4)}, RMS error ${rms[st.logN].toFixed(4)}`,
      `σ of one sample ${sigma.toFixed(3)}; σ/√N = ${(sigma / Math.sqrt(N)).toFixed(4)}`,
    ].join("\n");
  }

  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  sync();
}

autoMount(NAME, host => { injectStyle(); mount(host); });
