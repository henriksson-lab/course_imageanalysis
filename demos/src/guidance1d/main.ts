// Classifier-free guidance in 1D, exactly. Two classes A and B (Gaussian mixtures). Panels:
// the class densities and the unconditional density (what the network learns when the label is
// dropped); the sharpened density p(x | A) p(A | x)^w (exact); a histogram of 2 000 samples of
// guided diffusion with the classifier-free combination of the exact conditional and
// unconditional denoisers. Readouts: mean and sd against the class, fraction of samples beyond
// the class mean and beyond the most extreme 1 % of real class members.
//
// <div data-demo="guidance1d" data-preset="w0|w3|w8|sharp|submode"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import * as M from "./model";
import { injectStyle, choiceRow, helperCss, fmt } from "../scorematch1d/dview";

const NAME = "guidance1d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .g1-plots { flex: 1 1 600px; min-width: 0; max-width: 720px; display: flex; flex-direction: column; }
${S} .qb-panel { flex: 1 1 300px; gap: .4em; }
${S} .qb-readout { font-size: 15px; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .g1-plots { flex: 0 0 680px; }
.reveal ${S} .qb-panel { flex: 1 1 420px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

interface State { data: M.DataId; y: number; w: number; eta: number; showSharp: boolean }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { data: "two", y: 0, w: 0, eta: 1, showSharp: true };
const PRESETS: Preset[] = [
  { id: "w0", label: "w = 0: the class as it is", s: { w: 0 } },
  { id: "w3", label: "w = 3: narrower, pushed away", s: { w: 3 } },
  { id: "w8", label: "w = 8: beyond the real data", s: { w: 8, eta: 0 } },
  { id: "sharp", label: "guided sampling ≠ sharpened density", s: { w: 3, eta: 0 } },
  { id: "submode", label: "the sub-mode near B disappears", s: { data: "submode", w: 3 } },
];
const XR: [number, number] = [-5, 5];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  // larger labels on slides (the plots are scaled down there)
  const F = host.closest(".reveal") ? 23 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, ...p0.s };

  const plots = htmlEl("div", "g1-plots", "", host);
  const pA = new Plot(plots, { width: 700, height: 200, x: XR, y: [0, 0.5], ylabel: "density", xticks: [], yticks: 3, font: F, margin: { l: 3.9 * F, r: 10, t: 6, b: 6 } });
  const uncond = pA.band([], [], [], { color: "#999", fillOpacity: 0.3 });
  const dA = pA.line([], [], { color: seriesColor(0), width: 3 }), dB = pA.line([], [], { color: seriesColor(1), width: 3 });
  pA.legend([{ label: "class A", color: seriesColor(0) }, { label: "class B", color: seriesColor(1) }, { label: "no label (A + B)", color: "#999", kind: "box" }], "tl");
  const pB = new Plot(plots, { width: 700, height: 290, x: XR, y: [0, 1], xlabel: "x", ylabel: "density", xticks: 10, yticks: 3, font: F, margin: { l: 3.9 * F, r: 10, t: 6 } });
  const hist = pB.histogram([], [], { color: seriesColor(0), fillOpacity: 0.45 });
  const cls = pB.line([], [], { color: seriesColor(0), width: 2.5, dash: "3 4" });
  const sharp = pB.line([], [], { color: "var(--qb-fg)", width: 3, dash: "8 5" });
  const q99 = pB.vline(NaN, { color: "var(--qb-c4)", width: 2.5, text: "1 %" });
  pB.legend([{ label: "guided samples", color: seriesColor(0), kind: "box" }, { label: "class A (w = 0)", color: seriesColor(0), dash: "3 4" }, { label: "sharpened p(x|A) p(A|x)^w", color: "var(--qb-fg)", dash: "8 5" }], "tr");

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const dataC = choiceRow(panel, "classes", M.DATA_IDS.map(v => ({ v, label: M.DATA_LABEL[v] })), st.data, v => { st.data = v; later(); });
  const yC = choiceRow(panel, "condition on", [{ v: 0, label: "A" }, { v: 1, label: "B" }], st.y, v => { st.y = v; later(); });
  const wS = slider(panel, "guidance w", 0, 10, 0.5, st.w, v => { st.w = v; later(); });
  const etaC = choiceRow(panel, "sampler", [{ v: 1, label: "stochastic" }, { v: 0, label: "deterministic" }], st.eta, v => { st.eta = v; later(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Classifier-free guidance: one network learns both the conditional denoiser (with the label) and the unconditional one (label dropped at random during training); sampling uses (1 + w)·conditional − w·unconditional at every step. 2 000 samples, 100 steps, exact denoisers.", panel);

  let timer = 0;
  function later(): void { clearTimeout(timer); timer = window.setTimeout(draw, 60); }

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    dataC.set(st.data); yC.set(st.y); wS.set(st.w); etaC.set(st.eta);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }

  function draw(): void {
    const c = M.classes(st.data), xs = Array.from({ length: 401 }, (_, i) => XR[0] + (XR[1] - XR[0]) * i / 400);
    const pa = xs.map(x => M.classDensity(c, 0, x)), pb = xs.map(x => M.classDensity(c, 1, x)), pu = xs.map(x => M.classDensity(c, null, x));
    pA.setDomain(XR, [0, Math.max(...pa, ...pb) * 1.1]);
    uncond.set(xs, xs.map(() => 0), pu); dA.set(xs, pa); dB.set(xs, pb);
    const X = M.guidedSamples(c, st.y, st.w, 100, st.eta);
    const nb = 80, edges = Array.from({ length: nb + 1 }, (_, i) => XR[0] + (XR[1] - XR[0]) * i / nb), h = (XR[1] - XR[0]) / nb, cnt = new Float64Array(nb);
    for (const v of X) { const b = Math.floor((v - XR[0]) / h); if (b >= 0 && b < nb) cnt[b] += 1 / (X.length * h); }
    const sh = M.sharpened(c, st.y, st.w, xs), base = st.y === 0 ? pa : pb;
    pB.setDomain(XR, [0, Math.min(3, Math.max(...cnt, ...sh, ...base) * 1.08)]);
    hist.set(edges, cnt);
    cls.set(xs, base);
    sharp.set(xs, sh);
    const cs = M.classStats(c, st.y), sm = M.summary(c, st.y, X);
    q99.set(cs.q99);
    const name = st.y === 0 ? "A" : "B";
    ro.textContent =
      `class ${name}: mean ${fmt(cs.mean, 2)}, sd ${fmt(cs.sd, 2)}\n` +
      `guided samples: mean ${fmt(sm.mean, 2)}, sd ${fmt(sm.sd, 2)}\n` +
      `beyond the class mean: ${(100 * sm.beyondMean).toFixed(1)} %\n` +
      `beyond the 1 % most extreme real members: ${(100 * sm.beyondQ99).toFixed(1)} %`;
  }

  apply(p0);
}

autoMount(NAME, mount);
