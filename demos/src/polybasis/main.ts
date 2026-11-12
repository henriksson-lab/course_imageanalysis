// polybasis (moments elective, unit orthogonal-moments/): why monomials are a poor basis.
// Top left: the basis functions on [−1, 1] up to degree N: monomials 1, x, …, x^N, or
// Legendre polynomials, or the Gram–Schmidt view (step by step: the next monomial, its
// projections onto the functions already orthogonalised, and what is left).
// Top right: the Gram matrix of the shown functions as cosines ⟨b_j, b_k⟩/(‖b_j‖‖b_k‖)
// (1 = the same direction) and its condition number.
// Bottom: a 1D signal fitted with degree ≤ N by both bases (monomials through the Gram
// system, Legendre by projection) and the error against N, in 64- or 32-bit arithmetic.
// Options: data-preset (alike, gs, step, breakdown), data-n.
import { autoMount, htmlEl, button, slider } from "../lib/ui";
import { Plot, heatmapGrid, seriesColor } from "../lib/plot";
import { diverging } from "../lib/colour";
import { jacobiEigen } from "../lib/linalg";
import { legendre } from "../lib/moments";
import { XS, NS, SIGNALS, SignalId, signal, monomials, gram, fitMonomial, fitLegendre, relErr, gramSchmidt, condSym } from "./compute";

const NAME = "polybasis";
const NMAX = 30;
const CM = "var(--qb-c4, #ff7f0e)", CL = "var(--qb-c1, #1f77b4)";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr); gap: 1.1em .8em; align-items: start; }
.qb-demo[data-demo="${NAME}"] .pb-ctl { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: .35em 1em; align-items: center; }
.reveal .qb-demo[data-demo="${NAME}"] { grid-template-columns: 250px minmax(0, 1.45fr) minmax(0, 1fr); }
.reveal .qb-demo[data-demo="${NAME}"] .pb-ctl { grid-column: auto; grid-row: 1 / span 2; flex-direction: column; flex-wrap: nowrap; align-items: stretch; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .pb-lab { font-size: 15px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] .pb-lab { margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.qb-demo[data-demo="${NAME}"] .pb-note { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .pb-gram { text-align: center; font-size: 16px; }
@media (max-width: 700px) { .qb-demo[data-demo="${NAME}"] { grid-template-columns: 1fr; } }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const sup = (n: number) => String(n).split("").map(c => SUP[+c]).join("");
const sci = (v: number) => { if (!Number.isFinite(v) || v > 1e16) return "> 10¹⁶"; const e = Math.floor(Math.log10(v)); return e < 3 ? v.toFixed(1) : `${(v / 10 ** e).toFixed(1)}×10${sup(e)}`; };

function mount(host: HTMLElement): void {
  injectStyle();
  const ctl = htmlEl("div", "pb-ctl", "", host);
  const topL = htmlEl("div", "", "", host);
  const topR = htmlEl("div", "pb-gram", "", host);
  const botL = htmlEl("div", "", "", host);
  const botR = htmlEl("div", "", "", host);

  const st = { view: "mono" as "mono" | "leg" | "gs", N: 6, gs: 1, sig: "step" as SignalId, single: false };

  htmlEl("div", "pb-lab", "basis", ctl);
  const vrow = htmlEl("div", "qb-presets", "", ctl);
  const vb = {
    mono: button(vrow, "monomials", () => { st.view = "mono"; refresh(); }),
    leg: button(vrow, "Legendre", () => { st.view = "leg"; refresh(); }),
    gs: button(vrow, "Gram–Schmidt", () => { st.view = "gs"; refresh(); }),
  };
  const nS = slider(ctl, "degree N", 0, NMAX, 1, st.N, v => { st.N = v; if (st.gs > v + 1) st.gs = v + 1; refresh(); });
  const grow = htmlEl("div", "qb-presets", "", ctl);
  button(grow, "Gram–Schmidt step", () => { st.view = "gs"; st.gs = st.gs > st.N ? 1 : st.gs + 1; refresh(); });
  htmlEl("div", "pb-lab", "signal", ctl);
  const srow = htmlEl("div", "qb-presets", "", ctl);
  const sb = SIGNALS.map(s => button(srow, s.label, () => { st.sig = s.id; refresh(); }));
  htmlEl("div", "pb-lab", "arithmetic", ctl);
  const arow = htmlEl("div", "qb-presets", "", ctl);
  const ab = [button(arow, "64-bit", () => { st.single = false; refresh(); }), button(arow, "32-bit", () => { st.single = true; refresh(); })];
  const note = htmlEl("div", "pb-note", "", ctl);
  htmlEl("div", "pb-lab", "examples", ctl);
  const prow = htmlEl("div", "qb-presets", "", ctl);

  const onSlide = !!host.closest(".reveal");
  const basisPlot = new Plot(topL, { width: 560, height: onSlide ? 240 : 280, x: [-1, 1], y: [-1.15, 1.15], xlabel: "x", font: 17, margin: { b: 44 } });
  basisPlot.hline(0, { color: "var(--qb-axis, #444)", width: 1 });
  const gramTitle = htmlEl("div", "", "", topR);
  const gramBox = htmlEl("div", "", "", topR);
  const fitPlot = new Plot(botL, { width: 560, height: onSlide ? 215 : 250, x: [-1, 1], y: [-0.3, 1.3], xlabel: "x", font: 17, margin: { b: 44 } });
  const errPlot = new Plot(botR, { width: 380, height: onSlide ? 230 : 250, x: [0, NMAX], y: [1e-3, 10], ylog: true, xlabel: "degree N", ylabel: "rel. error", font: 17 });
  const eM = errPlot.line([], [], { color: CM, label: "monomials" });
  const eL = errPlot.line([], [], { color: CL, label: "Legendre" });
  const eNow = errPlot.vline(st.N, { color: "var(--qb-muted, #999)", dash: "4 3", width: 1.5 });
  errPlot.legend(undefined, "bl");
  const fSig = fitPlot.line([], [], { color: "var(--qb-fg, #222)", width: 2, label: "signal" });
  const fM = fitPlot.line([], [], { color: CM, width: 3, label: "monomials" });
  const fL = fitPlot.line([], [], { color: CL, width: 3, dash: "7 4", label: "Legendre" });
  fitPlot.legend(undefined, "tl");
  let errKey = "";

  function refresh(): void {
    Object.entries(vb).forEach(([k, b]) => b.classList.toggle("active", k === st.view));
    sb.forEach((b, i) => b.classList.toggle("active", SIGNALS[i].id === st.sig));
    ab.forEach((b, i) => b.classList.toggle("active", i === (st.single ? 1 : 0)));
    nS.set(st.N);
    const N = st.N;
    // ---- basis view ----
    basisPlot.clear();
    basisPlot.hline(0, { color: "var(--qb-axis, #444)", width: 1 });
    let shown: Float64Array[];
    if (st.view === "gs") {
      const g = gramSchmidt(N, st.gs);
      shown = g.set;
      for (let j = 0; j < st.gs - 1; j++) basisPlot.line(XS, g.set[j], { color: "var(--qb-muted, #999)", width: 1.5 });
      if (g.step) {
        basisPlot.line(XS, g.step.mono, { color: CM, width: 2.5, dash: "6 4", label: `x${sup(st.gs - 1)}` });
        const sum = XS.map((_, i) => g.step!.proj.reduce((s, p) => s + p[i], 0));
        if (g.step.proj.length) basisPlot.line(XS, sum, { color: "var(--qb-c3, #2ca02c)", width: 2.5, label: "projection on earlier" });
        basisPlot.line(XS, g.step.resid, { color: CL, width: 3.5, label: `new P${"₀₁₂₃₄₅₆₇₈₉"[st.gs - 1] ?? ""} (scaled)` });
      }
      basisPlot.setLabels({ title: `Gram–Schmidt step ${st.gs}: x${sup(st.gs - 1)} minus its projection` });
    } else {
      shown = st.view === "mono" ? monomials(N) : legendre(N, XS);
      shown.forEach((b, k) => basisPlot.line(XS, b, { color: seriesColor(k), width: k >= N - 1 && N >= 2 ? 3.5 : 2 }));
      basisPlot.setLabels({ title: st.view === "mono" ? `monomials 1, x, …, x${sup(N)}` : `Legendre P₀ … P${"₀₁₂₃₄₅₆₇₈₉"[N] ?? sup(N)}` });
    }
    if (st.view === "gs") basisPlot.legend(undefined, "br");
    else basisPlot.svg.querySelector(":scope > g.qb-legend")?.remove();
    // ---- Gram matrix (cosines) ----
    const n = shown.length, G = gram(shown), C = new Float64Array(n * n);
    for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) C[j * n + k] = G[j * n + k] / Math.sqrt(G[j * n + j] * G[k * n + k]);
    const kappa = condSym(G, n, (A, m) => jacobiEigen(A, m));
    gramTitle.innerHTML = `Gram matrix as cosines<br>condition number κ = <b>${sci(kappa)}</b>`;
    gramBox.replaceChildren();
    const cell = Math.max(10, Math.min(40, Math.floor((onSlide ? 210 : 300) / n)));
    heatmapGrid(gramBox, C, n, n, { map: diverging, lo: -1, hi: 1, numbers: n <= 8, fmt: v => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2), cell, font: Math.min(16, cell * 0.45) + 2, rowLabels: n <= 12 ? Array.from({ length: n }, (_, k) => String(k)) : undefined, colLabels: n <= 12 ? Array.from({ length: n }, (_, k) => String(k)) : undefined });
    // ---- fits ----
    const f = signal(st.sig);
    const fm = fitMonomial(f, N, st.single), fl = fitLegendre(f, N, st.single);
    fSig.set(XS, f); fM.set(XS, fm); fL.set(XS, fl);
    fitPlot.setLabels({ title: `degree ${N}: error monomials ${relErr(fm, f).toFixed(4)}, Legendre ${relErr(fl, f).toFixed(4)}` });
    const key = st.sig + st.single;
    if (key !== errKey) {
      errKey = key;
      const ns = Array.from({ length: NMAX + 1 }, (_, k) => k);
      eM.set(ns, ns.map(k => Math.min(10, relErr(fitMonomial(f, k, st.single), f))));
      eL.set(ns, ns.map(k => Math.min(10, relErr(fitLegendre(f, k, st.single), f))));
    }
    eNow.set(N);
    note.textContent = st.view === "gs" ? `Step ${st.gs} of ${N + 1}. The first ${st.gs} functions are orthogonal: the top-left block of the Gram matrix is diagonal.` : `${NS} samples on [−1, 1]; ⟨g, h⟩ = Σ g h Δx.`;
  }

  const presets: { id: string; label: string; s: Partial<typeof st> }[] = [
    { id: "alike", label: "monomials look alike", s: { view: "mono", N: 10, sig: "abs", single: false } },
    { id: "gs", label: "Gram–Schmidt to P₃", s: { view: "gs", N: 3, gs: 4, sig: "step", single: false } },
    { id: "step", label: "fit a step", s: { view: "leg", N: 12, sig: "step", single: false } },
    { id: "breakdown", label: "monomials break down", s: { view: "mono", N: 24, sig: "bump", single: false } },
  ];
  const btns = presets.map(p => button(prow, p.label, () => load(p)));
  const p0 = presets.find(p => p.id === host.dataset.preset) ?? presets[0];
  button(prow, "⟲ reset", () => load(p0));
  function load(p: typeof presets[number]): void {
    Object.assign(st, { view: "mono", N: 6, gs: 1, sig: "step", single: false }, p.s);
    if (p === p0 && host.dataset.n) st.N = Math.max(0, Math.min(NMAX, Number(host.dataset.n)));
    btns.forEach((b, i) => b.classList.toggle("active", presets[i] === p));
    refresh();
  }
  load(p0);
}

autoMount(NAME, mount);
