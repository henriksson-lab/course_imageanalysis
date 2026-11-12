// Continuous MRFs on a 1D profile: quadratic smoothing (Gaussian MRF), Huber and total
// variation. The profile (a step, a ramp and a narrow peak; plateaus; or a line across the
// micrograph) gets seeded Gaussian noise; the prior and lambda are chosen; the plot shows the
// noisy data, the truth and the estimate with the energy split and the error. Tabs: five draws
// from the quadratic and the TV prior (same step variance), and the matrix I + lambda L with
// the frequency response of quadratic smoothing.
// Auto-mounts on <div data-demo="mrf1d">. Options: data-preset="quad|tv|stair|peak|huber|none", data-tab="fit|prior|matrix".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { rng } from "../lib/random";
import { Plot, seriesColor, heatmapGrid } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { diverging } from "../lib/colour";
import { denoise1d, energy1d, Prior, samplePrior1d, gmrfMatrix, gmrfResponse } from "../lib/smooth";
import { N1, Profile, PRESETS, Preset, mixedProfile, stepsProfile, noisy, rmse } from "./model";

const NAME = "mrf1d";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] .m1-left { flex: 1 1 560px; max-width: 680px; min-width: 280px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .m1-two { display: grid; grid-template-columns: 1fr 1fr; gap: .4em; align-items: start; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .m1-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .m1-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .m1-cap { font-size: 16px; text-align: center; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .5em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const PRIORS: Prior[] = ["none", "quadratic", "huber", "tv"];
const PRIOR_LABEL: Record<Prior, string> = { none: "none", quadratic: "quadratic", huber: "Huber", tv: "TV" };
const lamOf = (u: number) => Number((10 ** u).toPrecision(2));
const uOf = (l: number) => Math.log10(l);
const NM = 10;   // size of the matrix shown

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { ...p0, tab: (["fit", "prior", "matrix"].includes(host.dataset.tab ?? "") ? host.dataset.tab : "fit") as "fit" | "prior" | "matrix" };
  let cellRow: Float64Array | null = null;

  const left = htmlEl("div", "m1-left", "", host);
  const tabRow = htmlEl("div", "qb-presets", "", left);
  const tabBtns = [
    button(tabRow, "denoise", () => { st.tab = "fit"; draw(); }),
    button(tabRow, "prior samples", () => { st.tab = "prior"; draw(); }),
    button(tabRow, "the matrix", () => { st.tab = "matrix"; draw(); }),
  ];
  // fit tab
  const fitBox = htmlEl("div", "", "", left);
  const pf = new Plot(fitBox, { width: 680, height: 380, x: [0, N1 - 1], y: [-1, 3.2], xlabel: "position", ylabel: "value" });
  const mY = pf.scatter([], [], { color: "var(--qb-muted)", r: 3, label: "noisy data y" });
  const mT = pf.line([], [], { color: seriesColor(2), width: 2.5, dash: "7 4", label: "truth" });
  const mX = pf.line([], [], { color: seriesColor(0), width: 3.5, label: "estimate x̂" });
  pf.legend(undefined, "tr");
  const credit = htmlEl("div", "m1-credit", CELL_CREDIT, fitBox);
  // prior tab
  const priorBox = htmlEl("div", "", "", left);
  const two = htmlEl("div", "m1-two", "", priorBox);
  const figQ = htmlEl("div", "", "", two), figT = htmlEl("div", "", "", two);
  htmlEl("div", "m1-cap", "quadratic prior: Gaussian steps", figQ);
  htmlEl("div", "m1-cap", "TV prior: Laplace steps (same sd)", figT);
  const pq = new Plot(figQ, { width: 340, height: 260, x: [0, N1 - 1], y: [-5, 5], xlabel: "position" });
  const pt = new Plot(figT, { width: 340, height: 260, x: [0, N1 - 1], y: [-5, 5], xlabel: "position" });
  const pdf = new Plot(priorBox, { width: 680, height: 170, x: [-4, 4], y: [0, 1], xlabel: "one step, in units of its standard deviation", yticks: 2, margin: { l: 56 } });
  // matrix tab
  const matBox = htmlEl("div", "m1-two", "", left);
  const matFig = htmlEl("div", "", "", matBox);
  htmlEl("div", "m1-cap", `I + λL for a path of ${NM} samples`, matFig);
  const matHost = htmlEl("div", "", "", matFig);
  const respFig = htmlEl("div", "", "", matBox);
  htmlEl("div", "m1-cap", "quadratic smoothing as a filter", respFig);
  const pr = new Plot(respFig, { width: 340, height: 320, x: [0, 0.5], y: [0, 1.05], xlabel: "frequency (cycles/sample)", ylabel: "gain" });
  const mResp = pr.fn(() => 0, { color: seriesColor(0), width: 3.5 }, 300);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "m1-lab", "profile", panel);
  const rowS = htmlEl("div", "qb-presets", "", panel);
  const profBtns = (["mixed", "steps", "cell"] as Profile[]).map(k => button(rowS, k === "mixed" ? "step · ramp · peak" : k === "steps" ? "plateaus" : "micrograph line", () => { st.profile = k; st.view = undefined; draw(); }));
  const sSd = slider(panel, "noise σn", 0, 1, 0.05, st.sd, v => { st.sd = v; draw(); });
  htmlEl("div", "m1-lab", "prior", panel);
  const rowP = htmlEl("div", "qb-presets", "", panel);
  const priorBtns = PRIORS.map(p => button(rowP, PRIOR_LABEL[p], () => { st.prior = p; draw(); }));
  const sLam = slider(panel, "λ", -1, 2, 0.01, uOf(st.lambda), u => { st.lambda = lamOf(u); draw(); }, u => String(lamOf(u)));
  const sDel = slider(panel, "δ (Huber)", 0.02, 1, 0.02, st.delta, v => { st.delta = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "m1-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "reset", () => { load(p0); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });

  function load(p: Preset): void {
    Object.assign(st, p);
    if (p.view === undefined) st.view = undefined;
    sSd.set(st.sd); sLam.set(uOf(st.lambda)); sDel.set(st.delta);
    draw();
  }

  function truthOf(): Float64Array | null {
    if (st.profile === "mixed") return mixedProfile();
    if (st.profile === "steps") return stepsProfile();
    if (!cellRow) {
      loadCell(256).then(g => {
        // row 150 of the micrograph, first 200 pixels, scaled to 0..3 (dark = high)
        cellRow = Float64Array.from({ length: N1 }, (_, i) => 3 * (1 - g.data[150 * 256 + 28 + i] / 255));
        draw();
      });
      return null;
    }
    return cellRow;
  }

  function draw(): void {
    profBtns.forEach((b, i) => b.classList.toggle("active", ["mixed", "steps", "cell"][i] === st.profile));
    priorBtns.forEach((b, i) => b.classList.toggle("active", PRIORS[i] === st.prior));
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["fit", "prior", "matrix"][i] === st.tab));
    (sDel.input.parentElement as HTMLElement).style.display = st.prior === "huber" ? "" : "none";
    (sLam.input.parentElement as HTMLElement).style.opacity = st.prior === "none" ? "0.4" : "1";
    fitBox.style.display = st.tab === "fit" ? "" : "none";
    priorBox.style.display = st.tab === "prior" ? "" : "none";
    matBox.style.display = st.tab === "matrix" ? "" : "none";
    credit.style.visibility = st.profile === "cell" ? "visible" : "hidden";
    const t = truthOf();
    if (!t) return;
    const y = noisy(t, st.sd, 1);
    const x = denoise1d(y, st.prior, st.lambda, st.delta);
    const e = energy1d(x, y, st.prior, st.lambda, st.delta);
    const idx = Array.from({ length: N1 }, (_, i) => i);
    pf.setDomain(st.view ?? [0, N1 - 1], [-1, 3.6]);
    mY.set(idx, y); mT.set(idx, t); mX.set(idx, x);
    const lam = st.prior === "none" ? "—" : String(st.lambda);
    ro.textContent = [
      `prior ${PRIOR_LABEL[st.prior]}, λ = ${lam}`,
      `E = ${e.data.toFixed(2)} (data) + ${e.prior.toFixed(2)} (prior)`,
      `  = ${e.total.toFixed(2)}`,
      `error vs truth (RMSE): ${rmse(x, t).toFixed(3)}`,
      `noisy data (RMSE):     ${rmse(y, t).toFixed(3)}`,
    ].join("\n");
    if (st.tab === "prior") drawPrior();
    if (st.tab === "matrix") drawMatrix();
  }

  function drawPrior(): void {
    // same step variance 1/lambda for both: TV scale b with 2 b^2 = 1/lambda, i.e. lambda_TV = sqrt(2 lambda)
    const lam = st.prior === "none" ? 1 : st.lambda, lamTV = Math.sqrt(2 * lam);
    pq.clear(); pt.clear(); pdf.clear();
    const idx = Array.from({ length: N1 }, (_, i) => i);
    let ext = 0;
    const draws: [Float64Array[], Float64Array[]] = [[], []];
    for (let k = 0; k < 5; k++) {
      draws[0].push(samplePrior1d(rng(100 + k), N1, "quadratic", lam));
      draws[1].push(samplePrior1d(rng(200 + k), N1, "tv", lamTV));
    }
    for (const d of [...draws[0], ...draws[1]]) for (const v of d) ext = Math.max(ext, Math.abs(v));
    ext = Math.max(ext * 1.05, 1e-3);
    pq.setDomain(undefined, [-ext, ext]); pt.setDomain(undefined, [-ext, ext]);
    draws[0].forEach((d, k) => pq.line(idx, d, { color: seriesColor(k), width: 2 }));
    draws[1].forEach((d, k) => pt.line(idx, d, { color: seriesColor(k), width: 2 }));
    // densities of one step in units of its sd
    const g = (z: number) => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
    const l = (z: number) => Math.exp(-Math.SQRT2 * Math.abs(z)) / Math.SQRT2;   // Laplace with variance 1
    pdf.setDomain([-4, 4], [0, 0.75]);
    pdf.fn(g, { color: seriesColor(0), width: 3, label: "Gaussian (quadratic)" }, 300);
    pdf.fn(l, { color: seriesColor(1), width: 3, label: "Laplace (TV)" }, 400);
    pdf.legend(undefined, "tr");
  }

  function drawMatrix(): void {
    const lam = st.prior === "none" ? 0 : st.lambda;
    matHost.replaceChildren();
    const M = gmrfMatrix(NM, lam);
    const amax = Math.max(1e-9, ...Array.from(M, Math.abs));
    heatmapGrid(matHost, Float64Array.from(M, v => v / amax), NM, NM, {
      numbers: NM <= 12, text: (_v, i) => { const v = M[i]; return v === 0 ? "" : Math.abs(v) >= 10 ? v.toFixed(0) : String(Number(v.toFixed(1))); },
      lo: -1, hi: 1, map: diverging, cell: 34, font: 15,
    });
    mResp.set(f => gmrfResponse(2 * Math.PI * f, lam));
  }

  load(st);
  presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
}

autoMount(NAME, mount);
