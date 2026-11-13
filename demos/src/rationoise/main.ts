// The noise of a ratio (and of a difference). Monte Carlo with seeded draws: 20 000 pairs of
// photon counts (Poisson, or Gaussian with the same variance) with means N_A and N_D,
// optionally multiplied by a shared illumination flicker, optionally binned (b pixels summed
// per draw). Tab "ratio": histogram of A/D (or log A/D) with the true ratio, the observed
// mean and the delta-method normal curve; readouts compare the mean of ratios, the ratio of
// sums over regions of m pixels, and the observed CV with 1/N_A + 1/N_D. Tab "difference":
// the three-image Fc = IDA − d·IDD − a·IAA. Tab "bleach": E = 1 − P/Q.
//
// <div data-demo="rationoise" data-preset="bright|dim|log|sums|flicker|lowfret|bleach" data-tab="ratio|diff|bleach"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, rng } from "../lib/ui";
import { Plot } from "../lib/plot";
import { poisson, gauss } from "../lib/random";
import * as F from "../lib/fret";
import { injectStyle, fmt, filterPresets, hideControl } from "../threecube/view";

const NAME = "rationoise";
const Q = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${Q} { gap: .8em; align-items: flex-start; }
${Q} .rn-plot { flex: 1 1 520px; min-width: 280px; max-width: 640px; display: flex; flex-direction: column; gap: .3em; }
${Q} .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .35em; }
${Q} .rn-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${Q} .qb-readout { font-size: 15px; }
${Q} .qb-btn { padding: .12em .5em; }
.reveal ${Q} { font-size: 18px; flex-wrap: nowrap; margin: 0; }
.reveal ${Q} .rn-plot { flex: 0 0 600px; }
.reveal ${Q} .qb-panel { flex: 0 0 470px; gap: .15em; }
.reveal ${Q} .qb-readout { font-size: 18px; padding: .2em .4em; line-height: 1.22; }
.reveal ${Q} .qb-btn { font-size: 18px; padding: .02em .3em; }
`;

type Tab = "ratio" | "diff" | "bleach";
interface State {
  tab: Tab; NA: number; ND: number; flicker: number; m: number; bin: number; log: boolean; gaussian: boolean;
  IDA: number; IDD: number; IAA: number; d: number; a: number;
  P: number; Qb: number;
}
const DEFAULT: State = { tab: "ratio", NA: 100, ND: 400, flicker: 0, m: 16, bin: 1, log: false, gaussian: false, IDA: 500, IDD: 400, IAA: 1000, d: 0.5, a: 0.1, P: 600, Qb: 800 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "bright", label: "bright: the formula works", s: { NA: 400, ND: 1000 } },
  { id: "dim", label: "dim denominator: bias and a long tail", s: { NA: 20, ND: 5 } },
  { id: "log", label: "log ratio is symmetric", s: { NA: 20, ND: 5, log: true } },
  { id: "sums", label: "ratio of sums beats mean of ratios", s: { NA: 10, ND: 10, m: 16 } },
  { id: "flicker", label: "shared flicker cancels", s: { NA: 500, ND: 500, flicker: 0.2 } },
  { id: "lowfret", label: "low FRET: the corrected signal is mostly noise", s: { tab: "diff", IDA: 320 } },
  { id: "bleach", label: "bleaching: E from 600 and 800 photons", s: { tab: "bleach" } },
];
const NDRAW = 20000;
const lg = (u: number) => Math.round(10 ** u);

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const pre = PRESETS.find(p => p.id === host.dataset.preset);
  const tab0 = (host.dataset.tab as Tab) || undefined;
  const start = (): State => ({ ...DEFAULT, ...(tab0 ? { tab: tab0 } : {}), ...(pre?.s ?? {}) });
  const st: State = start();

  const box = htmlEl("div", "rn-plot", "", host);
  const tabRow = htmlEl("div", "rn-row", "", box);
  const tabBtns = (["ratio", "diff", "bleach"] as Tab[]).map(t => button(tabRow, t === "ratio" ? "ratio A/D" : t === "diff" ? "difference Fc" : "bleach E = 1 − P/Q", () => { st.tab = t; off(); show(); draw(); }));
  const p = new Plot(box, { width: 600, height: 380, x: [0, 1], y: [0, 1], xlabel: "", ylabel: "density", yticks: [], margin: { l: 30, r: 14, t: 10, b: 56 } });
  const hist = p.histogram([0, 1], [0], { color: "#1f77b4", fillOpacity: 0.45 });
  const curve = p.fn(() => 0, { color: "#111", width: 3 }, 300);
  const vTrue = p.vline(0, { color: "#2ca02c", width: 4, text: "true" });
  const vMean = p.vline(0, { color: "#d62728", width: 3, dash: "7 5", text: "mean" });
  const legend = htmlEl("div", "qb-note", "green: true value · red dashes: mean of the draws · black: normal curve with the delta-method spread", box);
  void legend;

  const panel = htmlEl("div", "qb-panel", "", host);
  const PL = filterPresets(host, PRESETS);
  const presetBtns = presets(panel, PL.map(pp => ({ label: pp.label, apply: () => { Object.assign(st, DEFAULT, pp.s); sync(); show(); draw(); } })));
  const ratioBox = htmlEl("div", "", "", panel); ratioBox.style.display = "contents";
  const sNA = slider(ratioBox, "N_A (photons)", 0, 4, 0.01, Math.log10(st.NA), u => { st.NA = lg(u); off(); draw(); }, u => String(lg(u)));
  const sND = slider(ratioBox, "N_D (photons)", 0, 4, 0.01, Math.log10(st.ND), u => { st.ND = lg(u); off(); draw(); }, u => String(lg(u)));
  const sFl = slider(ratioBox, "shared flicker (sd)", 0, 0.4, 0.01, st.flicker, v => { st.flicker = v; off(); draw(); });
  const sM = slider(ratioBox, "pixels per region m", 1, 64, 1, st.m, v => { st.m = v; off(); draw(); });
  const sB = slider(ratioBox, "binning (pixels summed)", 1, 16, 1, st.bin, v => { st.bin = v; off(); draw(); });
  const rowC = htmlEl("div", "rn-row", "", ratioBox);
  const cLog = checkbox(rowC, "plot log ratio", st.log, v => { st.log = v; off(); draw(); });
  const cG = checkbox(rowC, "Gaussian instead of Poisson", st.gaussian, v => { st.gaussian = v; off(); draw(); });
  const diffBox = htmlEl("div", "", "", panel); diffBox.style.display = "contents";
  const sDA = slider(diffBox, "mean IDA", 0, 2000, 10, st.IDA, v => { st.IDA = v; off(); draw(); });
  const sDD = slider(diffBox, "mean IDD", 0, 2000, 10, st.IDD, v => { st.IDD = v; off(); draw(); });
  const sAA = slider(diffBox, "mean IAA", 0, 2000, 10, st.IAA, v => { st.IAA = v; off(); draw(); });
  const sd = slider(diffBox, "d", 0, 1, 0.01, st.d, v => { st.d = v; off(); draw(); });
  const sa = slider(diffBox, "a", 0, 0.4, 0.01, st.a, v => { st.a = v; off(); draw(); });
  const blBox = htmlEl("div", "", "", panel); blBox.style.display = "contents";
  const sP = slider(blBox, "P: donor before", 5, 2000, 5, st.P, v => { st.P = v; off(); draw(); });
  const sQ = slider(blBox, "Q: donor after", 5, 2000, 5, st.Qb, v => { st.Qb = v; off(); draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); show(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const off = () => presetBtns.forEach(b => b.classList.remove("active"));
  if (host.dataset.compact !== undefined) [sB].forEach(hideControl);
  function sync(): void {
    sNA.set(Math.log10(st.NA)); sND.set(Math.log10(st.ND)); sFl.set(st.flicker); sM.set(st.m); sB.set(st.bin); cLog.checked = st.log; cG.checked = st.gaussian;
    sDA.set(st.IDA); sDD.set(st.IDD); sAA.set(st.IAA); sd.set(st.d); sa.set(st.a); sP.set(st.P); sQ.set(st.Qb);
  }
  function show(): void {
    const vis = (el: HTMLElement, on: boolean) => el.querySelectorAll<HTMLElement>(":scope > *").forEach(c => { c.style.display = on ? "" : "none"; });
    vis(ratioBox, st.tab === "ratio"); vis(diffBox, st.tab === "diff"); vis(blBox, st.tab === "bleach");
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["ratio", "diff", "bleach"][i] === st.tab));
  }

  function histOf(v: ArrayLike<number>, lo: number, hi: number, bins = 80): { edges: number[]; dens: number[]; out: number } {
    const counts = new Array(bins).fill(0), w = (hi - lo) / bins;
    let out = 0, n = 0;
    for (let i = 0; i < v.length; i++) { const x = v[i]; if (!Number.isFinite(x)) { out++; continue; } n++; const b = Math.floor((x - lo) / w); if (b >= 0 && b < bins) counts[b]++; else out++; }
    return { edges: Array.from({ length: bins + 1 }, (_, i) => lo + i * w), dens: counts.map(c => c / (Math.max(1, n) * w)), out };
  }
  const meanSd = (v: ArrayLike<number>) => { let s = 0, ss = 0, n = 0; for (let i = 0; i < v.length; i++) if (Number.isFinite(v[i])) { s += v[i]; ss += v[i] * v[i]; n++; } const m = s / n; return { m, sd: Math.sqrt(Math.max(0, ss / n - m * m)), n }; };
  const npdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

  function draw(): void {
    const lines: string[] = [];
    if (st.tab === "ratio") {
      const b = st.bin;
      const mc = F.ratioDraws({ muA: st.NA, muD: st.ND, n: NDRAW, flicker: st.flicker, bin: b, gaussian: st.gaussian, seed: 7 });
      const truth = st.NA / st.ND;
      const cvP = F.ratioCVPredicted(st.NA * b, st.ND * b);
      const vals = st.log ? Float64Array.from(mc.ratios, r => (r > 0 && Number.isFinite(r) ? Math.log(r) : NaN)) : mc.ratios;
      const t = st.log ? Math.log(truth) : truth, spread = st.log ? cvP : truth * cvP;
      const lo = st.log ? t - 5 * Math.max(spread, 0.05) : 0, hi = st.log ? t + 5 * Math.max(spread, 0.05) : truth * (1 + Math.max(5 * cvP, 0.3));
      const h = histOf(vals, lo, hi);
      const ms = meanSd(vals);
      p.setDomain([lo, hi], [0, Math.max(...h.dens, npdf(t, t, spread)) * 1.1 || 1]);
      p.setLabels({ xlabel: st.log ? "log(A/D)" : "ratio A/D per draw" });
      hist.set(h.edges, h.dens); curve.set(x => npdf(x, t, spread)); vTrue.set(t); vMean.set(ms.m);
      // regions of m draws: ratio of sums vs mean of ratios
      const m = st.m, G = Math.floor(NDRAW / m);
      let sumRos = 0, sumMor = 0, nR = 0, nM = 0;
      for (let g = 0; g < G; g++) {
        let sa = 0, sdd = 0, s = 0, k = 0;
        for (let j = g * m; j < (g + 1) * m; j++) { sa += mc.A[j]; sdd += mc.D[j]; if (mc.D[j] > 0) { s += mc.A[j] / mc.D[j]; k++; } }
        if (sdd > 0) { sumRos += sa / sdd; nR++; }
        if (k) { sumMor += s / k; nM++; }
      }
      const rm = meanSd(mc.ratios);
      const inf = mc.ratios.reduce((c, r) => c + (Number.isFinite(r) ? 0 : 1), 0);
      lines.push(`true ratio N_A/N_D = ${fmt(truth, 3)}`);
      lines.push(`mean of pixel ratios = ${fmt(rm.m, 3)} (${fmt(100 * (rm.m / truth - 1), 1)} %)`);
      lines.push(`  delta method predicts ${fmt(100 * (F.ratioMeanPredicted(st.NA * b, st.ND * b) / (st.NA * b / (st.ND * b)) - 1), 1)} %`);
      lines.push(`regions of ${m}: ratio of sums ${fmt(sumRos / Math.max(1, nR), 3)}`);
      lines.push(`regions of ${m}: mean of ratios ${fmt(sumMor / Math.max(1, nM), 3)}`);
      lines.push(`CV per draw: observed ${fmt(100 * rm.sd / rm.m, 1)} %, predicted ${fmt(100 * cvP, 1)} %`);
      if (inf) lines.push(`draws with D = 0 (no ratio): ${inf}`);
    } else if (st.tab === "diff") {
      const r = rng(11), v = new Float64Array(NDRAW);
      for (let i = 0; i < NDRAW; i++) v[i] = poisson(r, st.IDA) - st.d * poisson(r, st.IDD) - st.a * poisson(r, st.IAA);
      const truth = F.correctFc(st.IDA, st.IDD, st.IAA, st.d, st.a), sdP = Math.sqrt(F.fcVariance(st.IDA, st.IDD, st.IAA, st.d, st.a));
      const ms = meanSd(v), lo = truth - 5 * sdP, hi = truth + 5 * sdP;
      const h = histOf(v, lo, hi);
      p.setDomain([lo, hi], [0, Math.max(...h.dens, npdf(truth, truth, sdP)) * 1.1 || 1]);
      p.setLabels({ xlabel: "Fc = IDA − d·IDD − a·IAA per pixel" });
      hist.set(h.edges, h.dens); curve.set(x => npdf(x, truth, sdP)); vTrue.set(truth); vMean.set(ms.m);
      lines.push(`true Fc = ${fmt(truth, 1)}`);
      lines.push(`sd predicted √(IDA + d²IDD + a²IAA) = ${fmt(sdP, 1)}`);
      lines.push(`sd observed = ${fmt(ms.sd, 1)}`);
      lines.push(`relative noise = ${fmt(100 * sdP / Math.abs(truth), 0)} %`);
      lines.push(`E_app (G = 2) = ${fmt(F.eFret(truth, st.IDD, 2), 3)}`);
    } else {
      const r = rng(13), v = new Float64Array(NDRAW);
      for (let i = 0; i < NDRAW; i++) { const P = poisson(r, st.P), Qq = poisson(r, st.Qb); v[i] = Qq > 0 ? 1 - P / Qq : NaN; }
      const truth = F.eFromBleach(st.P, st.Qb), sdP = F.bleachSigmaE(st.P, st.Qb);
      const ms = meanSd(v), lo = truth - 5 * sdP, hi = truth + 5 * sdP;
      const h = histOf(v, lo, hi);
      p.setDomain([lo, hi], [0, Math.max(...h.dens, npdf(truth, truth, sdP)) * 1.1 || 1]);
      p.setLabels({ xlabel: "E = 1 − P/Q per pixel" });
      hist.set(h.edges, h.dens); curve.set(x => npdf(x, truth, sdP)); vTrue.set(truth); vMean.set(ms.m);
      lines.push(`true E = ${fmt(truth, 3)}`);
      lines.push(`σ_E predicted (P/Q)√(1/P + 1/Q) = ${fmt(sdP, 3)}`);
      lines.push(`σ_E observed = ${fmt(ms.sd, 3)}, mean ${fmt(ms.m, 3)}`);
    }
    ro.textContent = lines.join("\n");
    void gauss;
  }

  sync(); show(); draw();
  const pi = PL.findIndex(pp => pp.id === host.dataset.preset);
  if (pi >= 0) presetBtns[pi].classList.add("active");
}

autoMount(NAME, mount);
