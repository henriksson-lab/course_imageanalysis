// Diffusion priors for an inverse problem, in 1D so the truth, the band and the null space are
// curves. A signal of 64 samples (one or two spots) is measured through an ideal low-pass
// (explicit cut-off), a blur, noise only, or with a gap. Prior: 400 training signals (narrow
// Gaussians around them), matched or mismatched. Panels: truth, measurement, Tikhonov (the
// regularised estimate of the deconvolution day, λ by the discrepancy principle) and the
// posterior mean; posterior samples with the 90 % band; and the part of each sample the
// measurement cannot see ("what came from the prior"). Samplers: exact posterior, or the
// approximate guided reverse diffusion (clean-estimate push; replacement for a gap).
//
// <div data-demo="posterior1d" data-preset="resolved|below|noclose|consistent|approx|coverage|gap"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import * as M from "./model";
import { injectStyle, choiceRow, helperCss, fmt } from "../scorematch1d/dview";

const NAME = "posterior1d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .p1-plots { flex: 1 1 600px; min-width: 0; max-width: 700px; display: flex; flex-direction: column; }
${S} .qb-panel { flex: 1 1 300px; gap: .35em; }
${S} .qb-readout { font-size: 15px; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .p1-plots { flex: 0 0 640px; }
.reveal ${S} .qb-panel { flex: 1 1 460px; gap: .12em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-slider { gap: .3em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .15em .4em; line-height: 1.2; }
.reveal ${S} .qb-note { display: none; }
`;

type Sampler = "exact" | "approx";
interface State { meas: M.MeasId; cutoff: number; blur: number; noise: number; sep: number; prior: M.PriorId; sampler: Sampler; zeta: number; truth: boolean; seed: number }
interface Preset { id: string; label: string; s: Partial<State>; cover?: boolean }
const BASE: State = { meas: "lowpass", cutoff: 6, blur: 2, noise: 0.05, sep: 4, prior: "matched", sampler: "exact", zeta: 1, truth: true, seed: 7 };
const PRESETS: Preset[] = [
  { id: "resolved", label: "two spots resolved: all agree", s: { sep: 8, cutoff: 24 } },
  { id: "below", label: "below the cut-off: samples split", s: { sep: 4, cutoff: 6 } },
  { id: "noclose", label: "prior without close pairs", s: { sep: 4, cutoff: 5, prior: "noclose" } },
  { id: "consistent", label: "consistent and wrong", s: { sep: 4, cutoff: 5, prior: "wide" } },
  { id: "approx", label: "approximate sampler", s: { sep: 4, cutoff: 6, sampler: "approx" } },
  { id: "coverage", label: "coverage: right vs wrong prior", s: { sep: 4, cutoff: 6 }, cover: true },
  { id: "gap", label: "gap: the prior fills it", s: { meas: "gap", sep: 6 } },
];
const GAP: [number, number] = [26, 38];
const NS = 32;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  // larger labels on slides (the plots are scaled down there)
  const F = host.closest(".reveal") ? 23 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const st: State = { ...BASE, ...p0.s };
  const trains: Partial<Record<M.PriorId, Float64Array[]>> = {};
  const train = (p: M.PriorId) => (trains[p] ??= M.trainingSet(p));

  const plots = htmlEl("div", "p1-plots", "", host);
  const xs = Array.from({ length: M.N }, (_, i) => i);
  const mk = (h: number, ylab: string, last = false, y: [number, number] = [-0.2, 1.5]) => new Plot(plots, { width: 660, height: h, x: [0, M.N - 1], y, ylabel: ylab, xticks: last ? 8 : [], yticks: 3, font: F, margin: { l: 3.9 * F, r: 10, t: 6, b: last ? 2.4 * F + 10 : 6 }, xlabel: last ? "position (samples)" : undefined });
  const pA = mk(170, "signal"), pB = mk(170, "samples"), pC = mk(150, "unseen", true, [-0.6, 0.6]);
  const gapA = pA.band([], [], [], { color: "var(--qb-c4)", fillOpacity: 0.12 });
  const meas = pA.line([], [], { color: "#999", width: 3 });
  const tik = pA.line([], [], { color: seriesColor(2), width: 3 });
  const mean = pA.line([], [], { color: seriesColor(0), width: 3 });
  const truthA = pA.line([], [], { color: "var(--qb-fg)", width: 2.5, dash: "6 4" });
  pA.legend([{ label: "measurement", color: "#999" }, { label: "Tikhonov", color: seriesColor(2) }, { label: "posterior mean", color: seriesColor(0) }, { label: "truth", color: "var(--qb-fg)", dash: "6 4" }], "tr");
  const bandB = pB.band([], [], [], { color: seriesColor(1), fillOpacity: 0.18 });
  const samp = pB.mark<[Float64Array[]]>((g, Ss) => drawLines(pB, g, Ss, seriesColor(1)), [[]]);
  const truthB = pB.line([], [], { color: "var(--qb-fg)", width: 2.5, dash: "6 4" });
  pB.legend([{ label: "5 samples", color: seriesColor(1) }, { label: "90 % band", color: seriesColor(1), kind: "box" }], "tr");
  const nullS = pC.mark<[Float64Array[]]>((g, Ss) => drawLines(pC, g, Ss, seriesColor(1)), [[]]);
  const nullT = pC.line([], [], { color: "var(--qb-fg)", width: 2.5, dash: "6 4" });
  pC.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" });
  pC.text(1, 0.45, "part of each sample the measurement cannot see", { size: F - 1 });

  function drawLines(p: Plot, g: SVGGElement, Ss: Float64Array[], col: string): void {
    for (const s of Ss) {
      let d = "";
      for (let i = 0; i < s.length; i++) d += (i ? "L" : "M") + p.x(i).toFixed(1) + "," + p.y(s[i]).toFixed(1);
      const e = document.createElementNS("http://www.w3.org/2000/svg", "path");
      e.setAttribute("d", d); e.style.fill = "none"; e.style.stroke = col; e.style.strokeWidth = "2"; e.style.opacity = "0.7";
      g.appendChild(e);
    }
  }

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const measC = choiceRow(panel, "measure", [{ v: "lowpass" as M.MeasId, label: "low-pass" }, { v: "blur" as M.MeasId, label: "blur" }, { v: "noise" as M.MeasId, label: "noise only" }, { v: "gap" as M.MeasId, label: "gap" }], st.meas, v => { st.meas = v; later(); });
  const cutS = slider(panel, "cut-off (cosines kept)", 2, 40, 1, st.cutoff, v => { st.cutoff = v; later(); });
  const blurS = slider(panel, "blur σ", 0.5, 5, 0.1, st.blur, v => { st.blur = v; later(); });
  const sepS = slider(panel, "truth: spot separation", 0, 14, 1, st.sep, v => { st.sep = v; later(); });
  const priC = choiceRow(panel, "prior", [{ v: "matched" as M.PriorId, label: "matched" }, { v: "noclose" as M.PriorId, label: "no close pairs" }, { v: "wide" as M.PriorId, label: "wider spots" }], st.prior, v => { st.prior = v; later(); });
  const smpC = choiceRow(panel, "sampler", [{ v: "exact" as Sampler, label: "exact posterior" }, { v: "approx" as Sampler, label: "approximate (diffusion)" }], st.sampler, v => { st.sampler = v; later(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const trC = checkbox(row, "truth", st.truth, v => { st.truth = v; later(); });
  button(row, "resample", () => { st.seed++; later(); });
  button(row, "coverage", () => runCoverage());
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Prior: 400 training signals with one or two spots (narrow Gaussians around each), so the posterior is exact. Coverage: 200 truths drawn from the matched prior, measured; the fraction of positions where the truth lies inside the 90 % band of the posterior under the chosen prior.", panel);
  let coverText = "", coverKey = "";
  const keyOf = () => [st.meas, st.cutoff, st.blur, st.prior].join(":");

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    measC.set(st.meas); cutS.set(st.cutoff); blurS.set(st.blur); sepS.set(st.sep); priC.set(st.prior); smpC.set(st.sampler); trC.checked = st.truth;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    coverText = "";
    draw();
    if (p.cover) runCoverage();
  }

  function runCoverage(): void {
    const m = measOf();
    ro.textContent += "\ncomputing coverage …";
    setTimeout(() => {
      const c = M.coverage(st.prior, m, 200), cm = st.prior === "matched" ? c : M.coverage("matched", m, 200);
      coverKey = keyOf();
      coverText = `\ncoverage of the 90 % band: ${(100 * c).toFixed(1)} % (${st.prior} prior)` + (st.prior === "matched" ? "" : `, matched ${(100 * cm).toFixed(1)} %`);
      draw();
    }, 30);
  }

  let timer = 0;
  function later(): void { clearTimeout(timer); timer = window.setTimeout(draw, st.sampler === "approx" ? 150 : 0); }

  const measOf = (): M.Meas => ({ id: st.meas, blur: st.blur, cutoff: st.cutoff, gap: GAP, noise: st.noise });

  function draw(): void {
    const m = measOf(), x = M.testSignal(st.sep), y = M.measure(x, m, 3);
    (cutS.input.parentElement as HTMLElement).style.display = st.meas === "lowpass" ? "" : "none";
    (blurS.input.parentElement as HTMLElement).style.display = st.meas === "blur" ? "" : "none";
    const tr = train(st.prior), post = M.posterior(tr, y, m), mu = M.postMeanSignal(post), tk = M.tikhonov(y, m);
    const Ss = st.sampler === "exact" ? M.postSamples(post, NS, st.seed) : M.approxSamples(tr, y, m, NS, 100, st.zeta, st.seed);
    const b = M.band(Ss);
    gapA.set(st.meas === "gap" ? [GAP[0] - 0.5, GAP[1] - 0.5] : [], [-0.2, -0.2], [1.5, 1.5]);
    meas.set(xs, M.measuredSignal(y, m));
    tik.set(xs, tk.x); mean.set(xs, mu);
    truthA.set(st.truth ? xs : [], st.truth ? x : []); truthB.set(st.truth ? xs : [], st.truth ? x : []);
    bandB.set(xs, b.lo, b.hi);
    samp.set(Ss.slice(0, 5));
    nullS.set(Ss.slice(0, 5).map(s => M.nullPart(s, m)));
    const nt = M.nullPart(x, m);
    nullT.set(st.truth ? xs : [], st.truth ? nt : []);
    const cnt = [0, 0, 0]; for (const s of Ss) cnt[Math.min(2, M.countSpots(s))]++;
    let inside = 0; for (let i = 0; i < M.N; i++) if (x[i] >= b.lo[i] && x[i] <= b.hi[i]) inside++;
    const rr = Ss.map(s => M.residualRatio(s, y, m)).sort((a, c) => a - c);
    ro.textContent =
      `spots: truth ${M.countSpots(x)}, Tikhonov ${M.countSpots(tk.x)}, mean ${M.countSpots(mu)}\n` +
      `  ${NS} samples: 1 spot ${cnt[1]}, 2 spots ${cnt[2]}` + (cnt[0] ? `, none ${cnt[0]}` : "") + `\n` +
      `residual / expected: Tikhonov ${fmt(M.residualRatio(tk.x, y, m), 2)}, mean ${fmt(M.residualRatio(mu, y, m), 2)}, samples (median) ${fmt(rr[rr.length >> 1], 2)}\n` +
      `truth inside the 90 % band at ${inside} of ${M.N} positions` + (coverKey === keyOf() ? coverText : "");
  }

  apply(p0);
}

autoMount(NAME, mount);
