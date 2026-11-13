// Frequency-domain lifetime measurement. The excitation is a wave at frequency f; the
// emission is the same wave, delayed by the phase φ and flattened (modulation m < 1), because
// it is the excitation convolved with the decay. Top: two periods of both waves (each
// normalised to the same mean). Bottom left: the decay (log axis) with its components. Bottom
// right: the phasor (m cos φ, m sin φ) on the semicircle. Readouts: φ, m, τ_φ and τ_m.
//
// <div data-demo="modulation" data-preset="one|two|low|high|best"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import * as D from "../lib/decay";
import { injectStyle } from "../canny/imgview";
import { logMap, phasorPlane, FLIM_CSS } from "../phasorplot/plane";

const NAME = "modulation";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { gap: .8em; flex-wrap: nowrap; align-items: flex-start; }
${SEL} .mo-plots { flex: 1 1 700px; min-width: 0; max-width: 760px; }
${SEL} .mo-row { display: flex; gap: .4em; }
${SEL} .mo-row > svg { flex: 1 1 50%; min-width: 0; }
${SEL} .qb-panel { flex: 0 1 330px; gap: .35em; }
.reveal ${SEL} .mo-plots { flex: 0 0 720px; }
.reveal ${SEL} .qb-panel { flex: 1 1 390px; font-size: 18px; }
@media (max-width: 760px) { ${SEL} { flex-wrap: wrap; } }
`;

interface State { tau1: number; tau2: number; frac: number; f: number }
const DEFAULT: State = { tau1: 2.5, tau2: 0.5, frac: 0, f: 80 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "one", label: "one lifetime: τ_φ = τ_m", s: { tau1: 2.5, frac: 0, f: 80 } },
  { id: "two", label: "two lifetimes: τ_φ < τ_m", s: { tau1: 4, tau2: 0.5, frac: 0.5, f: 80 } },
  { id: "low", label: "frequency too low: everything looks the same", s: { tau1: 2.5, frac: 0, f: 10 } },
  { id: "high", label: "frequency too high: the wave is flattened away", s: { tau1: 4, frac: 0, f: 200 } },
  { id: "best", label: "ωτ ≈ 1: best", s: { tau1: 2, frac: 0, f: 80 } },
];
const tauM = logMap(0.2, 8), fM = logMap(10, 200);

function mount(host: HTMLElement): void {
  injectStyle("flim-common", FLIM_CSS);
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const plots = htmlEl("div", "mo-plots fl-col", "", host);
  const wave = new Plot(plots, { width: 720, height: 270, x: [0, 2], y: [0, 2.15], xlabel: "time (periods of the excitation)", ylabel: "intensity", xticks: [0, 0.5, 1, 1.5, 2], margin: { l: 70, r: 12, t: 10, b: 60 } });
  const exc = wave.line([], [], { color: "var(--qb-c4)", width: 3, label: "excitation" });
  const em = wave.line([], [], { color: "var(--qb-c1)", width: 3.5, label: "emission" });
  const phiArrow = wave.arrow(0, 0, 0, 0, { color: "var(--qb-c2)", width: 2.5 });
  const phiTxt = wave.text(0, 0, "", { size: 18, color: "var(--qb-c2)" });
  const row = htmlEl("div", "mo-row", "", plots);
  const dec = new Plot(row, { width: 360, height: 280, x: [0, 10], y: [1e-3, 1.2], ylog: true, xlabel: "time (ns)", ylabel: "decay", margin: { l: 70, r: 10, t: 10, b: 60 } });
  const dSum = dec.line([], [], { color: "var(--qb-c1)", width: 3 });
  const d1 = dec.line([], [], { color: "var(--qb-c3)", width: 2, dash: "6 4" });
  const d2 = dec.line([], [], { color: "var(--qb-c5)", width: 2, dash: "6 4" });
  const pp = phasorPlane(row, D.omegaOf(st.f), { width: 360, height: 280, margin: { l: 56, r: 8, t: 8, b: 52 } });
  const chord = pp.plot.line([], [], { color: "var(--qb-muted)", width: 2, dash: "5 4" });
  const comps = pp.plot.scatter([], [], { r: 6 });
  const pt = pp.plot.scatter([], [], { color: "var(--qb-c2)", r: 9 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); draw(); } })));
  const s1 = slider(panel, "τ₁ (ns)", 0, 100, 0.5, tauM.u(st.tau1), u => { st.tau1 = tauM.v(u); draw(); }, u => tauM.v(u).toFixed(2));
  const s2 = slider(panel, "τ₂ (ns)", 0, 100, 0.5, tauM.u(st.tau2), u => { st.tau2 = tauM.v(u); draw(); }, u => tauM.v(u).toFixed(2));
  const sF = slider(panel, "photon fraction of τ₂", 0, 1, 0.01, st.frac, v => { st.frac = v; draw(); });
  const sFr = slider(panel, "frequency f (MHz)", 0, 100, 0.5, fM.u(st.f), u => { st.f = fM.v(u); draw(); }, u => fM.v(u).toFixed(0));
  button(panel, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout fl-ro", "", panel);

  function sync(): void { s1.set(tauM.u(st.tau1)); s2.set(tauM.u(st.tau2)); sF.set(st.frac); sFr.set(fM.u(st.f)); }

  function draw(): void {
    const w = D.omegaOf(st.f), dc: D.Decay = { tau: [st.tau1, st.tau2], f: [1 - st.frac, st.frac] };
    const P = D.decayPhasor(dc, w), pm = D.phaseModulation(P.g, P.s, w);
    const xs = Float64Array.from({ length: 241 }, (_, i) => i / 120);
    exc.set(xs, Float64Array.from(xs, x => 1 + Math.cos(2 * Math.PI * x)));
    em.set(xs, Float64Array.from(xs, x => 1 + pm.m * Math.cos(2 * Math.PI * x - pm.phi)));
    const x0 = 1, x1 = 1 + pm.phi / (2 * Math.PI);
    phiArrow.set(x0, 2.08, x1, 2.08);
    phiTxt.set(x1 + 0.02, 1.95, `φ = ${(pm.phi * 180 / Math.PI).toFixed(1)}°`);
    wave.legend(undefined, "br");
    const ts = Float64Array.from({ length: 201 }, (_, i) => i * 0.05);
    const c1 = (t: number) => (1 - st.frac) / st.tau1 * Math.exp(-t / st.tau1), c2 = (t: number) => st.frac / st.tau2 * Math.exp(-t / st.tau2);
    const nrm = c1(0) + c2(0);
    dSum.set(ts, Float64Array.from(ts, t => (c1(t) + c2(t)) / nrm));
    d1.set(ts, Float64Array.from(ts, t => (st.frac < 1 ? c1(t) / nrm : NaN)));
    d2.set(ts, Float64Array.from(ts, t => (st.frac > 0 ? c2(t) / nrm : NaN)));
    pp.setOmega(w);
    const q1 = D.decayPhasor({ tau: [st.tau1], f: [1] }, w), q2 = D.decayPhasor({ tau: [st.tau2], f: [1] }, w);
    const two = st.frac > 0 && st.frac < 1;
    chord.set(two ? [q1.g, q2.g] : [], two ? [q1.s, q2.s] : []);
    comps.set(two ? [q1.g, q2.g] : [], two ? [q1.s, q2.s] : [], ["var(--qb-c3)", "var(--qb-c5)"]);
    pt.set([P.g], [P.s]);
    const wt = (t: number) => (w * t).toFixed(2);
    ro.textContent =
      `ω = 2πf = ${w.toFixed(3)} rad/ns\n` +
      `ωτ₁ = ${wt(st.tau1)}` + (two ? `, ωτ₂ = ${wt(st.tau2)}` : "") + "\n" +
      `phase φ = ${(pm.phi * 180 / Math.PI).toFixed(1)}°, modulation m = ${pm.m.toFixed(3)}\n` +
      `τ_φ = tan φ / ω = ${pm.tauPhi.toFixed(2)} ns\n` +
      `τ_m = √(1/m² − 1) / ω = ${pm.tauM.toFixed(2)} ns\n` +
      `phasor (g, s) = (${P.g.toFixed(3)}, ${P.s.toFixed(3)})`;
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  draw();
}

autoMount(NAME, mount);
