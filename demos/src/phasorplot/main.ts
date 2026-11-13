// The phasor plot. Left: the phasor plane with the universal semicircle (lifetime ticks for
// the current frequency), up to three species that slide along the semicircle (drag them), the
// chord or triangle between them, and a draggable mixture point whose photon fractions follow
// the lever rule (barycentric coordinates for three). Middle: the mixture's expected histogram
// (log axis) through the IRF, with its components. Optional: a cloud of simulated pixels with
// N photons each. Calibration with a 4 ns reference removes the IRF's rotation and the shrink
// from coarse time bins; harmonic 2 uses 2ω.
//
// <div data-demo="phasorplot" data-preset="single|line|fractions|triangle|irf|cloud|bins"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { rng } from "../lib/random";
import * as D from "../lib/decay";
import * as Ph from "../lib/phasor";
import { injectStyle } from "../canny/imgview";
import { phasorPlane, handle, logMap, FLIM_CSS } from "./plane";

const NAME = "phasorplot";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { gap: .6em; flex-wrap: wrap; align-items: flex-start; }
${SEL} .pp-a { flex: 0 1 420px; min-width: 0; }
${SEL} .pp-b { flex: 0 1 280px; min-width: 0; }
${SEL} .pp-b .fl-ro { font-size: 13px; }
${SEL} .qb-panel { flex: 1 1 280px; max-width: 340px; gap: .25em; font-size: 15px; }
${SEL} .qb-panel .qb-btn { font-size: 13px; padding: .15em .5em; }
${SEL} .qb-panel .qb-presets { gap: .25em; }
.reveal ${SEL} { flex-wrap: nowrap; }
.reveal ${SEL} .pp-a { flex: 0 0 500px; }
.reveal ${SEL} .pp-b { flex: 0 0 330px; }
.reveal ${SEL} .pp-b .fl-ro { font-size: 16px; }
.reveal ${SEL} .qb-panel { flex: 1 1 280px; max-width: none; font-size: 17px; gap: .15em; }
.reveal ${SEL} .qb-panel .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal ${SEL} .qb-readout { font-size: 16px; }
`;

interface State { f: number; k: number; tau: number[]; w: number[]; N: number; irf: number; cal: boolean; bins: number; h: 1 | 2; cloud: boolean }
const DEFAULT: State = { f: 80, k: 2, tau: [0.5, 3, 8], w: [0.5, 0.5, 0], N: 1000, irf: 0.1, cal: true, bins: 256, h: 1, cloud: false };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "single", label: "single lifetimes live on the semicircle", s: { k: 1, tau: [2.5, 3, 8], w: [1, 0, 0] } },
  { id: "line", label: "two species: a straight line", s: { k: 2, tau: [0.5, 3, 8], w: [0.5, 0.5, 0] } },
  { id: "fractions", label: "photon fraction is not amplitude fraction", s: { k: 2, tau: [0.4, 4, 8], w: [0.0909, 0.9091, 0] } },
  { id: "triangle", label: "three species: a triangle", s: { k: 3, tau: [0.4, 2.5, 8], w: [0.3, 0.4, 0.3] } },
  { id: "irf", label: "uncalibrated IRF: everything rotates", s: { k: 2, cal: false, irf: 0.6, cloud: true } },
  { id: "cloud", label: "100 photons: a cloud", s: { k: 2, cloud: true, N: 100 } },
  { id: "bins", label: "coarse time bins distort the phasor", s: { k: 1, tau: [2.5, 3, 8], w: [1, 0, 0], bins: 4, cal: false } },
];
const COL = ["var(--qb-c3)", "var(--qb-c5)", "var(--qb-c4)"];
const BIN_CHOICES = [4, 8, 16, 32, 64, 128, 256];
const fM = logMap(10, 200), nM = logMap(30, 1e5);
const IRF_DELAY = 0.4, T_REF = 4, NPIX = 150;

function mount(host: HTMLElement): void {
  injectStyle("flim-common", FLIM_CSS);
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = structuredClone({ ...DEFAULT, ...(p0?.s ?? {}) });

  const colA = htmlEl("div", "pp-a fl-col", "", host);
  const pp = phasorPlane(colA, 1, { width: 520, height: 380 });
  const plot = pp.plot;
  const measured = plot.line([], [], { color: "var(--qb-c2)", width: 2, dash: "7 5" });
  const shape = plot.mark<[number[], number[]]>((g, xs, ys) => {
    if (xs.length < 2) return;
    const d = xs.map((x, i) => `${i ? "L" : "M"}${plot.x(x).toFixed(2)},${plot.y(ys[i]).toFixed(2)}`).join("") + (xs.length > 2 ? "Z" : "");
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.style.cssText = `fill:var(--qb-c1);fill-opacity:${xs.length > 2 ? 0.08 : 0};stroke:var(--qb-c1);stroke-width:2.5`;
    g.appendChild(p);
  }, [[], []]);
  const cloud = plot.scatter([], [], { color: "var(--qb-c1)", r: 3, opacity: 0.7 });
  const cloudMean = plot.scatter([], [], { color: "var(--qb-fg)", r: 5 });
  const hs = [0, 1, 2].map(i => handle(plot, COL[i], 11, (x, y) => dragSpecies(i, x, y), `τ${"₁₂₃"[i]}`));
  const mixH = handle(plot, "var(--qb-c2)", 12, (x, y) => dragMix(x, y), "mix");
  const capA = htmlEl("div", "fl-lab", "drag the species along the semicircle, and the mixture point", colA);
  void capA;

  const colB = htmlEl("div", "pp-b fl-col", "", host);
  const dec = new Plot(colB, { width: 360, height: 300, x: [0, 12.5], y: [1e-4, 0.1], ylog: true, xlabel: "time (ns)", ylabel: "fraction per bin", margin: { l: 92, r: 8, t: 10, b: 60 } });
  const comp = [0, 1, 2].map(i => dec.line([], [], { color: COL[i], width: 2, dash: "6 4" }));
  const sumL = dec.line([], [], { color: "var(--qb-c2)", width: 3 });
  const ro = htmlEl("div", "qb-readout fl-ro", "", colB);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, structuredClone(DEFAULT), structuredClone(p.s)); sync(); draw(); } })));
  htmlEl("div", "fl-lab", "number of species", panel);
  const kBtns = presets(panel, [1, 2, 3].map(k => ({ label: String(k), apply: () => { setK(k); draw(); } })));
  const sF = slider(panel, "frequency (MHz)", 0, 100, 0.5, fM.u(st.f), u => { st.f = fM.v(u); draw(); }, u => fM.v(u).toFixed(0));
  const sIrf = slider(panel, "IRF width σ (ns)", 0, 1, 0.02, st.irf, v => { st.irf = v; draw(); });
  const sBins = slider(panel, "time bins", 0, BIN_CHOICES.length - 1, 1, BIN_CHOICES.indexOf(st.bins), v => { st.bins = BIN_CHOICES[v]; draw(); }, v => String(BIN_CHOICES[v]));
  const sN = slider(panel, "photons per pixel", 0, 100, 0.5, nM.u(st.N), u => { st.N = Math.round(nM.v(u)); draw(); }, u => String(Math.round(nM.v(u))));
  const rowC = htmlEl("div", "qb-presets", "", panel);
  const cCal = checkbox(rowC, "calibrate", st.cal, v => { st.cal = v; draw(); });
  const cCloud = checkbox(rowC, "pixels", st.cloud, v => { st.cloud = v; draw(); });
  const cH = checkbox(rowC, "harmonic 2", st.h === 2, v => { st.h = v ? 2 : 1; draw(); });
  button(panel, "reset", () => { Object.assign(st, structuredClone(DEFAULT), structuredClone(p0?.s ?? {})); presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });

  const omega = () => D.omegaOf(st.f) * st.h;
  function setK(k: number): void {
    st.k = k;
    const w = st.w.slice(0, k);
    let s = w.reduce((a, b) => a + b, 0);
    if (!(s > 0)) { w.fill(1); s = k; }
    st.w = [0, 1, 2].map(i => (i < k ? w[i] / s : 0));
  }
  function dragSpecies(i: number, x: number, y: number): void {
    const th = Math.max(0.01, Math.min(Math.PI - 0.01, Math.atan2(Math.max(0, y), x - 0.5)));
    const q = { g: 0.5 + 0.5 * Math.cos(th), s: 0.5 * Math.sin(th) };
    st.tau[i] = Math.max(0.05, Math.min(30, q.s / (q.g * omega())));
    presetBtns.forEach(b => b.classList.remove("active"));
    draw();
  }
  function dragMix(x: number, y: number): void {
    const P = speciesPhasors().map(p => p.true);
    const pt = { g: x, s: y };
    if (st.k === 1) return;
    if (st.k === 2) { const t = Math.max(0, Math.min(1, Ph.fractionOnSegment(pt, P[0], P[1]))); st.w = [1 - t, t, 0]; }
    else {
      const b = Ph.barycentric3(pt, P[0], P[1], P[2]).map(v => Math.max(0, v));
      const s = b[0] + b[1] + b[2] || 1;
      st.w = b.map(v => v / s);
    }
    presetBtns.forEach(b => b.classList.remove("active"));
    draw();
  }

  /** Calibration factor from a noise-free 4 ns reference measured through the same IRF and bins. */
  function calib(ax: D.TimeAxis): Ph.Ph {
    const ref = Ph.phasor(D.decayBins({ tau: [T_REF], f: [1] }, { sigma: st.irf, delay: IRF_DELAY }, ax), st.h);
    return Ph.calibrationFactor(ref, T_REF, omega());
  }
  function measure(y: ArrayLike<number>, K: Ph.Ph): Ph.Ph {
    const p = Ph.phasor(y, st.h);
    return st.cal ? Ph.calibrate(p, K) : p;
  }
  function speciesPhasors(): { true: Ph.Ph; meas: Ph.Ph }[] {
    const ax = { T: 1000 / st.f, nBins: st.bins }, K = calib(ax);
    return st.tau.slice(0, st.k).map(t => ({ true: Ph.semicircle(omega() * t), meas: measure(D.decayBins({ tau: [t], f: [1] }, { sigma: st.irf, delay: IRF_DELAY }, ax), K) }));
  }

  function draw(): void {
    const w = omega(), T = 1000 / st.f, ax: D.TimeAxis = { T, nBins: st.bins }, K = calib(ax), irf = { sigma: st.irf, delay: IRF_DELAY };
    pp.setOmega(w);
    kBtns.forEach((b, i) => b.classList.toggle("active", i + 1 === st.k));
    const sp = speciesPhasors();
    // where single exponentials land without calibration
    if (!st.cal) {
      const gx: number[] = [], sy: number[] = [];
      for (let i = 0; i <= 120; i++) { const t = 0.02 * Math.pow(2000, i / 120); const q = measure(D.decayBins({ tau: [t], f: [1] }, irf, ax), K); gx.push(q.g); sy.push(q.s); }
      measured.set(gx, sy);
    } else measured.set([], []);
    // species: handles at the true points; the shape through the measured (= true when calibrated) points
    hs.forEach((h, i) => { h.el.style.display = i < st.k ? "" : "none"; if (i < st.k) h.set(sp[i].true.g, sp[i].true.s); });
    const shp = st.cal ? sp.map(p => p.true) : sp.map(p => p.meas);
    shape.set(shp.map(p => p.g), shp.map(p => p.s));
    const f = st.w.slice(0, st.k);
    const dc: D.Decay = { tau: st.tau.slice(0, st.k), f };
    const mixTrue = D.decayPhasor({ tau: dc.tau.map(t => t * st.h), f }, D.omegaOf(st.f));
    const y = D.decayBins(dc, irf, ax), mixMeas = measure(y, K);
    const mp = st.cal ? mixTrue : mixMeas;
    mixH.el.style.display = st.k > 1 ? "" : "none";
    mixH.set(mp.g, mp.s);
    // decay plot: the expected histogram (fractions per bin), its components
    dec.setDomain([0, T], [1e-4, 1]);
    const xs = D.binCentres(ax);
    let ymin = 1;
    comp.forEach((c, i) => {
      if (i >= st.k || st.k === 1) { c.set([], []); return; }
      const yi = D.decayBins({ tau: [st.tau[i]], f: [1] }, irf, ax).map(v => v * f[i]);
      c.set(xs, yi);
    });
    for (const v of y) if (v > 0) ymin = Math.min(ymin, v);
    const ymax = Math.max(...y);
    dec.setDomain([0, T], [Math.max(ymin * 0.5, ymax * 1e-4), ymax * 2]);
    sumL.set(xs, y);
    // cloud of pixels
    let cg: number[] = [], cs: number[] = [];
    if (st.cloud) {
      const r = rng(21);
      for (let p = 0; p < NPIX; p++) { const q = measure(D.poissonHistogram(r, y.map(v => v * st.N)), K); cg.push(q.g); cs.push(q.s); }
    }
    cloud.set(cg, cs);
    const mg = cg.length ? cg.reduce((a, b) => a + b, 0) / cg.length : NaN, ms = cs.length ? cs.reduce((a, b) => a + b, 0) / cs.length : NaN;
    cloudMean.set(st.cloud ? [mg] : [], st.cloud ? [ms] : []);
    let sd = NaN;
    if (cg.length) { let v = 0; for (let i = 0; i < cg.length; i++) v += (cg[i] - mg) ** 2 + (cs[i] - ms) ** 2; sd = Math.sqrt(v / (2 * (cg.length - 1))); }
    // readout
    const alpha = D.amplitudeFractions(f, dc.tau);
    const lt = Ph.lifetimeFromPhasor(mp, w);
    const lines = [`ω = ${w.toFixed(3)} rad/ns` + (st.h === 2 ? " (2ω)" : "") + "   τ in ns"];
    dc.tau.forEach((t, i) => lines.push(`τ${"₁₂₃"[i]} ${t.toFixed(2)} (${sp[i].true.g.toFixed(2)}, ${sp[i].true.s.toFixed(2)})` + (st.k > 1 ? ` f ${f[i].toFixed(2)} α ${alpha[i].toFixed(2)}` : "")));
    lines.push(`${st.cal ? "calibrated" : "measured"}: (${mp.g.toFixed(3)}, ${mp.s.toFixed(3)})`);
    lines.push(`τ_φ = ${lt.tauPhi.toFixed(2)} ns, τ_m = ${lt.tauM.toFixed(2)} ns`);
    if (!st.cal) { const k = Math.hypot(K.g, K.s); lines.push(`IRF + bins: ×${(1 / k).toFixed(3)}, turned ${(-Math.atan2(K.s, K.g) * 180 / Math.PI).toFixed(1)}°`); }
    if (st.cloud) lines.push(`${NPIX} pixels, spread ${sd.toFixed(3)}\n  (½/√N = ${(0.5 / Math.sqrt(st.N)).toFixed(3)})`);
    ro.textContent = lines.join("\n");
  }

  function sync(): void {
    setK(st.k);
    sF.set(fM.u(st.f)); sIrf.set(st.irf); sBins.set(BIN_CHOICES.indexOf(st.bins)); sN.set(nM.u(st.N));
    cCal.checked = st.cal; cCloud.checked = st.cloud; cH.checked = st.h === 2;
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  draw();
}

autoMount(NAME, mount);
