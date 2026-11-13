// Time-correlated single-photon counting. Top: the last few laser pulses on a time line, with
// the photons detected after them (at most one recorded per pulse). Bottom: the histogram of
// arrival times (linear or log), with the true decay, the IRF and the expected histogram
// (decay convolved with the IRF, plus background). Two adjacent gates (drag on the histogram)
// give the two-gate estimate; the readout also gives the mean arrival time, raw and corrected
// by the IRF's mean, the fraction wrapped round from earlier pulses and the pile-up loss.
//
// <div data-demo="tcspc" data-preset="slow|log|irf|wrap|pileup|background|few"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { rng } from "../lib/random";
import * as D from "../lib/decay";
import { injectStyle } from "../canny/imgview";
import { logMap, fmtCount, fmtShort, FLIM_CSS } from "../phasorplot/plane";

const NAME = "tcspc";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { gap: .8em; flex-wrap: nowrap; align-items: flex-start; }
${SEL} .tc-plots { flex: 1 1 680px; min-width: 0; max-width: 760px; }
${SEL} .qb-panel { flex: 0 1 340px; gap: .3em; }
${SEL} .qb-readout { font-size: 15px; }
.reveal ${SEL} .tc-plots { flex: 0 0 700px; }
.reveal ${SEL} .qb-panel { flex: 1 1 400px; gap: .15em; font-size: 17px; }
.reveal ${SEL} .qb-readout { font-size: 16px; padding: .2em .45em; line-height: 1.25; }
.reveal ${SEL} .qb-presets { gap: .2em; }
@media (max-width: 760px) { ${SEL} { flex-wrap: wrap; } }
`;

interface State { tau: number; irfSigma: number; irfDelay: number; bg: number; T: number; rate: number; N: number; log: boolean; gateA: number; gateW: number }
const DEFAULT: State = { tau: 2.5, irfSigma: 0.1, irfDelay: 1, bg: 0, T: 12.5, rate: 1, N: 20000, log: false, gateA: 1.5, gateW: 2 };
const PRESETS: { id: string; label: string; s: Partial<State>; slow?: boolean }[] = [
  { id: "slow", label: "one photon at a time", s: { N: 0, rate: 5 }, slow: true },
  { id: "log", label: "the log plot is a straight line", s: { N: 100000, log: true } },
  { id: "irf", label: "the IRF blurs the start", s: { N: 100000, log: true, tau: 0.8, irfSigma: 0.4, gateA: 2.5, gateW: 1 } },
  { id: "wrap", label: "τ = 8 ns at 80 MHz: the tail wraps around", s: { N: 100000, log: true, tau: 8, gateA: 2, gateW: 3 } },
  { id: "pileup", label: "too bright: pile-up shortens τ", s: { N: 100000, log: true, rate: 60 } },
  { id: "background", label: "background flattens the tail", s: { N: 100000, log: true, bg: 15 } },
  { id: "few", label: "100 photons: what can we say?", s: { N: 100 } },
];
const NB = 128, SHOW = 6;
const tauM = logMap(0.2, 8), nM = logMap(10, 1e6), rateM = logMap(0.1, 90);

function mount(host: HTMLElement): void {
  injectStyle("flim-common", FLIM_CSS);
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const plots = htmlEl("div", "tc-plots fl-col", "", host);
  const line = new Plot(plots, { width: 700, height: 118, x: [0, SHOW], y: [0, 1], xticks: [], yticks: [], grid: false, margin: { l: 14, r: 14, t: 26, b: 22 }, ariaLabel: "laser pulses and detected photons" });
  const hist = new Plot(plots, { width: 700, height: 400, x: [0, st.T], y: [0, 1], xlabel: "arrival time after the pulse (ns)", ylabel: "photons per bin", margin: { l: 86, r: 14, t: 12, b: 62 } });
  // time line: pulses as bars, detected photons as dots; x in units of pulses
  const pulseMark = line.mark<[number[], number[]]>((g, pulses, photons) => {
    for (let k = 0; k <= SHOW; k++) {
      const x = line.x(k);
      const l = svgEl("line", { x1: x, x2: x, y1: line.y(0.05), y2: line.y(0.95) }, g);
      l.style.cssText = "stroke:var(--qb-c4);stroke-width:4";
    }
    photons.forEach((t, i) => {
      if (!Number.isFinite(t)) return;
      const c = svgEl("circle", { cx: line.x(pulses[i] + t), cy: line.y(0.45), r: 8 }, g);
      c.style.cssText = "fill:var(--qb-c1);stroke:var(--qb-bg);stroke-width:1.5";
    });
  }, [[], []]);
  void pulseMark;
  const lineTxt = line.text(0.02, 1.12, "", { size: 17 });

  const gate1 = hist.band([], [], [], { color: "var(--qb-c3)", fillOpacity: 0.16 });
  const gate2 = hist.band([], [], [], { color: "var(--qb-c5)", fillOpacity: 0.16 });
  const dataMark = hist.histogram([], [], { color: "var(--qb-c1)", fillOpacity: 0.35 });
  const irfMark = hist.line([], [], { color: "var(--qb-c4)", width: 2.5, dash: "6 4", label: "IRF (scaled)" });
  const trueMark = hist.line([], [], { color: "var(--qb-muted)", width: 2.5, label: "decay without IRF" });
  const modelMark = hist.line([], [], { color: "var(--qb-c2)", width: 3, label: "expected: decay ∗ IRF + background" });
  const gTxt1 = hist.text(0, 1, "D1", { size: 18, color: "var(--qb-c3)", anchor: "middle" });
  const gTxt2 = hist.text(0, 1, "D2", { size: 18, color: "var(--qb-c5)", anchor: "middle" });
  hist.svg.style.cursor = "ew-resize"; hist.svg.style.touchAction = "none";
  let dragging = false;
  const setGate = (ev: PointerEvent) => { const [x] = hist.toData(ev); st.gateA = Math.max(0, Math.min(st.T - 2 * st.gateW, x - st.gateW)); drawHist(); };
  hist.svg.addEventListener("pointerdown", ev => { dragging = true; hist.svg.setPointerCapture(ev.pointerId); setGate(ev); });
  hist.svg.addEventListener("pointermove", ev => { if (dragging) setGate(ev); });
  hist.svg.addEventListener("pointerup", () => { dragging = false; });
  hist.svg.addEventListener("pointercancel", () => { dragging = false; });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  const sTau = slider(panel, "τ (ns)", 0, 100, 0.5, tauM.u(st.tau), u => { st.tau = +tauM.v(u).toFixed(2); resim(); }, u => tauM.v(u).toFixed(2));
  const sIrf = slider(panel, "IRF width σ (ns)", 0, 0.6, 0.02, st.irfSigma, v => { st.irfSigma = v; resim(); });
  const sDel = slider(panel, "IRF delay (ns)", 0, 4, 0.1, st.irfDelay, v => { st.irfDelay = v; resim(); });
  const sBg = slider(panel, "background (%)", 0, 40, 1, st.bg, v => { st.bg = v; resim(); });
  const sT = slider(panel, "pulse period (ns)", 12.5, 100, 0.5, st.T, v => { st.T = v; st.gateA = Math.min(st.gateA, v / 2); resim(); });
  const sRate = slider(panel, "count rate (% of pulses)", 0, 100, 0.5, rateM.u(st.rate), u => { st.rate = +rateM.v(u).toPrecision(2); resim(); }, u => rateM.v(u).toPrecision(2));
  const sN = slider(panel, "photons N", 0, 100, 0.5, nM.u(Math.max(10, st.N)), u => { st.N = Math.round(nM.v(u)); resim(); }, u => fmtShort(nM.v(u)));
  const sW = slider(panel, "gate width (ns)", 0.2, 6, 0.1, st.gateW, v => { st.gateW = v; drawHist(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const logC = checkbox(row, "log axis", st.log, v => { st.log = v; drawHist(); });
  const playB = button(row, "play", () => { playing = !playing; playB.textContent = playing ? "pause" : "play"; if (playing) loop(); });
  const sSpeed = slider(panel, "speed (pulses per frame)", 0, 100, 1, 0, () => { /* read on use */ }, u => fmtShort(logMap(1, 1e5).v(u)));
  button(row, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); presetBtns.forEach(b => b.classList.remove("active")); playing = false; playB.textContent = "play"; sync(); resim(); });
  const ro = htmlEl("div", "qb-readout fl-ro", "", panel);

  let r = rng(7), counts = new Float64Array(NB), nPhot = 0, nPulses = 0, nMulti = 0;
  let recent: { pulse: number; t: number }[] = [];
  let playing = false;
  const ax = () => ({ T: st.T, nBins: NB });
  const opts = (): D.TcspcOptions => ({ tau: [st.tau], f: [1], T: st.T, irfSigma: st.irfSigma, irfDelay: st.irfDelay, bgFraction: st.bg / 100, rate: st.rate / 100 });

  function addPhotons(k: number): void {
    const o = opts(), d = st.T / NB;
    for (let i = 0; i < k; i++) {
      const gap = D.pulsesUntilDetection(r, o.rate);
      nPulses += gap;
      const ev = D.detectPulse(r, o);
      if (ev.k > 1) nMulti++;
      counts[Math.min(NB - 1, Math.floor(ev.t / d))]++;
      nPhot++;
      if (k < 50) recent.push({ pulse: nPulses, t: ev.t });
    }
    if (k >= 50) recent = [];
    recent = recent.filter(e => e.pulse > nPulses - SHOW);
  }

  function resim(): void {
    r = rng(7); counts = new Float64Array(NB); nPhot = 0; nPulses = 0; nMulti = 0; recent = [];
    addPhotons(st.N);
    hist.setDomain([0, st.T]);
    draw();
  }

  // play: advance `speed` pulses per frame
  let pending = 0;
  function loop(): void {
    if (!playing || !host.isConnected) return;
    const speed = logMap(1, 1e5).v(sSpeed.get());
    pending += speed;
    const o = opts();
    while (pending >= 1) {
      // pulses until the next detection, consumed as the frame budget allows
      if (nextGap <= 0) nextGap = D.pulsesUntilDetection(r, o.rate);
      if (nextGap > pending) { nextGap -= Math.floor(pending); nPulses += Math.floor(pending); pending -= Math.floor(pending); break; }
      pending -= nextGap; nPulses += nextGap; nextGap = 0;
      const ev = D.detectPulse(r, o);
      if (ev.k > 1) nMulti++;
      counts[Math.min(NB - 1, Math.floor(ev.t / (st.T / NB)))]++;
      nPhot++;
      recent.push({ pulse: nPulses, t: ev.t });
    }
    recent = recent.filter(e => e.pulse > nPulses - SHOW);
    draw();
    requestAnimationFrame(loop);
  }
  let nextGap = 0;

  function drawLine(): void {
    const base = Math.max(0, nPulses - SHOW);
    pulseMark.set(recent.map(e => e.pulse - 1 - base), recent.map(e => e.t / st.T));
    lineTxt.set(0.02, 1.12, `laser pulses (every ${st.T} ns) and detected photons — pulse no. ${fmtCount(nPulses)}`);
  }

  function drawHist(): void {
    const a = ax(), d = st.T / NB;
    const xs = D.binCentres(a), edges = Float64Array.from({ length: NB + 1 }, (_, j) => j * d);
    const sig = nPhot * (1 - st.bg / 100);
    const mu = D.expectedCounts({ tau: [st.tau], f: [1] }, { sigma: st.irfSigma, delay: st.irfDelay }, a, Math.max(sig, 1e-9), nPhot * st.bg / 100 / NB);
    const tr = D.expectedCounts({ tau: [st.tau], f: [1] }, { sigma: 0, delay: st.irfDelay }, a, Math.max(sig, 1e-9), 0);
    const irf = D.gaussianIRF(st.irfSigma, st.irfDelay, a);
    let ymax = 1;
    for (let j = 0; j < NB; j++) ymax = Math.max(ymax, counts[j], mu[j]);
    const irfMax = Math.max(...irf), irfS = Float64Array.from(irf, v => v / irfMax * ymax * 0.9);
    let ylo = 0.5;
    if (st.log) { let mn = Infinity; for (let j = 0; j < NB; j++) if (mu[j] > 0) mn = Math.min(mn, mu[j]); ylo = Math.max(0.3, Math.min(mn * 0.5, 1)); }
    hist.opts.ylog = st.log;
    hist.setDomain([0, st.T], st.log ? [ylo, ymax * 2] : [0, ymax * 1.12]);
    dataMark.set(edges, counts);
    irfMark.set(xs, irfS);
    trueMark.set(xs, tr);
    modelMark.set(xs, mu);
    const tg = D.twoGateTau(counts, a, st.gateA, st.gateW);
    const g0 = tg.a, g1 = g0 + tg.w, g2 = g1 + tg.w;
    const lo = st.log ? ylo : 0, hi = st.log ? ymax * 2 : ymax * 1.12;
    gate1.set([g0, g1], [lo, lo], [hi, hi]);
    gate2.set([g1, g2], [lo, lo], [hi, hi]);
    const ty = st.log ? ylo * 2.2 : ymax * 0.04;
    gTxt1.set((g0 + g1) / 2, ty, "D1"); gTxt2.set((g1 + g2) / 2, ty, "D2");
    hist.legend(undefined, "tr");
    // readout
    const m = D.meanArrival(counts, a), mIrf = D.meanArrival(irf, a);
    const f = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");
    ro.textContent =
      `${fmtCount(nPhot)} photons, ${fmtCount(nPulses)} pulses\n` +
      `true τ = ${st.tau.toFixed(2)} ns\n` +
      `mean arrival ${f(m)} − IRF ${f(mIrf)}\n   = ${f(m - mIrf)} ns\n` +
      `gates ${tg.d1} / ${tg.d2} → τ = ${f(tg.tau)} ns\n` +
      `left at next pulse: ${(100 * D.wrappedFraction(st.tau, st.T)).toFixed(1)} %\n` +
      `pulses with ≥ 2 photons: ${nPhot ? (100 * nMulti / nPhot).toFixed(1) : "0"} %`;
  }

  function draw(): void { drawLine(); drawHist(); }

  function sync(): void {
    sTau.set(tauM.u(st.tau)); sIrf.set(st.irfSigma); sDel.set(st.irfDelay); sBg.set(st.bg); sT.set(st.T);
    sRate.set(rateM.u(st.rate)); sN.set(nM.u(Math.max(10, st.N))); sW.set(st.gateW); logC.checked = st.log;
  }

  function applyPreset(p: typeof PRESETS[number]): void {
    Object.assign(st, DEFAULT, p.s);
    sync();
    resim();
    if (p.slow) { sSpeed.set(0); playing = true; playB.textContent = "pause"; loop(); }
    else { playing = false; playB.textContent = "play"; }
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  resim();
  if (p0?.slow) sSpeed.set(0);
}

autoMount(NAME, mount);
