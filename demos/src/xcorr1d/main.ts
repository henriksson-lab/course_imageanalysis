// Sliding and scoring in 1D (FFT registration day). Image B (red) is placed at a shift t on
// image A (blue); the pixels on top of each other are compared by SSD, cross-correlation (CC)
// or normalised cross-correlation (NCC), and the score curve is drawn as B slides.
//   data-mode="direct" (default): the sliding picture, three scores, multiply-add counter.
//   data-mode="fft": the same correlation via the DFT; the FFT curve on top of the brute-force
//     dots, circular vs zero-padded (linear), each frequency's "vote" as an arrow, cost plot.
// Options: data-preset="<id>", data-mode. Convention: B(x) ≈ A(x + t), c(t) = Σ A(x+t) B(x).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, Slider } from "../lib/ui";
import { Plot, seriesColor, plotFigure } from "../lib/plot";
import { loadCell } from "../lib/cell";
import { bruteForce1d, correlate1d, signed, fftFlops } from "../lib/registration";
import { simulate1d, Sim1D, Kind, blobCentre } from "./signals";

const NAME = "xcorr1d";
const CREDIT = "Profile: row 128 of a transmission electron micrograph (Robert M. Hunt, CC BY 3.0)";

const STYLE = `
.qb-demo[data-demo="${NAME}"] .xc-plots { flex: 1 1 560px; min-width: 0; display: grid; gap: .3em .8em; }
.qb-demo[data-demo="${NAME}"].xc-direct .xc-plots { grid-template-columns: 1fr; max-width: 680px; }
.qb-demo[data-demo="${NAME}"].xc-fft .xc-plots { grid-template-columns: 1fr 1fr; max-width: 860px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 270px; max-width: 330px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .xc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .xc-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.reveal .qb-demo[data-demo="${NAME}"].xc-direct .xc-plots { max-width: 700px; }
.reveal .slides .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; line-height: 1.15; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type ScoreSel = "ssd" | "cc" | "ncc" | "all";
interface State { sim: Sim1D; cur: number; score: ScoreSel; minOv: number; pad: boolean; meanFree: boolean; phase: boolean }
interface Preset { id: string; label: string; st: Partial<Omit<State, "sim">> & { sim: Partial<Sim1D> } }

const BASE: Sim1D = { kind: "bumps", N: 64, L: 64, t: 9, gain: 1, offset: 0, ramp: 0, noise: 0, blob: 0, seed: 3 };

const DIRECT: Preset[] = [
  { id: "bump", label: "a matching bump", st: { sim: { L: 16, t: 22 }, cur: -4, score: "all", minOv: 8 } },
  { id: "bright", label: "brighter copy: SSD fails", st: { sim: { gain: 1.6, offset: 40 }, cur: 9, score: "ssd", minOv: 16 } },
  { id: "blob", label: "bright blob: CC fails", st: { sim: { L: 16, t: 12, blob: 150 }, cur: 12, score: "cc", minOv: 8 } },
  { id: "periodic", label: "periodic: many peaks", st: { sim: { kind: "periodic" }, cur: 9, score: "ncc", minOv: 16 } },
  { id: "tiny", label: "tiny overlap: NCC noisy", st: { sim: { noise: 15 }, cur: 9, score: "ncc", minOv: 2 } },
  { id: "profile", label: "micrograph profile", st: { sim: { kind: "profile", t: 9, L: 48 }, cur: 9, score: "all", minOv: 16 } },
  { id: "hand", label: "by hand: (1,3,1)", st: { sim: { kind: "hand", N: 8, L: 3, t: 1 }, cur: 0, score: "all", minOv: 3 } },
];
const FFT: Preset[] = [
  { id: "agree", label: "the curves agree", st: { sim: { N: 32, L: 32, t: 5, seed: 3 }, cur: 0, pad: false } },
  { id: "wrap", label: "wrap-around fools it", st: { sim: { N: 32, L: 32, t: 9, seed: 1 }, cur: -3, pad: false } },
  { id: "pad", label: "padding fixes it", st: { sim: { N: 32, L: 32, t: 9, seed: 1 }, cur: 9, pad: true } },
  { id: "waves", label: "the waves line up", st: { sim: { N: 16, L: 16, t: 3, seed: 1, circular: true }, cur: 3, pad: false } },
  { id: "hand4", label: "by hand: length 4", st: { sim: { kind: "hand", N: 4, L: 4, t: 1 }, cur: 1, pad: false, meanFree: false } },
];

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(d)) : "–");

function mount(host: HTMLElement): void {
  injectStyle();
  const mode = host.dataset.mode === "fft" ? "fft" : "direct";
  host.classList.add(mode === "fft" ? "xc-fft" : "xc-direct");
  const ALL = mode === "fft" ? FFT : DIRECT, ids = host.dataset.presets?.split(",").map(x => x.trim());
  const PRESETS = ids ? ALL.filter(p => ids.includes(p.id)) : ALL;
  const p0 = ALL.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { sim: { ...BASE }, cur: 0, score: "all", minOv: 8, pad: false, meanFree: true, phase: false };
  let row: Float64Array | null = null;

  const plots = htmlEl("div", "xc-plots", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  // ---- controls ----
  htmlEl("div", "xc-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "xc-lab", "signal", panel);
  const kinds: [Kind, string][] = [["bumps", "bumps"], ["profile", "micrograph"], ["step", "step"], ["periodic", "periodic"]];
  const kindBtns = presets(panel, kinds.map(([k, l]) => ({ label: l, apply: () => { st.sim.kind = k; if (st.sim.N < 16) { st.sim.N = mode === "fft" ? 32 : 64; st.sim.L = mode === "fft" ? 32 : 48; } update(true); } })));
  let scoreBtns: HTMLButtonElement[] = [], nBtns: HTMLButtonElement[] = [];
  if (mode === "direct") {
    htmlEl("div", "xc-lab", "score", panel);
    const sels: [ScoreSel, string][] = [["ssd", "SSD"], ["cc", "CC"], ["ncc", "NCC"], ["all", "all three"]];
    scoreBtns = presets(panel, sels.map(([s, l]) => ({ label: l, apply: () => { st.score = s; update(false); } })));
  } else {
    htmlEl("div", "xc-lab", "length N", panel);
    nBtns = presets(panel, [8, 16, 32, 64].map(n => ({ label: String(n), apply: () => { if (st.sim.kind === "hand") st.sim.kind = "bumps"; st.sim.N = n; st.sim.L = n; st.sim.t = Math.min(st.sim.t, n / 2 - 1); update(true); } })));
  }
  const curS = slider(panel, "shift t", -64, 64, 1, 0, v => { st.cur = v; update(false); });
  const trueS = slider(panel, "true shift", -32, 32, 1, 9, v => { st.sim.t = v; update(true); });
  let gainS: Slider | null = null, offS: Slider | null = null, rampS: Slider | null = null, lenS: Slider | null = null, ovS: Slider | null = null;
  if (mode === "direct") {
    gainS = slider(panel, "gain on B", 0.5, 2, 0.1, 1, v => { st.sim.gain = v; update(true); });
    offS = slider(panel, "offset on B", -60, 60, 5, 0, v => { st.sim.offset = v; update(true); });
    rampS = slider(panel, "ramp on B", -100, 100, 10, 0, v => { st.sim.ramp = v; update(true); });
  }
  const noiseS = slider(panel, "noise", 0, 40, 1, 0, v => { st.sim.noise = v; update(true); });
  if (mode === "direct") {
    lenS = slider(panel, "length of B", 4, 64, 1, 64, v => { st.sim.L = v; update(true); });
    ovS = slider(panel, "min. overlap", 2, 32, 1, 8, v => { st.minOv = v; update(false); });
  }
  let padC: HTMLInputElement | null = null, mfC: HTMLInputElement | null = null, phC: HTMLInputElement | null = null, circC: HTMLInputElement | null = null;
  if (mode === "fft") {
    padC = checkbox(panel, "zero-pad to 2N (linear)", false, v => { st.pad = v; update(false); });
    mfC = checkbox(panel, "subtract the means", true, v => { st.meanFree = v; update(false); });
    phC = checkbox(panel, "phase only (|·| = 1)", false, v => { st.phase = v; update(false); });
    circC = checkbox(panel, "B = A rolled around (circular copy)", false, v => { st.sim.circular = v; update(true); });
  }
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  hideControls(panel, host.dataset.controls);
  const credit = htmlEl("div", "xc-credit", CREDIT, panel);
  htmlEl("div", "qb-note", mode === "direct"
    ? "SSD and CC are averaged over the overlap, so shifts with different overlaps compare fairly; the readout gives the sums."
    : "Dots: brute force (every shift, directly). Line: inverse DFT of A·conj(B). Arrows: each frequency's contribution at the current shift t.", panel);

  // ---- plots ----
  const W = mode === "direct" ? 660 : 430, Hh = mode === "direct" ? 230 : 250;
  const f1 = plotFigure(plots, mode === "direct" ? "A (blue) and B (red) placed at shift t" : "A (blue) and B (red)");
  const sigP = uniqueClip(new Plot(f1, { width: W, height: Hh, x: [0, 64], y: [0, 255], xlabel: "x", ylabel: "grey level" }));
  const f2 = plotFigure(plots, mode === "direct" ? "score of each shift (drawn as B slides)" : "correlation c(t): brute force (dots) and FFT (line)");
  const scoreP = uniqueClip(new Plot(f2, { width: W, height: Hh, x: [-63, 63], y: [0, 1], xlabel: "shift t", ylabel: "score" }));
  let waveSvg: SVGSVGElement | null = null, costP: Plot | null = null;
  if (mode === "fft") {
    const f3 = plotFigure(plots, "each frequency's vote at shift t");
    waveSvg = svgEl("svg", { viewBox: "0 0 430 250", width: "100%", class: "qb-plot-svg" }, f3);
    const f4 = plotFigure(plots, "operations for two n × n images");
    costP = uniqueClip(new Plot(f4, { width: W, height: Hh, x: [16, 4096], y: [1e3, 1e15], xlog: true, ylog: true, xlabel: "image side n", yticks: [1e3, 1e6, 1e9, 1e12, 1e15], margin: { l: 70 } }));
    costP.fn(n => 2 * n ** 4, { color: seriesColor(1), label: "all shifts, directly" }, 60);
    costP.fn(n => 2 * 65 * 65 * n * n, { color: seriesColor(3), label: "±32 window, directly" }, 60);
    costP.fn(n => 3 * fftFlops(4 * n * n), { color: seriesColor(2), label: "FFT (padded)" }, 60);
    costP.vline(1024, { text: "1024", width: 2 });
    costP.legend(undefined, "tl");
  }

  const sigMarks = {
    band: sigP.band([0, 1], [0, 0], [0, 0], { color: "var(--qb-c4)", fillOpacity: 0.18 }),
    a: sigP.line([], [], { color: seriesColor(0), width: 3 }),
    b: sigP.line([], [], { color: seriesColor(1), width: 3 }),
    blob: sigP.text(0, 0, "", { anchor: "middle", size: 16 }),
  };
  const scoreMarks: ReturnType<Plot["line"]>[] = [];
  const extra: { remove(): void }[] = [];

  function load(p: Preset): void {
    st.sim = { ...BASE, ...(mode === "fft" ? { N: 32, L: 32 } : {}), ...p.st.sim };
    st.cur = p.st.cur ?? 0; st.score = p.st.score ?? "all"; st.minOv = p.st.minOv ?? 8;
    st.pad = p.st.pad ?? false; st.meanFree = p.st.meanFree ?? true; st.phase = p.st.phase ?? false;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    update(true);
  }

  let sig = simulate1d(st.sim);
  function update(regen: boolean): void {
    if (st.sim.kind === "profile" && !row) {
      loadCell().then(g => { row = Float64Array.from(g.data.subarray(128 * 256, 129 * 256)); update(true); });
      return;
    }
    if (regen) sig = simulate1d(st.sim, row);
    syncControls();
    if (mode === "direct") drawDirect(); else drawFFT();
  }

  function syncControls(): void {
    const s = st.sim, hand = s.kind === "hand";
    kindBtns.forEach((b, i) => b.classList.toggle("active", kinds[i][0] === s.kind));
    scoreBtns.forEach((b, i) => b.classList.toggle("active", ["ssd", "cc", "ncc", "all"][i] === st.score));
    nBtns.forEach((b, i) => b.classList.toggle("active", !hand && [8, 16, 32, 64][i] === s.N));
    const [lo, hi] = shiftRange();
    curS.input.min = String(lo); curS.input.max = String(hi);
    st.cur = Math.max(lo, Math.min(hi, st.cur));
    curS.set(st.cur);
    trueS.input.min = String(mode === "fft" ? -s.N / 2 + 1 : -s.L + 2); trueS.input.max = String(mode === "fft" ? s.N / 2 - 1 : s.N - 2);
    trueS.set(s.t);
    gainS?.set(s.gain); offS?.set(s.offset); rampS?.set(s.ramp); noiseS.set(s.noise);
    if (lenS) { lenS.set(s.L); lenS.input.disabled = hand; }
    if (ovS) { ovS.input.max = String(Math.max(2, Math.min(s.L, s.N))); ovS.set(st.minOv); }
    [trueS, gainS, offS, rampS, noiseS].forEach(x => { if (x) x.input.disabled = hand; });
    if (padC) padC.checked = st.pad;
    if (mfC) mfC.checked = st.meanFree;
    if (phC) phC.checked = st.phase;
    if (circC) circC.checked = !!st.sim.circular;
    credit.style.visibility = s.kind === "profile" ? "visible" : "hidden";
  }

  function shiftRange(): [number, number] {
    const s = st.sim;
    if (mode === "fft") return st.pad ? [-s.N + 1, s.N - 1] : [-s.N / 2 + 1, s.N / 2];
    return [-(s.L - st.minOv), s.N - st.minOv];
  }

  // ---------------- direct mode ----------------
  function drawDirect(): void {
    const { f, g } = sig, s = st.sim, N = f.length, L = g.length;
    const [lo, hi] = shiftRange(), t = st.cur;
    const bf = bruteForce1d(f, g, { tMin: lo, tMax: hi });
    extra.forEach(m => m.remove()); extra.length = 0;
    // signals
    const all = [...f, ...g];
    let ymin = Math.min(...all), ymax = Math.max(...all);
    const pad = 0.08 * (ymax - ymin || 1); ymin -= pad; ymax += pad;
    sigP.setDomain([Math.min(0, lo) - 0.5, Math.max(N, hi + L) - 0.5], [ymin, ymax]);
    const xa = Array.from(f, (_, i) => i), xb = Array.from(g, (_, i) => i + t);
    const o0 = Math.max(0, t) - 0.5, o1 = Math.min(N, t + L) - 0.5;
    sigMarks.band.set(o1 > o0 ? [o0, o1] : [], [ymin, ymin], [ymax, ymax]);
    sigMarks.a = replaceStep(sigMarks.a, xa, f, seriesColor(0), s.kind === "hand");
    sigMarks.b = replaceStep(sigMarks.b, xb, g, seriesColor(1), s.kind === "hand");
    sigMarks.blob.set(s.blob ? blobCentre(s) : -1e9, ymax - 0.08 * (ymax - ymin), s.blob ? "bright blob (only in A)" : "");
    // scores (per overlap pixel for SSD and CC)
    const n = bf.n, ssd = bf.ssd.map((v, k) => v / n[k]), cc = bf.cc.map((v, k) => v / n[k]), ncc = bf.ncc;
    const k0 = t - lo;
    let ops = 0;
    for (let k = 0; k <= k0; k++) ops += n[k];
    const best = (a: Float64Array, max: boolean) => { let b = -1; a.forEach((v, k) => { if (!Number.isFinite(v)) return; if (b < 0 || (max ? v > a[b] : v < a[b])) b = k; }); return b; };
    const bS = best(ssd, false), bC = best(cc, true), bN = best(ncc, true);
    scoreMarks.forEach(m => m.remove()); scoreMarks.length = 0;
    const ts = bf.t;
    const series: [string, Float64Array, string, number][] = [];
    if (st.score === "ssd" || st.score === "all") series.push(["SSD (low = good)", ssd, seriesColor(1), bS]);
    if (st.score === "cc" || st.score === "all") series.push(["CC", cc, seriesColor(2), bC]);
    if (st.score === "ncc" || st.score === "all") series.push(["NCC", ncc, seriesColor(0), bN]);
    let ylo = Infinity, yhi = -Infinity;
    const norm = st.score === "all";
    const shown = series.map(([lab, v, col, b]) => {
      let mn = Infinity, mx = -Infinity;
      v.forEach(x => { if (Number.isFinite(x)) { mn = Math.min(mn, x); mx = Math.max(mx, x); } });
      const y = norm ? v.map(x => (x - mn) / ((mx - mn) || 1)) : v;
      y.forEach(x => { if (Number.isFinite(x)) { ylo = Math.min(ylo, x); yhi = Math.max(yhi, x); } });
      return { lab, y, col, b };
    });
    if (!(yhi > ylo)) { ylo = 0; yhi = 1; }
    const yp = 0.06 * (yhi - ylo);
    scoreP.setDomain([lo - 0.5, hi + 0.5], [ylo - yp, yhi + yp]);
    scoreP.setLabels({ ylabel: norm ? "score (scaled)" : series[0][0].split(" ")[0] + " per pixel" });
    for (const sh of shown) {
      scoreMarks.push(scoreP.line(ts, sh.y, { color: sh.col, width: 1.5, opacity: 0.3 }));
      const yy = sh.y.map((v, k) => (k <= k0 ? v : NaN));
      scoreMarks.push(scoreP.line(ts, yy, { color: sh.col, width: 3.5, label: sh.lab }));
      extra.push(scoreP.scatter([ts[sh.b]], [sh.y[sh.b]], { color: sh.col, r: 8 }));
    }
    extra.push(scoreP.vline(t, { text: `t = ${t}`, width: 2 }));
    if (s.kind !== "hand") extra.push(scoreP.vline(s.t, { color: "var(--qb-c3)", dash: "2 5", width: 2 }));
    scoreP.legend(undefined, "tr");
    const k = k0, sc = (v: number) => fmt(v, 2);
    const lines = [
      `shift t = ${t}   overlap = ${n[k]} px`,
      `SSD = ${sc(bf.ssd[k])}  (per px ${sc(ssd[k])})`,
      `CC  = ${sc(bf.cc[k])}  (per px ${sc(cc[k])})`,
      `NCC = ${fmt(ncc[k], 3)}`,
      `best: SSD ${ts[bS]}, CC ${ts[bC]}, NCC ${ts[bN]}` + (s.kind === "hand" ? "" : `   true ${s.t}`),
      `multiply-adds so far: ${ops}`,
      `all ${ts.length} shifts: ${bf.ops}`,
    ];
    ro.textContent = lines.join("\n");
  }

  function replaceStep(m: ReturnType<Plot["line"]>, xs: number[], ys: Float64Array, col: string, dots: boolean): ReturnType<Plot["line"]> {
    m.remove();
    const nm = sigP.step(xs, ys, { color: col, width: 3 }, "mid");
    if (dots) extra.push(sigP.scatter(xs, ys, { color: col, r: 5 }));
    return nm;
  }

  // ---------------- FFT mode ----------------
  function drawFFT(): void {
    const s = st.sim;
    let f = sig.f, g = sig.g;
    if (s.kind === "hand") { f = Float64Array.from([0, 1, 2, 0]); g = Float64Array.from([1, 2, 0, 0]); }
    const N = f.length;
    const mf = (a: Float64Array) => { if (!st.meanFree) return a; const m = a.reduce((x, y) => x + y, 0) / a.length; return a.map(v => v - m); };
    const fa = mf(f), ga = mf(g);
    const res = correlate1d(fa, ga, { pad: st.pad, p: st.phase ? 1 : 0, eps: 1e-9 });
    const n = res.n, [lo, hi] = shiftRange(), t = st.cur;
    const bf = bruteForce1d(fa, ga, st.pad ? {} : { circular: true });
    extra.forEach(m => m.remove()); extra.length = 0;
    scoreMarks.forEach(m => m.remove()); scoreMarks.length = 0;
    // signals
    const all = [...f, ...g];
    let ymin = Math.min(...all), ymax = Math.max(...all);
    const pd = 0.08 * (ymax - ymin || 1);
    sigP.setDomain([-0.5, N - 0.5], [ymin - pd, ymax + pd]);
    sigMarks.band.set([], [], []);
    sigMarks.a.remove(); sigMarks.b.remove();
    sigMarks.a = sigP.step(Array.from(f, (_, i) => i), f, { color: seriesColor(0), width: 3 }, "mid");
    sigMarks.b = sigP.step(Array.from(g, (_, i) => i), g, { color: seriesColor(1), width: 3 }, "mid");
    sigMarks.blob.set(-1e9, 0, "");
    // correlation: FFT line and brute-force dots over signed shifts
    const ts: number[] = [], cf: number[] = [];
    for (let u = lo; u <= hi; u++) { ts.push(u); cf.push(res.c[((u % n) + n) % n]); }
    const bts: number[] = [], bvs: number[] = [];
    if (!st.phase) {
      if (st.pad) bf.t.forEach((u, k) => { bts.push(u); bvs.push(bf.cc[k]); });
      else bf.t.forEach((u, k) => { bts.push(signed(u, N)); bvs.push(bf.cc[k]); });
    }
    const ext = [...cf, ...bvs];
    let ylo = Math.min(...ext), yhi = Math.max(...ext);
    const yp = 0.08 * (yhi - ylo || 1);
    scoreP.setDomain([lo - 0.5, hi + 0.5], [ylo - yp, yhi + yp]);
    scoreP.setLabels({ ylabel: st.phase ? "phase correlation" : "c(t)" });
    scoreMarks.push(scoreP.line(ts, cf, { color: seriesColor(2), width: 3.5 }));
    if (!st.phase) extra.push(scoreP.scatter(bts, bvs, { color: seriesColor(0), r: N > 32 ? 4 : 6 }));
    extra.push(scoreP.vline(t, { text: `t = ${t}`, width: 2 }));
    if (s.kind !== "hand") extra.push(scoreP.vline(s.t, { color: "var(--qb-c3)", dash: "2 5", width: 2 }));
    let pk = 0;
    for (let k = 1; k < cf.length; k++) if (cf[k] > cf[pk]) pk = k;
    extra.push(scoreP.scatter([ts[pk]], [cf[pk]], { color: seriesColor(1), r: 8 }));
    let maxDiff = 0;
    if (!st.phase) bts.forEach((u, k) => { maxDiff = Math.max(maxDiff, Math.abs(res.c[((u % n) + n) % n] - bvs[k])); });
    drawWaves(res.R, n, t);
    const R = res.R, z = (v: number) => (Math.abs(v) < 1e-9 ? 0 : v);
    const cplx = (k: number) => `${fmt(z(R.re[k]), 2)}${z(R.im[k]) < 0 ? "−" : "+"}${fmt(Math.abs(z(R.im[k])), 2)}i`;
    const lines = [
      `transform length ${n}${st.pad ? " (padded)" : ""}`,
      `c(${t}) = ${fmt(res.c[((t % n) + n) % n], 3)}`,
      `peak at t = ${ts[pk]}` + (s.kind === "hand" ? "" : `   true ${s.t}`),
      st.phase ? "" : `max |FFT − brute force| = ${maxDiff.toExponential(1)}`,
    ];
    if (n <= 8) lines.push(`A·conj(B): ` + Array.from({ length: n }, (_, k) => cplx(k)).join(", "));
    ro.textContent = lines.filter(x => x).join("\n");
  }

  function drawWaves(R: { re: Float64Array; im: Float64Array }, n: number, t: number): void {
    const svg = waveSvg!;
    svg.replaceChildren();
    const K = Math.min(Math.floor(n / 2), 12), cols = 6, rows = Math.ceil(K / cols);
    let mx = 0;
    for (let k = 1; k <= K; k++) mx = Math.max(mx, Math.hypot(R.re[k], R.im[k]));
    const cw = 430 / cols, ch = Math.min(115, 230 / Math.max(1, rows)), rad = Math.min(cw, ch) * 0.36;
    for (let k = 1; k <= K; k++) {
      const c = (k - 1) % cols, r = Math.floor((k - 1) / cols);
      const cx = cw * (c + 0.5), cy = 14 + ch * (r + 0.5);
      const circ = svgEl("circle", { cx, cy, r: rad, fill: "none" }, svg);
      circ.style.stroke = "var(--qb-grid)"; circ.style.strokeWidth = "2";
      const ax = svgEl("line", { x1: cx, x2: cx + rad, y1: cy, y2: cy }, svg);
      ax.style.stroke = "var(--qb-grid)"; ax.style.strokeDasharray = "3 3";
      // contribution of frequency k at shift t: R_k e^{2πi k t / n}
      const a = 2 * Math.PI * k * t / n, re = R.re[k] * Math.cos(a) - R.im[k] * Math.sin(a), im = R.re[k] * Math.sin(a) + R.im[k] * Math.cos(a);
      const len = mx > 0 ? Math.sqrt(Math.hypot(re, im) / mx) : 0, ang = Math.atan2(im, re);
      const ex = cx + rad * len * Math.cos(ang), ey = cy - rad * len * Math.sin(ang);
      if (len > 0.02) {
        const ln = svgEl("line", { x1: cx, y1: cy, x2: ex, y2: ey }, svg);
        ln.style.stroke = Math.cos(ang) > 0.9 ? "var(--qb-c3)" : "var(--qb-c1)"; ln.style.strokeWidth = "4"; ln.style.strokeLinecap = "round";
        const dot = svgEl("circle", { cx: ex, cy: ey, r: 4.5 }, svg);
        dot.style.fill = ln.style.stroke;
      }
      const tx = svgEl("text", { x: cx, y: cy + rad + 17, "text-anchor": "middle" }, svg);
      tx.textContent = `k = ${k}`; tx.style.fontSize = "16px";
    }
    const note = svgEl("text", { x: 215, y: 246, "text-anchor": "middle" }, svg);
    note.textContent = "arrow pointing right = this wave agrees at t"; note.style.fontSize = "15px"; note.style.fill = "var(--qb-muted)";
  }

  load(p0);
}

/** data-controls="shift,true,noise": show only the sliders whose names start with these words. */
function hideControls(panel: HTMLElement, spec: string | undefined): void {
  if (!spec) return;
  const keep = spec.split(",").map(x => x.trim().toLowerCase());
  panel.querySelectorAll<HTMLElement>(".qb-slider, .qb-check").forEach(el => {
    const name = (el.querySelector(".qb-slider-name")?.textContent ?? el.textContent ?? "").trim().toLowerCase();
    if (!keep.some(k => name.startsWith(k))) el.style.display = "none";
  });
}


/** lib/plot.ts numbers its clip paths per bundle, so two demos on one page can share an id;
 *  give this plot's clip path a page-unique id (local workaround). */
function uniqueClip(p: Plot): Plot {
  const cp = p.svg.querySelector("clipPath");
  if (cp) {
    const id = "xcorr1d-clip-" + Math.random().toString(36).slice(2, 10);
    cp.id = id;
    p.layer.setAttribute("clip-path", `url(#${id})`);
  }
  return p;
}

autoMount(NAME, mount);
