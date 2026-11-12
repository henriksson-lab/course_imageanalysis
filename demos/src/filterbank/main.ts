// The discrete wavelet transform as a two-channel filter bank. A 1D signal (length 64) is
// filtered with the low-pass h and the high-pass g and every second sample is kept (a, d);
// the synthesis side upsamples, filters and adds the two branches. The output equals the
// input to rounding error. Break it on purpose: drop the high-pass branch (the low branch
// alone is aliased), use the wrong synthesis filters, or change the boundary rule
// (periodic / symmetric / zero; the last two keep a few extra coefficients). Views:
// "filter bank" (one level), "cascade" (repeat on the low branch: all levels stacked) and
// "basis functions" (φ and ψ by the cascade algorithm, iteration by iteration).
// Auto-mounts on <div data-demo="filterbank">. Optional: data-wavelet="haar|db2|db3|db4|db8|sym4",
// data-signal="step|…|quadratic", data-view="bank|cascade|basis", data-preset="…", data-n="64".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import {
  WAVELETS, OrthoWavelet, Boundary, branchSignals, synthesisStep, wavedec, waverec, cascade, testSignal,
  SignalName, SIGNAL_LABELS, Step,
} from "../lib/wavelet";

const NAME = "filterbank";
const WNAMES = ["haar", "db2", "db3", "db4", "db8", "sym4"] as const;
type WN = typeof WNAMES[number];
const SIGNALS: SignalName[] = ["step", "twosteps", "sine", "chirp", "heavisine", "doppler", "ramp", "quadratic", "cubic"];
const COL_X = "var(--qb-fg)", COL_LO = seriesColor(0), COL_HI = seriesColor(1), COL_OUT = seriesColor(3);
type View = "bank" | "cascade" | "basis";

interface Settings { wavelet: WN; signal: SignalName; mode: Boundary; dropHigh: boolean; wrong: boolean; view: View; levels: number; iters: number }
interface Preset { id: string; label: string; s: Partial<Settings> }
const DEFAULTS: Settings = { wavelet: "haar", signal: "step", mode: "periodic", dropHigh: false, wrong: false, view: "bank", levels: 3, iters: 5 };
const PRESETS: Preset[] = [
  { id: "haar", label: "Haar = unit 1", s: { wavelet: "haar", signal: "twosteps", view: "bank" } },
  { id: "pr", label: "perfect reconstruction", s: { wavelet: "db4", signal: "heavisine", view: "bank" } },
  { id: "alias", label: "remove one branch: aliasing", s: { wavelet: "db2", signal: "chirp", view: "bank", dropHigh: true } },
  { id: "quad", label: "quadratic: db3 details vanish", s: { wavelet: "db3", signal: "quadratic", view: "bank", mode: "periodic" } },
  { id: "border", label: "periodic boundary: false edge", s: { wavelet: "db4", signal: "ramp", view: "bank", mode: "periodic" } },
  { id: "cascade", label: "cascade: 4 levels", s: { wavelet: "db2", signal: "heavisine", view: "cascade", levels: 4 } },
  { id: "basis", label: "basis functions", s: { wavelet: "db2", view: "basis", iters: 1 } },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .fb-left { flex: 1 1 560px; min-width: 0; max-width: 800px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .fb-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .1em .8em; }
[data-demo="${NAME}"] .fb-cap { font-size: 16px; font-weight: 600; text-align: center; margin: 0; line-height: 1.15; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .4em; }
[data-demo="${NAME}"] .fb-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] .qb-slider.off, [data-demo="${NAME}"] .qb-check.off { opacity: .35; pointer-events: none; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .fb-left { flex: 1 1 640px; max-width: 660px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 420px; max-width: 440px; gap: .2em; }
.reveal [data-demo="${NAME}"] .fb-lab { display: none; }
.reveal [data-demo="${NAME}"] .qb-presets { gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-slider.off, .reveal [data-demo="${NAME}"] .qb-check.off { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .05em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Padding for a plot range that never collapses (tiny ranges would make the ticks loop). */
const padOf = (lo: number, hi: number, f: number) => Math.max(f * (hi - lo), 1e-3 * Math.max(1, Math.abs(lo), Math.abs(hi)), 0.05);
const sci = (v: number) => (v === 0 ? "0" : Math.abs(v) >= 1e-3 ? v.toFixed(4) : v.toExponential(1));

function mount(host: HTMLElement): void {
  injectStyle();
  const N = [16, 32, 64, 128, 256].includes(Number(host.dataset.n)) ? Number(host.dataset.n) : 64;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: Settings = { ...DEFAULTS };
  if (WNAMES.includes(host.dataset.wavelet as WN)) init.wavelet = host.dataset.wavelet as WN;
  if (SIGNALS.includes(host.dataset.signal as SignalName)) init.signal = host.dataset.signal as SignalName;
  if (["bank", "cascade", "basis"].includes(host.dataset.view ?? "")) init.view = host.dataset.view as View;
  if (p0) Object.assign(init, p0.s);
  const st: Settings = { ...init };

  const left = htmlEl("div", "fb-left", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "fb-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULTS, p.s); sync(); draw(); } })));
  htmlEl("div", "fb-lab", "view", panel);
  const vRow = htmlEl("div", "qb-presets", "", panel);
  const vBtns = (["bank", "cascade", "basis"] as View[]).map(v => button(vRow, v === "bank" ? "filter bank" : v === "cascade" ? "cascade" : "basis functions", () => { st.view = v; changed(); }));
  htmlEl("div", "fb-lab", "wavelet", panel);
  const wRow = htmlEl("div", "qb-presets", "", panel);
  const wBtns = WNAMES.map(w => button(wRow, WAVELETS[w].label, () => { st.wavelet = w; changed(); }));
  htmlEl("div", "fb-lab", "signal", panel);
  const sRow = htmlEl("div", "qb-presets", "", panel);
  const sBtns = SIGNALS.map(s => button(sRow, SIGNAL_LABELS[s], () => { st.signal = s; changed(); }));
  htmlEl("div", "fb-lab", "boundary", panel);
  const bRow = htmlEl("div", "qb-presets", "", panel);
  const bBtns = (["periodic", "symmetric", "zero"] as Boundary[]).map(m => button(bRow, m, () => { st.mode = m; changed(); }));
  const dropBox = checkbox(panel, "drop the high-pass branch", st.dropHigh, v => { st.dropHigh = v; changed(); });
  const wrongBox = checkbox(panel, "wrong synthesis filters", st.wrong, v => { st.wrong = v; changed(); });
  const levS = slider(panel, "levels", 1, 5, 1, st.levels, v => { st.levels = v; changed(); });
  const itS = slider(panel, "iterations", 0, 10, 1, st.iters, v => { st.iters = v; changed(); });
  button(panel, "reset", () => { Object.assign(st, init); sync(); draw(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); }
  function sync(): void {
    vBtns.forEach((b, i) => b.classList.toggle("active", ["bank", "cascade", "basis"][i] === st.view));
    wBtns.forEach((b, i) => b.classList.toggle("active", WNAMES[i] === st.wavelet));
    sBtns.forEach((b, i) => b.classList.toggle("active", SIGNALS[i] === st.signal));
    bBtns.forEach((b, i) => b.classList.toggle("active", ["periodic", "symmetric", "zero"][i] === st.mode));
    dropBox.checked = st.dropHigh; wrongBox.checked = st.wrong;
    levS.set(st.levels); itS.set(st.iters);
    const basis = st.view === "basis";
    for (const el of [...sBtns, ...bBtns]) el.disabled = basis;
    dropBox.parentElement!.classList.toggle("off", st.view !== "bank");
    wrongBox.parentElement!.classList.toggle("off", st.view !== "bank");
    levS.input.parentElement!.classList.toggle("off", st.view !== "cascade");
    itS.input.parentElement!.classList.toggle("off", !basis);
  }

  const cap = (parent: Element, text: string, color = "var(--qb-fg)") => { const c = htmlEl("div", "fb-cap", text, parent); c.style.color = color; return c; };

  function draw(): void {
    left.replaceChildren();
    const w = WAVELETS[st.wavelet] as OrthoWavelet;
    if (st.view === "basis") return drawBasis(w);
    const x = testSignal(st.signal, N);
    const lo = Math.min(...x), hi = Math.max(...x), pad = padOf(lo, hi, 0.15);
    if (st.view === "cascade") return drawCascade(w, x);

    const L = w.h.length, br = branchSignals(x, w, st.mode), step = br.step;
    const out = synthesisStep(step, w, st.mode, { dropHigh: st.dropHigh, wrongFilters: st.wrong });
    const lowBack = synthesisStep(step, w, st.mode, { dropHigh: true, wrongFilters: st.wrong });
    const highBack = synthesisStep(step, w, st.mode, { dropLow: true, wrongFilters: st.wrong });
    const idx = Float64Array.from({ length: N }, (_, i) => i);
    const xd: [number, number] = st.mode === "periodic" ? [-1, N] : [-L, N + 1];

    cap(left, st.dropHigh || st.wrong ? "input x (black) and output (orange): not equal" : "input x (black) and output (orange, dashed): the same", COL_X);
    const pIO = new Plot(left, { width: 800, height: 165, x: xd, y: [lo - pad, hi + pad], margin: { b: 28, l: 56 }, yticks: 3 });
    pIO.line(idx, x, { color: COL_X, width: 3 });
    pIO.line(idx, out, { color: COL_OUT, width: 3, dash: st.dropHigh || st.wrong ? undefined : "7 5" });
    pIO.vline(-0.5, { color: "var(--qb-muted)", width: 1.5 }); pIO.vline(N - 0.5, { color: "var(--qb-muted)", width: 1.5 });

    const grid = htmlEl("div", "fb-grid", "", left);
    const cells = [0, 1, 2, 3].map(() => htmlEl("div", "", "", grid));
    // analysis branches: full-rate filtered signal (faint) and kept samples (dots) at position 2k
    const posK = Float64Array.from({ length: step.a.length }, (_, j) => 2 * (step.k0 + j));
    const fullPos = Float64Array.from({ length: N }, (_, i) => i);
    const branch = (cell: HTMLElement, title: string, full: Float64Array, kept: Float64Array, color: string, faded: boolean) => {
      cap(cell, title, color);
      const all = [...kept, ...full];
      const a = Math.min(0, ...all), b = Math.max(0, ...all), pd = padOf(a, b, 0.12);
      const p = new Plot(cell, { width: 400, height: 170, x: xd, y: [a - pd, b + pd], margin: { b: 28, l: 56 }, yticks: 3, xticks: 3 });
      p.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" });
      p.line(fullPos, full, { color, width: 1.5, opacity: 0.35 });
      const stems = p.mark(g => {
        for (let j = 0; j < kept.length; j++) {
          const X = p.x(posK[j]), Y = p.y(kept[j]);
          svgEl("line", { x1: X, x2: X, y1: p.y(0), y2: Y, stroke: color, "stroke-width": 2, opacity: faded ? 0.3 : 1 }, g);
          svgEl("circle", { cx: X, cy: Y, r: 4, fill: color, opacity: faded ? 0.3 : 1 }, g);
        }
      }, []);
      void stems;
    };
    const backPlot = (cell: HTMLElement, title: string, y: Float64Array, color: string) => {
      cap(cell, title, color);
      const a = Math.min(lo, ...y), b = Math.max(hi, ...y), pd = padOf(a, b, 0.12);
      const p = new Plot(cell, { width: 400, height: 170, x: xd, y: [a - pd, b + pd], margin: { b: 28, l: 56 }, yticks: 3, xticks: 3 });
      p.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" });
      p.line(idx, x, { color: "var(--qb-muted)", width: 1.5, opacity: 0.5 });
      p.line(idx, y, { color, width: 2.5 });
    };
    branch(cells[0], `low-pass h, keep every 2nd: a (${step.a.length})`, br.lowFull, step.a, COL_LO, false);
    backPlot(cells[1], "low branch back: upsample, filter", lowBack, COL_LO);
    branch(cells[2], `high-pass g, keep every 2nd: d (${step.d.length})`, br.highFull, step.d, COL_HI, st.dropHigh);
    backPlot(cells[3], st.dropHigh ? "high branch: dropped" : "high branch back: upsample, filter", st.dropHigh ? new Float64Array(N) : highBack, COL_HI);

    let err = 0;
    for (let i = 0; i < N; i++) err = Math.max(err, Math.abs(out[i] - x[i]));
    // largest |d| whose filter window lies inside the signal
    let dIn = 0;
    for (let j = 0; j < step.d.length; j++) { const s = 2 * (step.k0 + j); if (s >= 0 && s + L - 1 < N) dIn = Math.max(dIn, Math.abs(step.d[j])); }
    let dAll = 0;
    for (const v of step.d) dAll = Math.max(dAll, Math.abs(v));
    const e = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v * v, 0);
    ro.textContent = [
      `${w.label}: ${L} taps, ${w.p} vanishing moment${w.p > 1 ? "s" : ""}`,
      `coefficients  ${step.a.length} + ${step.d.length} = ${2 * step.a.length} (N = ${N})`,
      `max |x − output|  ${sci(err)}`,
      `largest |d| inside   ${sci(dIn)}`,
      `largest |d| overall  ${sci(dAll)}`,
      `Σx² ${e(x).toFixed(3)}  Σa²+Σd² ${(e(step.a) + e(step.d)).toFixed(3)}`,
    ].join("\n");
    note.textContent = "Dots: the kept samples (every second one) of each filtered signal (faint line). Grey lines at the ends mark the borders of the signal.";
  }

  function drawCascade(w: OrthoWavelet, x: Float64Array): void {
    const J = st.levels, dec = wavedec(x, w, J, st.mode), rec = waverec(dec.steps, w, st.mode);
    cap(left, `cascade: ${J} level${J > 1 ? "s" : ""}, the low branch is split again each time`);
    const VW = 800, rows = J + 2, rh = 470 / rows, l = 90, r = 790;
    const L = w.h.length;
    const xd: [number, number] = st.mode === "periodic" ? [-1, N] : [-L * 2 ** (J - 1), N + 1];
    const X = (p: number) => l + (p - xd[0]) / (xd[1] - xd[0]) * (r - l);
    const svg = svgEl("svg", { viewBox: `0 0 ${VW} ${rh * rows + 10}`, width: "100%", class: "qb-plot-svg" }, left);
    const row = (k: number, label: string, pos: ArrayLike<number>, vals: ArrayLike<number>, color: string, line: boolean) => {
      const y0 = 6 + k * rh, h = rh - 8;
      let mx = 0;
      for (let i = 0; i < vals.length; i++) mx = Math.max(mx, Math.abs(vals[i]));
      const lo = line ? Math.min(...Array.from(vals)) : -mx, hi = line ? Math.max(...Array.from(vals)) : mx;
      const Y = (v: number) => y0 + h - (v - (lo - 1e-12)) / ((hi - lo) || 1) * h;
      svgEl("rect", { x: l, y: y0, width: r - l, height: h, fill: "var(--qb-panel)" }, svg);
      if (!line) svgEl("line", { x1: l, x2: r, y1: Y(0), y2: Y(0), stroke: "var(--qb-axis)", "stroke-width": 1 }, svg);
      if (line) {
        let d = "";
        for (let i = 0; i < vals.length; i++) d += `${i ? "L" : "M"}${X(pos[i])},${Y(vals[i])}`;
        svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 2.5 }, svg);
      } else for (let i = 0; i < vals.length; i++) {
        svgEl("line", { x1: X(pos[i]), x2: X(pos[i]), y1: Y(0), y2: Y(vals[i]), stroke: color, "stroke-width": 2 }, svg);
        svgEl("circle", { cx: X(pos[i]), cy: Y(vals[i]), r: 3.5, fill: color }, svg);
      }
      const t = svgEl("text", { x: l - 8, y: y0 + h / 2 + 6, "text-anchor": "end", "font-size": 18 }, svg);
      t.textContent = label;
      const m = svgEl("text", { x: r - 6, y: y0 + 18, "text-anchor": "end", "font-size": 15 }, svg);
      m.textContent = `${vals.length} · max ${mx.toPrecision(2)}`;
      (m as SVGTextElement).style.fill = "var(--qb-muted)";
    };
    const idx = Float64Array.from({ length: N }, (_, i) => i);
    row(0, "x", idx, x, COL_X, true);
    const pos = (s: Step, j: number) => Float64Array.from({ length: s.d.length }, (_, k) => 2 ** j * (s.k0 + k) + (2 ** j - 1) / 2);
    for (let j = 1; j <= J; j++) row(j, `d${j}`, pos(dec.steps[j - 1], j), dec.steps[j - 1].d, seriesColor(j), false);
    const last = dec.steps[J - 1];
    row(J + 1, `a${J}`, pos(last, J), last.a, COL_LO, false);
    let err = 0;
    for (let i = 0; i < N; i++) err = Math.max(err, Math.abs(rec[i] - x[i]));
    const count = dec.steps.reduce((s, t) => s + t.d.length, 0) + last.a.length;
    let work = 0, n = N;
    for (let j = 0; j < J; j++) { work += n; n = Math.ceil(n / 2); }
    ro.textContent = [
      `${w.label}, ${st.mode} boundary`,
      `coefficients  ${count} (N = ${N})`,
      `max |x − x̂|  ${sci(err)}`,
      `samples filtered  ${work} < 2N = ${2 * N}`,
    ].join("\n");
    note.textContent = "Each row is one band; coefficients are drawn at the centre of the samples they summarise. Finer levels (d1) have more, narrower coefficients.";
  }

  function drawBasis(w: OrthoWavelet): void {
    const it = st.iters, r = cascade(w, it), L = w.h.length;
    const dt = 2 ** -it;
    cap(left, `${w.label}: scaling function φ (blue) and wavelet ψ (red), cascade iteration ${it}`);
    const all = [...r.phi, ...r.psi];
    const lo = Math.min(...all), hi = Math.max(...all), pad = padOf(lo, hi, 0.1);
    const p = new Plot(left, { width: 800, height: 470, x: [0, L - 1 + 0.001], y: [lo - pad, hi + pad], xlabel: "t", yticks: 6 });
    p.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" });
    p.step(r.t, r.phi, { color: COL_LO, width: 3 }, "post");
    p.step(r.t, r.psi, { color: COL_HI, width: 3 }, "post");
    let s1 = 0, s2 = 0;
    for (let k = 0; k < r.phi.length; k++) { s1 += r.phi[k] * dt; s2 += r.psi[k] * dt; }
    ro.textContent = [
      `${w.label}: support [0, ${L - 1}]`,
      `${w.p} vanishing moment${w.p > 1 ? "s" : ""}`,
      `iteration ${it}: step 1/${2 ** it}`,
      `∫φ = ${s1.toFixed(3)}   ∫ψ = ${Math.abs(s2) < 5e-4 ? "0.000" : s2.toFixed(3)}`,
    ].join("\n");
    note.textContent = "Each iteration applies the two-scale relation φ(t) = √2 Σ h_n φ(2t − n) once more; the shape converges. More vanishing moments: smoother and wider.";
  }

  sync();
  draw();
  if (p0) presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
}

autoMount(NAME, mount);
