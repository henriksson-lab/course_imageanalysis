// What tracks measure: the mean squared displacement. Simulated 2D tracks (seeded) of one
// motion type: free diffusion, directed + diffusion, confined in a disc, anomalous (fractional
// Brownian motion), switching (waiting/running) or a persistent random walk; with localisation
// error σ, frame interval Δt and track length N. Panels: a few tracks; the MSD vs lag per track
// (thin) and over all tracks (thick), on linear or log–log axes, with the fitted line over the
// shaded fit range; a histogram of the D of each track (line through its first 4 lags). data-mode="speed": the apparent speed
// of one random walk measured at frame intervals Δt, 2Δt, 4Δt, … with √(π D/Δt).
//
// <div data-demo="msdplot" data-preset="free|directed|confined|locerr|short|speed|tail|anomalous|persistent"
//      data-panels="tracks,msd,hist" data-mode="msd|speed"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import type { MotionType } from "../lib/tracksim";
import { State, DEFAULT, PRESETS, simulate, analyse, speedCurve, apparentSpeed } from "./model";
import { persistentMSD } from "../lib/trackstats";

const NAME = "msdplot";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .6em; }
${S} .ms-tracks { flex: 0 1 300px; min-width: 200px; }
${S} .ms-msd { flex: 1 1 460px; max-width: 560px; min-width: 260px; }
${S} .ms-hist { flex: 0 1 320px; min-width: 220px; }
${S} .qb-panel { flex: 1 1 300px; max-width: 380px; gap: .25em; }
${S} .ms-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-size: 17px; white-space: pre-wrap; }
${S} .qb-presets { flex-basis: 100%; }
${S} .qb-btn.ms-on { background: var(--qb-accent); border-color: var(--qb-accent); color: #fff; }
.reveal ${S} { font-size: 18px; gap: .4em; }
.reveal ${S} .ms-tracks { flex: 0 0 250px; }
.reveal ${S} .ms-msd { flex: 0 0 560px; max-width: 560px; }
.reveal ${S} .qb-panel { flex: 0 0 400px; gap: .05em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .15em .4em; }
`;

const TYPES: [MotionType, string][] = [["free", "free"], ["directed", "directed"], ["confined", "confined"], ["anomalous", "anomalous"], ["switching", "switching"], ["persistent", "persistent"]];

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const fmt = (v: number) => (!Number.isFinite(v) ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : Math.abs(v) >= 0.01 ? v.toFixed(3) : v.toExponential(1));

autoMount(NAME, host => {
  injectStyle();
  const panels = (host.dataset.panels ?? "tracks,msd,hist").split(",").map(s => s.trim());
  let st: State = { ...DEFAULT, mode: host.dataset.mode === "speed" ? "speed" : "msd" };
  const trBox = htmlEl("div", "ms-tracks", "", host);
  const trPlot = new Plot(trBox, { width: 300, height: 300, x: [-1, 1], y: [-1, 1], equal: true, xlabel: "x (µm)", font: 22, margin: { l: 60 } });
  const msBox = htmlEl("div", "ms-msd", "", host);
  let msPlot: Plot;
  const makeMsPlot = () => {
    msBox.replaceChildren();
    const sp = st.mode === "speed";
    msPlot = new Plot(msBox, { width: 560, height: 400, x: [0.1, 10], y: [0.01, 1], xlog: st.loglog || sp, ylog: st.loglog || sp,
      xlabel: sp ? "frame interval Δt (s)" : "lag τ (s)", ylabel: sp ? "apparent speed (µm/s)" : "MSD (µm²)", font: 26 });
  };
  const panel = htmlEl("div", "qb-panel", "", host);
  const hiBox = htmlEl("div", "ms-hist", "", host);
  const hiPlot = new Plot(hiBox, { width: 320, height: 250, x: [0, 1], y: [0, 1], xlabel: "D per track (µm²/s), lags 1–4", ylabel: "tracks", font: 20 });
  const read = htmlEl("div", "qb-readout", "", panel);
  const typeRow = htmlEl("div", "ms-row", "", panel);
  const typeBtns = TYPES.map(([t, l]) => button(typeRow, l, () => { st.type = t; st.mode = "msd"; makeMsPlot(); recompute(); }));
  const lg = (lo: number, hi: number) => ({ lo: Math.log10(lo), hi: Math.log10(hi) });
  const logS = (name: string, lo: number, hi: number, key: "D" | "dt" | "sigma", digits = 2) => {
    const r = lg(lo, hi);
    return slider(panel, name, r.lo, r.hi, 0.02, Math.log10(Math.max(lo, st[key])), v => { st[key] = +(10 ** v).toPrecision(digits); recompute(); }, v => String(+(10 ** v).toPrecision(digits)));
  };
  const sl = {
    D: logS("D (µm²/s)", 0.001, 1, "D"),
    dt: logS("Δt (s)", 0.01, 2, "dt"),
    N: slider(panel, "points per track N", 10, 500, 10, st.N, v => { st.N = v; recompute(); }),
    n: slider(panel, "tracks", 1, 100, 1, st.n, v => { st.n = v; recompute(); }),
    sigma: slider(panel, "loc. error σ (µm)", 0, 0.1, 0.005, st.sigma, v => { st.sigma = v; recompute(); }, v => v.toFixed(3)),
    v: slider(panel, "speed v (µm/s)", 0, 2, 0.05, st.v, v => { st.v = v; recompute(); }, v => v.toFixed(2)),
    R: slider(panel, "cage radius R (µm)", 0.1, 2, 0.05, st.R, v => { st.R = v; recompute(); }, v => v.toFixed(2)),
    alpha: slider(panel, "α (anomalous)", 0.2, 1.8, 0.05, st.alpha, v => { st.alpha = v; recompute(); }, v => v.toFixed(2)),
    fitFrac: slider(panel, "fit first % of lags", 0.05, 1, 0.05, st.fitFrac, v => { st.fitFrac = v; recompute(); }, v => `${Math.round(100 * v)}`),
  };
  const chkRow = htmlEl("div", "ms-row", "", panel);
  const llC = checkbox(chkRow, "log–log axes", st.loglog, v => { st.loglog = v; makeMsPlot(); draw(); });
  const spBtn = button(chkRow, "speed trap", () => { st.mode = st.mode === "speed" ? "msd" : "speed"; makeMsPlot(); recompute(); });
  presets(host, [...PRESETS.map(p => ({ label: p.label, apply: () => apply(p.s) })), { label: "reset", apply: () => apply({}) }]);

  let tracks = simulate(st), fits = analyse(st, tracks);
  let timer = 0;
  function apply(p: Partial<State>): void {
    st = { ...DEFAULT, ...p };
    for (const k of Object.keys(sl) as (keyof typeof sl)[]) {
      const v = st[k] as number;
      sl[k].set(k === "D" || k === "dt" ? Math.log10(v) : k === "sigma" ? v : v);
    }
    llC.checked = st.loglog;
    makeMsPlot();
    recomputeNow();
  }
  function recompute(): void { window.clearTimeout(timer); timer = window.setTimeout(recomputeNow, 40); }
  function recomputeNow(): void {
    if (st.mode === "msd") { tracks = simulate(st); fits = analyse(st, tracks); }
    draw();
  }

  function draw(): void {
    typeBtns.forEach((b, i) => b.classList.toggle("ms-on", st.mode === "msd" && TYPES[i][0] === st.type));
    spBtn.classList.toggle("ms-on", st.mode === "speed");
    // in speed mode only D and σ matter
    for (const k of Object.keys(sl) as (keyof typeof sl)[]) (sl[k].input.parentElement as HTMLElement).style.display = st.mode === "speed" && k !== "D" && k !== "sigma" ? "none" : "";
    typeRow.style.display = st.mode === "speed" ? "none" : "";
    (llC.parentElement as HTMLElement).style.display = st.mode === "speed" ? "none" : "";
    trBox.style.display = st.mode === "msd" && panels.includes("tracks") ? "" : "none";
    hiBox.style.display = st.mode === "msd" && panels.includes("hist") ? "" : "none";
    if (st.mode === "speed") { drawSpeed(); return; }
    // tracks: the first 8, centred on their start; the first one highlighted
    trPlot.clear();
    const show = tracks.slice(0, 8);
    let ext = 0.05;
    for (const t of show) for (let i = 0; i < t.x.length; i++) ext = Math.max(ext, Math.abs(t.x[i] - t.x[0]), Math.abs(t.y[i] - t.y[0]));
    trPlot.setDomain([-ext * 1.05, ext * 1.05], [-ext * 1.05, ext * 1.05]);
    show.forEach((t, k) => trPlot.line(t.x.map(v => v - t.x[0]), t.y.map(v => v - t.y[0]), { color: k === 0 ? "#222" : `hsl(${(k * 47) % 360} 55% 60%)`, width: k === 0 ? 2.2 : 1.2, opacity: k === 0 ? 1 : 0.8 }));
    if (st.type === "confined") trPlot.ellipse([-tracks[0].x[0], -tracks[0].y[0]], [[st.R * st.R, 0], [0, st.R * st.R]], 1, { color: "#888", fill: "none", width: 1.5, dash: "4 3" });
    // MSD
    const p = msPlot, dt = st.dt, e = fits.ens;
    const tau = e.lag.map(L => L * dt);
    const tf = tau[fits.nFit - 1];
    // linear axes: show up to 4 × the fit range (the whole range when fitting more than a quarter)
    const xmax = st.loglog ? tau[tau.length - 1] : Math.min(tau[tau.length - 1], 4 * tf);
    const inRange = (m: { lag: number[]; msd: number[] }) => m.msd.filter((v, i) => v > 0 && m.lag[i] * dt <= xmax);
    const allY = [...inRange(e), ...fits.perTrack.slice(0, 20).flatMap(inRange)];
    let ylo = Math.min(...allY), yhi = Math.max(...allY);
    if (!st.loglog) { ylo = Math.min(0, fits.line.offset); yhi *= 1.05; } else { ylo *= 0.8; yhi *= 1.25; }
    p.clear();
    p.setDomain(st.loglog ? [tau[0] * 0.9, xmax * 1.1] : [0, xmax], [ylo, yhi]);
    p.band([st.loglog ? tau[0] * 0.9 : 0, tf], [ylo, ylo], [yhi, yhi], { color: "#f0e442", fill: "#f0e442", fillOpacity: 0.18 });
    fits.perTrack.slice(0, 20).forEach(m => p.line(m.lag.map(L => L * dt), m.msd, { color: "#9aa", width: 1, opacity: 0.6 }));
    p.line(tau, e.msd, { color: "var(--qb-c1, #1f77b4)", width: 3.5, label: "ensemble MSD" });
    if (st.type === "persistent") p.fn(x => persistentMSD(x, st.v, st.P), { color: "#d62728", width: 2.5, dash: "7 4", label: "2s²P(τ − P(1 − e^(−τ/P)))" });
    else p.fn(x => fits.line.offset + fits.line.slope * x, { color: "#d62728", width: 2.5, dash: "7 4", label: `fit: 4D τ + offset` });
    if (st.loglog) {
      // guide slopes: 1 through the last point, 2 through the first
      const x0 = tau[0], y0 = e.msd[0], x1 = tau[tau.length - 1], y1 = e.msd[e.msd.length - 1];
      p.fn(x => y1 * (x / x1), { color: "#777", width: 1.5, dash: "2 4", label: "slope 1" });
      p.fn(x => y0 * (x / x0) ** 2, { color: "#444", width: 1.5, dash: "1 3", label: "slope 2" });
    }
    if (st.type === "confined") p.hline(st.R * st.R, { color: "#2ca02c", dash: "5 4", width: 2, label: "R²" });
    p.legend(undefined, st.loglog ? "tl" : "tl");
    // histogram of per-track D
    const Ds = fits.perTrackD.filter(Number.isFinite);
    const lo = Math.min(0, ...Ds), hi = Math.max(...Ds, 1e-6);
    const nb = 16, w = (hi - lo) / nb || 1;
    const edges = Array.from({ length: nb + 1 }, (_, i) => lo + i * w), counts = new Array(nb).fill(0);
    for (const d of Ds) counts[Math.min(nb - 1, Math.max(0, Math.floor((d - lo) / w)))]++;
    hiPlot.clear();
    hiPlot.setDomain([lo, hi], [0, Math.max(...counts) + 1]);
    hiPlot.histogram(edges, counts, { color: "var(--qb-c1, #1f77b4)" });
    if (st.type === "free" || st.type === "directed" || st.type === "confined") hiPlot.vline(st.D, { color: "#d62728", width: 2.5, dash: "5 4" });
    const mean = Ds.reduce((s, x) => s + x, 0) / Math.max(1, Ds.length), sd = Math.sqrt(Ds.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, Ds.length - 1));
    const lines = [
      `fit over lags 1–${fits.nFit} (τ ≤ ${fmt(tf)} s)`,
      `D = slope/4 = ${fmt(fits.D)} µm²/s   (true ${st.type === "anomalous" || st.type === "persistent" ? "–" : fmt(st.D)})`,
      `offset = ${fmt(fits.line.offset)} µm²   (4σ² = ${fmt(4 * st.sigma * st.sigma)})`,
      `α (log–log slope) = ${fmt(fits.alpha)}`,
    ];
    if (st.type === "directed") lines.push(`v from τ² term = ${fmt(Math.sqrt(Math.max(0, fits.quad.b)))} µm/s   (true ${st.v})`);
    if (st.type === "confined") lines.push(`plateau = ${fmt(fits.plateau.plateau + fits.plateau.c)} µm²   (R² = ${fmt(st.R * st.R)})`);
    lines.push(`per-track D: mean ${fmt(mean)}, sd ${fmt(sd)}`);
    read.textContent = lines.join("\n");
  }

  function drawSpeed(): void {
    const c = speedCurve(st);
    const p = msPlot;
    p.clear();
    const lo = Math.min(...c.speed) * 0.7, hi = Math.max(...c.speed, apparentSpeed(st.D, c.dt[0], st.sigma)) * 1.3;
    p.setDomain([c.dt[0] * 0.8, c.dt[c.dt.length - 1] * 1.25], [lo, hi]);
    p.fn(x => Math.sqrt(Math.PI * st.D / x), { color: "#d62728", width: 2.5, dash: "7 4", label: "√(π D / Δt)" });
    if (st.sigma > 0) p.fn(x => apparentSpeed(st.D, x, st.sigma), { color: "#2ca02c", width: 2.5, dash: "3 3", label: "with loc. error σ" });
    p.line(c.dt, c.speed, { color: "var(--qb-c1, #1f77b4)", width: 3 });
    p.scatter(c.dt, c.speed, { color: "var(--qb-c1, #1f77b4)", r: 6, label: "measured" });
    p.legend(undefined, "tr");
    read.textContent = [
      "one random walk, D = " + st.D + " µm²/s,",
      "measured every Δt: speed = mean step / Δt",
      ...c.dt.filter((_, i) => i % 2 === 0).map((d, i) => `Δt ${String(+(1000 * d).toPrecision(3)).padStart(5)} ms → ${fmt(c.speed[2 * i])} µm/s`),
    ].join("\n");
  }

  const pre = host.dataset.preset ? PRESETS.find(p => p.id === host.dataset.preset) : undefined;
  apply({ ...(pre ? pre.s : {}), ...(host.dataset.mode === "speed" ? { mode: "speed" as const } : {}) });
});
