// Samplers in 1D, with time as an axis. Background: the exact density of the noised data
// (time τ from 1 = noise on the left to 0 = data on the right; position in the
// variance-preserving form). 30 highlighted sample paths; 2 000 endpoints as a histogram
// against the true density at the right. Controls: steps, η (0 = deterministic, 1 = ancestral),
// step spacing, Euler or two-stage, Langevin corrections, seed; "invert": the deterministic
// sampler run backwards from a data point to its noise, and forwards again.
//
// <div data-demo="samplerpaths" data-preset="same|one|five|spacing|forget|invert"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import type { Spacing } from "../lib/diffusion";
import * as M from "./model";
import { injectStyle, choiceRow, helperCss, fmt } from "../scorematch1d/dview";

const NAME = "samplerpaths";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .sp-plots { flex: 1 1 600px; min-width: 0; max-width: 760px; display: flex; gap: 0; }
${S} .sp-plots > svg:first-child { flex: 540 1 0; min-width: 0; }
${S} .sp-plots > svg:last-child { flex: 170 1 0; min-width: 0; }
${S} .qb-panel { flex: 1 1 300px; gap: .35em; }
${S} .qb-readout { font-size: 15px; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .sp-plots { flex: 0 0 700px; }
.reveal ${S} .qb-panel { flex: 1 1 400px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

interface State { data: M.DataId; steps: number; eta: number; spacing: Spacing; twoStage: boolean; langevin: number; seed: number; invert: boolean; x0: number; bad: boolean }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { data: "two", steps: 100, eta: 0, spacing: "time", twoStage: false, langevin: 0, seed: 1, invert: false, x0: 1.3, bad: false };
const PRESETS: Preset[] = [
  { id: "same", label: "same endpoints, different paths", s: { steps: 100, eta: 1 } },
  { id: "one", label: "one step: everything at the mean", s: { steps: 1, eta: 0 } },
  { id: "five", label: "5 steps: too narrow", s: { data: "three", steps: 5, eta: 0 } },
  { id: "spacing", label: "spacing matters", s: { data: "three", steps: 5, eta: 0, spacing: "logsnr" } },
  { id: "forget", label: "stochastic forgets a bad start", s: { data: "three", steps: 20, eta: 1, bad: true } },
  { id: "invert", label: "inversion and back", s: { steps: 20, eta: 0, invert: true, x0: 1.3 } },
];
const NPTS = 2000, NPATHS = 30;
const stepsOf = (u: number) => Math.max(1, Math.round(200 ** (u / 100)));
const uOf = (n: number) => 100 * Math.log(n) / Math.log(200);
const YR: [number, number] = [-3, 3];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  // larger labels on slides (the plots are scaled down there)
  const F = host.closest(".reveal") ? 23 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, ...p0.s };

  const plots = htmlEl("div", "sp-plots", "", host);
  const pl = new Plot(plots, { width: 540, height: 430, x: [1, 0], y: YR, xlabel: "time τ (noise → data)", ylabel: "position", xticks: [1, 0.75, 0.5, 0.25, 0], yticks: [-3, -2, -1, 0, 1, 2, 3], font: F, grid: false, margin: { r: 4 } });
  const heat = pl.heatmap(new Float64Array(4), 2, 2, { lo: 0, hi: 1, map: t => [1 - 0.55 * t, 1 - 0.4 * t, 1 - 0.08 * t], flipY: false });
  const pathMarks = pl.mark<[Float64Array, Float64Array[]]>((g, taus, paths) => {
    paths.forEach(p => {
      let d = "";
      for (let i = 0; i < p.length; i++) d += (i ? "L" : "M") + pl.x(taus[i]).toFixed(1) + "," + pl.y(p[i]).toFixed(1);
      const e = document.createElementNS("http://www.w3.org/2000/svg", "path");
      e.setAttribute("d", d); e.style.fill = "none"; e.style.stroke = seriesColor(0); e.style.strokeWidth = "1.6"; e.style.opacity = "0.75";
      g.appendChild(e);
    });
  }, [new Float64Array(0), []]);
  const invPath = pl.line([], [], { color: seriesColor(1), width: 4 });
  const invDots = pl.scatter([], [], { color: seriesColor(1), r: 8 });
  const side = new Plot(plots, { width: 170, height: 430, x: [0, 1], y: YR, xlabel: "endpoints", xticks: [], yticks: [], font: F, grid: false, margin: { l: 6, r: 6 } });
  const hist = side.mark<[Float64Array, Float64Array]>((g, edges, cnt) => {
    for (let b = 0; b < cnt.length; b++) {
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const ya = side.y(edges[b + 1]), yb = side.y(edges[b]);
      r.setAttribute("x", String(side.x(0))); r.setAttribute("y", String(ya)); r.setAttribute("width", String(Math.max(0, side.x(cnt[b]) - side.x(0)))); r.setAttribute("height", String(yb - ya));
      r.style.fill = seriesColor(0); r.style.fillOpacity = "0.55";
      g.appendChild(r);
    }
  }, [new Float64Array(0), new Float64Array(0)]);
  const trueD = side.line([], [], { color: "var(--qb-fg)", width: 2.5 });

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const dataC = choiceRow(panel, "data", M.DATA_IDS.map(v => ({ v, label: M.DATA_LABEL[v] })), st.data, v => { st.data = v; recompute(); });
  const stepS = slider(panel, "steps", 0, 100, 1, uOf(st.steps), u => { st.steps = stepsOf(u); recompute(); }, u => String(stepsOf(u)));
  const etaS = slider(panel, "η (0 = deterministic, 1 = ancestral)", 0, 1, 0.05, st.eta, v => { st.eta = v; recompute(); });
  const spC = choiceRow(panel, "spacing", [{ v: "time" as Spacing, label: "uniform in time" }, { v: "logsnr" as Spacing, label: "uniform in log-SNR" }, { v: "data" as Spacing, label: "dense near data" }], st.spacing, v => { st.spacing = v; recompute(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const twoC = checkbox(row, "two-stage", st.twoStage, v => { st.twoStage = v; recompute(); });
  const lanS = choiceRow(row, "Langevin", [0, 1, 2, 3].map(v => ({ v, label: String(v) })), st.langevin, v => { st.langevin = v; recompute(); });
  const row2 = htmlEl("div", "qb-presets", "", panel);
  const badC = checkbox(row2, "bad start (noise × 0.2)", st.bad, v => { st.bad = v; recompute(); });
  const invC = checkbox(row2, "invert a data point", st.invert, v => { st.invert = v; drawInvert(); });
  button(row2, "new seed", () => { st.seed++; recompute(); });
  button(row2, "reset", () => apply(p0));
  const x0S = slider(panel, "data point x₀", -2.5, 2.5, 0.05, st.x0, v => { st.x0 = v; if (st.invert) drawInvert(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Background: the exact density of the noised data at every time. Blue: 30 sample paths. Right: histogram of 2 000 endpoints against the true density (black). With 'invert', the orange path runs from the data point back to noise and the dots mark where it starts and where the sampler brings it back.", panel);
  pl.svg.addEventListener("pointerdown", ev => { if (!st.invert) return; const [, y] = pl.toData(ev); st.x0 = Math.max(-2.5, Math.min(2.5, Math.round(y * 20) / 20)); x0S.set(st.x0); drawInvert(); });

  let tr = M.truth(M.data(st.data), NPTS), res: M.RunResult | null = null, invText = "";

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    stepS.set(uOf(st.steps)); etaS.set(st.eta); spC.set(st.spacing); twoC.checked = st.twoStage; lanS.set(st.langevin); invC.checked = st.invert; badC.checked = st.bad; x0S.set(st.x0); dataC.set(st.data);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    recompute();
  }

  function drawHeat(m: M.Mix1): void {
    const nx = 240, ny = 160, V = new Float64Array(nx * ny);
    let mx = 0;
    for (let c = 0; c < nx; c++) {
      const tau = 1 - (c + 0.5) / nx;   // column 0 at the left = τ near 1
      const s = 0.008, f = (t: number) => Math.cos((t + s) / (1 + s) * Math.PI / 2) ** 2, ab = Math.max(1e-4, f(tau) / f(0));
      for (let r = 0; r < ny; r++) { const y = YR[0] + (YR[1] - YR[0]) * (r + 0.5) / ny; const v = Math.sqrt(M.densityVP(m, ab, y)); V[r * nx + c] = v; if (v > mx) mx = v; }
    }
    // normalise each column, so the narrow data end does not wash out the rest
    for (let c = 0; c < nx; c++) { let cm = 0; for (let r = 0; r < ny; r++) cm = Math.max(cm, V[r * nx + c]); for (let r = 0; r < ny; r++) V[r * nx + c] /= cm || 1; }
    heat.remove();
    const h2 = pl.heatmap(V, nx, ny, { lo: 0, hi: 1, map: t => [1 - 0.55 * t, 1 - 0.4 * t, 1 - 0.08 * t], flipY: false, x0: 1, x1: 0, y0: YR[0], y1: YR[1] });
    // keep the heat map below the paths
    pl.layer.insertBefore(h2.g, pl.layer.firstChild);
    Object.assign(heat, h2);
    void mx;
  }

  function recompute(): void {
    const m = M.data(st.data);
    tr = M.truth(m, NPTS);
    drawHeat(m);
    res = M.run(m, { steps: st.steps, eta: st.eta, spacing: st.spacing, twoStage: st.twoStage, langevin: st.langevin, seed: st.seed, n: NPTS, nPaths: NPATHS, startScale: st.bad ? 0.2 : 1 });
    pathMarks.set(res.taus, res.paths);
    // endpoint histogram (density units) and true density
    const nb = 60, edges = Float64Array.from({ length: nb + 1 }, (_, i) => YR[0] + (YR[1] - YR[0]) * i / nb), cnt = new Float64Array(nb), h = (YR[1] - YR[0]) / nb;
    for (const v of res.ends) { const b = Math.floor((v - YR[0]) / h); if (b >= 0 && b < nb) cnt[b] += 1 / (NPTS * h); }
    const ys = Array.from({ length: 301 }, (_, i) => YR[0] + (YR[1] - YR[0]) * i / 300), ds = ys.map(y => M.densityVP(m, 1, y));
    side.setDomain([0, Math.min(3, Math.max(...ds, ...cnt) * 1.05)], YR);
    hist.set(edges, cnt);
    trueD.set(ds, ys);
    drawInvert();
  }

  function drawInvert(): void {
    const m = M.data(st.data);
    invText = "";
    if (st.invert) {
      const iv = M.invert(m, st.x0, st.steps, st.spacing);
      const sig = res!.sigmas;
      invPath.set(res!.taus, iv.path);
      invDots.set([0, 1, 0], [st.x0, iv.z, M.toVP(iv.back, 0)]);
      invText = `\ninvert: x₀ = ${fmt(st.x0, 2)} → noise ${fmt(iv.z, 2)} → back to ${fmt(iv.back, 2)}`;
      void sig;
    } else { invPath.set([], []); invDots.set([], []); }
    readout();
  }

  function readout(): void {
    if (!res) return;
    const m = M.data(st.data), sm = M.summary(m, res.ends, tr), t = M.mixMeanVar(m);
    ro.textContent =
      `network evaluations per sample: ${res.nfe}\n` +
      `endpoints: mean ${fmt(sm.mean, 2)}, sd ${fmt(sm.sd, 2)}   (data: ${fmt(t.mean, 2)}, ${fmt(Math.sqrt(t.variance), 2)})\n` +
      `off the data (> 3 sd from every mode): ${(100 * sm.off).toFixed(1)} %\n` +
      `Fréchet ${fmt(sm.frechet, 3)}   Wasserstein ${fmt(sm.w1, 3)}` + invText;
  }

  apply(p0);
}

autoMount(NAME, mount);
