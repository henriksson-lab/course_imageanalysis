// How much data does a reconstruction need? One phantom, filtered back-projection (Hann),
// and four knobs: number of angles, angular range (a tilt series from -range/2 to +range/2,
// beam vertical at 0 degrees), detector bins (each bin integrates 1, 2 or 4 pixel widths)
// and photons per ray. Panels: reconstruction, error vs the truth, and the 2D spectrum of the
// reconstruction with the measured spokes drawn over it (a missing wedge = an empty sector).
// Auto-mounts on <div data-demo="tomosampling">. Options: data-preset="enough|streaks|wedge|bins|dose".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { diverging, grey } from "../lib/colour";
import * as R from "../lib/radon";
import { ObjectName, OBJECT_LABELS, objectImage, cellObject, imagePanel, injectStyle, baseCss, layout, drawHeat, CELL_CREDIT, uniqueClip } from "../radonproj/view";
import { centredSpectrum } from "../fourierslice/model";

const NAME = "tomosampling";
const N = 128;
const DOSES = [Infinity, 1e6, 1e5, 3e4, 1e4, 3e3, 1e3, 300];
const BINS = [128, 64, 32];

interface Settings { obj: ObjectName; nA: number; range: number; bins: number; dose: number; fixedTotal: boolean }
interface Preset { id: string; label: string; s: Settings }
const base: Settings = { obj: "shepp", nA: 200, range: 180, bins: 128, dose: Infinity, fixedTotal: false };
const PRESETS: Preset[] = [
  { id: "enough", label: "enough angles", s: { ...base } },
  { id: "streaks", label: "18 angles: streaks", s: { ...base, nA: 18 } },
  { id: "wedge", label: "±60°: missing wedge", s: { ...base, nA: 61, range: 120 } },
  { id: "lemon", label: "±60°: a disc becomes a lemon", s: { ...base, obj: "disc", nA: 61, range: 120 } },
  { id: "bins", label: "few bins: blur", s: { ...base, bins: 32 } },
  { id: "dose", label: "equal photons: many noisy views…", s: { ...base, nA: 180, dose: 1e3, fixedTotal: true } },
  { id: "dosefew", label: "…or few clean ones", s: { ...base, nA: 30, dose: 1e3, fixedTotal: true } },
];

const STYLE = baseCss(NAME);

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const pAttr = PRESETS.find(p => p.id === host.dataset.preset);
  const start: Settings = pAttr ? { ...pAttr.s } : { ...base };
  const st: Settings = { ...start };

  const L = layout(host, 3);
  const recP = imagePanel(L.figs, "reconstruction", N, 0, "rd-fig");
  const credit = htmlEl("div", "rd-credit", CELL_CREDIT, recP.fig);
  const errP = imagePanel(L.figs, "error", N, 0, "rd-fig");
  const specFig = plotFigure(L.figs, "spectrum + measured spokes", "rd-fig");
  const spec = uniqueClip(new Plot(specFig, { font: 22, width: 320, height: 320, x: [-0.5, 0.5], y: [-0.5, 0.5], equal: true, grid: false, xticks: [-0.5, 0, 0.5], yticks: [-0.5, 0, 0.5], margin: { l: 62, b: 34, t: 6, r: 12 } }));
  const [colA, colB, colC] = L.cols;

  htmlEl("div", "rd-lab", "presets", colA);
  const presetBtns = presets(colA, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s, p) })));
  htmlEl("div", "rd-lab", "object", colB);
  const objs: ObjectName[] = ["shepp", "disc", "cell"];
  const objRow = htmlEl("div", "qb-presets", "", colB);
  const objBtns = objs.map(o => button(objRow, OBJECT_LABELS[o], () => { st.obj = o; makeTruth(); }));
  const aS = slider(colB, "angles", 4, 360, 1, st.nA, v => { st.nA = v; recon(); });
  const rule = htmlEl("div", "qb-note", "", colB);
  const rS = slider(colB, "angular range", 90, 180, 2, st.range, v => { st.range = v; recon(); }, v => (v >= 180 ? "180°" : `±${v / 2}°`));
  const bS = slider(colB, "detector bins", 0, BINS.length - 1, 1, BINS.indexOf(st.bins), v => { st.bins = BINS[v]; recon(); }, v => String(BINS[v]));
  const dS = slider(colC, "photons per ray", 0, DOSES.length - 1, 1, DOSES.indexOf(st.dose), v => { st.dose = DOSES[v]; recon(); },
    v => (Number.isFinite(DOSES[v]) ? DOSES[v].toExponential(0).replace("e+", "e") : "∞"));
  const fixCb = checkbox(colC, "fixed total photons", st.fixedTotal, v => { st.fixedTotal = v; recon(); });
  button(htmlEl("div", "rd-row", "", colC), "reset", () => load(start));
  const ro = htmlEl("div", "qb-readout", "", colC);

  let truth = new Float64Array(N * N) as Float64Array, tmax = 1, fine = new Float64Array(0) as Float64Array;
  const mask = R.fovMask(N);
  let specVals: ArrayLike<number> = new Float64Array(N * N);
  spec.mark(g => drawHeat(spec, g, specVals, N, N, -0.5, 0.5, -0.5, 0.5, 0, undefined, grey), []);
  let spokeMarks: { remove(): void }[] = [];

  function makeTruth(): void {
    if (st.obj === "cell") { cellObject().then(() => { truth = objectImage("cell", N); tmax = Math.max(...truth); recon(); }); return; }
    truth = objectImage(st.obj, N); tmax = Math.max(...truth); recon();
  }

  function angles(): Float64Array {
    return st.range >= 180 ? R.angleSet(st.nA) : R.angleRange(-st.range / 2, st.range / 2, st.nA);
  }

  function recon(): void {
    const ang = angles(), g = R.geometry(N, ang);
    fine = R.project(truth, g);
    const I0 = st.fixedTotal && Number.isFinite(st.dose) ? st.dose * 180 / st.nA : st.dose;
    const ns = R.noisySinogram(fine, I0, 11);
    // detector bins of k pixel widths: average k neighbouring rays
    const k = N / st.bins, nB = st.bins, coarse = new Float64Array(ang.length * nB);
    for (let a = 0; a < ang.length; a++) for (let j = 0; j < nB; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += ns.sino[a * N + j * k + t];
      coarse[a * nB + j] = s / k;
    }
    const filt = R.filterSinogram(coarse, ang.length, nB, { kind: "hann" }, k);
    // back to pixel-spaced rays by linear interpolation, then back-project
    const up = new Float64Array(ang.length * N);
    for (let a = 0; a < ang.length; a++) for (let b = 0; b < N; b++) {
      const u = (b - (k - 1) / 2) / k, j0 = Math.floor(u), f = u - j0;
      const at = (j: number) => filt[a * nB + Math.max(0, Math.min(nB - 1, j))];
      up[a * N + b] = (1 - f) * at(j0) + f * at(j0 + 1);
    }
    const rec = R.backproject(up, g, R.angleStep(ang));
    recP.show(rec, 0, tmax);
    recP.cap.textContent = `FBP (Hann): ${st.nA} angles, ${st.range >= 180 ? "180°" : "±" + st.range / 2 + "°"}`;
    const err = rec.map((v, i) => (v - truth[i]) * mask[i]);
    errP.show(err, -0.5 * tmax, 0.5 * tmax, diverging);
    errP.cap.textContent = "error (reconstruction − truth)";
    // spectrum of the reconstruction and the spokes
    const S = centredSpectrum(rec, N, N), V = new Float64Array(N * N);
    for (let kk = 0; kk < N; kk++) for (let i = 0; i < N; i++) V[(N - 1 - kk) * N + i] = Math.log10(1 + Math.hypot(S.re[kk * N + i], S.im[kk * N + i]));
    specVals = V; spec.renderAll();
    spokeMarks.forEach(m => m.remove()); spokeMarks = [];
    const every = Math.max(1, Math.ceil(ang.length / 60));
    for (let a = 0; a < ang.length; a += every) {
      const c = Math.cos(ang[a]), s = Math.sin(ang[a]);
      spokeMarks.push(spec.line([-0.5 * c, 0.5 * c], [-0.5 * s, 0.5 * s], { color: "var(--qb-c4)", width: 1.2, opacity: 0.55 }));
    }
    // readout
    const rm = R.rmseOf(rec, truth, mask);
    rule.textContent = `angle rule for ${nB} bins: π/2 · ${nB} ≈ ${R.angleRule(nB)} angles over 180°`;
    ro.textContent = `RMSE ${rm.toFixed(3)}   PSNR ${(20 * Math.log10(tmax / rm)).toFixed(1)} dB\nphotons per ray: ${Number.isFinite(I0) ? Math.round(I0).toLocaleString("en") : "∞"}` +
      (ns.starved ? `\nphoton starvation: ${ns.starved} rays` : "") + `\nspokes drawn: ${Math.ceil(ang.length / every)} of ${ang.length}`;
    sync();
  }

  function sync(): void {
    objBtns.forEach((b, i) => b.classList.toggle("active", objs[i] === st.obj));
    aS.set(st.nA); rS.set(st.range); bS.set(BINS.indexOf(st.bins)); dS.set(DOSES.indexOf(st.dose)); fixCb.checked = st.fixedTotal;
    credit.style.display = st.obj === "cell" ? "" : "none";
  }

  function load(s: Settings, p?: Preset): void {
    Object.assign(st, s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeTruth();
  }
  load(start, pAttr);
}

autoMount(NAME, mount);
