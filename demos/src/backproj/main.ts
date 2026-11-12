// Back-projection and filtered back-projection (FBP), angle by angle. Left: the sinogram
// (measured, or after filtering) with the current angle marked. Middle: the filter's
// frequency response, and the current projection before and after filtering. Right: the
// reconstruction, built up by smearing each (filtered) projection back along its rays.
// Filters: none (plain back-projection), ramp, Shepp–Logan, cosine, Hann, with a cutoff;
// low dose = Poisson photon counts; "zero padding" off shows the wrap-around of circular
// convolution. Uses the projector pair of lib/radon.ts (back-projection = its transpose).
// Auto-mounts on <div data-demo="backproj">. Options: data-preset="star|glow|ramp|lowdose|hann|pointramp|nopad",
// data-filter="none|ramp|shepp-logan|cosine|hann" (start filter when no preset is given).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { diverging, grey } from "../lib/colour";
import * as R from "../lib/radon";
import { ObjectName, OBJECT_LABELS, objectImage, cellObject, imagePanel, sinoPanel, injectStyle, baseCss, layout, CELL_CREDIT, uniqueClip } from "../radonproj/view";

const NAME = "backproj";
const N = 128;
const DOSES = [Infinity, 1e6, 1e5, 1e4, 1e3, 300];
const FILTER_LABEL: Record<R.FilterKind, string> = { none: "none", ramp: "ramp", "shepp-logan": "Shepp–Logan", cosine: "cosine", hann: "Hann" };
const FLAT: Partial<Record<ObjectName, [number, number]>> = { shepp: [0, -28], disc: [10, 6], bigdisc: [0, 0] };

interface Settings { obj: ObjectName; filter: R.FilterKind; cutoff: number; nAngles: number; done: number; dose: number; pad: boolean; showFiltered: boolean }
interface Preset { id: string; label: string; s: Settings }
const base: Settings = { obj: "shepp", filter: "ramp", cutoff: 0.5, nAngles: 180, done: 180, dose: Infinity, pad: true, showFiltered: false };
const PRESETS: Preset[] = [
  { id: "star", label: "one point → star", s: { ...base, obj: "point", filter: "none", nAngles: 4, done: 4 } },
  { id: "glow", label: "plain: glow", s: { ...base, filter: "none" } },
  { id: "ramp", label: "ramp: sharp", s: { ...base } },
  { id: "pointramp", label: "one point: ramp cancels the star", s: { ...base, obj: "point", nAngles: 16, done: 16, showFiltered: true } },
  { id: "lowdose", label: "ramp, low dose: grain", s: { ...base, dose: 1e3 } },
  { id: "hann", label: "Hann: smoother, softer", s: { ...base, filter: "hann", dose: 1e3 } },
  { id: "nopad", label: "forgot to pad", s: { ...base, obj: "bigdisc", pad: false } },
];

const STYLE = baseCss(NAME) + `
[data-demo="${NAME}"] .bp-plots { flex: 1 1 230px; max-width: 320px; display: flex; flex-direction: column; gap: .1em; }
[data-demo="${NAME}"] .bp-plots figure { margin: 0; }
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const fAttr = host.dataset.filter as R.FilterKind | undefined;
  const pAttr = PRESETS.find(p => p.id === host.dataset.preset);
  const start: Settings = pAttr ? { ...pAttr.s } : fAttr && R.FILTERS.includes(fAttr) ? { ...base, filter: fAttr } : { ...base };
  const st: Settings = { ...start };
  let playing = false;

  // ---- layout ----
  const L = layout(host, 3);
  const sinoP = sinoPanel(L.figs, "sinogram", N / 2, [0, 180], "rd-fig wide", 300);
  const plots = htmlEl("div", "bp-plots", "", L.figs);
  const fFig = plotFigure(plots, "filter response", "");
  const fPlot = uniqueClip(new Plot(fFig, { font: 22, width: 320, height: 175, x: [0, 0.5], y: [0, 0.55], xticks: [0, 0.25, 0.5], yticks: [0, 0.25, 0.5], margin: { l: 62, b: 34 } }));
  const pFig = plotFigure(plots, "projection: before / after filter", "");
  const pPlot = uniqueClip(new Plot(pFig, { font: 22, width: 320, height: 175, x: [-64, 64], y: [-0.7, 1.1], xticks: [-64, 0, 64], yticks: [0, 1], margin: { l: 62, b: 34 } }));
  const recP = imagePanel(L.figs, "reconstruction", N, 0, "rd-fig");
  const credit = htmlEl("div", "rd-credit", CELL_CREDIT, recP.fig);
  const [colA, colB, colC] = L.cols;

  htmlEl("div", "rd-lab", "presets", colA);
  const presetBtns = presets(colA, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s, p) })));

  htmlEl("div", "rd-lab", "object", colB);
  const objNames: ObjectName[] = ["shepp", "point", "disc", "bigdisc", "cell"];
  const objRow = htmlEl("div", "qb-presets", "", colB);
  const objBtns = objNames.map(o => button(objRow, OBJECT_LABELS[o], () => { st.obj = o; full(); }));
  htmlEl("div", "rd-lab", "filter", colB);
  const fRow = htmlEl("div", "qb-presets", "", colB);
  const fBtns = R.FILTERS.map(f => button(fRow, FILTER_LABEL[f], () => { st.filter = f; refilter(); }));
  const cutS = slider(colB, "cutoff", 0.1, 0.5, 0.05, st.cutoff, v => { st.cutoff = v; refilter(); }, v => v.toFixed(2));
  const angS = slider(colB, "angles in the scan", 1, 360, 1, st.nAngles, v => { st.nAngles = v; st.done = v; full(); });

  const doneS = slider(colC, "back-projected", 0, 180, 1, st.done, v => { stop(); st.done = v; rebuild(); });
  const ctl = htmlEl("div", "rd-row", "", colC);
  const playBtn = button(ctl, "▶ play", () => (playing ? stop() : play()));
  button(ctl, "+1 angle", () => { stop(); addAngles(1); });
  button(ctl, "clear", () => { stop(); st.done = 0; rebuild(); });
  const doseS = slider(colC, "photons per ray", 0, DOSES.length - 1, 1, DOSES.indexOf(st.dose), v => { st.dose = DOSES[v]; renoise(); },
    v => (Number.isFinite(DOSES[v]) ? DOSES[v].toExponential(0).replace("e+", "e") : "∞"));
  const opts = htmlEl("div", "rd-row", "", colC);
  const padCb = checkbox(opts, "zero padding", st.pad, v => { st.pad = v; refilter(); });
  const showCb = checkbox(opts, "show filtered sinogram", st.showFiltered, v => { st.showFiltered = v; draw(); });
  button(opts, "reset", () => load(start));
  const ro = htmlEl("div", "qb-readout", "", colC);

  // ---- model ----
  let truth = new Float64Array(N * N) as Float64Array, geom = R.geometry(N, R.angleSet(180));
  let clean = new Float64Array(0) as Float64Array, meas = clean, filt = clean, rec = new Float64Array(N * N) as Float64Array;
  let starved = 0, truthMax = 1;
  const mask = R.fovMask(N);

  function full(): void {
    if (st.obj === "cell") { cellObject().then(() => full2()); return; }
    full2();
  }
  function full2(): void {
    truth = objectImage(st.obj, N);
    truthMax = Math.max(...truth);
    geom = R.geometry(N, R.angleSet(st.nAngles));
    clean = R.project(truth, geom);
    renoise();
  }
  function renoise(): void {
    const ns = R.noisySinogram(clean, st.dose, 7);
    meas = ns.sino; starved = ns.starved;
    refilter();
  }
  function refilter(): void {
    filt = R.filterSinogram(meas, st.nAngles, N, { kind: st.filter, cutoff: st.cutoff, pad: st.pad });
    rebuild();
  }
  /** Back-project the first st.done angles from scratch. */
  function rebuild(): void {
    st.done = Math.min(st.done, st.nAngles);
    rec = new Float64Array(N * N);
    const dth = Math.PI / st.nAngles;
    for (let a = 0; a < st.done; a++) R.backprojectAngle(filt, N, geom.angles[a], rec, N, 1, a * N, dth);
    sync(); draw();
  }
  function addAngles(k: number): void {
    const dth = Math.PI / st.nAngles;
    const to = Math.min(st.nAngles, st.done + k);
    for (let a = st.done; a < to; a++) R.backprojectAngle(filt, N, geom.angles[a], rec, N, 1, a * N, dth);
    st.done = to;
    sync(); draw();
  }

  function sync(): void {
    objBtns.forEach((b, i) => b.classList.toggle("active", objNames[i] === st.obj));
    fBtns.forEach((b, i) => b.classList.toggle("active", R.FILTERS[i] === st.filter));
    cutS.set(st.cutoff); angS.set(st.nAngles);
    doneS.input.max = String(st.nAngles); doneS.set(st.done);
    doseS.set(DOSES.indexOf(st.dose)); padCb.checked = st.pad; showCb.checked = st.showFiltered;
    credit.style.display = st.obj === "cell" ? "" : "none";
  }

  // overlays
  const curLine = sinoP.plot.vline(0, { color: "var(--qb-c4)", width: 2.5 });
  const respIdeal = fPlot.line([0], [0], { color: "var(--qb-muted)", width: 2, dash: "5 4" });
  const resp = fPlot.line([0], [0], { color: "var(--qb-c2)", width: 3 });
  const pRaw = pPlot.step([0], [0], { color: "var(--qb-muted)", width: 2 }, "mid");
  const pFilt = pPlot.step([0], [0], { color: "var(--qb-c2)", width: 2.5 }, "mid");
  pPlot.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" });

  function draw(): void {
    const na = st.nAngles, cur = Math.max(0, Math.min(na - 1, st.done - 1));
    // sinogram
    if (st.showFiltered && st.filter !== "none") {
      let m = 0; for (const v of filt) m = Math.max(m, Math.abs(v));
      sinoP.show(filt, na, N, 0, 180, na, -m, m, diverging);
    } else sinoP.show(meas, na, N, 0, 180, na, 0, undefined, grey);
    (sinoP.fig.querySelector("figcaption") as HTMLElement).textContent = (st.showFiltered && st.filter !== "none" ? "filtered sinogram" : "sinogram") + ` (${na} angles)`;
    curLine.set(st.done === 0 ? -10 : (cur + 0.5) * 180 / na);
    // filter response
    const L = R.filterLength(N, st.pad), H = R.filterResponse(st.filter, L, st.cutoff);
    const fs: number[] = [], hs: number[] = [];
    for (let k = 0; k <= L / 2; k++) { fs.push(k / L); hs.push(H[k]); }
    fPlot.opts.yticks = st.filter === "none" ? [0, 0.5, 1] : [0, 0.25, 0.5];
    fPlot.setDomain(undefined, [0, st.filter === "none" ? 1.1 : 0.55]);
    resp.set(fs, hs);
    respIdeal.set(fs, fs.map(f => (st.filter === "none" ? 1 : Math.abs(f))));
    (fFig.querySelector("figcaption") as HTMLElement).textContent = st.filter === "none" ? "filter: none (all 1)" : `filter response (dashed: |f|)`;
    // current projection, both scaled to max 1
    const rhos = Array.from({ length: N }, (_, b) => R.rhoOf(b, N));
    const raw = Array.from(meas.subarray(cur * N, cur * N + N)), fl = Array.from(filt.subarray(cur * N, cur * N + N));
    const m1 = Math.max(1e-12, ...raw.map(Math.abs)), m2 = Math.max(1e-12, ...fl.map(Math.abs));
    pRaw.set(rhos, raw.map(v => v / m1));
    pFilt.set(rhos, fl.map(v => v / m2));
    (pFig.querySelector("figcaption") as HTMLElement).textContent = `θ = ${(cur * 180 / na).toFixed(0)}°: grey before, red after (scaled)`;
    // reconstruction
    const autoC = st.filter === "none" || st.obj === "point" || st.done < st.nAngles;
    if (autoC) {
      let lo = Infinity, hi = -Infinity;
      for (let p = 0; p < rec.length; p++) if (mask[p] > 0.5) { lo = Math.min(lo, rec[p]); hi = Math.max(hi, rec[p]); }
      recP.show(rec, st.filter === "none" ? lo : Math.max(lo, -hi), hi);
    } else recP.show(rec, 0, truthMax);
    recP.cap.textContent = `reconstruction (${st.done} of ${na} angles)` + (autoC ? ", auto contrast" : "");
    // readout
    const rm = R.rmseOf(rec, truth, mask);
    let txt = `angles back-projected: ${st.done} / ${na}\n` + (st.filter === "none" ? "RMSE: – (plain back-projection is not normalised)" : `RMSE vs truth: ${rm.toFixed(3)}   PSNR: ${(20 * Math.log10(truthMax / rm)).toFixed(1)} dB`);
    const fl0 = FLAT[st.obj];
    if (fl0) {
      let s = 0, s2 = 0, k = 0;
      for (let y = fl0[1] - 4; y < fl0[1] + 4; y++) for (let x = fl0[0] - 4; x < fl0[0] + 4; x++) {
        const [c, r] = R.toPixel(x, y, N), v = rec[Math.round(r) * N + Math.round(c)];
        s += v; s2 += v * v; k++;
      }
      const mean = s / k;
      txt += `\nflat 8×8 region: mean ${mean.toFixed(3)}, std ${Math.sqrt(Math.max(0, s2 / k - mean * mean)).toFixed(3)}`;
    }
    if (starved) txt += `\nphoton starvation: ${starved} rays with 0 counts`;
    ro.textContent = txt;
  }

  // ---- play ----
  let timer = 0;
  function play(): void {
    if (st.done >= st.nAngles) { st.done = 0; rebuild(); }
    playing = true; playBtn.textContent = "❚❚ pause";
    const step = Math.max(1, Math.round(st.nAngles / 60));
    const tick = () => {
      if (!playing) return;
      addAngles(step);
      if (st.done >= st.nAngles) { stop(); return; }
      timer = window.setTimeout(tick, 60);
    };
    tick();
  }
  function stop(): void { playing = false; playBtn.textContent = "▶ play"; clearTimeout(timer); }

  function load(s: Settings, p?: Preset): void {
    stop();
    Object.assign(st, s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    full();
  }

  load(start, pAttr);
}

autoMount(NAME, mount);
