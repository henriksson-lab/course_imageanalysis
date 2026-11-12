// Richardson–Lucy step by step. Snapshots at about 30 log-spaced iterations are computed in
// the background after every change; the iteration slider scrubs through them.
// Auto-mounts on <div data-demo="rlsteps">. Options: data-preset="<id>", data-presets="id,...",
//   data-mode="2d" (default): photon-count image, error and misfit against the iteration,
//     PSF too narrow / too wide, a forgotten background, blind RL with PSF constraints;
//   data-mode="xz": a 3D stack (64 × 32 × 32), widefield or confocal 3D PSF, xz and xy slices,
//     the xz OTF with the missing cone, axial profiles;
//   data-mode="measure": a field with known truth (bead pairs, spots of flux 1:2:4, disks),
//     Wiener / Tikhonov / RL, and a table of measurements raw / restored / truth.
import { htmlEl, presets as presetRow, button, checkbox, autoMount, slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { diverging, viridis } from "../lib/colour";
import { rmse } from "../lib/metrics";
import * as P from "../lib/psf";
import * as D from "../lib/deconv";
import { phantom, star, beads, render, makePSF, blur, PsfSpec, BEADS, SceneFn } from "../deconvlab/scenes";
import { injectStyle, baseStyle, select, logSlider, imagePanel, Jobs, fmtSci, legendRow, ImagePanel } from "../deconvlab/widgets";
import * as M from "./model";

const NAME = "rlsteps";
const N = 128;

const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .rl-imgs { flex: 1 1 360px; max-width: 420px; display: grid; grid-template-columns: 1fr 1fr; gap: .4em .5em; }
[data-demo="${NAME}"] .rl-imgs3 { flex: 1 1 420px; max-width: 470px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .3em .4em; }
[data-demo="${NAME}"] .rl-plots { flex: 1 1 330px; max-width: 420px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .rl-free canvas { aspect-ratio: auto; }
[data-demo="${NAME}"] table.rl-table { border-collapse: collapse; font-size: 15px; width: 100%; }
[data-demo="${NAME}"] table.rl-table td, [data-demo="${NAME}"] table.rl-table th { border: 1px solid var(--qb-grid); padding: .15em .35em; text-align: right; }
[data-demo="${NAME}"] table.rl-table td:first-child, [data-demo="${NAME}"] table.rl-table th:first-child { text-align: left; }
[data-demo="${NAME}"] td.ok { background: #d9f2d9; } [data-demo="${NAME}"] td.mid { background: #fff1c2; } [data-demo="${NAME}"] td.bad { background: #f8d0cc; }
.reveal [data-demo="${NAME}"] .rl-imgs { flex: 0 1 330px; }
.reveal [data-demo="${NAME}"] .rl-imgs3 { flex: 0 1 400px; }
.reveal [data-demo="${NAME}"] .rl-plots { flex: 0 1 400px; max-width: 400px; }
`;

function panelHide(el: HTMLElement | null | undefined, hide: boolean): void { if (el) el.style.display = hide ? "none" : ""; }
const rowOf = (e: HTMLElement) => e.closest("label") as HTMLElement;

// =======================================================================================
// 2D mode

type Obj2 = "beads" | "star" | "phantom";
type Blind = "off" | "none" | "size" | "gauss";
interface S2 { obj: Obj2; psf: PsfSpec; photons: number; noiseFree: boolean; bg: number; bgModel: boolean; used: number; blind: Blind; kIdx: number }
const D2: S2 = { obj: "beads", psf: { kind: "gauss", width: 2 }, photons: 100, noiseFree: false, bg: 0, bgModel: true, used: 1, blind: "off", kIdx: 10 };
interface Preset<S> { id: string; label: string; s: Partial<S>; k?: number }
const P2: Preset<S2>[] = [
  { id: "nonoise", label: "no noise: keeps improving", s: { noiseFree: true, photons: 1000 }, k: 1000 },
  { id: "semi", label: "noise: semi-convergence", s: { photons: 100 }, k: 10 },
  { id: "fewer", label: "fewer photons: stop earlier", s: { photons: 20 }, k: 3 },
  { id: "speckle", label: "1000 iterations: speckles", s: { obj: "phantom", photons: 100 }, k: 1000 },
  { id: "ringing", label: "ringing at the edge", s: { photons: 10000, psf: { kind: "airy", width: 1.0 } }, k: 1000 },
  { id: "background", label: "forgot the background", s: { obj: "phantom", photons: 100, bg: 30, bgModel: false }, k: 100 },
  { id: "wide", label: "PSF too wide: halos", s: { photons: 1000, used: 1.3 }, k: 100 },
  { id: "narrow", label: "PSF too narrow: still blurred", s: { photons: 1000, used: 0.7 }, k: 100 },
  { id: "blind", label: "blind: the PSF shrinks to a point", s: { photons: 1000, blind: "none", used: 1.5 }, k: 300 },
  { id: "blindgauss", label: "blind with a Gaussian-shape constraint", s: { photons: 1000, blind: "gauss", used: 1.5 }, k: 300 },
  { id: "wiener", label: "RL vs best Wiener", s: { obj: "phantom", photons: 1000 }, k: 25 },
];
const BLIND_MAX = 300, MAX2 = 1000;

function mount2d(host: HTMLElement): void {
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? P2.filter(p => only.includes(p.id)) : P2;
  const p0 = P2.find(p => p.id === host.dataset.preset) ?? P2[1];
  const st: S2 = { ...D2, psf: { ...D2.psf } };
  let run: M.RLRun | null = null;
  const iters = () => run?.iters ?? M.snapshotIters(st.blind === "off" ? MAX2 : BLIND_MAX);

  const imgs = htmlEl("div", "rl-imgs", "", host);
  const pObs = imagePanel(imgs, "observed (photons)"), pEst = imagePanel(imgs, "RL estimate");
  const pErr = imagePanel(imgs, "error: estimate − truth"), pPsf = imagePanel(imgs, "PSF: true | used", "rl-free");
  const plots = htmlEl("div", "rl-plots", "", host);
  const prof = new Plot(plots, { width: 420, height: 240, x: [0, N], y: [0, 3], xlabel: "x along row 24 (px)", ylabel: "value", font: 19 });
  const lT = prof.line([], [], { color: "var(--qb-muted, #777)", dash: "7 5" }), lO = prof.line([], [], { color: seriesColor(0), width: 2 }), lE = prof.line([], [], { color: seriesColor(1) });
  legendRow(plots, [{ label: "truth", color: "var(--qb-muted, #777)", dash: true }, { label: "observed", color: seriesColor(0) }, { label: "RL", color: seriesColor(1) }]);
  const ek = new Plot(plots, { width: 420, height: 240, x: [1, MAX2], y: [0.5, 100], xlog: true, ylog: true, xlabel: "iteration k", ylabel: "", font: 19, xticks: [1, 10, 100, 1000] });
  const lErr = ek.line([], [], { color: seriesColor(3) }), lMis = ek.line([], [], { color: seriesColor(0) });
  const lWie = ek.hline(1, { color: seriesColor(2), dash: "6 4", width: 2 }), lNow = ek.vline(1, { color: seriesColor(1), width: 2.5 });
  legendRow(plots, [{ label: "error (% of peak)", color: seriesColor(3) }, { label: "misfit (deviance/px)", color: seriesColor(0) }, { label: "best Wiener", color: seriesColor(2), dash: true }]);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const objS = select<Obj2>(panel, "object", [["beads", "beads (1 : 4 pair) + edge"], ["star", "Siemens star"], ["phantom", "cell phantom"]], st.obj, v => { st.obj = v; remake(); });
  const psfS = select<"gauss" | "airy">(panel, "PSF", [["gauss", "Gaussian σ = 2 px"], ["airy", "Airy, NA 1.0"]], st.psf.kind as "gauss", v => { st.psf = { kind: v, width: v === "gauss" ? 2 : 1.0 }; remake(); });
  const phS = logSlider(panel, "photons at 1", 10, 1e5, st.photons, v => { st.photons = Math.round(v); remake(); }, v => fmtSci(Math.round(v)));
  const nfC = checkbox(panel, "no noise", st.noiseFree, v => { st.noiseFree = v; remake(); });
  const bgS = slider(panel, "background", 0, 50, 1, st.bg, v => { st.bg = v; remake(); });
  const bgC = checkbox(panel, "background in the model", st.bgModel, v => { st.bgModel = v; rerun(); });
  const usedS = slider(panel, "PSF used ×", 0.6, 1.6, 0.05, st.used, v => { st.used = v; rerun(); }, v => v.toFixed(2));
  const blS = select<Blind>(panel, "blind", [["off", "off (PSF given)"], ["none", "blind, no constraint"], ["size", "blind, size limit (r ≤ 6 px)"], ["gauss", "blind, Gaussian shape"]], st.blind, v => { st.blind = v; rerun(); });
  const kS = slider(panel, "iteration k", 0, 1, 1, 0, v => { st.kIdx = v; show(); }, v => String(iters()[Math.min(v, iters().length - 1)] ?? 0));
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Error: RMS of estimate − truth in % of the largest true value. Misfit: Poisson deviance per pixel, about 1 when the blurred estimate fits the data to the noise level. Blind: the guess is the true PSF widened by 'PSF used ×'.", panel);

  let pendingK: number | undefined;
  function load(p: Preset<S2>): void {
    Object.assign(st, { ...D2, psf: { ...D2.psf } }, JSON.parse(JSON.stringify(p.s)));
    pendingK = p.k;
    pbtns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    objS.set(st.obj); psfS.set(st.psf.kind as "gauss"); phS.setValue(st.photons); nfC.checked = st.noiseFree; bgS.set(st.bg); bgC.checked = st.bgModel; usedS.set(st.used); blS.set(st.blind);
    remake();
  }

  let truth: Float64Array = new Float64Array(N * N), y: Float64Array = new Float64Array(N * N), psfTrue: Float64Array = new Float64Array(N * N), hiT = 1;
  let wienerBest = NaN;
  function sceneOf(): SceneFn { return st.obj === "beads" ? beads(N) : st.obj === "star" ? star(N) : phantom(N); }
  function remake(): void {
    truth = render(sceneOf(), N);
    hiT = Math.max(...truth);
    psfTrue = makePSF(N, st.psf);
    y = M.photonData(blur(truth, psfTrue, N), st.photons, st.bg, 5, st.noiseFree);
    rerun();
  }
  // a run: plain RL (snapshots) or blind (object + PSF snapshots)
  let blindSnaps: { x: Float64Array; h: Float64Array }[] = [];
  let curve: { k: number; err: number; mis: number }[] = [];
  const jobs = new Jobs();
  function rerun(): void {
    const b = st.bgModel ? st.bg : 0, used = makePSF(N, st.psf, 1, st.used);
    curve = []; blindSnaps = [];
    // best Wiener (flat spectrum) on data − background, over 25 SNR values
    const ctx = D.gainContext(D.spectrum2d(P.toOrigin(used, N, N), N, N), N, N), G = D.spectrum2d(y.map(v => v - st.bg), N, N);
    wienerBest = Infinity;
    for (let i = 0; i < 25; i++) {
      const e = rmse(D.restoreSpectrum(G, { kind: "wiener", snr: 10 ** (i / 4), model: "flat" }, ctx).map(v => v / st.photons), truth);
      wienerBest = Math.min(wienerBest, e);
    }
    const A = D.fftOperator2d(used, N, N);
    run = new M.RLRun(y, A, b, st.blind === "off" ? MAX2 : BLIND_MAX);
    if (st.blind !== "off") blindSnaps.push({ x: run.x.slice(), h: used.slice() });
    kS.input.max = String(run.iters.length - 1);
    if (pendingK !== undefined) { st.kIdx = run.iters.findIndex(k => k >= pendingK!); pendingK = undefined; }
    st.kIdx = Math.min(st.kIdx, run.iters.length - 1);
    kS.set(st.kIdx);
    record(0, run.x);
    const r = run;
    let h: Float64Array = used.slice(), x: Float64Array = r.x;
    const sigmaOf = (hh: Float64Array) => { let s = 0; for (let yy = 0; yy < N; yy++) for (let xx = 0; xx < N; xx++) s += hh[yy * N + xx] * ((xx - N / 2) ** 2 + (yy - N / 2) ** 2); return Math.sqrt(s / 2); };
    const gen = function* () {
      while (!r.done) {
        if (st.blind === "off") { r.step(); if (r.snaps.length > curve.length) record(r.k, r.snaps[r.snaps.length - 1]); }
        else {
          ({ x, psf: h } = D.blindStep(x, h, y, N, N, { background: b, supportRadius: st.blind === "size" ? 6 : undefined }));
          if (st.blind === "gauss") h = P.gaussianPSF(N, N, sigmaOf(h));
          r.k++;
          if (r.k === r.iters[r.snaps.length]) { r.snaps.push(x.slice()); blindSnaps.push({ x: x.slice(), h: h.slice() }); record(r.k, x); }
        }
        yield;
      }
    };
    jobs.run(gen(), () => show(), 14, 120);
    show();
  }
  function record(k: number, x: Float64Array): void {
    const b = st.bgModel ? st.bg : 0;
    const err = 100 * rmse(x.map(v => v / st.photons), truth) / hiT;
    const used = st.blind === "off" ? null : blindSnaps[blindSnaps.length - 1]?.h;
    const A = used ? D.fftOperator2d(used, N, N) : run!.A;
    const m = A.forward(x).map(v => v + b);
    curve.push({ k, err, mis: M.deviance(y, m) });
  }
  function show(): void {
    if (!run) return;
    const i = Math.min(st.kIdx, run.snaps.length - 1), k = run.iters[i], x = run.snaps[i];
    const est = x.map(v => v / st.photons), obs = y.map(v => (v - st.bg) / st.photons);
    const hiD = Math.min(hiT, 1);
    pObs.draw(obs, N, N, { lo: 0, hi: hiD }); pEst.draw(est, N, N, { lo: 0, hi: hiD });
    pErr.draw(est.map((v, j) => v - truth[j]), N, N, { lo: -0.3 * hiT, hi: 0.3 * hiT, map: diverging });
    // PSF panel: true | used (or current blind estimate), central 25 × 25 each
    const c = 12, w = 2 * (2 * c + 1) + 1, img = new Float64Array(w * (2 * c + 1)).fill(NaN);
    const usedNow = st.blind === "off" ? makePSF(N, st.psf, 1, st.used) : blindSnaps[Math.min(i, blindSnaps.length - 1)].h;
    const pk = Math.max(...psfTrue);
    for (let yy = -c; yy <= c; yy++) for (let xx = -c; xx <= c; xx++) {
      img[(yy + c) * w + xx + c] = psfTrue[(yy + N / 2) * N + xx + N / 2] / pk;
      img[(yy + c) * w + xx + c + 2 * c + 2] = usedNow[(yy + N / 2) * N + xx + N / 2] / pk;
    }
    pPsf.draw(img, w, 2 * c + 1, { lo: 0, hi: 1, map: viridis });
    pPsf.cap.textContent = st.blind === "off" ? "PSF: true | used" : `PSF: true | blind estimate (k = ${k})`;
    const xs = Array.from({ length: N }, (_, j) => j), row = 24;
    lT.set(xs, xs.map(j => truth[row * N + j])); lO.set(xs, xs.map(j => obs[row * N + j])); lE.set(xs, xs.map(j => est[row * N + j]));
    prof.setDomain(undefined, [Math.min(0, ...xs.map(j => est[row * N + j])) - 0.05, 1.15 * Math.max(hiT, ...xs.map(j => est[row * N + j]))]);
    lErr.set(curve.map(q => Math.max(1, q.k)), curve.map(q => q.err));
    lMis.set(curve.map(q => Math.max(1, q.k)), curve.map(q => q.mis));
    lWie.set(100 * wienerBest / hiT);
    lNow.set(Math.max(1, k));
    const all = curve.flatMap(q => [q.err, q.mis]).concat([100 * wienerBest / hiT]).filter(v => v > 0 && Number.isFinite(v));
    if (all.length) ek.setDomain([1, run.iters[run.iters.length - 1]], [Math.min(...all) / 1.5, Math.max(...all) * 1.5]);
    const cur = curve[Math.min(i, curve.length - 1)];
    const ib = curve.reduce((b, q, j) => (q.err < curve[b].err ? j : b), 0);
    const lines = [`k = ${k}: error ${cur.err.toFixed(2)} %, misfit ${cur.mis.toFixed(2)}`,
      `best so far: k = ${curve[ib].k} (${curve[ib].err.toFixed(2)} %); best Wiener ${(100 * wienerBest / hiT).toFixed(2)} %`];
    const flux = x.reduce((a, v) => a + v, 0), fluxData = y.reduce((a, v) => a + v, 0) - (st.bgModel ? st.bg : 0) * N * N;
    lines.push(`total flux: estimate ${fmtSci(flux)}, data − background ${fmtSci(fluxData)}`);
    if (st.obj === "beads") {
      const [a, bb] = [BEADS[0], BEADS[1]];
      const pk2 = (cx: number, cy: number) => { let m = -Infinity; for (let yy = cy - 1; yy <= cy + 1; yy++) for (let xx = cx - 1; xx <= cx + 1; xx++) m = Math.max(m, est[yy * N + xx]); return m - 0.05; };
      const integ = (cx: number, cy: number) => { let s = 0; for (let yy = cy - 3; yy <= cy + 3; yy++) for (let xx = cx - 3; xx <= cx + 3; xx++) if ((xx - cx) ** 2 + (yy - cy) ** 2 <= 6.25) s += est[yy * N + xx] - 0.05; return s; };
      lines.push(`bead pair (truth 4.0): peak ratio ${(pk2(a[0], a[1]) / pk2(bb[0], bb[1])).toFixed(2)}, integrated ${(integ(a[0], a[1]) / integ(bb[0], bb[1])).toFixed(2)}`);
    }
    if (st.blind !== "off") lines.push(`blind PSF FWHM ${P.fwhm(usedNow.subarray((N / 2) * N, (N / 2 + 1) * N)).toFixed(2)} px (true ${P.fwhm(psfTrue.subarray((N / 2) * N, (N / 2 + 1) * N)).toFixed(2)})`);
    if (!run.done) lines.push(`computing… iteration ${run.k} of ${run.iters[run.iters.length - 1]}`);
    ro.textContent = lines.join("\n");
    kS.set(st.kIdx);
  }
  load(p0);
}

// =======================================================================================
// xz mode: a 3D stack

type Scope = "widefield" | "confocal";
interface S3 { scope: Scope; zstep: number; photons: number; kIdx: number; prof: "bead" | "sheet" | "filament" }
const D3: S3 = { scope: "widefield", zstep: 200, photons: 1000, kIdx: 0, prof: "bead" };
const P3: Preset<S3>[] = [
  { id: "wf", label: "widefield: blur from above and below", s: {}, k: 0 },
  { id: "bead", label: "RL restores the bead in z", s: { prof: "bead" }, k: 100 },
  { id: "sheet", label: "the sheet stays smeared: missing cone", s: { prof: "sheet" }, k: 100 },
  { id: "confocal", label: "confocal: less to gain, nothing missing", s: { scope: "confocal", prof: "sheet" }, k: 100 },
  { id: "zstep", label: "z-steps too large", s: { zstep: 800, prof: "bead" }, k: 100 },
];
const MAX3 = 200;

function mountXZ(host: HTMLElement): void {
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? P3.filter(p => only.includes(p.id)) : P3;
  const p0 = P3.find(p => p.id === host.dataset.preset) ?? P3[1];
  const st: S3 = { ...D3 };
  let run: M.RLRun | null = null;
  const NX = M.NX, NY = M.NY, NZ = M.NZ;

  const imgs = htmlEl("div", "rl-imgs3", "", host);
  const mk = (cap: string) => { const p = imagePanel(imgs, cap, "rl-free"); return p; };
  const xzT = mk("truth, xz"), xzO = mk("observed, xz"), xzR = mk("RL, xz");
  const xyT = mk("truth, xy"), xyO = mk("observed, xy"), xyR = mk("RL, xy");
  const plots = htmlEl("div", "rl-plots", "", host);
  const otfP = imagePanel(plots, "OTF |H(kx, kz)| (log), missing cone dashed", "rl-free");
  const otfSvgHost = otfP.fig;
  otfSvgHost.style.position = "relative";
  const ap = new Plot(plots, { width: 420, height: 230, x: [0, 1], y: [0, 1], xlabel: "z (µm)", ylabel: "÷ own maximum", font: 19 });
  const aT = ap.line([], [], { color: "var(--qb-muted, #777)", dash: "7 5" }), aO = ap.line([], [], { color: seriesColor(0) }), aR = ap.line([], [], { color: seriesColor(1) });
  legendRow(plots, [{ label: "truth", color: "var(--qb-muted, #777)", dash: true }, { label: "blurred (no noise)", color: seriesColor(0) }, { label: "RL", color: seriesColor(1) }]);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const scS = select<Scope>(panel, "microscope", [["widefield", "widefield"], ["confocal", "confocal (point pinhole)"]], st.scope, v => { st.scope = v; remake(); });
  const zS = select<"200" | "400" | "800">(panel, "z-step", [["200", "200 nm"], ["400", "400 nm"], ["800", "800 nm"]], "200", v => { st.zstep = Number(v); remake(); });
  const phS = logSlider(panel, "photons at 1", 10, 1e4, st.photons, v => { st.photons = Math.round(v); remake(); }, v => fmtSci(Math.round(v)));
  const prS = select<S3["prof"]>(panel, "axial profile", [["bead", "through the bead"], ["sheet", "through the sheet"], ["filament", "along the filament"]], st.prof, v => { st.prof = v; show(); });
  const kS = slider(panel, "iteration k", 0, 1, 1, 0, v => { st.kIdx = v; show(); }, v => String(run ? run.iters[Math.min(v, run.iters.length - 1)] : 0));
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", `Volume 64 × 32 × 32 voxels, 100 nm across, NA ${M.OPT3.NA} in air, λ ${M.OPT3.wavelength} nm. Objects: two beads (left), a vertical filament, a horizontal sheet (right, 200 nm thick) and a hollow sphere. The xz views are drawn to scale.`, panel);

  let pendingK: number | undefined;
  function load(p: Preset<S3>): void {
    Object.assign(st, D3, p.s);
    pendingK = p.k;
    pbtns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    scS.set(st.scope); zS.set(String(st.zstep) as "200"); phS.setValue(st.photons); prS.set(st.prof);
    remake();
  }
  let truth: Float64Array = new Float64Array(0), y: Float64Array = new Float64Array(0), clean3: Float64Array = new Float64Array(0), otfImg: Float64Array = new Float64Array(0);
  const jobs = new Jobs();
  function remake(): void {
    truth = M.volume3d(st.zstep);
    const psf = M.psf3d(st.scope, st.zstep), A = D.fftOperator3d(psf, NX, NY, NZ);
    clean3 = A.forward(truth);
    y = M.photonData(clean3, st.photons, 0, 9);
    // OTF magnitude at ky = 0, shifted: rows = kz, columns = kx
    otfImg = new Float64Array(NX * NZ);
    const H = A.H;
    let mx = 0;
    for (let kz = 0; kz < NZ; kz++) for (let kx = 0; kx < NX; kx++) {
      const i = (kz * NY) * NX + kx, v = Math.hypot(H.re[i], H.im[i]);
      otfImg[((kz + NZ / 2) % NZ) * NX + (kx + NX / 2) % NX] = v; mx = Math.max(mx, v);
    }
    otfImg = otfImg.map(v => Math.log10(Math.max(v / mx, 1e-3)));
    run = new M.RLRun(y, A, 0, MAX3, 8);
    kS.input.max = String(run.iters.length - 1);
    if (pendingK !== undefined) { st.kIdx = run.iters.findIndex(k => k >= pendingK!); pendingK = undefined; }
    st.kIdx = Math.max(0, Math.min(st.kIdx, run.iters.length - 1));
    kS.set(st.kIdx);
    const r = run;
    jobs.run((function* () { while (!r.done) { r.step(); yield; } })(), () => show(), 14, 60);
    show();
  }
  const cx = NX / 2, cy = NY / 2, cz = NZ / 2;
  const beadX = Math.round(M.OBJ3.beads[0][0] / M.PX3 + cx), sheetX = Math.round(800 / M.PX3 + cx), filX = Math.round(M.OBJ3.filamentX / M.PX3 + cx);
  function show(): void {
    if (!run) return;
    const i = Math.min(st.kIdx, run.snaps.length - 1), k = run.iters[i], x = run.snaps[i];
    const sc = (v: Float64Array) => v.map(a => a / st.photons);
    const est = sc(x), obs = sc(y);
    const aspect = `${NX * M.PX3} / ${NZ * st.zstep}`;
    const hiV = 1.2, hiO = Math.max(...xzSliceOf(obs)) * 1.05, hiR = Math.max(0.3, Math.min(8, Math.max(...xzSliceOf(est))));
    for (const [p, v, hi] of [[xzT, truth, hiV], [xzO, obs, hiO], [xzR, est, hiR]] as [ImagePanel, Float64Array, number][]) {
      p.draw(xzSliceOf(v), NX, NZ, { lo: 0, hi }); p.canvas.style.aspectRatio = aspect;
    }
    for (const [p, v, hi] of [[xyT, truth, hiV], [xyO, obs, hiO], [xyR, est, hiR]] as [ImagePanel, Float64Array, number][]) {
      p.draw(M.xy(v, cz), NX, NY, { lo: 0, hi }); p.canvas.style.aspectRatio = `${NX} / ${NY}`;
    }
    otfP.draw(otfImg, NX, NZ, { lo: -3, hi: 0, map: viridis });
    otfP.canvas.style.aspectRatio = `${1 / M.PX3} / ${1 / st.zstep}`;
    drawCone();
    // axial profile
    const px = st.prof === "bead" ? beadX : st.prof === "sheet" ? sheetX : filX;
    const zs = Array.from({ length: NZ }, (_, z) => (z - cz) * st.zstep / 1000);
    const at = (v: Float64Array, z: number) => v[(z * NY + cy) * NX + px];
    // each profile scaled to its own maximum, so the widths can be compared
    const norm = (v: number[]) => { const m = Math.max(...v) || 1; return v.map(a => a / m); };
    const tv = norm(zs.map((_, z) => at(truth, z))), ov = norm(zs.map((_, z) => at(clean3, z))), rv = norm(zs.map((_, z) => at(est, z)));
    aT.set(zs, tv); aO.set(zs, ov); aR.set(zs, rv);
    ap.setDomain([zs[0], zs[NZ - 1]], [0, 1.1]);
    const fw = (v: number[]) => P.fwhm(v) * st.zstep;
    const bz = (v: Float64Array) => zs.map((_, z) => v[(z * NY + cy) * NX + beadX]);
    const sz = (v: Float64Array) => zs.map((_, z) => v[(z * NY + cy) * NX + sheetX]);
    const lines = [`k = ${k}${run.done ? "" : `   (computing… ${run.k} of ${MAX3})`}`,
      `axial FWHM of the bead: blurred ${fw(bz(clean3)).toFixed(0)} nm, RL ${fw(bz(est)).toFixed(0)} nm`,
      `axial FWHM of the sheet: truth ${fw(sz(truth)).toFixed(0)} nm, blurred ${fw(sz(clean3)).toFixed(0)} nm, RL ${fw(sz(est)).toFixed(0)} nm`,
      "(blurred: the stack before noise, where a width can be measured)"];
    ro.textContent = lines.join("\n");
    kS.set(st.kIdx);
  }
  function xzSliceOf(v: Float64Array): Float64Array { return M.xz(v, cy); }
  // missing cone: lines at the aperture angle α from the kx axis, through the origin
  let coneSvg: SVGSVGElement | null = null;
  function drawCone(): void {
    if (!coneSvg) {
      coneSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      Object.assign(coneSvg.style, { position: "absolute", left: "0", width: "100%", pointerEvents: "none" });
      otfSvgHost.appendChild(coneSvg);
    }
    const c = otfP.canvas;
    coneSvg.style.top = c.offsetTop + "px"; coneSvg.style.height = c.offsetHeight + "px";
    coneSvg.setAttribute("viewBox", `0 0 ${NX} ${NZ}`); coneSvg.setAttribute("preserveAspectRatio", "none");
    const alpha = Math.asin(M.OPT3.NA / M.OPT3.ri);
    // in index units: kx index per cycle/nm = NX·PX3, kz index per cycle/nm = NZ·zstep
    const kr = 0.6 * M.OPT3.NA / M.OPT3.wavelength;                // draw up to 0.6 × NA/λ
    const dx = kr * NX * M.PX3, dz = kr * Math.tan(alpha) * NZ * st.zstep;
    const x0 = NX / 2 + 0.5, z0 = NZ / 2 + 0.5;
    coneSvg.innerHTML = st.scope === "widefield"
      ? [[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([a, b]) => `<line x1="${x0}" y1="${z0}" x2="${x0 + a * dx}" y2="${z0 + b * dz}" stroke="#fff" stroke-width="0.5" stroke-dasharray="1.5 1"/>`).join("")
      : "";
  }
  load(p0);
}

// =======================================================================================
// measure mode

type MethodM = "wiener" | "tikhonov" | "rl";
interface SM { method: MethodM; snr: number; lambda: number; photons: number; kIdx: number }
const DM: SM = { method: "rl", snr: 300, lambda: 1e-3, photons: 1000, kIdx: 0 };
const PM: Preset<SM>[] = [
  { id: "integrals", label: "integrals hold", s: { method: "rl", photons: 3000 }, k: 100 },
  { id: "peaks", label: "peaks lie", s: { method: "rl", photons: 3000 }, k: 316 },
  { id: "sizes", label: "sizes shrink to a floor", s: { method: "rl", photons: 3000 }, k: 1000 },
  { id: "pairs", label: "close pairs: the separation is biased", s: { method: "rl", photons: 1000 }, k: 100 },
  { id: "extra", label: "1000 iterations: extra spots", s: { method: "rl", photons: 1000 }, k: 1000 },
  { id: "wiener", label: "Wiener", s: { method: "wiener", snr: 300, photons: 1000 } },
];

function mountMeasure(host: HTMLElement): void {
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? PM.filter(p => only.includes(p.id)) : PM;
  const p0 = PM.find(p => p.id === host.dataset.preset) ?? PM[0];
  const st: SM = { ...DM };
  let run: M.RLRun | null = null;
  const imgs = htmlEl("div", "rl-imgs", "", host);
  const pT = imagePanel(imgs, "truth"), pO = imagePanel(imgs, "observed"), pR = imagePanel(imgs, "restored"), pE = imagePanel(imgs, "error: restored − truth");
  const mid = htmlEl("div", "rl-plots", "", host);
  const table = htmlEl("table", "rl-table", "", mid);
  const tnote = htmlEl("div", "qb-note", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const mS = select<MethodM>(panel, "method", [["wiener", "Wiener (flat)"], ["tikhonov", "Tikhonov (roughness)"], ["rl", "Richardson–Lucy"]], st.method, v => { st.method = v; sync(); restore(); });
  const snrS = logSlider(panel, "SNR", 1, 1e6, st.snr, v => { st.snr = v; restore(); });
  const lamS = logSlider(panel, "λ", 1e-6, 1, st.lambda, v => { st.lambda = v; restore(); });
  const kS = slider(panel, "iteration k", 0, 1, 1, 0, v => { st.kIdx = v; restore(); }, v => String(run ? run.iters[Math.min(v, run.iters.length - 1)] : 0));
  const phS = logSlider(panel, "photons at 1", 30, 1e5, st.photons, v => { st.photons = Math.round(v); remake(); }, v => fmtSci(Math.round(v)));
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", `Airy PSF, NA 1.0: Rayleigh distance ${M.RAYLEIGH.toFixed(2)} px. Top row: bead pairs at 0.8, 1.0, 1.2 and 1.5 × Rayleigh. Middle: spots of flux 1 : 2 : 4. Bottom: disks of radius 3, 5 and 8 px.`, panel);

  function sync(): void {
    panelHide(rowOf(snrS.input), st.method !== "wiener"); panelHide(rowOf(lamS.input), st.method !== "tikhonov"); panelHide(rowOf(kS.input), st.method !== "rl");
  }
  let pendingK: number | undefined;
  function load(p: Preset<SM>): void {
    Object.assign(st, DM, p.s); pendingK = p.k;
    pbtns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    mS.set(st.method); snrS.setValue(st.snr); lamS.setValue(st.lambda); phS.setValue(st.photons); sync();
    remake();
  }
  const psf = makePSF(N, { kind: "airy", width: 1.0 });
  const truth = render(M.measureScene(), N), clean = blur(truth, psf, N);
  const thr = M.COUNT_THR;
  const rTruth = { ...M.measure(truth, N, M.BASE, thr), count: M.TRUE_COUNT };
  let y: Float64Array = new Float64Array(0), G: D.Spectrum, ctx: D.GainContext;
  const jobs = new Jobs();
  function remake(): void {
    y = M.photonData(clean, st.photons, 0, 21);
    const A = D.fftOperator2d(psf, N, N);
    ctx = D.gainContext(A.H, N, N); G = D.spectrum2d(y, N, N);
    run = new M.RLRun(y, A, 0, 1000);
    kS.input.max = String(run.iters.length - 1);
    if (pendingK !== undefined) { st.kIdx = run.iters.findIndex(k => k >= pendingK!); pendingK = undefined; }
    st.kIdx = Math.max(0, Math.min(st.kIdx, run.iters.length - 1));
    kS.set(st.kIdx);
    const r = run;
    jobs.run((function* () { while (!r.done) { r.step(); yield; } })(), () => { if (st.method === "rl") restore(); }, 14, 120);
    restore();
  }
  function restore(): void {
    if (!run) return;
    let x: Float64Array, label = "";
    if (st.method === "rl") { const i = Math.min(st.kIdx, run.snaps.length - 1); x = run.snaps[i]; label = `RL, k = ${run.iters[i]}${run.done ? "" : ` (computing… ${run.k})`}`; }
    else if (st.method === "wiener") { x = D.restoreSpectrum(G, { kind: "wiener", snr: st.snr, model: "flat" }, ctx); label = `Wiener, SNR ${fmtSci(st.snr)}`; }
    else { x = D.restoreSpectrum(G, { kind: "tikhonov", lambda: st.lambda, penalty: "roughness" }, ctx); label = `Tikhonov, λ ${fmtSci(st.lambda)}`; }
    const est = x.map(v => v / st.photons), obs = y.map(v => v / st.photons);
    const hi = 1.0;
    pT.draw(truth, N, N, { lo: 0, hi }); pO.draw(obs, N, N, { lo: 0, hi }); pR.draw(est, N, N, { lo: 0, hi });
    pE.draw(est.map((v, i) => v - truth[i]), N, N, { lo: -0.5, hi: 0.5, map: diverging });
    const rRaw = M.measure(obs, N, M.BASE, thr), rRest = M.measure(est, N, M.BASE, thr);
    const rows: [string, keyof M.Readout, number][] = [["peak ratio, spots 4 : 1", "peak", 2], ["integrated ratio, spots 4 : 1", "integral", 2], ["FWHM of the bright spot (px)", "fwhm", 2], ["separation, 1.2 × Rayleigh pair (px)", "sep", 2], ["detected spots", "count", 0]];
    const cls = (v: number, t: number, key: string) => {
      if (!Number.isFinite(v)) return "bad";
      const e = key === "count" ? Math.abs(v - t) : Math.abs(v - t) / Math.abs(t);
      return key === "count" ? (e === 0 ? "ok" : e <= 2 ? "mid" : "bad") : e < 0.1 ? "ok" : e < 0.3 ? "mid" : "bad";
    };
    const f = (v: number, d: number) => (Number.isFinite(v) ? v.toFixed(d) : "merged");
    table.innerHTML = `<tr><th>measurement</th><th>raw</th><th>restored</th><th>truth</th></tr>` + rows.map(([lab, key, d]) =>
      `<tr><td>${lab}</td><td class="${cls(rRaw[key], rTruth[key], key)}">${f(rRaw[key], d)}</td><td class="${cls(rRest[key], rTruth[key], key)}">${f(rRest[key], d)}</td><td>${f(rTruth[key], d)}</td></tr>`).join("");
    tnote.textContent = "Green: within 10 % of the truth (counts: exact); yellow: within 30 % (counts: ±2); red: worse. Spots are counted as local maxima above one fixed threshold.";
    pR.cap.textContent = "restored: " + label;
    ro.textContent = `${label}\nRMSE ${rmse(est, truth).toFixed(4)} (truth peak ${Math.max(...truth).toFixed(2)})`;
    kS.set(st.kIdx);
  }
  load(p0);
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const mode = host.dataset.mode ?? "2d";
  if (mode === "xz") mountXZ(host); else if (mode === "measure") mountMeasure(host); else mount2d(host);
}

autoMount(NAME, mount);
