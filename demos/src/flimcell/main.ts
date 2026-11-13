// A FLIM image of the unmixing day's cell: a seeded x × y × t photon-count stack (128 × 128
// pixels × 64 time bins over 12.5 ns, Poisson, Gaussian IRF, background). Panels: the intensity
// image, a second image (lifetime as colour with intensity as brightness, or a fraction map),
// and a side panel that depends on the mode:
//   estimate  the histogram of the clicked pixel (binned), mean arrival time or two gates
//   fit       global analysis: two lifetimes fitted once to the summed decay, then every pixel's
//             amplitudes by linear least squares (≥ 0): a fraction map
//   phasor    the phasor histogram; a circular cursor highlights its pixels in the image, a brush
//             on the image shows its pixels on the plot; median filter of g, s; a line through
//             the cloud gives two lifetimes where it meets the semicircle
//   fret      donor-only point and the line from it through the cloud: τ_DA, E and an
//             interacting-fraction map
//   nadh      free / bound NADH (and a label as a third species): fraction-bound map
//
// <div data-demo="flimcell" data-mode="estimate|fit|phasor|fret" data-preset="..."></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot, colorbar } from "../lib/plot";
import { viridis, inferno, ColorMap } from "../lib/colour";
import { median, seSquare } from "../lib/morph";
import * as D from "../lib/decay";
import * as Ph from "../lib/phasor";
import { lmFit, linearAmplitudes } from "../lib/nlfit";
import { ImgView, injectStyle } from "../canny/imgview";
import { phasorPlane, handle, logMap, FLIM_CSS } from "../phasorplot/plane";

const NAME = "flimcell";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { gap: .6em; flex-wrap: wrap; align-items: flex-start; }
${SEL} .fc-imgs { flex: 1 1 520px; display: flex; gap: .4em; min-width: 0; max-width: 620px; }
${SEL} .fc-imgs > .ev-fig { flex: 1 1 50%; }
${SEL} .fc-side { flex: 1 1 360px; min-width: 0; max-width: 460px; }
${SEL} .qb-panel { flex: 1 1 260px; gap: .25em; }
${SEL} .fc-cb { max-width: 300px; }
.reveal ${SEL} { flex-wrap: nowrap; }
.reveal ${SEL} .fc-imgs { flex: 0 0 255px; flex-direction: column; gap: .1em; }
.reveal ${SEL} .fc-imgs > .ev-fig { flex: 0 0 auto; width: 232px; }
.reveal ${SEL} .fc-side { flex: 0 0 420px; }
.reveal ${SEL} .qb-panel { flex: 1 1 280px; font-size: 17px; gap: .1em; }
.reveal ${SEL} .ev-fig figcaption { font-size: 17px; }
`;

type Mode = "estimate" | "fit" | "phasor" | "fret" | "nadh";
interface State { mode: Mode; scene: D.SceneKind; photons: number; bin: number; minPh: number; est: "mean" | "gates"; cg: number; cs: number; cr: number; med: boolean; line: boolean; third: boolean; second: "tau" | "frac"; px: number; py: number }
const DEFAULT: State = { mode: "estimate", scene: "concentration", photons: 400, bin: 0, minPh: 20, est: "mean", cg: 0.5, cs: 0.4, cr: 0.05, med: false, line: false, third: false, second: "tau", px: 64, py: 50 };
type Preset = { id: string; label: string; s: Partial<State> };
const PRESETS: Record<string, Preset[]> = {
  estimate: [
    { id: "intensity", label: "intensity lies, lifetime does not", s: { scene: "concentration" } },
    { id: "twodyes", label: "same brightness, different lifetime", s: { scene: "twodyes", px: 56, py: 60 } },
    { id: "quencher", label: "a quencher shortens τ", s: { scene: "quencher" } },
    { id: "bin", label: "too few photons per pixel: bin", s: { scene: "concentration", photons: 40, minPh: 5 } },
    { id: "mask", label: "background pixels: mask them", s: { scene: "background", minPh: 0, px: 10, py: 10 } },
  ],
  fit: [
    { id: "global", label: "global analysis: a fraction map", s: { mode: "fit", scene: "mixture", second: "frac" } },
    { id: "globalfew", label: "50 photons per pixel still work", s: { mode: "fit", scene: "mixture", photons: 50, second: "frac", minPh: 5 } },
    { id: "globalfret", label: "global analysis of a FRET cell", s: { mode: "fit", scene: "fret", second: "frac" } },
  ],
  phasor: [
    { id: "line", label: "a cloud on a line", s: { mode: "phasor", scene: "mixture", line: true } },
    { id: "cursor", label: "cursor ↔ image", s: { mode: "phasor", scene: "twodyes", cg: 0.8, cs: 0.4, cr: 0.06 } },
    { id: "median", label: "median-filtered phasors", s: { mode: "phasor", scene: "mixture", photons: 60, minPh: 10, med: true } },
    { id: "background", label: "background pulls to the origin", s: { mode: "phasor", scene: "background", minPh: 5, cg: 0.2, cs: 0.2, cr: 0.12 } },
  ],
  fret: [
    { id: "fret", label: "FRET compartment", s: { mode: "fret", scene: "fret", second: "frac" } },
    { id: "fretfew", label: "too few photons", s: { mode: "fret", scene: "fret", photons: 40, minPh: 5, second: "frac" } },
    { id: "nadh", label: "fraction-bound map", s: { mode: "nadh", scene: "nadh", second: "frac" } },
    { id: "third", label: "a third species", s: { mode: "nadh", scene: "nadh", second: "frac", third: true, cg: 0.3, cs: 0.4, cr: 0.05 } },
  ],
};
const SCENES: Record<string, D.SceneKind[]> = {
  estimate: ["concentration", "twodyes", "quencher", "background"], fit: ["mixture", "twodyes", "fret"],
  phasor: ["twodyes", "mixture", "background"], fret: ["fret", "nadh"],
};
const TAU_MAX = 4, OMEGA = D.omegaOf(80), TAU_D = 2.6, NADH = [0.4, 2.5, 4.5];
const phM = logMap(10, 2000);

/** RGBA: colour = map(value), brightness = intensity / iMax; masked pixels black. */
function mergeRGBA(val: ArrayLike<number>, I: ArrayLike<number>, iMax: number, mask: ArrayLike<number>, map: ColorMap, lo: number, hi: number): Uint8ClampedArray {
  const n = val.length, out = new Uint8ClampedArray(4 * n);
  for (let p = 0; p < n; p++) {
    out[4 * p + 3] = 255;
    if (!mask[p] || !Number.isFinite(val[p])) continue;
    const c = map(Math.max(0, Math.min(1, (val[p] - lo) / (hi - lo)))), b = Math.min(1, Math.sqrt(I[p] / iMax));
    out[4 * p] = 255 * c[0] * b; out[4 * p + 1] = 255 * c[1] * b; out[4 * p + 2] = 255 * c[2] * b;
  }
  return out;
}

function mount(host: HTMLElement): void {
  injectStyle("flim-common", FLIM_CSS);
  injectStyle(NAME, STYLE);
  const group = (["estimate", "fit", "phasor", "fret"].includes(host.dataset.mode ?? "") ? host.dataset.mode : "estimate") as string;
  const presetList = PRESETS[group];
  const p0 = presetList.find(p => p.id === host.dataset.preset) ?? (host.dataset.preset ? undefined : presetList[0]);
  const base: State = { ...DEFAULT, mode: (group === "fret" ? "fret" : group) as Mode };
  const st: State = { ...base, ...(p0?.s ?? {}) };

  const imgs = htmlEl("div", "fc-imgs", "", host);
  const vI = new ImgView(imgs, "intensity (photons)");
  const vT = new ImgView(imgs, "lifetime");
  const side = htmlEl("div", "fc-side fl-col", "", host);
  // estimate mode: a pixel histogram; other modes: the phasor plane (both built, one shown)
  const hist = new Plot(side, { width: 440, height: 330, x: [0, 12.5], y: [0.5, 100], ylog: true, xlabel: "time (ns)", ylabel: "photons per bin", margin: { l: 78, r: 10, t: 10, b: 60 } });
  const hData = hist.step([], [], { color: "var(--qb-c1)", width: 2 }, "mid");
  const hIrf = hist.line([], [], { color: "var(--qb-c4)", width: 2, dash: "6 4", label: "IRF" });
  const hFit = hist.line([], [], { color: "var(--qb-c2)", width: 2.5, label: "global fit" });
  const hGates = [hist.band([], [], [], { color: "var(--qb-c3)", fillOpacity: 0.15 }), hist.band([], [], [], { color: "var(--qb-c5)", fillOpacity: 0.15 })];
  const pp = phasorPlane(side, OMEGA, { width: 440, height: 330 });
  const pl = pp.plot;
  let heat = pl.heatmap(new Float64Array(1), 1, 1, { maxCells: 0 });
  const brushPts = pl.scatter([], [], { color: "var(--qb-c4)", r: 2.5, opacity: 0.8 });
  const lineM = pl.line([], [], { color: "var(--qb-c2)", width: 2.5 });
  const refPts = pl.scatter([], [], { r: 8 });
  const triM = pl.line([], [], { color: "var(--qb-c3)", width: 2, dash: "6 4" });
  const curC = pl.mark<[number, number, number]>((g, cg, cs, cr) => {
    const c = svgEl("circle", { cx: pl.x(cg), cy: pl.y(cs), r: Math.abs(pl.x(cg + cr) - pl.x(cg)) }, g);
    c.style.cssText = "fill:none;stroke:#e0007a;stroke-width:3";
  }, [st.cg, st.cs, st.cr], {}, "line", pl.over);
  const curH = handle(pl, "#e0007a", 8, (x, y) => { st.cg = x; st.cs = y; drawSide(); drawImages(); });
  const cbWrap = htmlEl("div", "fc-cb", "", side);
  const ro = htmlEl("div", "qb-readout fl-ro", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fl-lab", "presets", panel);
  const presetBtns = presets(panel, presetList.map(p => ({ label: p.label, apply: () => { Object.assign(st, base, p.s); sync(); simulate(); } })));
  htmlEl("div", "fl-lab", "scene", panel);
  const sceneList = SCENES[group];
  const sceneBtns = presets(panel, sceneList.map(k => ({ label: k, apply: () => { st.scene = k; if (k === "nadh") st.mode = "nadh"; else if (group === "fret") st.mode = "fret"; simulate(); } })));
  const sPh = slider(panel, "photons (bright pixels)", 0, 100, 0.5, phM.u(st.photons), u => { st.photons = Math.round(phM.v(u)); simulate(); }, u => String(Math.round(phM.v(u))));
  const sBin = slider(panel, "binning", 0, 4, 1, st.bin, v => { st.bin = v; derive(); }, v => `${2 * v + 1}×${2 * v + 1}`);
  const sMin = slider(panel, "mask below (photons)", 0, 200, 1, st.minPh, v => { st.minPh = v; derive(); });
  const estRow = htmlEl("div", "qb-presets", "", panel);
  const estBtns = presets(estRow, [{ label: "mean arrival", apply: () => { st.est = "mean"; derive(); } }, { label: "two gates", apply: () => { st.est = "gates"; derive(); } }]);
  const sR = slider(panel, "cursor radius", 0.01, 0.2, 0.01, st.cr, v => { st.cr = v; drawSide(); drawImages(); });
  const chkRow = htmlEl("div", "qb-presets", "", panel);
  const cMed = checkbox(chkRow, "median g, s", st.med, v => { st.med = v; derive(); });
  const cLine = checkbox(chkRow, "line", st.line, v => { st.line = v; drawSide(); });
  const cThird = checkbox(chkRow, "third species", st.third, v => { st.third = v; derive(); });
  const secRow = htmlEl("div", "qb-presets", "", panel);
  const secBtns = presets(secRow, [{ label: "show τ", apply: () => { st.second = "tau"; drawImages(); } }, { label: "show fraction", apply: () => { st.second = "frac"; drawImages(); } }]);
  const actRow = htmlEl("div", "qb-presets", "", panel);
  button(actRow, "clear brush", () => { brush.fill(0); drawSide(); drawImages(); });
  button(actRow, "reset", () => { Object.assign(st, base, p0?.s ?? {}); presetBtns.forEach(b => b.classList.remove("active")); sync(); simulate(); });

  let sc!: D.FlimScene, P = 0, n = 128, nb = 64;
  let stack: Float32Array = new Float32Array(0), I: Float32Array = new Float32Array(0), mask = new Uint8Array(0), brush = new Uint8Array(0);
  let tauMap: Float32Array = new Float32Array(0), fracMap: Float32Array = new Float32Array(0), g: Float32Array = new Float32Array(0), s: Float32Array = new Float32Array(0);
  let K: Ph.Ph = { g: 1, s: 0 }, iMax = 1, info: string[] = [];
  let global: { tau: number[]; bg: number; fit: Float64Array; sum: Float64Array } | null = null;

  function simulate(): void {
    sc = D.flimScene(st.scene, { photons: st.photons, seed: 11 });
    n = sc.n; P = n * n; nb = sc.ax.nBins;
    if (brush.length !== P) brush = new Uint8Array(P);
    const ref = Ph.phasor(D.decayBins({ tau: [4], f: [1] }, sc.irfSpec, sc.ax));
    K = Ph.calibrationFactor(ref, 4, OMEGA);
    derive();
  }

  function derive(): void {
    stack = D.binStack(sc.stack, n, nb, st.bin);
    I = new Float32Array(P);
    for (let j = 0; j < nb; j++) for (let p = 0; p < P; p++) I[p] += stack[j * P + p];
    mask = Uint8Array.from(I, v => (v >= Math.max(1, st.minPh) ? 1 : 0));
    const sorted = Float32Array.from(I).sort();
    iMax = sorted[Math.floor(0.995 * (P - 1))] || 1;
    info = [];
    global = null;
    fracMap = new Float32Array(P).fill(NaN);
    if (st.mode === "estimate") {
      const ax = sc.ax, irfMean = D.meanArrival(sc.irf, ax), d = D.binWidth(ax);
      tauMap = new Float32Array(P);
      const ts = D.binCentres(ax);
      const j0 = Math.round(1.0 / d), nw = Math.round(2.5 / d);
      for (let p = 0; p < P; p++) {
        if (!mask[p]) { tauMap[p] = NaN; continue; }
        if (st.est === "mean") { let a = 0; for (let j = 0; j < nb; j++) a += stack[j * P + p] * ts[j]; tauMap[p] = a / I[p] - irfMean; }
        else {
          let d1 = 0, d2 = 0;
          for (let j = j0; j < j0 + nw; j++) d1 += stack[j * P + p];
          for (let j = j0 + nw; j < j0 + 2 * nw; j++) d2 += stack[j * P + p];
          tauMap[p] = d1 > 0 && d2 > 0 && d1 > d2 ? nw * d / Math.log(d1 / d2) : NaN;
        }
      }
    } else if (st.mode === "fit") fitGlobal();
    else {
      const ph = Ph.phasorStack(stack, nb, P);
      g = ph.g; s = ph.s;
      Ph.calibrateMaps(g, s, K);
      for (let p = 0; p < P; p++) if (!mask[p]) { g[p] = NaN; s[p] = NaN; }
      if (st.med) {
        const g0 = g.map(v => (Number.isFinite(v) ? v : 0)), s0 = s.map(v => (Number.isFinite(v) ? v : 0));
        const gm = median(g0, n, n, seSquare(1)), sm = median(s0, n, n, seSquare(1));
        for (let p = 0; p < P; p++) if (mask[p]) { g[p] = gm[p]; s[p] = sm[p]; }
      }
      tauMap = Float32Array.from({ length: P }, (_, p) => (mask[p] ? s[p] / (g[p] * OMEGA) : NaN));
      if (st.mode === "fret") fretMaps();
      if (st.mode === "nadh") nadhMaps();
    }
    drawSide();
    drawImages();
  }

  /** Global analysis: two lifetimes + background from the summed decay (Poisson fit), then
   *  each pixel's amplitudes by NNLS on the three basis curves. */
  function fitGlobal(): void {
    const sum = new Float64Array(nb);
    let npx = 0;
    for (let p = 0; p < P; p++) if (mask[p]) { npx++; for (let j = 0; j < nb; j++) sum[j] += stack[j * P + p]; }
    let tot = 0;
    for (const v of sum) tot += v;
    const model = (q: Float64Array) => { const a = q[0] + q[2]; return D.expectedCounts({ tau: [q[1], q[3]], f: [q[0] / a, q[2] / a] }, sc.irfSpec, sc.ax, a, q[4]); };
    const r = lmFit(model, sum, [tot / 2, 0.8, tot / 2, 3, Math.max(0.1, sum[1])], { cost: "poisson", positive: [true, true, true, true, false], lower: [0, 0.05, 0, 0.05, 0], maxIter: 200 });
    const t1 = Math.min(r.p[1], r.p[3]), t2 = Math.max(r.p[1], r.p[3]);
    global = { tau: [t1, t2], bg: r.p[4], fit: model(r.p), sum };
    const B1 = D.decayBins({ tau: [t1], f: [1] }, sc.irfSpec, sc.ax), B2 = D.decayBins({ tau: [t2], f: [1] }, sc.irfSpec, sc.ax);
    const B = new Float64Array(nb * 3);
    for (let j = 0; j < nb; j++) { B[3 * j] = B1[j]; B[3 * j + 1] = B2[j]; B[3 * j + 2] = 1; }
    tauMap = new Float32Array(P).fill(NaN);
    const y = new Float64Array(nb);
    for (let p = 0; p < P; p++) {
      if (!mask[p]) continue;
      for (let j = 0; j < nb; j++) y[j] = stack[j * P + p];
      const { a } = linearAmplitudes(B, nb, 3, y, "ls");
      const f2 = a[0] + a[1] > 0 ? a[1] / (a[0] + a[1]) : NaN;
      fracMap[p] = f2;
      tauMap[p] = (1 - f2) * t1 + f2 * t2;
    }
    info.push(`global fit to ${npx} pixels, ${Math.round(tot)} photons:`, `τ₁ = ${t1.toFixed(2)}, τ₂ = ${t2.toFixed(2)} ns (true ${sc.tau.map(t => t.toFixed(1)).join(", ")})`);
  }

  /** FRET: the line from the donor-only point through the cloud; where it meets the
   *  semicircle is the FRET state; position along it = photon fraction → amplitude fraction. */
  function fretMaps(): void {
    const D0 = Ph.semicircle(OMEGA * TAU_D);
    let sxx = 0, sxy = 0, syy = 0, W = 0;
    for (let p = 0; p < P; p++) {
      if (!mask[p] || !Number.isFinite(g[p])) continue;
      const w = I[p], dx = g[p] - D0.g, dy = s[p] - D0.s;
      sxx += w * dx * dx; sxy += w * dx * dy; syy += w * dy * dy; W += w;
    }
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    let dir = { g: Math.cos(th), s: Math.sin(th) };
    if (dir.g < 0) dir = { g: -dir.g, s: -dir.s };    // towards shorter lifetimes (larger g)
    const taus = Ph.lineSemicircle(D0, dir, OMEGA).filter(t => t < TAU_D * 0.98);
    const tauDA = taus.length ? taus[taus.length - 1] : NaN;
    const E = 1 - tauDA / TAU_D, DA = Ph.semicircle(OMEGA * tauDA);
    let fm = 0, am = 0, cnt = 0, at = 0;
    const truth = sc.regions.alpha;
    for (let p = 0; p < P; p++) {
      if (!mask[p] || !Number.isFinite(g[p])) continue;
      const f = Math.max(0, Math.min(1, Ph.fractionOnSegment({ g: g[p], s: s[p] }, D0, DA)));
      const a = D.amplitudeFractions([1 - f, f], [TAU_D, tauDA])[1];
      fracMap[p] = a; fm += f; am += a; cnt++; at += truth ? truth[p] : 0;
    }
    fretLine = { D0, DA };
    info.push(`donor alone τ_D = ${TAU_D} ns (control)`, `line meets the semicircle at τ_DA = ${tauDA.toFixed(2)} ns`, `E = 1 − τ_DA/τ_D = ${E.toFixed(2)} (true 0.50)`, `cell mean: photon fraction ${(fm / cnt).toFixed(2)}, α ${(am / cnt).toFixed(2)} (true ${(at / cnt).toFixed(2)})`);
  }
  let fretLine: { D0: Ph.Ph; DA: Ph.Ph } | null = null;

  /** NADH: fraction bound from the free → bound segment (or barycentric with the label). */
  function nadhMaps(): void {
    const F = Ph.semicircle(OMEGA * NADH[0]), Bd = Ph.semicircle(OMEGA * NADH[1]), L = Ph.semicircle(OMEGA * NADH[2]);
    let sb = 0, cnt = 0, bt = 0;
    const truth = sc.regions.bound;
    for (let p = 0; p < P; p++) {
      if (!mask[p] || !Number.isFinite(g[p])) continue;
      const q = { g: g[p], s: s[p] };
      let fb: number, ff: number;
      if (st.third) { const w = Ph.barycentric3(q, F, Bd, L).map(v => Math.max(0, v)); const t = w[0] + w[1] || 1; ff = w[0] / t; fb = w[1] / t; }
      else { fb = Math.max(0, Math.min(1, Ph.fractionOnSegment(q, F, Bd))); ff = 1 - fb; }
      const a = D.amplitudeFractions([ff, fb], [NADH[0], NADH[1]])[1];   // molecule fraction bound (equal brightness per unit lifetime)
      fracMap[p] = a; sb += a; cnt++; bt += truth ? truth[p] : 0;
    }
    info.push(`free ${NADH[0]} ns, bound ${NADH[1]} ns` + (st.third ? `, label ${NADH[2]} ns` : ""), `cell mean fraction bound (molecules): ${(sb / cnt).toFixed(2)}`, `true mean over the same pixels: ${(bt / cnt).toFixed(2)}`);
  }

  function drawImages(): void {
    // intensity, with the cursor's pixels (phasor modes) or the selected pixel
    const px = new Uint8ClampedArray(4 * P);
    const inCur = st.mode !== "estimate" && st.mode !== "fit" ? Ph.cursorMask(g, s, { g: st.cg, s: st.cs }, st.cr, mask) : null;
    let nCur = 0;
    for (let p = 0; p < P; p++) {
      const v = Math.min(255, 255 * Math.sqrt(I[p] / iMax));
      if (inCur?.[p]) { px[4 * p] = 224; px[4 * p + 1] = 0; px[4 * p + 2] = 122; nCur++; }
      else if (brush[p]) { px[4 * p] = 255; px[4 * p + 1] = 150 + 0.4 * v; px[4 * p + 2] = 30; }
      else { px[4 * p] = v; px[4 * p + 1] = v; px[4 * p + 2] = v; }
      px[4 * p + 3] = 255;
    }
    vI.rgba(px, n, n);
    vI.setCaption(st.bin ? `intensity, binned ${2 * st.bin + 1}×${2 * st.bin + 1}` : "intensity (photons)");
    const showFrac = st.second === "frac" && st.mode !== "estimate" && st.mode !== "phasor";
    const fracLabel = st.mode === "fret" ? "interacting fraction α" : st.mode === "nadh" ? "fraction bound" : "photon fraction long";
    if (showFrac) { vT.rgba(mergeRGBA(fracMap, I, iMax, mask, inferno, 0, 1), n, n); vT.setCaption(fracLabel); }
    else { vT.rgba(mergeRGBA(tauMap, I, iMax, mask, viridis, 0, TAU_MAX), n, n); vT.setCaption(st.mode === "estimate" ? (st.est === "mean" ? "τ by mean arrival" : "τ by two gates") : st.mode === "fit" ? "τ̄ (intensity-weighted)" : "τ_φ from the phasor"); }
    for (const v of [vI, vT]) { v.clear(); if (st.mode === "estimate") v.box(st.px, st.py, st.bin, "#ffd400", 2); }
    cbWrap.replaceChildren();
    colorbar(cbWrap, showFrac ? inferno : viridis, 0, showFrac ? 1 : TAU_MAX, { label: showFrac ? fracLabel : "τ (ns), brightness = intensity", width: 320, font: 18 });
    if (inCur) curInfo = `cursor: ${nCur} pixels`;
  }
  let curInfo = "";

  function drawSide(): void {
    const est = st.mode === "estimate", fit = st.mode === "fit";
    hist.svg.style.display = est || fit ? "" : "none";
    pl.svg.style.display = est || fit ? "none" : "";
    estRow.style.display = est ? "" : "none";
    for (const el of [sR.input.parentElement!, chkRow]) el.style.display = est || fit ? "none" : "";
    cThird.parentElement!.style.display = st.mode === "nadh" ? "" : "none";
    cMed.parentElement!.style.display = cLine.parentElement!.style.display = st.mode === "phasor" ? "" : "none";
    secRow.style.display = est || st.mode === "phasor" ? "none" : "";
    sceneBtns.forEach((b, i) => b.classList.toggle("active", sceneList[i] === st.scene));
    estBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.est === "mean")));
    secBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.second === "tau")));
    const lines: string[] = [];
    if (est || fit) {
      const xs = D.binCentres(sc.ax);
      let y: Float64Array, model: Float64Array | null = null;
      if (est) {
        y = D.pixelHistogram(stack, P, nb, st.py * n + st.px);
        const tot = y.reduce((a, b) => a + b, 0), irfMean = D.meanArrival(sc.irf, sc.ax);
        const tg = D.twoGateTau(y, sc.ax, 1.0, 2.5);
        lines.push(`pixel (${st.px}, ${st.py})${st.bin ? `, ${2 * st.bin + 1}×${2 * st.bin + 1} box` : ""}: ${tot} photons`,
          `mean arrival − IRF = ${(D.meanArrival(y, sc.ax) - irfMean).toFixed(2)} ns`,
          `two gates ${tg.d1} / ${tg.d2} → ${Number.isFinite(tg.tau) ? tg.tau.toFixed(2) : "–"} ns`,
          `true τ (intensity-weighted) ${Number.isFinite(sc.tauTrue[st.py * n + st.px]) ? sc.tauTrue[st.py * n + st.px].toFixed(2) : "– (outside)"} ns`);
        hGates[0].set([tg.a, tg.a + tg.w], [0.5, 0.5], [1e6, 1e6]); hGates[1].set([tg.a + tg.w, tg.a + 2 * tg.w], [0.5, 0.5], [1e6, 1e6]);
      } else {
        y = global!.sum; model = global!.fit;
        hGates.forEach(b => b.set([], [], []));
        lines.push(...info);
      }
      let ymax = 1;
      for (const v of y) ymax = Math.max(ymax, v);
      hist.setDomain([0, sc.ax.T], [0.5, ymax * 2]);
      hData.set(xs, y.map(v => Math.max(v, 1e-3)));
      const im = Math.max(...sc.irf);
      hIrf.set(xs, Array.from(sc.irf, v => v / im * ymax * 0.8));
      hFit.set(model ? xs : [], model ?? []);
      hist.legend(model ? undefined : [{ label: "IRF", color: "var(--qb-c4)", dash: "6 4" }], "tr");
    } else {
      // phasor histogram (log counts), cursor, brush, line / references
      const H = Ph.phasorHistogram(g, s, { bins: 110, gLo: -0.05, gHi: 1.05, sHi: 0.62, mask });
      const v = Array.from(H.counts, c => (c > 0 ? Math.log(1 + c) : NaN));
      heat.remove();
      heat = pl.heatmap(v, H.binsX, H.binsY, { x0: -0.05, x1: 1.05, y0: 0, y1: 0.62, flipY: false, maxCells: 0, map: t => inferno(0.92 - 0.8 * t), lo: 0 });
      for (const m of [brushPts, lineM, refPts, triM]) pl.layer.appendChild(m.g);
      curC.set(st.cg, st.cs, st.cr); curH.set(st.cg, st.cs);
      const bg: number[] = [], bs: number[] = [];
      for (let p = 0, k = 0; p < P && k < 2000; p++) if (brush[p] && mask[p]) { bg.push(g[p]); bs.push(s[p]); k++; }
      brushPts.set(bg, bs);
      lineM.set([], []); refPts.set([], []); triM.set([], []);
      if (st.mode === "phasor" && st.line) {
        const L = Ph.lineThroughCloud(g, s, OMEGA, Float32Array.from(I, (w, p) => (mask[p] ? w : 0)));
        const ends = L.taus.map(t => Ph.semicircle(OMEGA * t));
        if (ends.length === 2) { lineM.set(ends.map(e => e.g), ends.map(e => e.s)); refPts.set(ends.map(e => e.g), ends.map(e => e.s), ["var(--qb-c3)", "var(--qb-c5)"]); }
        lines.push(`line through the cloud meets the semicircle at`, `τ = ${L.taus.map(t => t.toFixed(2)).join(" and ")} ns (true ${sc.tau.join(", ")})`);
      }
      if (st.mode === "fret" && fretLine) {
        lineM.set([fretLine.D0.g, fretLine.DA.g], [fretLine.D0.s, fretLine.DA.s]);
        refPts.set([fretLine.D0.g, fretLine.DA.g], [fretLine.D0.s, fretLine.DA.s], ["var(--qb-c3)", "var(--qb-c2)"]);
      }
      if (st.mode === "nadh") {
        const pts = NADH.slice(0, st.third ? 3 : 2).map(t => Ph.semicircle(OMEGA * t));
        const loop = st.third ? [...pts, pts[0]] : pts;
        triM.set(loop.map(e => e.g), loop.map(e => e.s));
        refPts.set(pts.map(e => e.g), pts.map(e => e.s), ["var(--qb-c3)", "var(--qb-c5)", "var(--qb-c4)"].slice(0, pts.length));
      }
      lines.push(...info);
      // cursor and brush statistics
      const inCur = Ph.cursorMask(g, s, { g: st.cg, s: st.cs }, st.cr, mask);
      let nc = 0;
      for (const v of inCur) nc += v;
      lines.push(`cursor (${st.cg.toFixed(2)}, ${st.cs.toFixed(2)}): ${nc} pixels`);
      if (bg.length) {
        let fm = 0, cnt = 0;
        for (let p = 0; p < P; p++) if (brush[p] && mask[p] && Number.isFinite(fracMap[p])) { fm += fracMap[p]; cnt++; }
        const mg = bg.reduce((a, b) => a + b, 0) / bg.length, ms = bs.reduce((a, b) => a + b, 0) / bs.length;
        lines.push(`brush: ${bg.length} pixels, mean (${mg.toFixed(2)}, ${ms.toFixed(2)})` + (cnt ? `, fraction ${(fm / cnt).toFixed(2)}` : ""));
      }
    }
    void curInfo;
    ro.textContent = lines.join("\n");
  }

  const pick = (x: number, y: number) => {
    if (st.mode === "estimate") { st.px = x; st.py = y; drawSide(); drawImages(); return; }
    if (st.mode === "fit") return;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < n && yy < n && dx * dx + dy * dy <= 5) brush[yy * n + xx] = 1;
    }
    drawSide(); drawImages();
  };
  vI.onPick(pick); vT.onPick(pick);

  function sync(): void {
    sPh.set(phM.u(st.photons)); sBin.set(st.bin); sMin.set(st.minPh); sR.set(st.cr);
    cMed.checked = st.med; cLine.checked = st.line; cThird.checked = st.third;
    brush.fill(0);
  }

  sync();
  if (p0) presetBtns[presetList.indexOf(p0)].classList.add("active");
  simulate();
}

autoMount(NAME, mount);
