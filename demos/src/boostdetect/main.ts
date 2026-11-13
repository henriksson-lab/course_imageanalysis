// Boosted detector with an attentional cascade (Viola & Jones), on synthetic 24 × 24
// patches: cartoon faces or bright cells against background windows.
//  Tab "boost": add stumps one at a time (AdaBoost over a pool of rectangle features): the
//    chosen feature on the mean target patch, the examples' weights against their score
//    (mistakes grow), the heaviest examples, and training/test error against rounds.
//  Tab "cascade": build stages with a per-stage target detection rate; negatives for each
//    stage are background windows that the earlier stages wrongly accepted. Bars: windows of
//    the test scene surviving each stage; average features evaluated per window.
//  Tab "detect": sliding windows at 4 scales over a 256 × 256 test scene (features scaled,
//    not the image); detections, ground truth, and the stage that rejected each window;
//    compare with one boosted classifier without a cascade.
// Options: data-target="faces|cells", data-tab="boost|cascade|detect", data-preset.
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { drawScalarImage, colormapLUT, viridis } from "../lib/colour";
import { AdaBoost, Stage, Stump, FeatureMatrix, FeatureSpec } from "../lib/boost";
import { TEMPLATES } from "../lib/integral";
import { WIN, Target, makeDataset, testScene, Dataset, SceneTarget } from "./synth";
import { makePool, featureMatrix, PatchTables, patchTables, errorCurve, detect, DetectResult, nextStage, matches } from "./model";
import { DEFAULTS, VARIANTS, variantId } from "./defaults";
import { TRAINED } from "./trained";

const NAME = "boostdetect";
const { nPos: N_POS, nNeg: N_NEG, nTestPos: N_TEST_POS, nTestNeg: N_TEST_NEG } = DEFAULTS;

const STYLE = `
[data-demo="${NAME}"] figure { margin: 0; position: relative; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; }
[data-demo="${NAME}"] canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .bd-left { flex: 0 1 330px; min-width: 220px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .bd-mid { flex: 1 1 400px; min-width: 0; max-width: 480px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 280px; gap: .4em; }
[data-demo="${NAME}"] .bd-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
[data-demo="${NAME}"] .bd-tabs .qb-btn { font-weight: 600; }
[data-demo="${NAME}"] .bd-thumbs { display: grid; grid-template-columns: repeat(6, 1fr); gap: 3px; }
[data-demo="${NAME}"] .bd-thumbs canvas { border: 3px solid transparent; }
[data-demo="${NAME}"] table.bd-table { border-collapse: collapse; font-size: 15px; width: 100%; }
[data-demo="${NAME}"] table.bd-table td, [data-demo="${NAME}"] table.bd-table th { border-bottom: 1px solid var(--qb-grid); padding: .1em .35em; text-align: right; }
[data-demo="${NAME}"] .bd-mid > figure { max-width: 340px; }
.reveal [data-demo="${NAME}"] .bd-lab { display: none; }
[data-demo="${NAME}"] .bd-overlay { position: absolute; left: 0; width: 100%; pointer-events: none; }
[data-demo="${NAME}"] .bd-busy { color: var(--qb-c2); font-weight: 600; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .7em; }
.reveal [data-demo="${NAME}"] .bd-left { flex-basis: 320px; }
.reveal [data-demo="${NAME}"] .bd-mid { flex-basis: 430px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 300px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Tab = "boost" | "cascade" | "detect";
interface Preset { id: string; label: string; tab: Tab; rounds?: number; rotated?: boolean; noGlasses?: boolean; mono?: boolean; faces?: boolean }
const PRESETS: Preset[] = [
  { id: "first", label: "first two stumps", tab: "boost", rounds: 2 },
  { id: "hard", label: "boosting focuses on the hard examples", tab: "boost", rounds: 10 },
  { id: "cascade", label: "a cascade: few features per window", tab: "cascade" },
  { id: "mono", label: "one strong classifier vs a cascade", tab: "detect", mono: true },
  { id: "rotated", label: "a rotated face is missed", tab: "detect", rotated: true, faces: true },
  { id: "noglasses", label: "training set without glasses", tab: "detect", noGlasses: true, faces: true },
];

const C_POS = "#ff7f0e", C_NEG = "#1f77b4";

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const st = {
    target: (d.target === "cells" ? "cells" : "faces") as Target,
    tab: (["boost", "cascade", "detect"].includes(d.tab ?? "") ? d.tab : "boost") as Tab,
    noGlasses: false, rotated: false, mono: false, showStages: true,
    nStages: DEFAULTS.nStages, dStage: DEFAULTS.dStage, fStage: DEFAULTS.fStage, view: 0,
  };
  const p0 = PRESETS.find(p => p.id === d.preset);

  // ---- layout ----
  const leftBox = htmlEl("div", "bd-left", "", host);
  const left = htmlEl("div", "", "", leftBox);
  left.style.cssText = "display:flex;flex-direction:column;gap:.4em";
  const mid = htmlEl("div", "bd-mid", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  const tabRow = htmlEl("div", "qb-presets bd-tabs", "", panel);
  const tabBtns = (["boost", "cascade", "detect"] as Tab[]).map(t => button(tabRow, t, () => { st.tab = t; render(); }));
  htmlEl("div", "bd-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  htmlEl("div", "bd-lab", "target", panel);
  const tRow = htmlEl("div", "qb-presets", "", panel);
  const tBtns = (["faces", "cells"] as Target[]).map(t => button(tRow, t, () => { if (st.target !== t) { st.target = t; resetAll(); render(); } }));
  // boost controls
  const boostBox = htmlEl("div", "", "", panel); boostBox.style.cssText = "display:flex;flex-direction:column;gap:.35em";
  const bRow = htmlEl("div", "qb-presets", "", boostBox);
  button(bRow, "add one stump", () => addRounds(1));
  button(bRow, "add 10", () => addRounds(10));
  const viewS = slider(boostBox, "show stump", 1, 1, 1, 1, v => { st.view = v - 1; drawBoost(); });
  // cascade controls
  const casBox = htmlEl("div", "", "", panel); casBox.style.cssText = "display:flex;flex-direction:column;gap:.35em";
  const nS = slider(casBox, "stages", 2, 10, 1, st.nStages, v => { st.nStages = v; });
  const dS = slider(casBox, "detection / stage", 0.95, 0.999, 0.001, st.dStage, v => { st.dStage = v; });
  const fS = slider(casBox, "false pos. / stage", 0.2, 0.8, 0.05, st.fStage, v => { st.fStage = v; });
  button(casBox, "build cascade", () => { cascade = null; buildCascade(); });
  // detect controls
  const detBox = htmlEl("div", "", "", panel); detBox.style.cssText = "display:flex;flex-direction:column;gap:.35em";
  const monoC = checkbox(detBox, "no cascade", st.mono, v => { st.mono = v; drawDetect(); });
  const stageC = checkbox(detBox, "colour by stage", st.showStages, v => { st.showStages = v; drawDetect(); });
  const faceOpts = htmlEl("div", "", "", panel); faceOpts.style.cssText = "display:flex;flex-direction:column;gap:.35em";
  const rotC = checkbox(faceOpts, "rotated face in the scene", st.rotated, v => { st.rotated = v; detResult = null; render(); });
  const glC = checkbox(faceOpts, "train without glasses", st.noGlasses, v => { st.noGlasses = v; resetAll(); render(); });
  button(panel, "reset", () => { pbtns.forEach(b => b.classList.remove("active")); Object.assign(st, { noGlasses: false, rotated: false, mono: false }); resetAll(); render(); });
  const status = htmlEl("div", "bd-busy", "", panel);
  // pages: the readout and the note go in the panel, so the left column is no taller than its figure
  const roBox = host.closest(".reveal") ? leftBox : panel;
  const ro = htmlEl("div", "qb-readout", "", roBox);
  const note = htmlEl("div", "qb-note", "", roBox);

  // ---- data and models (built lazily) ----
  const pool: FeatureSpec[] = makePool();
  let train: Dataset | null = null, FM: FeatureMatrix | null = null, testTables: PatchTables[] | null = null, testY: Int8Array | null = null;
  let ab: AdaBoost | null = null, trainCurve: number[] = [], testCurve: number[] = [], bounds: number[] = [];
  let meanPos: Float64Array | null = null;
  let cascade: Stage[] | null = null, cascadeInfo: { stumps: number; d: number; f: number; pass: number; neg: number }[] = [];
  let detResult: DetectResult | null = null, monoResult: DetectResult | null = null, scene: { img: Float64Array; targets: SceneTarget[] } | null = null;
  let busy = false;

  function resetAll(): void {
    train = null; FM = null; testTables = null; ab = null; trainCurve = []; testCurve = []; bounds = [];
    cascade = null; cascadeInfo = []; detResult = null; monoResult = null; scene = null; st.view = 0;
  }

  // while mounting, work runs synchronously (the page opens complete); afterwards it is
  // deferred so the busy message can show
  let mounting = true;
  const later = (fn: () => void) => { if (mounting) fn(); else window.setTimeout(fn, 20); };
  function setBusy(msg: string): void { busy = !!msg; status.textContent = msg; }

  function ensureData(then: () => void): void {
    if (FM) { then(); return; }
    setBusy("preparing the training set …");
    later(() => {
      const glasses = st.noGlasses ? "never" : "random";
      const v = VARIANTS.find(x => x.id === variantId(st.target, st.noGlasses))!;
      train = makeDataset(st.target, N_POS, N_NEG, v.seed, { glasses });
      const test = makeDataset(st.target, N_TEST_POS, N_TEST_NEG, v.testSeed);
      FM = featureMatrix(pool, train).M;
      testTables = test.patches.map(patchTables); testY = test.y;
      ab = new AdaBoost(FM, train.y);
      meanPos = new Float64Array(WIN * WIN);
      for (let i = 0; i < N_POS; i++) for (let k = 0; k < WIN * WIN; k++) meanPos[k] += train.patches[i][k] / N_POS;
      setBusy("");
      then();
    });
  }

  function addRounds(n: number, then?: () => void): void {
    if (busy) return;
    ensureData(() => {
      for (let k = 0; k < n; k++) {
        ab!.round();
        trainCurve.push(ab!.trainingError());
        bounds.push(ab!.bound());
      }
      testCurve = errorCurve(pool, ab!.stumps, testTables!, testY!);
      st.view = ab!.stumps.length - 1;
      monoResult = null;
      if (then) then(); else render();
    });
  }

  function buildCascade(then?: () => void): void {
    if (busy) return;
    ensureData(() => {
      cascade = []; cascadeInfo = []; detResult = null;
      const pos = train!.patches.slice(0, N_POS);
      const step = (k: number) => {
        if (k >= st.nStages) { setBusy(""); if (then) then(); else render(); return; }
        setBusy(`training stage ${k + 1} of ${st.nStages} …`);
        later(() => {
          const r = nextStage(st.target, pool, pos, cascade!, st.dStage, st.fStage, DEFAULTS.maxStumps, DEFAULTS.nMine, 100 + k);
          if (!r.stage) { setBusy(""); render(); return; }
          cascade!.push(r.stage);
          cascadeInfo.push({ stumps: r.report!.stumps, d: r.report!.detection, f: r.report!.falsePositive, pass: r.passBefore, neg: r.negatives });
          render();
          step(k + 1);
        });
      };
      step(0);
    });
  }

  /** The precomputed cascade for the default settings (trained.ts), if the settings are the defaults. */
  function defaultCascade(): boolean {
    if (st.nStages !== DEFAULTS.nStages || st.dStage !== DEFAULTS.dStage || st.fStage !== DEFAULTS.fStage) return false;
    const t = TRAINED[variantId(st.target, st.noGlasses)];
    cascade = t.cascade; cascadeInfo = t.info; detResult = null;
    return true;
  }

  function ensureDetect(then: () => void): void {
    if ((!cascade || cascade.length === 0) && !defaultCascade()) { buildCascade(() => ensureDetect(then)); return; }
    if (!scene) scene = testScene(st.target, 5, { rotated: st.rotated && st.target === "faces" });
    if (!detResult) detResult = detect(scene.img, 256, 256, pool, cascade!, null);
    if (!monoResult) monoResult = detect(scene.img, 256, 256, pool, [], { stumps: TRAINED[variantId(st.target, st.noGlasses)].mono, threshold: 0 });
    then();
  }

  function applyPreset(p: Preset): void {
    if (busy) return;
    const needReset = (!!p.noGlasses) !== st.noGlasses || (!!p.faces && st.target !== "faces");
    if (p.faces) st.target = "faces";
    st.noGlasses = !!p.noGlasses; st.rotated = !!p.rotated; st.mono = !!p.mono; st.tab = p.tab;
    if (needReset) resetAll();
    scene = null; detResult = null; monoResult = null;
    if (p.rounds !== undefined) {
      ab = null; trainCurve = []; testCurve = []; bounds = [];
      if (FM) ab = new AdaBoost(FM, train!.y);
      addRounds(p.rounds);
      return;
    }
    render();
  }

  // ---- drawing ----
  function sync(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["boost", "cascade", "detect"][i] === st.tab));
    pbtns.forEach((b, i) => { b.style.display = PRESETS[i].tab === st.tab && (!PRESETS[i].faces || st.target === "faces") ? "" : "none"; });
    tBtns.forEach((b, i) => b.classList.toggle("active", ["faces", "cells"][i] === st.target));
    boostBox.style.display = st.tab === "boost" ? "flex" : "none";
    casBox.style.display = st.tab === "cascade" ? "flex" : "none";
    detBox.style.display = st.tab === "detect" ? "flex" : "none";
    faceOpts.style.display = st.target === "faces" && st.tab !== "boost" ? "flex" : "none";
    monoC.checked = st.mono; stageC.checked = st.showStages; rotC.checked = st.rotated; glC.checked = st.noGlasses;
    nS.set(st.nStages); dS.set(st.dStage); fS.set(st.fStage);
  }

  function render(): void {
    sync();
    if (st.tab === "boost") {
      if (!ab || ab.stumps.length === 0) { addRounds(1); return; }
      drawBoost();
    } else if (st.tab === "cascade") {
      if (!cascade && !defaultCascade()) { if (!busy) buildCascade(); else drawCascade(); return; }
      drawCascade();
    } else {
      if (busy) return;
      ensureDetect(drawDetect);
    }
  }

  function patchCanvas(parent: Element, p: Float64Array, scale = 1): HTMLCanvasElement {
    const c = htmlEl("canvas", "", "", parent);
    drawScalarImage(c, p, WIN, WIN, scale === 1 ? {} : { lo: 0, hi: 255 });
    return c;
  }

  function drawFeature(c: HTMLCanvasElement, f: FeatureSpec): void {
    const S = 12;
    const big = document.createElement("canvas");
    big.width = WIN * S; big.height = WIN * S;
    const ctx = big.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(c, 0, 0, WIN * S, WIN * S);
    const t = TEMPLATES[f.t];
    for (const r of [...t.rects].sort((a, b) => b.w * b.h - a.w * a.h)) {
      const x = (f.x + r.x * f.sx) * S, y = (f.y + r.y * f.sy) * S, w = r.w * f.sx * S, h = r.h * f.sy * S;
      ctx.fillStyle = r.weight > 0 ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.55)";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = r.weight > 0 ? "#fff" : "#000"; ctx.lineWidth = 3; ctx.strokeRect(x, y, w, h);
    }
    c.replaceWith(big);
  }

  function featureText(s: Stump): string {
    const f = pool[s.feature], t = TEMPLATES[f.t];
    return `${t.label}\n  at (${f.x}, ${f.y}), cells ${f.sx} × ${f.sy}`;
  }

  let boostPlots: { err: Plot; w: Plot } | null = null;
  function drawBoost(): void {
    if (!ab || !meanPos) return;
    left.replaceChildren();
    const T = ab.stumps.length, v = Math.min(st.view, T - 1);
    viewS.input.max = String(T); viewS.set(v + 1);
    const fig = htmlEl("figure", "", "", left);
    htmlEl("figcaption", "", `stump ${v + 1} on the mean ${st.target === "faces" ? "face" : "cell"} (white +, black −)`, fig);
    const c = patchCanvas(fig, meanPos);
    drawFeature(c, pool[ab.stumps[v].feature]);
    // heaviest examples (weights before the shown stump's round, i.e. what it was trained on)
    const w = ab.history[v] ?? ab.w;
    const idx = Array.from(w.keys()).sort((a, b) => w[b] - w[a]).slice(0, 6);
    const tf = htmlEl("figure", "", "", left);
    htmlEl("figcaption", "", `heaviest examples in round ${v + 1}`, tf);
    const th = htmlEl("div", "bd-thumbs", "", tf);
    for (const i of idx) { const cc = patchCanvas(th, train!.patches[i]); cc.style.borderColor = train!.y[i] > 0 ? C_POS : C_NEG; }
    // plots
    if (!boostPlots || !mid.contains(boostPlots.err.svg)) {
      mid.replaceChildren();
      boostPlots = {
        err: new Plot(mid, { width: 470, height: 250, x: [0, 10], y: [0, 0.3], xlabel: "rounds (stumps)", ylabel: "error rate" }),
        w: new Plot(mid, { width: 470, height: 250, x: [-5, 5], y: [-1.6, 1.6], margin: { l: 96 }, xlabel: "score Σ α h (> 0: target)", ylabel: "", yticks: [-1, 1], yfmt: v => (v > 0 ? "target" : "backgr.") }),
      };
    }
    const E = boostPlots.err;
    E.clear();
    const xs = trainCurve.map((_, i) => i + 1);
    E.setDomain([0, Math.max(10, T)], [0, Math.max(0.15, ...testCurve.slice(0, 3), ...trainCurve.slice(0, 3)) * 1.1]);
    E.line(xs, bounds, { color: "#888", dash: "6 4", width: 2, label: "bound" });
    E.line(xs, trainCurve, { color: "#1f77b4", label: "training" });
    E.line(xs, testCurve, { color: "#d62728", label: "test" });
    E.vline(v + 1, { color: "#555", width: 1.5 });
    E.legend(undefined, "tr");
    // weights against score: dot area ∝ weight
    const Wp = boostPlots.w;
    Wp.clear();
    const score = new Float64Array(ab.M.nEx);
    for (let t = 0; t <= v; t++) {
      const s = ab.stumps[t], off = s.feature * ab.M.nEx;
      for (let i = 0; i < ab.M.nEx; i++) score[i] += s.alpha * (s.polarity * (ab.M.values[off + i] - s.threshold) > 0 ? 1 : -1);
    }
    let smax = 1;
    for (const s of score) smax = Math.max(smax, Math.abs(s));
    Wp.setDomain([-smax * 1.05, smax * 1.05], [-1.6, 1.6]);
    const wn = ab.history[v + 1] ?? ab.w, n = ab.M.nEx;
    Wp.mark(g => {
      for (let i = 0; i < n; i++) {
        const jit = ((i * 7919) % 97) / 97 - 0.5;
        const yv = (train!.y[i] > 0 ? 1 : -1) + 0.9 * jit;
        const r = Math.max(1.2, Math.min(16, 2.2 * Math.sqrt(wn[i] * n)));
        const ci = svgEl("circle", { cx: Wp.x(score[i]).toFixed(1), cy: Wp.y(yv).toFixed(1), r: r.toFixed(1) }, g);
        ci.style.fill = train!.y[i] > 0 ? C_POS : C_NEG; ci.style.fillOpacity = "0.55";
      }
    }, [] as []);
    Wp.vline(0, { color: "#333", width: 1.5 });
    const s = ab.stumps[v];
    ro.textContent = `round ${v + 1} of ${T}\n${featureText(s)}\nweighted error ε = ${s.error.toFixed(3)}\nvote α = ${s.alpha.toFixed(2)}\n` +
      `training error ${(100 * trainCurve[v]).toFixed(1)} %\ntest error     ${(100 * testCurve[v]).toFixed(1)} %\n` +
      `pool: ${pool.length} features`;
    note.textContent = "Bottom plot: every training example, at its score after this round; dot area = its weight for the next round. Mistakes (orange left of 0, blue right of 0) grow.";
  }

  let casPlot: Plot | null = null;
  function drawCascade(): void {
    left.replaceChildren();
    const tbl = htmlEl("table", "bd-table", "", left);
    const hd = htmlEl("tr", "", "", tbl);
    for (const h of ["stage", "stumps", "d", "f", "bg. reaching"]) htmlEl("th", "", h, hd);
    cascadeInfo.forEach((c, k) => {
      const tr = htmlEl("tr", "", "", tbl);
      for (const v of [String(k + 1), String(c.stumps), c.d.toFixed(3), c.f.toFixed(2), c.pass < 0.001 ? c.pass.toExponential(1) : c.pass.toFixed(3)]) htmlEl("td", "", v, tr);
    });
    htmlEl("div", "qb-note", "d, f: detection and false-positive rate of the stage on its own training set. bg. reaching: share of random background windows that passed all earlier stages.", left);
    if (!casPlot || !mid.contains(casPlot.svg)) {
      mid.replaceChildren();
      casPlot = new Plot(mid, { width: 470, height: 380, x: [0, 9], y: [1, 1e5], ylog: true, xlabel: "stage", ylabel: "windows of the test scene", xticks: [1, 2, 3, 4, 5, 6, 7, 8] });
    }
    casPlot.clear();
    if (!cascade || cascade.length < cascadeInfo.length || busy) {
      ro.textContent = "building …"; return;
    }
    if (!scene) scene = testScene(st.target, 5, { rotated: st.rotated && st.target === "faces" });
    if (!detResult) detResult = detect(scene.img, 256, 256, pool, cascade, null);
    const r = detResult;
    casPlot.setDomain([0.3, cascade.length + 0.7], [1, Math.max(10, r.windows * 1.5)]);
    casPlot.bars(r.reached.slice(0, cascade.length).map((_, k) => k + 1), r.reached.slice(0, cascade.length), { color: "#1f77b4", barWidth: 0.6 });
    casPlot.hline(Math.max(1, r.reached[cascade.length]), { color: "#d62728" });
    let D = 1, F = 1, feats = 0;
    cascadeInfo.forEach(c => { D *= c.d; F *= c.f; });
    for (const s of cascade) feats += s.stumps.length;
    ro.textContent = `${cascade.length} stages, ${feats} stumps in all\n` +
      `windows in the scene: ${r.windows}\n` +
      `pass all stages (red line): ${r.reached[cascade.length]}\n` +
      `features per window: ${(r.totalFeatures / r.windows).toFixed(2)}\n  (all stumps: ${feats})\n` +
      `product of stage d: ${D.toFixed(3)}\nproduct of stage f: ${F.toExponential(1)}`;
    note.textContent = "Each stage keeps almost all targets and rejects a share of the background; most windows are rejected by the first stage after one or two features.";
  }

  function drawDetect(): void {
    if (!scene || !detResult || !monoResult) return;
    const r = st.mono ? monoResult : detResult;
    left.replaceChildren(); mid.replaceChildren(); casPlot = null; boostPlots = null;
    const fig = htmlEl("figure", "", "", left);
    htmlEl("figcaption", "", `detections (${st.mono ? "no cascade" : "cascade"}); dashed: true targets`, fig);
    const c = htmlEl("canvas", "", "", fig);
    drawScalarImage(c, scene.img, 256, 256, { lo: 0, hi: 255 });
    const svg = svgEl("svg", { viewBox: "0 0 256 256", class: "bd-overlay" }, fig);
    svg.style.top = c.offsetTop + "px";
    const place = () => { svg.style.top = c.offsetTop + "px"; svg.style.height = c.offsetHeight + "px"; };
    requestAnimationFrame(place);
    for (const t of scene.targets) svgEl("rect", { x: t.x, y: t.y, width: t.size, height: t.size, fill: "none", stroke: "#fff", "stroke-width": 1.2, "stroke-dasharray": "3 2" }, svg);
    let falses = 0;
    for (const m of r.merged) {
      const ok = scene.targets.some(t => matches(m, t));
      if (!ok) falses++;
      svgEl("rect", { x: m.x, y: m.y, width: m.size, height: m.size, fill: "none", stroke: ok ? "#2ca02c" : "#d62728", "stroke-width": 2.5 }, svg);
    }
    const found = scene.targets.filter(t => r.merged.some(m => matches(m, t)));
    const missed = scene.targets.filter(t => !found.includes(t));
    // stage map
    const f2 = htmlEl("figure", "", "", mid);
    htmlEl("figcaption", "", st.mono ? "windows (scale 1): accepted (yellow) or rejected" : "windows (scale 1): stage that rejected them", f2);
    const c2 = htmlEl("canvas", "", "", f2);
    const nS = Math.max(1, cascade!.length);
    const lut = colormapLUT(viridis, 256);
    const id = new ImageData(256, 256);
    for (let i = 0; i < 256 * 256; i++) {
      const g = scene.img[i] * 0.5 + 60;
      let rgb = [g, g, g];
      const s = r.stageMap[i];
      if (st.showStages && Number.isFinite(s)) {
        const li = s < 0 ? 255 : Math.round(200 * s / nS);
        rgb = [lut[4 * li], lut[4 * li + 1], lut[4 * li + 2]];
      }
      id.data[4 * i] = rgb[0]; id.data[4 * i + 1] = rgb[1]; id.data[4 * i + 2] = rgb[2]; id.data[4 * i + 3] = 255;
    }
    c2.width = 256; c2.height = 256;
    c2.getContext("2d")!.putImageData(id, 0, 0);
    if (!st.mono) {
      const leg = htmlEl("div", "qb-note", "", mid);
      leg.textContent = `dark purple: rejected by stage 1 … green: by stage ${nS}; yellow: passed all stages. Each colour marks the centre of a 24 × 24 window.`;
    }
    const fpw = (x: DetectResult) => (x.totalFeatures / x.windows).toFixed(2);
    ro.textContent = `windows: ${r.windows} (4 scales)\n` +
      `targets found: ${found.length} of ${scene.targets.length}\n` +
      (missed.length ? `missed: ${missed.map(t => t.kind).join(", ")}\n` : "") +
      `false detections: ${falses}\n` +
      `features per window:\n  cascade      ${fpw(detResult)}\n  no cascade   ${fpw(monoResult)}\n  (one classifier, ${DEFAULTS.monoRounds} stumps)`;
    note.textContent = "Green: detections that match a target; red: false detections. Overlapping windows are merged.";
  }

  if (p0) applyPreset(p0); else render();
  if (p0) pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
  mounting = false;
}

autoMount(NAME, mount);
