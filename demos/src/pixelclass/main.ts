// Pixel classification from painted strokes (the paint – train – look – correct loop).
// Left: the image with the painted labels (paint with the mouse; brush, erase, undo, clear).
// Middle: the result: segmentation, the probability of one class, the uncertainty
// (1 − largest probability), one channel of the feature stack, or the ground truth (test
// image). Right: presets, the classifier and its settings, the feature families and scales,
// class balancing, a held-out region, and the readout (training pixels, built-in OOB error,
// accuracy on the held-out region, and on the test image Dice and the confusion matrix
// against the ground truth). Every pixel's feature vector comes from the multiscale stack of
// lib/featurestack.ts; the classifier is any `Classifier` of lib/classifier.ts.
//
// Auto-mounts on <div data-demo="pixelclass">. Options:
//   data-preset      one of the ids in presets.ts (default "two")
//   data-image       "synth" (test image with ground truth) | "cell" (micrograph); overrides the preset
//   data-classifier  id of the classifier to start with (default "forest")
//   data-classifiers comma-separated ids offered in the switch (default: all registered)
//   data-stack       comma-separated feature families (default intensity,edges,blobs,ridges,texture)
//   data-size        256 (default) or 128 (faster; e.g. for slower classifiers)
//   data-tabs        comma-separated ids of plug-in tabs to show (default none)
//
// ======================================================================================
// PLUG-IN API (for other days: SVM, integral images, neural networks, ...)
// ======================================================================================
// 1. A classifier: implement `Classifier` (lib/classifier.ts) in your own lib file, e.g.
//    `export const SVM_CLASSIFIER: Classifier = { id: "svm", name: "SVM (RBF)", params: [...],
//    train(data, params, seed) { ...; return model; } }`, where model.predict(X, n, out,
//    from, to) fills n × k scores in [0, 1] (rows sum to 1) for the samples from..to-1 of the
//    FEATURE-MAJOR X (X[j * n + i]). Then add ONE import and ONE registerClassifier(...) line
//    to the PLUGINS block below. It appears in the switch; data-classifier="svm" selects it,
//    data-classifiers="forest,svm" limits the switch. Parameters with `maxIsD: true` are
//    clamped to the number of channels; `log: true` sliders hold log10 of the value.
//    The shell standardises nothing: a classifier that needs standardised features does it
//    inside train()/predict() (that contrast with the forest is a lesson of the SVM day).
// 2. A tab (an extra panel section, e.g. the SVM day's cross-validate tab): implement
//    `PixelclassTab` (pixelclass/plugins.ts) and add registerPixelclassTab(...) to the
//    PLUGINS block; data-tabs="cv" shows it. Its mount() gets a ShellContext with the stack,
//    the training set (groups = stroke ids, for by-stroke folds), the classifier, its
//    parameters (get and set) and an onTrained() hook.
// 3. A feature family (e.g. the integral-images day's box mean / local variance group):
//    registerFeatureGroup(...) from lib/featurestack.ts in the PLUGINS block, and list its
//    id in data-stack (e.g. data-stack="intensity,box").
// ======================================================================================
import { htmlEl, slider, presets, button, checkbox, autoMount, Slider } from "../lib/ui";
import { registerClassifier, getClassifier, listClassifiers, Classifier, Model, defaultParams, labelsAndUncertainty } from "../lib/classifier";
import { FOREST_CLASSIFIER, TREE_CLASSIFIER } from "../lib/forest";
import { featureStack, selectChannels, FeatureStack, channelImage } from "../lib/featurestack";
import { LabelPainter } from "../lib/paint";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Grey } from "../lib/imageops";
import { hexToRgb, colormapLUT, viridis, inferno } from "../lib/colour";
import { synthScene, SCENE_CLASSES } from "./scene";
import { PIXEL_PRESETS, PixelPreset, strokesToMap, HOLDOUT_X, ALL_FAMILIES, SMALL, LARGE } from "./presets";
import { trainingSet, regionScores, inRegion, TrainingInfo } from "./pipeline";
import { ShellContext, getPixelclassTab } from "./plugins";
import { registerPixelclassTab } from "./plugins";
import { SVM_CLASSIFIER } from "../lib/svm";
import { SVM_CV_TAB } from "../svm2d/pixeltab";

// ---------------------------------------------------------------- PLUGINS (other days add lines here)
registerClassifier(FOREST_CLASSIFIER);
registerClassifier(TREE_CLASSIFIER);
registerClassifier(SVM_CLASSIFIER);       // SVM day (lib/svm.ts)
registerPixelclassTab(SVM_CV_TAB);        // SVM day: data-tabs="svmcv"
// ---------------------------------------------------------------- end of PLUGINS

const NAME = "pixelclass";
const CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"];
const CELL_CLASSES = ["background", "nucleus", "cytoplasm", "other"];
const CHUNK_ROWS = 16;

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .pc-col { flex: 0 1 340px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-paint { flex: none; width: 100%; }
[data-demo="${NAME}"] .pc-col > .qb-panel { flex: none; }
[data-demo="${NAME}"] .qb-paint-canvas { cursor: crosshair; }
[data-demo="${NAME}"] .pc-view { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
[data-demo="${NAME}"] .pc-cap { font-size: 15px; text-align: center; color: var(--qb-muted); }
[data-demo="${NAME}"] .pc-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 340px; gap: .35em; }
[data-demo="${NAME}"] .pc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .pc-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .05em .4em; font-size: .9em; }
[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre; overflow-x: auto; flex: none; }
[data-demo="${NAME}"] .pc-small .qb-btn { font-size: 14px; padding: .05em .4em; }
[data-demo="${NAME}"] .pc-tab { border-top: 1px solid var(--qb-grid); padding-top: .3em; }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .pc-col { flex: 0 0 300px; }
.reveal [data-demo="${NAME}"] > .qb-panel { gap: .2em; max-height: 570px; overflow-y: auto; flex: 1 1 380px; max-width: 440px; font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .05em .4em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .05em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 12px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type View = "seg" | "prob" | "unc" | "feature" | "truth";
interface ImageEntry { img: Grey; truth: Uint8Array | null; scene: ReturnType<typeof synthScene> | null; full: FeatureStack | null }

const pct = (v: number) => (Number.isFinite(v) ? (100 * v).toFixed(1) + "%" : "–");

function mount(host: HTMLElement): void {
  injectStyle();
  const size = host.dataset.size === "128" ? 128 : 256, N = size * size;
  const offered = (host.dataset.classifiers ?? "").split(",").map(s => s.trim()).filter(Boolean);
  const clfs = listClassifiers().filter(c => !offered.length || offered.includes(c.id));
  const families = (host.dataset.stack ?? ALL_FAMILIES.join(",")).split(",").map(s => s.trim()).filter(Boolean);
  const p0 = PIXEL_PRESETS.find(p => p.id === host.dataset.preset) ?? PIXEL_PRESETS[0];
  const startClf = getClassifier(host.dataset.classifier ?? "") ?? clfs[0] ?? FOREST_CLASSIFIER;
  const st = {
    image: (host.dataset.image as "synth" | "cell") ?? p0.image,
    fam: new Set(p0.families.filter(f => families.includes(f))), small: p0.small, large: p0.large,
    balance: p0.balance, heldout: p0.heldout, auto: true, view: p0.view as View,
    probClass: 1, featIdx: 0, clf: startClf as Classifier, params: defaultParams(startClf),
  };
  if (!st.fam.size) families.forEach(f => st.fam.add(f));
  const images: Record<string, ImageEntry | undefined> = {};
  let stack: FeatureStack | null = null, stackKey = "";
  let model: Model | null = null, info: TrainingInfo | null = null, P: Float32Array | null = null, done = 0, job = 0;
  let pred: Uint8Array | null = null, unc: Float32Array | null = null, trainMs = 0;
  const trainedHooks: (() => void)[] = [];

  // ---- layout
  const colL = htmlEl("div", "pc-col", "", host);
  const painterBox = htmlEl("div", "", "", colL);
  const painterCtl = htmlEl("div", "qb-panel pc-small", "", colL);
  const credit = htmlEl("div", "pc-credit", "", colL);
  const colM = htmlEl("div", "pc-col", "", host);
  const viewRow = htmlEl("div", "qb-presets pc-small", "", colM);
  const viewBtns: Record<View, HTMLButtonElement> = {} as Record<View, HTMLButtonElement>;
  ([["seg", "segmentation"], ["prob", "probability"], ["unc", "uncertainty"], ["feature", "features"], ["truth", "truth"]] as [View, string][])
    .forEach(([v, l]) => { viewBtns[v] = button(viewRow, l, () => { st.view = v; syncView(); drawView(); }); });
  const subRow = htmlEl("div", "qb-presets pc-small", "", colM);
  const viewCv = htmlEl("canvas", "pc-view", "", colM);
  viewCv.width = size; viewCv.height = size;
  const viewCap = htmlEl("div", "pc-cap", "", colM);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "pc-lab", "presets", panel);
  const presetBtns = presets(panel, PIXEL_PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  // classifier switch and its parameters
  let clfBtns: HTMLButtonElement[] = [];
  if (clfs.length > 1) {
    htmlEl("div", "pc-lab", "classifier", panel);
    clfBtns = presets(panel, clfs.map(c => ({ label: c.name, apply: () => { st.clf = c; st.params = defaultParams(c); buildParams(); retrain(); } })));
  }
  const paramBox = htmlEl("div", "", "", panel);
  htmlEl("div", "pc-lab", "feature families", panel);
  const famBox = htmlEl("div", "pc-grid", "", panel);
  const famCbs = families.map(f => checkbox(famBox, f, st.fam.has(f), v => { if (v) st.fam.add(f); else st.fam.delete(f); restack(); }));
  const scaleBox = htmlEl("div", "pc-grid", "", panel);
  const smallCb = checkbox(scaleBox, "σ ≤ 1.6", st.small, v => { st.small = v; restack(); });
  const largeCb = checkbox(scaleBox, "σ ≥ 3.5", st.large, v => { st.large = v; restack(); });
  const balCb = checkbox(panel, "balance classes", st.balance, v => { st.balance = v; retrain(); });
  const hoCb = checkbox(panel, "hold out the right part", st.heldout, v => { st.heldout = v; drawPainterOverlay(); retrain(); });
  const trainRow = htmlEl("div", "qb-presets", "", panel);
  const autoCb = checkbox(trainRow, "train after each stroke", st.auto, v => { st.auto = v; });
  button(trainRow, "train", () => retrain());
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Paint with the mouse; the classifier retrains when you lift it.", panel);

  // plug-in tabs
  const ctx: ShellContext = {
    stack: () => stack, training: () => info?.data ?? null, classifier: () => st.clf, params: () => ({ ...st.params }),
    setParams: p => { Object.assign(st.params, p); buildParams(); retrain(); }, model: () => model,
    classNames: () => classNames(), width: () => size, height: () => size, retrain: () => retrain(),
    onTrained: fn => { trainedHooks.push(fn); },
  };
  for (const id of (host.dataset.tabs ?? "").split(",").map(s => s.trim()).filter(Boolean)) {
    const t = getPixelclassTab(id);
    if (!t) continue;
    const box = htmlEl("div", "pc-tab", "", panel);
    htmlEl("div", "pc-lab", t.label, box);
    t.mount(box, ctx);
  }

  let paramSliders: Slider[] = [];
  function buildParams(): void {
    paramBox.replaceChildren();
    clfBtns.forEach((b, i) => b.classList.toggle("active", clfs[i] === st.clf));
    paramSliders = st.clf.params.map(ps => {
      const max = ps.maxIsD && stack ? Math.min(ps.max, stack.d) : ps.max;
      const v = Math.min(max, st.params[ps.key] ?? ps.value);
      st.params[ps.key] = v;
      const fmt = ps.log ? (x: number) => Math.pow(10, x).toPrecision(2) : undefined;
      return slider(paramBox, ps.label, ps.min, max, ps.step, v, x => { st.params[ps.key] = x; retrain(); }, fmt);
    });
  }

  // ---- painter
  let painter: LabelPainter | null = null, painterImage = "";
  function classNames(): string[] { return st.image === "synth" ? SCENE_CLASSES : CELL_CLASSES; }
  function makePainter(): void {
    painterBox.replaceChildren(); painterCtl.replaceChildren();
    painter = new LabelPainter(painterBox, {
      width: size, height: size, brush: Math.round(2 * size / 256), maxBrush: 8, controlsParent: painterCtl, caption: "paint labels here",
      labels: classNames().slice(0, st.image === "synth" ? 3 : 4).map((n, i) => ({ value: i + 1, name: n, color: CLASS_COLORS[i] })),
      onChange: (_, final) => { if (final && st.auto) retrain(); },
    });
    painterImage = st.image;
    credit.textContent = st.image === "cell" ? CELL_CREDIT : "Synthetic test image (seeded) with ground truth: background, nuclei (some dim), granular cells.";
  }

  function drawPainterOverlay(): void {
    if (!painter) return;
    if (!st.heldout) { painter.setOverlay(null); return; }
    const o = new ImageData(size, size), x0 = Math.round(HOLDOUT_X * size);
    for (let y = 0; y < size; y++) for (let x = x0; x < size; x++) {
      const q = 4 * (y * size + x), edge = x - x0 < Math.max(1, size / 128);
      o.data[q] = 255; o.data[q + 1] = 220; o.data[q + 2] = 0; o.data[q + 3] = edge ? 255 : 40;
    }
    painter.setOverlay(o);
  }

  // ---- images and stacks
  async function ensureImage(which: "synth" | "cell"): Promise<ImageEntry> {
    const have = images[which];
    if (have) return have;
    let e: ImageEntry;
    if (which === "synth") { const sc = synthScene(size, 1); e = { img: sc.image, truth: sc.truth, scene: sc, full: null }; }
    else e = { img: await loadCell(size === 256 ? undefined : size), truth: null, scene: null, full: null };
    images[which] = e;
    return e;
  }

  function currentStack(e: ImageEntry): FeatureStack {
    if (!e.full) e.full = featureStack(e.img, families, [...SMALL, ...LARGE]);
    const key = st.image + "|" + [...st.fam].sort().join(",") + "|" + st.small + st.large;
    if (key !== stackKey || !stack) {
      stack = selectChannels(e.full, (g, s) => st.fam.has(g) && (s <= 1.6 ? st.small : st.large));
      stackKey = key;
    }
    return stack;
  }

  function restack(): void { stack = null; retrain(); }

  // ---- training and progressive prediction
  function retrain(sync = false): void {
    const myJob = ++job;
    const e = images[st.image];
    if (!e || !painter) return;
    if (!e.full) ro.textContent = "computing the feature stack …";
    const run = () => {
      if (myJob !== job) return;
      const S = currentStack(e);
      if (!S.d) { ro.textContent = "choose at least one feature family and scale"; model = null; P = null; drawView(); return; }
      buildParamsIfNeeded(S);
      const exclude = st.heldout ? (p: number) => inRegion(p, size, HOLDOUT_X) : undefined;
      info = trainingSet(S, painter!.labels, { exclude, cap: 4000, balance: st.balance, seed: 1 });
      if (!info) { model = null; P = null; pred = null; ro.textContent = "paint at least two classes"; drawView(); return; }
      const t0 = performance.now();
      model = st.clf.train(info.data, st.params, 1);
      trainMs = performance.now() - t0;
      const k = info.data.k;
      P = new Float32Array(N * k); done = 0; pred = null; unc = null;
      // predict in chunks of rows for about 40 ms at a time, so painting never blocks
      const step = () => {
        if (myJob !== job || !model || !P) return;
        const t1 = performance.now();
        while (done < N && (sync || performance.now() - t1 < 40)) {
          const to = Math.min(N, done + CHUNK_ROWS * size);
          model.predict(S.data, N, P, done, to);
          done = to;
        }
        drawView();
        if (done < N) setTimeout(step, 0);
        else finish();
      };
      readout(true);
      if (sync) step(); else setTimeout(step, 0);
    };
    if (sync) run(); else setTimeout(run, e.full ? 0 : 30);
  }

  let lastD = -1;
  function buildParamsIfNeeded(S: FeatureStack): void { if (S.d !== lastD) { lastD = S.d; buildParams(); } }

  function finish(): void {
    if (!P || !info) return;
    const r = labelsAndUncertainty(P, N, info.data.k);
    pred = r.label; unc = r.uncertainty;
    drawView(); readout(false);
    trainedHooks.forEach(f => f());
  }

  // ---- drawing the result view
  const vctx = viewCv.getContext("2d")!;
  const lutV = colormapLUT(viridis, 256), lutI = colormapLUT(inferno, 256);
  const rgb = CLASS_COLORS.map(h => hexToRgb(h).map(v => 255 * v));
  function drawView(): void {
    const e = images[st.image];
    if (!e) return;
    const id = vctx.createImageData(size, size), D = id.data, img = e.img.data;
    const k = info?.data.k ?? 2;
    let chan: Float32Array | null = null, lo = 0, hi = 1;
    if (st.view === "feature" && stack && stack.d) {
      st.featIdx = Math.min(st.featIdx, stack.d - 1);
      chan = channelImage(stack, st.featIdx).data;
      const sorted = Float32Array.from(chan).sort();
      lo = sorted[Math.floor(0.01 * sorted.length)]; hi = sorted[Math.floor(0.99 * sorted.length)];
      if (hi <= lo) hi = lo + 1;
    }
    for (let p = 0; p < N; p++) {
      const g = img[p];
      let r = g, gg = g, b = g;
      const have = P && p < done;
      if (st.view === "seg" && have) {
        let best = 0; for (let c = 1; c < k; c++) if (P![p * k + c] > P![p * k + best]) best = c;
        const col = rgb[best]; r = 0.45 * g + 0.55 * col[0]; gg = 0.45 * g + 0.55 * col[1]; b = 0.45 * g + 0.55 * col[2];
      } else if (st.view === "prob" && have) {
        const v = st.probClass < k ? P![p * k + st.probClass] : 0, i = Math.round(255 * v);
        r = lutV[4 * i]; gg = lutV[4 * i + 1]; b = lutV[4 * i + 2];
      } else if (st.view === "unc" && have) {
        let m = 0; for (let c = 0; c < k; c++) m = Math.max(m, P![p * k + c]);
        const u = (1 - m) / (1 - 1 / k), i = Math.max(0, Math.min(255, Math.round(255 * u)));
        r = lutI[4 * i]; gg = lutI[4 * i + 1]; b = lutI[4 * i + 2];
      } else if (st.view === "feature" && chan) {
        const v = Math.max(0, Math.min(255, 255 * (chan[p] - lo) / (hi - lo)));
        r = gg = b = v;
      } else if (st.view === "truth" && e.truth) {
        const col = rgb[e.truth[p] - 1]; r = 0.45 * g + 0.55 * col[0]; gg = 0.45 * g + 0.55 * col[1]; b = 0.45 * g + 0.55 * col[2];
      }
      if (st.heldout && (p % size) === Math.round(HOLDOUT_X * size)) { r = 255; gg = 220; b = 0; }
      D[4 * p] = r; D[4 * p + 1] = gg; D[4 * p + 2] = b; D[4 * p + 3] = 255;
    }
    vctx.putImageData(id, 0, 0);
    const names = classNames();
    viewCap.textContent = st.view === "seg" ? "segmentation: colour = most probable class"
      : st.view === "prob" ? `probability of "${names[st.probClass] ?? "?"}" (dark 0 → yellow 1)`
      : st.view === "unc" ? "uncertainty 1 − max P (black: sure, yellow: undecided)"
      : st.view === "feature" ? (stack && stack.d ? `channel ${st.featIdx + 1}/${stack.d}: ${stack.names[st.featIdx]}` : "no channels")
      : e.truth ? "ground truth" : "no ground truth for the micrograph";
  }

  function syncView(): void {
    (Object.keys(viewBtns) as View[]).forEach(v => viewBtns[v].classList.toggle("active", v === st.view));
    viewBtns.truth.style.display = st.image === "synth" ? "" : "none";
    subRow.replaceChildren();
    if (st.view === "prob") {
      const k = info?.data.k ?? 2;
      presets(subRow, classNames().slice(0, k).map((n, c) => ({ label: n, apply: () => { st.probClass = c; drawView(); } })))
        .forEach((b, c) => b.classList.toggle("active", c === st.probClass));
    } else if (st.view === "feature") {
      button(subRow, "◀ channel", () => { if (stack) st.featIdx = (st.featIdx + stack.d - 1) % stack.d; drawView(); });
      button(subRow, "channel ▶", () => { if (stack) st.featIdx = (st.featIdx + 1) % stack.d; drawView(); });
    }
  }

  // ---- readout
  function readout(training: boolean): void {
    if (!info || !model) return;
    const names = classNames(), k = info.data.k;
    let s = `training pixels (${info.strokes} strokes):\n`;
    s += names.slice(0, k).map((n, c) => `  ${n.padEnd(14)} ${String(info!.painted[c]).padStart(5)}` + (info!.used[c] !== info!.painted[c] ? ` → ${info!.used[c]}` : "")).join("\n");
    s += `\n${st.clf.name}${model.summary ? ": " + model.summary() : ""}\nfeatures d = ${info.data.d} · trained in ${trainMs.toFixed(0)} ms`;
    const oob = model.oobError?.();
    if (oob !== undefined && oob !== null) s += `\nOOB accuracy (painted px)  ${pct(1 - oob)}`;
    if (training) { ro.textContent = s + "\npredicting …"; return; }
    const e = images[st.image]!;
    const region = st.heldout ? (p: number) => inRegion(p, size, HOLDOUT_X) : undefined;
    const sc = regionScores(pred!, painter!.labels, e.truth, e.truth ? 3 : k, region);
    const where = st.heldout ? "held-out region" : "whole image";
    if (st.heldout) s += `\nheld-out painted px: ${sc.paintedN ? pct(sc.paintedAcc) + ` of ${sc.paintedN}` : "none painted"}`;
    if (e.truth && sc.confusion) {
      s += `\ntruth, ${where}: accuracy ${pct(sc.truthAcc)}`;
      s += `\n  Dice ${["bg", "nuc", "gran"].map((n, c) => `${n} ${Number.isFinite(sc.dice[c]) ? sc.dice[c].toFixed(2) : "–"}`).join(" · ")}`;
      const tot = sc.confusion.reduce((a, b) => a + b, 0);
      s += `\n  truth \\ predicted (% of px)\n` + SCENE_CLASSES.map((n, t) => `  ${n.split(" ")[0].padEnd(11)}` + [0, 1, 2].map(q => (100 * sc.confusion![t * 3 + q] / tot).toFixed(1).padStart(6)).join("")).join("\n");
    }
    ro.textContent = s;
  }

  // ---- presets
  async function load(p: PixelPreset): Promise<void> {
    const image = (host.dataset.image as "synth" | "cell" | undefined) && p === p0 ? (host.dataset.image as "synth" | "cell") : p.image;
    st.image = image;
    st.fam = new Set(p.families.filter(f => families.includes(f)));
    if (!st.fam.size) families.forEach(f => st.fam.add(f));
    st.small = p.small; st.large = p.large; st.balance = p.balance; st.heldout = p.heldout; st.view = p.view;
    st.probClass = 1;
    famCbs.forEach((cb, i) => (cb.checked = st.fam.has(families[i])));
    smallCb.checked = st.small; largeCb.checked = st.large; balCb.checked = st.balance; hoCb.checked = st.heldout;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PIXEL_PRESETS[i] === p));
    ro.textContent = "loading …";
    const e = await ensureImage(image);
    if (!painter || painterImage !== image) makePainter();
    painter!.setImage(e.img.data);
    const strokes = e.scene ? p.strokes(e.scene) : p.image === "cell" ? p.strokes(null as never) : [];
    painter!.setLabels(strokesToMap(strokes, size, size));
    drawPainterOverlay();
    stack = null;
    syncView();
    drawView();
    retrain(true);
  }

  buildParams();
  void load(p0);
}

autoMount(NAME, mount);

