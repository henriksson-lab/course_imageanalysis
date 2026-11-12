// Colour (feature-space) segmentation with k-means or a Gaussian mixture. Left: the colour
// image (synthetic, with known truth). Middle: its pixels as a 2D histogram in two of the
// feature dimensions (a*–b* or L*–a* for Lab; G–B or R–G for RGB), with the fitted components
// as projected 2σ ellipses (k-means: its centres). Right: the label image, the posterior of
// one component, or the pixels that are further than a Mahalanobis distance t from every
// component (low likelihood). Features: RGB (scaled to 0..100), Lab, or Lab plus position
// (x, y scaled by a weight: compact regions). Dice per class against the truth (each cluster
// is mapped to the class it overlaps most).
//
// <div data-demo="colourgmm" data-preset="shadow|rgblab|position|tissue|odd"></div>
import { autoMount, button, htmlEl, presets, slider, type Slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { QUALITATIVE, hexToRgb, viridis, colormapLUT } from "../lib/colour";
import { rng } from "../lib/random";
import { type Mixture, type CovType, eStep, mStep, fromCentres, initMixture, hardLabels, prepare, componentMahalanobis2, cov2 } from "../lib/gmm";
import { kmeans, kmeansPlusPlus, assignStep, updateStep } from "../lib/kmeans";
import { histogram2d } from "../lib/histogram";
import { N, makeImage, features, score, FEATURE_NAMES, type ImageId, type Features, type ColourImage } from "./model";

const NAME = "colourgmm";
const STYLE = `
[data-demo="${NAME}"] { gap: .6em 1em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin-bottom: .15em; }
[data-demo="${NAME}"] .cg-row { display: flex; gap: .8em; width: 100%; align-items: flex-start; }
[data-demo="${NAME}"] .cg-row > * { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .cg-ctl { display: flex; gap: .4em 1.2em; width: 100%; flex-wrap: wrap; align-items: flex-start; }
[data-demo="${NAME}"] .cg-ctl > div { flex: 1 1 340px; display: flex; flex-direction: column; gap: .3em; min-width: 0; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] .cg-lab { font-size: 14px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 16px; gap: .2em; }
.reveal [data-demo="${NAME}"] .cg-row { max-height: 300px; }
.reveal [data-demo="${NAME}"] .cg-row > * { max-width: 270px; }
.reveal [data-demo="${NAME}"] .cg-row > .cg-ro { max-width: none; flex: 1.6 1 0; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .45em; }
.reveal [data-demo="${NAME}"] .cg-ctl > div { gap: .15em; }
@media (max-width: 640px) { [data-demo="${NAME}"] .cg-row { flex-wrap: wrap; } [data-demo="${NAME}"] .cg-row > * { flex-basis: 45%; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Algo = "kmeans" | "gmm";
type View = "labels" | "posterior" | "outliers";
interface Preset { id: string; label: string; img: ImageId; feat: Features; k: number; algo: Algo; cov: CovType; seed: number; wxy?: number; view?: View; proj?: number; t?: number }
const PRESETS: Preset[] = [
  { id: "shadow", label: "shadowed ball: k-means splits it, full GMM keeps it", img: "scene", feat: "rgb", k: 4, algo: "kmeans", cov: "full", seed: 5 },
  { id: "rgblab", label: "RGB vs Lab", img: "scene", feat: "lab", k: 4, algo: "kmeans", cov: "full", seed: 5, proj: 1 },
  { id: "position", label: "add position: compact regions", img: "scene", feat: "labxy", k: 6, algo: "kmeans", cov: "full", seed: 1, wxy: 1 },
  { id: "tissue", label: "stained tissue: three stains", img: "tissue", feat: "lab", k: 3, algo: "gmm", cov: "full", seed: 1 },
  { id: "odd", label: "an odd object has low likelihood", img: "odd", feat: "lab", k: 4, algo: "gmm", cov: "full", seed: 2, view: "outliers", t: 4.5 },
];
const COLS = QUALITATIVE.slice(0, 6), RGB = COLS.map(h => hexToRgb(h).map(v => 255 * v)), VIR = colormapLUT(viridis, 256);
const f2 = (v: number) => v.toFixed(2);
const PROJ: Record<Features, [number, number][]> = { rgb: [[1, 2], [0, 1]], lab: [[1, 2], [0, 1]], labxy: [[1, 2], [3, 4]] };
const RANGE: Record<string, [number, number]> = { R: [0, 100], G: [0, 100], B: [0, 100], "L*": [0, 100], "a*": [-60, 80], "b*": [-70, 70], x: [0, 100], y: [0, 100] };

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    img: p0.img as ImageId, feat: p0.feat as Features, k: p0.k, algo: p0.algo as Algo, cov: p0.cov as CovType, seed: p0.seed, wxy: p0.wxy ?? 0.5,
    view: (p0.view ?? "labels") as View, proj: p0.proj ?? 0, cls: 0, tmaha: 3.5,
    m: null as Mixture | null, centres: null as Float64Array | null, iter: 0, done: false, crit: NaN, best: "",
  };
  let image: ColourImage = makeImage(st.img), F = features(image, st.feat, st.wxy), timer = 0;

  // ---- layout ----
  const row = htmlEl("div", "cg-row", "", host);
  const fig0 = htmlEl("figure", "", "", row);
  htmlEl("figcaption", "", "image (synthetic)", fig0);
  const cv0 = htmlEl("canvas", "", "", fig0); cv0.width = N; cv0.height = N;
  const figP = htmlEl("figure", "", "", row);
  const capP = htmlEl("figcaption", "", "", figP);
  const pp = new Plot(figP, { width: 300, height: 300, x: [-60, 80], y: [-70, 70], font: 17, margin: { l: 50, b: 40, t: 6, r: 6 } });
  const heat = pp.heatmap(new Float64Array(64 * 64), 64, 64, { map: t => { const v = 1 - 0.85 * t; return [v, v, v]; }, flipY: false });
  const ells = [0, 1, 2, 3, 4, 5].map(j => pp.ellipse([0, 0], [[1, 0], [0, 1]], 2, { color: COLS[j], width: 3, fillOpacity: 0.08 }));
  const cents = pp.scatter([], [], { r: 8, stroke: "#000" });
  const fig1 = htmlEl("figure", "", "", row);
  const cap1 = htmlEl("figcaption", "", "", fig1);
  const cv1 = htmlEl("canvas", "", "", fig1); cv1.width = N; cv1.height = N;
  const roWrap = htmlEl("div", "cg-ro", "", row);
  const ro = htmlEl("div", "qb-readout", "", roWrap);

  const ctl = htmlEl("div", "cg-ctl", "", host);
  const c1 = htmlEl("div", "", "", ctl), c2 = htmlEl("div", "", "", ctl);
  const r1 = htmlEl("div", "qb-presets", "", c1);
  button(r1, "step", () => { stop(); stepOnce(); draw(); });
  const runBtn = button(r1, "run", () => { if (timer) stop(); else run(); });
  button(r1, "restart", () => { stop(); start(); });
  button(r1, "new start", () => { stop(); st.seed++; start(); });
  button(r1, "best of 6 starts", () => { stop(); bestOf(6); });
  const r2 = htmlEl("div", "qb-presets", "", c1);
  htmlEl("span", "cg-lab", "fit:", r2);
  const algoBtns = (["kmeans", "gmm"] as Algo[]).map(a => button(r2, a === "kmeans" ? "k-means" : "Gaussian mixture", () => { st.algo = a; start(); }));
  htmlEl("span", "cg-lab", " Σ:", r2);
  const covBtns = (["spherical", "diagonal", "full"] as CovType[]).map(c => button(r2, c, () => { st.cov = c; if (st.algo === "gmm") start(); else draw(); }));
  const kS: Slider = slider(c1, "K", 2, 6, 1, st.k, v => { st.k = v; start(); });
  const r3 = htmlEl("div", "qb-presets", "", c2);
  htmlEl("span", "cg-lab", "features:", r3);
  const featBtns = (["rgb", "lab", "labxy"] as Features[]).map(f => button(r3, f === "rgb" ? "RGB" : f === "lab" ? "Lab" : "Lab + position", () => { st.feat = f; st.proj = 0; rebuild(); }));
  htmlEl("span", "cg-lab", " plot:", r3);
  const projBtns = [0, 1].map(i => button(r3, "", () => { st.proj = i; draw(); }));
  const wS: Slider = slider(c2, "position weight", 0.1, 2, 0.1, st.wxy, v => { st.wxy = v; if (st.feat === "labxy") rebuild(); });
  const r4 = htmlEl("div", "qb-presets", "", c2);
  htmlEl("span", "cg-lab", "image:", r4);
  const imgBtns = (["scene", "odd", "tissue"] as ImageId[]).map(i => button(r4, i === "scene" ? "scene" : i === "odd" ? "scene + odd object" : "stained tissue", () => { st.img = i; rebuild(); }));
  htmlEl("span", "cg-lab", " show:", r4);
  const viewBtns = (["labels", "posterior", "outliers"] as View[]).map(v => button(r4, v === "outliers" ? "d_M > t to all" : v === "posterior" ? "posterior (click again: next)" : v, () => { if (v === "posterior" && st.view === "posterior") st.cls = (st.cls + 1) % st.k; st.view = v; draw(); }));
  const tS: Slider = slider(c2, "t (Mahalanobis)", 2, 6, 0.25, st.tmaha, v => { st.tmaha = v; draw(); });
  const presetBtns = presets(c1, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(c1, "Reset", () => load(p0)).style.alignSelf = "flex-start";

  // ---- model ----
  const n = N * N;
  function rebuild(): void { image = makeImage(st.img); F = features(image, st.feat, st.wxy); start(); }
  function load(p: Preset): void {
    stop();
    Object.assign(st, { img: p.img, feat: p.feat, k: p.k, algo: p.algo, cov: p.cov, seed: p.seed, wxy: p.wxy ?? 0.5, view: p.view ?? "labels", proj: p.proj ?? 0, cls: 0, tmaha: p.t ?? 3.5 });
    kS.set(st.k); wS.set(st.wxy); tS.set(st.tmaha);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    image = makeImage(st.img); F = features(image, st.feat, st.wxy);
    start();
    fitToEnd();
  }
  function start(): void {
    stop();
    const C = kmeansPlusPlus(F.X, n, F.d, st.k, rng(st.seed));
    st.centres = C; st.iter = 0; st.done = false; st.best = "";
    st.m = st.algo === "gmm" ? fromCentres(F.X, n, F.d, st.k, C, { covType: st.cov, floor: 0.01 }) : null;
    st.crit = NaN;
    draw();
  }
  function stepOnce(): void {
    if (st.done) return;
    if (st.algo === "kmeans") {
      const a = assignStep(F.X, n, F.d, st.centres!, st.k);
      const u = updateStep(F.X, n, F.d, a.assign, st.k, undefined, st.centres!);
      const b = assignStep(F.X, n, F.d, u.centres, st.k);
      st.done = b.assign.every((v, i) => v === a.assign[i]);
      st.centres = u.centres; st.crit = b.J;
    } else {
      const e = eStep(st.m!, F.X, n);
      st.m = mStep(F.X, n, F.d, e.resp, st.k, { covType: st.cov, floor: 0.01 }, undefined, st.m!);
      const L = eStep(st.m, F.X, n).logLik;
      st.done = Math.abs(L - st.crit) < 1e-3;
      st.crit = L;
    }
    st.iter++;
  }
  function fitToEnd(): void { for (let i = 0; i < 300 && !st.done; i++) stepOnce(); draw(); }
  function run(): void {
    runBtn.textContent = "pause";
    timer = window.setInterval(() => { for (let i = 0; i < 2; i++) stepOnce(); draw(); if (st.done || st.iter > 300) stop(); }, 30);
  }
  function stop(): void { if (timer) { clearInterval(timer); timer = 0; } runBtn.textContent = "run"; }
  function bestOf(R: number): void {
    const vals: string[] = [];
    let bestScore = Infinity, bestState: { m: Mixture | null; c: Float64Array | null; crit: number; seed: number } | null = null;
    for (let t = 0; t < R; t++) {
      const seed = 100 + t;
      if (st.algo === "kmeans") {
        const r = kmeans(F.X, n, F.d, st.k, { init: kmeansPlusPlus(F.X, n, F.d, st.k, rng(seed)) });
        vals.push(r.J.toExponential(3));
        if (r.J < bestScore) { bestScore = r.J; bestState = { m: null, c: r.centres, crit: r.J, seed }; }
      } else {
        let m = initMixture(F.X, n, F.d, st.k, "kmeans++", rng(seed), { covType: st.cov, floor: 0.01 });
        let L = -Infinity;
        for (let it = 0; it < 150; it++) {
          const e = eStep(m, F.X, n);
          if (Math.abs(e.logLik - L) < 1e-3) break;
          L = e.logLik;
          m = mStep(F.X, n, F.d, e.resp, st.k, { covType: st.cov, floor: 0.01 }, undefined, m);
        }
        L = eStep(m, F.X, n).logLik;
        vals.push(L.toFixed(0));
        if (-L < bestScore) { bestScore = -L; bestState = { m, c: null, crit: L, seed }; }
      }
    }
    if (bestState) { st.m = bestState.m; st.centres = bestState.c ?? st.centres; st.crit = bestState.crit; st.done = true; }
    st.best = (st.algo === "kmeans" ? "J of 6 starts: " : "log p of 6 starts: ") + vals.join(", ");
    draw();
  }

  // ---- drawing ----
  function labelsNow(): { lab: Int32Array; post: Float64Array | null; out: Uint8Array | null } {
    if (st.algo === "kmeans" || !st.m) {
      const a = assignStep(F.X, n, F.d, st.centres!, st.k);
      let out: Uint8Array | null = null;
      if (st.view === "outliers") {
        // k-means has no covariance: use the distance to the nearest centre in units of the
        // overall RMS cluster spread
        const s2 = a.J / (n * F.d);
        out = new Uint8Array(n);
        for (let p = 0; p < n; p++) { let q = 0; for (let t = 0; t < F.d; t++) q += (F.X[p * F.d + t] - st.centres![a.assign[p] * F.d + t]) ** 2; out[p] = q / s2 > st.tmaha * st.tmaha ? 1 : 0; }
      }
      return { lab: a.assign, post: null, out };
    }
    const e = eStep(st.m, F.X, n);
    let out: Uint8Array | null = null;
    if (st.view === "outliers") {
      const prep = prepare(st.m), t2 = st.tmaha * st.tmaha;
      out = new Uint8Array(n);
      for (let p = 0; p < n; p++) out[p] = componentMahalanobis2(st.m, F.X.subarray(p * F.d, p * F.d + F.d), prep).every(v => v > t2) ? 1 : 0;
    }
    return { lab: hardLabels(e.resp, n, st.k), post: e.resp, out };
  }

  function draw(): void {
    // image
    const ctx0 = cv0.getContext("2d")!, id0 = ctx0.createImageData(N, N);
    for (let p = 0; p < n; p++) { for (let q = 0; q < 3; q++) id0.data[4 * p + q] = 255 * image.rgb[3 * p + q]; id0.data[4 * p + 3] = 255; }
    ctx0.putImageData(id0, 0, 0);
    // result
    const L = labelsNow(), ctx1 = cv1.getContext("2d")!, id1 = ctx1.createImageData(N, N);
    const cls = Math.min(st.cls, st.k - 1);
    let flagged = 0;
    for (let p = 0; p < n; p++) {
      let c: number[];
      if (st.view === "posterior" && L.post) { const t = Math.round(255 * L.post[p * st.k + cls]); c = [VIR[4 * t], VIR[4 * t + 1], VIR[4 * t + 2]]; }
      else if (st.view === "outliers" && L.out) {
        const g = 255 * (0.3 * image.rgb[3 * p] + 0.59 * image.rgb[3 * p + 1] + 0.11 * image.rgb[3 * p + 2]);
        c = L.out[p] ? [255, 0, 170] : [g, g, g];
        flagged += L.out[p];
      } else c = RGB[L.lab[p]];
      id1.data[4 * p] = c[0]; id1.data[4 * p + 1] = c[1]; id1.data[4 * p + 2] = c[2]; id1.data[4 * p + 3] = 255;
    }
    ctx1.putImageData(id1, 0, 0);
    cap1.textContent = st.view === "labels" ? "label per pixel" : st.view === "posterior" ? (L.post ? `posterior of component ${cls + 1}` : "posterior (GMM only)") : `d_M > ${st.tmaha} to every component: ${flagged} px`;
    // projection plot
    const names = FEATURE_NAMES[st.feat], [i, j] = PROJ[st.feat][st.proj], ni = names[i], nj = names[j];
    const xr = st.feat === "labxy" && i >= 3 ? [0, 100 * st.wxy] as [number, number] : RANGE[ni], yr = st.feat === "labxy" && j >= 3 ? [0, 100 * st.wxy] as [number, number] : RANGE[nj];
    const xs = new Float64Array(n), ys = new Float64Array(n);
    for (let p = 0; p < n; p++) { xs[p] = F.X[p * F.d + i]; ys[p] = F.X[p * F.d + j]; }
    const H = histogram2d(xs, ys, { binsX: 64, binsY: 64, minX: xr[0], maxX: xr[1], minY: yr[0], maxY: yr[1] });
    pp.setDomain(xr, yr);
    pp.setLabels({ xlabel: undefined, ylabel: undefined });
    heat.set(H.counts.map(v => Math.log1p(v)));
    capP.textContent = `pixels: ${ni} across, ${nj} up`;
    projBtns.forEach((b, q) => { const [a, c] = PROJ[st.feat][q]; b.textContent = `${names[a]}–${names[c]}`; b.classList.toggle("active", q === st.proj); });
    ells.forEach((e, q) => {
      if (st.algo === "gmm" && st.m && q < st.k) {
        e.set([st.m.means[q * F.d + i], st.m.means[q * F.d + j]], cov2(st.m, q, i, j));
        e.g.style.display = "";
      } else e.g.style.display = "none";
    });
    if (st.algo === "kmeans" && st.centres) cents.set(Array.from({ length: st.k }, (_, q) => st.centres![q * F.d + i]), Array.from({ length: st.k }, (_, q) => st.centres![q * F.d + j]), COLS.slice(0, st.k));
    else cents.set([], []);
    // readout
    const sc = score(L.lab, image.truth, st.k, image.classNames.length);
    let s = `${st.algo === "kmeans" ? "k-means" : `GMM (${st.cov})`}, ${st.feat === "rgb" ? "RGB" : st.feat === "lab" ? "Lab" : "Lab + position"}, K = ${st.k}, start ${st.seed}\n`;
    s += `iteration ${st.iter}${st.done ? ", converged" : ""}   ` + (Number.isFinite(st.crit) ? (st.algo === "kmeans" ? `J = ${st.crit.toExponential(4)}` : `log p(D) = ${st.crit.toFixed(0)}`) : "") + "\n";
    s += "Dice per class (each cluster → the class it overlaps most):\n";
    s += image.classNames.map((nm, c) => `  ${nm}: ${f2(sc.dice[c])}`).join("\n") + `\n  mean: ${f2(sc.mean)}`;
    s += "\nclusters → classes: " + sc.map.map((c, q) => `${q + 1}→${image.classNames[c]}`).join(", ");
    if (st.best) s += "\n" + st.best;
    ro.textContent = s;
    // buttons
    algoBtns.forEach((b, q) => b.classList.toggle("active", ["kmeans", "gmm"][q] === st.algo));
    covBtns.forEach((b, q) => { b.classList.toggle("active", ["spherical", "diagonal", "full"][q] === st.cov && st.algo === "gmm"); b.disabled = st.algo !== "gmm"; });
    featBtns.forEach((b, q) => b.classList.toggle("active", ["rgb", "lab", "labxy"][q] === st.feat));
    imgBtns.forEach((b, q) => b.classList.toggle("active", ["scene", "odd", "tissue"][q] === st.img));
    viewBtns.forEach((b, q) => b.classList.toggle("active", ["labels", "posterior", "outliers"][q] === st.view));
    (wS.input.parentElement as HTMLElement).style.opacity = st.feat === "labxy" ? "1" : "0.4";
    (tS.input.parentElement as HTMLElement).style.display = st.view === "outliers" ? "" : "none";
  }

  load(p0);
}

autoMount(NAME, mount);
