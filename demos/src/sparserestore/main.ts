// Restoration with a sparsity prior in 2D: denoising, deblurring and inpainting as one problem,
// min ½‖Ax − y‖² + λ·g(x), with different forward operators A (identity, Gaussian blur, a mask
// of known pixels) and priors g: none (least squares), Tikhonov (½‖∇x‖²), wavelet ℓ1 (Haar,
// db4, optional cycle spinning), total variation; with data-priors including "dct learned
// nlmeans": patch dictionaries (overcomplete 8×8 DCT, or a dictionary learned by K-SVD from
// another part of the image / the patchdict demo on the same page) and non-local means as a
// baseline. Solver: FISTA with the prior's proximal step (step 1, ‖A‖ ≤ 1), iterated
// progressively; patch priors use their own restoration (sparse coding of overlapping patches;
// for deblurring a gradient step alternated with patch denoising, "plug-and-play").
// Panels: observed, restored, error (restored − truth) and PSNR against the iteration.
// Auto-mounts on <div data-demo="sparserestore">. Optional: data-priors="none tikhonov haar db4 tv"
// (+ "dct learned nlmeans"), data-size="128|256", data-preset="…", data-layout="full".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { WAVELETS } from "../lib/wavelet";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { diverging } from "../lib/colour";
import { psnr } from "../lib/metrics";
import { tv2d } from "../lib/smooth";
import { fft2d } from "../lib/fft";
import { nonLocalMeans } from "../lib/denoise";
import { LinOp, blurOp2d, maskOp, identityOp, waveletProx2d, Vec } from "../lib/sparse";
import { dctDictionary, ksvdDenoise, patchInpaint, extractPatches, randomPatchDictionary, learnStep, getDictionary } from "../lib/patches";
import { rng } from "../lib/random";
import { imgFig, injectStyle, scheduler, progressive } from "../csmri/view";
import { shapesImage, textureImage, randomMask, scratchMask, textMask } from "./images";

const NAME = "sparserestore";
type Img = "cell" | "shapes" | "texture";
type Task = "denoise" | "deblur" | "random" | "scratch" | "text";
type Prior = "none" | "tikhonov" | "haar" | "db4" | "tv" | "dct" | "learned" | "nlmeans";
interface State { img: Img; task: Task; prior: Prior; noise: number; blur: number; missing: number; lam: number; spin: boolean; iters: number }
const DEF: State = { img: "shapes", task: "deblur", prior: "db4", noise: 5, blur: 2, missing: 0.5, lam: 0.7, spin: false, iters: 100 };
const PRIOR_LABEL: Record<Prior, string> = { none: "none (least squares)", tikhonov: "Tikhonov", haar: "Haar ℓ1", db4: "db4 ℓ1", tv: "TV", dct: "8×8 DCT patches", learned: "learned dictionary", nlmeans: "NL-means" };
const TASKS: { id: Task; label: string }[] = [
  { id: "denoise", label: "denoise" }, { id: "deblur", label: "deblur" }, { id: "random", label: "missing pixels" }, { id: "scratch", label: "scratches + hole" }, { id: "text", label: "remove text" },
];
interface Preset { id: string; label: string; s: Partial<State>; needs?: Prior }
const PRESETS: Preset[] = [
  { id: "tikhonov", label: "deblur: Tikhonov rings", s: { img: "shapes", task: "deblur", prior: "tikhonov", lam: 0.3, noise: 5, blur: 2 } },
  { id: "wavelet", label: "deblur: wavelet ℓ1 does not", s: { img: "shapes", task: "deblur", prior: "db4", lam: 0.7, noise: 5, blur: 2, spin: true } },
  { id: "tv", label: "TV: cartoon", s: { img: "cell", task: "deblur", prior: "tv", lam: 1.0, noise: 5, blur: 1.5 } },
  { id: "texture", label: "λ too large: texture gone", s: { img: "texture", task: "denoise", prior: "db4", lam: 1.9, noise: 10 } },
  { id: "missing", label: "50 % of the pixels missing", s: { img: "cell", task: "random", prior: "db4", missing: 0.5, lam: 0.5, noise: 0, spin: true, iters: 150 } },
  { id: "hole", label: "a scratch is fine, a large hole is not", s: { img: "shapes", task: "scratch", prior: "db4", lam: 0.5, noise: 0, spin: true, iters: 150 } },
  { id: "text", label: "remove the text", s: { img: "cell", task: "text", prior: "db4", lam: 0.5, noise: 0, spin: true, iters: 150 } },
  { id: "dctden", label: "patch DCT denoising", s: { img: "cell", task: "denoise", prior: "dct", noise: 20 }, needs: "dct" },
  { id: "learnden", label: "learned dictionary denoising", s: { img: "cell", task: "denoise", prior: "learned", noise: 20 }, needs: "learned" },
  { id: "nlm", label: "NL-means baseline", s: { img: "cell", task: "denoise", prior: "nlmeans", noise: 20 }, needs: "nlmeans" },
  { id: "patchinp", label: "patch inpainting, 50 %", s: { img: "cell", task: "random", prior: "learned", missing: 0.5, noise: 0 }, needs: "learned" },
];

const CSS = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.15; margin: 0 0 .15em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 2px; }
.qb-demo[data-demo="${NAME}"] .sr-left { flex: 0 1 620px; min-width: 280px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .sr-big { display: grid; grid-template-columns: 1fr 1fr; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .sr-row { display: grid; grid-template-columns: 1fr 3fr; gap: .4em; align-items: start; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .sr-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .sr-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"][data-layout="full"] .sr-left { flex-basis: 760px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .sr-left { flex-basis: 540px; max-width: 540px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 360px; max-width: 380px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 13px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .sr-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;

const lamOf = (s: number) => Math.pow(10, s);   // slider value → λ in grey levels

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const n = Number(host.dataset.size) === 256 ? 256 : 128;
  const priors = ((host.dataset.priors ?? "none tikhonov haar db4 tv").split(/[\s,]+/).filter(p => p in PRIOR_LABEL)) as Prior[];
  const presetList = PRESETS.filter(p => !p.needs || priors.includes(p.needs)).filter(p => priors.includes(p.s.prior as Prior));
  const p0 = presetList.find(p => p.id === host.dataset.preset);
  const init: State = { ...DEF, prior: priors.includes(DEF.prior) ? DEF.prior : priors[0], ...(p0?.s ?? {}) };
  const st: State = { ...init };

  const leftCol = htmlEl("div", "sr-left", "", host);
  const big = htmlEl("div", "sr-big", "", leftCol);
  const fObs = imgFig(big, "observed"), fRes = imgFig(big, "restored");
  const row = htmlEl("div", "sr-row", "", leftCol);
  const fErr = imgFig(row, "error");
  const pP = new Plot(htmlEl("div", "", "", row), { width: 480, height: 170, x: [0, 100], y: [10, 40], xlabel: "iteration", ylabel: "PSNR (dB)", font: 20, margin: { t: 8 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sr-lab", "presets", panel);
  const presetBtns = presets(panel, presetList.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEF, { prior: st.prior }, p.s); sync(); newScene(); } })));
  htmlEl("div", "sr-lab", "image", panel);
  const iRow = htmlEl("div", "qb-presets", "", panel);
  const iBtns = (["cell", "shapes", "texture"] as Img[]).map(v => button(iRow, { cell: "micrograph", shapes: "shapes", texture: "texture" }[v], () => { st.img = v; changed(); sync(); newScene(); }));
  htmlEl("div", "sr-lab", "task", panel);
  const tRow = htmlEl("div", "qb-presets", "", panel);
  const tBtns = TASKS.map(t => button(tRow, t.label, () => { st.task = t.id; changed(); sync(); newScene(); }));
  htmlEl("div", "sr-lab", "prior", panel);
  const pRow = htmlEl("div", "qb-presets", "", panel);
  const pBtns = priors.map(p => button(pRow, PRIOR_LABEL[p], () => { st.prior = p; changed(); sync(); go(); }));
  const nS = slider(panel, "noise σ", 0, 40, 1, st.noise, v => { st.noise = v; changed(); newScene(); });
  const bS = slider(panel, "blur σ", 0.5, 4, 0.25, st.blur, v => { st.blur = v; changed(); newScene(); });
  const mS = slider(panel, "missing", 0.1, 0.9, 0.05, st.missing, v => { st.missing = v; changed(); newScene(); }, v => Math.round(100 * v) + " %");
  const lS = slider(panel, "λ (log)", -1, 2.5, 0.05, st.lam, v => { st.lam = v; changed(); go(); }, v => lamOf(v).toPrecision(2));
  const itS = slider(panel, "iterations", 10, 400, 10, st.iters, v => { st.iters = v; changed(); go(); });
  const cRow = htmlEl("div", "qb-presets", "", panel);
  const spinC = checkbox(cRow, "cycle spinning", st.spin, v => { st.spin = v; changed(); go(); });
  button(cRow, "reset", () => { Object.assign(st, init); sync(); newScene(); presetBtns.forEach((b, i) => b.classList.toggle("active", presetList[i] === p0)); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const credit = htmlEl("div", "sr-credit", "", panel);
  htmlEl("div", "qb-note", "λ is in grey levels (images 0–255). Wavelet and TV priors: FISTA with the prior's proximal step. Patch priors: sparse coding of overlapping 8 × 8 patches; NL-means only denoises.", panel);
  if (p0) presetBtns[presetList.indexOf(p0)].classList.add("active");

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    iBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "shapes", "texture"][i] === st.img));
    tBtns.forEach((b, i) => b.classList.toggle("active", TASKS[i].id === st.task));
    pBtns.forEach((b, i) => b.classList.toggle("active", priors[i] === st.prior));
    nS.set(st.noise); bS.set(st.blur); mS.set(st.missing); lS.set(st.lam); itS.set(st.iters); spinC.checked = st.spin;
  }

  // ---- scene ----
  let truth: Vec = new Float64Array(n * n), train: Vec = new Float64Array(n * n), y: Vec = new Float64Array(n * n);
  let A: LinOp = identityOp(n * n), known: Uint8Array = new Uint8Array(n * n).fill(1), cellFull: Float64Array | null = null;
  async function newScene(): Promise<void> {
    if (st.img === "cell") {
      if (!cellFull) cellFull = Float64Array.from((await loadCell(256)).data);
      const crop = (x0: number, y0: number) => { const o = new Float64Array(n * n); for (let yy = 0; yy < n; yy++) for (let xx = 0; xx < n; xx++) o[yy * n + xx] = cellFull![(y0 + yy) * 256 + x0 + xx]; return o; };
      truth = n === 256 ? Float64Array.from(cellFull) : crop(64, 64);
      train = n === 256 ? Float64Array.from(cellFull, (_, i) => cellFull![(255 - Math.floor(i / 256)) * 256 + (i % 256)]) : crop(128, 0);
    } else if (st.img === "shapes") { truth = shapesImage(n); train = shapesImage(n, 37); }
    else { truth = textureImage(n); train = textureImage(n, 23); }
    credit.textContent = st.img === "cell" ? CELL_CREDIT : "Synthetic test image.";
    known = st.task === "random" ? randomMask(n, st.missing) : st.task === "scratch" ? scratchMask(n) : st.task === "text" ? textMask(n) : new Uint8Array(n * n).fill(1);
    A = st.task === "deblur" ? blurOp2d(n, n, st.blur) : st.task === "denoise" ? identityOp(n * n) : maskOp(known);
    const r = rng(11), clean = A.forward(truth);
    y = Float64Array.from(clean, (v, i) => (known[i] ? v + st.noise * r.norm() : 0));
    learned = null;
    go();
  }

  // ---- dictionaries ----
  let dct: Vec | null = null, learned: Vec | null = null;
  const P = 8;
  function getDct(): Vec { return (dct ??= dctDictionary(P, 256)); }
  function getLearned(): Vec | null {
    const shared = getDictionary(`${st.img}:${n}:8`);
    if (shared && shared.p === P) return shared.D;
    return learned;
  }

  let job: { cancel(): void } | null = null;
  const go = scheduler(run);
  function run(): void {
    job?.cancel();
    const lam = lamOf(st.lam);
    const showObs = Float64Array.from(y, (v, i) => (known[i] ? v : NaN));
    fObs.draw(showObs, n, n, 0, 255);
    fObs.cap.textContent = st.task === "deblur" ? `observed (blur σ = ${st.blur}, noise ${st.noise})` : st.task === "denoise" ? `observed (noise σ = ${st.noise})` : `observed (${Math.round(100 * (1 - known.reduce((a, b) => a + b, 0) / (n * n)))} % missing)`;
    const hist: number[] = [];
    let x: Vec = Float64Array.from(y);
    if (st.task !== "deblur" && st.task !== "denoise") { let s = 0, c = 0; for (let i = 0; i < n * n; i++) if (known[i]) { s += y[i]; c++; } const m = s / (c || 1); for (let i = 0; i < n * n; i++) if (!known[i]) x[i] = m; }
    const p0 = psnr(x, truth, 255);
    const show = (label: string) => {
      fRes.draw(x, n, n, 0, 255);
      fRes.cap.textContent = label;
      fErr.draw(Float64Array.from(x, (v, i) => v - truth[i]), n, n, -60, 60, diverging);
      pP.clear();
      const its = Math.max(10, hist.length);
      const lo = Math.min(p0, ...hist) - 1, hi = Math.max(p0, ...hist) + 1;
      pP.setDomain([0, its], [Math.floor(lo), Math.ceil(hi)]);
      pP.line([0, ...hist.map((_, i) => i + 1)], [p0, ...hist], { color: seriesColor(0) });
      pP.hline(p0, { color: "var(--qb-muted)", dash: "4 4", width: 1.5 });
      const cur = hist.length ? hist[hist.length - 1] : p0;
      ro.textContent = `${PRIOR_LABEL[st.prior]}, λ = ${lam.toPrecision(3)}\nPSNR observed ${psnr(Float64Array.from(y, (v, i) => (known[i] ? v : x[i])), truth, 255).toFixed(1)} dB\nPSNR restored ${cur.toFixed(1)} dB`;
    };

    // patch priors and NL-means: their own algorithms
    if (st.prior === "nlmeans" || st.prior === "dct" || st.prior === "learned") {
      if (st.prior === "nlmeans" && st.task !== "denoise") { show("NL-means only denoises"); ro.textContent = "NL-means is a denoiser: choose the denoise task."; return; }
      const needLearn = st.prior === "learned" && !getLearned();
      let lstate: { D: Vec; it: number } | null = null;
      if (needLearn) {
        const Ptr = extractPatches(train, n, n, P, 3);
        const sub = { ...Ptr, n: Math.min(Ptr.n, 1500), Y: Ptr.Y.subarray(0, Math.min(Ptr.n, 1500) * P * P) };
        lstate = { D: randomPatchDictionary(sub, 128, rng(5)), it: 0 };
        let done = false;
        job = progressive(() => {
          if (lstate!.it < 8) { learnStep(lstate!.D, P * P, sub, 4, "ksvd"); lstate!.it++; return true; }
          if (!done) { learned = lstate!.D; done = true; }
          return false;
        }, () => { show(learned ? "dictionary learned" : `learning a dictionary from another region… ${lstate!.it}/8`); if (learned) setTimeout(go, 0); }, 40);
        return;
      }
      const D = st.prior === "dct" ? getDct() : getLearned()!;
      let k = 0;
      const sig = Math.max(st.noise, 1);
      job = progressive(() => {
        if (st.prior === "nlmeans") { x = nonLocalMeans(y, n, n, 2, 7, 0.9 * sig); hist.push(psnr(x, truth, 255)); return false; }
        if (st.task === "denoise") { x = ksvdDenoise(y, n, n, D, P, sig, { stride: 2 }); hist.push(psnr(x, truth, 255)); return false; }
        if (st.task !== "deblur") { x = patchInpaint(y, known, n, n, D, P, { stride: 2, K: 6, errPerPixel: st.noise * st.noise }); hist.push(psnr(x, truth, 255)); return false; }
        // deblur: plug-and-play: gradient step, then patch denoising at the level λ
        const r = A.forward(x); for (let i = 0; i < r.length; i++) r[i] = y[i] - r[i];
        const g = A.adjoint(r), v = Float64Array.from(x, (xi, i) => xi + g[i]);
        x = ksvdDenoise(v, n, n, D, P, Math.max(lam, 0.5), { stride: 4, maxK: 8 });
        hist.push(psnr(x, truth, 255)); k++;
        return k < Math.min(st.iters, 30);
      }, () => show(`${PRIOR_LABEL[st.prior]}` + (st.task === "deblur" ? ` (${k} it.)` : "")), 40);
      show("computing…");
      return;
    }

    // FISTA with a proximal step
    const w = st.prior === "haar" ? WAVELETS.haar : WAVELETS.db4, levels = 4, spins = st.spin ? 4 : 1;
    let tikh: Float64Array | null = null;
    if (st.prior === "tikhonov") { tikh = new Float64Array(n * n); for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) { const L = 4 - 2 * Math.cos(2 * Math.PI * u / n) - 2 * Math.cos(2 * Math.PI * v / n); tikh[v * n + u] = 1 / (1 + lam * L); } }
    const prox = (v: Vec): Vec => {
      switch (st.prior) {
        case "none": return v;
        case "tikhonov": {
          const re = Float64Array.from(v), im = new Float64Array(n * n);
          fft2d(re, im, n); for (let i = 0; i < re.length; i++) { re[i] *= tikh![i]; im[i] *= tikh![i]; } fft2d(re, im, n, true);
          return re;
        }
        case "tv": return tv2d(v, n, n, lam, 15);
        default: return waveletProx2d(v, n, n, w, levels, lam, spins);
      }
    };
    let z: Vec = Float64Array.from(x), t = 1, k = 0;
    const maxIt = st.task === "denoise" ? 1 : st.iters;
    job = progressive(() => {
      if (k >= maxIt) return false;
      const r = A.forward(z); for (let i = 0; i < r.length; i++) r[i] = y[i] - r[i];
      const g = A.adjoint(r), v = new Float64Array(n * n);
      for (let i = 0; i < v.length; i++) v[i] = z[i] + g[i];
      const xn = prox(v), tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2, b = (t - 1) / tn;
      for (let i = 0; i < v.length; i++) z[i] = xn[i] + b * (xn[i] - x[i]);
      x = xn; t = tn; k++;
      hist.push(psnr(x, truth, 255));
      return k < maxIt;
    }, () => show(`restored: ${PRIOR_LABEL[st.prior]} (${k} it.)`), 30);
    show("restored");
  }

  window.addEventListener("qb-patchdict", () => { if (st.prior === "learned") go(); });
  sync();
  newScene();
}

autoMount(NAME, mount);
