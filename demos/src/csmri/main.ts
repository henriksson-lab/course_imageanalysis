// Fast MRI by compressed sensing. A phantom (Shepp–Logan, the same with small faint lesions, or
// the course micrograph as a non-MRI test) is "scanned" in k-space: the scanner measures Fourier
// coefficients line by line, and we keep only some lines (centre only, every r-th line + centre,
// random lines + centre, variable-density random lines, or radial spokes). Reconstruction:
// zero-filled (inverse FFT with the missing lines set to 0) and compressed sensing:
// min ½‖PFx − y‖² + λ‖Wx‖₁ (+ λ_TV TV(x)) by FISTA with step 1 (PF has orthonormal rows),
// the wavelet prox being soft shrinkage of the details and TV a few Chambolle iterations.
// Iterations run progressively (refresh every 5). Panels: zero-filled, CS result, the mask,
// k-space (log magnitude, mask overlaid), the point spread function of the mask and the error.
// Masks are symmetric (k and −k) because the phantom is real.
// Auto-mounts on <div data-demo="csmri">. Optional: data-n="128|256", data-preset="…", data-layout="full".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { WAVELETS } from "../lib/wavelet";
import { sheppLogan } from "../lib/synth";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { fft2d } from "../lib/fft";
import { diverging, viridis } from "../lib/colour";
import { psnr, ssim } from "../lib/metrics";
import { tv2d } from "../lib/smooth";
import { kspaceMask, partialFourierOp2d, maskPsf2d, maskFraction, MaskKind, LESION_PHANTOM, waveletProx2d, Vec } from "../lib/sparse";
import { rng } from "../lib/random";
import { imgFig, injectStyle, scheduler, progressive } from "./view";

const NAME = "csmri";
type Phantom = "shepp" | "lesions" | "cell";
type Recon = "wavelet" | "wavelet+tv";
interface State { phantom: Phantom; mask: MaskKind; accel: number; q: number; noise: number; recon: Recon; wav: "haar" | "db4"; lam: number; lamTv: number; iters: number }
const DEF: State = { phantom: "shepp", mask: "vardens", accel: 4, q: 3, noise: 0.002, recon: "wavelet", wav: "haar", lam: 0.01, lamTv: 0.005, iters: 100 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "centre", label: "centre only: blur", s: { mask: "centre", accel: 4 } },
  { id: "fold", label: "every 2nd line: fold", s: { mask: "regular", accel: 2 } },
  { id: "random", label: "random: noise-like", s: { mask: "random", accel: 4 } },
  { id: "cs4", label: "variable density + wavelets: 4×", s: { mask: "vardens", accel: 4, recon: "wavelet", wav: "haar", lam: 0.01 } },
  { id: "lesions", label: "8×: lesions vanish", s: { phantom: "lesions", mask: "vardens", accel: 8, recon: "wavelet", wav: "haar", lam: 0.015 } },
  { id: "tv", label: "TV cartoon", s: { phantom: "cell", mask: "vardens", accel: 4, recon: "wavelet+tv", lam: 0.005, lamTv: 0.03 } },
  { id: "radial", label: "radial spokes", s: { mask: "radial", accel: 4 } },
];
const MASKS: { id: MaskKind; label: string }[] = [
  { id: "centre", label: "centre" }, { id: "regular", label: "every r-th" }, { id: "random", label: "random" },
  { id: "vardens", label: "variable density" }, { id: "radial", label: "radial" },
];

const CSS = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.15; margin: 0 0 .15em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 2px; background: #000; }
.qb-demo[data-demo="${NAME}"] .mr-left { flex: 0 1 min(620px, calc((100vh - 140px) / 0.75)); min-width: 280px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .mr-big { display: grid; grid-template-columns: 1fr 1fr; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .mr-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .25em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { padding: .12em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-slider { gap: .35em; }
.qb-demo[data-demo="${NAME}"] details summary { cursor: pointer; font-size: 14px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .mr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.15em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre; line-height: 1.3; }
.qb-demo[data-demo="${NAME}"] .mr-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"][data-layout="full"] .mr-left { flex-basis: min(720px, calc((100vh - 140px) / 0.75)); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .mr-left { flex-basis: 560px; max-width: 560px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 330px; max-width: 350px; gap: .15em; font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] details { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 13px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .mr-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;

function fftshift(a: ArrayLike<number>, n: number): Vec {
  const o = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) o[((y + n / 2) % n) * n + ((x + n / 2) % n)] = a[y * n + x];
  return o;
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const n = Number(host.dataset.n) === 256 ? 256 : 128;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: State = { ...DEF, ...(p0?.s ?? {}) };
  const st: State = { ...init };

  const leftCol = htmlEl("div", "mr-left", "", host);
  const big = htmlEl("div", "mr-big", "", leftCol);
  const fZf = imgFig(big, "zero-filled"), fCs = imgFig(big, "compressed sensing");
  const strip = htmlEl("div", "mr-strip", "", leftCol);
  const fMask = imgFig(strip, "lines kept"), fK = imgFig(strip, "k-space"), fPsf = imgFig(strip, "PSF of mask"), fErr = imgFig(strip, "CS error");
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "mr-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEF, p.s); sync(); newPhantom(); } })));
  htmlEl("div", "mr-lab", "object", panel);
  const phRow = htmlEl("div", "qb-presets", "", panel);
  const phBtns = (["shepp", "lesions", "cell"] as Phantom[]).map(v => button(phRow, { shepp: "head phantom", lesions: "+ faint lesions", cell: "micrograph" }[v], () => { st.phantom = v; changed(); sync(); newPhantom(); }));
  htmlEl("div", "mr-lab", "k-space sampling", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = MASKS.map(m => button(mRow, m.label, () => { st.mask = m.id; changed(); sync(); go(); }));
  const aS = slider(panel, "acceleration", 1, 10, 0.5, st.accel, v => { st.accel = v; changed(); go(); }, v => v.toFixed(1) + "×");
  const qS = slider(panel, "density exponent q", 1, 6, 0.5, st.q, v => { st.q = v; changed(); go(); });
  const nS = slider(panel, "noise", 0, 0.02, 0.001, st.noise, v => { st.noise = v; changed(); go(); });
  htmlEl("div", "mr-lab", "reconstruction", panel);
  const rRow = htmlEl("div", "qb-presets", "", panel);
  const rBtns = [
    button(rRow, "Haar ℓ1", () => { st.recon = "wavelet"; st.wav = "haar"; changed(); sync(); go(); }),
    button(rRow, "db4 ℓ1", () => { st.recon = "wavelet"; st.wav = "db4"; changed(); sync(); go(); }),
    button(rRow, "+ TV", () => { st.recon = "wavelet+tv"; changed(); sync(); go(); }),
  ];
  const lS = slider(panel, "λ wavelet", 0, 0.05, 0.001, st.lam, v => { st.lam = v; changed(); go(); });
  const tS = slider(panel, "λ TV", 0, 0.08, 0.002, st.lamTv, v => { st.lamTv = v; changed(); go(); });
  const iS = slider(panel, "iterations", 5, 300, 5, st.iters, v => { st.iters = v; changed(); go(); });
  button(panel, "reset", () => { Object.assign(st, init); sync(); newPhantom(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const credit = htmlEl("div", "mr-credit", "", panel);
  const about = htmlEl("details", "", "", panel);
  htmlEl("summary", "", "about the phantom", about);
  htmlEl("div", "qb-note", "Synthetic phantom (Shepp–Logan, a head-like test object). Scan time assumes a 3D scan of 256 × 128 phase-encoding lines at 10 ms each, divided by the acceleration.", about);
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    phBtns.forEach((b, i) => b.classList.toggle("active", ["shepp", "lesions", "cell"][i] === st.phantom));
    mBtns.forEach((b, i) => b.classList.toggle("active", MASKS[i].id === st.mask));
    rBtns.forEach((b, i) => b.classList.toggle("active", i === 2 ? st.recon === "wavelet+tv" : st.recon === "wavelet" && st.wav === ["haar", "db4"][i]));
    aS.set(st.accel); qS.set(st.q); nS.set(st.noise); lS.set(st.lam); tS.set(st.lamTv); iS.set(st.iters);
  }

  let truth: Vec = new Float64Array(n * n), cellImg: Vec | null = null;
  async function newPhantom(): Promise<void> {
    if (st.phantom === "cell") {
      if (!cellImg) { const g = await loadCell(n); cellImg = Float64Array.from(g.data, v => v / 255); }
      truth = cellImg;
    } else truth = Float64Array.from(sheppLogan(n, st.phantom === "lesions" ? LESION_PHANTOM : undefined).data);
    credit.textContent = st.phantom === "cell" ? CELL_CREDIT : "";
    // k-space of the truth (display)
    const re = Float64Array.from(truth), im = new Float64Array(n * n);
    fft2d(re, im, n);
    kLog = fftshift(Float64Array.from(re, (v, i) => Math.log10(1 + Math.hypot(v, im[i]))), n);
    go();
  }
  let kLog: Vec = new Float64Array(n * n);

  let job: { cancel(): void } | null = null;
  const go = scheduler(run);
  function run(): void {
    job?.cancel();
    const mask = kspaceMask(n, st.mask, st.accel, rng(17), { q: st.q });
    const A = partialFourierOp2d(mask, n);
    const rn = rng(5);
    const y = A.forward(truth);
    for (let i = 0; i < y.length; i++) if (y[i] !== 0) y[i] += st.noise * rn.norm() / Math.SQRT2;
    const zf = A.adjoint(y);
    // displays that do not depend on the iterations
    const ms = fftshift(mask, n);
    fMask.draw(ms, n, n, 0, 1);
    const ko = new Float64Array(n * n), kmax = Math.max(...kLog);
    for (let i = 0; i < n * n; i++) ko[i] = ms[i] ? kLog[i] / kmax : 0.15 * kLog[i] / kmax;
    fK.draw(ko, n, n, 0, 1, viridis);
    const psf = maskPsf2d(mask, n), pk = psf[(n / 2) * n + n / 2] || 1;
    fPsf.draw(Float64Array.from(psf, v => Math.sqrt(Math.abs(v) / pk)), n, n, 0, 0.6, viridis);
    fZf.draw(zf, n, n, 0, 1);
    const frac = maskFraction(mask);
    const pZf = psnr(zf, truth, 1), sZ = ssim(zf, truth, n, n, { peak: 1 }).mean;
    const w = WAVELETS[st.wav], levels = n === 256 ? 5 : 4;
    // FISTA, step 1
    let x: Vec = Float64Array.from(zf), z: Vec = Float64Array.from(zf), t = 1, k = 0;
    const prox = (v: Vec): Vec => {
      let o = st.lam > 0 ? waveletProx2d(v, n, n, w, levels, st.lam) : v;
      if (st.recon === "wavelet+tv" && st.lamTv > 0) o = tv2d(o, n, n, st.lamTv, 12);
      return o;
    };
    const show = () => {
      fCs.draw(x, n, n, 0, 1);
      const e = Float64Array.from(x, (v, i) => v - truth[i]);
      fErr.draw(e, n, n, -0.15, 0.15, diverging);
      fCs.cap.textContent = `compressed sensing (${k} it.)`;
      const pCs = psnr(x, truth, 1), sC = ssim(x, truth, n, n, { peak: 1 }).mean;
      const full = 256 * 128 * 0.01, acc = 1 / frac;
      ro.textContent = [
        `sampled ${(100 * frac).toFixed(1)} % of k-space (${acc.toFixed(1)}×)`,
        `zero-filled  PSNR ${pZf.toFixed(1)} dB  SSIM ${sZ.toFixed(3)}`,
        `CS           PSNR ${pCs.toFixed(1)} dB  SSIM ${sC.toFixed(3)}`,
        `3D scan: ${(full / 60).toFixed(1)} min → ${(full / acc / 60).toFixed(1)} min`,
      ].join("\n");
    };
    job = progressive(() => {
      if (k >= st.iters) return false;
      const r = A.forward(z);
      for (let i = 0; i < r.length; i++) r[i] = y[i] - r[i];
      const g = A.adjoint(r), v = new Float64Array(n * n);
      for (let i = 0; i < v.length; i++) v[i] = z[i] + g[i];
      const xn = prox(v), tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2, b = (t - 1) / tn;
      for (let i = 0; i < v.length; i++) z[i] = xn[i] + b * (xn[i] - x[i]);
      x = xn; t = tn; k++;
      return k < st.iters;
    }, show, 30);
    show();
  }

  sync();
  newPhantom();
}

autoMount(NAME, mount);
