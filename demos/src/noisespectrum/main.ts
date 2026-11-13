// What does noise destroy first? A 128 × 128 image (a crop of the course micrograph, or
// synthetic nuclei), normalised to mean 0 and variance 1, plus white noise of level σ
// (the forward process in noise-level form, x/√ā = x₀ + σε). Panels: the noisy image; the
// Wiener denoiser for the "Gaussian image model" (the image's own radially averaged power
// spectrum P(f)); a sample of that Gaussian model (random phases, same spectrum); the radially
// averaged power spectrum with the noise floor and the crossover frequency; the schedule ā(τ)
// with the time at which this noise level is reached.
//
// <div data-demo="noisespectrum" data-preset="fine|snr1|blur|gauss"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { standardise, powerModel, analyse, noiseField, gaussianSample, N } from "./model";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { nuclei } from "../lib/synth";
import { alphaBarAt, type Schedule } from "../lib/mixture";
import { abarFromLogSnr, tauFromAbar } from "../lib/diffusion";
import { injectStyle, choiceRow, helperCss, imgPanel, fmt } from "../scorematch1d/dview";

const NAME = "noisespectrum";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .ns-imgs { flex: 0 1 min(400px, calc(100vh - 300px)); display: grid; grid-template-columns: 1fr 1fr; gap: .3em .5em; min-width: 260px; }
${S} .ns-plots { flex: 0 1 330px; min-width: 240px; display: flex; flex-direction: column; gap: .2em; }
${S} .qb-panel { flex: 1 1 250px; max-width: 340px; gap: .3em; font-size: 15px; }
${S} .qb-btn { font-size: 13.5px; padding: .12em .5em; }
${S} .qb-presets { gap: .25em; }
${S} .qb-readout { font-size: 14px; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .ns-imgs { flex: 0 0 400px; }
.reveal ${S} .ns-plots { flex: 0 0 390px; }
.reveal ${S} .qb-panel { flex: 1 1 300px; max-width: none; gap: .2em; font-size: 18px; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

type ImgId = "cell" | "nuclei";
interface State { img: ImgId; lsnr: number; sched: Schedule; seed: number }
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "fine", label: "fine detail goes first", s: { img: "cell", lsnr: 2.3 } },
  { id: "snr1", label: "SNR 1: what is left", s: { img: "cell", lsnr: 0 } },
  { id: "blur", label: "high noise: Wiener is a blur", s: { img: "cell", lsnr: -4.6 } },
  { id: "gauss", label: "a Gaussian prior knows the spectrum, not the cells", s: { img: "nuclei", lsnr: 1 } },
];
const BASE: State = { img: "cell", lsnr: 2.3, sched: "cosine", seed: 1 };

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  // larger labels on slides (the plots are scaled down there)
  const F = host.closest(".reveal") ? 23 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, ...p0.s };
  const imgs: Partial<Record<ImgId, { x: Float64Array; model: ReturnType<typeof powerModel> }>> = {};
  const noise = noiseField();

  const grid = htmlEl("div", "ns-imgs", "", host);
  const pClean = imgPanel(grid, "clean x₀"), pNoisy = imgPanel(grid, "noisy x₀ + σε"), pWien = imgPanel(grid, "Wiener denoiser"), pSamp = imgPanel(grid, "sample of the Gaussian model");
  const plots = htmlEl("div", "ns-plots", "", host);
  const sp = new Plot(plots, { width: 420, height: 300, x: [1, 64], y: [1e-1, 1e8], xlog: true, ylog: true, xlabel: "frequency (cycles per image)", ylabel: "power", font: F, margin: { l: 4.3 * F, r: 12, t: 8 } });
  const specLine = sp.line([], [], { color: seriesColor(0), width: 3 });
  const floor = sp.hline(1, { color: seriesColor(1), width: 3, dash: "" });
  const cross = sp.vline(10, { color: "var(--qb-fg)", width: 2 });
  sp.legend([{ label: "image power", color: seriesColor(0) }, { label: "noise floor", color: seriesColor(1) }], "bl");
  const schP = new Plot(plots, { width: 420, height: 170, x: [0, 1], y: [0, 1], xlabel: "time τ", ylabel: "ā", xticks: [0, 0.25, 0.5, 0.75, 1], yticks: [0, 0.5, 1], font: F, margin: { l: 4.3 * F, r: 12, t: 8 } });
  const schLine = schP.line([], [], { color: seriesColor(2), width: 3 });
  const schDot = schP.scatter([], [], { color: seriesColor(1), r: 8 });

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const imgC = choiceRow(panel, "image", [{ v: "cell" as ImgId, label: "micrograph" }, { v: "nuclei" as ImgId, label: "nuclei (synthetic)" }], st.img, v => { st.img = v; draw(); });
  const lS = slider(panel, "log-SNR", -7, 7, 0.1, st.lsnr, v => { st.lsnr = v; draw(); });
  const schC = choiceRow(panel, "schedule", [{ v: "cosine" as Schedule, label: "cosine" }, { v: "linear" as Schedule, label: "linear" }], st.sched, v => { st.sched = v; draw(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "resample", () => { st.seed++; draw(); });
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  const credit = htmlEl("div", "qb-note", "", panel);

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    lS.set(st.lsnr); imgC.set(st.img); schC.set(st.sched);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }

  function draw(): void {
    const I = imgs[st.img];
    credit.textContent = st.img === "cell" ? CELL_CREDIT + ". Crop 128 × 128, standardised." : "Synthetic nuclei (seeded), standardised.";
    if (!I) return;
    const snr = Math.exp(st.lsnr), sigma = 1 / Math.sqrt(snr), x = I.x;
    const A = analyse(x, I.model, sigma, noise), sr = gaussianSample(I.model.P, 100 + st.seed), nf = N * N * sigma * sigma;
    const lo = -2.5, hi = 2.5;
    pClean.draw(x, N, N, lo, hi); pNoisy.draw(A.noisy, N, N, lo - 1.5 * sigma, hi + 1.5 * sigma); pWien.draw(A.wiener, N, N, lo, hi); pSamp.draw(sr, N, N, lo, hi);
    pNoisy.cap.textContent = `noisy x₀ + σε (σ = ${fmt(sigma, 2)})`;
    const fr: number[] = [], pw: number[] = [];
    for (let b = 1; b <= N / 2; b++) { fr.push(b); pw.push(I.model.radial[b]); }
    specLine.set(fr, pw);
    floor.set(nf);
    const fc = A.crossover;
    cross.set(fc >= 1 ? fc : NaN);
    const eN = A.mseNoisy * N * N, eW = A.mseWiener * N * N;
    const ab = abarFromLogSnr(st.lsnr), tau = tauFromAbar(ab, st.sched);
    const ts = Array.from({ length: 101 }, (_, i) => i / 100);
    schLine.set(ts, ts.map(t => alphaBarAt(t, st.sched)));
    schDot.set([tau], [ab]);
    ro.textContent =
      `SNR = ${snr >= 10 ? snr.toFixed(0) : fmt(snr, 2)}   σ = ${fmt(sigma, 2)}   ā = ${fmt(ab, 3)}   τ = ${fmt(tau, 2)}\n` +
      `power above the noise floor up to ${fc >= 1 ? fc : "<1"} cycles per image\n` +
      `mean squared error: noisy ${fmt(eN / (N * N), 3)}, Wiener ${fmt(eW / (N * N), 3)}`;
  }

  // images: the synthetic one now, the micrograph when loaded
  const nu = nuclei({ width: N, height: N, count: 9, radius: [8, 13], noise: 0, seed: 4 });
  const nx = standardise(nu.image.data);
  imgs.nuclei = { x: nx, model: powerModel(nx) };
  loadCell(256).then(g => {
    const c = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let xx = 0; xx < N; xx++) c[y * N + xx] = g.data[(y + 64) * 256 + xx + 64];
    const cx = standardise(c);
    imgs.cell = { x: cx, model: powerModel(cx) };
    draw();
  }).catch(() => { /* image unavailable: the nuclei still work */ });
  pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
  draw();
}

autoMount(NAME, mount);
