// Classical denoisers compared by what they destroy. A 128 × 128 test image (a step, a ramp
// crossed by a 1-pixel line, a repeated dot lattice, one unique spot) or a micrograph crop;
// Gaussian, Poisson or salt-and-pepper noise (seeded). Methods: Gaussian, median, bilateral,
// non-local means, Wiener (no blur, 1/f² signal model). Panels: noisy, denoised, method
// noise (noisy − denoised), error vs truth; a line profile; PSNR (peak 255).
// "compare with" shows a second method (with its own last settings) next to the first.
// Auto-mounts on <div data-demo="denoisers">. Options: data-preset="<id>", data-presets="id,...".
import { htmlEl, presets as presetRow, button, autoMount, slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { diverging } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { psnr } from "../lib/metrics";
import { rng } from "../lib/ui";
import { poisson } from "../lib/random";
import * as Dn from "../lib/denoise";
import { injectStyle, baseStyle, select, logSlider, imagePanel, fmtSci, legendRow, ImagePanel } from "../deconvlab/widgets";
import { testImage } from "./testimage";

const NAME = "denoisers";
const N = 128;
type Img = "test" | "cell";
type Noise = "gauss" | "poisson" | "sp";
type Meth = "gauss" | "median" | "bilateral" | "nlm" | "wiener";
interface Params { sigma: number; radius: number; sS: number; sR: number; patch: number; search: number; h: number; snr: number }
interface State { img: Img; noise: Noise; level: number; photons: number; sp: number; m: Meth; cmp: Meth | "none"; p: Params; row: number }
const DP: Params = { sigma: 1.5, radius: 1, sS: 2, sR: 30, patch: 2, search: 5, h: 20, snr: 30 };
const DEFAULT: State = { img: "test", noise: "gauss", level: 20, photons: 30, sp: 0.1, m: "gauss", cmp: "none", p: { ...DP }, row: 32 };
interface Preset { id: string; label: string; s: Partial<State>; p?: Partial<Params> }
const PRESETS: Preset[] = [
  { id: "gaussstep", label: "Gaussian blurs the step", s: { m: "gauss" }, p: { sigma: 2 } },
  { id: "sp", label: "salt and pepper: median wins", s: { noise: "sp", sp: 0.1, m: "median", cmp: "gauss" }, p: { radius: 1, sigma: 1 } },
  { id: "line", label: "median eats the thin line", s: { level: 5, m: "median" }, p: { radius: 2 } },
  { id: "bilateral", label: "bilateral keeps edges", s: { level: 15, m: "bilateral", cmp: "gauss" }, p: { sS: 2, sR: 30, sigma: 2 } },
  { id: "cartoon", label: "bilateral at high noise: cartoon", s: { level: 50, m: "bilateral" }, p: { sS: 3, sR: 100 } },
  { id: "nlm", label: "non-local means on texture", s: { level: 25, m: "nlm", cmp: "bilateral", row: 96 }, p: { patch: 1, search: 7, h: 25, sS: 2, sR: 50 } },
  { id: "spot", label: "non-local means smears the unique spot", s: { level: 25, m: "nlm", row: 98 }, p: { patch: 2, search: 7, h: 45 } },
  { id: "methodnoise", label: "method noise shows the removed edge", s: { level: 10, m: "gauss" }, p: { sigma: 3 } },
  { id: "psnr", label: "best PSNR, wrong look", s: { img: "cell", level: 25, m: "wiener", cmp: "bilateral", row: 64 }, p: { snr: 3, sS: 1.5, sR: 50 } },
];
const LABEL: Record<Meth, string> = { gauss: "Gaussian", median: "median", bilateral: "bilateral", nlm: "non-local means", wiener: "Wiener (no blur)" };

const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .dn-imgs { flex: 1 1 380px; max-width: 440px; display: grid; grid-template-columns: 1fr 1fr; gap: .4em .5em; }
[data-demo="${NAME}"] .dn-imgs.three { grid-template-columns: 1fr 1fr 1fr; max-width: 520px; }
[data-demo="${NAME}"] .dn-plots { flex: 1 1 320px; max-width: 420px; display: flex; flex-direction: column; gap: .2em; }
.reveal [data-demo="${NAME}"] .dn-imgs { flex: 0 1 340px; }
.reveal [data-demo="${NAME}"] .dn-plots { flex: 0 1 400px; max-width: 400px; }
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? PRESETS.filter(p => only.includes(p.id)) : PRESETS;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...DEFAULT, p: { ...DP } };

  const imgs = htmlEl("div", "dn-imgs", "", host);
  const pNoisy = imagePanel(imgs, "noisy"), pA = imagePanel(imgs, "denoised"), pB = imagePanel(imgs, "B"), pMA = imagePanel(imgs, "method noise"), pMB = imagePanel(imgs, "method noise B"), pErr = imagePanel(imgs, "error vs truth");
  const credit = htmlEl("div", "dc-credit", CELL_CREDIT, imgs);
  credit.style.gridColumn = "1 / -1";
  const plots = htmlEl("div", "dn-plots", "", host);
  const prof = new Plot(plots, { width: 420, height: 270, x: [0, N], y: [0, 255], xlabel: "x along the marked row (px)", ylabel: "grey level", font: 19 });
  const lT = prof.line([], [], { color: "var(--qb-muted, #777)", dash: "7 5" }), lN = prof.line([], [], { color: seriesColor(0), width: 1.2, opacity: 0.6 });
  const lA = prof.line([], [], { color: seriesColor(1) }), lB = prof.line([], [], { color: seriesColor(2) });
  const leg = htmlEl("div", "", "", plots);
  const ro = htmlEl("div", "qb-readout", "", plots);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const imgS = select<Img>(panel, "image", [["test", "test image"], ["cell", "micrograph crop"]], st.img, v => { st.img = v; st.row = v === "cell" ? 64 : 32; rowS.set(st.row); makeNoisy(); });
  const noiseS = select<Noise>(panel, "noise", [["gauss", "Gaussian σn"], ["poisson", "Poisson (photons)"], ["sp", "salt and pepper"]], st.noise, v => { st.noise = v; sync(); makeNoisy(); });
  const levS = slider(panel, "σn (grey levels)", 0, 80, 1, st.level, v => { st.level = v; makeNoisy(); });
  const phS = logSlider(panel, "photons at 255", 3, 1000, st.photons, v => { st.photons = v; makeNoisy(); }, v => fmtSci(Math.round(v)));
  const spS = slider(panel, "fraction hit", 0, 0.3, 0.01, st.sp, v => { st.sp = v; makeNoisy(); }, v => `${Math.round(v * 100)} %`);
  const mS = select<Meth>(panel, "method", (Object.keys(LABEL) as Meth[]).map(k => [k, LABEL[k]]), st.m, v => { st.m = v; sync(); run(); });
  const sigS = slider(panel, "σ (px)", 0.3, 5, 0.1, st.p.sigma, v => { st.p.sigma = v; run(); });
  const radS = slider(panel, "radius (px)", 1, 4, 1, st.p.radius, v => { st.p.radius = v; run(); }, v => `${v} (${2 * v + 1}×${2 * v + 1})`);
  const sSS = slider(panel, "σs (px)", 0.5, 5, 0.1, st.p.sS, v => { st.p.sS = v; run(); });
  const sRS = slider(panel, "σr (grey levels)", 5, 150, 1, st.p.sR, v => { st.p.sR = v; run(); });
  const paS = slider(panel, "patch radius", 1, 3, 1, st.p.patch, v => { st.p.patch = v; run(); }, v => `${2 * v + 1}×${2 * v + 1}`);
  const seS = slider(panel, "search radius", 2, 10, 1, st.p.search, v => { st.p.search = v; run(); }, v => `${2 * v + 1}×${2 * v + 1}`);
  const hS = slider(panel, "h (grey levels)", 2, 100, 1, st.p.h, v => { st.p.h = v; run(); });
  const snrS = logSlider(panel, "SNR", 0.1, 1e4, st.p.snr, v => { st.p.snr = v; run(); });
  const cmpS = select<Meth | "none">(panel, "compare with", [["none", "—"], ...(Object.keys(LABEL) as Meth[]).map(k => [k, LABEL[k]] as [Meth, string])], st.cmp, v => { st.cmp = v; sync(); run(); });
  const rowS = slider(panel, "profile row", 0, N - 1, 1, st.row, v => { st.row = v; draw(); });
  button(panel, "reset", () => load(p0));
  htmlEl("div", "qb-note", "Method noise = noisy − denoised: it should look like pure noise. Any structure in it (an edge, a line, the lattice) is signal the filter removed. The second method uses its own last settings.", panel);

  const rowOf = (e: HTMLElement) => e.closest("label") as HTMLElement;
  function sync(): void {
    const show = (el: HTMLElement, on: boolean) => { rowOf(el).style.display = on ? "" : "none"; };
    show(levS.input, st.noise === "gauss"); show(phS.input, st.noise === "poisson"); show(spS.input, st.noise === "sp");
    show(sigS.input, st.m === "gauss"); show(radS.input, st.m === "median"); show(sSS.input, st.m === "bilateral"); show(sRS.input, st.m === "bilateral");
    show(paS.input, st.m === "nlm"); show(seS.input, st.m === "nlm"); show(hS.input, st.m === "nlm"); show(snrS.input, st.m === "wiener");
    const two = st.cmp !== "none";
    imgs.classList.toggle("three", two);
    // order: noisy, A, B / (error), method noise A, method noise B
    const order: ImagePanel[] = two ? [pNoisy, pA, pB, pErr, pMA, pMB] : [pNoisy, pA, pMA, pErr];
    [pNoisy, pA, pB, pMA, pMB, pErr].forEach(p => { p.fig.style.display = order.includes(p) ? "" : "none"; });
    order.forEach(p => imgs.insertBefore(p.fig, credit));
    leg.replaceChildren(legendRow(leg, [{ label: "truth", color: "var(--qb-muted, #777)", dash: true }, { label: "noisy", color: seriesColor(0) }, { label: LABEL[st.m], color: seriesColor(1) }, ...(two ? [{ label: LABEL[st.cmp as Meth], color: seriesColor(2) }] : [])]));
  }
  function load(p: Preset): void {
    Object.assign(st, { ...DEFAULT, p: { ...DP, ...(p.p ?? {}) } }, p.s);
    pbtns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    imgS.set(st.img); noiseS.set(st.noise); levS.set(st.level); phS.setValue(st.photons); spS.set(st.sp); mS.set(st.m); cmpS.set(st.cmp); rowS.set(st.row);
    sigS.set(st.p.sigma); radS.set(st.p.radius); sSS.set(st.p.sS); sRS.set(st.p.sR); paS.set(st.p.patch); seS.set(st.p.search); hS.set(st.p.h); snrS.setValue(st.p.snr);
    sync(); makeNoisy();
  }

  let cell: Float64Array | null = null;
  let truth: Float64Array = testImage(), noisy: Float64Array = truth.slice(), outA: Float64Array = truth.slice(), outB: Float64Array | null = null;
  function makeNoisy(): void {
    if (st.img === "cell" && !cell) { loadCell().then(g => { const c = new Float64Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) c[y * N + x] = g.data[(y + 64) * 256 + x + 64]; cell = c; makeNoisy(); }); return; }
    credit.style.display = st.img === "cell" ? "" : "none";
    truth = st.img === "cell" ? cell! : testImage();
    const r = rng(17);
    if (st.noise === "gauss") noisy = truth.map(v => v + st.level * r.norm());
    else if (st.noise === "poisson") noisy = truth.map(v => poisson(r, v / 255 * st.photons) * 255 / st.photons);
    else noisy = truth.map(v => { const u = r.unif(); return u < st.sp / 2 ? 0 : u < st.sp ? 255 : v; });
    run();
  }
  function apply(m: Meth): Float64Array {
    const p = st.p;
    switch (m) {
      case "gauss": return Dn.gaussianDenoise(noisy, N, N, p.sigma);
      case "median": return Dn.medianFilter(noisy, N, N, p.radius);
      case "bilateral": return Dn.bilateral(noisy, N, N, p.sS, p.sR);
      case "nlm": return Dn.nonLocalMeans(noisy, N, N, p.patch, p.search, p.h);
      case "wiener": return Dn.wienerDenoise(noisy, N, N, p.snr);
    }
  }
  let pending = 0, ms = 0;
  function run(): void {
    if (pending) return;
    pending = requestAnimationFrame(() => {
      pending = 0;
      const t0 = performance.now();
      outA = apply(st.m);
      outB = st.cmp !== "none" ? apply(st.cmp) : null;
      ms = performance.now() - t0;
      draw();
    });
    setTimeout(() => { if (pending) { cancelAnimationFrame(pending); pending = 0; outA = apply(st.m); outB = st.cmp !== "none" ? apply(st.cmp) : null; draw(); } }, 80);
  }
  function draw(): void {
    pNoisy.draw(noisy, N, N, { lo: 0, hi: 255 });
    pA.draw(outA, N, N, { lo: 0, hi: 255 }); pA.cap.textContent = LABEL[st.m];
    pMA.draw(Dn.methodNoise(noisy, outA), N, N, { lo: -60, hi: 60, map: diverging }); pMA.cap.textContent = "method noise";
    pErr.draw(outA.map((v, i) => v - truth[i]), N, N, { lo: -60, hi: 60, map: diverging }); pErr.cap.textContent = `error: ${LABEL[st.m]} − truth`;
    if (outB) {
      pB.draw(outB, N, N, { lo: 0, hi: 255 }); pB.cap.textContent = LABEL[st.cmp as Meth];
      pMB.draw(Dn.methodNoise(noisy, outB), N, N, { lo: -60, hi: 60, map: diverging }); pMB.cap.textContent = `method noise, ${LABEL[st.cmp as Meth]}`;
      pMA.cap.textContent = `method noise, ${LABEL[st.m]}`;
    }
    const xs = Array.from({ length: N }, (_, i) => i), row = (v: Float64Array) => xs.map(i => v[st.row * N + i]);
    lT.set(xs, row(truth)); lN.set(xs, row(noisy)); lA.set(xs, row(outA)); lB.set(outB ? xs : [], outB ? row(outB) : []);
    // mark the profile row in the noisy image
    const ctx = pNoisy.canvas.getContext("2d")!;
    ctx.fillStyle = "rgba(255, 127, 14, 0.8)";
    ctx.fillRect(0, st.row, 3, 1); ctx.fillRect(N - 3, st.row, 3, 1);
    const lines = [`PSNR (peak 255): noisy ${psnr(noisy, truth, 255).toFixed(1)} dB`, `${LABEL[st.m]}: ${psnr(outA, truth, 255).toFixed(1)} dB`];
    if (outB) lines.push(`${LABEL[st.cmp as Meth]}: ${psnr(outB, truth, 255).toFixed(1)} dB`);
    if (st.img === "test") {   // the dot lattice (bottom-left quarter) on its own
      const q = (a: Float64Array) => { const o: number[] = []; for (let y = 64; y < N; y++) for (let x = 0; x < 64; x++) o.push(a[y * N + x]); return o; };
      const tq = q(truth);
      lines.push(`lattice quarter: noisy ${psnr(q(noisy), tq, 255).toFixed(1)}, ${LABEL[st.m]} ${psnr(q(outA), tq, 255).toFixed(1)}${outB ? `, ${LABEL[st.cmp as Meth]} ${psnr(q(outB), tq, 255).toFixed(1)}` : ""} dB`);
    }
    lines.push(`computed in ${ms.toFixed(0)} ms`);
    ro.textContent = lines.join("\n");
  }
  load(p0);
}

autoMount(NAME, mount);
