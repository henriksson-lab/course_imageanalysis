// blindspot: self-supervised denoising with a linear filter trained on the noisy image itself
// (unit self-denoise/). The filter (5 × 5 by default) is fitted by least squares: Noise2Void
// (blind spot: the centre weight is forced to 0, the target is the noisy pixel), without the
// blind spot (it learns to copy the pixel), Noise2Noise (a second noisy image as the target),
// or supervised (clean target, for comparison). Panels: noisy, the filter's result, a
// classical filter (Gaussian or non-local means), the clean image, and the learned weights.
// A table gives the PSNR (peak 255) of every training kind. Noise: independent, correlated
// (blurred) or stripes. See model.ts.
//
// <div data-demo="blindspot" data-preset="n2v"></div>
import { autoMount, button, checkbox, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { heatmapGrid } from "../lib/plot";
import { diverging } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { baseStyle, imagePanel, injectStyle, select } from "../deconvlab/widgets";
import { testImage } from "../denoisers/testimage";
import { evaluate, TRAIN_LABEL, W, type Cmp, type NoiseKind, type Settings, type Train } from "./model";
import { DEFAULT, PRESETS, type Img, type Preset } from "./presets";

const NAME = "blindspot";
const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .bs-imgs { flex: 1 1 360px; max-width: 420px; display: grid; grid-template-columns: 1fr 1fr; gap: .35em .5em; }
[data-demo="${NAME}"] .bs-mid { flex: 1 1 250px; max-width: 300px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .bs-cap { font-size: 15px; text-align: center; }
[data-demo="${NAME}"] .bs-table { font-size: 14px; border-collapse: collapse; width: 100%; }
[data-demo="${NAME}"] .bs-table td { padding: .05em .3em; border-bottom: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .bs-table td:last-child { text-align: right; font-family: ui-monospace, monospace; }
[data-demo="${NAME}"] .bs-table tr.sel td { font-weight: 700; color: var(--qb-accent); }
.reveal [data-demo="${NAME}"] .bs-imgs { flex: 0 1 330px; }
.reveal [data-demo="${NAME}"] .bs-mid { flex: 0 1 250px; }
.reveal [data-demo="${NAME}"] .bs-table { font-size: 13px; }
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let img: Img = p0.img;
  const s: Settings = { ...p0.s };

  const imgs = htmlEl("div", "bs-imgs", "", host);
  const pNoisy = imagePanel(imgs, "noisy"), pOut = imagePanel(imgs, "learned filter"), pCmp = imagePanel(imgs, "classical"), pClean = imagePanel(imgs, "clean (truth)");
  const credit = htmlEl("div", "dc-credit", CELL_CREDIT, imgs);
  credit.style.gridColumn = "1 / -1";

  const mid = htmlEl("div", "bs-mid", "", host);
  const kcap = htmlEl("div", "bs-cap", "learned weights", mid);
  const kHost = htmlEl("div", "", "", mid);
  let kgrid: { svg: SVGSVGElement; set(v: ArrayLike<number>): void } | null = null;
  let kSize = 0;
  const table = htmlEl("table", "bs-table", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const imgS = select<Img>(panel, "image", [["cell", "micrograph crop"], ["test", "test image"]], img, v => { img = v; run(); });
  const noiseS = select<NoiseKind>(panel, "noise", [["white", "independent"], ["corr", "correlated (blurred)"], ["stripes", "stripes (per row)"]], s.noise, v => { s.noise = v; run(); });
  const sigS = slider(panel, "σn (grey levels)", 5, 60, 1, s.sigma, v => { s.sigma = v; s.nlmH = v; run(); });
  const trS = select<Train>(panel, "training", (Object.keys(TRAIN_LABEL) as Train[]).map(k => [k, TRAIN_LABEL[k]]), s.train, v => { s.train = v; run(); });
  const rS = slider(panel, "filter size", 1, 3, 1, s.r, v => { s.r = v; run(); }, v => `${2 * v + 1}×${2 * v + 1}`);
  const mixC = checkbox(panel, "put the pixel back (σn known)", s.mix, v => { s.mix = v; run(); });
  const cmpS = select<Cmp>(panel, "classical", [["gauss", "Gaussian blur"], ["nlm", "non-local means"]], s.cmp, v => { s.cmp = v; run(); });
  const gS = slider(panel, "Gaussian σ (px)", 0.3, 3, 0.1, s.gsig, v => { s.gsig = v; run(); });
  button(panel, "reset", () => load(p0));
  htmlEl("div", "qb-note", "Every filter is trained on this one noisy image (least squares over all its pixels); the clean image is used only for the PSNR, and as the target of the supervised filters. Non-local means: 3×3 patches, 11×11 search, h = σn.", panel);

  function load(p: Preset): void {
    img = p.img; Object.assign(s, DEFAULT, p.s);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    imgS.set(img); noiseS.set(s.noise); sigS.set(s.sigma); trS.set(s.train); rS.set(s.r); mixC.checked = s.mix; cmpS.set(s.cmp); gS.set(s.gsig);
    run();
  }

  let cell: Float64Array | null = null;
  let pending = 0;
  function run(): void {
    if (img === "cell" && !cell) {
      loadCell().then(g => { const c = new Float64Array(W * W); for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) c[y * W + x] = g.data[(y + 64) * 256 + x + 64]; cell = c; run(); });
      return;
    }
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = 0; draw(); });
  }

  function draw(): void {
    (gS.input.closest("label") as HTMLElement).style.display = s.cmp === "gauss" ? "" : "none";
    mixC.disabled = !(s.train === "n2v" || s.train === "supblind");
    credit.style.display = img === "cell" ? "" : "none";
    const clean = img === "cell" ? cell! : testImage();
    const r = evaluate(clean, s);
    const o = { lo: 0, hi: 255 };
    pNoisy.draw(r.noisy, W, W, o); pOut.draw(r.out, W, W, o); pCmp.draw(r.cmpOut, W, W, o); pClean.draw(clean, W, W, o);
    pNoisy.cap.textContent = `noisy: ${r.psnr.noisy.toFixed(2)} dB`;
    pOut.cap.textContent = `${s.train === "n2v" ? "Noise2Void" : TRAIN_LABEL[s.train]}${Number.isFinite(r.c) ? " + pixel" : ""}: ${r.psnr.out.toFixed(2)} dB`;
    pCmp.cap.textContent = `${s.cmp === "gauss" ? `Gaussian σ ${s.gsig.toFixed(1)}` : "non-local means"}: ${r.psnr.cmp.toFixed(2)} dB`;
    // weights
    const n = 2 * s.r + 1;
    if (!kgrid || kSize !== n) {
      kHost.replaceChildren();
      kgrid = heatmapGrid(kHost, r.filter.weights, n, n, { cell: n === 7 ? 36 : n === 5 ? 46 : 64, font: 16, map: diverging, lo: -0.5, hi: 0.5, fmt: v => (Math.abs(v) < 0.005 ? "0" : v.toFixed(2)),
        text: (_v, i) => (i === (n * n - 1) / 2 && (s.train === "n2v" || s.train === "supblind") ? "×" : undefined) });
      kSize = n;
    } else kgrid.set(r.filter.weights);
    let sum = 0; for (const v of r.filter.weights) sum += v;
    kcap.textContent = `learned weights (sum ${sum.toFixed(2)})` + (Number.isFinite(r.c) ? `; pixel mixed back, c = ${r.c.toFixed(2)}` : "");
    // table
    table.replaceChildren();
    const rows: [string, number, boolean][] = [
      ["noisy image", r.psnr.noisy, false],
      ...(Object.keys(TRAIN_LABEL) as Train[]).map(k => [TRAIN_LABEL[k], r.psnr.all[k], k === s.train] as [string, number, boolean]),
      ["Noise2Void + pixel back", r.psnr.mixed, s.mix && s.train === "n2v"],
      [s.cmp === "gauss" ? "Gaussian blur" : "non-local means", r.psnr.cmp, false],
    ];
    const head = htmlEl("tr", "", "", table);
    htmlEl("td", "", "PSNR against the truth", head); htmlEl("td", "", "dB", head);
    for (const [lab, v, sel] of rows) {
      const tr = htmlEl("tr", sel ? "sel" : "", "", table);
      htmlEl("td", "", lab, tr); htmlEl("td", "", v.toFixed(2), tr);
    }
  }

  load(p0);
}

autoMount(NAME, mount);
