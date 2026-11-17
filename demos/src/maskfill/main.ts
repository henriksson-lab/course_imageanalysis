// maskfill: masked image modelling in miniature (unit pretext/). Hide a fraction of the 6 × 6
// patches (or single pixels) of a test image and fill them in two ways: local interpolation,
// and a linear (PCA) model of 18 × 18 windows learnt without labels from other images of the
// same kind. Panels: the image, what the model sees, the two fills; the first principal
// components (what the model learnt); PSNR over the hidden pixels. See model.ts.
//
// <div data-demo="maskfill" data-preset="p75"></div>
import { autoMount, button, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { drawScalarImage, diverging, grey } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { baseStyle, imagePanel, injectStyle, select } from "../deconvlab/widgets";
import { fitPca, fillInterp, fillPca, hiddenPsnr, makeMask, scene, KMAX, N, WIN, type MaskKind, type PcaModel, type Scene } from "./model";
import { DEFAULT, PRESETS, type Preset, type Settings } from "./presets";

const NAME = "maskfill";
const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .mf-imgs { flex: 1 1 380px; max-width: 440px; display: grid; grid-template-columns: 1fr 1fr; gap: .35em .5em; }
[data-demo="${NAME}"] .mf-mid { flex: 1 1 220px; max-width: 260px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .mf-comps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; }
[data-demo="${NAME}"] .mf-comps canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border-radius: 2px; }
[data-demo="${NAME}"] .mf-cap { font-size: 14px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] .mf-imgs { flex: 0 1 360px; }
.reveal [data-demo="${NAME}"] .mf-mid { flex: 0 1 220px; }
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const s: Settings = { ...p0.s };

  const imgs = htmlEl("div", "mf-imgs", "", host);
  const pTruth = imagePanel(imgs, "image"), pSeen = imagePanel(imgs, "what the model sees (white = hidden)"), pInt = imagePanel(imgs, "interpolation"), pPca = imagePanel(imgs, "learnt (PCA)");
  const credit = htmlEl("div", "dc-credit", CELL_CREDIT, imgs);
  credit.style.gridColumn = "1 / -1";
  const mid = htmlEl("div", "mf-mid", "", host);
  htmlEl("div", "mf-cap", "the first 8 principal components of 18 × 18 windows (learnt without labels):", mid);
  const comps = htmlEl("div", "mf-comps", "", mid);
  const cc = Array.from({ length: 8 }, () => htmlEl("canvas", "", "", comps));
  const ro = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const scS = select<Scene>(panel, "images", [["nuclei", "synthetic nuclei"], ["cell", "micrograph"]], s.scene, v => { s.scene = v; run(); });
  const kS = select<MaskKind>(panel, "hide", [["patches", "6 × 6 patches"], ["pixels", "single pixels"]], s.kind, v => { s.kind = v; run(); });
  const rS = slider(panel, "fraction hidden", 0.05, 0.95, 0.05, s.ratio, v => { s.ratio = v; run(); }, v => `${Math.round(100 * v)} %`);
  const kSl = slider(panel, "components k", 0, KMAX, 1, s.k, v => { s.k = v; run(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "new random mask", () => { s.seed++; run(); });
  button(row, "reset", () => load(p0));
  htmlEl("div", "qb-note", "The PSNR is computed over the hidden pixels only. The PCA model was fitted to other images of the same kind (nuclei: another synthetic image; micrograph: its upper half; the test crop is from the lower half).", panel);

  const models: Partial<Record<Scene, { m: PcaModel; test: Float64Array }>> = {};
  let cell: Float64Array | null = null;
  function load(p: Preset): void {
    Object.assign(s, DEFAULT, p.s);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    scS.set(s.scene); kS.set(s.kind); rS.set(s.ratio); kSl.set(s.k);
    run();
  }
  let pending = 0;
  function run(): void {
    if (s.scene === "cell" && !cell) {
      loadCell().then(g => { cell = Float64Array.from(g.data); run(); });
      return;
    }
    if (!models[s.scene]) {
      ro.textContent = "learning the PCA model from the training images …";
      setTimeout(() => { const sc = scene(s.scene, cell ?? undefined); models[s.scene] = { m: fitPca(sc.windows), test: sc.test }; run(); }, 30);
      return;
    }
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = 0; draw(); });
  }
  function draw(): void {
    const { m, test } = models[s.scene]!;
    credit.style.display = s.scene === "cell" ? "" : "none";
    const mask = makeMask(s.kind, s.ratio, s.seed);
    const seen = Float64Array.from(test, (v, i) => (mask[i] ? NaN : v));
    const a = fillInterp(test, mask), b = fillPca(test, mask, m, s.k);
    const o = { lo: 0, hi: 255 };
    pTruth.draw(test, N, N, o);
    drawScalarImage(pSeen.canvas, seen, N, N, { map: grey, lo: 0, hi: 255 });
    pInt.draw(a, N, N, o); pPca.draw(b, N, N, o);
    const pa = hiddenPsnr(a, test, mask), pb = hiddenPsnr(b, test, mask);
    pInt.cap.textContent = `interpolation: ${pa.toFixed(2)} dB`;
    pPca.cap.textContent = `learnt, k = ${s.k}: ${pb.toFixed(2)} dB`;
    let hidden = 0; for (const v of mask) hidden += v;
    ro.textContent = `hidden: ${hidden} pixels (${(100 * hidden / (N * N)).toFixed(0)} %)\n` +
      `PSNR on the hidden pixels:\n  interpolation  ${pa.toFixed(2)} dB\n  learnt (k=${s.k})   ${pb.toFixed(2)} dB\n` +
      `(the outer ring of patches is never hidden)`;
    m.comps.slice(0, 8).forEach((c, i) => {
      let mx = 0; for (const v of c) mx = Math.max(mx, Math.abs(v));
      drawScalarImage(cc[i], c, WIN, WIN, { map: diverging, lo: -mx, hi: mx });
    });
  }
  load(p0);
}

autoMount(NAME, mount);
