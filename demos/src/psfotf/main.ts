// PSF and OTF of a microscope: NA, wavelength, defocus, motion blur or the Gaussian
// approximation. Panels: the PSF (linear or log), the OTF as an image with the cut-off
// circle 2NA/λ (magnitude or signed), the OTF along the horizontal frequency axis with the
// ideal in-focus curve, and a test object with its blurred image. The PSF is
// |F⁻¹(pupil · e^{i·defocus phase})|² on a 256² grid of 40 nm pixels (one FFT); the blur is
// one forward and one inverse FFT of the object.
// Auto-mounts on <div data-demo="psfotf">. Options: data-preset="oil|dry|bluered|defocus|motion|gauss"
// (default oil); data-presets="id,id,..." limits the preset buttons; data-object="cell|star|beads|bars".
import { htmlEl, presets as presetRow, button, checkbox, autoMount, slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { inferno, diverging, grey } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { airyOTF } from "../lib/psf";
import { injectStyle, baseStyle, select, imagePanel, legendRow } from "../deconvlab/widgets";
import { N, PIXEL, Optics, PsfKind, ObjectId, computePSF, blurWith, otfCut, spectrumCrop, centreCrop, psfFWHM, starObject, beadObject, barObject, wavelengthColour, immersion, BAR_PERIODS } from "./model";

const NAME = "psfotf";
const FMAX = 8;                                         // cycles/µm shown in the OTF panels
const KMAX = Math.round(FMAX / 1000 * N * PIXEL);       // spectrum bins up to FMAX (82)
const KPSF = 32;                                        // PSF crop half-width (pixels): 65 × 40 nm = 2.6 µm

interface State extends Optics { object: ObjectId; logPsf: boolean; signed: boolean; stretch: boolean }
const DEFAULT: State = { kind: "pupil", NA: 1.4, wavelength: 520, defocus: 0, motion: 1000, angle: 0, object: "cell", logPsf: false, signed: false, stretch: false };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "oil", label: "oil objective, NA 1.4", s: { NA: 1.4, object: "cell" } },
  { id: "dry", label: "dry objective, NA 0.3", s: { NA: 0.3, object: "cell" } },
  { id: "bluered", label: "blue vs red", s: { NA: 0.8, wavelength: 450, object: "bars" } },
  { id: "defocus", label: "defocus: contrast reversal", s: { NA: 0.7, defocus: 2000, object: "star", signed: true, stretch: true } },
  { id: "motion", label: "motion blur: zeros in the OTF", s: { kind: "motion", NA: 1.4, motion: 1000, object: "star", signed: true } },
  { id: "gauss", label: "Gaussian approximation", s: { kind: "gauss", NA: 1.4, object: "beads", logPsf: true } },
];

const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .po-imgs { flex: 1 1 380px; max-width: 440px; display: grid; grid-template-columns: 1fr 1fr; gap: .4em .5em; }
[data-demo="${NAME}"] .po-right { flex: 1 1 330px; max-width: 420px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .po-otf { position: relative; }
[data-demo="${NAME}"] .qb-slider.po-lambda input[type=range] { accent-color: var(--po-lambda); }
.reveal [data-demo="${NAME}"] .po-imgs { flex: 0 1 430px; max-width: 430px; }
.reveal [data-demo="${NAME}"] .po-right { flex: 0 1 400px; max-width: 400px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 1 280px; }
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? PRESETS.filter(p => only.includes(p.id)) : PRESETS;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const objOverride = host.dataset.object as ObjectId | undefined;
  const st: State = { ...DEFAULT };

  // ---- layout ----
  const imgs = htmlEl("div", "po-imgs", "", host);
  const pPsf = imagePanel(imgs, "PSF (2.6 µm wide)");
  const pOtf = imagePanel(imgs, `OTF (±${FMAX} cycles/µm)`, "po-otf");
  const pObj = imagePanel(imgs, "object (10 µm field)");
  const pImg = imagePanel(imgs, "image = object ∗ PSF");
  const credit = htmlEl("div", "dc-credit", CELL_CREDIT + ". Treated as the true object with 40 nm pixels (it is itself already a blurred image).", imgs);
  credit.style.gridColumn = "1 / 3";

  const right = htmlEl("div", "po-right", "", host);
  const plot = new Plot(right, { width: 420, height: 290, x: [0, FMAX], y: [-0.2, 1], xlabel: "spatial frequency (cycles/µm)", ylabel: "OTF (contrast kept)", font: 23, margin: { l: 98 } });
  const zero = plot.hline(0, { color: "var(--qb-muted, #777)", width: 1.5, dash: "" });
  const lIdeal = plot.line([], [], { color: "var(--qb-muted, #777)", dash: "7 5", width: 2.5 });
  const lNow = plot.line([], [], { color: seriesColor(0), width: 3.5 });
  const vCut = plot.vline(1, { color: seriesColor(1), text: "2NA/λ" });
  legendRow(right, [
    { label: "this PSF (along fx)", color: seriesColor(0) },
    { label: "ideal lens, in focus", color: "var(--qb-muted, #777)", dash: true },
    { label: "cut-off 2NA/λ", color: seriesColor(1), dash: true },
  ]);
  const ro = htmlEl("div", "qb-readout", "", right);
  void zero;

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const kindS = select<PsfKind>(panel, "PSF", [["pupil", "lens (diffraction + defocus)"], ["motion", "lens + motion blur"], ["gauss", "Gaussian approximation"]], st.kind, v => { st.kind = v; syncVis(); recompute(); });
  const naS = slider(panel, "NA", 0.1, 1.45, 0.05, st.NA, v => { st.NA = v; recompute(); }, v => v.toFixed(2));
  const lamS = slider(panel, "λ (nm)", 400, 700, 10, st.wavelength, v => { st.wavelength = v; colourLambda(); recompute(); }, v => v.toFixed(0));
  lamS.input.closest("label")!.classList.add("po-lambda");
  const dzS = slider(panel, "defocus (µm)", 0, 4, 0.1, st.defocus / 1000, v => { st.defocus = v * 1000; recompute(); }, v => v.toFixed(1));
  const mlS = slider(panel, "motion (µm)", 0.1, 2, 0.05, st.motion / 1000, v => { st.motion = v * 1000; recompute(); }, v => v.toFixed(2));
  const maS = slider(panel, "angle (°)", 0, 180, 5, st.angle, v => { st.angle = v; recompute(); }, v => v.toFixed(0));
  const objS = select<ObjectId>(panel, "object", [["cell", "cell micrograph"], ["star", "Siemens star"], ["beads", "point beads"], ["bars", "bar groups"]], st.object, v => { st.object = v; reobject(); });
  const logC = checkbox(panel, "PSF on a log scale (rings)", st.logPsf, v => { st.logPsf = v; draw(); });
  const sgnC = checkbox(panel, "signed OTF (negative = reversed)", st.signed, v => { st.signed = v; draw(); });
  const strC = checkbox(panel, "stretch the image contrast", st.stretch, v => { st.stretch = v; draw(); });
  button(panel, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", panel);

  function colourLambda(): void { host.style.setProperty("--po-lambda", wavelengthColour(st.wavelength)); }
  function syncVis(): void {
    const show = (s: { input: HTMLInputElement }, on: boolean) => { s.input.closest("label")!.style.display = on ? "" : "none"; };
    show(dzS, st.kind === "pupil"); show(mlS, st.kind === "motion"); show(maS, st.kind === "motion");
  }
  function syncAll(): void {
    kindS.set(st.kind); naS.set(st.NA); lamS.set(st.wavelength); dzS.set(st.defocus / 1000); mlS.set(st.motion / 1000); maS.set(st.angle);
    objS.set(st.object); logC.checked = st.logPsf; sgnC.checked = st.signed; strC.checked = st.stretch;
    colourLambda(); syncVis();
  }
  function load(p: Preset): void {
    Object.assign(st, DEFAULT, p.s);
    if (objOverride && p === p0) st.object = objOverride;
    pbtns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    syncAll(); reobject(false); recompute();
  }

  // ---- computation ----
  let cell: Float64Array | null = null;
  let obj: Float64Array = new Float64Array(N * N), img: Float64Array = new Float64Array(N * N);
  let res = computePSF(st);
  function objectData(): Float64Array | null {
    if (st.object === "star") return starObject();
    if (st.object === "beads") return beadObject();
    if (st.object === "bars") return barObject();
    if (!cell) { loadCell(N).then(g => { cell = Float64Array.from(g.data, v => v / 255); reobject(); }); return null; }
    return cell;
  }
  function reobject(blurNow = true): void {
    const o = objectData();
    credit.style.visibility = st.object === "cell" ? "visible" : "hidden";
    if (!o) return;
    obj = o;
    if (blurNow) { img = blurWith(obj, res); draw(); }
  }
  let pending = false;
  function recompute(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      res = computePSF(st);
      img = blurWith(obj, res);
      draw();
    });
  }

  function draw(): void {
    // PSF
    const crop = centreCrop(res.psf, KPSF);
    const pk = Math.max(...crop);
    if (st.logPsf) pPsf.draw(crop.map(v => Math.log10(Math.max(v / pk, 1e-4))), 2 * KPSF + 1, 2 * KPSF + 1, { lo: -4, hi: 0, map: inferno });
    else pPsf.draw(crop, 2 * KPSF + 1, 2 * KPSF + 1, { lo: 0, hi: pk, map: inferno });
    pPsf.cap.textContent = st.logPsf ? "PSF, log scale (2.6 µm wide)" : "PSF (2.6 µm wide)";
    // OTF image + cut-off circle
    const m = 2 * KMAX + 1, fcBins = 2 * st.NA / st.wavelength * N * PIXEL;
    if (st.signed) pOtf.draw(spectrumCrop(res.re, KMAX), m, m, { lo: -0.1, hi: 0.1, map: diverging });
    else pOtf.draw(spectrumCrop(res.re.map((v, i) => Math.hypot(v, res.im[i])), KMAX), m, m, { lo: 0, hi: 1, map: grey });
    const ctx = pOtf.canvas.getContext("2d")!;
    ctx.strokeStyle = "#d62728"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(KMAX + 0.5, KMAX + 0.5, fcBins, 0, 2 * Math.PI); ctx.stroke();
    pOtf.cap.textContent = st.signed ? "OTF: red > 0, blue < 0" : `|OTF| (±${FMAX} cycles/µm)`;
    // object and image
    let lo = 0, hi = Math.max(...obj);
    if (st.stretch) { lo = Infinity; hi = -Infinity; for (const v of img) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    pObj.draw(obj, N, N, { lo: 0, hi: Math.max(...obj) });
    pImg.draw(img, N, N, st.object === "beads" && !st.stretch ? { lo: 0, hi: Math.max(...img) } : { lo, hi });
    pImg.cap.textContent = st.stretch || st.object === "beads" ? "image (contrast stretched)" : "image = object ∗ PSF";
    // profile plot
    const fc = 2 * st.NA / st.wavelength * 1000;    // cycles/µm
    const cut = otfCut(res, KMAX);
    lNow.set(cut.f, st.signed ? cut.v : cut.v.map(Math.abs));
    lIdeal.set(cut.f, cut.f.map(f => airyOTF(f / fc)));
    plot.setDomain(undefined, st.signed ? [-0.2, 1] : [0, 1]);
    vCut.set(Math.min(fc, FMAX));
    // readouts
    const ray = 0.61 * st.wavelength / st.NA, abbe = st.wavelength / (2 * st.NA);
    const lines = [
      `NA ${st.NA.toFixed(2)} (${immersion(st.NA) > 1 ? "oil, n = 1.518" : "air"}), λ ${st.wavelength} nm`,
      `Rayleigh 0.61λ/NA = ${ray.toFixed(0)} nm`,
      `Abbe λ/2NA = ${abbe.toFixed(0)} nm`,
      `cut-off 2NA/λ = ${fc.toFixed(2)} cycles/µm`,
      `PSF FWHM (along x) = ${psfFWHM(res.psf).toFixed(0)} nm`,
    ];
    if (st.kind === "gauss") lines.push(`Gaussian σ = 0.21λ/NA = ${(0.21 * st.wavelength / st.NA).toFixed(0)} nm`);
    if (st.kind === "motion") lines.push(`motion L = ${st.motion.toFixed(0)} nm: OTF zeros\n  every 1/L = ${(1000 / st.motion).toFixed(2)} cycles/µm`);
    if (st.kind === "pupil" && st.defocus > 0) lines.push(`defocus ${(st.defocus / 1000).toFixed(1)} µm`);
    ro.textContent = lines.join("\n");
    note.textContent = st.object === "bars" ? `Bar groups, periods ${BAR_PERIODS.join(", ")} nm (finest top left).`
      : st.object === "star" ? "Siemens star: 16 bright spokes; the period shrinks towards the centre, so the grey disc in the middle shows where detail is lost."
      : "The blur is circular (the field wraps around at the borders).";
  }

  syncAll();
  load(p0);
}

autoMount(NAME, mount);
