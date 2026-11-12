// momentrecon (moments elective, units orthogonal-moments/ and zernike/): an image rebuilt
// from its moments up to order N. Panels: the input, then monomials (least squares through
// the Gram system; coordinates in [−1, 1] or [0, 1]), Legendre (sampled at pixel centres,
// averaged over pixels, or the discrete orthogonal polynomials), Zernike (inscribed disc,
// zero outside), and a Fourier low-pass with the same number of coefficients. Each panel
// gives the number of coefficients and the relative error against the clean input; the plot
// shows the error against N for all four. Inputs: a letter, a nucleus outline, a square and
// a crop of the course micrograph; optional Gaussian noise (fixed seed).
// Options: data-preset (blurry, explode, ringing, outside, noise), data-presets (show only
// these preset buttons), data-n, data-image.
import { autoMount, htmlEl, button, slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { cropGrey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { countUpTo, relError } from "../lib/moments";
import { W, NMAX, NMON, LETTER_E, SQUARE, nucleusPoly, polyImage, addNoise, computeAll, monomial, AllRecons, Variant } from "./compute";

const NAME = "momentrecon";
const CM = "var(--qb-c4, #ff7f0e)", CLg = "var(--qb-c1, #1f77b4)", CZ = "var(--qb-c3, #2ca02c)", CF = "var(--qb-c5, #9467bd)";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; gap: .8em; align-items: flex-start; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .mr-ctl { flex: 0 0 250px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .mr-grid { flex: 1 1 600px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .3em .7em; }
.qb-demo[data-demo="${NAME}"] .mr-panel { text-align: center; font-size: 15px; line-height: 1.25; }
.qb-demo[data-demo="${NAME}"] .mr-panel b { font-size: 17px; }
.qb-demo[data-demo="${NAME}"] .mr-img { position: relative; width: 100%; max-width: 180px; margin: 0 auto; aspect-ratio: 1; }
.qb-demo[data-demo="${NAME}"] .mr-img canvas { width: 100%; height: 100%; image-rendering: pixelated; display: block; border: 1px solid var(--qb-grid, #ddd); }
.qb-demo[data-demo="${NAME}"] .mr-img svg { position: absolute; inset: 0; pointer-events: none; }
.qb-demo[data-demo="${NAME}"] .mr-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.qb-demo[data-demo="${NAME}"] .mr-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type ImageId = "E" | "nucleus" | "square" | "cell";
const IMAGES: { id: ImageId; label: string }[] = [
  { id: "E", label: "letter E" }, { id: "nucleus", label: "nucleus" }, { id: "square", label: "square" }, { id: "cell", label: "micrograph" },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const ctl = htmlEl("div", "mr-ctl", "", host);
  const grid = htmlEl("div", "mr-grid", "", host);
  const st = { img: "nucleus" as ImageId, N: 4, variant: "point" as Variant, domain: "sym" as "sym" | "unit", noise: 0 };

  const nS = slider(ctl, "order N", 0, NMAX, 1, st.N, v => { st.N = v; show(); });
  htmlEl("div", "mr-lab", "image", ctl);
  const irow = htmlEl("div", "qb-presets", "", ctl);
  const ib = IMAGES.map(im => button(irow, im.label, () => { st.img = im.id; recompute(); }));
  const NOISES = [0, 0.05, 0.1, 0.2, 0.3];
  const noS = slider(ctl, "noise σ", 0, NOISES.length - 1, 1, 0, v => { st.noise = NOISES[v]; recompute(); }, v => String(NOISES[v]));
  htmlEl("div", "mr-lab", "pixels", ctl);
  const vrow = htmlEl("div", "qb-presets", "", ctl);
  const VARS: [Variant, string][] = [["point", "point samples"], ["integrated", "pixel-integrated"], ["discrete", "discrete (exact)"]];
  const vbt = VARS.map(([v, l]) => button(vrow, l, () => { st.variant = v; recompute(); }));
  htmlEl("div", "mr-lab", "monomial coordinates", ctl);
  const drow = htmlEl("div", "qb-presets", "", ctl);
  const dbt = [button(drow, "[−1, 1]", () => { st.domain = "sym"; monoCurve(); show(); }), button(drow, "[0, 1]", () => { st.domain = "unit"; monoCurve(); show(); })];
  htmlEl("div", "mr-lab", "examples", ctl);
  const prow = htmlEl("div", "qb-presets", "", ctl);
  const credit = htmlEl("div", "mr-credit", "", ctl);

  // ---- panels ----
  function panel(title: string, colour: string, disc = false) {
    const p = htmlEl("div", "mr-panel", "", grid);
    const t = htmlEl("div", "", "", p);
    t.innerHTML = `<b style="color:${colour}">${title}</b>`;
    const box = htmlEl("div", "mr-img", "", p);
    const can = htmlEl("canvas", "", "", box);
    can.width = can.height = W;
    if (disc) {
      const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      s.setAttribute("viewBox", "0 0 64 64");
      s.innerHTML = `<circle cx="32" cy="32" r="32" fill="none" stroke="var(--qb-c3, #2ca02c)" stroke-width="0.7" stroke-dasharray="2 1.5"/>`;
      box.appendChild(s);
    }
    const info = htmlEl("div", "", "", p);
    return { can, info };
  }
  const pIn = panel("input", "var(--qb-fg, #222)");
  const pM = panel("monomials", CM);
  const pL = panel("Legendre", CLg);
  const pZ = panel("Zernike", CZ, true);
  const pF = panel("Fourier (same count)", CF);
  const plotBox = htmlEl("div", "mr-panel", "", grid);
  const plot = new Plot(plotBox, { width: 360, height: 330, x: [0, NMAX], y: [0.05, 2], ylog: true, xlabel: "order N", ylabel: "rel. error", font: 18, margin: { l: 78 } });
  const lM = plot.line([], [], { color: CM, label: "monomials" });
  const lL = plot.line([], [], { color: CLg, label: "Legendre" });
  const lZ = plot.line([], [], { color: CZ, label: "Zernike" });
  const lF = plot.line([], [], { color: CF, label: "Fourier" });
  const vN = plot.vline(st.N, { color: "var(--qb-muted, #999)", dash: "4 3", width: 1.5 });
  plot.legend(undefined, "tr");

  let clean: Float64Array = new Float64Array(W * W), noisy: Float64Array = clean, all: AllRecons | null = null, lo = 0, hi = 1, cell: Float64Array | null = null;
  let monoToken = 0;
  const monoErr: number[] = [];

  function draw(can: HTMLCanvasElement, v: Float64Array): void {
    const c2 = can.getContext("2d")!, id = c2.createImageData(W, W);
    for (let i = 0; i < W * W; i++) {
      const x = v[i], g = Number.isFinite(x) ? Math.max(0, Math.min(255, Math.round(255 * (x - lo) / (hi - lo)))) : 128;
      id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = g; id.data[4 * i + 3] = 255;
    }
    c2.putImageData(id, 0, 0);
  }

  function input(): Float64Array | null {
    if (st.img === "E") return polyImage(LETTER_E);
    if (st.img === "square") return polyImage(SQUARE);
    if (st.img === "nucleus") return polyImage(nucleusPoly());
    return cell;
  }

  function recompute(): void {
    ib.forEach((b, i) => b.classList.toggle("active", IMAGES[i].id === st.img));
    vbt.forEach((b, i) => b.classList.toggle("active", VARS[i][0] === st.variant));
    noS.set(NOISES.indexOf(st.noise));
    credit.textContent = st.img === "cell" ? CELL_CREDIT : "";
    const f = input();
    if (!f) { pIn.info.textContent = "loading…"; return; }
    clean = f; noisy = addNoise(f, st.noise);
    // display range: the input's range plus 30 % on both sides, so over- and undershoots show
    let a = Math.min(...clean), b = Math.max(...clean);
    if (b <= a) b = a + 1;
    lo = a - 0.3 * (b - a); hi = b + 0.3 * (b - a);
    all = computeAll(noisy, clean, st.variant);
    lL.set(all.errLeg.map((_, i) => i), all.errLeg);
    lZ.set(all.errZer.map((_, i) => i), all.errZer);
    lF.set(all.errFour.map((_, i) => i), all.errFour);
    monoCurve();
    show();
  }

  /** The monomial error curve, one order per animation frame (the solves get slow). */
  function monoCurve(): void {
    dbt.forEach((b, i) => b.classList.toggle("active", i === (st.domain === "sym" ? 0 : 1)));
    const token = ++monoToken;
    monoErr.length = 0; lM.set([], []);
    let N = 0;
    const step = () => {
      if (token !== monoToken) return;
      monoErr.push(Math.min(1e3, relError(monomial(noisy, N, st.variant, st.domain), clean)));
      lM.set(monoErr.map((_, i) => i), monoErr.map(v => Math.min(v, 2)));
      if (++N <= NMON) setTimeout(step, 0);
    };
    step();
  }

  function show(): void {
    if (!all) return;
    nS.set(st.N);
    vN.set(st.N);
    const N = st.N, K = countUpTo(N);
    draw(pIn.can, noisy);
    pIn.info.textContent = st.noise > 0 ? `with noise σ = ${st.noise} (errors vs clean)` : `${W} × ${W} = ${W * W} pixels`;
    if (N <= NMON) {
      const m = monomial(noisy, N, st.variant, st.domain);
      draw(pM.can, m);
      pM.info.textContent = `${K} coeff., error ${relError(m, clean).toFixed(3)}`;
    } else {
      const c2 = pM.can.getContext("2d")!; c2.fillStyle = "#888"; c2.fillRect(0, 0, W, W);
      pM.info.textContent = `not computed above N = ${NMON}`;
    }
    draw(pL.can, all.leg[N]); pL.info.textContent = `${K} coeff., error ${all.errLeg[N].toFixed(3)}`;
    draw(pZ.can, all.zer[N]);
    pZ.info.textContent = `${K} real numbers, error ${all.errZer[N].toFixed(3)}`;
    draw(pF.can, all.four[N]); pF.info.textContent = `${K} coeff., error ${all.errFour[N].toFixed(3)}`;
  }

  const presets: { id: string; label: string; s: Partial<typeof st> }[] = [
    { id: "blurry", label: "low orders = blurry ellipse", s: { img: "nucleus", N: 2 } },
    { id: "explode", label: "monomials explode", s: { img: "E", N: 14, domain: "unit" } },
    { id: "ringing", label: "Legendre vs Fourier ringing", s: { img: "square", N: 16 } },
    { id: "outside", label: "Zernike outside the disc", s: { img: "cell", N: 20 } },
    { id: "noise", label: "noise in high orders", s: { img: "nucleus", N: 40, noise: 0.3 } },
  ];
  const only = host.dataset.presets?.split(",").map(x => x.trim());
  const shown = only ? presets.filter(p => only.includes(p.id)) : presets;
  const btns = shown.map(p => button(prow, p.label, () => load(p)));
  const p0 = presets.find(p => p.id === host.dataset.preset) ?? presets[0];
  button(prow, "⟲ reset", () => load(p0));
  function load(p: typeof presets[number]): void {
    Object.assign(st, { img: "nucleus", N: 4, variant: "point", domain: "sym", noise: 0 }, p.s);
    if (p === p0 && host.dataset.n) st.N = Math.max(0, Math.min(NMAX, Number(host.dataset.n)));
    if (p === p0 && host.dataset.image) st.img = host.dataset.image as ImageId;
    btns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    recompute();
  }
  load(p0);
  loadCell(96).then(g => {
    const c = cropGrey(g, 24, 12, W, W);
    cell = Float64Array.from(c.data, v => v / 255);
    if (st.img === "cell") recompute();
  });
}

autoMount(NAME, mount);
