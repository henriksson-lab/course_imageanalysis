// Neighbourhood filters opened up: the weights of the bilateral filter and of non-local means at
// one pixel. Click a pixel of the image (test card with seeded noise, or the micrograph):
//  - bilateral: its spatial factor, range factor and their product over the window;
//  - non-local means: the weight of every pixel in the search window (from patch similarity),
//    with the five best-matching patches outlined on the image;
//  - readouts: effective number of averaged pixels N_eff = (Σw)²/Σw², estimate and truth.
// The large image can show the noisy input, the bilateral result (iterated 1, 3 or 10 times),
// non-local means, or the joint bilateral filter guided by a second, clean channel that has a
// step the noisy image lacks (across the ramp at x = 50).
//
// <div data-demo="patchweights" data-preset="flat|edge|corner|noisy|lattice|unique|hsmall|search|iterate|joint"></div>
import { autoMount, htmlEl, button, slider, checkbox, presets as presetRow } from "../lib/ui";
import { CELL_CREDIT } from "../lib/cell";
import { bilateralWeights, nlmWeights, jointBilateral, testCard, addNoise } from "../lib/pde";
import { bilateral, nonLocalMeans } from "../lib/denoise";
import { ImgView, injectStyle } from "../canny/imgview";
import { choice, cell128, toGrey, PS_CSS } from "../heatflow/common";

const NAME = "patchweights";
const STYLE = PS_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .pw-main { flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .pw-top { display: flex; gap: .7em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .pw-big { flex: 0 0 330px; }
.qb-demo[data-demo="${NAME}"] .pw-maps { flex: 1 1 0; display: grid; grid-template-columns: 1fr 1fr; gap: .3em .6em; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .pw-maps .ev-fig figcaption { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 460px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 345px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .pw-main { flex: 0 0 770px; }
.reveal .qb-demo[data-demo="${NAME}"] .pw-maps { max-width: 380px; }
.reveal .qb-demo[data-demo="${NAME}"] .pw-maps .ev-fig figcaption { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .2em .45em; }
@media (max-width: 860px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } .qb-demo[data-demo="${NAME}"] .pw-top { flex-wrap: wrap; } }
`;

type View = "noisy" | "bilateral" | "nlm" | "joint";
interface State { image: "card" | "cell"; noise: number; px: number; py: number; ss: number; sr: number; P: number; S: number; h: number; selfFix: boolean; iters: number; view: View }
const DEFAULT: State = { image: "card", noise: 10, px: 22, py: 22, ss: 2, sr: 25, P: 2, S: 7, h: 15, selfFix: false, iters: 1, view: "noisy" };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "flat", label: "flat: a Gaussian", s: { px: 22, py: 22 } },
  { id: "edge", label: "edge: a half-moon", s: { px: 6, py: 22 } },
  { id: "corner", label: "corner: a quarter", s: { px: 6, py: 6 } },
  { id: "noisy", label: "noise ≈ edge height", s: { px: 6, py: 22, noise: 60, sr: 120 } },
  { id: "lattice", label: "NLM, lattice: all copies", s: { px: 82, py: 20, S: 10 } },
  { id: "unique", label: "NLM, unique spot: only itself", s: { px: 46, py: 76, S: 10 } },
  { id: "hsmall", label: "h too small", s: { px: 82, py: 20, S: 10, h: 5 } },
  { id: "search", label: "search window too small", s: { px: 82, py: 20, S: 2 } },
  { id: "iterate", label: "iterated: cartoon", s: { view: "bilateral", iters: 10, sr: 30, px: 50, py: 112 } },
  { id: "joint", label: "joint: copies the guide's edge", s: { view: "joint", px: 50, py: 112, sr: 20, ss: 4 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "pw-main", "", host);
  const top = htmlEl("div", "pw-top", "", main);
  const bigDiv = htmlEl("div", "pw-big", "", top);
  const big = new ImgView(bigDiv, "noisy image: click a pixel");
  const maps = htmlEl("div", "pw-maps", "", top);
  const mS = new ImgView(maps, "bilateral: spatial"), mR = new ImgView(maps, "bilateral: range");
  const mP = new ImgView(maps, "bilateral: product"), mN = new ImgView(maps, "NLM weights");
  const ro = htmlEl("div", "qb-readout", "", main);
  const credit = htmlEl("div", "ps-credit", CELL_CREDIT, main);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const setView = choice(panel, "show", [["noisy", "noisy"], ["bilateral", "bilateral"], ["nlm", "NLM"], ["joint", "joint bilateral"]], id => { st.view = id as View; recompute(); });
  const setIt = choice(panel, "bilateral passes", [["1", "1"], ["3", "3"], ["10", "10"]], id => { st.iters = Number(id); if (st.view === "bilateral") recompute(); else setIt(id); });
  const ssS = slider(panel, "σs", 0.5, 5, 0.25, st.ss, v => { st.ss = v; drawMaps(); });
  const srS = slider(panel, "σr", 2, 150, 1, st.sr, v => { st.sr = v; drawMaps(); });
  const setP = choice(panel, "patch", [["1", "3 × 3"], ["2", "5 × 5"], ["3", "7 × 7"]], id => { st.P = Number(id); drawMaps(); });
  const sS = slider(panel, "search", 2, 15, 1, st.S, v => { st.S = v; drawMaps(); }, v => `${2 * v + 1}`);
  const hS = slider(panel, "h", 2, 60, 1, st.h, v => { st.h = v; drawMaps(); });
  const noiseS = slider(panel, "noise σn", 0, 60, 1, st.noise, v => { st.noise = v; });
  const fixC = checkbox(panel, "NLM: own weight = best other weight", st.selfFix, v => { st.selfFix = v; drawMaps(); });
  const setImg = choice(panel, "image", [["card", "test card"], ["cell", "micrograph"]], id => { st.image = id as State["image"]; makeImage(); });
  for (const s of [ssS, srS, sS, hS]) s.input.addEventListener("change", () => { if (st.view !== "noisy") recompute(); });
  noiseS.input.addEventListener("change", () => makeImage());
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImage(); });

  big.onPick((x, y) => { st.px = x; st.py = y; drawMaps(); });

  const card = testCard();
  let cellImg: Float64Array | null = null, truth: Float64Array = card.truth, noisy: Float64Array = card.truth, shown: Float64Array = card.truth;
  // the guide for the joint bilateral: the clean card with an extra step across the ramp (x = 50)
  const guide = Float64Array.from(card.truth);
  for (let y = 98; y < 126; y++) for (let x = 34; x < 66; x++) guide[y * 128 + x] = x < 50 ? 150 : 90;

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); makeImage(); }
  function sync(): void { setView(st.view); setIt(String(st.iters)); ssS.set(st.ss); srS.set(st.sr); setP(String(st.P)); sS.set(st.S); hS.set(st.h); noiseS.set(st.noise); fixC.checked = st.selfFix; setImg(st.image); }
  function makeImage(): void {
    if (st.image === "cell" && !cellImg) { cell128().then(g => { cellImg = Float64Array.from(g.data); makeImage(); }); return; }
    truth = st.image === "cell" ? cellImg! : card.truth;
    noisy = st.noise > 0 ? addNoise(truth, st.noise, 9) : Float64Array.from(truth);
    recompute();
  }
  function recompute(): void {
    setView(st.view); setIt(String(st.iters));
    if (st.view === "noisy") shown = noisy;
    else if (st.view === "bilateral") { let u: Float64Array = noisy; for (let k = 0; k < st.iters; k++) u = bilateral(u, 128, 128, st.ss, st.sr); shown = u; }
    else if (st.view === "nlm") shown = nonLocalMeans(noisy, 128, 128, st.P, st.S, st.h);
    else shown = jointBilateral(noisy, st.image === "cell" ? truth : guide, 128, 128, st.ss, st.sr);
    drawMaps();
  }

  function drawMaps(): void {
    const x = st.px, y = st.py;
    big.grey(toGrey(shown, 128, 128), { mode: "clip", lo: 0, hi: 255 });
    const vname = st.view === "noisy" ? "noisy image" : st.view === "bilateral" ? `bilateral × ${st.iters}` : st.view === "nlm" ? "non-local means" : "joint bilateral (guide: 2nd channel)";
    big.setCaption(`${vname}: click a pixel`);
    credit.style.display = st.image === "cell" ? "" : "none";
    // weights are computed on the noisy image (joint view: range from the guide)
    const g = st.view === "joint" ? (st.image === "cell" ? truth : guide) : undefined;
    const B = bilateralWeights(noisy, 128, 128, x, y, st.ss, st.sr, Math.ceil(2 * st.ss), g);
    const show = (v: ImgView, a: Float64Array, n: number) => { let m = 0; for (const q of a) m = Math.max(m, q); v.grey(toGrey(a, n, n), { mode: "clip", lo: 0, hi: m || 1 }); };
    show(mS, B.spatial, B.size); show(mR, B.range, B.size); show(mP, B.product, B.size);
    const N = nlmWeights(noisy, 128, 128, x, y, st.P, st.S, st.h, st.selfFix);
    show(mN, N.weights, N.size);
    for (const v of [mS, mR, mP]) { v.clear(); v.box(B.r, B.r, 0, "#e8402a", 2); }
    mN.clear(); mN.box(st.S, st.S, 0, "#e8402a", 2);
    for (const t of N.top) mN.box(t.x - x + st.S, t.y - y + st.S, 0, "#ffd400", 1.5);
    mN.setCaption(`NLM weights (${N.size} × ${N.size})`);
    // overlays on the big image
    big.clear();
    big.box(x, y, B.r, "#22d3ee", 2);
    big.box(x, y, st.S, "#ff9f1c", 2);
    for (const t of N.top) big.box(t.x, t.y, st.P, "#ffd400", 1.5);
    big.box(x, y, st.P, "#e8402a", 2.5);
    const tr = truth[y * 128 + x];
    ro.textContent =
      `pixel (${x}, ${y}): noisy ${noisy[y * 128 + x].toFixed(1)}, truth ${tr.toFixed(1)}\n` +
      `bilateral (cyan window): estimate ${B.estimate.toFixed(1)}, N_eff = ${B.neff.toFixed(1)} → noise ÷ ${Math.sqrt(B.neff).toFixed(1)}\n` +
      `NLM (orange search window, yellow: 5 best patches): estimate ${N.estimate.toFixed(1)}, N_eff = ${N.neff.toFixed(1)}, own weight ${(100 * N.selfWeight).toFixed(0)} %`;
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
