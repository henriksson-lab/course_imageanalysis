// What "aligned" means for two images that may differ in contrast. Fixed: a 96² crop of the
// micrograph; moving: the same crop with its grey values mapped (same contrast, gain + offset,
// inverted, non-monotone "second stain") plus noise. Slide the moving image (shift, rotation):
// the overlay, the joint histogram of the overlapping pixels (fixed value across, moving value
// up), and SSD / NCC / mutual information against the x shift, with the current shift marked.
//
// <div data-demo="jointhist" data-preset="aligned|misaligned|inverted|nonmono|bins|overlap"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Grey, sampleBilinear } from "../lib/imageops";
import { remap, Remap, REMAP_NAMES, mutualInformation, ssdNcc } from "../lib/deform";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { rng } from "../lib/random";
import { Plot, seriesColor } from "../lib/plot";
import { inferno } from "../lib/colour";
import { injectCss } from "../lkflow/flowui";

const NAME = "jointhist", S = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${S} { gap: .8em; flex-wrap: wrap; }
${S} .jh-col { flex: 0 1 300px; min-width: 200px; display: flex; flex-direction: column; gap: .2em; }
${S} .jh-col canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
${S} .jh-cap { font-size: 17px; text-align: center; }
${S} .jh-mid { flex: 0 1 320px; min-width: 220px; display: flex; flex-direction: column; gap: .2em; }
${S} .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .3em; }
${S} .qb-readout { font-size: 15px; }
${S} .jh-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${S} .jh-credit { font-size: 12px; color: var(--qb-muted); }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; }
.reveal ${S} .jh-col { flex: 0 0 280px; }
.reveal ${S} .jh-mid { flex: 0 0 330px; }
.reveal ${S} .jh-cap { font-size: 18px; }
.reveal ${S} .qb-panel { flex: 1 1 320px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .jh-lab { font-size: 16px; }
.reveal ${S} .jh-credit { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;

const N = 96, RANGE = 84;
const MAPS: Remap[] = ["identity", "gain", "inverted", "nonmonotone"];
interface State { map: Remap; tx: number; ty: number; rot: number; bins: number; noise: number }
const DEFAULT: State = { map: "identity", tx: 0, ty: 0, rot: 0, bins: 16, noise: 0.03 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "aligned", label: "aligned: a thin line", s: {} },
  { id: "misaligned", label: "misaligned: a cloud", s: { tx: 6, ty: 3 } },
  { id: "inverted", label: "inverted: NCC −1, MI high, SSD worst", s: { map: "inverted" } },
  { id: "nonmono", label: "non-monotone: only MI peaks at the truth", s: { map: "nonmonotone", tx: 4 } },
  { id: "bins", label: "too many bins: MI noisy", s: { bins: 128, noise: 0.05, tx: 3 } },
  { id: "overlap", label: "tiny overlap: MI misleads", s: { tx: 80, bins: 64 } },
];

function mount(host: HTMLElement): void {
  injectCss(NAME, CSS);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const col = htmlEl("div", "jh-col", "", host);
  htmlEl("div", "jh-cap", "overlay: fixed magenta, moving green", col);
  const cv = htmlEl("canvas", "", "", col); cv.width = N; cv.height = N;
  htmlEl("div", "jh-credit", CELL_CREDIT, col);

  const mid = htmlEl("div", "jh-mid", "", host);
  const hp = new Plot(mid, { width: 320, height: 250, x: [0, 1], y: [0, 1], equal: true, xlabel: "fixed value", ylabel: "moving value", font: 18, xticks: [0, 0.5, 1], yticks: [0, 0.5, 1], margin: { l: 64, r: 8, t: 8, b: 58 } });
  let heat = hp.heatmap(new Float64Array(4), 2, 2, { map: inferno, flipY: false, lo: 0 });
  const cp = new Plot(mid, { width: 320, height: 210, x: [-RANGE, RANGE], y: [0, 1.05], xticks: [-80, -40, 0, 40, 80], xlabel: "x shift (px)", font: 18, yticks: [], margin: { l: 14, r: 8, t: 8, b: 58 } });
  const cS = cp.line([], [], { color: seriesColor(0), width: 3, label: "−SSD" });
  const cN = cp.line([], [], { color: seriesColor(1), width: 3, label: "NCC" });
  const cM = cp.line([], [], { color: seriesColor(2), width: 3, label: "MI" });
  const vNow = cp.vline(0, { color: "var(--qb-fg)", width: 2, dash: "5 4" });
  cp.legend(undefined, "tl");

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "jh-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); make(); } })));
  htmlEl("div", "jh-lab", "moving image contrast", panel);
  const mapBtns = presets(panel, MAPS.map(m => ({ label: REMAP_NAMES[m], apply: () => { st.map = m; make(); } })));
  const txS = slider(panel, "x shift", -RANGE, RANGE, 1, st.tx, v => { st.tx = v; update(); });
  const tyS = slider(panel, "y shift", -10, 10, 1, st.ty, v => { st.ty = v; make(); });
  const rS = slider(panel, "rotation (°)", -20, 20, 1, st.rot, v => { st.rot = v; make(); });
  const bS = slider(panel, "bins", 2, 128, 1, st.bins, v => { st.bins = v; make(); });
  const nS = slider(panel, "noise", 0, 0.2, 0.01, st.noise, v => { st.noise = v; make(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); make(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  function sync(): void { txS.set(st.tx); tyS.set(st.ty); rS.set(st.rot); bS.set(st.bins); nS.set(st.noise); }

  let cell: Grey | null = null;
  loadCell().then(g => { cell = g; make(); });
  // fixed: crop at (80, 80) of the 256² micrograph, values 0..1; moving: remapped copy, sampled with a pose
  let F = new Float64Array(N * N), M = new Float64Array(N * N);
  const X0 = 80, Y0 = 80;
  /** Moving image as seen at trial pose: M(x) = remap(cell(R(x − c) + c − t)) restricted to its own 96² frame. */
  function movingAt(tx: number): { vals: Float64Array; mask: Uint8Array } {
    const vals = new Float64Array(N * N), mask = new Uint8Array(N * N);
    const a = st.rot * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), cx = (N - 1) / 2;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      // position of fixed pixel (x, y) in the moving frame
      const dx = x - tx - cx, dy = y - st.ty - cx, mx = cx + c * dx + s * dy, my = cx - s * dx + c * dy;
      if (mx < 0 || my < 0 || mx > N - 1 || my > N - 1) continue;
      const i = y * N + x;
      mask[i] = 1;
      vals[i] = sampleBilinear({ width: N, height: N, data: M } as unknown as Grey, mx, my);
    }
    return { vals, mask };
  }
  function make(): void {
    if (!cell) return;
    mapBtns.forEach((b, i) => b.classList.toggle("active", MAPS[i] === st.map));
    const R = rng(9);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const v = cell.data[(Y0 + y) * 256 + X0 + x] / 255;
      F[y * N + x] = v;
      M[y * N + x] = Math.max(0, Math.min(1, remap(v, st.map) + st.noise * R.norm()));
    }
    // curves over the x shift
    const xs: number[] = [], ss: number[] = [], nc: number[] = [], mi: number[] = [];
    for (let t = -RANGE; t <= RANGE; t++) {
      const m = movingAt(t), a = ssdNcc(F, m.vals, m.mask), b = mutualInformation(F, m.vals, st.bins, m.mask);
      xs.push(t); ss.push(-a.ssd); nc.push(a.ncc); mi.push(b.mi);
    }
    const norm = (v: number[]) => { const lo = Math.min(...v), hi = Math.max(...v); return v.map(x => (x - lo) / (hi - lo || 1)); };
    cS.set(xs, norm(ss)); cN.set(xs, norm(nc)); cM.set(xs, norm(mi));
    curves = { ss, nc, mi };
    update();
  }
  let curves = { ss: [] as number[], nc: [] as number[], mi: [] as number[] };
  function update(): void {
    if (!cell) return;
    const m = movingAt(st.tx);
    // overlay
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const f = 255 * F[i], g = m.mask[i] ? 255 * m.vals[i] : 0, o = 4 * i;
      id.data[o] = f; id.data[o + 1] = g; id.data[o + 2] = f; id.data[o + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const r = mutualInformation(F, m.vals, st.bins, m.mask), a = ssdNcc(F, m.vals, m.mask);
    // joint histogram: counts[a*bins + b], a = fixed (x), b = moving (y); heatmap wants rows = y
    const B = st.bins, H = new Float64Array(B * B);
    for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) H[j * B + i] = Math.log(1 + r.counts[i * B + j]);
    heat.remove();
    heat = hp.heatmap(H, B, B, { map: inferno, flipY: false, lo: 0, maxCells: 1024 });
    vNow.set(st.tx);
    ro.textContent = `overlap ${a.n} pixels\nSSD ${a.ssd.toFixed(4)}   NCC ${a.ncc.toFixed(3)}\nMI ${r.mi.toFixed(3)} bits  (H fixed ${r.ha.toFixed(2)}, H moving ${r.hb.toFixed(2)}, joint ${r.hab.toFixed(2)})` +
      (curves.mi.length ? `\nbest x shift: SSD ${argmax(curves.ss) - RANGE}, NCC ${argmax(curves.nc) - RANGE}, MI ${argmax(curves.mi) - RANGE}` : "");
  }
}

function argmax(v: number[]): number { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b; }

autoMount(NAME, mount);
