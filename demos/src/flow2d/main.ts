// A normalising flow in 2D, stepped through rather than trained live: six affine coupling layers
// trained on two moons (weights at 0, 300, 1 000 and 5 000 training steps, precomputed with this
// demo's own code by tools/figs/flows/train.ts). Left, data space: the exact density
// p(x) = N(f(x); 0, I)·exp(Σ s) as a heat map, held-out data points, and samples made by running
// the layers backwards on Gaussian noise. Right: the data points after the first k layers;
// at k = 6 they should look like a standard Gaussian (circles at 1 and 2 sd). Click on the left
// plot to probe a point: its z, log N(z), the log-determinant Σ s and log p(x).
//
// <div data-demo="flow2d" data-preset="untrained|layers|all|probe|early"></div>
import { autoMount, htmlEl, slider, presets, checkbox, rng } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { viridis, toCss } from "../lib/colour";
import { injectStyle, choiceRow, helperCss, fmt } from "../scorematch1d/dview";
import { Flow, sampleData, importWeights, K_LAYERS, HIDDEN } from "./model";
import WEIGHTS from "./weights.json";

const NAME = "flow2d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const P = `${S}:not(.reveal *)`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .f2-fig { flex: 0 1 360px; min-width: 220px; }
${S} .f2-fig figcaption { text-align: center; font-size: 16px; }
${S} .qb-panel { flex: 1 1 280px; gap: .35em; max-width: 360px; }
${S} .qb-readout { font-size: 14px; }
@media (min-width: 641px) {
  ${P} { flex-wrap: nowrap; }
  ${P} .f2-fig { flex: 0 1 min(360px, calc(100vh - 160px)); min-width: 0; }
  ${P} .qb-note { font-size: 13px; }
}
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .f2-fig { flex: 0 0 390px; }
.reveal ${S} .qb-panel { flex: 1 1 330px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 17px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 17px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

interface Ckpt { step: number; nll: number; w: number[][][] }
const CKPTS = WEIGHTS as unknown as Ckpt[];
interface State { ck: number; k: number; density: boolean; samples: boolean; probe: [number, number] }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { ck: 3, k: 6, density: true, samples: false, probe: [0.5, 0.3] };
const PRESETS: Preset[] = [
  { id: "untrained", label: "untrained: almost the identity", s: { ck: 0, k: 6 } },
  { id: "layers", label: "trained: data to noise, layer by layer", s: { k: 2, density: false } },
  { id: "all", label: "trained: all six layers", s: { k: 6 } },
  { id: "samples", label: "sampling: noise to data", s: { samples: true, density: false } },
  { id: "probe", label: "density: a point off the moons", s: { probe: [0.5, 1.2] } },
  { id: "early", label: "after 300 steps", s: { ck: 1 } },
];
const R = 3, G = 72;
const N_PTS = 400;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const F = host.closest(".reveal") ? 22 : 17;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, ...p0.s };
  const flows = CKPTS.map(c => { const f = new Flow(K_LAYERS, HIDDEN, 1); importWeights(f, c.w); return f; });
  const data = sampleData("moons", N_PTS, rng(5));
  const colours = Array.from({ length: N_PTS }, (_, i) => toCss(viridis(Math.min(1, Math.max(0, (data[2 * i] + 2.2) / 4.4)))));
  const noise = Float64Array.from({ length: 2 * N_PTS }, (() => { const r = rng(6); return () => r.norm(); })());
  const dens: Float64Array[] = [];

  const f1 = htmlEl("figure", "f2-fig", "", host); f1.style.margin = "0";
  const left = new Plot(f1, { width: 380, height: 380, x: [-R, R], y: [-R, R], equal: true, xticks: [-2, 0, 2], yticks: [-2, 0, 2], font: F });
  const heat = left.heatmap(new Float64Array(G * G), G, G, { map: viridis, lo: 0, hi: 0.35 });
  const dataM = left.scatter([], [], { r: 2.6, stroke: "none", opacity: 0.85 });
  const sampM = left.scatter([], [], { color: "#fff", r: 2.6, stroke: "#333", opacity: 0.9 });
  const probeM = left.scatter([], [], { color: seriesColor(1), r: 8, stroke: "#fff" });
  htmlEl("figcaption", "", "data space: density p(x)", f1);
  const f2 = htmlEl("figure", "f2-fig", "", host); f2.style.margin = "0";
  const right = new Plot(f2, { width: 380, height: 380, x: [-R, R], y: [-R, R], equal: true, xticks: [-2, 0, 2], yticks: [-2, 0, 2], font: F });
  const circ = (rad: number) => right.line(Array.from({ length: 81 }, (_, i) => rad * Math.cos(2 * Math.PI * i / 80)), Array.from({ length: 81 }, (_, i) => rad * Math.sin(2 * Math.PI * i / 80)), { color: "var(--qb-muted)", width: 1.5, dash: "5 4" });
  circ(1); circ(2);
  const movedM = right.scatter([], [], { r: 2.6, stroke: "none", opacity: 0.85 });
  const probeR = right.scatter([], [], { color: seriesColor(1), r: 8, stroke: "#fff" });
  const cap2 = htmlEl("figcaption", "", "", f2);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const ckC = choiceRow(panel, "training steps", CKPTS.map((c, i) => ({ v: i, label: String(c.step) })), st.ck, v => { st.ck = v; draw(); });
  const kS = slider(panel, "layers applied k", 0, K_LAYERS, 1, st.k, v => { st.k = v; draw(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const cD = checkbox(row, "density", st.density, v => { st.density = v; draw(); });
  const cS = checkbox(row, "samples", st.samples, v => { st.samples = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click the left plot to probe a point. Points are coloured by their x position, so you can follow them through the layers. Each layer moves only one coordinate (x after even, y after odd layers), by a shift and a stretch that depend on the other coordinate.", panel);

  left.svg.addEventListener("pointerdown", ev => { const [x, y] = left.toData(ev); st.probe = [x, y]; draw(); });

  function density(i: number): Float64Array {
    if (!dens[i]) {
      const f = flows[i], v = new Float64Array(G * G);
      for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) v[r * G + c] = Math.exp(f.logDensity(-R + 2 * R * (c + 0.5) / G, R - 2 * R * (r + 0.5) / G));
      dens[i] = v;
    }
    return dens[i];
  }
  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    ckC.set(st.ck); kS.set(st.k); cD.checked = st.density; cS.checked = st.samples;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }
  function draw(): void {
    const f = flows[st.ck];
    heat.set(st.density ? density(st.ck) : new Float64Array(G * G).fill(NaN));
    dataM.set(Array.from({ length: N_PTS }, (_, i) => data[2 * i]), Array.from({ length: N_PTS }, (_, i) => data[2 * i + 1]), colours);
    if (st.samples) {
      const xs: number[] = [], ys: number[] = [];
      for (let i = 0; i < 250; i++) { const [x, y] = f.inverse(noise[2 * i], noise[2 * i + 1]); xs.push(x); ys.push(y); }
      sampM.set(xs, ys);
    } else sampM.set([], []);
    const mx: number[] = [], my: number[] = [];
    for (let i = 0; i < N_PTS; i++) { const z = f.forward(data[2 * i], data[2 * i + 1], st.k).z; mx.push(z[0]); my.push(z[1]); }
    movedM.set(mx, my, colours);
    cap2.textContent = st.k === 0 ? "the data, before any layer" : st.k === K_LAYERS ? `after all ${K_LAYERS} layers: should be N(0, I)` : `after ${st.k} of ${K_LAYERS} layers`;
    const [px, py] = st.probe, full = f.forward(px, py), part = f.forward(px, py, st.k);
    probeM.set([px], [py]); probeR.set([part.z[0]], [part.z[1]]);
    const logN = -0.5 * (full.z[0] ** 2 + full.z[1] ** 2) - Math.log(2 * Math.PI);
    ro.textContent = `after ${CKPTS[st.ck].step} training steps: held-out −log p = ${fmt(CKPTS[st.ck].nll, 2)}\n` +
      `probe x = (${fmt(px, 2)}, ${fmt(py, 2)}) → z = (${fmt(full.z[0], 2)}, ${fmt(full.z[1], 2)})\n` +
      `log N(z) ${fmt(logN, 2)} + Σ s ${fmt(full.logdet, 2)} = log p(x) ${fmt(logN + full.logdet, 2)}\n` +
      `p(x) = ${fmt(Math.exp(logN + full.logdet), 3)}`;
  }

  apply(p0);
}

autoMount(NAME, mount);
