// Self-attention on patch tokens (modern deep learning day, unit attention/). A 64 × 64 image
// cut into 8 × 8 patches = 64 tokens. Click a patch (the query): its 64 attention weights are
// shown over the image, next to the fixed footprint of a 3 × 3 convolution (after L layers).
// Scores mix content similarity (hand-made features, or raw pixels = non-local means) and
// position closeness, with weights w_c, w_p and a temperature T; the right panel shows the
// image after one attention layer (every output patch = weighted average of all patches).
// "Shuffle patches" with no position code shows permutation equivariance. Hand-set weights,
// no training: the point is the mechanism.
// Auto-mounts on <div data-demo="selfattn">. Options: data-preset="position|content|nlm|sharp|shuffle|conv".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { loadCell } from "../lib/cell";
import * as M from "./model";

const NAME = "selfattn";
const { S, P, G, NT } = M;

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; align-items: flex-start; }
[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 340px; min-width: 200px; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; }
[data-demo="${NAME}"] .sa-stage { position: relative; width: 100%; aspect-ratio: 1; }
[data-demo="${NAME}"] .sa-stage canvas, [data-demo="${NAME}"] .sa-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 3px; }
[data-demo="${NAME}"] .sa-stage canvas { image-rendering: pixelated; }
[data-demo="${NAME}"] .sa-stage svg { touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 400px; gap: .35em; }
[data-demo="${NAME}"] .sa-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; line-height: 1.3; }
[data-demo="${NAME}"] .sa-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .sa-dis { opacity: .4; pointer-events: none; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .6em; }
.reveal [data-demo="${NAME}"] figure { flex-basis: 360px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] figure { flex: 0 0 330px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 420px; max-width: 480px; }
.reveal [data-demo="${NAME}"] .sa-lab { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; line-height: 1.2; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type ImgId = "synth" | "noisy" | "cell";
interface State { img: ImgId; wc: number; wp: number; T: number; features: M.Features; noPos: boolean; shuffle: boolean; L: number; q: number }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { img: "synth", wc: 1, wp: 0.5, T: 1, features: "hand", noPos: false, shuffle: false, L: 1, q: 3 * G + 1 };
const PRESETS: Preset[] = [
  { id: "position", label: "position only: a blur kernel", s: { wc: 0, wp: 1, T: 1, q: 3 * G + 3 } },
  { id: "content", label: "content only: similar patches, however far", s: { wc: 1, wp: 0, T: 0.5, q: 1 * G + 4 } },
  { id: "nlm", label: "non-local means: denoise", s: { img: "noisy", features: "pixels", wc: 1, wp: 0, T: 0.7, q: 1 * G + 4 } },
  { id: "sharp", label: "temperature → 0: one patch", s: { wc: 1, wp: 0.5, T: 0.03, q: 6 * G + 1 } },
  { id: "shuffle", label: "no position code: shuffling only reorders", s: { wc: 1, wp: 0, T: 0.5, noPos: true, shuffle: true, q: 1 * G + 4 } },
  { id: "conv", label: "a 3 × 3 conv needs many layers", s: { wc: 1, wp: 0, T: 0.5, L: 3, q: 0 } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE };
  const base: Record<ImgId, Float32Array | null> = { synth: M.synthImage(), noisy: null, cell: null };
  base.noisy = M.addNoise(base.synth!, 35);
  const perm = M.permutation(5);

  const mk = (cap: string) => {
    const f = htmlEl("figure", "", "", host);
    const c = htmlEl("figcaption", "", cap, f);
    const stg = htmlEl("div", "sa-stage", "", f);
    const cv = htmlEl("canvas", "", "", stg); cv.width = S; cv.height = S;
    const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}` }, stg);
    return { f, c, cv, svg };
  };
  const pIn = mk("input: click a patch (the query)"), pOut = mk("after one attention layer");
  const credit = htmlEl("div", "sa-credit", "", pOut.f);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "sa-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  htmlEl("div", "sa-lab", "image · content features", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  const imgBtns = (["synth", "noisy", "cell"] as ImgId[]).map(k => button(row, k === "synth" ? "test image" : k === "noisy" ? "noisy" : "micrograph", () => setImg(k)));
  const featBtns = (["hand", "pixels"] as M.Features[]).map(k => button(row, k === "hand" ? "features" : "raw patch", () => { st.features = k; run(); }));
  const wcS = slider(panel, "content weight", 0, 3, 0.05, st.wc, v => { st.wc = v; run(); });
  const wpS = slider(panel, "position weight", 0, 3, 0.05, st.wp, v => { st.wp = v; run(); });
  const tS = slider(panel, "temperature T", 0, 1.5, 0.01, Math.log10(st.T * 100) / 2, v => { st.T = tOf(v); run(); }, v => tOf(v).toFixed(2));
  const posCb = checkbox(panel, "no position code", st.noPos, v => { st.noPos = v; run(); });
  const shCb = checkbox(panel, "shuffle the patches", st.shuffle, v => { st.shuffle = v; run(); });
  const lS = slider(panel, "conv layers L", 1, 7, 1, st.L, v => { st.L = v; drawOverlay(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "Colour on the input: attention weight of each patch for the query (outlined). Dashed square: what a stack of L 3 × 3 convolutions on the patch grid can see, the same at every position.", panel);
  button(panel, "reset", () => applyPreset(p0));
  void note;

  function tOf(u: number): number { return Math.round(0.01 * 100 ** u * 1000) / 1000; }   // 0.01 … 10
  function uOf(T: number): number { return Math.log10(T / 0.01) / 2; }

  let A: Float64Array = new Float64Array(0), inImg: Float32Array = base.synth!, outImg: Float32Array = base.synth!;

  pIn.svg.addEventListener("pointerdown", ev => {
    const [x, y] = svgPoint(pIn.svg, ev);
    const gx = Math.max(0, Math.min(G - 1, Math.floor(x / P))), gy = Math.max(0, Math.min(G - 1, Math.floor(y / P)));
    st.q = gy * G + gx; drawOverlay(); readout();
  });

  function setImg(k: ImgId): void {
    st.img = k;
    if (k === "cell" && !base.cell) {
      loadCell(S).then(g => { base.cell = Float32Array.from(g.data); run(); }).catch(() => { st.img = "synth"; run(); });
      return;
    }
    run();
  }

  function applyPreset(p: Preset): void {
    Object.assign(st, BASE, p.s);
    wcS.set(st.wc); wpS.set(st.wp); tS.set(uOf(st.T)); posCb.checked = st.noPos; shCb.checked = st.shuffle; lS.set(st.L);
    setImg(st.img);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  function run(): void {
    const src = base[st.img] ?? base.synth!;
    inImg = st.shuffle ? M.shufflePatches(src, perm) : src;
    const wp = st.noPos ? 0 : st.wp;
    (wpS.input.parentElement as HTMLElement).classList.toggle("sa-dis", st.noPos);
    A = M.attentionWeights(M.features(inImg, st.features), { wc: st.wc, wp, T: st.T, features: st.features });
    outImg = M.attend(inImg, A);
    imgBtns.forEach((b, i) => b.classList.toggle("active", ["synth", "noisy", "cell"][i] === st.img));
    featBtns.forEach((b, i) => b.classList.toggle("active", ["hand", "pixels"][i] === st.features));
    credit.textContent = st.img === "cell" ? "Image: TEM micrograph of a mesenchymal stem cell, R. M. Hunt, CC BY 3.0, Wikimedia Commons" : "";
    draw();
  }

  function put(cv: HTMLCanvasElement, img: Float32Array): void {
    const id = new ImageData(S, S);
    for (let i = 0; i < S * S; i++) { const v = Math.max(0, Math.min(255, img[i])); id.data[4 * i] = v; id.data[4 * i + 1] = v; id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
    cv.getContext("2d")!.putImageData(id, 0, 0);
  }

  function draw(): void {
    put(pIn.cv, inImg);
    put(pOut.cv, outImg);
    drawOverlay();
    readout();
  }

  function drawOverlay(): void {
    const svg = pIn.svg;
    svg.innerHTML = "";
    let mx = 0; for (let j = 0; j < NT; j++) mx = Math.max(mx, A[st.q * NT + j]);
    for (let j = 0; j < NT; j++) {
      const a = A[st.q * NT + j] / (mx || 1), [gx, gy] = M.pos(j);
      svgEl("rect", { x: gx * P, y: gy * P, width: P, height: P, fill: "#ff7f0e", "fill-opacity": 0.75 * a, stroke: "rgba(255,255,255,.25)", "stroke-width": 0.15 }, svg);
    }
    const [qx, qy] = M.pos(st.q), r = st.L;
    svgEl("rect", { x: (qx - r) * P, y: (qy - r) * P, width: (2 * r + 1) * P, height: (2 * r + 1) * P, fill: "none", stroke: "#56b4e9", "stroke-width": 0.7, "stroke-dasharray": "2 1.2" }, svg);
    svgEl("rect", { x: qx * P, y: qy * P, width: P, height: P, fill: "none", stroke: "#fff", "stroke-width": 0.8 }, svg);
    const po = pOut.svg; po.innerHTML = "";
    svgEl("rect", { x: qx * P, y: qy * P, width: P, height: P, fill: "none", stroke: "#fff", "stroke-width": 0.6 }, po);
  }

  function readout(): void {
    const row = Array.from({ length: NT }, (_, j) => A[st.q * NT + j]);
    const top = row.map((a, j) => [a, j]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    let ent = 0; for (const a of row) if (a > 0) ent -= a * Math.log(a);
    const [qx, qy] = M.pos(st.q);
    // permutation check: attention on the shuffled image vs the shuffled output of the original
    const src = base[st.img] ?? base.synth!, wp = st.noPos ? 0 : st.wp, opt = { wc: st.wc, wp, T: st.T, features: st.features };
    const sh = M.shufflePatches(src, perm);
    const outSh = M.attend(sh, M.attentionWeights(M.features(sh, st.features), opt));
    const shOut = M.shufflePatches(M.attend(src, M.attentionWeights(M.features(src, st.features), opt)), perm);
    let d = 0; for (let i = 0; i < S * S; i++) d = Math.max(d, Math.abs(outSh[i] - shOut[i]));
    const span = Math.min(G, 2 * st.L + 1);
    ro.textContent =
      `query (${qx}, ${qy}); weights sum to ${row.reduce((a, b) => a + b, 0).toFixed(3)}\n` +
      `top: ${top.map(([a, j]) => `${(100 * a).toFixed(0)}% (${M.pos(j).join(",")})`).join("  ")}\n` +
      `effective no. of patches: ${Math.exp(ent).toFixed(1)}\n` +
      `attention: ${NT} patches in 1 layer\n` +
      `conv, ${st.L} layer${st.L > 1 ? "s" : ""}: ${span}×${span} = ${span * span} patches\n` +
      `shuffle test: max difference ${d.toFixed(1)}`;
  }

  applyPreset(p0);
}

autoMount(NAME, mount);
