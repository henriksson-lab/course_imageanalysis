// Generative restoration and hallucination (modern deep learning day, unit diffusion/).
// A 32 × 32 synthetic cell is blurred and noisy (or has holes); the prior is a mixture of
// small Gaussians around 300 training cells, so the posterior is exact (genrestore/model.ts).
// Panels: truth ("you never have this"), measurement, classical restoration (Wiener / smooth
// filling), posterior mean, four posterior samples, the disagreement (pixel sd over 32
// samples), and the training cells the posterior trusts most. Under each restoration:
// nuclei, nucleoli, area and PSNR against the truth.
// Auto-mounts on <div data-demo="genrestore">. Options: data-preset="mild|heavy|meanvs|nodiv|ring|hole|psnr".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import * as M from "./model";

const NAME = "genrestore";
const N = M.N;

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .gr-grid { flex: 1 1 640px; min-width: 300px; display: grid; grid-template-columns: repeat(4, 1fr); gap: .35em .5em; }
[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 15px; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .gr-m { font-size: 13px; text-align: center; font-family: ui-monospace, monospace; line-height: 1.2; white-space: pre; }
[data-demo="${NAME}"] .gr-samples { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; }
[data-demo="${NAME}"] .gr-top { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; }
[data-demo="${NAME}"] .gr-top div { font-size: 12px; text-align: center; }
[data-demo="${NAME}"] .gr-truth canvas { outline: 2px dashed var(--qb-muted); }
[data-demo="${NAME}"] .gr-meas canvas { cursor: crosshair; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .3em; }
[data-demo="${NAME}"] .gr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; line-height: 1.3; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .6em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .gr-grid { flex: 0 1 600px; gap: .2em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 440px; max-width: 500px; }
.reveal [data-demo="${NAME}"] .gr-lab { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; line-height: 1.15; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State { truth: M.CellSpec; prior: M.PriorId; mode: M.Mode; sigma: number; noise: number; hole: Uint8Array; seed: number }
interface Preset { id: string; label: string; s: Partial<State> }

function squareHole(x0: number, y0: number, w: number, h: number): Uint8Array {
  const m = new Uint8Array(N * N);
  for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) if (x >= 0 && y >= 0 && x < N && y < N) m[y * N + x] = 1;
  return m;
}

const T_PLAIN: M.CellSpec = { kind: "plain", nucleoli: 2, seed: 5 };
const PRESETS: Preset[] = [
  { id: "mild", label: "mild blur: everything agrees", s: { truth: T_PLAIN, sigma: 1, noise: 0.02 } },
  { id: "heavy", label: "heavy blur: samples invent nucleoli", s: { truth: T_PLAIN, sigma: 3.5, noise: 0.08 } },
  { id: "nodiv", label: "prior without dividing cells", s: { truth: { kind: "dividing", nucleoli: 0, seed: 7 }, prior: "nodiv", sigma: 2.5, noise: 0.05 } },
  { id: "ring", label: "unusual nucleus: made normal", s: { truth: { kind: "ring", nucleoli: 0, seed: 9 }, sigma: 2, noise: 0.05 } },
  { id: "three", label: "three nucleoli: one is lost", s: { truth: { kind: "three", nucleoli: 3, seed: 11 }, sigma: 2, noise: 0.05 } },
  { id: "hole", label: "inpainting: what is in the hole?", s: { truth: { kind: "plain", nucleoli: 1, seed: 13 }, mode: "inpaint", noise: 0.02, hole: squareHole(10, 10, 12, 12) } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const priors = new Map<M.PriorId, M.Prior>();
  const prior = (id: M.PriorId) => { let p = priors.get(id); if (!p) { p = M.makePrior(id); priors.set(id, p); } return p; };
  const st: State = { truth: T_PLAIN, prior: "matched", mode: "blur", sigma: 2, noise: 0.05, hole: new Uint8Array(N * N), seed: 5 };

  const grid = htmlEl("div", "gr-grid", "", host);
  const fig = (cap: string, cls = "") => {
    const f = htmlEl("figure", cls, "", grid);
    const c = htmlEl("figcaption", "", cap, f);
    const cv = htmlEl("canvas", "", "", f); cv.width = N; cv.height = N;
    const m = htmlEl("div", "gr-m", "", f);
    return { f, c, cv, m };
  };
  const fT = fig("truth (you never have it)", "gr-truth"), fM = fig("measurement", "gr-meas"), fC = fig("classical"), fMean = fig("posterior mean");
  const fS = htmlEl("figure", "", "", grid);
  htmlEl("figcaption", "", "four posterior samples", fS);
  const sg = htmlEl("div", "gr-samples", "", fS);
  const sCv = Array.from({ length: 4 }, () => { const c = htmlEl("canvas", "", "", sg); c.width = N; c.height = N; return c; });
  const sM = htmlEl("div", "gr-m", "", fS);
  const fD = fig("disagreement (sd)");
  const fTop = htmlEl("figure", "", "", grid);
  fTop.style.gridColumn = "span 2";
  htmlEl("figcaption", "", "training cells the posterior trusts", fTop);
  const tg = htmlEl("div", "gr-top", "", fTop);
  const topCells = Array.from({ length: 6 }, () => { const d = htmlEl("div", "", "", tg); const c = htmlEl("canvas", "", "", d); c.width = N; c.height = N; const t = htmlEl("div", "", "", d); return { c, t }; });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "gr-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  htmlEl("div", "gr-lab", "true cell", panel);
  const r1 = htmlEl("div", "qb-presets", "", panel);
  button(r1, "new cell", () => { st.seed++; st.truth = { kind: "plain", nucleoli: st.seed % 3, seed: 500 + st.seed }; run(); });
  button(r1, "dividing", () => { st.seed++; st.truth = { kind: "dividing", nucleoli: 0, seed: 600 + st.seed }; run(); });
  button(r1, "protrusion", () => { st.seed++; st.truth = { kind: "protrusion", nucleoli: 1, seed: 700 + st.seed }; run(); });
  button(r1, "3 nucleoli", () => { st.truth = { kind: "three", nucleoli: 3, seed: 11 }; run(); });
  button(r1, "ring", () => { st.truth = { kind: "ring", nucleoli: 0, seed: 9 }; run(); });
  htmlEl("div", "gr-lab", "prior (training cells)", panel);
  const r2 = htmlEl("div", "qb-presets", "", panel);
  const prBtns = (["matched", "nodiv", "noprot"] as M.PriorId[]).map(p => button(r2, p === "matched" ? "all kinds" : p === "nodiv" ? "no dividing" : "no protrusions", () => { st.prior = p; run(); }));
  htmlEl("div", "gr-lab", "measurement", panel);
  const r3 = htmlEl("div", "qb-presets", "", panel);
  const mdBtns = (["blur", "inpaint"] as M.Mode[]).map(m => button(r3, m === "blur" ? "blur + noise" : "holes (click the measurement)", () => { st.mode = m; run(); }));
  const sigS = slider(panel, "blur σ", 0.5, 4, 0.25, st.sigma, v => { st.sigma = v; run(); });
  const noiS = slider(panel, "noise", 0.01, 0.15, 0.01, st.noise, v => { st.noise = v; run(); });
  const r4 = htmlEl("div", "qb-presets", "", panel);
  button(r4, "resample", () => { sampleSeed++; run(); });
  button(r4, "clear holes", () => { st.hole = new Uint8Array(N * N); run(); });
  button(r4, "reset", () => applyPreset(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Prior: a narrow Gaussian around each of 300 training cells (six nucleus outlines; nucleoli, protrusions and divisions vary), so the posterior is exact. Measures: nuclei and nucleoli counted by thresholds 0.3 and 0.7.", panel);
  let sampleSeed = 5;

  fM.cv.addEventListener("pointerdown", ev => {
    if (st.mode !== "inpaint") return;
    const b = fM.cv.getBoundingClientRect(), x = Math.floor((ev.clientX - b.left) / b.width * N), y = Math.floor((ev.clientY - b.top) / b.height * N);
    const add = !st.hole[y * N + x];
    for (let dy = -2; dy <= 3; dy++) for (let dx = -2; dx <= 3; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < N && yy < N) st.hole[yy * N + xx] = add ? 1 : 0; }
    run();
  });

  function applyPreset(p: Preset): void {
    Object.assign(st, { truth: T_PLAIN, prior: "matched", mode: "blur", sigma: 2, noise: 0.05, hole: new Uint8Array(N * N) }, p.s);
    st.hole = Uint8Array.from(st.hole);
    sigS.set(st.sigma); noiS.set(st.noise); sampleSeed = 5;
    run();
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  function put(cv: HTMLCanvasElement, x: ArrayLike<number>, lo = 0, hi = 1, hole?: Uint8Array): void {
    const id = new ImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const v = 255 * Math.max(0, Math.min(1, (x[i] - lo) / (hi - lo)));
      const h = hole && hole[i];
      id.data[4 * i] = h ? 40 : v; id.data[4 * i + 1] = h ? 90 : v; id.data[4 * i + 2] = h ? 200 : v; id.data[4 * i + 3] = 255;
    }
    cv.getContext("2d")!.putImageData(id, 0, 0);
  }
  const fmt = (m: M.CellMeasures, p?: number) => `nuclei ${m.nuclei} · nucleoli ${m.nucleoli}\narea ${m.area}` + (p !== undefined ? ` · ${p.toFixed(1)} dB` : "");

  function run(): void {
    const P = prior(st.prior), x = M.cell(st.truth);
    const meas = M.measure(x, st.mode, st.sigma, st.noise, st.hole);
    const R = M.restore(P, meas, 32, sampleSeed);
    put(fT.cv, x); fT.m.textContent = fmt(M.measureCell(x));
    put(fM.cv, meas.mode === "inpaint" ? meas.y : meas.shown, 0, 1, meas.mode === "inpaint" ? st.hole : undefined); fM.m.textContent = st.mode === "blur" ? `σ ${st.sigma} · noise ${st.noise}` : "blue: not measured";
    put(fC.cv, R.classical); fC.m.textContent = fmt(M.measureCell(R.classical), M.psnr(R.classical, x));
    put(fMean.cv, R.mean); fMean.m.textContent = fmt(M.measureCell(R.mean), M.psnr(R.mean, x));
    R.samples.slice(0, 4).forEach((s, k) => put(sCv[k], s));
    const cnt = R.samples.map(s => M.measureCell(s));
    const hist = new Map<string, number>();
    for (const c of cnt) { const k = `${c.nuclei}/${c.nucleoli}`; hist.set(k, (hist.get(k) ?? 0) + 1); }
    sM.textContent = `first 4: nucleoli ${cnt.slice(0, 4).map(c => c.nucleoli).join(" ")}\n${M.psnr(R.samples[0], x).toFixed(1)} dB (first)`;
    let mx = 0; for (const v of R.sd) mx = Math.max(mx, v);
    put(fD.cv, R.sd, 0, 0.25); fD.m.textContent = `max sd ${mx.toFixed(2)}`;
    const order = Array.from(R.post.w).map((w, i) => [w, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, 6);
    order.forEach(([w, i], k) => { put(topCells[k].c, P.images[i]); topCells[k].t.textContent = `${(100 * w).toFixed(w < 0.01 ? 1 : 0)} % ${P.specs[i].kind === "plain" ? P.specs[i].nucleoli + " ncl" : P.specs[i].kind}`; });
    prBtns.forEach((b, i) => b.classList.toggle("active", ["matched", "nodiv", "noprot"][i] === st.prior));
    mdBtns.forEach((b, i) => b.classList.toggle("active", ["blur", "inpaint"][i] === st.mode));
    (sigS.input.parentElement as HTMLElement).style.opacity = st.mode === "blur" ? "1" : ".4";
    ro.textContent = `32 posterior samples, nuclei/nucleoli:\n` + [...hist.entries()].sort().map(([k, v]) => `  ${k}: ${v}`).join("\n") +
      `\ntruth: ${M.measureCell(x).nuclei}/${M.measureCell(x).nucleoli}   prior: ${P.images.length} cells`;
  }

  applyPreset(p0);
}

autoMount(NAME, mount);
