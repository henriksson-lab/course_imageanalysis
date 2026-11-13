// Latent diffusion in the 2D latent space of the cell-shape VAE (VAE day). Left: the latent
// plane with the 800 training codes coloured by kind, the circle that holds 95 % of the VAE
// prior N(0, I), and the reverse-diffusion paths of 16 samples (exact denoiser of the noised
// codes). Right: 16 decoded samples, and the kind proportions among 200 samples (kind of the
// nearest code) against the training set. Controls: source (latent diffusion / VAE prior),
// kind + guidance w, spread of the codes (encoder sds or tiny: a memorising model), steps,
// sampler.
//
// <div data-demo="latentdiff" data-preset="prior|latent|dividing|caricature|copies"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import * as M from "./model";
import { CODE_MU, CODE_KIND } from "./latent_codes";
import { injectStyle, choiceRow, helperCss, imgPanel, fmt, type ImgPanel } from "../scorematch1d/dview";

const NAME = "latentdiff";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .ld-main { flex: 0 1 440px; min-width: 260px; }
${S} .ld-side { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${S} .ld-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; }
${S} .ld-grid figure.dv-img canvas { border: 1px solid var(--qb-grid); }
${S} .qb-panel { flex: 1 1 280px; gap: .35em; }
${S} .qb-readout { font-size: 15px; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .ld-main { flex: 0 0 420px; }
.reveal ${S} .ld-side { flex: 0 0 290px; }
.reveal ${S} .qb-panel { flex: 1 1 400px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

const KCOL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"];
interface State { source: M.Source; kind: number | null; w: number; steps: number; eta: number; tiny: boolean; seed: number }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { source: "diffusion", kind: null, w: 0, steps: 40, eta: 0, tiny: false, seed: 5 };
const PRESETS: Preset[] = [
  { id: "prior", label: "VAE prior: some nonsense shapes", s: { source: "prior" } },
  { id: "latent", label: "latent diffusion: where real codes are", s: { source: "diffusion" } },
  { id: "dividing", label: "dividing cells on demand", s: { kind: 2, w: 0 } },
  { id: "caricature", label: "strong guidance: caricatures", s: { kind: 2, w: 4 } },
  { id: "copies", label: "tiny spreads: it only copies", s: { tiny: true } },
];
const N = 200, NPATHS = 16;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  // larger labels on slides (the plots are scaled down there)
  const F = host.closest(".reveal") ? 23 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const st: State = { ...BASE, ...p0.s };

  const main = htmlEl("div", "ld-main", "", host);
  const pl = new Plot(main, { width: 460, height: 460, x: [-4, 4], y: [-4, 4], equal: true, xlabel: "latent z₁", ylabel: "latent z₂", xticks: [-4, -2, 0, 2, 4], yticks: [-4, -2, 0, 2, 4], font: F });
  const circ = pl.ellipse([0, 0], [[1, 0], [0, 1]], 2.448, { color: "var(--qb-muted)", dash: "6 5", fillOpacity: 0, width: 2 });
  void circ;
  const xs = Float64Array.from({ length: M.NCODES }, (_, i) => CODE_MU[2 * i]), ys = Float64Array.from({ length: M.NCODES }, (_, i) => CODE_MU[2 * i + 1]);
  pl.scatter(xs, ys, { r: 2.6, colors: CODE_KIND.map(k => KCOL[k]), opacity: 0.55, stroke: "none" });
  const paths = pl.mark<[Float64Array[]]>((g, P) => {
    for (const p of P) {
      let d = "";
      for (let i = 0; i < p.length / 2; i++) d += (i ? "L" : "M") + pl.x(p[2 * i]).toFixed(1) + "," + pl.y(p[2 * i + 1]).toFixed(1);
      const e = document.createElementNS("http://www.w3.org/2000/svg", "path");
      e.setAttribute("d", d); e.style.fill = "none"; e.style.stroke = "var(--qb-fg)"; e.style.strokeWidth = "1.5"; e.style.opacity = "0.6";
      g.appendChild(e);
    }
  }, [[]]);
  const ends = pl.scatter([], [], { color: "#000", r: 5, stroke: "#fff" });
  pl.legend(M.KINDS.map((k, i) => ({ label: k, color: KCOL[i], kind: "dot" as const })), "bl");

  const side = htmlEl("div", "ld-side", "", host);
  const grid = htmlEl("div", "ld-grid", "", side);
  const cells: ImgPanel[] = Array.from({ length: 16 }, () => imgPanel(grid, ""));
  cells.forEach(c => c.cap.remove());
  const bp = new Plot(side, { width: 300, height: 200, x: [-0.6, 3.6], y: [0, 1], xticks: [], yticks: [0, 0.5, 1], ylabel: "share", font: F, margin: { b: 1.6 * F + 6, l: 3.4 * F } });
  const bars = bp.bars([-0.17, 0.83, 1.83, 2.83], [0, 0, 0, 0], { barWidth: 0.32, colors: KCOL });
  const trainBars = bp.bars([0.17, 1.17, 2.17, 3.17], [0.25, 0.25, 0.25, 0.25], { barWidth: 0.32, color: "#bbb" });
  void trainBars;
  M.KINDS.forEach((k, i) => bp.text(i, -0.14, k.slice(0, 1).toUpperCase(), { anchor: "middle", size: F - 1 }));

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const srcC = choiceRow(panel, "source", [{ v: "diffusion" as M.Source, label: "latent diffusion" }, { v: "prior" as M.Source, label: "VAE prior N(0, I)" }], st.source, v => { st.source = v; later(); });
  const kindC = choiceRow<string>(panel, "kind", [{ v: "none", label: "any" }, ...M.KINDS.map((k, i) => ({ v: String(i), label: k }))], st.kind === null ? "none" : String(st.kind), v => { st.kind = v === "none" ? null : Number(v); later(); });
  const wS = slider(panel, "guidance w", 0, 10, 0.5, st.w, v => { st.w = v; later(); });
  const stS = slider(panel, "steps", 2, 100, 1, st.steps, v => { st.steps = v; later(); });
  const etaC = choiceRow(panel, "sampler", [{ v: 0, label: "deterministic" }, { v: 1, label: "stochastic" }], st.eta, v => { st.eta = v; later(); });
  const sprC = choiceRow(panel, "code spread", [{ v: "enc", label: "encoder sds" }, { v: "tiny", label: "tiny (memorise)" }], st.tiny ? "tiny" : "enc", v => { st.tiny = v === "tiny"; later(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "new samples", () => { st.seed++; later(); });
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Dots: the latent codes (encoder means) of 800 synthetic training cells. Dashed circle: where 95 % of the VAE prior lies. Black: 16 sample paths and their ends; their decoded images are on the right. Bars: kinds among 200 samples (R round, E elongated, D dividing, P protrusion; kind of the nearest training code), grey = training set.", panel);

  let timer = 0;
  function later(): void { clearTimeout(timer); ro.textContent = "sampling …"; timer = window.setTimeout(draw, 120); }

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    srcC.set(st.source); kindC.set(st.kind === null ? "none" : String(st.kind)); wS.set(st.w); stS.set(st.steps); etaC.set(st.eta); sprC.set(st.tiny ? "tiny" : "enc");
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }

  function draw(): void {
    const r = M.sampleLatents({ source: st.source, kind: st.kind, w: st.w, steps: st.steps, eta: st.eta, sdScale: st.tiny ? 0.1 : 1, n: N, nPaths: NPATHS, seed: st.seed });
    paths.set(r.paths);
    ends.set(Float64Array.from({ length: 16 }, (_, i) => r.Z[2 * i]), Float64Array.from({ length: 16 }, (_, i) => r.Z[2 * i + 1]));
    for (let i = 0; i < 16; i++) cells[i].draw(M.decode([r.Z[2 * i], r.Z[2 * i + 1]]), 28, 28, 0, 1, t => [255 * (1 - t), 255 * (1 - t), 255 * (1 - t)]);
    const nc = M.nearestCode(r.Z), sh = M.kindShares(nc.kind);
    bars.set([-0.17, 0.83, 1.83, 2.83], sh);
    const d = Array.from(nc.dist).sort((a, b) => a - b);
    let far = 0; for (const v of nc.dist) if (v > 0.3) far++;
    ro.textContent =
      `kinds among ${N} samples: ` + sh.map(v => (100 * v).toFixed(0) + " %").join(", ") + `\n` +
      `median distance to the nearest training code: ${fmt(d[N >> 1], 3)}\n` +
      `  (codes among themselves: ${fmt(spacing, 3)})\n` +
      `samples far from every code (> 0.3): ${(100 * far / N).toFixed(1)} %`;
  }

  const spacing = M.codeSpacing();
  apply(p0);
}

autoMount(NAME, mount);
