// What does a denoiser learn? 1D data = a Gaussian mixture (exact answers). One x-axis, three
// rows: (top) the data density and the noised density p_σ; (middle) 400 training pairs
// (noisy x, target) with the exact best denoiser, the binned average of the pairs, and a tiny
// learned network; the target is the clean value, the noise or the score (same curve, other
// units); (bottom) the score ∇ log p_σ: exact and learned (by Tweedie). Slider: noise level σ.
// Learned network: three precomputed snapshots (train_dsm.ts) and "train 500 more steps".
//
// <div data-demo="scorematch1d" data-preset="binned|high|low|under|rare|target"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Adam, type Mlp } from "../lib/nn";
import * as M from "./model";
import { DSM_WEIGHTS } from "./dsm_weights";
import { injectStyle, choiceRow, helperCss, fmt } from "./dview";

const NAME = "scorematch1d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .8em; flex-wrap: nowrap; }
${S} .sm-plots { flex: 1 1 640px; min-width: 0; max-width: 720px; display: flex; flex-direction: column; gap: 0; }
${S} .qb-panel { flex: 0 1 340px; gap: .4em; }
${S} .qb-readout { font-size: 15px; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .sm-plots { flex: 0 0 690px; }
.reveal ${S} .qb-panel { flex: 1 1 400px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
@media (max-width: 760px) { ${S} { flex-wrap: wrap; } }
`;

type Target = "clean" | "noise" | "score";
type Net = "none" | 0 | 1 | 2;
interface State { data: M.DataId; s: number; target: Target; net: Net }
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "binned", label: "the binned average is the denoiser", s: { data: "two", s: 0.6, target: "clean", net: "none" } },
  { id: "high", label: "high noise: towards the mean", s: { data: "two", s: 2.5, target: "clean", net: "none" } },
  { id: "low", label: "low noise: a staircase", s: { data: "two", s: 0.15, target: "clean", net: "none" } },
  { id: "under", label: "under-trained: wrong far out", s: { data: "widenarrow", s: 0.3, target: "clean", net: 0 } },
  { id: "rare", label: "rare mode: the learned score", s: { data: "rare", s: 0.3, target: "score", net: 1 } },
  { id: "target", label: "predict noise or clean: same curve", s: { data: "two", s: 0.6, target: "noise", net: 2 } },
];
const BASE: State = { data: "two", s: 0.6, target: "clean", net: "none" };
const XR: [number, number] = [-4, 4];
const NB = 20;
const sOf = (u: number) => Math.round(M.S_MIN * (M.S_MAX / M.S_MIN) ** (u / 100) * 1000) / 1000;
const uOf = (s: number) => 100 * Math.log(s / M.S_MIN) / Math.log(M.S_MAX / M.S_MIN);

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  // larger labels on slides (the plots are scaled down there)
  const F = host.closest(".reveal") ? 23 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, ...p0.s };
  let d = M.data1d(st.data), exact = M.exactDenoiser(d);
  let live: { net: Mlp; opt: Adam; steps: number } | null = null;

  const plots = htmlEl("div", "sm-plots", "", host);
  const mk = (h: number, ylab: string, last = false) => new Plot(plots, {
    width: 700, height: h, x: XR, y: [0, 1], ylabel: ylab, xticks: last ? 9 : [], yticks: 3, font: F,
    margin: { l: 3.9 * F, r: 10, t: 6, b: last ? 2.4 * F + 10 : 6 }, xlabel: last ? "noisy value x = x₀ + σε" : undefined,
  });
  const pA = mk(120, "density"), pB = mk(210, "target"), pC = mk(170, "score", true);
  const dens0 = pA.band([], [], [], { color: "#999", fillOpacity: 0.35 });
  const densS = pA.line([], [], { color: seriesColor(0), width: 3 });
  pA.legend([{ label: "data p₀", color: "#999", kind: "box" }, { label: "noised p_σ", color: seriesColor(0) }], "tl");
  const sparseB = pB.mark<[number[][]]>((g, rs) => { for (const [a, b] of rs) drawRect(pB, g, a, b); }, [[]]);
  const sparseC = pC.mark<[number[][]]>((g, rs) => { for (const [a, b] of rs) drawRect(pC, g, a, b); }, [[]]);
  const zeroB = pB.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" }), zeroC = pC.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" });
  void zeroB; void zeroC;
  const pts = pB.scatter([], [], { color: "#888", r: 3, opacity: 0.55 });
  const binM = pB.step([], [], { color: seriesColor(2), width: 3.5 }, "mid");
  const exB = pB.line([], [], { color: seriesColor(0), width: 3 });
  const lnB = pB.line([], [], { color: seriesColor(1), width: 3, dash: "7 5" });
  pB.legend([{ label: "pairs", color: "#888", kind: "dot" }, { label: "binned average", color: seriesColor(2) }, { label: "exact", color: seriesColor(0) }, { label: "learned", color: seriesColor(1), dash: "7 5" }], "tl");
  const exC = pC.line([], [], { color: seriesColor(0), width: 3 });
  const lnC = pC.line([], [], { color: seriesColor(1), width: 3, dash: "7 5" });

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const dataC = choiceRow(panel, "data", M.DATA_IDS.map(v => ({ v, label: M.DATA_LABEL[v] })), st.data, v => { st.data = v; live = null; refreshData(); });
  const sS = slider(panel, "noise level σ", 0, 100, 1, uOf(st.s), u => { st.s = sOf(u); draw(); }, u => sOf(u).toFixed(2));
  const tgtC = choiceRow(panel, "target", [{ v: "clean" as Target, label: "clean x₀" }, { v: "noise" as Target, label: "noise ε" }, { v: "score" as Target, label: "score" }], st.target, v => { st.target = v; draw(); });
  const netC = choiceRow<string>(panel, "network", [{ v: "none", label: "none" }, { v: "0", label: "200 steps" }, { v: "1", label: "2 000" }, { v: "2", label: "20 000" }], String(st.net), v => { st.net = v === "none" ? "none" : (Number(v) as Net); live = null; draw(); });
  const trainRow = htmlEl("div", "qb-presets", "", panel);
  button(trainRow, "train 500 more steps", () => trainMore());
  button(trainRow, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Grey dots: training pairs (noisy value, what the network must predict). Shaded: fewer than 3 pairs per bin, where a learned network has little to go on.", panel);

  function drawRect(p: Plot, g: SVGGElement, a: number, b: number): void {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", String(p.x(a))); r.setAttribute("width", String(p.x(b) - p.x(a)));
    r.setAttribute("y", String(p.box.t)); r.setAttribute("height", String(p.box.b - p.box.t));
    r.style.fill = "var(--qb-c4)"; r.style.fillOpacity = "0.12";
    g.appendChild(r);
  }

  function currentNet(): Mlp | null {
    if (live) return live.net;
    if (st.net === "none") return null;
    return M.unpackNet(DSM_WEIGHTS[st.data][st.net]);
  }
  let cached: { key: string; net: Mlp | null } = { key: "", net: null };
  function net(): Mlp | null {
    const key = st.data + ":" + st.net + ":" + (live ? live.steps : 0);
    if (cached.key !== key) cached = { key, net: currentNet() };
    return cached.net;
  }

  function trainMore(): void {
    if (!live) {
      const base = st.net === "none" ? M.newNet() : M.unpackNet(DSM_WEIGHTS[st.data][st.net]);
      const steps = st.net === "none" ? 0 : M.SNAP_STEPS[st.net];
      live = { net: base, opt: new Adam(base, M.LR), steps };
    }
    M.trainDsm(live.net, live.opt, d, live.steps, 500);
    live.steps += 500;
    draw();
  }

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    live = null;
    sS.set(uOf(st.s)); tgtC.set(st.target); netC.set(String(st.net)); dataC.set(st.data);
    refreshData();
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  function refreshData(): void { d = M.data1d(st.data); exact = M.exactDenoiser(d); draw(); }

  function draw(): void {
    const s = st.s, n = net();
    const xs = Array.from({ length: 241 }, (_, i) => XR[0] + (XR[1] - XR[0]) * i / 240);
    const p0v = xs.map(x => M.density(d, 0, x)), psv = xs.map(x => M.density(d, s, x));
    const top = Math.max(...psv, ...p0v.map(v => Math.min(v, 1.2)));
    pA.setDomain(XR, [0, Math.min(1.25, top * 1.08)]);
    dens0.set(xs, xs.map(() => 0), p0v);
    densS.set(xs, psv);
    const P = M.pairs(d, s);
    // target of each pair and of the curves
    const tgt = (x: number, x0: number) => st.target === "clean" ? x0 : st.target === "noise" ? (x - x0) / s : -(x - x0) / (s * s);
    const fromD = (x: number, D: number) => st.target === "clean" ? D : st.target === "noise" ? (x - D) / s : (D - x) / (s * s);
    const T = Array.from(P.x, (x, i) => tgt(x, P.x0[i]));
    const yr: [number, number] = st.target === "clean" ? [-3, 3] : st.target === "noise" ? [-3.5, 3.5] : [-3.5 / s, 3.5 / s];
    pB.setDomain(XR, yr);
    pB.setLabels({ ylabel: st.target === "clean" ? "clean x₀" : st.target === "noise" ? "noise ε" : "−ε/σ" });
    pts.set(P.x, T);
    const bn = M.binned(P.x, T, XR[0], XR[1], NB);
    binM.set(bn.c, bn.m);
    const Dex = xs.map(x => exact(x, s));
    exB.set(xs, xs.map((x, i) => fromD(x, Dex[i])));
    const Dl = n ? xs.map(x => M.learnedDenoiser(n, x, s)) : [];
    lnB.set(n ? xs : [], n ? xs.map((x, i) => fromD(x, Dl[i])) : []);
    const sc = xs.map((x, i) => (Dex[i] - x) / (s * s));
    const smax = Math.max(1, ...sc.map(Math.abs).filter((v, i) => psv[i] > 1e-3));
    pC.setDomain(XR, [-Math.min(smax, 40) * 1.1, Math.min(smax, 40) * 1.1]);
    exC.set(xs, sc);
    lnC.set(n ? xs : [], n ? xs.map((x, i) => (Dl[i] - x) / (s * s)) : []);
    // sparse bins (fewer than 3 pairs)
    const h = (XR[1] - XR[0]) / NB, rs: number[][] = [];
    for (let b = 0; b < NB; b++) if (bn.n[b] < 3) rs.push([XR[0] + b * h, XR[0] + (b + 1) * h]);
    sparseB.set(rs); sparseC.set(rs);
    // errors of the denoisers on the 400 pairs, in clean-value units
    let eEx = 0, eBin = 0, eLn = 0, nb = 0;
    for (let i = 0; i < P.x.length; i++) {
      eEx += (exact(P.x[i], s) - P.x0[i]) ** 2;
      const b = Math.floor((P.x[i] - XR[0]) / h);
      if (b >= 0 && b < NB) { eBin += (bn.m[b] - (st.target === "clean" ? P.x0[i] : T[i])) ** 2; nb++; }
      if (n) eLn += (M.learnedDenoiser(n, P.x[i], s) - P.x0[i]) ** 2;
    }
    const N = P.x.length;
    const snr = 1 / (s * s);
    const netName = live ? `${live.steps} steps (live)` : st.net === "none" ? "none" : `${M.SNAP_STEPS[st.net]} steps`;
    ro.textContent =
      `σ = ${fmt(s, 2)}   SNR = 1/σ² = ${snr >= 10 ? snr.toFixed(0) : fmt(snr, 2)}\n` +
      `denoising error (x̂₀ − x₀)², mean of ${N} pairs:\n` +
      `  exact ${fmt(eEx / N, 3)}` + (n ? `   learned ${fmt(eLn / N, 3)}` : "") + `\n` +
      `network: ${netName}\n` +
      `bins with < 3 pairs: ${rs.length} of ${NB}`;
    void nb; void eBin;
  }

  refreshData();
  pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
}

autoMount(NAME, mount);
