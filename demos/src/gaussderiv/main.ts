// Smoothing before differentiating, in 1D. A profile of 256 samples (a step, a ramp, a thin
// line, two close steps; or a line through the micrograph) with seeded noise. Rows: the
// signal and its Gaussian blur; the finite difference of the raw signal; the derivative-of-
// Gaussian response; the kernel itself, to scale, at a chosen position (click any row).
// Order 0 / 1 / 2: smoothing, slope, curvature. Readout: kernel taps and the cost of a 2D
// filter applied directly vs separably.
//
// <div data-demo="gaussderiv" data-preset="noise|clean|merge|line|curv|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { gaussianKernel } from "../lib/imageops";
import { gaussian1d } from "../lib/edges";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { makeProfile, N } from "./profile";
import { injectStyle } from "../canny/imgview";

const NAME = "gaussderiv";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .gd-plots { flex: 1 1 640px; min-width: 0; max-width: 760px; display: flex; flex-direction: column; gap: 2px; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .gd-plots { flex: 0 0 720px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .25em .5em; }
@media (max-width: 760px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

type Profile = "mixed" | "step" | "line" | "close" | "cell";
interface State { profile: Profile; sigma: number; noise: number; order: 0 | 1 | 2; pos: number; same: boolean }
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "noise", label: "finite difference on noise", s: { profile: "mixed", sigma: 0.5, noise: 8, order: 1, pos: 48 } },
  { id: "clean", label: "σ = 2 cleans it", s: { profile: "mixed", sigma: 2, noise: 8, order: 1, pos: 48 } },
  { id: "merge", label: "σ = 8: two close steps merge", s: { profile: "mixed", sigma: 8, noise: 8, order: 1, pos: 203 } },
  { id: "line", label: "a thin line gives two lobes", s: { profile: "mixed", sigma: 2, noise: 4, order: 1, pos: 160 } },
  { id: "curv", label: "curvature: zero crossing at the edge", s: { profile: "step", sigma: 3, noise: 3, order: 2, pos: 128 } },
  { id: "cell", label: "a line through the micrograph", s: { profile: "cell", sigma: 2, noise: 0, order: 1, pos: 128 } },
];
const DEFAULT: State = { profile: "mixed", sigma: 2, noise: 8, order: 1, pos: 48, same: true };
const SIG_MIN = 0.5, SIG_MAX = 16;
const sigOf = (u: number) => Math.round(SIG_MIN * Math.pow(SIG_MAX / SIG_MIN, u / 100) * 100) / 100;
const uOf = (s: number) => 100 * Math.log(s / SIG_MIN) / Math.log(SIG_MAX / SIG_MIN);
const ORDER_NAMES = ["smoothing", "slope", "curvature"];
const FD_NAMES = ["raw signal (no smoothing)", "finite difference ½[−1 0 1]", "finite difference [1 −2 1]"];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };
  let cellRow: Float64Array | null = null;

  const plots = htmlEl("div", "gd-plots", "", host);
  const mk = (h: number, last = false) => new Plot(plots, {
    width: 720, height: h, x: [0, N - 1], y: [0, 1], xticks: last ? 8 : [], font: 18,
    margin: { l: 62, r: 10, t: 6, b: last ? 52 : 6 }, xlabel: last ? "position x (pixels)" : undefined,
  });
  const pA = mk(132), pB = mk(118), pC = mk(118), pD = mk(150, true);
  const raw = pA.line([], [], { color: "#999", width: 1.5 });
  const smooth = pA.line([], [], { color: seriesColor(0), width: 3 });
  const fd = pB.line([], [], { color: "#888", width: 1.5 });
  const dg = pC.line([], [], { color: seriesColor(1), width: 3 });
  const kern = pD.mark<[Float64Array, number]>((g, k, c) => {
    const r = k.length >> 1;
    for (let i = -r; i <= r; i++) {
      const x = pD.x(c + i), y0 = pD.y(0), y1 = pD.y(k[i + r]);
      const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
      l.setAttribute("x1", String(x)); l.setAttribute("x2", String(x)); l.setAttribute("y1", String(y0)); l.setAttribute("y2", String(y1));
      l.style.stroke = seriesColor(2); l.style.strokeWidth = "3";
      g.appendChild(l);
    }
  }, [new Float64Array([1]), 128]);
  const zeroB = pB.hline(0, { color: "var(--qb-axis)", width: 1 }), zeroC = pC.hline(0, { color: "var(--qb-axis)", width: 1 }), zeroD = pD.hline(0, { color: "var(--qb-axis)", width: 1 });
  void zeroB; void zeroC; void zeroD;
  const vl = [pA, pB, pC].map(p => p.vline(128, { color: "var(--qb-c4)", width: 2, dash: "5 4" }));
  const tA = pA.text(4, 0, "signal f (grey) and blurred G_σ ∗ f (blue)", { anchor: "start", size: 17 });
  const tB = pB.text(4, 0, "", { anchor: "start", size: 17 });
  const tC = pC.text(4, 0, "", { anchor: "start", size: 17 });
  const tD = pD.text(4, 0, "", { anchor: "start", size: 17 });
  for (const p of [pA, pB, pC, pD]) p.svg.addEventListener("pointerdown", ev => { st.pos = Math.max(0, Math.min(N - 1, Math.round(p.toData(ev)[0]))); draw(); });
  for (const p of [pA, pB, pC, pD]) p.svg.style.cursor = "crosshair";

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); compute(); } })));
  htmlEl("div", "ev-lab", "profile", panel);
  const profIds: Profile[] = ["mixed", "step", "line", "close", "cell"];
  const profBtns = presets(panel, ["mixed", "step", "line", "two steps", "micrograph"].map((l, i) => ({ label: l, apply: () => { st.profile = profIds[i]; compute(); } })));
  const sigS = slider(panel, "σ", 0, 100, 0.5, uOf(st.sigma), u => { st.sigma = sigOf(u); compute(); }, u => sigOf(u).toFixed(2));
  const noiseS = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; compute(); });
  htmlEl("div", "ev-lab", "derivative order", panel);
  const ordBtns = presets(panel, [0, 1, 2].map(o => ({ label: `${o}: ${ORDER_NAMES[o]}`, apply: () => { st.order = o as 0 | 1 | 2; compute(); } })));
  const sameC = checkbox(panel, "rows 2 and 3 on the same scale", st.same, v => { st.same = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); presetBtns.forEach(b => b.classList.remove("active")); sync(); compute(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, panel);

  function sync(): void {
    sigS.set(uOf(st.sigma)); noiseS.set(st.noise); sameC.checked = st.same;
  }

  let f: Float64Array = new Float64Array(N), sm: Float64Array = f, d1: Float64Array = f, d2: Float64Array = f;
  function compute(): void {
    if (st.profile === "cell" && !cellRow) {
      loadCell(256).then(img => { cellRow = Float64Array.from(img.data.subarray(150 * 256, 151 * 256)); compute(); });
      return;
    }
    if (st.profile === "cell") {
      f = cellRow!.slice();
      if (st.noise > 0) { const n = makeProfile("step", st.noise, 1), c = makeProfile("step", 0, 1); for (let i = 0; i < N; i++) f[i] += n[i] - c[i]; }
    } else f = makeProfile(st.profile, st.noise, 1);
    sm = gaussian1d(f, st.sigma, 0);
    d1 = st.order === 0 ? f : gaussian1d(f, 0, st.order);
    d2 = gaussian1d(f, st.sigma, st.order);
    draw();
  }

  const xs = Float64Array.from({ length: N }, (_, i) => i);
  function draw(): void {
    profBtns.forEach((b, i) => b.classList.toggle("active", profIds[i] === st.profile));
    ordBtns.forEach((b, i) => b.classList.toggle("active", i === st.order));
    credit.style.display = st.profile === "cell" ? "" : "none";
    let lo = Infinity, hi = -Infinity;
    for (const v of f) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = 0.08 * (hi - lo || 1);
    pA.setDomain(undefined, [lo - pad, hi + 3.5 * pad]);
    raw.set(xs, f); smooth.set(xs, sm);
    const amax = (a: Float64Array) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m || 1; };
    if (st.order === 0) {
      pB.setDomain(undefined, [lo - pad, hi + 3.5 * pad]); pC.setDomain(undefined, [lo - pad, hi + 3.5 * pad]);
    } else {
      const mb = amax(d1), mc = amax(d2), m = st.same ? Math.max(mb, mc) : 0;
      pB.setDomain(undefined, [-(m || mb) * 1.1, (m || mb) * 1.45]);
      pC.setDomain(undefined, [-(m || mc) * 1.1, (m || mc) * 1.45]);
    }
    fd.set(xs, d1); dg.set(xs, d2);
    const k = gaussianKernel(st.sigma, st.order), km = amax(k);
    pD.setDomain(undefined, [st.order === 0 ? -0.15 * km : -1.15 * km, 1.6 * km]);
    kern.set(k, st.pos);
    for (const v of vl) v.set(st.pos);
    const top = (p: Plot) => p.y.domain[1] - 0.12 * (p.y.domain[1] - p.y.domain[0]);
    tA.set(4, top(pA), "signal f (grey) and blurred G_σ ∗ f (blue)");
    tB.set(4, top(pB), FD_NAMES[st.order]);
    tC.set(4, top(pC), st.order === 0 ? "Gaussian blur G_σ ∗ f" : `derivative of Gaussian, order ${st.order} (${ORDER_NAMES[st.order]})`);
    tD.set(4, top(pD), `the kernel at x = ${st.pos}, to scale`);
    const taps = k.length, dir = taps * taps, sep = 2 * taps;
    const p = st.pos, fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
    ro.textContent =
      `kernel: ${taps} taps (cut at ±${taps >> 1} = ⌈4σ⌉)\n` +
      `2D filter per pixel: ${dir} direct, ${sep} separable\n` +
      `at x = ${p}: f = ${fmt(f[p])}, blurred = ${fmt(sm[p])}\n` +
      (st.order ? `  finite diff = ${fmt(d1[p])}, Gaussian = ${fmt(d2[p])}` : "");
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  compute();
}

autoMount(NAME, mount);
