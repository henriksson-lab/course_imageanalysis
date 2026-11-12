// From a spectrum to three numbers. Top: one or two spectra on 380–730 nm (36 bins of
// 10 nm), drawn by dragging. Middle: the observer's three sensitivity curves (eye: L, M, S
// cones; a synthetic camera's R, G, B; a deuteranope: L and S only), with the products
// spectrum × sensitivity shaded for spectrum A. Right: the two colours as the observer sees
// them, and the three numbers. "Make a metamer" adds a seeded random vector from the null
// space of the observer's 3 x 36 matrix: the numbers stay equal while the curves differ.
// Mode "surfaces": the spectra are reflectances lit by a lamp (metameric failure).
// Auto-mounts on <div data-demo="spectrum">. Options: data-preset (white|metamer|camera|
// tungsten|dichromat|yellow), data-mode (lights|surfaces).
import { htmlEl, slider, presets, button, autoMount, rng } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { toCss, linearToSrgb, RGB } from "../lib/colour";
import * as SP from "../lib/spectra";
import { simulateDichromat } from "../colourcloud/colourx";

const NAME = "spectrum";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .sp-plots { flex: 1 1 520px; min-width: 300px; max-width: 640px; display: flex; flex-direction: column; gap: .1em; }
[data-demo="${NAME}"] .sp-plots svg.sp-edit { touch-action: none; cursor: ns-resize; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 420px; gap: .4em; }
[data-demo="${NAME}"] .sp-sw { display: flex; gap: .5em; }
[data-demo="${NAME}"] .sp-sw div { flex: 1; height: 92px; border-radius: 6px; border: 1px solid #999; display: flex; align-items: flex-end; justify-content: center; font-size: 15px; padding-bottom: 4px; box-sizing: border-box; }
[data-demo="${NAME}"] .sp-match { font-size: 1.1em; font-weight: 600; text-align: center; }
[data-demo="${NAME}"] .sp-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
[data-demo="${NAME}"] .sp-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .sp-plots { flex-basis: 560px; max-width: 600px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .sp-sw div { height: 70px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

type Obs = "lms" | "camera" | "deutan";
type Lamp = "d65" | "tungsten" | "fluorescent" | "led";
const OBS_LABEL: Record<Obs, string> = { lms: "eye", camera: "camera", deutan: "deuteranope" };
const CH_NAMES: Record<Obs, string[]> = { lms: ["L", "M", "S"], camera: ["R", "G", "B"], deutan: ["L", "–", "S"] };
const CH_COLS = ["#d62728", "#2ca02c", "#1f77b4"];

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const metamericLED = SP.metamericLED;

/** A smooth reflectance (0..1) from a seed: a sum of a few broad bumps. */
export function smoothReflectance(seed: number, base = 0.3): Float64Array {
  const r = rng(seed), out = new Float64Array(SP.NB).fill(base);
  for (let k = 0; k < 3; k++) {
    const c = 400 + 320 * r.unif(), w = 30 + 50 * r.unif(), a = (r.unif() - 0.3) * 0.5;
    for (let i = 0; i < SP.NB; i++) out[i] += a * Math.exp(-0.5 * ((SP.WL[i] - c) / w) ** 2);
  }
  return out.map(v => Math.min(0.95, Math.max(0.05, v)));
}

function lampSpd(l: Lamp, T: number): Float64Array {
  return l === "d65" ? SP.D65_SPD : l === "tungsten" ? SP.blackbody(T) : l === "fluorescent" ? SP.fluorescent() : metamericLED();
}

autoMount(NAME, host => {
  injectStyle();
  const st = {
    mode: (host.dataset.mode === "surfaces" ? "surfaces" : "lights") as "lights" | "surfaces",
    obs: "lms" as Obs, lamp: "d65" as Lamp, T: 2800, edit: 0 as 0 | 1,
    a: Float64Array.from(SP.D65_SPD) as Float64Array, b: null as Float64Array | null, seed: 1,
  };

  // ---- layout ----
  const plots = htmlEl("div", "sp-plots", "", host);
  const f1 = plotFigure(plots, "", "sp-fig");
  const top = new Plot(f1, { width: 620, height: 250, x: [375, 735], y: [0, 200], ylabel: "power", xticks: [400, 450, 500, 550, 600, 650, 700], margin: { b: 30 } });
  top.svg.classList.add("sp-edit");
  const f2 = plotFigure(plots, "", "sp-fig");
  const bot = new Plot(f2, { width: 620, height: 230, x: [375, 735], y: [0, 1.05], xlabel: "wavelength (nm)", ylabel: "sensitivity", xticks: [400, 450, 500, 550, 600, 650, 700] });
  // wavelength colour strip under the top plot's axis
  top.mark(g => {
    for (let i = 0; i < SP.NB; i++) {
      const x0 = top.x(SP.WL[i] - 5), x1 = top.x(SP.WL[i] + 5);
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", String(x0)); r.setAttribute("width", String(x1 - x0 + 0.5));
      r.setAttribute("y", String(top.box.b - 10)); r.setAttribute("height", "10");
      r.setAttribute("fill", toCss(SP.wavelengthToSrgb(SP.WL[i])));
      g.appendChild(r);
    }
  }, []);
  const lampLine = top.line([], [], { color: "var(--qb-muted)", width: 2, dash: "3 4", label: "lamp" });
  const lineB = top.step([], [], { color: "#ff7f0e", width: 3.5, dash: "7 4", label: "B" }, "mid");
  const lineA = top.step([], [], { color: "#222", width: 3.5, label: "A" }, "mid");
  const bands = [0, 1, 2].map(c => bot.band([], [], [], { color: CH_COLS[c], fill: CH_COLS[c], fillOpacity: 0.25 }));
  const curves = [0, 1, 2].map(c => bot.line([], [], { color: CH_COLS[c], width: 3 }));
  const curveNames = [0, 1, 2].map(c => bot.text(0, 0, "", { color: CH_COLS[c], anchor: "middle", size: 18 }));

  const panel = htmlEl("div", "qb-panel", "", host);
  const sw = htmlEl("div", "sp-sw", "", panel);
  const swA = htmlEl("div", "", "A", sw), swB = htmlEl("div", "", "B", sw);
  const match = htmlEl("div", "sp-match", "", panel);
  const read = htmlEl("div", "qb-readout", "", panel);

  htmlEl("div", "sp-lab", "Observer", panel);
  const obsRow = htmlEl("div", "sp-row", "", panel);
  const obsBtns = (Object.keys(OBS_LABEL) as Obs[]).map(o => { const b = button(obsRow, OBS_LABEL[o], () => { st.obs = o; update(); }); b.dataset.id = o; return b; });
  htmlEl("div", "sp-lab", "The curves are", panel);
  const modeRow = htmlEl("div", "sp-row", "", panel);
  const modeBtns = (["lights", "surfaces"] as const).map(m => { const b = button(modeRow, m === "lights" ? "lights" : "surfaces under a lamp", () => setMode(m)); b.dataset.id = m; return b; });
  const lampRow = htmlEl("div", "sp-row", "", panel);
  const LAMPS: Record<Lamp, string> = { d65: "daylight", tungsten: "tungsten", fluorescent: "fluorescent", led: "white LED" };
  const lampBtns = (Object.keys(LAMPS) as Lamp[]).map(l => { const b = button(lampRow, LAMPS[l], () => { st.lamp = l; update(); }); b.dataset.id = l; return b; });
  const tS = slider(panel, "tungsten T (K)", 2000, 7000, 100, st.T, v => { st.T = v; update(); });
  const actRow = htmlEl("div", "sp-row", "", panel);
  button(actRow, "make a metamer", () => { makeMetamer(); });
  button(actRow, "B = A", () => { st.b = Float64Array.from(st.a); update(); });
  button(actRow, "no B", () => { st.b = null; st.edit = 0; update(); });
  const editRow = htmlEl("div", "sp-row", "", panel);
  htmlEl("span", "sp-lab", "drag to draw: ", editRow);
  const editBtns = [0, 1].map(k => { const b = button(editRow, k ? "B" : "A", () => { if (k && !st.b) st.b = Float64Array.from(st.a); st.edit = k as 0 | 1; update(); }); return b; });
  htmlEl("div", "qb-note", "Numbers: each channel relative to its response to daylight (lights), or to a white surface under the same lamp (surfaces: the eye adapts). Swatches assume an sRGB screen; colours outside its gamut are clipped and marked.", panel);

  const P = {
    white: () => { setMode("lights", false); st.obs = "lms"; st.a = Float64Array.from(SP.D65_SPD); st.b = metamericLED(); },
    metamer: () => { setMode("lights", false); st.obs = "lms"; st.a = Float64Array.from(SP.WL, (l, i) => 60 + 50 * Math.exp(-0.5 * ((l - 600) / 60) ** 2) + 0 * i); st.b = null; st.seed = 3; makeMetamer(false); },
    camera: () => { P.white(); st.obs = "camera"; },
    tungsten: () => {
      setMode("surfaces", false); st.obs = "lms"; st.a = smoothReflectance(5, 0.35); st.T = 2800; tS.set(2800);
      // among seeds 1..30, the D65 metamer that fails most under tungsten (seeded, so fixed)
      let best = -1, bestB: Float64Array | null = null, bestSeed = 1;
      for (let k = 1; k <= 30; k++) {
        st.lamp = "d65"; st.seed = k; makeMetamer(false);
        st.lamp = "tungsten";
        const na = numbers(st.a), nb = numbers(st.b!), d = Math.max(...na.map((v, c) => Math.abs(v - nb[c])));
        if (d > best) { best = d; bestB = st.b; bestSeed = k; }
      }
      st.b = bestB; st.seed = bestSeed + 1;
    },
    dichromat: () => {
      // B - A in the null space of the deuteranope's (L, S) map, chosen along the M cone's
      // sensitivity: the largest change that only M cones could see
      setMode("lights", false); st.obs = "deutan"; st.a = Float64Array.from(SP.WL, l => 70 + 40 * Math.exp(-0.5 * ((l - 520) / 50) ** 2));
      const lms = SP.sensitivity("lms"), m = lms.slice(SP.NB, 2 * SP.NB);
      st.b = SP.makeMetamer(st.a, SP.sensitivity("deutan"), () => 0, 0.92, Infinity, m);
    },
    yellow: () => {
      setMode("lights", false); st.obs = "lms"; st.a = SP.monochromatic(580, 150);
      const comps = [SP.monochromatic(540, 1), SP.monochromatic(640, 1)], w = SP.fitMixture(st.a, comps, SP.sensitivity("lms"));
      st.b = Float64Array.from(comps[0], (v, i) => w[0] * v + w[1] * comps[1][i]);
    },
  };
  const presetList: [keyof typeof P, string][] = [["white", "daylight vs LED: same white"], ["metamer", "make a metamer"], ["camera", "the camera disagrees"], ["tungsten", "metameric failure under tungsten"], ["dichromat", "a dichromat sees more matches"], ["yellow", "580 nm vs red + green"]];
  const pbtns = presets(plots, [
    ...presetList.map(([id, label]) => ({ label, apply: () => { P[id](); st.edit = 0; update(); } })),
    { label: "reset", apply: () => { setMode("lights", false); st.obs = "lms"; st.a = Float64Array.from(SP.D65_SPD); st.b = null; st.edit = 0; st.lamp = "d65"; update(); } },
  ]);

  function setMode(m: "lights" | "surfaces", upd = true): void {
    if (m !== st.mode) {
      st.mode = m;
      if (m === "surfaces") { st.a = smoothReflectance(5, 0.35); st.b = null; }
      else { st.a = Float64Array.from(SP.D65_SPD); st.b = null; }
    }
    top.setDomain(undefined, m === "surfaces" ? [0, 1.05] : [0, 200]);
    top.setLabels({ ylabel: m === "surfaces" ? "reflectance" : "power" });
    if (upd) update();
  }

  /** The observer's matrix for the current mode (surfaces: times the lamp). */
  function effMatrix(): Float64Array {
    const A = SP.sensitivity(st.obs);
    if (st.mode === "lights") return A;
    const E = lampSpd(st.lamp, st.T), out = new Float64Array(A.length);
    for (let c = 0; c < 3; c++) for (let i = 0; i < SP.NB; i++) out[c * SP.NB + i] = A[c * SP.NB + i] * E[i];
    return out;
  }

  function makeMetamer(upd = true): void {
    const r = rng(1000 + st.seed++);
    st.b = SP.makeMetamer(st.a, effMatrix(), r.unif, 0.92, st.mode === "surfaces" ? 1 : Infinity);
    st.edit = 0;
    if (upd) update();
  }

  /** Light reaching the eye for spectrum s. */
  const lightOf = (s: Float64Array): Float64Array => {
    if (st.mode === "lights") return s;
    const E = lampSpd(st.lamp, st.T);
    return Float64Array.from(s, (v, i) => v * E[i]);
  };

  /** The three numbers, relative to D65 (lights) or to a white surface under the lamp. */
  function numbers(s: Float64Array): number[] {
    const A = SP.sensitivity(st.obs), L = lightOf(s);
    const ref = st.mode === "lights" ? SP.D65_SPD : lampSpd(st.lamp, st.T);
    const v = SP.apply(A, L), w = SP.apply(A, ref), d = SP.apply(A, SP.D65_SPD);
    // lights: relative to D65 at the same power; surfaces: relative to a white surface
    return [0, 1, 2].map(c => (st.mode === "lights" ? (d[c] ? v[c] / d[c] : 0) : (w[c] ? v[c] / w[c] : 0)));
  }

  /** Display colour of what the observer sees. The eye is adapted to the reference white
   *  (D65 for lights, the lamp for surfaces), so its colour is rebuilt from the three
   *  relative numbers: LMS = numbers ⊙ LMS(D65 white) -> XYZ -> sRGB. The deuteranope's
   *  view is the trichromat colour through the rank-2 dichromat projection; the camera's
   *  relative R, G, B are shown directly as linear sRGB. */
  function swatch(s: Float64Array, exposure: number): { css: string; out: boolean; text: string } {
    let lin: RGB, out = false;
    if (st.obs === "camera") {
      const n = numbers(s).map(v => v * exposure);
      const mx = Math.max(...n);
      out = mx > 1;
      lin = (mx > 1 ? n.map(v => v / mx) : n) as RGB;
    } else {
      const obs = st.obs;
      st.obs = "lms"; const n = numbers(s); st.obs = obs;
      const lmsW = SP.mat3v(SP.M_XYZ_TO_LMS, SP.WHITE_XYZ);
      const xyz = SP.mat3v(SP.inv3(SP.M_XYZ_TO_LMS), [n[0] * lmsW[0] * exposure, n[1] * lmsW[1] * exposure, n[2] * lmsW[2] * exposure]);
      const dc = SP.displayColour(xyz);
      lin = dc.lin; out = dc.outOfGamut;
    }
    let e: RGB = lin.map(v => linearToSrgb(Math.min(1, Math.max(0, v)))) as RGB;
    if (st.obs === "deutan") e = simulateDichromat(e, "deutan");
    return { css: toCss(e), out, text: e.reduce((s, v) => s + v, 0) > 1.6 ? "#111" : "#fff" };
  }

  // drag to draw on the top plot
  let dragging = false;
  const drawAt = (ev: PointerEvent) => {
    const [l, v] = top.toData(ev);
    const s = st.edit === 1 && st.b ? st.b : st.a;
    const hi = st.mode === "surfaces" ? 1 : top.y.domain[1];
    for (let i = 0; i < SP.NB; i++) {
      const w = Math.exp(-0.5 * ((SP.WL[i] - l) / 9) ** 2);
      s[i] = Math.min(hi, Math.max(0, s[i] + w * (Math.min(hi, Math.max(0, v)) - s[i])));
    }
    update();
  };
  top.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; top.svg.setPointerCapture(ev.pointerId); drawAt(ev); });
  top.svg.addEventListener("pointermove", ev => { if (dragging) drawAt(ev); });
  top.svg.addEventListener("pointerup", () => { dragging = false; });

  function fitDomain(): void {
    if (st.mode !== "lights") return;
    const m = Math.max(...st.a, ...(st.b ?? [0]));
    top.setDomain(undefined, [0, Math.max(200, 50 * Math.ceil(1.08 * m / 50))]);
  }

  function update(): void {
    if (!dragging) fitDomain();
    obsBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.obs));
    modeBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.mode));
    lampBtns.forEach(b => { b.classList.toggle("active", b.dataset.id === st.lamp); b.disabled = st.mode !== "surfaces"; b.style.opacity = st.mode === "surfaces" ? "1" : ".45"; });
    tS.input.disabled = st.mode !== "surfaces" || st.lamp !== "tungsten";
    editBtns.forEach((b, k) => b.classList.toggle("active", st.edit === k));
    const wl = Array.from(SP.WL);
    lineA.set(wl, Array.from(st.a));
    lineB.set(st.b ? wl : [], st.b ? Array.from(st.b) : []);
    if (st.mode === "surfaces") {
      const E = lampSpd(st.lamp, st.T), m = Math.max(...E);
      lampLine.set(wl, Array.from(E, v => v / m));
    } else lampLine.set([], []);
    top.legend([{ label: "A", color: "#222" }, ...(st.b ? [{ label: "B", color: "#ff7f0e", dash: "7 4" }] : []), ...(st.mode === "surfaces" ? [{ label: "lamp", color: "var(--qb-muted)", dash: "3 4" }] : [])], "tr");
    // sensitivities (normalised to peak 1) and products with A's light
    const A = SP.sensitivity(st.obs), La = lightOf(st.a);
    const pk = [0, 1, 2].map(c => { let m = 0; for (let i = 0; i < SP.NB; i++) m = Math.max(m, A[c * SP.NB + i]); return m || 1; });
    let maxProd = 0;
    for (let c = 0; c < 3; c++) for (let i = 0; i < SP.NB; i++) maxProd = Math.max(maxProd, A[c * SP.NB + i] / pk[c] * La[i]);
    for (let c = 0; c < 3; c++) {
      const ys = wl.map((_, i) => A[c * SP.NB + i] / pk[c]);
      if (st.obs === "deutan" && c === 1) { curves[c].set([], []); bands[c].set([], [], []); curveNames[c].set(0, 0, ""); continue; }
      curves[c].set(wl, ys);
      bands[c].set(wl, wl.map(() => 0), wl.map((_, i) => 0.95 * A[c * SP.NB + i] / pk[c] * La[i] / (maxProd || 1)));
      const ipk = ys.indexOf(Math.max(...ys));
      curveNames[c].set(wl[ipk] + (c === 0 ? 22 : c === 1 ? -18 : -12), 1.0, CH_NAMES[st.obs][c]);
    }
    bot.setLabels({ title: undefined });
    // swatches and numbers
    const na = numbers(st.a), nb = st.b ? numbers(st.b) : null;
    let exposure = 1;
    if (st.mode === "lights") {
      const ya = SP.spectrumToXyz(st.a)[1], yb = st.b ? SP.spectrumToXyz(st.b)[1] : 0;
      exposure = 0.8 / Math.max(ya, yb, 1e-6);
      if (st.obs === "camera") { const m = Math.max(...na, ...(nb ?? [0])); exposure = 0.8 / Math.max(m, 1e-6); }
    }
    const sa = swatch(st.a, exposure);
    swA.style.background = sa.css; swA.style.color = sa.text; swA.textContent = "A" + (sa.out ? " (clipped)" : "");
    if (st.b) {
      const sb = swatch(st.b, exposure);
      swB.style.background = sb.css; swB.style.color = sb.text; swB.textContent = "B" + (sb.out ? " (clipped)" : ""); swB.style.visibility = "visible";
    } else swB.style.visibility = "hidden";
    const nm = CH_NAMES[st.obs], used = st.obs === "deutan" ? [0, 2] : [0, 1, 2];
    const f = (v: number[]) => used.map(c => v[c].toFixed(3).padStart(7)).join("");
    const lines = ["     " + used.map(c => nm[c].padStart(7)).join(""), "A    " + f(na)];
    if (nb) {
      lines.push("B    " + f(nb));
      const big = Math.max(...used.map(c => Math.max(Math.abs(na[c]), Math.abs(nb[c]))), 1e-9);
      const rel = Math.max(...used.map(c => Math.abs(na[c] - nb[c]))) / big;
      match.textContent = rel < 0.005 ? `A and B match for the ${OBS_LABEL[st.obs]}` : `A and B differ (by up to ${(100 * rel).toFixed(1)} % of the largest number)`;
      match.style.color = rel < 0.005 ? "var(--qb-c3)" : "var(--qb-c2)";
    } else match.textContent = "";
    const xa = SP.xyzToXy(SP.spectrumToXyz(lightOf(st.a)));
    lines.push(`A: x = ${xa[0].toFixed(3)}, y = ${xa[1].toFixed(3)}`);
    if (st.b) { const xb = SP.xyzToXy(SP.spectrumToXyz(lightOf(st.b))); lines.push(`B: x = ${xb[0].toFixed(3)}, y = ${xb[1].toFixed(3)}`); }
    read.textContent = lines.join("\n");
  }

  setMode(st.mode, false);
  const pre = host.dataset.preset as keyof typeof P | undefined;
  if (pre && P[pre]) { P[pre](); pbtns[presetList.findIndex(p => p[0] === pre)]?.classList.add("active"); }
  update();
});
