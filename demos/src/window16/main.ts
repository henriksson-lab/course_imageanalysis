// Showing a 12/16-bit image on an 8-bit screen. A synthetic 12-bit fluorescence image
// (stored in uint16, LSB- or MSB-aligned) or a synthetic CT slice in Hounsfield units
// (int16, or uint16 stored as HU + 1024). A linear window [min, max] maps raw values to
// the 256 display levels; the histogram (log counts) shows where the data are and what the
// window cuts off. For the fluorescence image, a background can be subtracted with uint16
// wrap-around, uint16 saturation, or in float. LUT: grey or hi-lo (blue = at or below the
// window, red = at or above). Click the image for the numbers of one pixel.
// Auto-mounts on <div data-demo="window16">. Optional: data-image="fluo|ct",
// data-preset (an id from the preset lists below, e.g. "full", "w12", "lung").
import { htmlEl, slider, presets, button, autoMount, svgEl, Slider } from "../lib/ui";
import { N, fluorescence, ctPhantom, subtractU16, windowLevel, percentile, Arith } from "./data";

const NAME = "window16";
const BINS = 256;

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .9em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .w16-img { flex: 0 1 290px; min-width: 200px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; text-align: center; line-height: 1.2; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; cursor: crosshair; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .w16-mid { flex: 1 1 380px; min-width: 260px; max-width: 480px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .w16-lab { font-size: 16px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-slider { grid-template-columns: 6.5em 1fr 4.6em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .off { opacity: .35; pointer-events: none; }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 19px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .w16-lab { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] svg text { font-size: 17px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type ImgId = "fluo" | "ct";
type Lut = "grey" | "hilo";
interface State {
  image: ImgId; container: "a" | "b";   // fluo: a = LSB, b = MSB; ct: a = int16 HU, b = uint16 HU + 1024
  arith: Arith; bg: number; lo: number; hi: number; lut: Lut; histFull: boolean; sel: [number, number];
}
interface Preset { id: string; label: string; apply: (s: State, d: Float64Array) => void; }

const win = (L: number, W: number) => (s: State) => { s.lo = L - W / 2; s.hi = L + W / 2; };
const PRESETS: Record<ImgId, Preset[]> = {
  fluo: [
    { id: "full", label: "0–65535", apply: s => { s.lo = 0; s.hi = 65535; } },
    { id: "w12", label: "0–4095", apply: s => { s.lo = 0; s.hi = 4095 * (s.container === "b" ? 16 : 1); } },
    { id: "minmax", label: "min–max", apply: (s, d) => { [s.lo, s.hi] = extent(d); } },
    { id: "pct", label: "0.5–99.5 %", apply: (s, d) => { s.lo = percentile(d, 0.5); s.hi = percentile(d, 99.5); } },
  ],
  ct: [
    { id: "ctfull", label: "full −1024…3071", apply: s => { s.lo = -1024; s.hi = 3071; } },
    { id: "soft", label: "soft tissue W400 L40", apply: win(40, 400) },
    { id: "lung", label: "lung W1500 L−600", apply: win(-600, 1500) },
    { id: "bone", label: "bone W1800 L400", apply: win(400, 1800) },
    { id: "narrow", label: "narrow W150 L70", apply: win(70, 150) },
  ],
};

function mount(host: HTMLElement): void {
  injectStyle();
  const imgCol = htmlEl("div", "w16-img", "", host);
  const fig = htmlEl("figure", "", "", imgCol);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  cv.width = N; cv.height = N;
  const ro = htmlEl("div", "qb-readout", "", imgCol);

  const mid = htmlEl("div", "w16-mid", "", host);
  const W = 470, H = 250, PL = 46, PR = 14, PT = 10, PB = 58;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, mid);
  const panel = htmlEl("div", "qb-panel", "", host);

  const data = { fluo: fluorescence(), ct: ctPhantom() };
  const img0: ImgId = host.dataset.image === "ct" ? "ct" : "fluo";
  const st: State = { image: img0, container: "a", arith: "none", bg: 160, lo: 0, hi: 4095, lut: "grey", histFull: true, sel: [128, 128] };

  // ---- the values shown, after container and arithmetic ----
  function values(): { v: Float64Array; amin: number; amax: number; typeName: string } {
    if (st.image === "ct") {
      const off = st.container === "b" ? 1024 : 0;
      return { v: Float64Array.from(data.ct, x => x + off), amin: -1024 + off, amax: 3071 + off,
        typeName: st.container === "b" ? "uint16, stored = HU + 1024" : "int16, Hounsfield units" };
    }
    const f = st.container === "b" ? 16 : 1;
    const v = Float64Array.from(data.fluo, x => subtractU16(x * f, st.bg * f, st.arith));
    if (st.arith === "float") {
      let lo = Infinity, hi = -Infinity;
      for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
      return { v, amin: Math.min(-st.bg * f, lo), amax: 65535, typeName: "float32" };
    }
    return { v, amin: 0, amax: 65535, typeName: "uint16" + (st.container === "b" ? ", 12 bits MSB-aligned (× 16)" : ", 12 bits LSB-aligned") };
  }

  // ---- panel ----
  htmlEl("div", "w16-lab", "image", panel);
  const imgBtns = presets(panel, [
    { label: "fluorescence 12 bit", apply: () => { st.image = "fluo"; st.container = "a"; st.arith = "none"; applyPreset("w12"); } },
    { label: "CT slice (HU)", apply: () => { st.image = "ct"; st.container = "a"; st.arith = "none"; applyPreset("soft"); } },
  ]);
  htmlEl("div", "w16-lab", "stored as", panel);
  const contBtns = presets(panel, [
    { label: "", apply: () => { st.container = "a"; fixPresetRanges(); schedule(); } },
    { label: "", apply: () => { st.container = "b"; fixPresetRanges(); schedule(); } },
  ]);
  htmlEl("div", "w16-lab", "window presets", panel);
  const presetBox = htmlEl("div", "", "", panel);
  let presetBtns: HTMLButtonElement[] = [];
  let presetFor: ImgId | null = null;
  const arithBox = htmlEl("div", "", "", panel);
  arithBox.style.display = "flex"; arithBox.style.flexDirection = "column"; arithBox.style.gap = ".3em";
  htmlEl("div", "w16-lab", "subtract the background", arithBox);
  const arithItems: { label: string; id: Arith }[] = [
    { label: "no", id: "none" }, { label: "uint16 wrap", id: "wrap" }, { label: "uint16 saturate", id: "saturate" }, { label: "float", id: "float" },
  ];
  const arithBtns = presets(arithBox, arithItems.map(a => ({ label: a.label, apply: () => { st.arith = a.id; schedule(); } })));
  const bgS = slider(arithBox, "background", 0, 400, 10, st.bg, v => { st.bg = v; schedule(); });
  htmlEl("div", "w16-lab", "colour map (LUT)", panel);
  const lutBtns = presets(panel, [
    { label: "grey", apply: () => { st.lut = "grey"; schedule(); } },
    { label: "hi-lo", apply: () => { st.lut = "hilo"; schedule(); } },
  ]);
  button(htmlEl("div", "", "", panel), "reset", () => { reset(); });

  // window sliders (signed quadratic scale: fine steps near 0) and histogram axis
  let M = 65535, pmin = 0;
  const toV = (p: number) => Math.round(Math.sign(p) * p * p * M);
  const toP = (v: number) => Math.sign(v) * Math.sqrt(Math.min(1, Math.abs(v) / M));
  const loS: Slider = slider(mid, "window min", 0, 1, 0.001, 0, p => { st.lo = toV(p); if (st.lo >= st.hi) st.hi = st.lo + 1; markCustom(); schedule(); }, p => String(toV(p)));
  const hiS: Slider = slider(mid, "window max", 0, 1, 0.001, 1, p => { st.hi = toV(p); if (st.hi <= st.lo) st.lo = st.hi - 1; markCustom(); schedule(); }, p => String(toV(p)));
  const axisRow = htmlEl("div", "", "", mid);
  axisRow.style.display = "flex"; axisRow.style.gap = ".5em"; axisRow.style.alignItems = "center";
  htmlEl("span", "w16-lab", "histogram axis:", axisRow);
  const axisBtns = presets(axisRow, [
    { label: "type range", apply: () => { st.histFull = true; schedule(); } },
    { label: "data range", apply: () => { st.histFull = false; schedule(); } },
  ]);
  const note = htmlEl("div", "qb-note", "", mid);

  function markCustom(): void { presetBtns.forEach(b => b.classList.remove("active")); }

  function buildPresets(): void {
    if (presetFor === st.image) return;
    presetFor = st.image;
    presetBox.innerHTML = "";
    const items = PRESETS[st.image].map(p => ({ label: p.label, apply: () => applyPreset(p.id) }));
    if (st.image === "fluo") {
      items.push({ label: "− bg: wrap", apply: () => { st.arith = "wrap"; applyPreset("w12"); setActive("− bg: wrap"); } });
      items.push({ label: "− bg: saturate", apply: () => { st.arith = "saturate"; applyPreset("w12"); setActive("− bg: saturate"); } });
    }
    presetBtns = presets(presetBox, items);
  }
  function setActive(label: string): void { presetBtns.forEach(b => b.classList.toggle("active", b.textContent === label)); }

  /** MSB/LSB switch: keep a 0–4095 window meaning "the 12-bit range". */
  function fixPresetRanges(): void {
    const act = presetBtns.find(b => b.classList.contains("active"))?.textContent;
    const p = PRESETS[st.image].find(q => q.label === act);
    if (p) p.apply(st, values().v);
  }

  function applyPreset(id: string): void {
    buildPresets();
    const p = PRESETS[st.image].find(q => q.id === id) ?? PRESETS[st.image][0];
    p.apply(st, values().v);
    setActive(p.label);
    schedule();
  }

  function reset(): void {
    st.image = img0; st.container = "a"; st.arith = "none"; st.bg = 160; st.lut = "grey"; st.histFull = true; st.sel = [128, 128];
    bgS.set(st.bg);
    presetFor = null;
    applyPreset(host.dataset.preset ?? (img0 === "ct" ? "soft" : "w12"));
  }

  cv.addEventListener("pointerdown", ev => {
    const r = cv.getBoundingClientRect();
    const c = Math.floor((ev.clientX - r.left) / r.width * N), rr = Math.floor((ev.clientY - r.top) / r.height * N);
    st.sel = [Math.max(0, Math.min(N - 1, rr)), Math.max(0, Math.min(N - 1, c))];
    schedule();
  });

  let pending = false;
  function schedule(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }

  function lutColour(g: number, v: number): [number, number, number] {
    if (st.lut === "hilo") {
      if (v <= st.lo) return [0, 60, 255];
      if (v >= st.hi) return [255, 0, 0];
    }
    return [g, g, g];
  }

  function draw(): void {
    buildPresets();
    const { v, amin, amax, typeName } = values();
    const isFluo = st.image === "fluo";
    // slider scale
    M = Math.max(Math.abs(amin), Math.abs(amax));
    pmin = amin < 0 ? -Math.sqrt(-amin / M) : 0;
    for (const s of [loS, hiS]) s.input.min = String(Math.floor(pmin * 1000) / 1000);
    loS.set(toP(st.lo)); hiS.set(toP(st.hi));
    (loS.input.nextElementSibling as HTMLOutputElement).value = String(st.lo);
    (hiS.input.nextElementSibling as HTMLOutputElement).value = String(st.hi);

    // image
    const ctx = cv.getContext("2d")!;
    const id = ctx.createImageData(N, N);
    let below = 0, above = 0;
    for (let k = 0; k < N * N; k++) {
      const g = windowLevel(v[k], st.lo, st.hi);
      if (v[k] <= st.lo) below++;
      if (v[k] >= st.hi) above++;
      const [r, gg, b] = lutColour(g, v[k]);
      id.data[4 * k] = r; id.data[4 * k + 1] = gg; id.data[4 * k + 2] = b; id.data[4 * k + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const [sr, sc] = st.sel;
    ctx.lineWidth = 2; ctx.strokeStyle = "#ffd400";
    ctx.beginPath(); ctx.arc(sc + 0.5, sr + 0.5, 6, 0, 2 * Math.PI); ctx.stroke();
    cap.textContent = isFluo ? "fluorescence, 12 bits" : "CT slice (synthetic)";

    // buttons
    imgBtns.forEach((b, i) => b.classList.toggle("active", i === (isFluo ? 0 : 1)));
    contBtns[0].textContent = isFluo ? "LSB-aligned" : "int16 HU";
    contBtns[1].textContent = isFluo ? "MSB-aligned (× 16)" : "uint16 HU+1024";
    contBtns.forEach((b, i) => b.classList.toggle("active", i === (st.container === "a" ? 0 : 1)));
    arithBox.classList.toggle("off", !isFluo);
    arithBtns.forEach((b, i) => b.classList.toggle("active", arithItems[i].id === st.arith));
    bgS.input.parentElement!.classList.toggle("off", st.arith === "none");
    lutBtns.forEach((b, i) => b.classList.toggle("active", i === (st.lut === "grey" ? 0 : 1)));
    axisBtns.forEach((b, i) => b.classList.toggle("active", i === (st.histFull ? 0 : 1)));

    // histogram
    let dmin = Infinity, dmax = -Infinity;
    for (const x of v) { if (x < dmin) dmin = x; if (x > dmax) dmax = x; }
    const h0 = st.histFull ? amin : dmin, h1 = st.histFull ? amax + 1 : dmax + 1;
    const counts = new Array<number>(BINS).fill(0);
    for (const x of v) counts[Math.min(BINS - 1, Math.max(0, Math.floor((x - h0) / (h1 - h0) * BINS)))]++;
    const cmax = Math.log10(Math.max(...counts) + 1);
    svg.innerHTML = "";
    const X = (x: number) => PL + (x - h0) / (h1 - h0) * (W - PL - PR);
    const Y = (c: number) => H - PB - Math.log10(c + 1) / cmax * (H - PT - PB);
    const wl = Math.max(PL, Math.min(W - PR, X(st.lo))), wr = Math.max(PL, Math.min(W - PR, X(st.hi)));
    svgEl("rect", { x: wl, y: PT, width: Math.max(1, wr - wl), height: H - PT - PB, fill: "var(--qb-c4)", opacity: 0.18 }, svg);
    counts.forEach((c, i) => {
      if (!c) return;
      const x0 = PL + i / BINS * (W - PL - PR), x1 = PL + (i + 1) / BINS * (W - PL - PR);
      svgEl("rect", { x: x0, y: Y(c), width: Math.max(0.8, x1 - x0), height: H - PB - Y(c), fill: "var(--qb-c1)" }, svg);
    });
    for (const x of [st.lo, st.hi]) if (x >= h0 && x <= h1)
      svgEl("line", { x1: X(x), x2: X(x), y1: PT, y2: H - PB, stroke: "var(--qb-c4)", "stroke-width": 2.5 }, svg);
    // grey ramp of the window under the axis
    const gy = H - PB + 4;
    for (let i = 0; i < 64; i++) {
      const x0 = PL + i / 64 * (W - PL - PR), xm = h0 + (i + 0.5) / 64 * (h1 - h0);
      const [r, g, b] = lutColour(windowLevel(xm, st.lo, st.hi), xm);
      svgEl("rect", { x: x0, y: gy, width: (W - PL - PR) / 64 + 0.5, height: 10, fill: `rgb(${r},${g},${b})` }, svg);
    }
    svgEl("line", { x1: PL, x2: W - PR, y1: H - PB, y2: H - PB, class: "axis" }, svg);
    svgEl("text", { x: PL, y: H - PB + 32, "text-anchor": "start" }, svg).textContent = fmtN(h0);
    svgEl("text", { x: W - PR, y: H - PB + 32, "text-anchor": "end" }, svg).textContent = fmtN(h1 - 1);
    svgEl("text", { x: (PL + W - PR) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = "raw value" + (st.histFull ? " (whole type range)" : " (data range)");
    const yl = svgEl("text", { x: 14, y: (PT + H - PB) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(PT + H - PB) / 2})` }, svg);
    yl.textContent = "count (log)";

    // readout
    const raw = isFluo ? data.fluo[sr * N + sc] * (st.container === "b" ? 16 : 1) : data.ct[sr * N + sc] + (st.container === "b" ? 1024 : 0);
    const val = v[sr * N + sc];
    const lines = [
      typeName,
      `pixel (${sr}, ${sc}): stored ${fmtN(raw)}` + (raw >= 0 && raw < 65536 && Number.isInteger(raw) ? ` = 0x${raw.toString(16).toUpperCase().padStart(4, "0")}` : ""),
    ];
    if (isFluo && st.arith !== "none") lines.push(`  − ${st.bg * (st.container === "b" ? 16 : 1)} (${arithItems.find(a => a.id === st.arith)!.label}) = ${fmtN(val)}`);
    lines.push(`  → display level ${windowLevel(val, st.lo, st.hi)} of 255`);
    lines.push(`window ${fmtN(st.lo)} … ${fmtN(st.hi)}`);
    lines.push(`  ${((st.hi - st.lo) / 256).toFixed(1)} raw values per display level`);
    lines.push(`clipped: ${(100 * below / v.length).toFixed(1)}% low, ${(100 * above / v.length).toFixed(1)}% high`);
    lines.push(`data ${fmtN(dmin)} … ${fmtN(dmax)}`);
    ro.textContent = lines.join("\n");
    note.textContent = isFluo
      ? "Histogram: counts per bin, log scale; shaded = the window. Values at or below the window are black, at or above white (hi-lo LUT: blue / red)."
      : "Hounsfield units: air −1000, fat ≈ −100, water 0, soft tissue 20–80, bone several hundred to 2000+. A window W/L shows L ± W/2.";
  }

  reset();
}

function extent(d: ArrayLike<number>): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < d.length; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i]; }
  return [lo, hi];
}

const fmtN = (x: number) => (Number.isInteger(x) ? String(x) : x.toFixed(1)).replace("-", "−");

autoMount(NAME, mount);
