// Floating point, two views.
//  data-view="bits" (default): the bits of a float32, float16 or bfloat16. Click a bit to
//    flip it, type a number (rounded to the nearest representable value), step by one
//    ulp, jump to special values. Readout: fields, formula, exact decimal value, class,
//    spacing. Plot: spacing (ulp) against |x| on log-log axes for all three formats, with
//    the line "spacing = 1" where integers stop being exact.
//  data-view="sum": add up to 2^22 simulated 16-bit pixel values (seeded) in float32
//    (naive, pairwise, Kahan) and in float16 (values / 65535; naive, pairwise), and plot
//    the relative error against the number of terms; table at the chosen n, plus the
//    variance computed in one pass (E[x^2] - E[x]^2) and in two passes in float32.
// Optional: data-format="float16|float32|bfloat16", data-value="0.1",
// data-mean="1000", data-sd="50", data-n="22" (log2 of the number of terms).
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import {
  F32, F16, FORMATS, encode, decode, fields, classify, ulp, bias, nbits,
  maxFinite, minNormal, minSubnormal, hexString, exactDecimal, rounder, varOnePass, varTwoPass,
} from "./fp";
import { pixels, sumTable, METHOD_LABELS } from "./sum";

const NAME = "floatbits";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { flex-direction: column; align-items: stretch; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .fb-row { display: flex; gap: .6em; align-items: center; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .fb-bits { display: flex; gap: 2px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 2px; }
.qb-demo[data-demo="${NAME}"] .fb-bit { font: inherit; font-family: ui-monospace, monospace; font-size: 18px; width: 27px; height: 38px; padding: 0;
  border-radius: 4px; border: 2px solid; cursor: pointer; background: var(--qb-bg); color: var(--qb-fg); flex: 0 0 auto; }
.qb-demo[data-demo="${NAME}"] .fb-bit.one { color: #fff; }
.qb-demo[data-demo="${NAME}"] .fb-bit.s { border-color: var(--qb-c2); } .qb-demo[data-demo="${NAME}"] .fb-bit.s.one { background: var(--qb-c2); }
.qb-demo[data-demo="${NAME}"] .fb-bit.e { border-color: var(--qb-c1); } .qb-demo[data-demo="${NAME}"] .fb-bit.e.one { background: var(--qb-c1); }
.qb-demo[data-demo="${NAME}"] .fb-bit.m { border-color: var(--qb-c3); } .qb-demo[data-demo="${NAME}"] .fb-bit.m.one { background: var(--qb-c3); }
.qb-demo[data-demo="${NAME}"] .fb-bit.gap { margin-left: 8px; }
.qb-demo[data-demo="${NAME}"] .fb-legend { font-size: 16px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .fb-legend b.s { color: var(--qb-c2); } .qb-demo[data-demo="${NAME}"] .fb-legend b.e { color: var(--qb-c1); } .qb-demo[data-demo="${NAME}"] .fb-legend b.m { color: var(--qb-c3); }
.qb-demo[data-demo="${NAME}"] input.fb-val { font: inherit; font-family: ui-monospace, monospace; width: 9em; padding: .15em .3em; border: 1px solid #bbb; border-radius: 4px; background: var(--qb-bg); color: var(--qb-fg); }
.qb-demo[data-demo="${NAME}"] .fb-main { display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .fb-main > .qb-readout { flex: 1 1 420px; min-width: 0; font-size: 16px; }
.qb-demo[data-demo="${NAME}"] .fb-plot { flex: 1 1 420px; min-width: 280px; max-width: 560px; }
.qb-demo[data-demo="${NAME}"] .fb-lab { font-size: 16px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-slider { grid-template-columns: 5.2em 1fr 5em; }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 15px; }
@media (min-width: 641px) { .qb-demo[data-demo="${NAME}"] .fb-main { flex-wrap: nowrap; } .qb-demo[data-demo="${NAME}"] .fb-main > .qb-panel { flex: 0 0 340px; } }
.reveal .qb-demo[data-demo="${NAME}"] .fb-main { flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .fb-main > .qb-panel { flex: 0 0 360px; }
.reveal .qb-demo[data-demo="${NAME}"] .fb-plot { max-width: 600px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 19px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .fb-main > .qb-readout { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] svg text { font-size: 18px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const fmtByName = (s: string | undefined) => FORMATS.find(f => f.name === s) ?? F16;
const minus = (s: string) => s.replace(/-/g, "−");
const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
/** "2⁻³⁰" style label. */
const pow = (base: number, k: number) => base + String(k).split("").map(c => SUP[c] ?? c).join("");
const COL: Record<string, string> = { float32: "var(--qb-c1)", float16: "var(--qb-c2)", bfloat16: "var(--qb-c3)" };

// ======================================================================================
// bits view
// ======================================================================================
function mountBits(host: HTMLElement): void {
  const f0 = fmtByName(host.dataset.format);
  const v0 = Number(host.dataset.value ?? 0.1);
  const st = { f: f0, bits: encode(f0, Number.isFinite(v0) ? v0 : 0.1) };

  const top = htmlEl("div", "fb-row", "", host);
  const fBtns = presets(top, FORMATS.map(f => ({ label: f.name, apply: () => {
    const v = decode(st.f, st.bits); st.f = f; st.bits = encode(f, v); draw();
  } })));
  htmlEl("span", "fb-lab", "value:", top);
  const inp = htmlEl("input", "fb-val", "", top);
  inp.type = "text";
  inp.addEventListener("change", () => {
    const t = inp.value.trim().toLowerCase();
    const v = t === "inf" || t === "infinity" ? Infinity : t === "-inf" ? -Infinity : t === "nan" ? NaN : Number(t);
    if (!Number.isNaN(v) || t === "nan") { st.bits = encode(st.f, v); }
    draw(); inp.blur();
  });
  inp.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "Escape") inp.blur(); });
  const maxBits = () => 2 ** nbits(st.f);
  const signBit = () => 2 ** (nbits(st.f) - 1);
  button(top, "−1 ulp", () => { st.bits = step(-1); draw(); });
  button(top, "+1 ulp", () => { st.bits = step(1); draw(); });
  button(top, "× 2", () => { st.bits = encode(st.f, 2 * decode(st.f, st.bits)); draw(); });
  button(top, "÷ 2", () => { st.bits = encode(st.f, decode(st.f, st.bits) / 2); draw(); });

  /** Next representable value up (+1) or down (-1) in value, through zero. */
  function step(d: number): number {
    const v = decode(st.f, st.bits);
    if (Number.isNaN(v)) return st.bits;
    const neg = st.bits >= signBit();
    const mag = st.bits % signBit();
    if ((d > 0 && !neg) || (d < 0 && neg)) return Math.min(st.bits + 1, (neg ? signBit() : 0) + (2 ** st.f.e - 1) * 2 ** st.f.m);
    if (mag === 0) return neg ? 1 : signBit() + 1;   // through zero
    return st.bits - 1;
  }

  const bitRow = htmlEl("div", "fb-bits", "", host);
  const legend = htmlEl("div", "fb-legend", "", host);
  const pre = htmlEl("div", "fb-row", "", host);
  const setV = (v: number) => { st.bits = encode(st.f, v); draw(); };
  presets(pre, [
    { label: "1", apply: () => setV(1) },
    { label: "0.1", apply: () => setV(0.1) },
    { label: "1/3", apply: () => setV(1 / 3) },
    { label: "2049", apply: () => setV(2049) },
    { label: "4095", apply: () => setV(4095) },
    { label: "65535", apply: () => setV(65535) },
    { label: "largest", apply: () => setV(maxFinite(st.f)) },
    { label: "smallest normal", apply: () => setV(minNormal(st.f)) },
    { label: "smallest subnormal", apply: () => setV(minSubnormal(st.f)) },
    { label: "−0", apply: () => setV(-0) },
    { label: "∞", apply: () => setV(Infinity) },
    { label: "NaN", apply: () => { st.bits = (2 ** st.f.e - 1) * 2 ** st.f.m + 2 ** (st.f.m - 1); draw(); } },
  ]);
  button(pre, "reset", () => { st.f = f0; st.bits = encode(f0, Number.isFinite(v0) ? v0 : 0.1); draw(); });

  const main = htmlEl("div", "fb-main", "", host);
  const ro = htmlEl("div", "qb-readout", "", main);
  const plotBox = htmlEl("div", "fb-plot", "", main);
  const W = 520, H = 330, PL = 66, PR = 12, PT = 10, PB = 44;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, plotBox);

  function draw(): void {
    const f = st.f;
    st.bits = Math.max(0, Math.min(maxBits() - 1, st.bits));
    fBtns.forEach((b, i) => b.classList.toggle("active", FORMATS[i] === f));
    const v = decode(f, st.bits);
    const n = nbits(f);
    // bit boxes
    bitRow.innerHTML = "";
    const str = st.bits.toString(2).padStart(n, "0");
    for (let i = 0; i < n; i++) {
      const kind = i === 0 ? "s" : i <= f.e ? "e" : "m";
      const b = htmlEl("button", `fb-bit ${kind}${str[i] === "1" ? " one" : ""}${i === 1 || i === f.e + 1 ? " gap" : ""}`, str[i], bitRow);
      b.type = "button";
      b.title = kind === "s" ? "sign" : kind === "e" ? `exponent bit, worth ${2 ** (f.e - i)} in E` : `fraction bit, worth 2^-${i - f.e}`;
      b.addEventListener("click", () => {
        const w = 2 ** (n - 1 - i);
        st.bits = str[i] === "1" ? st.bits - w : st.bits + w;
        draw(); b.blur();
      });
    }
    legend.innerHTML = `<b class="s">sign</b> 1 bit · <b class="e">exponent</b> ${f.e} bits (bias ${bias(f)}) · <b class="m">fraction</b> ${f.m} bits · ${f.name}, ${n} bits in total`;
    if (document.activeElement !== inp) inp.value = Number.isNaN(v) ? "NaN" : minus(String(v)).replace(/−/g, "-");

    const { s, E, F } = fields(f, st.bits);
    const cls = classify(f, st.bits);
    const b = bias(f);
    const Fbin = F.toString(2).padStart(f.m, "0");
    const lines: string[] = [];
    lines.push(`${hexString(f, st.bits)}   ${cls}`);
    lines.push(`sign s = ${s}, exponent field E = ${E}, fraction F = ${F}`);
    if (cls === "normal") lines.push(`value = (−1)^${s} × 2^(${E} − ${b}) × 1.${Fbin}₂`, `      = ${s ? "−" : ""}2^${E - b} × ${(1 + F / 2 ** f.m)}`);
    else if (cls === "subnormal" || cls === "zero") lines.push(`E = 0: subnormal, value = (−1)^${s} × 2^${1 - b} × 0.${Fbin}₂`);
    else if (cls === "infinity") lines.push("E all ones, F = 0: infinity");
    else lines.push("E all ones, F ≠ 0: not a number");
    lines.push(`exactly ${minus(exactDecimal(v, 30))}`);
    if (Number.isFinite(v)) {
      const u = ulp(f, v);
      const up = decode(f, step(1)), dn = decode(f, step(-1));
      lines.push(`spacing here (1 ulp) = ${minus(fmtG(u))}` + (v !== 0 ? `, relative ${fmtG(u / Math.abs(v))}` : ""));
      lines.push(`neighbours ${minus(fmtG(dn))}  <  x  <  ${minus(fmtG(up))}`);
    }
    lines.push(`largest ${fmtG(maxFinite(f))}, smallest normal ${fmtG(minNormal(f))}`);
    lines.push(`machine ε = 2^−${f.m} = ${fmtG(2 ** -f.m)}; integers exact up to 2^${f.m + 1} = ${2 ** (f.m + 1)}`);
    ro.textContent = lines.join("\n");
    drawPlot(v);
  }

  function drawPlot(v: number): void {
    svg.innerHTML = "";
    const xa = -30, xb = 40, ya = -50, yb = 20;   // log2 ranges
    const X = (l: number) => PL + (l - xa) / (xb - xa) * (W - PL - PR);
    const Y = (l: number) => H - PB - (l - ya) / (yb - ya) * (H - PT - PB);
    for (let k = xa; k <= xb; k += 10) {
      svgEl("line", { x1: X(k), x2: X(k), y1: PT, y2: H - PB, class: "grid" }, svg);
      svgEl("text", { x: X(k), y: H - PB + 18, "text-anchor": "middle" }, svg).textContent = pow(2, k);
    }
    for (let k = ya; k <= yb; k += 10) {
      svgEl("line", { x1: PL, x2: W - PR, y1: Y(k), y2: Y(k), class: "grid" }, svg);
      svgEl("text", { x: PL - 5, y: Y(k) + 5, "text-anchor": "end" }, svg).textContent = pow(2, k);
    }
    svgEl("line", { x1: PL, x2: W - PR, y1: Y(0), y2: Y(0), stroke: "var(--qb-fg)", "stroke-dasharray": "5 4" }, svg);
    svgEl("text", { x: W - PR - 4, y: Y(0) - 6, "text-anchor": "end" }, svg).textContent = "spacing 1";
    svgEl("text", { x: (PL + W - PR) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = "|x|";
    const yl = svgEl("text", { x: 14, y: (PT + H - PB) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(PT + H - PB) / 2})` }, svg);
    yl.textContent = "spacing (ulp)";
    svg.appendChild(svgEl("clipPath", { id: "fbclip" })).appendChild(svgEl("rect", { x: PL, y: PT, width: W - PL - PR, height: H - PT - PB }));
    const g = svgEl("g", { "clip-path": "url(#fbclip)" }, svg);
    FORMATS.forEach((f, fi) => {
      const b = bias(f);
      const pts: string[] = [];
      const lo = Math.max(xa, 1 - b - f.m), hi = Math.min(xb, b + 1);
      for (let k = Math.floor(lo); k < hi; k++) {
        const sp = Math.log2(ulp(f, 2 ** k));
        pts.push(`${X(k)},${Y(sp)}`, `${X(k + 1)},${Y(sp)}`);
      }
      svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: COL[f.name], "stroke-width": f === st.f ? 3.5 : 2, opacity: f === st.f ? 1 : 0.55 }, g);
      const ly = PT + 18 + 20 * fi;
      svgEl("line", { x1: PL + 10, x2: PL + 34, y1: ly - 5, y2: ly - 5, stroke: COL[f.name], "stroke-width": 3 }, svg);
      svgEl("text", { x: PL + 40, y: ly }, svg).textContent = `${f.name}: integers exact to ${pow(2, f.m + 1)}`;
    });
    if (Number.isFinite(v) && v !== 0) {
      const l = Math.log2(Math.abs(v)), sp = Math.log2(ulp(st.f, v));
      if (l >= xa && l <= xb) svgEl("circle", { cx: X(l), cy: Y(sp), r: 7, fill: "#fff", stroke: COL[st.f.name], "stroke-width": 3 }, svg);
    }
  }

  draw();
}

function fmtG(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const a = Math.abs(x);
  if (a >= 1e-4 && a < 1e7) return String(Number(x.toPrecision(8)));
  return x.toExponential(4);
}

// ======================================================================================
// sum view
// ======================================================================================
const KMAX = 22, KMIN = 4;

interface Method { label: string; colour: string; dash: string; }
const METHODS: Method[] = METHOD_LABELS.map((label, i) => ({ label, colour: ["var(--qb-c1)", "var(--qb-c3)", "var(--qb-c5)", "var(--qb-c2)", "var(--qb-c4)"][i], dash: i >= 3 ? "6 4" : "" }));

function mountSum(host: HTMLElement): void {
  const d0 = {
    mean: Number(host.dataset.mean ?? 1000), sd: Number(host.dataset.sd ?? 50),
    k: Math.max(KMIN, Math.min(KMAX, Number(host.dataset.n ?? 22))),
  };
  const st = { ...d0 };
  const main = htmlEl("div", "fb-main", "", host);
  const plotBox = htmlEl("div", "fb-plot", "", main);
  const W = 560, H = 420, PL = 70, PR = 12, PT = 74, PB = 44;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, plotBox);
  const panel = htmlEl("div", "qb-panel", "", main);
  const meanS = slider(panel, "mean", 0, 60000, 100, st.mean, v => { st.mean = v; later(); });
  const sdS = slider(panel, "SD", 0, 2000, 10, st.sd, v => { st.sd = v; later(); });
  const kS = slider(panel, "n", KMIN, KMAX, 1, st.k, v => { st.k = v; draw(); }, v => `2^${v}`);
  const row = htmlEl("div", "qb-presets", "", panel);
  const set = (mean: number, sd: number, k: number) => {
    st.mean = mean; st.sd = sd; st.k = k; meanS.set(mean); sdS.set(sd); kS.set(k); recompute();
  };
  presets(row, [
    { label: "2048² image, mean 1000", apply: () => set(1000, 50, 22) },
    { label: "bright and flat", apply: () => set(30000, 20, 22) },
    { label: "small image", apply: () => set(1000, 50, 12) },
  ]);
  button(row, "reset", () => set(d0.mean, d0.sd, d0.k));
  const ro = htmlEl("div", "qb-readout", "", panel);
  ro.style.fontSize = "14px";

  let px: Uint16Array = new Uint16Array(0);
  let tab: ReturnType<typeof sumTable> | null = null;
  let timer = 0;
  function later(): void { clearTimeout(timer); timer = window.setTimeout(recompute, 150); }
  function recompute(): void {
    px = pixels(2 ** KMAX, st.mean, st.sd);
    tab = sumTable(px, KMIN);
    draw();
  }

  function draw(): void {
    if (!tab) return;
    svg.innerHTML = "";
    const ya = -12, yb = 1;
    const X = (k: number) => PL + (k - KMIN) / (KMAX - KMIN) * (W - PL - PR);
    const Y = (l: number) => H - PB - (Math.max(ya, Math.min(yb, l)) - ya) / (yb - ya) * (H - PT - PB);
    for (let k = KMIN; k <= KMAX; k += 2) {
      svgEl("line", { x1: X(k), x2: X(k), y1: PT, y2: H - PB, class: "grid" }, svg);
      svgEl("text", { x: X(k), y: H - PB + 18, "text-anchor": "middle" }, svg).textContent = pow(2, k);
    }
    for (let l = ya; l <= 0; l += 2) {
      svgEl("line", { x1: PL, x2: W - PR, y1: Y(l), y2: Y(l), class: "grid" }, svg);
      svgEl("text", { x: PL - 5, y: Y(l) + 5, "text-anchor": "end" }, svg).textContent = l === 0 ? "1" : pow(10, l);
    }
    svgEl("text", { x: (PL + W - PR) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = "number of pixels summed";
    const yl = svgEl("text", { x: 14, y: (PT + H - PB) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(PT + H - PB) / 2})` }, svg);
    yl.textContent = "relative error of the sum";
    svgEl("line", { x1: X(st.k), x2: X(st.k), y1: PT, y2: H - PB, stroke: "var(--qb-c4)", "stroke-width": 2 }, svg);
    const t = tab;
    const relErr = (m: number, i: number) => {
      const v = t.res[m][i];
      if (!Number.isFinite(v)) return yb;
      const e = Math.abs(v - t.exact[i]) / t.exact[i];
      return e === 0 ? ya : Math.log10(e);
    };
    METHODS.forEach((m, mi) => {
      const pts = t.ks.map((k, i) => `${X(k)},${Y(relErr(mi, i))}`).join(" ");
      svgEl("polyline", { points: pts, fill: "none", stroke: m.colour, "stroke-width": 3, "stroke-dasharray": m.dash }, svg);
      t.ks.forEach((k, i) => svgEl("circle", { cx: X(k), cy: Y(relErr(mi, i)), r: 3.5, fill: m.colour }, svg));
      // legend above the plot, in two columns
      const lx = mi < 3 ? 10 : 290, ly = 18 + 21 * (mi < 3 ? mi : mi - 3);
      svgEl("line", { x1: lx, x2: lx + 24, y1: ly - 5, y2: ly - 5, stroke: m.colour, "stroke-width": 3, "stroke-dasharray": m.dash }, svg);
      svgEl("text", { x: lx + 30, y: ly }, svg).textContent = m.label;
    });
    svgEl("text", { x: W - PR - 4, y: H - PB - 6, "text-anchor": "end", fill: "var(--qb-muted)" }, svg).textContent = "bottom row: exact";

    const i = t.ks.indexOf(st.k), n = 2 ** st.k;
    const lines = [`n = 2^${st.k} = ${n.toLocaleString("en-US")} pixels`, `exact sum ${t.exact[i].toLocaleString("en-US")}`, `exact mean ${(t.exact[i] / n).toFixed(4)}`];
    METHODS.forEach((m, mi) => {
      const v = t.res[mi][i];
      lines.push(`${m.label}:`, `  mean ${Number.isFinite(v) ? (v / n).toFixed(4) : String(v)}, rel. err. ${Number.isFinite(v) ? (Math.abs(v - t.exact[i]) / t.exact[i]).toExponential(1) : "∞"}`);
    });
    const r32 = rounder(F32);
    let m = 0;
    for (let q = 0; q < n; q++) m += px[q];
    m /= n;
    let vx = 0;
    for (let q = 0; q < n; q++) vx += (px[q] - m) ** 2;
    vx /= n;
    lines.push(`variance: exact ${vx.toFixed(2)}`,
      `  float32, one pass       ${varOnePass(px, n, r32).toFixed(2)}`,
      `  float32, two pass       ${varTwoPass(px, n, r32).toFixed(2)}`,
      `  two pass, pairwise sums ${varTwoPass(px, n, r32, true).toFixed(2)}`);
    ro.textContent = minus(lines.join("\n"));
  }

  recompute();
}

autoMount(NAME, host => {
  injectStyle();
  if (host.dataset.view === "sum") mountSum(host); else mountBits(host);
});

