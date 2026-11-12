// Bleed-through and compensation (adapted from the earlier course's flow-cytometry
// "compensation" demo, reframed for images). A synthetic two-channel image of a cell
// (seeded; dye 1 in the nucleus, dye 2 in the membrane, optional double-labelled vesicles,
// Poisson noise), measured = S × true, S = [[1, s12], [s21, 1]]. Left: one channel (or the
// merge) before and after compensation with the student's guess. Middle: the 2D pixel
// scatter (channel 2 against channel 1) before compensation; right: after, S_guess⁻¹ ×
// measured. A least-squares line through the nucleus-only pixels (dye 1 alone) and through
// the membrane-only pixels shows the tilt: up = under-compensated, flat = right, down =
// over-compensated (same verdict logic as the old demo). Readouts: S, S⁻¹, slopes, medians,
// Pearson's r of the two channels inside the cell.
// data-mode="cells": the old demo's four cytometry populations (one point per cell).
// Auto-mounts on <div data-demo="bleedthrough">. Options: data-s21, data-s12 (true
// spillover, default 0.25 and 0.05; cells: 0.25 and 0.02), data-start="s21 s12" (initial
// guess), data-seed, data-photons (image mode, default 600), data-mode (image|cells),
// data-preset (none|under|over|true|similar).
import { htmlEl, slider, presets, checkbox, button, numList, autoMount } from "../lib/ui";
import { Plot } from "../lib/plot";
import { RGB } from "../lib/colour";
import * as U from "../lib/unmix";
import { poisson, rng } from "../lib/random";
import { injectStyle, uniqueClip, canvasFigure, drawMerge, fmt, quantile, COMMON_CSS } from "../lsunmix/common";

const NAME = "bleedthrough";
const CSS = COMMON_CSS + `
[data-demo="${NAME}"] { gap: .7em; align-items: flex-start; }
[data-demo="${NAME}"] .bt-imgs { flex: 0 1 200px; min-width: 150px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .bt-sc { flex: 1 1 290px; min-width: 220px; max-width: 360px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 380px; gap: .35em; }
[data-demo="${NAME}"] .bt-status { display: flex; flex-direction: column; gap: .3em; font-size: .9em; }
[data-demo="${NAME}"] .bt-status div { border-left: 5px solid #ccc; padding: .1em .5em; }
[data-demo="${NAME}"] .bt-ok { color: #1a7f37; font-weight: 600; }
[data-demo="${NAME}"] .bt-under { color: #b35900; font-weight: 600; }
[data-demo="${NAME}"] .bt-over { color: #c62828; font-weight: 600; }
.reveal [data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .bt-imgs { flex: 0 0 165px; }
.reveal [data-demo="${NAME}"] .bt-sc { flex: 0 0 300px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 350px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 16px; }
.reveal [data-demo="${NAME}"] .bt-status { font-size: 16px; }
`;

const FLAT = 0.015;   // |slope| below this counts as flat (as in the old demo)
type Grp = 0 | 1 | 2 | 3 | 4;
const IMG_GROUPS = ["background", "nucleus (dye 1 only)", "membrane (dye 2 only)", "vesicles (both)", "other"];
const CELL_GROUPS = ["unstained", "dye 1 only", "dye 2 only", "both"];
const PAL = ["#8c8c8c", "#009E73", "#D55E00", "#7B3294", "#c8c8c8"];
const DARK = ["#444", "#00563f", "#8a3000", "#4b1d5c", "#999"];
const C1: RGB = [0.1, 1, 0.25], C2: RGB = [1, 0.25, 0.85];

interface Data { x: Float64Array; y: Float64Array; g: Uint8Array; w: number; h: number; inCell?: Uint8Array }

function imageData(s21: number, s12: number, photons: number, seed: number, vesicles: boolean): Data {
  const sc = U.cellScene(128, 3), P = 128 * 128, r = rng(seed);
  const x = new Float64Array(P), y = new Float64Array(P), g = new Uint8Array(P);
  // smooth brightness variation (staining is never uniform): a wide spread of intensities
  // along each population, so the scatter shows a line and its slope is well defined
  const field = (sd: number) => {
    const q = rng(sd), f = U.blurMap(Float32Array.from({ length: P }, () => q.unif()), 128, 128, 4);
    let lo = Infinity, hi = -Infinity;
    for (const v of f) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return f.map(v => (v - lo) / (hi - lo));
  };
  const f1 = field(21), f2 = field(22);
  for (let p = 0; p < P; p++) {
    const v = vesicles ? sc.vesicles[p] : 0;
    const a1 = 0.01 + sc.nucleus[p] * (0.25 + 1.1 * f1[p]) + 0.8 * v, a2 = 0.01 + sc.membrane[p] * (0.3 + 1.1 * f2[p]) + 0.8 * v;
    x[p] = poisson(r, photons * (a1 + s12 * a2));
    y[p] = poisson(r, photons * (s21 * a1 + a2));
    g[p] = v > 0.3 ? 3 : sc.nucleus[p] > 0.5 && sc.membrane[p] < 0.02 ? 1 : sc.membrane[p] > 0.5 && sc.nucleus[p] < 0.02 ? 2
      : sc.nucleus[p] < 0.01 && sc.membrane[p] < 0.01 && v < 0.01 ? 0 : 4;
  }
  return { x, y, g, w: 128, h: 128, inCell: Uint8Array.from(sc.cell) };
}

function cellsData(s21: number, s12: number, seed: number): Data {
  const c = U.cytometryCells(s21, s12, seed);
  return { x: c.mx, y: c.my, g: Uint8Array.from(c.group), w: 0, h: 0 };
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const cells = host.dataset.mode === "cells";
  const num = (s: string | undefined, d: number) => (s !== undefined && s !== "" && Number.isFinite(Number(s)) ? Number(s) : d);
  const trueDefault = { s21: num(host.dataset.s21, 0.25), s12: num(host.dataset.s12, cells ? 0.02 : 0.05) };
  const seed = num(host.dataset.seed, 1), photons = num(host.dataset.photons, 600);
  const start = numList(host.dataset.start, [0, 0]);
  const T = { ...trueDefault };
  const gs = { s21: start[0] ?? 0, s12: start[1] ?? 0 };
  let vesicles = true, view: "1" | "2" | "merge" = "2", revealed = false;
  let D: Data = cells ? cellsData(T.s21, T.s12, seed) : imageData(T.s21, T.s12, photons, seed, vesicles);
  const groups = cells ? CELL_GROUPS : IMG_GROUPS;

  // ---- images (image mode) ----
  let cBefore: HTMLCanvasElement | null = null, cAfter: HTMLCanvasElement | null = null, viewBtns: HTMLButtonElement[] = [];
  if (!cells) {
    const imgs = htmlEl("div", "bt-imgs", "", host);
    cBefore = canvasFigure(imgs, "measured", 128, 128).canvas;
    cAfter = canvasFigure(imgs, "compensated", 128, 128).canvas;
    const row = htmlEl("div", "um-row", "", imgs);
    viewBtns = (["1", "2", "merge"] as const).map(v => button(row, v === "merge" ? "merge" : "ch " + v, () => { view = v; viewBtns.forEach((b, i) => b.classList.toggle("active", ["1", "2", "merge"][i] === v)); draw(); }));
    viewBtns[1].classList.add("active");
  }

  // ---- scatter plots ----
  const mkPlot = (title: string, xl: string, yl: string) => {
    const box = htmlEl("div", "bt-sc", "", host);
    const p = new Plot(box, { width: 380, height: 380, x: [0, 1], y: [0, 1], title, xlabel: xl, ylabel: yl, xticks: 4, yticks: 4, font: 18 });
    uniqueClip(p.svg, NAME);
    return p;
  };
  const pB = mkPlot("before: measured", cells ? "detector 1" : "channel 1", cells ? "detector 2" : "channel 2");
  const pA = mkPlot("after: S⁻¹ × measured", "dye 1 (compensated)", "dye 2 (compensated)");

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const leg = htmlEl("div", "um-row", "", panel);
  groups.forEach((name, i) => { if (i === 4) return; const sp = htmlEl("span", "", "● " + name, leg); sp.style.color = PAL[i]; sp.style.fontSize = "15px"; });
  const sl21 = slider(panel, "s₂₁ dye 1 → ch 2", 0, 1, 0.01, gs.s21, v => { gs.s21 = v; manual(); });
  const sl12 = slider(panel, "s₁₂ dye 2 → ch 1", 0, 1, 0.01, gs.s12, v => { gs.s12 = v; manual(); });
  const r2 = (v: number) => Math.round(v * 100) / 100;
  const setGuess = (a: number, b: number) => { gs.s21 = a; gs.s12 = b; sl21.set(a); sl12.set(b); draw(); };
  const setTrue = (a: number, b: number) => {
    T.s21 = a; T.s12 = b;
    D = cells ? cellsData(a, b, seed) : imageData(a, b, photons, seed, vesicles);
    setRanges();
  };
  const presetDefs = [
    { id: "none", label: "no compensation", apply: () => { setTrue(trueDefault.s21, trueDefault.s12); revealed = false; setGuess(0, 0); } },
    { id: "under", label: "under", apply: () => { setTrue(trueDefault.s21, trueDefault.s12); setGuess(Math.max(0, r2(T.s21 - 0.1)), r2(T.s12)); } },
    { id: "over", label: "over", apply: () => { setTrue(trueDefault.s21, trueDefault.s12); setGuess(Math.min(1, r2(T.s21 + 0.1)), r2(T.s12)); } },
    { id: "true", label: "reveal true S", apply: () => { revealed = true; setGuess(r2(T.s21), r2(T.s12)); } },
    { id: "similar", label: "similar dyes: noise explodes", apply: () => { setTrue(0.9, 0.9); revealed = true; setGuess(0.9, 0.9); } },
  ];
  const btns = presets(panel, presetDefs.map(p => ({ label: p.label, apply: p.apply })));
  if (!cells) checkbox(panel, "double-labelled vesicles", vesicles, v => { vesicles = v; setTrue(T.s21, T.s12); draw(); });
  const manual = () => { btns.forEach(b => b.classList.remove("active")); draw(); };
  const mat = htmlEl("div", "qb-readout um-readout", "", panel);
  const status = htmlEl("div", "bt-status", "", panel);
  const st1 = htmlEl("div", "", "", status), st2 = htmlEl("div", "", "", status);
  st1.style.borderColor = PAL[1]; st2.style.borderColor = PAL[2];
  const extra = htmlEl("div", "qb-readout um-readout", "", panel);

  // ---- ranges and point layers ----
  let lo = 0, hi = 1;
  function setRanges(): void {
    if (cells) { lo = -1000; hi = 5000; }
    else { hi = 1.1 * Math.max(quantile(D.x, 0.999), quantile(D.y, 0.999)); lo = -0.25 * hi; }
    for (const p of [pB, pA]) p.setDomain([lo, hi], [lo, hi]);
  }
  const dotsMark = (p: Plot) => p.mark((g: SVGGElement, xs: Float64Array, ys: Float64Array) => {
    // one path per group: "M x y h0" + round caps = a dot
    const r = cells ? 4.5 : 3.2, op = cells ? 0.55 : 0.4;
    for (const grp of [4, 0, 2, 1, 3] as Grp[]) {
      let d = "";
      for (let i = 0; i < xs.length; i++) if (D.g[i] === grp && (!D.inCell || D.inCell[i] || grp !== 4)) {
        const X = p.x(xs[i]), Y = p.y(ys[i]);
        if (X < p.box.l - 5 || X > p.box.r + 5 || Y < p.box.t - 5 || Y > p.box.b + 5) continue;
        d += `M${X.toFixed(1)} ${Y.toFixed(1)}h0`;
      }
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d); path.setAttribute("stroke", PAL[grp]); path.setAttribute("stroke-width", String(r));
      path.setAttribute("stroke-linecap", "round"); path.setAttribute("stroke-opacity", String(op)); path.setAttribute("fill", "none");
      g.appendChild(path);
    }
  }, [new Float64Array(0), new Float64Array(0)] as [Float64Array, Float64Array]);
  const dB = dotsMark(pB), dA = dotsMark(pA);
  const trend1 = pA.line([0, 0], [0, 0], { color: DARK[1], width: 5 }), trend2 = pA.line([0, 0], [0, 0], { color: DARK[2], width: 5 });
  const ref1 = pA.hline(0, { color: "#333", width: 2, dash: "7 5" }), ref2 = pA.vline(0, { color: "#333", width: 2, dash: "7 5" });
  const axB0 = pB.hline(0, { color: "#999", width: 1.5 }), axB1 = pB.vline(0, { color: "#999", width: 1.5 });
  void axB0; void axB1;

  function verdict(b: number, which: string, up: string, down: string): string {
    if (Math.abs(b) < FLAT) return `<span class="bt-ok">flat: compensated</span>`;
    return b > 0 ? `<span class="bt-under">${up}: under (${which} too small)</span>` : `<span class="bt-over">${down}: over (${which} too large)</span>`;
  }

  function draw(): void {
    const k = 1 - gs.s21 * gs.s12, n = D.x.length;
    const cx = new Float64Array(n), cy = new Float64Array(n);
    for (let i = 0; i < n; i++) { cx[i] = (D.x[i] - gs.s12 * D.y[i]) / k; cy[i] = (-gs.s21 * D.x[i] + D.y[i]) / k; }
    dB.set(D.x, D.y); dA.set(cx, cy);
    const pick = (v: Float64Array, g: number) => Array.from(v).filter((_, i) => D.g[i] === g);
    const f1x = pick(cx, 1), f1y = pick(cy, 1), f2x = pick(cx, 2), f2y = pick(cy, 2);
    const l1 = U.lsLine(f1x, f1y), l2 = U.lsLine(f2y, f2x);
    const mx1 = Math.min(...f1x), Mx1 = Math.max(...f1x), my2 = Math.min(...f2y), My2 = Math.max(...f2y);
    trend1.set([mx1, Mx1], [l1.a + l1.b * mx1, l1.a + l1.b * Mx1]);
    trend2.set([l2.a + l2.b * my2, l2.a + l2.b * My2], [my2, My2]);
    const bgX = U.median(pick(cx, 0)), bgY = U.median(pick(cy, 0));
    ref1.set(bgY); ref2.set(bgX);
    const med1 = U.median(f1y), med2 = U.median(f2x);
    const Si = [1 / k, -gs.s12 / k, -gs.s21 / k, 1 / k];
    mat.textContent =
      `S   = [${fmt(1)}  ${fmt(gs.s12)}]\n      [${fmt(gs.s21)}  ${fmt(1)}]\n` +
      `S⁻¹ = [${fmt(Si[0], 3)} ${fmt(Si[1], 3)}]\n      [${fmt(Si[2], 3)} ${fmt(Si[3], 3)}]`;
    const d1 = cells ? "dye 1 only" : "nucleus", d2 = cells ? "dye 2 only" : "membrane";
    st1.innerHTML = `${d1}: slope <b>${fmt(l1.b)}</b>: ${verdict(l1.b, "s₂₁", "tilts up", "tilts down")}<br>median dye 2 <b>${fmt(med1, 0)}</b> (${cells ? "unstained" : "background"} <b>${fmt(bgY, 0)}</b>)`;
    st2.innerHTML = `${d2}: slope <b>${fmt(l2.b)}</b>: ${verdict(l2.b, "s₁₂", "leans right", "leans left")}<br>median dye 1 <b>${fmt(med2, 0)}</b> (${cells ? "unstained" : "background"} <b>${fmt(bgX, 0)}</b>)`;
    const lines: string[] = [];
    if (!cells && D.inCell) {
      lines.push(`Pearson r (cell): ${fmt(U.pearson(D.x, D.y, D.inCell))} → ${fmt(U.pearson(cx, cy, D.inCell))}`);
      lines.push(`nucleus, before: Σxy/Σx² = ${fmt(U.slopeThroughOrigin(pick(D.x, 1), pick(D.y, 1)), 3)}`);
      const bg = pick(cy, 0), m = bg.reduce((s, v) => s + v, 0) / bg.length, sd = Math.sqrt(bg.reduce((s, v) => s + (v - m) ** 2, 0) / bg.length);
      const bx = pick(D.y, 0), mb = bx.reduce((s, v) => s + v, 0) / bx.length, sdb = Math.sqrt(bx.reduce((s, v) => s + (v - mb) ** 2, 0) / bx.length);
      lines.push(`background sd, ch 2: ${fmt(sdb, 1)} → ${fmt(sd, 1)}`);
    } else {
      const u = pick(cy, 0), m = u.reduce((s, v) => s + v, 0) / u.length;
      lines.push(`unstained sd, dye 2: ${fmt(Math.sqrt(u.reduce((s, v) => s + (v - m) ** 2, 0) / u.length), 0)}`);
    }
    lines.push(revealed ? `true: s₂₁ = ${fmt(T.s21)}, s₁₂ = ${fmt(T.s12)}` : "true S hidden (preset \"reveal true S\")");
    extra.textContent = lines.join("\n");
    if (cBefore && cAfter) {
      const sc = 1 / (0.9 * hi);
      const show = (c: HTMLCanvasElement, a: Float64Array, b: Float64Array) => {
        const maps = view === "merge" ? [a, b] : view === "1" ? [a, a] : [b, b];
        const cols: RGB[] = view === "merge" ? [C1, C2] : [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]];
        drawMerge(c, maps, cols, [1 / sc, 1 / sc], D.w, D.h);
      };
      show(cBefore, D.x, D.y); show(cAfter, cx, cy);
    }
  }

  setRanges();
  const pre = presetDefs.findIndex(p => p.id === host.dataset.preset);
  if (pre >= 0) { presetDefs[pre].apply(); btns[pre].classList.add("active"); } else draw();
}

autoMount(NAME, mount);
