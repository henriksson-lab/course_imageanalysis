// activelearn: which cells should a person label next? A pool of 500 unlabelled cells in a 2D
// feature space (two classes, plus a small rare subtype of class 1 far from the boundary). The
// classifier (logistic regression on Gaussian bumps) is retrained after every new label; the
// background shows its p(class 1). A rule proposes the next cell (ringed): at random, the most
// uncertain, the most different from everything labelled, both in turn, or a cell the model gets
// wrong (a person reviewing the predictions). Clicking a cell labels it by hand. Right: test
// accuracy against the number of labels, averaged over 6 different starts, for every rule.
//
// <div data-demo="activelearn" data-preset="uncertain"></div>
//   data-preset  random | uncertain | rare | both | mistakes
import { autoMount, button, checkbox, htmlEl, presets as presetRow, svgEl, svgPoint } from "../lib/ui";
import { Plot, seriesColor, type Mark } from "../lib/plot";
import { makePool, Labeller, startSet, learningCurve, score, RULES, RULE_LABEL, type Rule, type Curve, prob } from "./model";

const NAME = "activelearn";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .al-left { flex: 0 1 450px; min-width: 260px; }
[data-demo="${NAME}"] .al-left svg { width: 100%; touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .al-right { flex: 1 1 400px; min-width: 0; max-width: 520px; display: flex; flex-direction: column; gap: .45em; }
[data-demo="${NAME}"] .al-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .al-left { flex-basis: 420px; }
.reveal [data-demo="${NAME}"] .al-right { max-width: 460px; gap: .3em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

export const POOL = { n: 500, seed: 1 }, TEST = { n: 600, seed: 2 }, MAXN = 40, STARTS = [1, 2, 3, 4, 5, 6];
interface Preset { id: string; label: string; rule: Rule; n: number; showTrue: boolean }
const PRESETS: Preset[] = [
  { id: "random", label: "random", rule: "random", n: 20, showTrue: false },
  { id: "uncertain", label: "most uncertain", rule: "uncertain", n: 20, showTrue: false },
  { id: "rare", label: "uncertain misses the rare cells", rule: "uncertain", n: 40, showTrue: true },
  { id: "both", label: "uncertain + different", rule: "both", n: 40, showTrue: true },
  { id: "mistakes", label: "a person fixes mistakes", rule: "mistakes", n: 20, showTrue: true },
];
const C0 = "#2a6f97", C1 = "#e08a1e";
const RULE_COLOUR: Record<Rule, string> = { random: "var(--qb-muted)", uncertain: seriesColor(1), diverse: seriesColor(2), both: seriesColor(4), mistakes: seriesColor(0) };

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const pool = makePool(POOL.n, POOL.seed), test = makePool(TEST.n, TEST.seed);
  let rule: Rule = start.rule, showTrue = start.showTrue;
  let L = new Labeller(pool, startSet(pool, 1), 51);
  let history: { n: number; acc: number }[] = [];

  // ---- left: the pool ----
  const left = htmlEl("div", "al-left", "", host);
  const S = 460, M = 8, sx = (v: number) => M + v * (S - 2 * M), sy = (v: number) => S - M - v * (S - 2 * M);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, role: "img", "aria-label": "pool of cells in a 2D feature space" }, left);
  const bg = svgEl("image", { x: M, y: M, width: S - 2 * M, height: S - 2 * M, preserveAspectRatio: "none" }, svg);
  svgEl("rect", { x: M, y: M, width: S - 2 * M, height: S - 2 * M, fill: "none", stroke: "var(--qb-axis)", "stroke-width": 1.5 }, svg);
  const contour = svgEl("path", { fill: "none", stroke: "var(--qb-fg)", "stroke-width": 2.5 }, svg);
  const rareRing = svgEl("ellipse", { cx: sx(0.12), cy: sy(0.88), rx: 0.11 * (S - 2 * M), ry: 0.11 * (S - 2 * M), fill: "none", stroke: "var(--qb-fg)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, svg);
  const rareText = svgEl("text", { x: sx(0.12), y: sy(0.88) + 0.11 * (S - 2 * M) + 18, "text-anchor": "middle", "font-size": 16, class: "qb-halo" }, svg);
  rareText.textContent = "rare subtype";
  const ptsG = svgEl("g", {}, svg);
  const nextRing = svgEl("circle", { r: 13, fill: "none", stroke: "#c00", "stroke-width": 3 }, svg);
  const dots = pool.map(p => svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 3.5 }, ptsG));
  svg.addEventListener("pointerdown", ev => {
    const [px, py] = svgPoint(svg, ev);
    let best = -1, bd = 14 * 14;
    pool.forEach((p, j) => { const d = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2; if (d < bd && !L.isLab[j]) { bd = d; best = j; } });
    if (best >= 0) { L.add(best); record(); draw(); }
  });
  const legend = htmlEl("div", "qb-note", "", left);
  legend.innerHTML = `<span style="color:${C0}">●</span> class A &nbsp; <span style="color:${C1}">●</span> class B (big dots: labelled) &nbsp; background: the model's p(B); <span style="color:#c00">○</span> proposed next. Click a cell to label it yourself.`;

  // ---- right: rules, buttons, readout, learning curves ----
  const right = htmlEl("div", "al-right", "", host);
  presetRow(right, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  const ruleRow = htmlEl("div", "al-row", "", right);
  htmlEl("span", "", "next label:", ruleRow);
  const ruleBtns = RULES.map(rl => button(ruleRow, RULE_LABEL[rl], () => { rule = rl; setRuleBtns(); draw(); }));
  function setRuleBtns(): void { ruleBtns.forEach((b, i) => b.classList.toggle("active", RULES[i] === rule)); }
  const actRow = htmlEl("div", "al-row", "", right);
  button(actRow, "label 1", () => { labelMore(1); });
  button(actRow, "label 10", () => { labelMore(10); });
  button(actRow, "start again", () => { reset(); draw(); });
  const cbTrue = checkbox(actRow, "show true classes", showTrue, v => { showTrue = v; draw(); });
  const readout = htmlEl("div", "qb-readout", "", right);
  const plot = new Plot(right, { width: 520, height: 280, x: [4, MAXN], y: [0.5, 1], xlabel: "labelled cells", ylabel: "test accuracy", margin: { t: 10, r: 12 } });
  const curveMarks: Partial<Record<Rule, Mark<any>>> = {};
  for (const rl of RULES) curveMarks[rl] = plot.line([], [], { color: RULE_COLOUR[rl], width: 2.5, label: RULE_LABEL[rl] });
  const mine: Mark<any> = plot.scatter([], [], { color: "var(--qb-fg)", r: 3.5 });
  plot.legend(RULES.map(rl => ({ label: RULE_LABEL[rl], color: RULE_COLOUR[rl] })), "br");
  const curveNote = htmlEl("div", "qb-note", "Lines: mean of 6 runs from different starting labels. Dots: this run.", right);

  // learning curves in the background, one rule and start at a time
  const curves: Partial<Record<Rule, Curve>> = {};
  const queue = RULES.slice();
  function computeNext(): void {
    const rl = queue.shift();
    if (!rl) { curveNote.textContent = "Lines: mean of 6 runs from different starting labels. Dots: this run."; return; }
    curveNote.textContent = `computing the learning curves (${RULES.length - queue.length} of ${RULES.length})…`;
    curves[rl] = learningCurve(pool, test, rl, MAXN, STARTS);
    curveMarks[rl]!.set(curves[rl]!.n, curves[rl]!.acc);
    setTimeout(computeNext, 10);
  }
  setTimeout(computeNext, 50);

  function record(): void { history.push({ n: L.labelled.length, acc: score(L.w, test).acc }); }
  function reset(): void { L = new Labeller(pool, startSet(pool, 1), 51); history = []; record(); }
  function labelMore(k: number): void {
    for (let i = 0; i < k && L.labelled.length < POOL.n; i++) { L.step(rule); record(); }
    draw();
  }
  function applyPreset(p: Preset): void {
    rule = p.rule; showTrue = p.showTrue; cbTrue.checked = showTrue; setRuleBtns();
    reset();
    while (L.labelled.length < p.n) { L.step(rule); record(); }
    draw();
  }

  // background p(B) on a grid, and the 0.5 contour (marching squares, line segments)
  const R = 48;
  const cv = document.createElement("canvas"); cv.width = R; cv.height = R;
  function drawBackground(): void {
    const ctx = cv.getContext("2d")!, img = ctx.createImageData(R, R), grid = new Float64Array((R + 1) * (R + 1));
    for (let j = 0; j <= R; j++) for (let i = 0; i <= R; i++) grid[j * (R + 1) + i] = prob(L.w, i / R, j / R);
    const a = [0x2a, 0x6f, 0x97], b = [0xe0, 0x8a, 0x1e];
    for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
      const p = prob(L.w, (i + 0.5) / R, 1 - (j + 0.5) / R), t = Math.abs(p - 0.5) * 2 * 0.45, col = p > 0.5 ? b : a, k = 4 * (j * R + i);
      for (let c = 0; c < 3; c++) img.data[k + c] = Math.round(255 + (col[c] - 255) * t);
      img.data[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    bg.setAttribute("href", cv.toDataURL());
    let d = "";
    const at = (i: number, j: number) => grid[j * (R + 1) + i] - 0.5;
    for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
      const v = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)], c = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
      const pts: [number, number][] = [];
      for (let e = 0; e < 4; e++) {
        const v0 = v[e], v1 = v[(e + 1) % 4];
        if ((v0 > 0) !== (v1 > 0)) {
          const t = v0 / (v0 - v1), [x0, y0] = c[e], [x1, y1] = c[(e + 1) % 4];
          pts.push([sx((x0 + t * (x1 - x0)) / R), sy((y0 + t * (y1 - y0)) / R)]);
        }
      }
      if (pts.length >= 2) d += `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}L${pts[1][0].toFixed(1)},${pts[1][1].toFixed(1)}`;
      if (pts.length === 4) d += `M${pts[2][0].toFixed(1)},${pts[2][1].toFixed(1)}L${pts[3][0].toFixed(1)},${pts[3][1].toFixed(1)}`;
    }
    contour.setAttribute("d", d);
  }

  function draw(): void {
    drawBackground();
    pool.forEach((p, j) => {
      const lab = !!L.isLab[j], col = p.c === 1 ? C1 : C0;
      const e = dots[j];
      e.setAttribute("r", lab ? "7.5" : "3.5");
      e.setAttribute("fill", lab || showTrue ? col : "#8a8a8a");
      e.setAttribute("fill-opacity", lab ? "1" : showTrue ? "0.75" : "0.6");
      e.setAttribute("stroke", lab ? "var(--qb-fg)" : "none");
      e.setAttribute("stroke-width", "1.5");
    });
    // labelled dots on top
    for (const j of L.labelled) ptsG.appendChild(dots[j]);
    rareRing.style.display = rareText.style.display = showTrue ? "" : "none";
    const nx = L.propose(rule);
    if (nx >= 0) { nextRing.setAttribute("cx", String(sx(pool[nx].x))); nextRing.setAttribute("cy", String(sy(pool[nx].y))); nextRing.style.display = ""; }
    else nextRing.style.display = "none";
    const sc = score(L.w, test);
    const nRare = L.labelled.filter(j => pool[j].rare).length;
    readout.textContent = `${L.labelled.length} labelled cells (${nRare} of them rare)\ntest accuracy ${(100 * sc.acc).toFixed(1)} %   rare subtype correct ${(100 * sc.rare).toFixed(0)} %`;
    const hs = history.filter(h => h.n <= MAXN);
    mine.set(hs.map(h => h.n), hs.map(h => h.acc));
  }

  setRuleBtns();
  applyPreset(start);
}

autoMount(NAME, mount);
