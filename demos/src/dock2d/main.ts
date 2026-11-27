// FFT docking in 2D (shape-matching day, unit fft-docking). A synthetic receptor with a pocket
// (core blue, surface layer orange) and a key-shaped ligand. Drag the ligand and turn it: the
// live score counts its cells on the surface layer (+1 each) and in the core (ρ each). The
// middle panel is the score of every shift at the current angle: one correlation, two FFTs.
// "search" runs every angle (animated), draws the best score per angle and the top five poses.
// The shapes stand in for molecules; nothing here is a real protein.
// Options: data-preset="<id>", data-presets="id,id,…" (the preset buttons shown).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { drawScalarImage, diverging } from "../lib/colour";
import { Plot } from "../lib/plot";
import { Docker, scoreSparse, fftFlops, rotatePolyCW, Pose } from "../lib/docking";
import {
  BOX, DEFAULTS, Settings, Scene, makeScene, toCell, toWorld, ligandAt, ligandPoly, handScore, anglesOf,
  scoreAngle, rankPoses, AngleResult, TRUE_POSE, isTrueFit,
} from "./model";
import { PRESETS, Preset } from "./presets";

const NAME = "dock2d";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
.qb-demo[data-demo="${NAME}"] .dk-stack { position: relative; width: 100%; aspect-ratio: 1; }
.qb-demo[data-demo="${NAME}"] .dk-stack canvas { width: 100%; height: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .dk-stack svg { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .dk-main { flex: 0 1 min(360px, calc(100vh - 240px)); min-width: 230px; }
.qb-demo[data-demo="${NAME}"] .dk-mid { flex: 0 1 min(240px, calc(100vh - 430px)); min-width: 180px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 225px; max-width: 320px; min-width: 0; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre; overflow-x: auto; min-width: 0; flex-shrink: 0; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 13.5px; padding: .1em .45em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .dk-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .dk-key { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .dk-key span { display: inline-block; width: .9em; height: .9em; vertical-align: -.1em; margin: 0 .2em 0 .5em; border-radius: 2px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .dk-main { flex-basis: 430px; }
.reveal .qb-demo[data-demo="${NAME}"] .dk-mid { flex-basis: 260px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 12.5px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .slides .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; line-height: 1.15; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const COL = { core: [74, 108, 168], surface: [245, 170, 70], empty: [250, 250, 250] };
const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(1));
const SPACINGS = [1, 1.5, 2, 3];

function stack(parent: Element, caption: string): { fig: HTMLElement; cv: HTMLCanvasElement; svg: SVGSVGElement; cap: HTMLElement } {
  const fig = htmlEl("figure", "", "", parent);
  const cap = htmlEl("figcaption", "", caption, fig);
  const st = htmlEl("div", "dk-stack", "", fig);
  const cv = htmlEl("canvas", "", "", st);
  const svg = svgEl("svg", { viewBox: `0 0 ${BOX} ${BOX}`, preserveAspectRatio: "none" }, st);
  return { fig, cv, svg, cap };
}

function polyPath(p: [number, number][]): string {
  return "M" + p.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("L") + "Z";
}

function mount(host: HTMLElement): void {
  injectStyle();
  const ids = host.dataset.presets?.split(",").map(x => x.trim());
  const LIST = ids ? PRESETS.filter(p => ids.includes(p.id)) : PRESETS;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? LIST[0];
  const st: Settings = { ...DEFAULTS };

  // ---- layout ----
  const left = htmlEl("div", "dk-main", "", host);
  const big = stack(left, "drag the ligand; turn it by its handle");
  const key = htmlEl("div", "dk-key", "", left);
  key.innerHTML = `<span style="background:rgb(${COL.core})"></span>core (ρ)<span style="background:rgb(${COL.surface})"></span>surface layer (+1)<span style="background:#2ca02c"></span>contact<span style="background:#d62728"></span>clash`;
  const poseG = svgEl("g", {}, big.svg);
  const cellG = svgEl("g", {}, big.svg);
  const lig = svgEl("path", { fill: "rgba(44,160,44,0.18)" }, big.svg);
  lig.style.stroke = "#1a7a1a"; lig.style.strokeWidth = "2.5"; lig.setAttribute("vector-effect", "non-scaling-stroke");
  lig.style.cursor = "move"; lig.style.pointerEvents = "all";
  const arm = svgEl("line", {}, big.svg);
  arm.style.stroke = "#1a7a1a"; arm.style.strokeWidth = "2"; arm.setAttribute("vector-effect", "non-scaling-stroke");
  const handle = svgEl("circle", { r: 2.2, fill: "#1a7a1a" }, big.svg);
  handle.style.cursor = "grab"; handle.style.touchAction = "none";

  const mid = htmlEl("div", "dk-mid", "", host);
  const mp = stack(mid, "score of every shift");
  const peakM = svgEl("circle", { r: 1.6, fill: "none" }, mp.svg);
  peakM.style.stroke = "#00c8e0"; peakM.style.strokeWidth = "3"; peakM.setAttribute("vector-effect", "non-scaling-stroke");
  const handM = svgEl("path", { fill: "none" }, mp.svg);
  handM.style.stroke = "#111"; handM.style.strokeWidth = "2"; handM.setAttribute("vector-effect", "non-scaling-stroke");
  const plotHost = htmlEl("div", "", "", mid);
  const plot = new Plot(plotHost, { width: 300, height: 180, x: [0, 360], y: [0, 60], xticks: [0, 180, 360], xlabel: "ligand angle (°)", ylabel: "best score", font: 18, margin: { l: 54, b: 46, t: 8, r: 14 } });
  const curve = plot.line([], [], { color: "var(--qb-c1)", width: 2.5 });
  const curDot = plot.vline(0, { color: "#111", width: 1.5, dash: "4 3" });
  const trueLine = plot.vline(TRUE_POSE.angle, { color: "var(--qb-c3)", width: 2 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dk-lab", "presets", panel);
  const presetBtns = presets(panel, LIST.map(p => ({ label: p.label, apply: () => load(p) })));
  const angS = slider(panel, "ligand angle", 0, 359, 1, st.angle, v => { st.angle = v; changed(false); }, v => v + "°");
  const rhoS = slider(panel, "core value ρ", -15, 0, 0.1, st.rho, v => { st.rho = v; changed(true); }, v => v.toFixed(1));
  const thS = slider(panel, "surface (cells)", 0, 4, 1, st.thickness, v => { st.thickness = v; changed(true); });
  const spS = slider(panel, "grid spacing", 0, SPACINGS.length - 1, 1, 0, v => { st.spacing = SPACINGS[v]; changed(true); }, v => String(SPACINGS[v]));
  const stepS = slider(panel, "angle step", 1, 45, 1, st.step, v => { st.step = v; changed(true); }, v => v + "°");
  const bumpC = checkbox(panel, "flexible bump on the ligand", false, v => { st.bump = v; changed(true); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const searchBtn = button(row, "search", () => runSearch());
  button(row, "direct sum (this angle)", () => direct());
  button(row, "reset", () => load(p0));
  const poseRow = htmlEl("div", "qb-presets", "", panel);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const ro2 = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Synthetic shapes stand in for two molecules. Score map: the value at each point is the score with the ligand's centre there, at the current angle (red: contact, blue: clash). Cyan ring: the best shift at this angle; black cross: the ligand by hand. Green line in the curve: the angle of the true fit.", panel);

  // ---- state ----
  let sc: Scene = makeScene(st), dk = new Docker(sc.a, sc.M);
  let map: Float64Array = new Float64Array(0), mapAngle = NaN;
  let results: AngleResult[] = [], top: Pose[] = [], searching = 0, searchMs = 0, searchFfts = 0;
  let directOut = "";

  // ---- dragging ----
  let drag: { x: number; y: number; px: number; py: number } | null = null;
  lig.addEventListener("pointerdown", ev => {
    ev.preventDefault(); lig.setPointerCapture(ev.pointerId);
    const [x, y] = svgPoint(big.svg, ev); drag = { x, y, px: st.x, py: st.y };
  });
  lig.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [x, y] = svgPoint(big.svg, ev);
    st.x = Math.max(0, Math.min(BOX, drag.px + x - drag.x)); st.y = Math.max(0, Math.min(BOX, drag.py + y - drag.y));
    presetBtns.forEach(b => b.classList.remove("active")); draw();
  });
  const up = () => { drag = null; };
  lig.addEventListener("pointerup", up); lig.addEventListener("pointercancel", up);
  let turning = false;
  handle.addEventListener("pointerdown", ev => { ev.preventDefault(); handle.setPointerCapture(ev.pointerId); turning = true; });
  handle.addEventListener("pointermove", ev => {
    if (!turning) return;
    const [x, y] = svgPoint(big.svg, ev), [cx, cy] = snapped();
    // the handle points "up" (−y) at angle 0; clockwise on screen is positive
    st.angle = (Math.round(Math.atan2(x - cx, -(y - cy)) * 180 / Math.PI) + 360) % 360;
    presetBtns.forEach(b => b.classList.remove("active")); changed(false);
  });
  const upH = () => { turning = false; };
  handle.addEventListener("pointerup", upH); handle.addEventListener("pointercancel", upH);

  function snapped(): [number, number] { const [gx, gy] = toCell(sc, st.x, st.y); return toWorld(sc, gx, gy); }

  function load(p: Preset): void {
    Object.assign(st, DEFAULTS, p.s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", LIST[i] === p));
    sync(); changed(true);
  }

  function sync(): void {
    angS.set(st.angle); rhoS.set(st.rho); thS.set(st.thickness); spS.set(Math.max(0, SPACINGS.indexOf(st.spacing)));
    stepS.set(st.step); bumpC.checked = st.bump;
  }

  /** Something changed: rebuild the receptor (if `scene`), recompute the map of the current angle. */
  function changed(scene: boolean): void {
    if (scene) {
      searching++;   // cancels a running search
      sc = makeScene(st); dk = new Docker(sc.a, sc.M);
      results = []; top = []; searchMs = 0; searchFfts = 0; directOut = "";
      poseRow.replaceChildren();
    }
    map = dk.map(ligandAt(sc, st, st.angle)); mapAngle = st.angle;
    draw();
  }

  // ---- search, animated: a few angles per frame ----
  function runSearch(): void {
    const id = ++searching, angles = anglesOf(st.step);
    results = []; top = []; poseRow.replaceChildren();
    const f0 = dk.ffts;
    let i = 0, ms = 0;
    searchBtn.disabled = true;
    const frame = () => {
      if (id !== searching) { searchBtn.disabled = false; return; }
      const t0 = performance.now();
      for (let k = 0; k < 3 && i < angles.length; k++, i++) {
        const r = scoreAngle(sc, st, dk, angles[i]);
        results.push(r.res); map = r.map; mapAngle = angles[i];
      }
      ms += performance.now() - t0;
      searchMs = ms; searchFfts = dk.ffts - f0;
      if (i < angles.length) { draw(); requestAnimationFrame(frame); return; }
      searchBtn.disabled = false;
      top = rankPoses(sc, results);
      top.forEach((p, k) => button(poseRow, `#${k + 1}`, () => {
        [st.x, st.y] = toWorld(sc, p.x, p.y); st.angle = p.angle; sync();
        presetBtns.forEach(b => b.classList.remove("active")); changed(false);
      }));
      map = dk.map(ligandAt(sc, st, st.angle)); mapAngle = st.angle;
      draw();
    };
    requestAnimationFrame(frame);
  }

  function direct(): void {
    const L = ligandAt(sc, st, st.angle), M = sc.M;
    const t0 = performance.now();
    const S = scoreSparse(sc.a, M, L);
    const tDirect = performance.now() - t0;
    const t1 = performance.now();
    const F = dk.map(L);
    const tFft = performance.now() - t1;
    let diff = 0;
    for (let k = 0; k < M * M; k++) diff = Math.max(diff, Math.abs(S[k] - F[k]));
    const adds = M * M * L.dx.length, flops = 2 * fftFlops(M * M) + 6 * M * M;
    directOut = [
      `this angle, ${(M * M).toLocaleString("en")} shifts:`,
      ` direct, ${L.dx.length} cells: ${(adds / 1e6).toFixed(1)} M adds`,
      `   ${tDirect.toFixed(1)} ms`,
      ` FFT route: ≈ ${(flops / 1e6).toFixed(1)} M flops`,
      `   ${tFft.toFixed(1)} ms`,
      ` dense direct: ${(M * M * M * M / 1e6).toFixed(0)} M multiply-adds`,
      ` largest difference ${diff.toExponential(1)}`,
    ].join("\n");
    draw();
  }

  // ---- drawing ----
  function draw(): void {
    const { n, o, M, h, R } = sc;
    // receptor
    const img = new ImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const g = (j + o) * M + i + o, c = R.core[g] ? COL.core : R.surface[g] ? COL.surface : COL.empty, p = 4 * (j * n + i);
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
    }
    big.cv.width = n; big.cv.height = n;
    big.cv.getContext("2d")!.putImageData(img, 0, 0);
    // top poses
    poseG.replaceChildren();
    top.forEach((p, k) => {
      const [x, y] = toWorld(sc, p.x, p.y);
      const q = rotatePolyCW(ligandPoly(st.bump), p.angle).map(([u, v]) => [u + x, v + y] as [number, number]);
      const e = svgEl("path", { d: polyPath(q), fill: "none" }, poseG);
      e.style.stroke = k === 0 ? "#00a0c0" : "#555"; e.style.strokeWidth = k === 0 ? "2.5" : "1.5";
      e.style.strokeDasharray = k === 0 ? "" : "4 3"; e.setAttribute("vector-effect", "non-scaling-stroke");
      const t = svgEl("text", { x: x, y: y + 1, "text-anchor": "middle", "font-size": 4.2, "font-weight": 700 }, poseG);
      t.textContent = String(k + 1); t.style.fill = k === 0 ? "#007a94" : "#333";
    });
    // the ligand by hand: its cells, coloured by what they hit
    const hs = handScore(sc, st), L = ligandAt(sc, st, st.angle);
    cellG.replaceChildren();
    for (let k = 0; k < L.dx.length; k++) {
      const gx = hs.gx + L.dx[k], gy = hs.gy + L.dy[k];
      if (gx < 0 || gy < 0 || gx >= M || gy >= M) continue;
      const g = gy * M + gx, kind = R.core[g] ? "#d62728" : R.surface[g] ? "#2ca02c" : "";
      if (!kind) continue;
      svgEl("rect", { x: (gx - o) * h, y: (gy - o) * h, width: h, height: h, fill: kind, opacity: 0.85 }, cellG);
    }
    const [cx, cy] = toWorld(sc, hs.gx, hs.gy);
    const q = rotatePolyCW(ligandPoly(st.bump), st.angle).map(([u, v]) => [u + cx, v + cy] as [number, number]);
    lig.setAttribute("d", polyPath(q));
    const a = st.angle * Math.PI / 180, hx = cx + 13 * Math.sin(a), hy = cy - 13 * Math.cos(a);
    arm.setAttribute("x1", String(cx)); arm.setAttribute("y1", String(cy)); arm.setAttribute("x2", String(hx)); arm.setAttribute("y2", String(hy));
    handle.setAttribute("cx", String(hx)); handle.setAttribute("cy", String(hy));

    // score map of the box, at mapAngle
    const sm = new Float64Array(n * n);
    let hi = 1, bx = 0, by = 0, best = -Infinity;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = map[(j + o) * M + i + o];
      sm[j * n + i] = v;
      if (v > hi) hi = v;
      if (v > best + 1e-9) { best = v; bx = i; by = j; }
    }
    drawScalarImage(mp.cv, sm, n, n, { map: diverging, lo: -hi, hi });
    mp.cap.textContent = `score of every shift at ${mapAngle}°`;
    if (best > 1e-9) { peakM.style.display = ""; peakM.setAttribute("cx", String((bx + 0.5) * h)); peakM.setAttribute("cy", String((by + 0.5) * h)); }
    else peakM.style.display = "none";
    const c = 2.2;
    handM.setAttribute("d", `M${cx - c},${cy}H${cx + c}M${cx},${cy - c}V${cy + c}`);
    handM.style.display = mapAngle === st.angle ? "" : "none";

    // best score per angle
    const xs = results.map(r => r.angle), ys = results.map(r => r.best);
    const ymax = Math.max(10, ...ys) * 1.1, ymin = Math.min(0, ...ys);
    plot.setDomain([0, 360], [ymin, ymax]);
    curve.set(xs, ys);
    curDot.set(mapAngle); trueLine.set(TRUE_POSE.angle);

    // readout
    const lines = [
      `by hand: ${st.angle}° at (${cx.toFixed(1)}, ${cy.toFixed(1)})`,
      `score ${fmt(hs.score)} = ${hs.contact} contacts`,
      `        + (${fmt(st.rho)})·${hs.clash} clashes`,
      `grid ${n}×${n} in a ${M}×${M} FFT grid`,
      `ligand ${L.dx.length} cells; FFTs done ${dk.ffts}`,
    ];
    if (results.length) {
      const nA = anglesOf(st.step).length;
      lines.push(`search: ${results.length}/${nA} angles, ${searchFfts} FFTs`, `        ${searchMs.toFixed(0)} ms`);
    }
    if (top.length) {
      lines.push("top poses: angle (centre) score");
      top.forEach((p, k) => {
        const [x, y] = toWorld(sc, p.x, p.y);
        lines.push(`#${k + 1} ${String(p.angle).padStart(3)}° (${x.toFixed(1)}, ${y.toFixed(1)}) ${fmt(p.score)}${isTrueFit(sc, p) ? " ✓ fit" : ""}`);
      });
      if (!top.some(p => isTrueFit(sc, p))) lines.push("true fit (20°) not in the top 5");
    }
    ro.textContent = lines.join("\n");
    ro2.textContent = directOut;
    ro2.style.display = directOut ? "" : "none";
  }

  load(p0);
}

autoMount(NAME, mount);
