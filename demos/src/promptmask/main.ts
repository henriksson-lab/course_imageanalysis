// A toy promptable segmenter (Networks for cells and objects day, unit sam/). Built like the
// real one (samnet): an "image encoder" run once per image (smoothed copies at three scales
// and a background map: the feature stack of the random-forest day), and a cheap "mask
// decoder" run per click that returns THREE nested candidates (part / object / group) with a
// predicted score each. The score here is an honest proxy (the stability of the mask at a
// wider threshold offset), shown next to the TRUE IoU with the ground truth at each level, so
// one can see when the score misleads. Negative clicks and a box remove ambiguity.
// Automatic mode: a grid of single clicks, then the real pipeline's filters (score,
// stability, box NMS: samnet/post.ts) with the fate of every grid point and F1 against the
// true nuclei. A cost panel counts encoder time (once) against decoder time (per click).
// Auto-mounts on <div data-demo="promptmask">. Options:
// data-preset="edge|between|negative|box|auto|strict|loose".
import { htmlEl, button, slider, checkbox, presets, autoMount, svgEl, type Slider } from "../lib/ui";
import type { RGB } from "../lib/colour";
import { labelColor } from "../lib/colour";
import * as P from "../samnet/post";
import { makeScene, W } from "./scene";
import * as M from "./model";

const NAME = "promptmask";
const C_POS = "#2ca02c", C_NEG = "#d62728", C_BOX = "#ff7f0e";
const CAND_RGB: RGB[] = [[31, 119, 180], [255, 127, 14], [148, 103, 189]];
const LEVEL_NAME = ["part", "object", "group"];
const TRUTH_NAME = ["nucleolus", "nucleus", "group"];
const FATE_COL: Record<string, string> = { kept: "#ffffff", nms: "#b07cff", stability: "#ff9f1c", iou: "#888888", empty: "#888888" };

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .pm-left { flex: 0 1 420px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .pm-stage { position: relative; width: 100%; aspect-ratio: 1; background: #000; border-radius: 4px; overflow: hidden; }
.qb-demo[data-demo="${NAME}"] .pm-stage canvas, .qb-demo[data-demo="${NAME}"] .pm-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .pm-stage svg { touch-action: none; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .pm-credit { font-size: 14px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 400px; max-width: 640px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .pm-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .pm-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .pm-figs { display: flex; gap: .4em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .pm-figs figure { margin: 0; flex: 1 1 0; min-width: 0; cursor: pointer; }
.qb-demo[data-demo="${NAME}"] .pm-figs canvas { width: 100%; max-width: 190px; aspect-ratio: 1; display: block; border-radius: 3px; border: 3px solid transparent; background: #000; }
.qb-demo[data-demo="${NAME}"] .pm-figs figure.pm-sel canvas { border-color: var(--qb-accent); }
.qb-demo[data-demo="${NAME}"] .pm-figs figcaption { font-size: 15px; text-align: center; line-height: 1.25; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] .qb-btn:disabled { opacity: .45; cursor: default; }
.qb-demo[data-demo="${NAME}"] .qb-slider { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .pm-left { flex: 0 0 400px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 600px; max-width: 700px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .pm-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .pm-figs canvas { max-width: 150px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .45em; line-height: 1.25; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { font-size: 16px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Mode = "pos" | "neg" | "box";

interface Preset { id: string; label: string; points?: M.Point[]; box?: boolean; auto?: Partial<M.AutoFilters> & { grid?: number } }

function mount(host: HTMLElement): void {
  injectStyle();
  const scene = makeScene();
  const sp = scene.spots;
  const PRESETS: Preset[] = [
    { id: "edge", label: "click on a nucleus edge", points: [{ x: sp.edge[0], y: sp.edge[1], label: 1 }] },
    { id: "between", label: "click between two nuclei", points: [{ x: sp.between[0], y: sp.between[1], label: 1 }] },
    { id: "negative", label: "+ negative click", points: [{ x: sp.between[0], y: sp.between[1], label: 1 }, { x: sp.pairRight[0], y: sp.pairRight[1], label: 0 }] },
    { id: "box", label: "box", box: true },
    { id: "auto", label: "automatic 8 × 8", auto: { grid: 8, ...M.AUTO_DEFAULT } },
    { id: "strict", label: "strict filters", auto: { score: 0.9, stability: 0.95, nms: 0.7 } },
    { id: "loose", label: "loose filters", auto: { score: 0.5, stability: 0.6, nms: 0.95 } },
  ];

  // ---- DOM ------------------------------------------------------------------------------------
  const left = htmlEl("div", "pm-left", "", host);
  const stage = htmlEl("div", "pm-stage", "", left);
  const cv = htmlEl("canvas", "", "", stage);
  cv.width = cv.height = W;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${W}` }, stage);
  htmlEl("div", "pm-credit", "Synthetic image (256 × 256): nuclei with nucleoli, a touching pair, a cluster of three, a cell with dim cytoplasm, bright debris. Truth known at three levels.", left);
  const cbTruth = checkbox(left, "outline the true nuclei", false, () => redraw());

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pm-lab", "Prompts (click or drag on the image)", panel);
  const modeRow = htmlEl("div", "pm-row", "", panel);
  const modeBtns: Record<Mode, HTMLButtonElement> = {
    pos: button(modeRow, "+ object click", () => setMode("pos")),
    neg: button(modeRow, "− background click", () => setMode("neg")),
    box: button(modeRow, "box", () => setMode("box")),
  };
  button(modeRow, "undo", () => { if (S.box && !S.points.length) S.box = null; else S.points.pop(); promptsChanged(); });
  button(modeRow, "clear", () => { S.points = []; S.box = null; promptsChanged(); });
  const figs = htmlEl("div", "pm-figs", "", panel);
  const fCand = [0, 1, 2].map((k) => {
    const f = htmlEl("figure", "", "", figs);
    const c = htmlEl("canvas", "", "", f);
    const fc = htmlEl("figcaption", "", LEVEL_NAME[k], f);
    f.addEventListener("click", () => { if (S.dec && S.dec.cands[k]) { S.sel = k; S.view = "prompt"; redraw(); } });
    return { f, c, fc };
  });
  const readout = htmlEl("div", "qb-readout", "", panel);

  htmlEl("div", "pm-lab", "Automatic masks (a grid of single clicks, then filters)", panel);
  const autoRow = htmlEl("div", "pm-row", "", panel);
  const runBtn = button(autoRow, "run on the grid", () => runAuto());
  const viewBtn = button(autoRow, "show click result", () => { S.view = S.view === "auto" ? "prompt" : "auto"; redraw(); });
  const sGrid: Slider = slider(panel, "grid points per side", 2, 16, 1, 8, (v) => { S.grid = v; }, (v) => `${v}×${v}`);
  const sScore: Slider = slider(panel, "keep if score >", 0.3, 1, 0.01, M.AUTO_DEFAULT.score, (v) => { S.filt.score = v; refilter(); });
  const sStab: Slider = slider(panel, "keep if stability ≥", 0.3, 1, 0.01, M.AUTO_DEFAULT.stability, (v) => { S.filt.stability = v; refilter(); });
  const sNms: Slider = slider(panel, "remove if box IoU >", 0.3, 1, 0.01, M.AUTO_DEFAULT.nms, (v) => { S.filt.nms = v; refilter(); });
  const presetBtns = presets(panel, PRESETS.map((p) => ({ label: p.label, apply: () => applyPreset(p.id) })));
  button(presetBtns[0].parentElement!, "reset", () => reset());

  // ---- State ----------------------------------------------------------------------------------
  const E = M.encode(scene.img, W, W);
  const S = {
    mode: "pos" as Mode, points: [] as M.Point[], box: null as M.Box | null,
    dec: null as M.Decoded | null, sel: 0,
    clicks: 0, decMs: 0,
    grid: 8, filt: { ...M.AUTO_DEFAULT } as M.AutoFilters,
    cands: [] as P.Candidate[], autoPoints: 0, autoDone: 0, autoMs: 0, autoRun: 0,
    filtered: null as ReturnType<typeof M.filterAuto> | null,
    view: "prompt" as "prompt" | "auto",
  };

  function setMode(m: Mode): void {
    S.mode = m;
    (Object.keys(modeBtns) as Mode[]).forEach((k) => modeBtns[k].classList.toggle("active", k === m));
  }
  setMode("pos");

  // ---- Drawing --------------------------------------------------------------------------------
  const base = new ImageData(W, W);
  for (let p = 0; p < W * W; p++) { const v = Math.round(scene.img[p]); base.data[4 * p] = base.data[4 * p + 1] = base.data[4 * p + 2] = v; base.data[4 * p + 3] = 255; }

  function paintMask(d: Uint8ClampedArray, mask: ArrayLike<number>, c: RGB, fill = 0.45): void {
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const p = y * W + x;
      if (!mask[p]) continue;
      const edge = x === 0 || y === 0 || x === W - 1 || y === W - 1 || !mask[p - 1] || !mask[p + 1] || !mask[p - W] || !mask[p + W];
      const a = edge ? 1 : fill, o = 4 * p;
      d[o] = d[o] * (1 - a) + c[0] * a; d[o + 1] = d[o + 1] * (1 - a) + c[1] * a; d[o + 2] = d[o + 2] * (1 - a) + c[2] * a;
    }
  }
  function paintTruth(d: Uint8ClampedArray): void {
    const t = scene.object;
    for (let y = 1; y < W - 1; y++) for (let x = 1; x < W - 1; x++) {
      const p = y * W + x;
      if (t[p] && (t[p - 1] !== t[p] || t[p + 1] !== t[p] || t[p - W] !== t[p] || t[p + W] !== t[p])) {
        d[4 * p] = 255; d[4 * p + 1] = 255; d[4 * p + 2] = 0;
      }
    }
  }

  function svgClear(): void { while (svg.firstChild) svg.removeChild(svg.firstChild); }
  function drawPrompts(): void {
    const r = 4.5;
    if (S.box) {
      const [x0, y0, x1, y1] = S.box;
      svgEl("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "none", stroke: C_BOX, "stroke-width": 1.6 }, svg);
    }
    for (const p of S.points) {
      svgEl("circle", { cx: p.x, cy: p.y, r, fill: p.label ? C_POS : C_NEG, stroke: "#fff", "stroke-width": 1.2 }, svg);
      svgEl("path", { d: p.label ? `M${p.x - 2.5},${p.y}h5M${p.x},${p.y - 2.5}v5` : `M${p.x - 2.5},${p.y}h5`, stroke: "#fff", "stroke-width": 1.2 }, svg);
    }
  }

  /** Square region the thumbnails zoom into: around the prompts and the masks. */
  function zoomRegion(): [number, number, number] {
    const xs: number[] = [], ys: number[] = [];
    for (const p of S.points) { xs.push(p.x); ys.push(p.y); }
    if (S.box) { xs.push(S.box[0], S.box[2]); ys.push(S.box[1], S.box[3]); }
    if (!xs.length) return [0, 0, W];
    for (const c of S.dec?.cands ?? []) {
      const b = P.maskBox(c.mask, W, W);
      if (b && b.area < 0.25 * W * W) { xs.push(b.box[0], b.box[2] + 1); ys.push(b.box[1], b.box[3] + 1); }
    }
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const side = Math.min(W, Math.max(56, 1.3 * Math.max(x1 - x0, y1 - y0)));
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    return [Math.min(Math.max(0, cx - side / 2), W - side), Math.min(Math.max(0, cy - side / 2), W - side), side];
  }

  const off = document.createElement("canvas"); off.width = off.height = W;
  function drawThumb(c: HTMLCanvasElement, mask: Uint8Array | null, col: RGB, z: [number, number, number]): void {
    const side = 150; c.width = c.height = side;
    const g = c.getContext("2d")!;
    const id = new ImageData(new Uint8ClampedArray(base.data), W, W);
    if (mask) paintMask(id.data, mask, col, 0.5);
    off.getContext("2d")!.putImageData(id, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, z[0], z[1], z[2], z[2], 0, 0, side, side);
  }

  function clickXY(): [number, number] | null {
    const pos = S.points.find((p) => p.label === 1);
    if (pos) return [pos.x, pos.y];
    if (S.box) return [(S.box[0] + S.box[2]) / 2, (S.box[1] + S.box[3]) / 2];
    return null;
  }

  function costLines(): string[] {
    const enc = E.ms, n = S.clicks;
    const lines = [`Encoder: ${enc.toFixed(0)} ms, once for this image.`];
    if (n) {
      const per = S.decMs / n;
      lines.push(`Decoder: ${S.dec ? S.dec.ms.toFixed(0) : "–"} ms this prompt; ${n} prompt${n > 1 ? "s" : ""} so far, ${per.toFixed(0)} ms each.`);
      lines.push(`Total ${(enc + S.decMs).toFixed(0)} ms; re-encoding for every prompt would take ${(n * (enc + per)).toFixed(0)} ms.`);
    }
    return lines;
  }

  function drawPrompt(): void {
    const id = new ImageData(new Uint8ClampedArray(base.data), W, W);
    const d = S.dec;
    if (d && d.cands[S.sel]) paintMask(id.data, d.cands[S.sel].mask, CAND_RGB[S.sel], 0.4);
    if (cbTruth.checked) paintTruth(id.data);
    cv.getContext("2d")!.putImageData(id, 0, 0);
    svgClear(); drawPrompts();
    const z = zoomRegion(), xy = clickXY();
    fCand.forEach((fc, k) => {
      const c = d?.cands[k];
      fc.f.classList.toggle("pm-sel", !!c && k === S.sel);
      drawThumb(fc.c, c ? c.mask : null, CAND_RGB[k], z);
      if (!c || !xy) { fc.fc.textContent = LEVEL_NAME[k]; return; }
      // True IoU: the best of the three truth levels at the click.
      let bestI = -1, bestL = -1;
      [scene.part, scene.object, scene.group].forEach((t, l) => { const v = M.trueIoU(c.mask, t, W, xy[0], xy[1]); if (v > bestI) { bestI = v; bestL = l; } });
      fc.fc.textContent = `${LEVEL_NAME[k]}: score ${c.score.toFixed(2)}\n` + (bestL >= 0 ? `true IoU ${bestI.toFixed(2)} (${TRUTH_NAME[bestL]})` : "no true object here");
      fc.fc.style.whiteSpace = "pre-line";
    });
    const lines: string[] = [];
    if (d && xy) {
      const c = d.cands[S.sel];
      const ious = [scene.part, scene.object, scene.group].map((t) => M.trueIoU(c.mask, t, W, xy[0], xy[1]));
      lines.push(`Shown: ${LEVEL_NAME[S.sel]} (${S.sel === d.best ? "highest score" : "your choice"}), ${c.area} px, stability ${c.stability.toFixed(2)}.`);
      lines.push(`True IoU with nucleolus ${fmt(ious[0])} · nucleus ${fmt(ious[1])} · group ${fmt(ious[2])}`);
    } else lines.push("Click on the image (or pick a preset).");
    lines.push(...costLines());
    readout.textContent = lines.join("\n");
    updateButtons();
  }
  const fmt = (v: number) => (v < 0 ? "–" : v.toFixed(2));

  function pointFates(): string[] {
    const f = S.filtered, rank: Record<string, number> = { kept: 4, nms: 3, stability: 2, empty: 1, iou: 0 };
    const out: string[] = new Array(S.autoPoints).fill("");
    if (!f) return out;
    S.cands.forEach((c, i) => { const fa = f.fate[i]; if (!out[c.point] || rank[fa] > rank[out[c.point]]) out[c.point] = fa; });
    return out;
  }

  function drawAuto(): void {
    const id = new ImageData(new Uint8ClampedArray(base.data), W, W);
    const f = S.filtered;
    if (f) {
      const order = f.kept.map((_, i) => i).sort((a, b) => f.kept[b].area - f.kept[a].area);
      for (const i of order) paintMask(id.data, f.kept[i].mask!, labelColor(i + 1).map((v) => 255 * v) as RGB, 0.45);
    }
    if (cbTruth.checked) paintTruth(id.data);
    cv.getContext("2d")!.putImageData(id, 0, 0);
    svgClear();
    const fates = pointFates();
    const grid = P.pointGrid(Math.round(Math.sqrt(S.autoPoints)) || S.grid);
    grid.forEach(([gx, gy], i) => {
      const fa = fates[i], done = i < S.autoDone;
      svgEl("circle", { cx: gx * W - 0.5, cy: gy * W - 0.5, r: done ? 3 : 1.8, fill: fa === "kept" ? "#fff" : "none",
        stroke: done ? FATE_COL[fa || "iou"] : "#666", "stroke-width": fa === "kept" ? 1 : 1.6 }, svg);
    });
    const lines: string[] = [];
    if (f) {
      const c = f.counts, n = S.cands.length;
      lines.push(`${S.autoDone} points → ${n} candidates → score filter drops ${c.iou} → stability drops ${c.stability + c.empty} → NMS drops ${c.nms} → ${c.kept} masks`);
      const fc: Record<string, number> = { kept: 0, nms: 0, stability: 0, iou: 0, empty: 0 };
      for (const fa of fates) if (fa) fc[fa]++;
      lines.push(`Grid points: ${fc.kept} with a kept mask (white), ${fc.nms} only duplicates (purple), ${fc.stability + fc.empty} unstable (orange), ${fc.iou} low score (grey)`);
      if (f.scores) lines.push(`Against the ${scene.nObjects} true nuclei (IoU ≥ 0.5): TP ${f.scores.tp}, FP ${f.scores.fp}, FN ${f.scores.fn}, F1 ${f.scores.f1.toFixed(2)}`);
      lines.push(`Encoder ${E.ms.toFixed(0)} ms once; decoder ${S.autoMs.toFixed(0)} ms for ${S.autoDone} points (${(S.autoMs / Math.max(1, S.autoDone)).toFixed(0)} ms each)`);
    } else lines.push(`Running the grid …`);
    readout.textContent = lines.join("\n");
    // Thumbnails: blank in automatic view.
    fCand.forEach((fc, k) => { fc.f.classList.remove("pm-sel"); drawThumb(fc.c, null, CAND_RGB[k], [0, 0, W]); fc.fc.textContent = LEVEL_NAME[k]; });
    updateButtons();
  }

  function redraw(): void { if (S.view === "auto") drawAuto(); else drawPrompt(); }
  function updateButtons(): void {
    runBtn.disabled = S.autoRun > 0 && S.autoDone < S.autoPoints;
    viewBtn.disabled = !S.cands.length;
    viewBtn.textContent = S.view === "auto" ? "show click result" : "show automatic masks";
  }

  // ---- Actions --------------------------------------------------------------------------------
  function promptsChanged(): void {
    S.view = "prompt";
    if (!S.points.some((p) => p.label === 1) && !S.box) { S.dec = null; redraw(); return; }
    S.dec = M.decode(E, S.points, S.box);
    S.sel = Math.max(0, S.dec.best);
    S.clicks++; S.decMs += S.dec.ms;
    redraw();
  }

  function refilter(): void {
    if (!S.cands.length) return;
    S.filtered = M.filterAuto(S.cands, S.autoDone, S.filt, scene.object, W, W);
    S.view = "auto";
    redraw();
  }

  function runAuto(): void {
    const run = ++S.autoRun, n = S.grid;
    S.cands = []; S.autoPoints = n * n; S.autoDone = 0; S.autoMs = 0; S.filtered = null; S.view = "auto";
    redraw();
    const step = () => {
      if (run !== S.autoRun) return;
      const t0 = performance.now();
      const k1 = Math.min(S.autoPoints, S.autoDone + 8);
      M.autoSlice(E, n, S.autoDone, k1, S.cands);
      S.autoMs += performance.now() - t0;
      S.autoDone = k1;
      refilter();
      if (S.autoDone < S.autoPoints) setTimeout(step, 0);
      else updateButtons();
    };
    setTimeout(step, 0);
  }

  function setFilters(f: Partial<M.AutoFilters>): void {
    Object.assign(S.filt, f);
    sScore.set(S.filt.score); sStab.set(S.filt.stability); sNms.set(S.filt.nms);
  }

  function applyPreset(id: string): void {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].id === id));
    if (p.auto) {
      const { grid, ...f } = p.auto;
      setFilters(f);
      if (grid) { S.grid = grid; sGrid.set(grid); }
      if (!S.cands.length || (grid && grid * grid !== S.autoPoints)) runAuto();
      else refilter();
      return;
    }
    S.points = (p.points ?? []).map((q) => ({ ...q }));
    S.box = p.box ? [...scene.box] as M.Box : null;
    setMode(p.box ? "box" : "pos");
    promptsChanged();
  }

  function reset(): void {
    S.autoRun++;
    S.points = []; S.box = null; S.dec = null; S.clicks = 0; S.decMs = 0;
    S.cands = []; S.autoPoints = S.autoDone = 0; S.filtered = null; S.view = "prompt";
    S.grid = 8; sGrid.set(8); setFilters(M.AUTO_DEFAULT);
    cbTruth.checked = false;
    presetBtns.forEach((b) => b.classList.remove("active"));
    setMode("pos");
    redraw();
  }

  // ---- Pointer input --------------------------------------------------------------------------
  let drag: { x0: number; y0: number; rect: SVGRectElement | null } | null = null;
  const toImg = (ev: PointerEvent): [number, number] => {
    const r = svg.getBoundingClientRect();
    return [Math.min(W - 1, Math.max(0, (ev.clientX - r.left) / r.width * W)), Math.min(W - 1, Math.max(0, (ev.clientY - r.top) / r.height * W))];
  };
  svg.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    const [x, y] = toImg(ev);
    drag = { x0: x, y0: y, rect: null };
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener("pointermove", (ev) => {
    if (!drag || S.mode !== "box") return;
    const [x, y] = toImg(ev);
    if (!drag.rect) drag.rect = svgEl("rect", { fill: "none", stroke: C_BOX, "stroke-width": 1.4, "stroke-dasharray": "4 3" }, svg);
    const a = Math.min(drag.x0, x), b = Math.min(drag.y0, y);
    Object.entries({ x: a, y: b, width: Math.abs(x - drag.x0), height: Math.abs(y - drag.y0) }).forEach(([k, v]) => drag!.rect!.setAttribute(k, String(v)));
  });
  svg.addEventListener("pointerup", (ev) => {
    if (!drag) return;
    const [x, y] = toImg(ev);
    const d = drag; drag = null;
    if (S.mode === "box") {
      if (Math.hypot(x - d.x0, y - d.y0) < 4) { d.rect?.remove(); return; }
      S.box = [Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y)];
    } else S.points.push({ x, y, label: S.mode === "pos" ? 1 : 0 });
    presetBtns.forEach((b) => b.classList.remove("active"));
    promptsChanged();
  });

  // ---- Start ----------------------------------------------------------------------------------
  redraw();
  if (host.dataset.preset) applyPreset(host.dataset.preset);
}

autoMount(NAME, mount);
