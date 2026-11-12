// raster: a software rasteriser on a coarse pixel grid (unit `rasterise/`, 3D-graphics day).
// One or two triangles in window coordinates (x right, y up, pixel centres at i + 0.5), each
// vertex with a depth w (the distance from the camera). Step through the pixels of each
// triangle's bounding box: the current pixel shows its three edge functions, its barycentric
// coordinates, its interpolated depth and the depth test. Views: colour buffer, depth buffer,
// barycentric coordinates (as RGB), the edge function E_ab of the current triangle.
// Toggles: back-face culling, top-left rule, perspective-correct interpolation; depth-buffer
// bits and the near plane for the depth-precision preset (with the z_ndc(d) curve).
// Drag a vertex (snaps to half pixels); click a pixel to read what happened there.
// Auto-mounts on <div data-demo="raster">. Optional: data-preset="one|intersect|shared|cull|
// perspective|precision", data-view="colour|depth|bary|edge".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { WinVertex, Fragment, rasteriseSteps, ndcDepth, edge, FragmentShader } from "../lib/raster";
import { diverging, grey } from "../lib/colour";
import { Plot } from "../lib/plot";

const NAME = "raster";
const FAR = 10;

const STYLE = `
.qb-demo[data-demo="${NAME}"] .rs-left { flex: 1 1 520px; min-width: 280px; max-width: 640px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] canvas.rs-grid { width: 100%; display: block; touch-action: none; cursor: crosshair; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; line-height: 1.3; }
.qb-demo[data-demo="${NAME}"] .rs-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .rs-curve { max-width: 330px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; gap: .8em; }
.reveal .qb-demo[data-demo="${NAME}"] .rs-left { flex: 0 0 600px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 420px; max-width: 520px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 17px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .25em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;

type View = "colour" | "depth" | "bary" | "edge";
interface Tri { p: [number, number][]; w: number[]; col: [number, number, number][]; uv?: [number, number][] }
interface State { res: number; tris: Tri[]; cull: boolean; topLeft: boolean; persp: boolean; near: number; bits: number; view: View }
interface Preset { id: string; label: string; make: () => State }

const RES: [number, number][] = [[16, 12], [32, 24], [64, 48]];
const RGBV: [number, number, number][] = [[1, 0.15, 0.15], [0.15, 0.85, 0.2], [0.2, 0.35, 1]];
const ORANGE: [number, number, number][] = [[1, 0.65, 0.1], [1, 0.65, 0.1], [1, 0.65, 0.1]];
const base = (o: Partial<State>): State => ({ res: 1, tris: [], cull: false, topLeft: true, persp: true, near: 1, bits: 24, view: "colour", ...o });

/** A floor quad seen in perspective: two triangles with texture coordinates and true w. */
function floorQuad(): Tri[] {
  const [W, H] = RES[2], f = 1.2, asp = W / H;
  const P: [number, number, number][] = [[-1.6, -1, -1.3], [1.6, -1, -1.3], [1.6, -1, -9], [-1.6, -1, -9]];
  const s = P.map(([x, y, z]) => { const w = -z; return [(f * x / asp / w + 1) / 2 * W, (f * y / w + 1) / 2 * H, w] as [number, number, number]; });
  const uv: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const mk = (ix: number[]): Tri => ({ p: ix.map(k => [s[k][0], s[k][1]] as [number, number]), w: ix.map(k => s[k][2]), col: ix.map(() => [1, 1, 1] as [number, number, number]), uv: ix.map(k => uv[k]) });
  return [mk([0, 1, 2]), mk([0, 2, 3])];
}

const PRESETS: Preset[] = [
  { id: "one", label: "one triangle", make: () => base({ tris: [{ p: [[4, 3], [26, 8], [12, 21]], w: [2, 4, 3], col: RGBV }] }) },
  { id: "intersect", label: "two triangles intersect", make: () => base({ tris: [
    { p: [[3, 3], [27, 5], [10, 21]], w: [2, 8, 2], col: RGBV },
    { p: [[8, 18], [14, 2], [29, 15]], w: [4, 4, 4], col: ORANGE }] }) },
  { id: "shared", label: "a shared edge", make: () => base({ res: 0, tris: [
    { p: [[1.5, 1.5], [13.5, 1.5], [1.5, 10.5]], w: [3, 3, 3], col: RGBV },
    { p: [[13.5, 1.5], [13.5, 10.5], [1.5, 10.5]], w: [3, 3, 3], col: ORANGE }] }) },
  { id: "cull", label: "flipped winding is culled", make: () => base({ cull: true, tris: [
    { p: [[3, 3], [15, 4], [8, 20]], w: [3, 3, 3], col: RGBV },
    { p: [[17, 4], [24, 20], [29, 3]], w: [3, 3, 3], col: ORANGE }] }) },
  { id: "perspective", label: "affine vs perspective-correct", make: () => base({ res: 2, persp: false, tris: floorQuad() }) },
  { id: "precision", label: "near plane 0.01 vs 1", make: () => base({ near: 0.01, bits: 8, view: "colour", tris: [
    { p: [[3, 4], [29, 6], [12, 21]], w: [8, 9, 8], col: RGBV },
    { p: [[6, 18], [16, 3], [29, 16]], w: [8.5, 8.5, 8.5], col: ORANGE }] }) },
];

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const f3 = (v: number) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);
const f2 = (v: number) => (Math.abs(v) < 5e-3 ? 0 : v).toFixed(2);

function mount(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let st = p0.make();
  if (host.dataset.view && ["colour", "depth", "bary", "edge"].includes(host.dataset.view)) st.view = host.dataset.view as View;
  let steps: Fragment[] = [], culled: boolean[] = [], k = 0, sel: [number, number] | null = null;

  // ---- layout ----
  const left = htmlEl("div", "rs-left", "", host);
  const cv = htmlEl("canvas", "rs-grid", "", left);
  const viewRow = htmlEl("div", "rs-row", "", left);
  const viewBtns = (["colour", "depth", "bary", "edge"] as View[]).map(v =>
    button(viewRow, { colour: "colour buffer", depth: "depth buffer", bary: "barycentrics", edge: "edge function E_ab" }[v], () => { st.view = v; draw(); }));
  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const stepRow = htmlEl("div", "rs-row", "", panel);
  button(stepRow, "next pixel", () => step(1));
  button(stepRow, "next row", () => stepRow1());
  button(stepRow, "all", () => { k = steps.length; draw(); });
  button(stepRow, "start", () => { k = 0; draw(); });
  button(stepRow, "reset", () => load(p0));
  const resRow = htmlEl("div", "rs-row", "", panel);
  const resBtns = RES.map((r, i) => button(resRow, `${r[0]}×${r[1]}`, () => setRes(i)));
  const chk = htmlEl("div", "rs-row", "", panel);
  const cullC = checkbox(chk, "back-face culling", st.cull, v => { st.cull = v; recompute(); });
  const tlC = checkbox(chk, "top-left rule", st.topLeft, v => { st.topLeft = v; recompute(); });
  const pcC = checkbox(chk, "perspective-correct", st.persp, v => { st.persp = v; recompute(); });
  const wS = slider(panel, "depth w of vertex", 1, 10, 0.1, 3, v => { if (sel) { st.tris[sel[0]].w[sel[1]] = v; recompute(); } });
  const nS = slider(panel, "near plane n", -2, 0, 0.1, Math.log10(st.near), v => { st.near = 10 ** v; recompute(); }, v => (10 ** v).toPrecision(2));
  const bS = slider(panel, "depth-buffer bits", 4, 24, 1, st.bits, v => { st.bits = v; recompute(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const curveBox = htmlEl("div", "rs-curve", "", panel);
  const curve = new Plot(curveBox, { width: 330, height: 150, x: [0, FAR], y: [-1, 1], xlabel: "distance d", ylabel: "z_ndc", margin: { l: 58, b: 46, t: 8, r: 10 }, font: 16 });
  const cLine = curve.fn(d => ndcDepth(d, st.near, FAR), { width: 2.5 });
  const cDots = curve.scatter([], [], { r: 5, color: "var(--qb-c2)" });
  htmlEl("div", "qb-note", "Drag a vertex (select it to change its depth w); click a pixel to read what happened there.", panel);

  function load(p: Preset): void {
    st = p.make(); sel = null;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    cullC.checked = st.cull; tlC.checked = st.topLeft; pcC.checked = st.persp;
    nS.set(Math.log10(st.near)); bS.set(st.bits);
    recompute(true);
  }

  function setRes(i: number): void {
    const f = RES[i][0] / RES[st.res][0];
    st.tris.forEach(t => t.p.forEach(q => { q[0] *= f; q[1] *= f; }));
    st.res = i; recompute(true);
  }

  // ---- rasterise ----
  function winTris(): WinVertex[][] {
    return st.tris.map(t => t.p.map((q, v) => ({
      x: q[0], y: q[1], w: t.w[v], z: ndcDepth(t.w[v], st.near, FAR),
      attr: [...t.col[v], ...(t.uv ? t.uv[v] : [0, 0]), t.uv ? 1 : 0],
    })));
  }
  const shader: FragmentShader = a => {
    if (a[5] > 0.5) {
      const c = (Math.floor(a[3] * 8) + Math.floor(a[4] * 8)) % 2 ? 0.92 : 0.22;
      return [c, c, c * 0.9 + 0.08];
    }
    return [a[0], a[1], a[2]];
  };
  function recompute(all = false): void {
    const [W, H] = RES[st.res];
    const r = rasteriseSteps(winTris(), W, H, { cull: st.cull, topLeft: st.topLeft, perspective: st.persp, depthBits: st.bits >= 24 ? undefined : st.bits }, () => shader);
    steps = r.steps; culled = r.culled;
    if (all || k > steps.length) k = steps.length;
    draw();
  }
  function step(n: number): void { if (k >= steps.length) k = 0; k = Math.min(steps.length, k + n); pickedAt = null; draw(); }
  function stepRow1(): void {
    if (k >= steps.length) k = 0;
    const s0 = steps[k];
    let n = k;
    while (n < steps.length && steps[n].j === s0.j && steps[n].tri === s0.tri) n++;
    k = n; pickedAt = null;
    draw();
  }

  // ---- drawing ----
  let cell = 10;
  function draw(): void {
    const [W, H] = RES[st.res];
    viewBtns.forEach((b, i) => b.classList.toggle("active", (["colour", "depth", "bary", "edge"] as View[])[i] === st.view));
    resBtns.forEach((b, i) => b.classList.toggle("active", i === st.res));
    // replay the first k fragments
    const col = new Float32Array(3 * W * H).fill(0.13), dep = new Float32Array(W * H).fill(Infinity);
    const bary = new Float32Array(3 * W * H), cover = new Uint8Array(W * H), drawn = [0, 0];
    for (let s = 0; s < k; s++) {
      const f = steps[s], p = f.j * W + f.i;
      if (f.inside) cover[p]++;
      if (f.passed && f.color) {
        col.set(f.color, 3 * p); dep[p] = f.z; bary.set(st.persp ? f.baryP : f.bary, 3 * p); drawn[f.tri]++;
      }
    }
    const cur = k > 0 ? steps[k - 1] : null;
    const curTri = cur ? cur.tri : 0;
    // canvas size
    const cssW = cv.clientWidth || 600, dpr = window.devicePixelRatio || 1;
    cell = cssW / W;
    cv.style.aspectRatio = `${W} / ${H}`;
    cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssW * H / W * dpr);
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = (x: number) => x * cell, Y = (y: number) => (H - y) * cell;
    const wt = winTris();
    const eTri = wt[Math.min(curTri, wt.length - 1)];
    let eMax = 1;
    if (st.view === "edge" && eTri) for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) eMax = Math.max(eMax, Math.abs(edge(eTri[0], eTri[1], i + 0.5, j + 0.5)));
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const p = j * W + i;
      let c: [number, number, number];
      if (st.view === "colour") c = [col[3 * p], col[3 * p + 1], col[3 * p + 2]];
      else if (st.view === "depth") c = dep[p] === Infinity ? [0.1, 0.12, 0.2] : grey(1 - (dep[p] + 1) / 2);
      else if (st.view === "bary") c = dep[p] === Infinity ? [0.13, 0.13, 0.13] : [bary[3 * p], bary[3 * p + 1], bary[3 * p + 2]];
      else c = eTri ? diverging(0.5 + 0.5 * edge(eTri[0], eTri[1], i + 0.5, j + 0.5) / eMax) : [0.5, 0.5, 0.5];
      g.fillStyle = `rgb(${255 * c[0]},${255 * c[1]},${255 * c[2]})`;
      g.fillRect(X(i), Y(j + 1), cell + 0.5, cell + 0.5);
      if (st.view === "edge" && eTri && cell >= 34) {
        g.fillStyle = "#000"; g.font = `${Math.min(15, cell * 0.36)}px system-ui`; g.textAlign = "center"; g.textBaseline = "middle";
        g.fillText(String(Math.round(edge(eTri[0], eTri[1], i + 0.5, j + 0.5) * 10) / 10), X(i + 0.5), Y(j + 0.5));
      }
    }
    // grid lines and pixel centres
    if (cell >= 9) {
      g.strokeStyle = "rgba(255,255,255,0.18)"; g.lineWidth = 1; g.beginPath();
      for (let i = 0; i <= W; i++) { g.moveTo(X(i), 0); g.lineTo(X(i), Y(0)); }
      for (let j = 0; j <= H; j++) { g.moveTo(0, Y(j)); g.lineTo(X(W), Y(j)); }
      g.stroke();
      if (cell >= 18 && st.view !== "edge") {
        g.fillStyle = "rgba(255,255,255,0.35)";
        for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) { g.beginPath(); g.arc(X(i + 0.5), Y(j + 0.5), 1.6, 0, 7); g.fill(); }
      }
    }
    // pixels covered twice (only interesting for shared edges)
    for (let p = 0; p < W * H; p++) if (cover[p] > 1) {
      const i = p % W, j = Math.floor(p / W);
      g.strokeStyle = "#ff3df5"; g.lineWidth = 3; g.strokeRect(X(i) + 2, Y(j + 1) + 2, cell - 4, cell - 4);
    }
    // triangle outlines and vertices
    st.tris.forEach((t, ti) => {
      g.strokeStyle = culled[ti] ? "rgba(255,255,255,0.45)" : "#fff"; g.lineWidth = 2;
      g.setLineDash(culled[ti] ? [6, 5] : []);
      g.beginPath(); t.p.forEach((q, v) => (v ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1])))); g.closePath(); g.stroke();
      g.setLineDash([]);
      t.p.forEach((q, v) => {
        const isSel = sel && sel[0] === ti && sel[1] === v;
        g.beginPath(); g.arc(X(q[0]), Y(q[1]), isSel ? 9 : 7, 0, 7);
        g.fillStyle = `rgb(${t.col[v].map(c => 255 * c).join(",")})`; g.fill();
        g.lineWidth = isSel ? 3 : 2; g.strokeStyle = isSel ? "#ffe600" : "#000"; g.stroke();
        const lab = (ti ? ["a'", "b'", "c'"] : ["a", "b", "c"])[v];
        g.font = "bold 18px system-ui"; g.textAlign = "left"; g.textBaseline = "bottom";
        g.lineWidth = 4; g.strokeStyle = "#000"; g.strokeText(lab, X(q[0]) + 8, Y(q[1]) - 4);
        g.fillStyle = "#fff"; g.fillText(lab, X(q[0]) + 8, Y(q[1]) - 4);
      });
    });
    // current pixel
    if (cur && k < steps.length) {
      g.strokeStyle = "#ffe600"; g.lineWidth = 3; g.strokeRect(X(cur.i), Y(cur.j + 1), cell, cell);
      g.beginPath(); g.arc(X(cur.i + 0.5), Y(cur.j + 0.5), 3, 0, 7); g.fillStyle = "#ffe600"; g.fill();
    } else if (cur && k === steps.length && pickedAt !== null) {
      const f = steps[pickedAt];
      g.strokeStyle = "#ffe600"; g.lineWidth = 3; g.strokeRect(X(f.i), Y(f.j + 1), cell, cell);
    }
    // readout
    const lines: string[] = [];
    const fr = pickedAt !== null && k === steps.length ? steps[pickedAt] : cur && k < steps.length ? cur : null;
    if (fr) {
      const tn = fr.tri ? "2 (a'b'c')" : "1 (abc)";
      lines.push(`triangle ${tn}, pixel (${fr.i}, ${fr.j}), centre (${fr.i + 0.5}, ${fr.j + 0.5})`);
      lines.push(`E_bc ${f2(fr.E[0])}  E_ca ${f2(fr.E[1])}  E_ab ${f2(fr.E[2])}`);
      lines.push(fr.inside ? (fr.onEdge ? "inside (on an edge: kept by the top-left rule)" : "inside") : (fr.onEdge && fr.E.every(e => e >= 0) ? "on an edge: left to the neighbour (top-left rule)" : "outside"));
      if (fr.inside) {
        lines.push(`bary (screen)  ${fr.bary.map(f3).join("  ")}`);
        lines.push(`bary (3D)      ${fr.baryP.map(f3).join("  ")}`);
        lines.push(`z_ndc ${f3(fr.z)}, stored ${fr.zBefore === Infinity ? "empty" : f3(fr.zBefore)} → ${fr.passed ? "drawn" : "hidden"}`);
      }
    } else lines.push(`${steps.length} pixel visits (bounding boxes)`);
    lines.push(`step ${k} / ${steps.length}; drawn: ${st.tris.map((_, i) => (culled[i] ? "culled" : drawn[i])).join(" + ")}`);
    let twice = 0; for (let p = 0; p < W * H; p++) if (cover[p] > 1) twice++;
    if (st.tris.length > 1) lines.push(`pixels covered by both: ${twice}`);
    ro.textContent = lines.join("\n");
    // depth curve
    curveBox.style.display = st.view === "depth" || st.near < 0.5 || st.bits < 24 ? "" : "none";
    cLine.set(d => ndcDepth(d, st.near, FAR));
    const ds = st.tris.flatMap(t => t.w);
    cDots.set(ds, ds.map(d => ndcDepth(d, st.near, FAR)));
    wS.input.disabled = !sel;
    if (sel) wS.set(st.tris[sel[0]].w[sel[1]]);
  }

  // ---- pointer ----
  let pickedAt: number | null = null;
  const toGrid = (ev: PointerEvent): [number, number] => {
    const r = cv.getBoundingClientRect(), [W, H] = RES[st.res];
    return [(ev.clientX - r.left) / r.width * W, H - (ev.clientY - r.top) / r.height * H];
  };
  let drag: [number, number] | null = null;
  cv.addEventListener("pointerdown", ev => {
    const [x, y] = toGrid(ev), [W] = RES[st.res], tol = 14 / (cv.getBoundingClientRect().width / W);
    let best: [number, number] | null = null, bd = tol;
    st.tris.forEach((t, ti) => t.p.forEach((q, v) => { const d = Math.hypot(q[0] - x, q[1] - y); if (d < bd) { bd = d; best = [ti, v]; } }));
    if (best) {
      sel = best; drag = best; pickedAt = null;
      cv.setPointerCapture(ev.pointerId); ev.preventDefault(); draw();
      return;
    }
    // pick a pixel: the last visit of that pixel
    const i = Math.floor(x), j = Math.floor(y);
    pickedAt = null;
    for (let s = steps.length - 1; s >= 0; s--) if (steps[s].i === i && steps[s].j === j && steps[s].inside) { pickedAt = s; break; }
    if (pickedAt === null) for (let s = steps.length - 1; s >= 0; s--) if (steps[s].i === i && steps[s].j === j) { pickedAt = s; break; }
    k = steps.length; draw();
  });
  cv.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [x, y] = toGrid(ev), [W, H] = RES[st.res];
    const q = st.tris[drag[0]].p[drag[1]];
    q[0] = Math.max(0, Math.min(W, Math.round(x * 2) / 2)); q[1] = Math.max(0, Math.min(H, Math.round(y * 2) / 2));
    recompute(true);
  });
  const up = () => { drag = null; };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe(cv);

  load(p0);
  if (host.dataset.view) { st.view = host.dataset.view as View; draw(); }
}

autoMount(NAME, host => { injectStyle(); mount(host); });
