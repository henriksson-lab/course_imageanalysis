// Epipolar geometry (lib/camera.ts, lib/twoview.ts). A synthetic scene (points on a house, a
// ground grid and three posts) seen by two cameras (640 × 480, f = 700 px). Click a point in
// the left image: its epipolar line in the right image, with the true match on it; the
// epipoles are marked (or an arrow when they lie outside the image); a small 3D view (WebGL2)
// shows the cameras, the points and the epipolar plane. Sliders: baseline length, its
// direction (0° sideways … 90° forward) and the rotation of camera 2. "Estimate F": the matches
// get seeded pixel noise and a fraction of outliers; the eight-point algorithm (unnormalised or
// normalised) or RANSAC + eight-point estimates F, whose epipolar line is drawn dashed.
//
// <div data-demo="epipolar" data-preset="sideways|forward|general|rotation|planar|unnormalised|outliers|ransac"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { GLView, OrbitCamera } from "../lib/gl3d";
import { Camera, makeK, lookAtCamera, project, cameraCentre, V3, V2 } from "../lib/camera";
import { fundamentalFromCameras, eightPoint, ransacF, epipolarLine, epipoles, pointLineDistance, symmetricDistance } from "../lib/twoview";
import { rng } from "../lib/random";
import { injectStyle } from "../triangulate/flatview";

const NAME = "epipolar";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { flex-wrap: nowrap; gap: .7em; }
${SEL} .ep-main { flex: 0 1 760px; min-width: 300px; display: flex; flex-direction: column; gap: .3em; }
${SEL} .ep-imgs { display: grid; grid-template-columns: 1fr 1fr; gap: .5em; }
${SEL} .ep-imgs figure { margin: 0; }
${SEL} .ep-imgs figcaption { font-size: 17px; text-align: center; }
${SEL} .ep-imgs svg { width: 100%; display: block; border-radius: 4px; }
${SEL} .ep-low { display: grid; grid-template-columns: 1fr 1fr; gap: .5em; align-items: start; }
${SEL} .qb-panel { flex: 1 1 260px; max-width: 320px; gap: .25em; }
${SEL} .qb-readout { font-size: 14px; }
${SEL} .ep-row { display: flex; flex-wrap: wrap; gap: .25em; align-items: center; }
.reveal ${SEL} { font-size: 17px; }
.reveal ${SEL} .ep-main { flex: 0 0 760px; }
.reveal ${SEL} .qb-btn { font-size: 15px; padding: .04em .32em; }
.reveal ${SEL} .qb-readout { font-size: 15px; padding: .2em .4em; }
@media (max-width: 860px) { ${SEL} { flex-wrap: wrap; } }
`;

const W = 640, H = 480, K = makeK(700, 320, 240);
type Method = "unnorm" | "norm" | "ransac";
interface State { B: number; dir: number; rot: number; estimate: boolean; sigma: number; outl: number; method: Method; planar: boolean; pick: number }
const DEFAULT: State = { B: 1.2, dir: 20, rot: -10, estimate: false, sigma: 0.5, outl: 0, method: "norm", planar: false, pick: 4 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "sideways", label: "sideways: parallel lines", s: { dir: 0, rot: 0 } },
  { id: "forward", label: "forward: focus of expansion", s: { dir: 90, rot: 0, B: 1.5 } },
  { id: "general", label: "general motion", s: {} },
  { id: "rotation", label: "pure rotation: no geometry", s: { B: 0, rot: -15 } },
  { id: "planar", label: "planar scene: degenerate", s: { planar: true, estimate: true, sigma: 0.2 } },
  { id: "unnormalised", label: "no normalisation", s: { estimate: true, method: "unnorm", sigma: 1 } },
  { id: "outliers", label: "20 % outliers break least squares", s: { estimate: true, outl: 0.2 } },
  { id: "ransac", label: "RANSAC repairs it", s: { estimate: true, outl: 0.2, method: "ransac" } },
];

function scenePoints(planar: boolean): V3[] {
  const pts: V3[] = [];
  const g = planar ? 7 : 5;
  for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) pts.push([-1.5 + 3 * i / (g - 1), 0, -1.5 + 3 * j / (g - 1)]);
  if (planar) return pts;
  // house: box corners and edge midpoints, roof ridge
  const xs = [-0.6, 0, 0.6], ys = [0, 0.4, 0.8], zs = [-0.5, 0.5];
  for (const x of xs) for (const y of ys) for (const z of zs) pts.push([x, y, z]);
  for (const x of xs) pts.push([x, 1.2, 0]);
  for (const z of [-0.25, 0.25]) for (const x of [-0.3, 0.3]) pts.push([x, 1.0, z]);
  for (const [x, z] of [[-1.2, 0.9], [1.1, -1.0], [1.3, 1.1]]) for (const y of [0.3, 0.6, 0.9, 1.2]) pts.push([x, y, z]);
  return pts;
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const main = htmlEl("div", "ep-main", "", host);
  const imgs = htmlEl("div", "ep-imgs", "", main);
  const mk = (cap: string) => { const f = htmlEl("figure", "", "", imgs); htmlEl("figcaption", "", cap, f); const s = svgEl("svg", { viewBox: `-30 -30 ${W + 60} ${H + 60}` }, f); return { s, g: svgEl("g", {}, s) }; };
  const L = mk("image 1 (click a point)"), R = mk("image 2");
  const low = htmlEl("div", "ep-low", "", main);
  const glDiv = htmlEl("div", "", "", low);
  const orbit = new OrbitCamera({ distance: 10, azimuth: 0.7, elevation: 0.6, target: [0, 0.6, 2], far: 200 });
  const view = new GLView(glDiv, { aspect: 4 / 3, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: orbit, label: "3D (drag)" }], fallback: "The two images still work." });
  const gPts = view.addPoints({ positions: [] }, { size: 7, color: [0.2, 0.2, 0.25] });
  const gCams = view.addLines({ positions: [] }, { width: 2.5, color: [0.12, 0.47, 0.71] });
  const gPlane = view.addMesh({ positions: [] }, { color: [0.95, 0.75, 0.1], lighting: "unlit", opacity: 0.35 });
  const gSel = view.addPoints({ positions: [] }, { size: 13, color: [0.9, 0.25, 0.15] });
  const ro = htmlEl("div", "qb-readout", "", low);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  const bS = slider(panel, "baseline", 0, 3, 0.1, st.B, v => { st.B = v; rebuild(); });
  const dS = slider(panel, "direction°", 0, 90, 5, st.dir, v => { st.dir = v; rebuild(); });
  const rS = slider(panel, "rotation°", -30, 30, 1, st.rot, v => { st.rot = v; rebuild(); });
  const eC = checkbox(panel, "estimate F from noisy matches", st.estimate, v => { st.estimate = v; rebuild(); });
  const sS = slider(panel, "noise σ (px)", 0, 3, 0.1, st.sigma, v => { st.sigma = v; rebuild(); });
  const oS = slider(panel, "outliers", 0, 0.5, 0.05, st.outl, v => { st.outl = v; rebuild(); }, v => `${Math.round(100 * v)} %`);
  const mRow = htmlEl("div", "ep-row", "", panel);
  const mBtns = (["unnorm", "norm", "ransac"] as Method[]).map(m => button(mRow, m === "unnorm" ? "8-point" : m === "norm" ? "8-point normalised" : "RANSAC + 8-point", () => { st.method = m; rebuild(); }));
  const plC = checkbox(panel, "planar scene (ground only)", st.planar, v => { st.planar = v; rebuild(); });
  button(panel, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); pBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });
  function sync(): void { bS.set(st.B); dS.set(st.dir); rS.set(st.rot); eC.checked = st.estimate; sS.set(st.sigma); oS.set(st.outl); plC.checked = st.planar; }

  let c1: Camera, c2: Camera, X: V3[] = [], x1: V2[] = [], x2: V2[] = [], m1: V2[] = [], m2: V2[] = [], isOut: boolean[] = [], Ftrue: Float64Array | null = null, Fest: Float64Array | null = null, info = "", inl = new Set<number>();
  function rebuild(): void {
    const E1: V3 = [0.3, 1.5, 4.0], T: V3 = [0, 0.45, 0];
    c1 = lookAtCamera(K, E1, T);
    const xa = [c1.R[0], c1.R[1], c1.R[2]], za = [c1.R[6], c1.R[7], c1.R[8]], a = st.dir * Math.PI / 180;
    const E2: V3 = [0, 1, 2].map(k => E1[k] + st.B * (Math.cos(a) * xa[k] + Math.sin(a) * za[k]));
    const ro_ = st.rot * Math.PI / 180, d = [T[0] - E1[0], T[1] - E1[1], T[2] - E1[2]];
    const dr: V3 = [Math.cos(ro_) * d[0] + Math.sin(ro_) * d[2], d[1], -Math.sin(ro_) * d[0] + Math.cos(ro_) * d[2]];
    c2 = lookAtCamera(K, E2, [E2[0] + dr[0], E2[1] + dr[1], E2[2] + dr[2]]);
    Ftrue = st.B > 1e-6 ? fundamentalFromCameras(c1, c2) : null;
    X = scenePoints(st.planar).filter(P => { const p = project(c1, P), q = project(c2, P); return p.Z > 0 && q.Z > 0 && p.u > 0 && p.u < W && p.v > 0 && p.v < H && q.u > 0 && q.u < W && q.v > 0 && q.v < H; });
    x1 = X.map(P => { const p = project(c1, P); return [p.u, p.v] as V2; });
    x2 = X.map(P => { const p = project(c2, P); return [p.u, p.v] as V2; });
    st.pick = Math.min(st.pick, X.length - 1);
    // noisy matches with outliers
    const r = rng(11);
    m1 = x1.map(p => [p[0] + st.sigma * r.norm(), p[1] + st.sigma * r.norm()] as V2);
    m2 = x2.map(p => [p[0] + st.sigma * r.norm(), p[1] + st.sigma * r.norm()] as V2);
    const ro2 = rng(23);
    isOut = X.map(() => ro2.unif() < st.outl);
    isOut.forEach((o, i) => { if (o) m2[i] = [20 + ro2.unif() * (W - 40), 20 + ro2.unif() * (H - 40)]; });
    Fest = null; info = ""; inl = new Set();
    if (st.estimate) {
      if (st.method === "ransac") {
        const res = ransacF(m1, m2, 2, 1177, 1);
        Fest = res.F; res.inliers.forEach(i => inl.add(i));
        info = `RANSAC: ${res.inliers.length} of ${X.length} inliers (within 2 px), best sample at iteration ${res.foundAt} of 1177`;
      } else {
        const e = eightPoint(m1, m2, st.method === "norm");
        if (e) {
          Fest = e.F;
          const S = e.SA;
          info = `singular values of A (smallest 3): ${Array.from(S.slice(6)).map(v => v.toExponential(1)).join(", ")}\nof F before rank 2: ${Array.from(e.SF).map(v => v.toExponential(2)).join(", ")}`;
        }
      }
    }
    draw();
  }

  function clickPick(ev: PointerEvent): void {
    const pt = L.s.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const q = pt.matrixTransform(L.s.getScreenCTM()!.inverse());
    let best = -1, bd = 40;
    x1.forEach((p, i) => { const d = Math.hypot(p[0] - q.x, p[1] - q.y); if (d < bd) { bd = d; best = i; } });
    if (best >= 0) { st.pick = best; draw(); }
  }
  L.s.addEventListener("pointerdown", clickPick);

  function lineInImage(g: SVGGElement, l: V3, color: string, width: number, dash = ""): void {
    // intersect a u + b v + c = 0 with the image rectangle
    const pts: V2[] = [];
    const [a, b, c] = l;
    if (Math.abs(b) > 1e-12) { for (const u of [0, W]) { const v = -(a * u + c) / b; if (v >= 0 && v <= H) pts.push([u, v]); } }
    if (Math.abs(a) > 1e-12) { for (const v of [0, H]) { const u = -(b * v + c) / a; if (u >= 0 && u <= W) pts.push([u, v]); } }
    if (pts.length < 2) return;
    const ln = svgEl("line", { x1: pts[0][0], y1: pts[0][1], x2: pts[1][0], y2: pts[1][1], "stroke-width": width, "stroke-linecap": "round" }, g);
    ln.style.stroke = color;
    if (dash) ln.setAttribute("stroke-dasharray", dash);
  }
  function epipoleMark(g: SVGGElement, e: V3, label: string): string {
    if (Math.abs(e[2]) < 1e-9 * Math.hypot(e[0], e[1])) {
      // at infinity: an arrow along its direction at the image centre
      const dx = e[0], dy = e[1], n = Math.hypot(dx, dy), ux = dx / n, uy = dy / n;
      const a = svgEl("path", { d: `M${W / 2 - 60 * ux},${H / 2 - 60 * uy}L${W / 2 + 60 * ux},${H / 2 + 60 * uy}`, "stroke-width": 4, "marker-end": "" }, g); a.style.stroke = "#9467bd";
      return `${label} at infinity (direction ${Math.round(Math.atan2(uy, ux) * 180 / Math.PI)}°)`;
    }
    const u = e[0] / e[2], v = e[1] / e[2];
    if (u > -25 && u < W + 25 && v > -25 && v < H + 25) {
      const c = svgEl("circle", { cx: u, cy: v, r: 10, fill: "none", "stroke-width": 3.5 }, g); c.style.stroke = "#9467bd";
    } else {
      const cu = Math.max(-20, Math.min(W + 20, u)), cv = Math.max(-20, Math.min(H + 20, v));
      const dx = u - W / 2, dy = v - H / 2, n = Math.hypot(dx, dy);
      const a = svgEl("path", { d: `M${cu - 40 * dx / n},${cv - 40 * dy / n}L${cu},${cv}`, "stroke-width": 5 }, g); a.style.stroke = "#9467bd";
    }
    return `${label} at (${u.toFixed(0)}, ${v.toFixed(0)})`;
  }

  function draw(): void {
    mBtns.forEach((b, i) => b.classList.toggle("active", ["unnorm", "norm", "ransac"][i] === st.method));
    for (const P of [L, R]) {
      P.g.replaceChildren();
      const bg = svgEl("rect", { x: 0, y: 0, width: W, height: H, "stroke-width": 2 }, P.g);
      bg.style.fill = "#f4f6f9"; bg.style.stroke = "#666";
    }
    const k = st.pick, txt: string[] = [];
    // all points
    x1.forEach((p, i) => { const c = svgEl("circle", { cx: (st.estimate ? m1 : x1)[i][0], cy: (st.estimate ? m1 : x1)[i][1], r: i === k ? 9 : 6 }, L.g); c.style.fill = i === k ? "#e8402a" : "#555"; });
    (st.estimate ? m2 : x2).forEach((p, i) => {
      const out = st.estimate && isOut[i], ransacOut = st.method === "ransac" && st.estimate && !inl.has(i);
      if (out || ransacOut) { const c = svgEl("path", { d: `M${p[0] - 6},${p[1] - 6}l12,12M${p[0] + 6},${p[1] - 6}l-12,12`, "stroke-width": 2.5 }, R.g); c.style.stroke = out ? "#e8402a" : "#ff9f1c"; }
      else { const c = svgEl("circle", { cx: p[0], cy: p[1], r: 6 }, R.g); c.style.fill = st.method === "ransac" && st.estimate ? "#2ca02c" : "#555"; }
    });
    // epipolar lines of the picked point
    if (Ftrue && k >= 0) {
      lineInImage(R.g, epipolarLine(Ftrue, x1[k]), "#2ca02c", 3.5);
      // a few more lines, faint, to show the pencil
      x1.forEach((p, i) => { if (i % 5 === 0 && i !== k) lineInImage(R.g, epipolarLine(Ftrue!, p), "rgba(44,160,44,0.3)", 1.5); });
      const { e1, e2 } = epipoles(Ftrue);
      txt.push(epipoleMark(L.g, e1, "epipole in image 1"), epipoleMark(R.g, e2, "epipole in image 2"));
    }
    if (Fest && k >= 0) {
      lineInImage(R.g, epipolarLine(Fest, m1[k]), "#e8402a", 3, "12 7");
      const good = X.map((_, i) => i).filter(i => !isOut[i]);
      const md = good.reduce((s, i) => s + pointLineDistance(epipolarLine(Fest!, x1[i]), x2[i]), 0) / Math.max(1, good.length);
      const sd = good.reduce((s, i) => s + symmetricDistance(Fest!, m1[i], m2[i]), 0) / Math.max(1, good.length);
      txt.push(`estimated F: true matches are ${md.toFixed(2)} px from their lines (noisy matches ${sd.toFixed(2)} px)`);
    }
    if (k >= 0) { const c = svgEl("circle", { cx: x2[k][0], cy: x2[k][1], r: 11, fill: "none", "stroke-width": 3 }, R.g); c.style.stroke = "#e8402a"; }
    if (!Ftrue) txt.push("no baseline: the cameras share a centre; no depth, no epipolar lines — the two images are related by a homography");
    // 3D
    const pos: number[] = [];
    X.forEach(P => pos.push(...P));
    gPts.setData({ positions: pos });
    const lines: number[] = [];
    for (const c of [c1, c2]) {
      const C = cameraCentre(c), Rt = [c.R[0], c.R[3], c.R[6], c.R[1], c.R[4], c.R[7], c.R[2], c.R[5], c.R[8]], s = 0.45;
      const corner = (u: number, v: number): V3 => { const xn = (u - 320) / 700, yn = (v - 240) / 700; return [0, 1, 2].map(i => C[i] + s * (Rt[i * 3] * xn + Rt[i * 3 + 1] * yn + Rt[i * 3 + 2])); };
      const cs = [corner(0, 0), corner(W, 0), corner(W, H), corner(0, H)];
      for (let q = 0; q < 4; q++) lines.push(...C, ...cs[q], ...cs[q], ...cs[(q + 1) % 4]);
    }
    gCams.setData({ positions: lines });
    if (k >= 0 && Ftrue) {
      const C1 = cameraCentre(c1), C2 = cameraCentre(c2), P = X[k];
      const ext = (A: V3, Bp: V3, f: number): V3 => [A[0] + f * (Bp[0] - A[0]), A[1] + f * (Bp[1] - A[1]), A[2] + f * (Bp[2] - A[2])];
      gPlane.setData({ positions: [...C1, ...C2, ...ext(C1, P, 1.3), ...C2, ...ext(C2, P, 1.3), ...ext(C1, P, 1.3)] });
      gSel.setData({ positions: P });
    } else { gPlane.setData({ positions: [] }); gSel.setData({ positions: X[k] ?? [] }); }
    const Fs = Fest ?? Ftrue;
    ro.textContent = txt.join("\n") + (info ? "\n" + info : "") +
      (Fs ? `\n${Fest ? "estimated" : "true"} F (unit norm):\n` + [0, 1, 2].map(i => [0, 1, 2].map(j => Fs[3 * i + j].toExponential(2).padStart(10)).join("")).join("\n") : "");
  }

  sync();
  if (p0) pBtns[PRESETS.indexOf(p0)].classList.add("active");
  rebuild();
}

autoMount(NAME, mount);
