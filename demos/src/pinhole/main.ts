// The pinhole camera (lib/camera.ts). Left: a 3D scene (a checkerboard floor, a box, three
// posts) seen from outside (WebGL2, lib/gl3d.ts; drag to orbit), with the camera drawn as a
// pyramid and, after a click in the image, the ray of that pixel. Right: the camera's image in
// pixels (640 × 480) with the principal point. Sliders: focal length f (px), principal point,
// skew, camera position (azimuth, elevation, distance) and roll, radial distortion k1, k2.
// Readouts: K, [R | t] and P = K [R | t]; or, with "graphics convention", the 4 × 4 view and
// projection matrices of the 3D graphics day for the same camera.
//
// <div data-demo="pinhole" data-preset="dolly|offcentre|barrel|pincushion|behind|telecentric"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { GLView, OrbitCamera, cubeMesh, MeshData } from "../lib/gl3d";
import { Camera, makeK, lookAtCamera, project, pixelRay, cameraMatrix, cameraCentre, glProjectionFromK, V3, toCamera } from "../lib/camera";
import { injectStyle } from "../triangulate/flatview";

const NAME = "pinhole";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { flex-wrap: nowrap; gap: .7em; }
${SEL} .ph-left { flex: 0 1 410px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
${SEL} .ph-mid { flex: 0 1 420px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
${SEL} .ph-mid svg { width: 100%; display: block; cursor: crosshair; border-radius: 4px; }
${SEL} .qb-panel { flex: 1 1 270px; max-width: 320px; gap: .15em; }
${SEL} .qb-readout { font-size: 14px; }
${SEL} .ph-cap { font-size: 16px; color: var(--qb-muted); }
${SEL} .qb-slider { gap: .3em; }
.reveal ${SEL} { font-size: 17px; }
.reveal ${SEL} .ph-left { flex: 0 0 400px; }
.reveal ${SEL} .ph-mid { flex: 0 0 410px; }
.reveal ${SEL} .qb-btn { font-size: 15px; padding: .04em .32em; }
.reveal ${SEL} .qb-readout { font-size: 15px; padding: .2em .4em; }
.reveal ${SEL} .ph-cap { font-size: 17px; }
@media (max-width: 860px) { ${SEL} { flex-wrap: wrap; } }
`;

const W = 640, H = 480, TARGET: V3 = [0, 0.25, 0];
interface State { f: number; cx: number; cy: number; skew: number; az: number; el: number; dist: number; roll: number; k1: number; k2: number; gl: boolean; behind: boolean; dolly: boolean; pick: [number, number] | null }
const DEFAULT: State = { f: 600, cx: 320, cy: 240, skew: 0, az: 35, el: 25, dist: 3, roll: 0, k1: 0, k2: 0, gl: false, behind: false, dolly: false, pick: [400, 300] };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "dolly", label: "long lens from far away (dolly zoom)", s: { f: 1500, dist: 7.5, dolly: true } },
  { id: "offcentre", label: "principal point off centre", s: { cx: 200, cy: 150 } },
  { id: "barrel", label: "barrel distortion", s: { k1: -0.35 } },
  { id: "pincushion", label: "pincushion", s: { k1: 0.3 } },
  { id: "behind", label: "a point behind the camera", s: { dist: 0.9, el: 8, az: 20, f: 300, behind: true, pick: null } },
  { id: "telecentric", label: "nearly orthographic (very long f, far away)", s: { f: 8000, dist: 40 } },
];

// ---- the scene (world: y up, metres) ----
interface Quad { pts: V3[]; fill: string }
function sceneQuads(): Quad[] {
  const q: Quad[] = [];
  const n = 8, s = 0.25;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x0 = -1 + i * s, z0 = -1 + j * s;
    q.push({ pts: [[x0, 0, z0], [x0 + s, 0, z0], [x0 + s, 0, z0 + s], [x0, 0, z0 + s]], fill: (i + j) % 2 ? "#2b2b2b" : "#e9e9e9" });
  }
  return q;
}
function boxQuads(min: V3, max: V3, colours: string[]): Quad[] {
  const [a, b, c] = min, [d, e, f] = max;
  return [
    { pts: [[a, b, f], [d, b, f], [d, e, f], [a, e, f]], fill: colours[0] }, // +z
    { pts: [[a, b, c], [a, e, c], [d, e, c], [d, b, c]], fill: colours[0] }, // -z
    { pts: [[d, b, c], [d, e, c], [d, e, f], [d, b, f]], fill: colours[1] }, // +x
    { pts: [[a, b, c], [a, b, f], [a, e, f], [a, e, c]], fill: colours[1] }, // -x
    { pts: [[a, e, c], [a, e, f], [d, e, f], [d, e, c]], fill: colours[2] }, // top
  ];
}
const BOX: [V3, V3] = [[0.15, 0, -0.65], [0.6, 0.45, -0.2]];
const POSTS: [number, number][] = [[-0.7, -0.6], [-0.55, 0.55], [0.7, 0.65]];
const OBJECTS: Quad[] = [
  ...boxQuads(BOX[0], BOX[1], ["#d98c3f", "#b86f2a", "#f0b070"]),
  ...POSTS.flatMap(([x, z]) => boxQuads([x - 0.04, 0, z - 0.04], [x + 0.04, 0.8, z + 0.04], ["#3b7dd8", "#2c62ad", "#6aa2ec"])),
];

function meshOf(quads: Quad[]): MeshData {
  const pos: number[] = [], col: number[] = [];
  for (const q of quads) {
    const c = [1, 3, 5].map(k => parseInt(q.fill.slice(k, k + 2), 16) / 255);
    for (const k of [0, 1, 2, 0, 2, 3]) { pos.push(...q.pts[k]); col.push(...c); }
  }
  return { positions: pos, colors: col };
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const left = htmlEl("div", "ph-left", "", host);
  const orbit = new OrbitCamera({ distance: 7, azimuth: 1.0, elevation: 0.55, target: [0, 0.3, 0], far: 200 });
  const view = new GLView(left, { aspect: 4 / 3, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: orbit, label: "the scene (drag to turn)" }], fallback: "The image on the right still works." });
  view.addMesh(meshOf(sceneQuads()), { lighting: "unlit", vertexColors: true });
  view.addMesh(meshOf(OBJECTS), { lighting: "lambert", vertexColors: true, ambient: 0.55, shading: "flat" });
  const gPyr = view.addLines({ positions: [] }, { width: 2.5, color: [0.85, 0.2, 0.6] });
  const gRay = view.addLines({ positions: [] }, { width: 3, color: [1, 0.75, 0] });
  const gPt = view.addPoints({ positions: [] }, { size: 12, color: [1, 0.75, 0] });
  const roL = htmlEl("div", "qb-readout", "", left);

  const mid = htmlEl("div", "ph-mid", "", host);
  const svg = svgEl("svg", { viewBox: `-20 -20 ${W + 40} ${H + 40}` }, mid);
  const gImg = svgEl("g", {}, svg);
  htmlEl("div", "ph-cap", "the image in pixels (u right, v down); click a pixel to see its ray", mid);
  const roM = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); draw(); } })));
  const fS = slider(panel, "f (px)", 100, 2000, 10, st.f, v => { if (st.dolly) { st.dist = st.dist * v / st.f; dS.set(st.dist); } st.f = v; draw(); });
  const cxS = slider(panel, "cx", 100, 540, 5, st.cx, v => { st.cx = v; draw(); });
  const cyS = slider(panel, "cy", 80, 400, 5, st.cy, v => { st.cy = v; draw(); });
  const skS = slider(panel, "skew", -300, 300, 10, st.skew, v => { st.skew = v; draw(); });
  const azS = slider(panel, "azimuth°", -180, 180, 1, st.az, v => { st.az = v; draw(); });
  const elS = slider(panel, "elevation°", 2, 85, 1, st.el, v => { st.el = v; draw(); });
  const dS = slider(panel, "distance", 0.5, 12, 0.1, st.dist, v => { st.dist = v; draw(); }, v => v.toFixed(1));
  const rS = slider(panel, "roll°", -45, 45, 1, st.roll, v => { st.roll = v; draw(); });
  const k1S = slider(panel, "k1", -0.5, 0.5, 0.05, st.k1, v => { st.k1 = v; draw(); });
  const k2S = slider(panel, "k2", -0.3, 0.3, 0.05, st.k2, v => { st.k2 = v; draw(); });
  const dollyC = checkbox(panel, "dolly: distance follows f", st.dolly, v => { st.dolly = v; });
  const behindC = checkbox(panel, "show points behind the camera (wrongly)", st.behind, v => { st.behind = v; draw(); });
  const glC = checkbox(panel, "graphics convention (4 × 4)", st.gl, v => { st.gl = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); pBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  function sync(): void {
    fS.set(st.f); cxS.set(st.cx); cyS.set(st.cy); skS.set(st.skew); azS.set(st.az); elS.set(st.el); dS.set(st.dist); rS.set(st.roll);
    k1S.set(st.k1); k2S.set(st.k2); dollyC.checked = st.dolly; behindC.checked = st.behind; glC.checked = st.gl;
  }

  svg.addEventListener("pointerdown", ev => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    if (q.x < 0 || q.y < 0 || q.x > W || q.y > H) return;
    st.pick = [Math.round(q.x), Math.round(q.y)];
    draw();
  });

  function camera(): Camera {
    const a = st.az * Math.PI / 180, e = st.el * Math.PI / 180;
    const eye: V3 = [TARGET[0] + st.dist * Math.cos(e) * Math.sin(a), TARGET[1] + st.dist * Math.sin(e), TARGET[2] + st.dist * Math.cos(e) * Math.cos(a)];
    return lookAtCamera(makeK(st.f, st.cx, st.cy, st.f, st.skew), eye, TARGET, st.roll * Math.PI / 180, st.k1, st.k2);
  }

  const fmt = (v: number, d = 3) => (Math.abs(v) < 5e-4 && d >= 3 ? "0" : v.toFixed(d)).padStart(9);
  function matStr(rows: number[][], d = 3): string { return rows.map(r => "[" + r.map(v => fmt(v, d)).join("") + " ]").join("\n"); }

  function draw(): void {
    const cam = camera();
    // ---- the image ----
    gImg.replaceChildren();
    const bg = svgEl("rect", { x: 0, y: 0, width: W, height: H, "stroke-width": 2 }, gImg);
    bg.style.fill = "#dfe8f2"; bg.style.stroke = "#555";
    const clip = svgEl("clipPath", { id: "ph-clip-" + uid }, gImg);
    svgEl("rect", { x: 0, y: 0, width: W, height: H }, clip);
    const gc = svgEl("g", { "clip-path": `url(#ph-clip-${uid})` }, gImg);
    const projPoly = (pts: V3[]): string | null => {
      let d = "";
      for (let k = 0; k < pts.length; k++) {
        const a = pts[k], b = pts[(k + 1) % pts.length];
        for (let s = 0; s < 6; s++) {
          const t = s / 6, P = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
          const q = project(cam, P);
          if (q.Z <= 0.02 || !Number.isFinite(q.u) || Math.abs(q.u) > 1e5 || Math.abs(q.v) > 1e5) return null;
          d += (d ? "L" : "M") + q.u.toFixed(1) + "," + q.v.toFixed(1);
        }
      }
      return d + "Z";
    };
    let hidden = 0;
    for (const q of sceneQuads()) { const d = projPoly(q.pts); if (d) { const p = svgEl("path", { d }, gc); p.style.fill = q.fill; } else hidden++; }
    const depth = (q: Quad) => q.pts.reduce((s, p) => s + toCamera(cam, p)[2], 0) / q.pts.length;
    // painter's order, and back faces dropped
    const objs = OBJECTS.map(q => ({ q, z: depth(q) })).sort((a, b) => b.z - a.z);
    for (const { q } of objs) {
      const d = projPoly(q.pts);
      if (!d) { hidden++; continue; }
      const p = svgEl("path", { d, "stroke-width": 1 }, gc);
      p.style.fill = q.fill; p.style.stroke = "rgba(0,0,0,.35)";
    }
    // points behind the camera, projected anyway (the division by a negative Z flips them)
    let nBehind = 0;
    const corners: V3[] = [...OBJECTS.flatMap(q => q.pts), ...sceneQuads().flatMap(q => q.pts)];
    for (const P of corners) {
      const q = project(cam, P);
      if (q.Z > 0) continue;
      nBehind++;
      if (st.behind && Math.abs(q.u) < 1e4) { const c = svgEl("path", { d: `M${q.u - 7},${q.v - 7}l14,14M${q.u + 7},${q.v - 7}l-14,14`, "stroke-width": 2.5 }, gc); c.style.stroke = "#e8402a"; }
    }
    const pp = svgEl("circle", { cx: st.cx, cy: st.cy, r: 7, fill: "none", "stroke-width": 2.5 }, gImg);
    pp.style.stroke = "#d62728";
    const ppl = svgEl("text", { x: st.cx + 10, y: st.cy - 10, "font-size": 24, class: "qb-halo" }, gImg); ppl.textContent = "(cx, cy)";
    // ---- the pyramid and the ray in 3D ----
    const C = cameraCentre(cam), sc = Math.min(0.6, 0.25 * st.dist);
    const cornerRay = (u: number, v: number): V3 => { const r = pixelRay({ ...cam, k1: 0, k2: 0 }, u, v); const Rz = [cam.R[6], cam.R[7], cam.R[8]]; const z = r.dir[0] * Rz[0] + r.dir[1] * Rz[1] + r.dir[2] * Rz[2]; return [C[0] + sc * r.dir[0] / z, C[1] + sc * r.dir[1] / z, C[2] + sc * r.dir[2] / z]; };
    const cs = [cornerRay(0, 0), cornerRay(W, 0), cornerRay(W, H), cornerRay(0, H)];
    const pyr: number[] = [];
    for (let k = 0; k < 4; k++) pyr.push(...C, ...cs[k], ...cs[k], ...cs[(k + 1) % 4]);
    const up = cornerRay(W / 2, -H * 0.25);
    pyr.push(...cs[0], ...up, ...up, ...cs[1]);
    gPyr.setData({ positions: pyr });
    let floorTxt = "";
    if (st.pick) {
      const [u, v] = st.pick, r = pixelRay(cam, u, v);
      const tFloor = r.dir[1] < -1e-9 ? -C[1] / r.dir[1] : Infinity;
      const L = Number.isFinite(tFloor) ? Math.min(tFloor, 60) : 30;
      gRay.setData({ positions: [...C, C[0] + L * r.dir[0], C[1] + L * r.dir[1], C[2] + L * r.dir[2]] });
      if (Number.isFinite(tFloor)) {
        const X = [C[0] + tFloor * r.dir[0], C[1] + tFloor * r.dir[1], C[2] + tFloor * r.dir[2]];
        gPt.setData({ positions: X });
        floorTxt = `\nthe ray meets the floor at (${X.map(x => x.toFixed(2)).join(", ")})`;
      } else gPt.setData({ positions: [] });
      const m = svgEl("path", { d: `M${u - 10},${v}h20M${u},${v - 10}v20`, "stroke-width": 3 }, gImg); m.style.stroke = "#f2b705";
      const Ki = [(u - st.cx - st.skew * (v - st.cy) / st.f) / st.f, (v - st.cy) / st.f];
      roM.textContent = `pixel (${u}, ${v}) → K⁻¹(u, v, 1) = (${Ki[0].toFixed(3)}, ${Ki[1].toFixed(3)}, 1)` +
        `\nray direction (world): (${r.dir.map(x => x.toFixed(3)).join(", ")})` + floorTxt;
    } else { gRay.setData({ positions: [] }); gPt.setData({ positions: [] }); roM.textContent = "click a pixel"; }
    if (nBehind) roM.textContent += `\n${nBehind} corners are behind the camera (Z < 0)${st.behind ? ": red ×, wrongly projected" : ""}`;
    // ---- matrices ----
    const P = cameraMatrix(cam), R = cam.R, t = cam.t;
    const hfov = 2 * Math.atan(W / 2 / st.f) * 180 / Math.PI;
    if (!st.gl) {
      roL.textContent =
        `K =\n${matStr([[st.f, st.skew, st.cx], [0, st.f, st.cy], [0, 0, 1]], 1)}\n[R | t] =\n` +
        matStr([0, 1, 2].map(i => [R[3 * i], R[3 * i + 1], R[3 * i + 2], t[i]])) +
        `\nP = K[R | t] =\n${matStr([0, 1, 2].map(i => [P[4 * i], P[4 * i + 1], P[4 * i + 2], P[4 * i + 3]]), 1)}` +
        `\ncentre c = (${C.map(v => v.toFixed(2)).join(", ")}), horizontal field of view ${hfov.toFixed(1)}°`;
    } else {
      const G = glProjectionFromK([st.f, st.skew, st.cx, 0, st.f, st.cy, 0, 0, 1], W, H, 0.1, 100);
      const V = [0, 1, 2].map(i => [R[3 * i], R[3 * i + 1], R[3 * i + 2], t[i]].map(x => (i === 0 ? x : -x)));
      roL.textContent =
        `graphics view matrix V = diag(1, −1, −1)·[R | t]:\n${matStr([...V, [0, 0, 0, 1]])}\n` +
        `projection P_GL (near 0.1, far 100):\n${matStr([0, 1, 2, 3].map(i => [G[4 * i], G[4 * i + 1], G[4 * i + 2], G[4 * i + 3]]))}` +
        `\nits first two rows are K's, rescaled to −1 … 1 (NDC)`;
    }
  }
  const uid = Math.floor(Math.random() * 1e9).toString(36);

  sync();
  if (p0) pBtns[PRESETS.indexOf(p0)].classList.add("active");
  draw();
  void cubeMesh;
}

autoMount(NAME, mount);
