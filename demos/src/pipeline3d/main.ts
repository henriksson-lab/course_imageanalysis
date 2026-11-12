// pipeline3d: follow a small house through model → world → view → clip/NDC → screen
// (3D-graphics day, unit `pipeline3d/`). One WebGL2 canvas (lib/gl3d.ts) with two viewports:
// left, a "god's eye" view of the world with the object, the camera (a pyramid) and its view
// frustum; right, the object in the selected space: model, world, view (the camera at the
// origin looking down −z), clip → NDC (the frustum has become the cube [−1, 1]³), or screen
// (the final image on a coarse pixel grid, drawn by the software rasteriser lib/raster.ts).
// Under it the matrices M, V, P with live numbers. Click a vertex (in either view) to see its
// coordinates in every space, including w before the divide. Drag to rotate either view.
// Auto-mounts on <div data-demo="pipeline3d">. Optional: data-space="model|world|view|ndc|
// screen", data-preset="ortho|dolly|near|behind|scale".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { GLView, OrbitCamera, axesLines, boxLines, LineData } from "../lib/gl3d";
import * as M from "../lib/mat4";
import type { Mat4, Vec3 } from "../lib/mat4";
import { createTarget, drawClipTriangle, ClipVertex } from "../lib/raster";

import { HV, HT, houseMesh, State, DEF, PRESETS, matrices, Space, SCREEN } from "./scene";

const NAME = "pipeline3d";

const STYLE = `
.qb-demo[data-demo="${NAME}"] .p3-left { flex: 1 1 600px; min-width: 300px; max-width: 760px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 400px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .p3-mats { display: flex; flex-wrap: wrap; gap: .3em .8em; font-family: ui-monospace, monospace; font-size: 14px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .p3-mats div { background: var(--qb-panel); border-radius: 5px; padding: .2em .45em; }
.qb-demo[data-demo="${NAME}"] .p3-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] canvas.p3-screen { position: absolute; top: 0; height: 100%; image-rendering: pixelated; pointer-events: none; }
.qb-demo[data-demo="${NAME}"] .qb-slider { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .p3-left { flex: 0 0 700px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 420px; max-width: 450px; gap: .1em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .02em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .p3-mats { font-size: 13px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; line-height: 1.2; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const fmt = (v: number, d = 3) => { const s = (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d); return s.padStart(d + 4); };

function mount(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const spaceAttr = host.dataset.space as Space | undefined;
  const init = (): State => ({ ...DEF, ...(p0?.s ?? {}), ...(spaceAttr && !p0 ? { space: spaceAttr } : {}) });
  let st = init();

  // ---- layout ----
  const left = htmlEl("div", "p3-left", "", host);
  const godCam = new OrbitCamera({ distance: 12, azimuth: 0.95, elevation: 0.5, fovy: 40 * Math.PI / 180, far: 200 });
  const spCam = new OrbitCamera({ distance: 6, azimuth: 0.6, elevation: 0.35, fovy: 40 * Math.PI / 180, far: 200 });
  const view = new GLView(left, {
    aspect: 2, fallback: "The matrices and the vertex table below still work.",
    viewports: [{ x: 0, y: 0, w: 0.55, h: 1, camera: godCam, label: "world: god's eye" }, { x: 0.55, y: 0, w: 0.45, h: 1, camera: spCam, label: "view space", background: [0.93, 0.95, 0.98] }],
  });
  const scr = htmlEl("canvas", "p3-screen", "", view.root);
  scr.width = SCREEN[0]; scr.height = SCREEN[1];
  scr.style.left = "55%"; scr.style.width = "45%";
  const mats = htmlEl("div", "p3-mats", "", left);
  const mBox = htmlEl("div", "", "", mats), vBox = htmlEl("div", "", "", mats), pBox = htmlEl("div", "", "", mats);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { st = { ...DEF, ...p.s }; frameSpace(); syncUI(); } })));
  const spRow = htmlEl("div", "p3-row", "", panel);
  const SPACES: [Space, string][] = [["model", "model"], ["world", "world"], ["view", "view"], ["ndc", "clip → NDC"], ["screen", "screen"]];
  const spBtns = SPACES.map(([s, t]) => button(spRow, t, () => { st.space = s; frameSpace(); update(); }));
  const sl = {
    yaw: slider(panel, "object: turn (°)", -180, 180, 5, st.yaw, v => { st.yaw = v; update(); }),
    sx: slider(panel, "object: scale x", 0.5, 3, 0.1, st.sx, v => { st.sx = v; update(); }),
    tz: slider(panel, "object: move z", -2, 2, 0.1, st.tz, v => { st.tz = v; update(); }),
    az: slider(panel, "camera: azimuth (°)", -180, 180, 5, st.az, v => { st.az = v; update(); }),
    el: slider(panel, "camera: elevation (°)", -60, 80, 5, st.el, v => { st.el = v; update(); }),
    dist: slider(panel, "camera: distance", 1, 12, 0.1, st.dist, v => { st.dist = v; update(); }),
    fov: slider(panel, "field of view (°)", 5, 120, 0.1, st.fov, v => { st.fov = v; update(); }),
    near: slider(panel, "near n", -2, 1, 0.05, Math.log10(st.near), v => { st.near = 10 ** v; update(); }, v => (10 ** v).toPrecision(2)),
    far: slider(panel, "far f", 2, 40, 0.5, st.far, v => { st.far = v; update(); }),
  };
  const row = htmlEl("div", "p3-row", "", panel);
  const orthoC = checkbox(row, "orthographic", st.ortho, v => { st.ortho = v; update(); });
  button(row, "reset", () => { st = init(); pb.forEach(b => b.classList.remove("active")); frameSpace(); syncUI(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag to rotate either view; click a vertex to follow it through the spaces.", panel);

  // ---- 3D objects ----
  const gHouse = view.addMesh(houseMesh(HV), { shading: "flat", lighting: "lambert", ambient: 0.5, wireframe: "overlay", wireWidth: 1 });
  gHouse.viewports = [0];
  const gAxes = view.addLines(axesLines(1), { width: 3 }); gAxes.viewports = [0];
  const gFrustum = view.addLines({ positions: [] }, { width: 2, color: [0.2, 0.2, 0.25] }); gFrustum.viewports = [0];
  const gPyr = view.addLines({ positions: [] }, { width: 3, color: [0.85, 0.2, 0.6] }); gPyr.viewports = [0];
  const gPick = view.addPoints({ positions: [] }, { size: 12, color: [1, 0.85, 0] }); gPick.viewports = [0];
  const sHouse = view.addMesh(houseMesh(HV), { shading: "flat", lighting: "lambert", ambient: 0.5, wireframe: "overlay", wireWidth: 1 });
  sHouse.viewports = [1];
  const sAxes = view.addLines(axesLines(1), { width: 3 }); sAxes.viewports = [1];
  const sBox = view.addLines({ positions: [] }, { width: 2, color: [0.2, 0.2, 0.25] }); sBox.viewports = [1];
  const sPyr = view.addLines({ positions: [] }, { width: 3, color: [0.85, 0.2, 0.6] }); sPyr.viewports = [1];
  const sPick = view.addPoints({ positions: [] }, { size: 12, color: [1, 0.85, 0] }); sPick.viewports = [1];

  const SCREEN_ASPECT = SCREEN[0] / SCREEN[1];

  /** The frustum's 12 edges (and the pyramid from the eye) in some space, given clip → that space. */
  function frustumLines(inv: Mat4, eye: Vec3 | null): { box: LineData; pyr: LineData } {
    const c = (i: number): Vec3 => M.transformPoint(inv, [i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1]);
    const P: number[] = [];
    for (let i = 0; i < 8; i++) for (const b of [1, 2, 4]) if (!(i & b)) P.push(...c(i), ...c(i | b));
    const Q: number[] = [];
    if (eye) for (const i of [0, 1, 2, 3]) Q.push(...eye, ...c(i));
    return { box: { positions: P }, pyr: { positions: Q } };
  }

  function frameSpace(): void {
    const t: Record<Space, [Vec3, number, number, number]> = {
      model: [[0, 0.2, 0], 4, 0.6, 0.35], world: [[0, 0, 0], 9, 0.9, 0.45], view: [[0, 0, -st.dist], 1.7 * st.far + 3, 1.1, 0.4],
      ndc: [[0, 0, 0], 5.2, Math.PI + 0.55, 0.35], screen: [[0, 0, 0], 5, 0, 0],
    };
    const [tg, d, az, el] = t[st.space];
    spCam.target = tg; spCam.distance = d; spCam.azimuth = az; spCam.elevation = el;
  }

  // ---- update ----
  function update(): void {
    spBtns.forEach((b, i) => b.classList.toggle("active", SPACES[i][0] === st.space));
    const { Mm, V, P, cam } = matrices(st);
    const VM = M.multiply(V, Mm), PVM = M.multiply(P, VM);
    // left: world
    gHouse.setModel(Mm);
    const invPV = M.invert(M.multiply(P, V))!;
    const fw = frustumLines(invPV, st.ortho ? null : cam.eye());
    gFrustum.setData(fw.box); gPyr.setData(fw.pyr);
    // right: the selected space
    const S = st.space;
    const lab = { model: "model space", world: "world space", view: "view space: camera at 0, looking down −z", ndc: "NDC: after the divide by w", screen: "screen: 40 × 30 pixels" }[S];
    if (view.viewports[1].label !== lab) { view.viewports[1].label = lab; view.setViewports(view.viewports); }
    let pts: Vec3[];
    sAxes.setVisible(S !== "screen"); sHouse.setVisible(S !== "screen"); sBox.setVisible(S === "world" || S === "view" || S === "ndc"); sPyr.setVisible(S === "world" || S === "view");
    if (S === "model") { pts = HV; sAxes.setData(axesLines(1)); }
    else if (S === "world") { pts = HV.map(p => M.transformPoint(Mm, p)); sAxes.setData(axesLines(1)); sBox.setData(fw.box); sPyr.setData(fw.pyr); }
    else if (S === "view") {
      pts = HV.map(p => M.transformPoint(VM, p));
      const fv = frustumLines(M.invert(P)!, st.ortho ? null : [0, 0, 0]);
      sAxes.setData(axesLines(1)); sBox.setData(fv.box); sPyr.setData(fv.pyr);
    } else {
      pts = HV.map(p => { const c = M.transformVec4(PVM, [...p, 1]); const cl = (x: number) => Math.max(-20, Math.min(20, x)); return [cl(c[0] / c[3]), cl(c[1] / c[3]), cl(c[2] / c[3])] as Vec3; });
      sAxes.setData({ positions: [] }); sBox.setData(boxLines([-1, -1, -1], [1, 1, 1]));
    }
    sHouse.setData(houseMesh(pts));
    scr.style.display = S === "screen" ? "" : "none";
    if (S === "screen") drawScreen(PVM);
    // picked vertex
    const wp = M.transformPoint(Mm, HV[st.pick]);
    gPick.setData({ positions: wp });
    sPick.setData({ positions: S === "screen" ? [] : pts[st.pick] });
    shown = pts;
    // matrices
    mBox.textContent = "M (model → world)\n" + M.formatMat(Mm, 3);
    vBox.textContent = "V (world → view)\n" + M.formatMat(V, 3);
    pBox.textContent = `P (view → clip), ${st.ortho ? "orthographic" : "perspective"}\n` + M.formatMat(P, 3);
    // vertex table
    const m4 = [...HV[st.pick], 1] as [number, number, number, number];
    const w4 = M.transformVec4(Mm, m4), v4 = M.transformVec4(VM, m4), c4 = M.transformVec4(PVM, m4);
    const nd: Vec3 = [c4[0] / c4[3], c4[1] / c4[3], c4[2] / c4[3]];
    const sx = (nd[0] + 1) / 2 * SCREEN[0], sy = (nd[1] + 1) / 2 * SCREEN[1];
    const t3 = (v: number[]) => v.slice(0, 3).map(x => fmt(x)).join(" ");
    const inside = c4[3] > 0 && Math.abs(nd[0]) <= 1 && Math.abs(nd[1]) <= 1 && Math.abs(nd[2]) <= 1;
    ro.textContent = [
      `vertex ${st.pick}            x       y       z       w`,
      `model   ${t3(m4)}`, `world   ${t3(w4)}`, `view    ${t3(v4)}`,
      `clip    ${t3(c4)} ${fmt(c4[3])}`, `NDC     ${t3(nd)}`,
      `screen  ${fmt(sx, 1)}  ${fmt(sy, 1)}  (pixels, y up)`,
      c4[3] < 0 ? "w < 0: BEHIND the camera; the divide flips it" : inside ? "inside the frustum" : "outside the frustum: clipped",
    ].join("\n");
    view.render();
  }

  let shown: Vec3[] = HV;

  function drawScreen(PVM: Mat4): void {
    const [W, H] = SCREEN, t = createTarget(W, H, [0.93, 0.95, 0.98]);
    const L = M.normalize([-0.35, 0.55, 0.75]);
    HT.forEach(([a, b, c, k], id) => {
      const tri: ClipVertex[] = [a, b, c].map(v => ({ p: M.transformVec4(PVM, [...HV[v], 1]), attr: [] }));
      // flat shading with the face normal in model space (enough for a picture)
      const n = M.normalize(M.cross(M.sub(HV[b], HV[a]), M.sub(HV[c], HV[a])));
      const s = 0.55 + 0.45 * Math.abs(M.dot(n, L));
      drawClipTriangle(t, tri, { cull: true }, () => [k[0] * s, k[1] * s, k[2] * s], id);
    });
    const rgba = new Uint8ClampedArray(4 * W * H);
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const p = j * W + i, q = 4 * ((H - 1 - j) * W + i);
      rgba[q] = 255 * t.color[3 * p]; rgba[q + 1] = 255 * t.color[3 * p + 1]; rgba[q + 2] = 255 * t.color[3 * p + 2]; rgba[q + 3] = 255;
    }
    // the selected vertex as a yellow pixel
    const c4 = M.transformVec4(PVM, [...HV[st.pick], 1]);
    if (c4[3] > 0) {
      const i = Math.floor((c4[0] / c4[3] + 1) / 2 * W), j = Math.floor((c4[1] / c4[3] + 1) / 2 * H);
      if (i >= 0 && i < W && j >= 0 && j < H) { const q = 4 * ((H - 1 - j) * W + i); rgba[q] = 255; rgba[q + 1] = 215; rgba[q + 2] = 0; }
    }
    scr.getContext("2d")!.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, W, H), 0, 0);
    // keep 4:3: letterbox inside the right viewport
    const r = view.root.getBoundingClientRect(), vw = 0.45 * r.width, vh = r.height;
    const w = Math.min(vw, vh * SCREEN_ASPECT), h = w / SCREEN_ASPECT;
    scr.style.width = `${w}px`; scr.style.height = `${h}px`; scr.style.left = `${0.55 * r.width + (vw - w) / 2}px`; scr.style.top = `${(vh - h) / 2}px`;
  }

  // ---- picking: click (not drag) near a vertex ----
  let down: [number, number] | null = null;
  view.root.addEventListener("pointerdown", ev => { down = [ev.clientX, ev.clientY]; });
  view.root.addEventListener("pointerup", ev => {
    if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) return;
    const ray = view.pointerRay(ev);
    if (!ray) return;
    const { Mm } = matrices(st);
    const cand = ray.vp === 0 ? HV.map(p => M.transformPoint(Mm, p)) : shown;
    let best = -1, bd = Infinity;
    cand.forEach((p, i) => {
      const d = M.sub(p, ray.origin), t = M.dot(d, ray.dir), q = M.sub(d, M.scale3(ray.dir, t));
      const dist = M.length(q) / Math.max(0.5, t);
      if (dist < bd) { bd = dist; best = i; }
    });
    if (best >= 0 && bd < 0.06) { st.pick = best; update(); }
  });

  function syncUI(): void {
    sl.yaw.set(st.yaw); sl.sx.set(st.sx); sl.tz.set(st.tz); sl.az.set(st.az); sl.el.set(st.el); sl.dist.set(st.dist);
    sl.fov.set(st.fov); sl.near.set(Math.log10(st.near)); sl.far.set(st.far); orthoC.checked = st.ortho;
    update();
  }
  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  frameSpace();
  syncUI();
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => { if (st.space === "screen") update(); }).observe(view.root);
}

autoMount(NAME, host => { injectStyle(); mount(host); });
