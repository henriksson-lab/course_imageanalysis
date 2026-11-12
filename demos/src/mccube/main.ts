// One marching-cubes cell, rotatable by dragging (WebGL2 via lib/gl3d.ts). Click a corner to
// flip it across c (orange = above c, inside); the slider sets the value of the selected
// corner and c, so the vertices slide along the edges. Readouts: the 8 bits, the case index
// 0..255, its class among the 23 (rotations) and the 15 (rotations + complement), and the
// rotation that maps it to the class representative ("rotate" animates it). Ambiguous faces
// (a 2D saddle) are tinted; the face rule (separate / join / asymptotic decider / complement
// as in the original 15-case table) chooses their triangulation. "Tetrahedra" shows the
// 5- or 6-tetrahedron split and its triangles instead. "Neighbour cube" adds a second cube
// sharing the +x face, with its own face rule: mismatched choices leave a crack (red).
// All geometry from lib/marching.ts (generated triangle table).
// Auto-mounts on <div data-demo="mccube">. Optional: data-preset="one|edge|classes|ambiguous|values|crack|tets".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { GLView, OrbitCamera, viewButtons, MeshObject, LinesObject, PointsObject } from "../lib/gl3d";
import * as M from "../lib/mat4";
import {
  CUBE_CORNERS, CUBE_EDGES, CUBE_FACES, FACE_NAMES, CLASSES_15, CLASSES_23, CUBE_ROTATIONS, ambiguousFaces,
  faceJoinMask, caseTriangulation, marchingCubes, singleCube, FaceRule, TETS5A, TETS6, edgeT,
} from "../lib/marching";
import { boundarySegments } from "../lib/mesh";

const NAME = "mccube";
type Alg = "mc" | "mt5" | "mt6";
const HI = 0.8, LO = 0.2;

interface Preset { id: string; label: string; values: number[]; c: number; rule?: FaceRule; alg?: Alg; neighbour?: boolean; nrule?: FaceRule; classes?: boolean; sel?: number }
const fromBits = (cs: number) => Array.from({ length: 8 }, (_, k) => ((cs >> k) & 1 ? HI : LO));
const PRESETS: Preset[] = [
  { id: "one", label: "one corner", values: fromBits(1), c: 0.5 },
  { id: "edge", label: "an edge (two corners)", values: fromBits(3), c: 0.5 },
  { id: "classes", label: "the 15 classes", values: fromBits(1), c: 0.5, classes: true },
  { id: "ambiguous", label: "an ambiguous face", values: fromBits(1 | 8), c: 0.5, rule: "separate" },
  { id: "values", label: "values: vertices slide", values: [0.9, 0.35, 0.6, 0.1, 0.45, 0.2, 0.3, 0.05], c: 0.5, sel: 0 },
  { id: "crack", label: "a crack", values: [0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8], c: 0.5, rule: "separate", neighbour: true, nrule: "join" },
  { id: "tets", label: "tetrahedra instead", values: fromBits(1 | 8), c: 0.5, alg: "mt6" },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; }
.qb-demo[data-demo="${NAME}"] .mc-view { flex: 1 1 440px; min-width: 260px; max-width: 560px; position: relative; }
.qb-demo[data-demo="${NAME}"] .mc-lbl { position: absolute; font: 600 18px ui-monospace, monospace; pointer-events: none; transform: translate(-50%, -135%);
  padding: 0 .25em; border-radius: 4px; background: rgba(255,255,255,.75); color: #222; white-space: nowrap; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; max-width: 420px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .mc-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .mc-bits { font: 600 20px ui-monospace, monospace; letter-spacing: .05em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .12em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .mc-view { max-width: 520px; }
.reveal .qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .mc-view { flex: 0 0 470px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 520px; max-width: 560px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .3em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider, .reveal .qb-demo[data-demo="${NAME}"] .qb-check { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .mc-bits { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .mc-lab { font-size: 13px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const W = (p: number[]): M.Vec3 => [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5];
const RULE_NAMES: Record<FaceRule, string> = { separate: "separate", join: "join", decider: "decider", complement: "complement (15-case table)" };

/** Mesh of two cubes side by side (left x 0..1, right x 1..2) with their own face rules. */
function twoCubes(vals: number[], c: number, rL: FaceRule, rR: FaceRule) {
  // grid 3 × 2 × 2: point (x, y, z) id = (z * 2 + y) * 3 + x
  const pid = (x: number, y: number, z: number) => (z * 2 + y) * 3 + x;
  const pos = (p: number): number[] => [p % 3, Math.floor(p / 3) % 2, Math.floor(p / 6)];
  const P: number[] = [], I: number[] = [], ids = new Map<number, number>();
  for (const [cx, rule] of [[0, rL], [1, rR]] as [number, FaceRule][]) {
    const corner = CUBE_CORNERS.map(([x, y, z]) => pid(cx + x, y, z));
    const v = corner.map(p => vals[p]);
    let cs = 0;
    v.forEach((a, k) => { if (a > c) cs |= 1 << k; });
    const tr = caseTriangulation(cs, faceJoinMask(cs, rule, v, c));
    const centre = new Map<number, number>();
    const vert = (e: number): number => {
      if (e >= 12) {
        if (!centre.has(e)) {
          const L = tr.loops[e - 12], m = [0, 0, 0];
          for (const f of L) { const q = edgePoint(f); for (let a = 0; a < 3; a++) m[a] += q[a] / L.length; }
          P.push(...m); centre.set(e, P.length / 3 - 1);
        }
        return centre.get(e)!;
      }
      const [a, b] = CUBE_EDGES[e], pa = corner[a], pb = corner[b], key = Math.min(pa, pb) * 12 + Math.max(pa, pb);
      if (!ids.has(key)) { P.push(...edgePoint(e)); ids.set(key, P.length / 3 - 1); }
      return ids.get(key)!;
    };
    const edgePoint = (e: number): number[] => {
      const [a, b] = CUBE_EDGES[e], A = pos(corner[a]), B = pos(corner[b]), t = edgeT(vals[corner[a]], vals[corner[b]], c);
      return [A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]), A[2] + t * (B[2] - A[2])];
    };
    for (let i = 0; i < tr.tris.length; i++) I.push(vert(tr.tris[i]));
  }
  const positions = Float32Array.from(P.map((p, i) => p - 0.5)), indices = Uint32Array.from(I);
  // boundary edges on the shared face x = 1 (world x = 0.5) are cracks
  const seg = boundarySegments({ positions, indices }), crack: number[] = [];
  for (let i = 0; i < seg.length; i += 6) if (Math.abs(seg[i] - 0.5) < 1e-6 && Math.abs(seg[i + 3] - 0.5) < 1e-6) crack.push(...seg.slice(i, i + 6));
  return { positions, indices, crack: Float32Array.from(crack) };
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    values: [...p0.values], c: p0.c, rule: (p0.rule ?? "separate") as FaceRule, alg: (p0.alg ?? "mc") as Alg,
    neighbour: !!p0.neighbour, nrule: (p0.nrule ?? "join") as FaceRule, sel: p0.sel ?? -1, classIdx: 0,
    far: [0.2, 0.2, 0.2, 0.2], // neighbour's far corners (x = 2): y,z = 00, 10, 01, 11
  };

  const vbox = htmlEl("div", "mc-view", "", host);
  const cam = new OrbitCamera({ distance: 2.9, azimuth: 0.55, elevation: 0.42 });
  const view = new GLView(vbox, { aspect: 1.15, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam }], fallback: "The figure on the page shows the corner numbering." });
  const panel = htmlEl("div", "qb-panel", "", host);

  // scene objects
  const faceTint = view.addMesh({ positions: [0, 0, 0], indices: [] }, { color: [0.58, 0.4, 0.74], opacity: 0.28, lighting: "unlit", shading: "flat" });
  const surf = view.addMesh({ positions: [0, 0, 0], indices: [] }, { color: [0.35, 0.6, 0.9], shading: "flat", lighting: "lambert", wireframe: "overlay", wireColor: [0.1, 0.2, 0.45], wireWidth: 1.6 });
  const box = view.addLines({ positions: [] }, { width: 2.5, color: [0.35, 0.35, 0.4] });
  const tetLines = view.addLines({ positions: [] }, { width: 1.5, color: [0.55, 0.55, 0.6] });
  const crackLines = view.addLines({ positions: [] }, { width: 6, color: [0.85, 0.1, 0.1] });
  const corners = view.addPoints({ positions: [] }, { size: 24 });
  const vpts = view.addPoints({ positions: [] }, { size: 11, color: [0.1, 0.3, 0.7] });
  const all: (MeshObject | LinesObject | PointsObject)[] = [faceTint, surf, box, tetLines, crackLines, corners, vpts];

  // corner labels (HTML over the canvas)
  const labels = Array.from({ length: 8 }, () => htmlEl("div", "mc-lbl", "", vbox));
  let model = M.identity();
  const placeLabels = () => {
    const r = view.root.getBoundingClientRect(), PV = M.multiply(cam.projection(r.width / Math.max(1, r.height)), cam.view());
    CUBE_CORNERS.forEach((p, k) => {
      const q = M.transformPoint(M.multiply(PV, model), W(p));
      labels[k].style.left = `${((q[0] + 1) / 2 * view.root.clientWidth).toFixed(1)}px`;
      labels[k].style.top = `${((1 - q[1]) / 2 * view.root.clientHeight).toFixed(1)}px`;
      labels[k].style.display = view.active ? "" : "none";
    });
  };
  view.opts.onRender = placeLabels;

  // picking: a click (no drag) on the view selects/flips the nearest corner
  let down: [number, number] | null = null;
  view.root.addEventListener("pointerdown", ev => { down = [ev.clientX, ev.clientY]; });
  view.root.addEventListener("pointerup", ev => {
    if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) { down = null; return; }
    down = null;
    const r = view.root.getBoundingClientRect(), PV = M.multiply(cam.projection(r.width / r.height), cam.view());
    let best = -1, bd = 30;
    CUBE_CORNERS.forEach((p, k) => {
      const q = M.transformPoint(PV, W(p));
      const sx = r.left + (q[0] + 1) / 2 * r.width, sy = r.top + (1 - q[1]) / 2 * r.height, d = Math.hypot(sx - ev.clientX, sy - ev.clientY);
      if (d < bd) { bd = d; best = k; }
    });
    if (best < 0) return;
    const v = st.values[best];
    st.values[best] = Math.max(0, Math.min(1, Math.round((2 * st.c - v) * 100) / 100));
    if ((st.values[best] > st.c) === (v > st.c)) st.values[best] = v > st.c ? Math.max(0, st.c - 0.3) : Math.min(1, st.c + 0.3);
    st.sel = best;
    clearPresets(); update();
  });

  // ---- panel ----
  htmlEl("div", "mc-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const bits = htmlEl("div", "mc-bits", "", panel);
  const classRow = htmlEl("div", "qb-presets", "", panel);
  const prevB = button(classRow, "◀ class", () => stepClass(-1));
  const nextB = button(classRow, "class ▶", () => stepClass(1));
  button(classRow, "rotate to its representative", () => animateToRep());
  const vS = slider(panel, "value of the selected corner", 0, 1, 0.01, 0.5, v => { if (st.sel >= 0) { st.values[st.sel] = v; clearPresets(); update(); } });
  const cS = slider(panel, "iso-value c", 0.05, 0.95, 0.01, st.c, v => { st.c = v; update(); });
  htmlEl("div", "mc-lab", "face rule for ambiguous faces", panel);
  const ruleRow = htmlEl("div", "qb-presets", "", panel);
  const ruleBtns = (["separate", "join", "decider", "complement"] as FaceRule[]).map(r => button(ruleRow, RULE_NAMES[r], () => { st.rule = r; update(); }));
  htmlEl("div", "mc-lab", "algorithm", panel);
  const algRow = htmlEl("div", "qb-presets", "", panel);
  const algBtns = (["mc", "mt5", "mt6"] as Alg[]).map(a => button(algRow, a === "mc" ? "marching cubes" : a === "mt5" ? "5 tetrahedra" : "6 tetrahedra", () => { st.alg = a; update(); }));
  const nbChk = checkbox(panel, "neighbour cube (+x), its rule:", st.neighbour, v => { st.neighbour = v; update(); });
  const nbRow = htmlEl("div", "qb-presets", "", panel);
  const nbBtns = (["separate", "join", "decider"] as FaceRule[]).map(r => button(nbRow, r, () => { st.nrule = r; update(); }));
  viewButtons(panel, view, 0, ["front", "right", "top", "iso"], false);
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag to rotate; click a corner to flip it across c and select it. Corner k = x + 2y + 4z (x right, y up, z towards you in the front view). Orange: above c. Purple: an ambiguous face.", panel);

  function clearPresets(): void { presetBtns.forEach(b => b.classList.remove("active")); }

  function load(p: Preset): void {
    st.values = [...p.values]; st.c = p.c; st.rule = p.rule ?? "separate"; st.alg = p.alg ?? "mc";
    st.neighbour = !!p.neighbour; st.nrule = p.nrule ?? "join"; st.sel = p.sel ?? -1; st.classIdx = 0;
    cS.set(st.c); nbChk.checked = st.neighbour;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    showClassButtons(!!p.classes);
    if (p.classes) { st.classIdx = 1; st.values = fromBits(CLASSES_15.reps[1]); }
    cam.target = st.neighbour ? [0.5, 0, 0] : [0, 0, 0];
    cam.distance = st.neighbour ? 3.6 : 2.9;
    update();
  }
  function showClassButtons(on: boolean): void { prevB.style.display = on ? "" : "none"; nextB.style.display = on ? "" : "none"; }
  function stepClass(d: number): void {
    st.classIdx = (st.classIdx + d + 15) % 15;
    st.values = fromBits(CLASSES_15.reps[st.classIdx]); st.c = 0.5; cS.set(0.5); st.sel = -1;
    update();
  }

  let anim = 0;
  function animateToRep(): void {
    const cs = caseOf();
    const k = CLASSES_23.of[cs], R = CUBE_ROTATIONS[k.rotation];
    if (R.angle < 1e-6) return;
    cancelAnimationFrame(anim);
    const t0 = performance.now(), dur = 1200;
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / dur), e = t * t * (3 - 2 * t);
      model = M.rotation(R.axis as M.Vec3, R.angle * e);
      for (const o of all) o.setModel(model);
      if (t < 1) anim = requestAnimationFrame(step);
      else {
        const nv = new Array(8).fill(0);
        st.values.forEach((v, i) => { nv[R.perm[i]] = v; });
        st.values = nv;
        if (st.sel >= 0) st.sel = R.perm[st.sel];
        model = M.identity();
        for (const o of all) o.setModel(model);
        update();
      }
    };
    anim = requestAnimationFrame(step);
  }

  const caseOf = () => st.values.reduce((s, v, k) => s | (v > st.c ? 1 << k : 0), 0);

  function update(): void {
    ruleBtns.forEach((b, i) => b.classList.toggle("active", ["separate", "join", "decider", "complement"][i] === st.rule));
    algBtns.forEach((b, i) => b.classList.toggle("active", ["mc", "mt5", "mt6"][i] === st.alg));
    nbBtns.forEach((b, i) => b.classList.toggle("active", ["separate", "join", "decider"][i] === st.nrule));
    nbRow.style.display = st.neighbour ? "" : "none";
    if (st.sel >= 0) vS.set(st.values[st.sel]);
    vS.input.disabled = st.sel < 0;
    const c = st.c, cs = caseOf(), amb = ambiguousFaces(cs);

    // corners
    const cp: number[] = [], cc: number[] = [];
    CUBE_CORNERS.forEach((p, k) => {
      cp.push(...W(p));
      const above = st.values[k] > c;
      cc.push(...(k === st.sel ? (above ? [0.85, 0.3, 0] : [0.55, 0.55, 0.6]) : above ? [1, 0.5, 0.05] : [0.97, 0.97, 0.97]));
      const plain = Math.abs(st.values[k] - HI) < 1e-9 || Math.abs(st.values[k] - LO) < 1e-9;
      labels[k].textContent = plain ? `${k}` : `${k}: ${st.values[k].toFixed(2)}`;
    });
    if (st.neighbour) {
      st.far.forEach((v, i) => { cp.push(1.5, (i & 1) - 0.5, (i >> 1) - 0.5); cc.push(...(v > c ? [1, 0.5, 0.05] : [0.97, 0.97, 0.97])); });
    }
    corners.setData({ positions: cp, colors: cc });
    // box edges
    const bl: number[] = [];
    for (const [a, b] of CUBE_EDGES) bl.push(...W(CUBE_CORNERS[a]), ...W(CUBE_CORNERS[b]));
    if (st.neighbour) for (const [a, b] of CUBE_EDGES) bl.push(...W(CUBE_CORNERS[a]).map((v, i) => v + (i ? 0 : 1)), ...W(CUBE_CORNERS[b]).map((v, i) => v + (i ? 0 : 1)));
    box.setData({ positions: bl });
    // ambiguous faces
    const fp: number[] = [], fi: number[] = [];
    for (const f of amb) {
      const base = fp.length / 3;
      for (const k of CUBE_FACES[f]) fp.push(...W(CUBE_CORNERS[k]));
      fi.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    faceTint.setData({ positions: fp.length ? fp : [0, 0, 0], indices: fi });

    // surface
    let pos: ArrayLike<number> = [], idx: ArrayLike<number> = [];
    const tl: number[] = [];
    let crack = new Float32Array(0);
    if (st.neighbour && st.alg === "mc") {
      const vals = new Array(12).fill(0);
      CUBE_CORNERS.forEach(([x, y, z], k) => { vals[(z * 2 + y) * 3 + x] = st.values[k]; });
      st.far.forEach((v, i) => { vals[((i >> 1) * 2 + (i & 1)) * 3 + 2] = v; });
      const T = twoCubes(vals, c, st.rule, st.nrule);
      pos = T.positions; idx = T.indices; crack = T.crack;
    } else if (st.alg === "mc") {
      const S = singleCube(st.values, c, st.rule);
      const P: number[] = [];
      S.edgePoints.forEach(p => P.push(...(p ? W(p) : [0, 0, 0])));
      pos = P; idx = S.triangles;
    } else {
      // a 2 × 2 × 2 volume through the library (tetrahedra), stored (z, y, x)
      const d = new Float32Array(8);
      CUBE_CORNERS.forEach(([x, y, z], k) => { d[(z * 2 + y) * 2 + x] = st.values[k]; });
      const m = marchingCubes(d, 2, 2, 2, c, { algorithm: st.alg });
      pos = Array.from(m.positions, v => v - 0.5); idx = m.indices;
      const tets = st.alg === "mt6" ? TETS6 : TETS5A;
      for (const T of tets) for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) tl.push(...W(CUBE_CORNERS[T[i]]), ...W(CUBE_CORNERS[T[j]]));
    }
    surf.setData({ positions: pos.length ? pos : [0, 0, 0], indices: idx });
    tetLines.setData({ positions: tl });
    crackLines.setData({ positions: crack });
    const vp: number[] = [];
    const used = new Set<number>(Array.from(idx));
    used.forEach(i => vp.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]));
    vpts.setData({ positions: vp });

    // readout
    const b = st.values.map(v => (v > c ? 1 : 0));
    bits.textContent = `bits 7…0: ${b.slice().reverse().join("")}  → case ${cs}`;
    const k23 = CLASSES_23.of[cs], k15 = CLASSES_15.of[cs];
    const R = CUBE_ROTATIONS[k23.rotation];
    const lines = [
      `case ${cs} = ${b.map((x, k) => (x ? `2^${k}` : "")).filter(s => s).join(" + ") || "0"}`,
      `class ${k15.id} of 15 (representative ${k15.rep}${k15.complemented ? ", via the complement" : ""})`,
      `class ${k23.id} of 23 (representative ${k23.rep})`,
      R.angle < 1e-6 ? "already the representative" : `rotation to it: ${Math.round(R.angle * 180 / Math.PI)}° about (${R.axis.map(a => Math.round(a * 100) / 100).join(", ")})`,
      `ambiguous faces: ${amb.length ? amb.map(f => FACE_NAMES[f]).join(", ") : "none"}`,
      `triangles: ${Math.floor(idx.length / 3)}${st.neighbour && st.alg === "mc" ? ` (both cubes)   crack edges: ${crack.length / 6}` : ""}`,
    ];
    if (st.classIdx > 0 || presetBtns[2].classList.contains("active")) lines.unshift(`showing class ${st.classIdx} of the 15`);
    ro.textContent = lines.join("\n");
    view.render();
  }

  load(p0);
}

autoMount(NAME, mount);
