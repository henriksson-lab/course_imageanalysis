// raystep: a step-through ray tracer on a coarse pixel grid (ray-tracing day, units raycast/ and
// whitted/). Left: the image, filled in as rays are traced (next pixel, next row, all; click a
// pixel to inspect it). Middle: a top view (x–z plane) of the scene with the camera, the image
// plane and the inspected ray with every candidate hit (sphere roots, plane, triangle inside or
// outside); with recursion, the whole ray tree (primary, shadow, reflected, refracted). Readout:
// the quadratic's discriminant and roots, t for the plane, t, u, v for the triangle, the nearest
// hit, normal and Lambert term; or the ray tree with weights. Views: shaded / depth / normal /
// object / barycentrics.
// Options: data-depth="0" (ray casting, unit 1; default) or 1–6 (Whitted, unit 2), data-preset.
import { htmlEl, svgEl, slider, button, presets, checkbox, autoMount, svgPoint } from "../lib/ui";
import * as RT from "../lib/raytrace";
import { linearToSrgb8, categoryColor } from "../lib/colour";
import { Vec3, dot, normalize, sub } from "../lib/mat4";
import { State, DEF0, DEF1, PRESETS0, PRESETS1, makeScene } from "./scenes";

const NAME = "raystep";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .7em; }
${S} .rs-c1 { flex: 0 1 420px; min-width: 260px; display: flex; flex-direction: column; gap: .3em; }
${S} .rs-c2 { flex: 0 1 330px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
${S} .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .35em; }
${S} canvas.rs-img { width: 100%; aspect-ratio: 4 / 3; image-rendering: pixelated; display: block; cursor: crosshair; border-radius: 3px; touch-action: none; }
${S} svg.rs-top { width: 100%; display: block; background: var(--qb-panel); border-radius: 4px; touch-action: none; }
${S} .rs-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-family: system-ui, sans-serif; white-space: pre; overflow: hidden; line-height: 1.3; }
${S} .rs-count { font-weight: 600; }
.reveal ${S} { flex-wrap: nowrap; font-size: 18px; gap: .5em; }
.reveal ${S} .rs-c1 { flex: 0 0 400px; }
.reveal ${S} .rs-c2 { flex: 0 0 330px; }
.reveal ${S} .qb-panel { flex: 0 0 360px; gap: .2em; padding: .3em .45em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .qb-presets { gap: .2em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

const KIND_COL: Record<RT.RayKind, string> = { primary: "#d49b00", shadow: "#666", reflect: "#1f77b4", refract: "#2ca02c" };
const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : v > 0 ? "∞" : "—");
const vec = (v: Vec3) => `(${v.map(x => x.toFixed(2)).join(", ")})`;

// top view window: x ∈ [−4.5, 4.5], z ∈ [−6.5, 4.5]
const X0 = -4.5, X1 = 4.5, Z0 = -6.5, Z1 = 4.5, VW = 300, VH = VW * (Z1 - Z0) / (X1 - X0);
const tx = (x: number) => (x - X0) / (X1 - X0) * VW, tz = (z: number) => (z - Z0) / (Z1 - Z0) * VH;

function mount(host: HTMLElement): void {
  const whitted = Number(host.dataset.depth ?? "0") > 0;
  const DEF = whitted ? { ...DEF1, depth: Number(host.dataset.depth) } : DEF0;
  const PRE = whitted ? PRESETS1 : PRESETS0;
  const p0 = PRE.find(p => p.id === host.dataset.preset);
  const fresh = (s: Partial<State> = {}): State => ({ ...DEF, ...s, eye: [...(s.eye ?? DEF.eye)] as Vec3, inspect: [...(s.inspect ?? DEF.inspect)] as [number, number] });
  let st = fresh(p0?.s);

  // ---------------- column 1: the image
  const c1 = htmlEl("div", "rs-c1", "", host);
  const cv = htmlEl("canvas", "rs-img", "", c1);
  const r1 = htmlEl("div", "rs-row", "", c1);
  button(r1, "next pixel", () => { advance(1); });
  button(r1, "next row", () => { advance(st.W - ((st.traced < 0 ? 0 : st.traced) % st.W)); });
  button(r1, "all", () => { st.traced = -1; draw(); });
  button(r1, "clear", () => { st.traced = 0; draw(); });
  const r2 = htmlEl("div", "rs-row", "", c1);
  const viewBtns = (["shaded", "depth", "normal", "id", "bary"] as const).map(v => {
    const b = button(r2, v === "id" ? "object" : v === "bary" ? "barycentric" : v, () => { st.view = v; draw(); });
    return { v, b };
  });
  const r3 = htmlEl("div", "rs-row", "", c1);
  const gridBtns = ([[16, 12], [32, 24], [64, 48]] as const).map(([w, h]) => {
    const b = button(r3, `${w}×${h}`, () => { const fx = (st.inspect[0] + 0.5) / st.W, fy = (st.inspect[1] + 0.5) / st.H; st.W = w; st.H = h; st.inspect = [Math.floor(fx * w), Math.floor(fy * h)]; if (st.traced > 0) st.traced = 0; draw(); });
    return { w, b };
  });
  const counts = htmlEl("span", "rs-count", "", r3);

  // ---------------- column 2: top view and readout
  const c2 = htmlEl("div", "rs-c2", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${VW} ${VH}`, class: "rs-top" }, c2);
  const gScene = svgEl("g", {}, svg), gRays = svgEl("g", {}, svg), gCam = svgEl("g", {}, svg);
  const ro = htmlEl("div", "qb-readout", "", c2);

  // ---------------- column 3: presets and controls
  const panel = htmlEl("div", "qb-panel", "", host);
  const pb = presets(panel, [...PRE.map(p => ({ label: p.label, apply: () => { st = fresh(p.s); sync(); } })), { label: "reset", apply: () => { st = fresh(); sync(); pb.forEach(b => b.classList.remove("active")); } }]);
  const fovS = slider(panel, "field of view (°)", 20, 120, 5, st.fov, v => { st.fov = v; draw(); });
  const radS = slider(panel, "sphere A radius", 0.3, 1.5, 0.05, st.radius, v => { st.radius = v; draw(); });
  const rowC = htmlEl("div", "rs-row", "", panel);
  const orthoC = checkbox(rowC, "orthographic", st.ortho, v => { st.ortho = v; draw(); });
  let depS: ReturnType<typeof slider> | null = null, epsS: ReturnType<typeof slider> | null = null;
  const toggles: { k: "shadows" | "reflection" | "refraction" | "fresnel"; c: HTMLInputElement }[] = [];
  if (whitted) {
    depS = slider(panel, "max depth", 0, 6, 1, st.depth, v => { st.depth = v; draw(); });
    epsS = slider(panel, "offset ε", -8, 0, 0.1, st.epsLog, v => { st.epsLog = v; draw(); }, v => (v <= -7.95 ? "0" : (10 ** v).toPrecision(1)));
    const tr = htmlEl("div", "rs-row", "", panel);
    for (const k of ["shadows", "reflection", "refraction", "fresnel"] as const) toggles.push({ k, c: checkbox(tr, k === "fresnel" ? "Fresnel" : k, st[k], v => { st[k] = v; draw(); }) });
    fovS.input.closest("label")!.style.display = "none";
  } else radS.input.closest("label")!.style.display = "";
  htmlEl("div", "qb-note", "Drag the camera (C) in the top view. Click a pixel to inspect its ray.", panel);

  function sync(): void {
    fovS.set(st.fov); radS.set(st.radius); orthoC.checked = st.ortho;
    depS?.set(st.depth); epsS?.set(st.epsLog);
    toggles.forEach(t => (t.c.checked = st[t.k]));
    draw();
  }
  function advance(k: number): void {
    const n = st.W * st.H;
    if (st.traced < 0 || st.traced >= n) st.traced = 0;
    st.traced = Math.min(n, st.traced + k);
    const last = st.traced - 1;
    st.inspect = [last % st.W, Math.floor(last / st.W)];
    draw();
  }

  const cam = (): RT.Camera => ({ eye: st.eye, fov: st.fov, ortho: st.ortho, height: 4 });
  const opts = (tree: boolean): RT.WhittedOptions => whitted
    ? { depth: st.depth, shadows: st.shadows, reflection: st.reflection, refraction: st.refraction, fresnel: st.fresnel, eps: st.epsLog <= -7.95 ? 0 : 10 ** st.epsLog, float32: true, minWeight: 1e-3, tree }
    : { depth: 0, shadows: false, reflection: false, refraction: false, fresnel: false, eps: 0, float32: true, tree };

  function pixelColour(sc: RT.Scene, i: number, j: number, cnt: RT.Counter): [number, number, number] {
    const ray = RT.cameraRay(i, j, st.W, st.H, cam());
    if (st.view === "shaded") { const c = RT.traceWhitted(sc, ray.o, ray.d, opts(false), cnt).colour; return [linearToSrgb8(c[0]), linearToSrgb8(c[1]), linearToSrgb8(c[2])]; }
    cnt.rays++;
    const h = RT.nearestHit(sc.objects, ray.o, ray.d, 0, Infinity, cnt);
    if (!h) return [40, 40, 48];
    if (st.view === "depth") { const v = Math.max(0, Math.min(255, 255 - h.t * 28)); return [v, v, v]; }
    if (st.view === "normal") { const n = dot(ray.d, h.n) < 0 ? h.n : [-h.n[0], -h.n[1], -h.n[2]]; return n.map(x => Math.round(127.5 + 127.5 * x)) as [number, number, number]; }
    if (st.view === "id") return categoryColor(h.obj).map(x => Math.round(255 * x)) as [number, number, number];
    const ob = sc.objects[h.obj];
    if (ob.type === "triangle") return [1 - h.u - h.v, h.u, h.v].map(x => Math.round(255 * x)) as [number, number, number];
    return [90, 90, 90];
  }

  function draw(): void {
    viewBtns.forEach(({ v, b }) => b.classList.toggle("active", v === st.view));
    gridBtns.forEach(({ w, b }) => b.classList.toggle("active", w === st.W));
    const sc = makeScene(st.scene, st.radius);
    const n = st.W * st.H, done = st.traced < 0 ? n : Math.min(st.traced, n);
    // ---- image
    const img = new ImageData(st.W, st.H), cnt = RT.newCounter();
    for (let p = 0; p < n; p++) {
      const i = p % st.W, j = Math.floor(p / st.W);
      let c: [number, number, number];
      if (p < done) c = pixelColour(sc, i, j, cnt);
      else c = (i + j) % 2 ? [205, 205, 210] : [190, 190, 196];
      img.data.set([c[0], c[1], c[2], 255], 4 * p);
    }
    const scale = 8;
    cv.width = st.W * (st.W <= 16 ? 24 : st.W <= 32 ? 12 : 6); cv.height = cv.width * st.H / st.W;
    const g = cv.getContext("2d")!;
    const off = document.createElement("canvas"); off.width = st.W; off.height = st.H; off.getContext("2d")!.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false; g.drawImage(off, 0, 0, cv.width, cv.height);
    const ps = cv.width / st.W;
    if (st.W <= 32) { g.strokeStyle = "rgba(0,0,0,0.12)"; g.lineWidth = 1; for (let i = 1; i < st.W; i++) { g.beginPath(); g.moveTo(i * ps, 0); g.lineTo(i * ps, cv.height); g.stroke(); } for (let j = 1; j < st.H; j++) { g.beginPath(); g.moveTo(0, j * ps); g.lineTo(cv.width, j * ps); g.stroke(); } }
    g.strokeStyle = "#ffcc00"; g.lineWidth = Math.max(2, ps / 6);
    g.strokeRect(st.inspect[0] * ps, st.inspect[1] * ps, ps, ps);
    g.strokeStyle = "#000"; g.lineWidth = 1; g.strokeRect(st.inspect[0] * ps - 1, st.inspect[1] * ps - 1, ps + 2, ps + 2);
    void scale;
    // ---- inspected ray
    const [ii, jj] = st.inspect, ray = RT.cameraRay(ii, jj, st.W, st.H, cam());
    const pc = RT.newCounter(), res = RT.traceWhitted(sc, ray.o, ray.d, opts(true), pc);
    counts.textContent = `rays ${cnt.rays} · tests ${cnt.tests}`;
    drawTop(sc, ray, res.node);
    ro.textContent = whitted ? treeText(sc, res.node, pc) : castText(sc, ray, ii, jj);
  }

  function castText(sc: RT.Scene, ray: RT.Ray, i: number, j: number): string {
    const L: string[] = [`pixel (${i}, ${j}): d = ${vec(ray.d)}`];
    sc.objects.forEach(ob => {
      if (ob.type === "sphere") {
        const s = RT.solveSphere(ray.o, ray.d, ob.c, ob.r);
        L.push(s.disc < 0 ? `${ob.name}: disc ${f2(s.disc)} < 0: miss` : `${ob.name}: disc ${f2(s.disc)}, t ${f2(s.t1)}, ${f2(s.t2)}`);
      } else if (ob.type === "plane") {
        const t = RT.hitPlane(ray.o, ray.d, ob.p, ob.n);
        L.push(`${ob.name}: ${t === Infinity ? "parallel: no hit" : `t ${f2(t)}${t <= 0 ? " (behind)" : ""}`}`);
      } else if (ob.type === "triangle") {
        const s = RT.solveTriangle(ray.o, ray.d, ob.a, ob.b, ob.c);
        L.push(`${ob.name}: t ${f2(s.t)}, u ${f2(s.u)}, v ${f2(s.v)}: ${RT.insideTriangle(s.u, s.v) ? "in" : "out"}`);
      }
    });
    const h = RT.nearestHit(sc.objects, ray.o, ray.d);
    if (!h) L.push("nearest: none (background)");
    else {
      const nf = dot(ray.d, h.n) < 0 ? h.n : [-h.n[0], -h.n[1], -h.n[2]] as Vec3;
      const l = normalize(sub(sc.lights[0].p, h.p));
      L.push(`nearest: ${sc.objects[h.obj].name}, t ${f2(h.t)}`);
      L.push(`n = ${vec(nf)}, n·l = ${f2(Math.max(0, dot(nf, l)))}`);
    }
    return L.join("\n");
  }

  function treeText(sc: RT.Scene, root: RT.RayNode, pc: RT.Counter): string {
    const L: string[] = [];
    const walk = (nd: RT.RayNode, ind: string) => {
      if (L.length >= 8) return;
      const name = nd.obj >= 0 ? sc.objects[nd.obj].name : nd.kind === "shadow" ? "light" : "sky";
      if (nd.kind === "shadow") L.push(`${ind}shadow → ${nd.obj >= 0 ? "blocked by " + name : "light reached"}`);
      else L.push(`${ind}${nd.kind} → ${name}, w ${nd.weight.toFixed(nd.weight < 0.01 ? 3 : 2)}${nd.note ? " (" + nd.note + ")" : ""}`);
      nd.children.forEach(c => walk(c, ind + "  "));
    };
    walk(root, "");
    const size = RT.treeSize(root);
    if (size > L.length) L.push(`… (${size - L.length} more)`);
    L.unshift(`pixel (${st.inspect[0]}, ${st.inspect[1]}): ${size} rays, ${pc.tests} tests`);
    return L.join("\n");
  }

  function drawTop(sc: RT.Scene, ray: RT.Ray, root: RT.RayNode): void {
    gScene.replaceChildren(); gRays.replaceChildren(); gCam.replaceChildren();
    const hasGround = sc.objects.some(o => o.type === "plane" && Math.abs(o.n[1]) > 0.9);
    if (hasGround) { const t = svgEl("text", { x: 6, y: 18, "font-size": 15 }, gScene); t.textContent = "ground plane: everywhere"; t.style.fill = "var(--qb-muted)"; }
    sc.objects.forEach((ob, k) => {
      const col = ob.mat.kind === "mirror" ? "#9aa" : ob.mat.kind === "glass" ? "#bde" : `rgb(${ob.mat.colour.map(x => Math.round(255 * x)).join(",")})`;
      if (ob.type === "sphere") {
        svgEl("circle", { cx: tx(ob.c[0]), cy: tz(ob.c[2]), r: ob.r / (X1 - X0) * VW, fill: col, "fill-opacity": 0.55, stroke: "#333", "stroke-width": 1.5 }, gScene);
        lab(ob.c[0], ob.c[2], ob.name.replace("sphere ", ""));
      } else if (ob.type === "triangle") {
        const pts = [ob.a, ob.b, ob.c].map(p => `${tx(p[0])},${tz(p[2])}`).join(" ");
        svgEl("polygon", { points: pts, fill: col, "fill-opacity": 0.6, stroke: "#333", "stroke-width": 2 }, gScene);
        lab((ob.a[0] + ob.b[0] + ob.c[0]) / 3, (ob.a[2] + ob.b[2] + ob.c[2]) / 3 - 0.6, "T");
      } else if (ob.type === "plane" && Math.abs(ob.n[0]) > 0.9) {
        svgEl("line", { x1: tx(ob.p[0]), x2: tx(ob.p[0]), y1: 0, y2: VH, stroke: col, "stroke-width": 5 }, gScene);
      }
      void k;
    });
    for (const L of sc.lights) { svgEl("circle", { cx: tx(L.p[0]), cy: tz(L.p[2]), r: 7, fill: "#ffd84a", stroke: "#a80", "stroke-width": 1.5 }, gScene); lab(L.p[0] + 0.55, L.p[2] + 0.2, "light"); }
    // image plane at distance 1 (perspective), with the pixel columns
    const { r: rr, f } = RT.cameraBasis(cam());
    const th = Math.tan(st.fov * Math.PI / 360) * st.W / st.H;
    const e = st.eye;
    const half = st.ortho ? 2 * st.W / st.H : th;
    const pc: Vec3 = st.ortho ? e : [e[0] + f[0], e[1] + f[1], e[2] + f[2]];
    const a: Vec3 = [pc[0] - half * rr[0], 0, pc[2] - half * rr[2]], b: Vec3 = [pc[0] + half * rr[0], 0, pc[2] + half * rr[2]];
    svgEl("line", { x1: tx(a[0]), y1: tz(a[2]), x2: tx(b[0]), y2: tz(b[2]), stroke: "var(--qb-fg)", "stroke-width": 3 }, gCam);
    if (st.W <= 32) for (let i = 1; i < st.W; i++) { const s = i / st.W; svgEl("line", { x1: tx(a[0] + s * (b[0] - a[0])), y1: tz(a[2] + s * (b[2] - a[2])) - 3, x2: tx(a[0] + s * (b[0] - a[0])), y2: tz(a[2] + s * (b[2] - a[2])) + 3, stroke: "var(--qb-fg)", "stroke-width": 1 }, gCam); }
    if (!st.ortho) for (const q of [a, b]) svgEl("line", { x1: tx(e[0]), y1: tz(e[2]), x2: tx(e[0] + 6 * (q[0] - e[0])), y2: tz(e[2] + 6 * (q[2] - e[2])), stroke: "var(--qb-muted)", "stroke-width": 1, "stroke-dasharray": "4 4" }, gCam);
    // rays
    const drawNode = (nd: RT.RayNode) => {
      const t = Number.isFinite(nd.t) ? nd.t : 30;
      const end = RT.pointAt(nd.o, nd.d, t);
      svgEl("line", { x1: tx(nd.o[0]), y1: tz(nd.o[2]), x2: tx(end[0]), y2: tz(end[2]), stroke: KIND_COL[nd.kind], "stroke-width": nd.kind === "shadow" ? 1.5 : 2.5 + 1.5 * nd.weight, "stroke-dasharray": nd.kind === "shadow" ? "5 4" : "", "stroke-opacity": 0.4 + 0.6 * Math.min(1, nd.weight * 3) }, gRays);
      nd.children.forEach(drawNode);
    };
    drawNode(root);
    if (!whitted) {
      // every candidate hit of the primary ray
      sc.objects.forEach(ob => {
        const ts: [number, boolean][] = [];
        if (ob.type === "sphere") { const s = RT.solveSphere(ray.o, ray.d, ob.c, ob.r); if (s.disc >= 0) ts.push([s.t1, true], [s.t2, true]); }
        else if (ob.type === "plane") { const t = RT.hitPlane(ray.o, ray.d, ob.p, ob.n); if (Number.isFinite(t)) ts.push([t, true]); }
        else if (ob.type === "triangle") { const s = RT.solveTriangle(ray.o, ray.d, ob.a, ob.b, ob.c); if (Number.isFinite(s.t)) ts.push([s.t, RT.insideTriangle(s.u, s.v)]); }
        for (const [t, ins] of ts) {
          const p = RT.pointAt(ray.o, ray.d, t);
          if (t <= 0) svgEl("line", { x1: tx(ray.o[0]), y1: tz(ray.o[2]), x2: tx(p[0]), y2: tz(p[2]), stroke: "#d62728", "stroke-width": 1.5, "stroke-dasharray": "3 3" }, gRays);
          svgEl("circle", { cx: tx(p[0]), cy: tz(p[2]), r: 6, fill: ins && t > 0 ? "#d49b00" : "none", stroke: t > 0 ? "#000" : "#d62728", "stroke-width": 2 }, gRays);
        }
      });
    }
    // the camera handle
    const ch = svgEl("g", {}, gCam);
    svgEl("circle", { cx: tx(e[0]), cy: tz(e[2]), r: 11, fill: "#fff", stroke: "#000", "stroke-width": 2 }, ch);
    const ct = svgEl("text", { x: tx(e[0]), y: tz(e[2]) + 6, "text-anchor": "middle", "font-size": 16, "font-weight": 700 }, ch); ct.textContent = "C";
    ch.style.cursor = "grab";
    ch.addEventListener("pointerdown", ev => { dragging = true; svg.setPointerCapture(ev.pointerId); ev.preventDefault(); });
  }
  const lab = (x: number, z: number, s: string) => { const t = svgEl("text", { x: tx(x), y: tz(z) + 6, "text-anchor": "middle", "font-size": 17, "font-weight": 600 }, gScene); t.textContent = s; t.classList.add("qb-halo"); };

  let dragging = false;
  svg.addEventListener("pointermove", ev => {
    if (!dragging) return;
    const [px, py] = svgPoint(svg, ev);
    const x = X0 + px / VW * (X1 - X0), z = Z0 + py / VH * (Z1 - Z0);
    st.eye = [Math.round(Math.max(X0, Math.min(X1, x)) * 10) / 10, st.eye[1], Math.round(Math.max(Z0, Math.min(Z1, z)) * 10) / 10];
    draw();
  });
  const stop = () => { dragging = false; };
  svg.addEventListener("pointerup", stop); svg.addEventListener("pointercancel", stop);

  cv.addEventListener("pointerdown", ev => {
    const rc = cv.getBoundingClientRect();
    const i = Math.floor((ev.clientX - rc.left) / rc.width * st.W), j = Math.floor((ev.clientY - rc.top) / rc.height * st.H);
    if (i < 0 || j < 0 || i >= st.W || j >= st.H) return;
    st.inspect = [i, j];
    if (st.traced >= 0 && j * st.W + i >= st.traced) st.traced = j * st.W + i + 1;
    draw();
  });

  if (p0) pb[PRE.indexOf(p0)].classList.add("active");
  sync();
}

autoMount(NAME, host => { injectStyle(); mount(host); });
