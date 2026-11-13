// Triangulation in flatland: a top-down 2D world with 1D cameras (lib/flatland.ts). Drag the
// cameras (centre handle), their heading (small handle in front) and the point. Each camera
// shows its pixel row, the ray through the observed pixel and a wedge of ±δ pixels; the
// wedges meet in the uncertainty region (shaded). A cloud of 200 triangulations with seeded
// pixel noise σ shows the actual spread. Side plot: RMS error vs depth along the current
// viewing direction, with the rule of thumb √2 σ Z² / (f B). A third camera can be switched on
// (then midpoint/least-squares and optimal triangulation differ).
//
// <div data-demo="triangulate" data-preset="short|wide|far|parallel|third"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Cam1D, P, project1d, lookAt1d, triangulateRays, triangulateOptimal, uncertaintyQuad, rayDir } from "../lib/flatland";
import { Plot } from "../lib/plot";
import { rng } from "../lib/random";
import { FlatView, injectStyle, CAM_COLORS } from "./flatview";

const NAME = "triangulate";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { flex-wrap: nowrap; gap: .8em; }
${SEL} .tr-main { flex: 0 1 500px; min-width: 260px; }
${SEL} .tr-side { flex: 0 1 330px; min-width: 220px; display: flex; flex-direction: column; gap: .4em; }
${SEL} .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .4em; }
${SEL} .qb-readout { font-size: 15px; }
${SEL} .tr-cap { font-size: 16px; color: var(--qb-muted); }
.reveal ${SEL} { font-size: 18px; }
.reveal ${SEL} .tr-main { flex: 0 0 470px; }
.reveal ${SEL} .tr-side { flex: 0 0 310px; }
.reveal ${SEL} .qb-panel { gap: .25em; }
.reveal ${SEL} .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal ${SEL} .qb-readout { font-size: 16px; padding: .25em .45em; }
@media (max-width: 860px) { ${SEL} { flex-wrap: wrap; } }
`;

const F = 200, N = 160;
interface CamState { pos: P; th: number }
interface State { cams: CamState[]; p: P; delta: number; sigma: number; third: boolean; method: "ls" | "optimal"; cloud: boolean }
const aim = (pos: P, t: P): CamState => ({ pos, th: Math.atan2(t[1] - pos[1], t[0] - pos[0]) });
function setup(c1: P, c2: P, p: P, extra: Partial<State> = {}, c3: P = [-2.6, 0.4]): State {
  return { cams: [aim(c1, p), aim(c2, p), aim(c3, p)], p, delta: 1, sigma: 1, third: false, method: "optimal", cloud: true, ...extra };
}
const DEFAULT = (): State => setup([-0.6, -3], [0.6, -3], [0, 0.4]);
interface Preset { id: string; label: string; s: () => State }
const PRESETS: Preset[] = [
  { id: "short", label: "short baseline: a long thin region", s: () => setup([-0.25, -3], [0.25, -3], [0, 0.4]) },
  { id: "wide", label: "wide baseline", s: () => setup([-2.2, -2.4], [2.2, -2.4], [0, 0.4]) },
  { id: "far", label: "point far away", s: () => setup([-0.6, -3], [0.6, -3], [0, 2.3]) },
  { id: "parallel", label: "nearly parallel rays", s: () => setup([-0.12, -3], [0.12, -3], [0, 1.6]) },
  { id: "third", label: "a third view", s: () => setup([-0.4, -3], [0.4, -3], [0, 0.6], { third: true }, [-2.7, 0.6]) },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => (p0 ? p0.s() : DEFAULT());
  let st = start();

  const main = htmlEl("div", "tr-main", "", host);
  const view = new FlatView(main, 520, 500, -3, 3, -3.6, 2.7);
  htmlEl("div", "tr-cap", "drag the cameras (big circles), their heading (small circles) and the point (black)", main);
  const side = htmlEl("div", "tr-side", "", host);
  const plot = new Plot(side, { width: 330, height: 250, x: [0, 6], y: [0, 0.3], xlabel: "depth Z (distance to baseline)", ylabel: "RMS error", font: 17, margin: { l: 62, r: 10, t: 10, b: 52 } });
  const rule = plot.line([], [], { color: "var(--qb-c2)", width: 2.5, dash: "6 4", label: "√2σZ²/(fB)" });
  const sim = plot.scatter([], [], { color: "var(--qb-c1)", r: 4, label: "simulated" });
  const here = plot.scatter([], [], { color: "#111", r: 7 });
  plot.legend(undefined, "tl");
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtn = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { st = p.s(); sync(); draw(); } })));
  const dS = slider(panel, "wedge ±δ (px)", 0.5, 4, 0.5, st.delta, v => { st.delta = v; draw(); });
  const sS = slider(panel, "noise σ (px)", 0, 3, 0.25, st.sigma, v => { st.sigma = v; draw(); });
  const thirdC = checkbox(panel, "third camera", st.third, v => { st.third = v; draw(); });
  const cloudC = checkbox(panel, "cloud of 200 noisy triangulations", st.cloud, v => { st.cloud = v; draw(); });
  const mrow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = [
    button(mrow, "least squares on rays", () => { st.method = "ls"; draw(); }),
    button(mrow, "optimal (reprojection)", () => { st.method = "optimal"; draw(); }),
  ];
  button(panel, "reset", () => { st = start(); pbtn.forEach(b => b.classList.remove("active")); sync(); draw(); });
  function sync(): void { dS.set(st.delta); sS.set(st.sigma); thirdC.checked = st.third; cloudC.checked = st.cloud; }

  const cam = (c: CamState): Cam1D => ({ x: c.pos[0], y: c.pos[1], theta: c.th, f: F, c: (N - 1) / 2, n: N });
  const hCam = st.cams.map((_, k) => view.handle(11, CAM_COLORS[k], q => { st.cams[k].pos = q; draw(); }, "drag: move camera"));
  const hDir = st.cams.map((_, k) => view.handle(7, CAM_COLORS[k], q => { const c = st.cams[k]; c.th = Math.atan2(q[1] - c.pos[1], q[0] - c.pos[0]); draw(); }, "drag: turn camera"));
  const hP = view.handle(10, "#111", q => { st.p = q; draw(); }, "drag: move the point");

  function triang(cams: Cam1D[], us: number[]): P {
    return st.method === "optimal" ? triangulateOptimal(cams, us) : triangulateRays(cams, us);
  }

  function draw(): void {
    view.clear();
    mBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.method === "ls")));
    const nc = st.third ? 3 : 2, cams = st.cams.slice(0, nc).map(cam);
    const obs = cams.map(c => project1d(c, st.p));
    const vis = obs.map(o => o.Z > 0 && o.u >= 0 && o.u <= N - 1);
    // wedges
    cams.forEach((c, k) => {
      if (!vis[k]) return;
      const L = 9, d0 = rayDir(c, obs[k].u - st.delta), d1 = rayDir(c, obs[k].u + st.delta);
      view.poly([[c.x, c.y], [c.x + L * d0[0], c.y + L * d0[1]], [c.x + L * d1[0], c.y + L * d1[1]]], CAM_COLORS[k], "none", 0.13);
    });
    // uncertainty quadrilateral of the first two cameras
    let quadLen = NaN, quadWid = NaN;
    if (vis[0] && vis[1]) {
      const q = uncertaintyQuad(cams[0], obs[0].u, cams[1], obs[1].u, st.delta);
      if (q.every(v => Number.isFinite(v[0]) && Math.hypot(v[0] - st.p[0], v[1] - st.p[1]) < 40)) {
        view.poly(q, "#f2b705", "#b8860b", 0.55, 1.5);
        quadLen = Math.max(Math.hypot(q[0][0] - q[2][0], q[0][1] - q[2][1]), Math.hypot(q[1][0] - q[3][0], q[1][1] - q[3][1]));
        quadWid = Math.min(Math.hypot(q[0][0] - q[2][0], q[0][1] - q[2][1]), Math.hypot(q[1][0] - q[3][0], q[1][1] - q[3][1]));
      }
    }
    // rays
    cams.forEach((c, k) => { if (vis[k]) { const d = rayDir(c, obs[k].u); view.line([c.x, c.y], [c.x + 9 * d[0], c.y + 9 * d[1]], CAM_COLORS[k], 1.8); } });
    // cloud
    const r = rng(42), cloud: P[] = [];
    if (st.cloud && vis.every(v => v)) {
      for (let t = 0; t < 200; t++) {
        const us = obs.map(o => o.u + st.sigma * r.norm());
        const q = triang(cams, us);
        if (Number.isFinite(q[0])) cloud.push(q);
      }
      for (const q of cloud) view.dot(q, 2.2, "rgba(20,20,20,0.55)");
    }
    // cameras with observed pixel
    cams.forEach((c, k) => {
      view.camera(c, CAM_COLORS[k], { len: 0.55, label: `camera ${k + 1}`, frustum: 7 });
      if (vis[k]) view.dot(view.imagePoint(c, obs[k].u, 0.55), 5, "#111");
    });
    view.dot(st.p, 6, "#111");
    st.cams.forEach((c, k) => {
      const on = k < nc;
      hCam[k].show(on); hDir[k].show(on);
      hCam[k].set(c.pos); hDir[k].set([c.pos[0] + 1.0 * Math.cos(c.th), c.pos[1] + 1.0 * Math.sin(c.th)]);
    });
    hP.set(st.p);

    // readouts
    const B = Math.hypot(st.cams[1].pos[0] - st.cams[0].pos[0], st.cams[1].pos[1] - st.cams[0].pos[1]);
    const m: P = [(st.cams[0].pos[0] + st.cams[1].pos[0]) / 2, (st.cams[0].pos[1] + st.cams[1].pos[1]) / 2];
    const Z = Math.hypot(st.p[0] - m[0], st.p[1] - m[1]);
    const dir: P = [(st.p[0] - m[0]) / Z, (st.p[1] - m[1]) / Z];
    const a0 = Math.atan2(st.p[1] - st.cams[0].pos[1], st.p[0] - st.cams[0].pos[0]), a1 = Math.atan2(st.p[1] - st.cams[1].pos[1], st.p[0] - st.cams[1].pos[0]);
    let ang = Math.abs(a0 - a1) * 180 / Math.PI; if (ang > 180) ang = 360 - ang;
    let along = 0, across = 0;
    for (const q of cloud) { const e: P = [q[0] - st.p[0], q[1] - st.p[1]]; along += (e[0] * dir[0] + e[1] * dir[1]) ** 2; across += (e[0] * dir[1] - e[1] * dir[0]) ** 2; }
    along = Math.sqrt(along / Math.max(1, cloud.length)); across = Math.sqrt(across / Math.max(1, cloud.length));
    ro.textContent =
      `baseline B = ${B.toFixed(2)}, depth Z = ${Z.toFixed(2)}\nangle between the rays: ${ang.toFixed(1)}°\n` +
      (Number.isFinite(quadLen) ? `region (±${st.delta} px): ${quadLen.toFixed(3)} long, ${quadWid.toFixed(3)} wide\n` : "region: rays (nearly) parallel\n") +
      (vis.every(v => v) ? (st.cloud ? `cloud RMS error: ${along.toFixed(3)} in depth, ${across.toFixed(3)} sideways` : "") : "the point is outside a camera's view");

    // error vs depth along the current direction
    const zs: number[] = [], es: number[] = [], rx: number[] = [], ry: number[] = [];
    let emax = 0;
    for (let z = 0.5; z <= 6.001; z += 0.25) {
      const q: P = [m[0] + z * dir[0], m[1] + z * dir[1]], o = cams.map(c => project1d(c, q));
      if (!o.every(v => v.Z > 0 && v.u >= 0 && v.u <= N - 1) || st.sigma === 0) continue;
      const rr = rng(7);
      let s = 0, n = 0;
      for (let t = 0; t < 200; t++) {
        const x = triang(cams, o.map(v => v.u + st.sigma * rr.norm()));
        if (Number.isFinite(x[0])) { s += (x[0] - q[0]) ** 2 + (x[1] - q[1]) ** 2; n++; }
      }
      zs.push(z); es.push(Math.sqrt(s / Math.max(1, n)));
    }
    for (let z = 0.2; z <= 6.001; z += 0.1) { rx.push(z); ry.push(Math.SQRT2 * st.sigma * z * z / (F * Math.max(B, 1e-6))); }
    emax = Math.max(0.05, ...es.slice(0, 40));
    plot.setDomain([0, 6], [0, Math.min(2, 1.15 * emax)]);
    rule.set(rx, ry);
    sim.set(zs, es);
    const ez = st.cloud && vis.every(v => v) ? Math.hypot(along, across) : NaN;
    here.set(Number.isFinite(ez) && Z <= 6 ? [Z] : [], Number.isFinite(ez) ? [ez] : []);
  }

  if (p0) pbtn[PRESETS.indexOf(p0)].classList.add("active");
  sync();
  draw();
}

autoMount(NAME, mount);
