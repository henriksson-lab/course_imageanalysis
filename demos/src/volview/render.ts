// volview, mode "render" (3D-graphics day, unit `volrender/`): ray casting through a synthetic
// head (lib/volrender.ts headPhantom: 64 × 64 × 40 voxels of 1 × 1 × 1.6 mm with a skull, brain,
// ventricles, a bright vessel tree and a dark air cavity). Modes: MIP, MinIP, average (≈ an
// X-ray), direct volume rendering with a transfer function (front-to-back compositing), and a
// first-hit isosurface. Transfer-function editor over the histogram; step size, opacity
// correction, early termination (with a samples-per-pixel heat map), nearest/trilinear
// sampling, z spacing correct/ignored, orthographic/perspective. Drag to rotate.
// The samples-per-pixel numbers in the readout are computed on the CPU (48 × 48 rays, the same
// algorithm), so they do not depend on the graphics card.
import { htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { VolumeRenderer, headPhantom, transferEditor, TF, RayMode, castImage, RenderSettings } from "../lib/volrender";
import { voxelToWorld, formatAffine } from "../lib/volume";
import type { StandardView } from "../lib/gl3d";

const STYLE = `
.qb-demo[data-mode="render"] .vr-left { flex: 0 1 440px; min-width: 260px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-mode="render"] .qb-panel { flex: 1 1 340px; max-width: 520px; gap: .3em; }
.qb-demo[data-mode="render"] .vr-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
.qb-demo[data-mode="render"] .vr-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-mode="render"] .qb-readout { font-size: 14px; white-space: pre; }
.qb-demo[data-mode="render"] .qb-tfedit { max-width: 400px; }
.reveal .qb-demo[data-mode="render"] { flex-wrap: nowrap; font-size: 17px; }
.reveal .qb-demo[data-mode="render"] .vr-left { flex: 0 0 420px; }
.reveal .qb-demo[data-mode="render"] .qb-panel { flex: 1 1 600px; max-width: 680px; gap: .12em; }
.reveal .qb-demo[data-mode="render"] .qb-btn { font-size: 16px; padding: .02em .35em; }
.reveal .qb-demo[data-mode="render"] .qb-readout { font-size: 13px; padding: .2em .4em; }
.reveal .qb-demo[data-mode="render"] .qb-tfedit { max-width: 360px; }
.reveal .qb-demo[data-mode="render"] .vr-cols { display: flex; gap: .6em; align-items: flex-start; }
`;

const SKULL: TF = { points: [[140, 0], [165, 0.9], [255, 0.9]], map: "bone" };
const SOFT: TF = { points: [[45, 0], [60, 0.03], [115, 0.03], [135, 0], [150, 0], [170, 0.12], [200, 0.12], [222, 0.95]], map: "bone" };
const VESSELS: TF = { points: [[180, 0], [215, 0.9]], map: "inferno" };

interface VState extends RenderSettings { tf: TF; view: StandardView | [number, number]; ortho: boolean }
const DEF: VState = { mode: "dvr", step: 0.5, correction: true, eps: 0, iso: 135, nearest: false, ignoreSpacing: false, showCount: false, tf: SOFT, view: [0.5, 0.25], ortho: false };
const PRESETS: { id: string; label: string; s: Partial<VState> }[] = [
  { id: "mip", label: "MIP finds the vessels", s: { mode: "mip", view: "front" } },
  { id: "mipside", label: "MIP has no depth: rotate", s: { mode: "mip", view: "right" } },
  { id: "minip", label: "MinIP finds the air", s: { mode: "minip", view: "front" } },
  { id: "avg", label: "average = an X-ray", s: { mode: "avg", view: "front" } },
  { id: "skull", label: "only the skull", s: { mode: "dvr", tf: SKULL } },
  { id: "soft", label: "semi-transparent soft tissue", s: { mode: "dvr", tf: SOFT } },
  { id: "vessels", label: "vessels only", s: { mode: "dvr", tf: VESSELS } },
  { id: "wood", label: "step too large: wood grain", s: { mode: "dvr", tf: SKULL, step: 3, view: [0.3, 0.15] } },
  { id: "nocorr", label: "no opacity correction", s: { mode: "dvr", tf: SOFT, step: 0.25, correction: false } },
  { id: "early", label: "early termination saves samples", s: { mode: "dvr", tf: SKULL, eps: 0.05, showCount: true } },
  { id: "zspacing", label: "ignored z spacing", s: { mode: "dvr", tf: SKULL, ignoreSpacing: true, view: "right" } },
  { id: "iso", label: "isosurface by ray casting", s: { mode: "iso", iso: 135, tf: SKULL } },
];

function injectStyle(): void {
  if (document.getElementById("qb-style-volview-render")) return;
  const s = document.createElement("style");
  s.id = "qb-style-volview-render"; s.textContent = STYLE;
  document.head.appendChild(s);
}

export function mountRender(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init = (): VState => ({ ...DEF, ...(p0?.s ?? {}) });
  let st = init();
  const vol = headPhantom();

  const left = htmlEl("div", "vr-left", "", host);
  const R = new VolumeRenderer(left, "The figures on this page show the same views.");
  const vrow = htmlEl("div", "vr-row", "", left);
  for (const [v, t] of [["front", "front"], ["right", "side"], ["top", "top"], ["iso", "3/4 view"]] as [StandardView, string][]) button(vrow, t, () => { st.view = v; applyView(); stats(); });
  const orthoC = checkbox(vrow, "orthographic", st.ortho, v => { st.ortho = v; applyView(); stats(); });

  const panel = htmlEl("div", "qb-panel", "", host);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { st = { ...DEF, ...p.s }; sync(); } })));
  const mrow = htmlEl("div", "vr-row", "", panel);
  htmlEl("span", "vr-lab", "mode", mrow);
  const MODES: [RayMode, string][] = [["mip", "MIP"], ["minip", "MinIP"], ["avg", "average"], ["dvr", "volume rendering"], ["iso", "isosurface"]];
  const mBtns = MODES.map(([m, t]) => button(mrow, t, () => { st.mode = m; apply(); }));
  const cols = htmlEl("div", "vr-cols", "", panel);
  const c1 = htmlEl("div", "", "", cols), c2 = htmlEl("div", "", "", cols);
  const ed = transferEditor(c1, st.tf, vol.data, tf => { st.tf = tf; R.setTF(tf); stats(); });
  const stepS = slider(c2, "step Δs (voxels)", 0.25, 4, 0.25, st.step, v => { st.step = v; apply(); });
  const isoS = slider(c2, "isovalue c", 0, 255, 1, st.iso, v => { st.iso = v; apply(); });
  const epsS = slider(c2, "stop when T <", 0, 0.2, 0.01, st.eps, v => { st.eps = v; apply(); }, v => (v === 0 ? "off" : v.toFixed(2)));
  const chk = htmlEl("div", "vr-row", "", c2);
  const corrC = checkbox(chk, "opacity correction", st.correction, v => { st.correction = v; apply(); });
  const countC = checkbox(chk, "samples per pixel", st.showCount, v => { st.showCount = v; apply(); });
  const nearC = checkbox(chk, "nearest voxel", st.nearest, v => { st.nearest = v; apply(); });
  const spC = checkbox(chk, "ignore z spacing", st.ignoreSpacing, v => { st.ignoreSpacing = v; apply(); });
  button(c2, "reset", () => { st = init(); pb.forEach(b => b.classList.remove("active")); sync(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function applyView(): void {
    const c = R.camera;
    if (Array.isArray(st.view)) { c.azimuth = st.view[0]; c.elevation = st.view[1]; } else c.setView(st.view);
    c.distance = 2.5; c.orthographic = st.ortho;
    R.render();
  }
  function apply(): void {
    mBtns.forEach((b, i) => b.classList.toggle("active", MODES[i][0] === st.mode));
    R.set({ mode: st.mode, step: st.step, correction: st.correction, eps: st.eps, iso: st.iso, nearest: st.nearest, ignoreSpacing: st.ignoreSpacing, showCount: st.showCount });
    stats();
  }
  function sync(): void {
    stepS.set(st.step); isoS.set(st.iso); epsS.set(st.eps); corrC.checked = st.correction; countC.checked = st.showCount; nearC.checked = st.nearest; spC.checked = st.ignoreSpacing; orthoC.checked = st.ortho;
    ed.set(st.tf); R.setTF(st.tf);
    applyView(); apply();
  }

  let timer = 0;
  function stats(): void {
    clearTimeout(timer);
    timer = window.setTimeout(() => {
      const A = voxelToWorld(st.ignoreSpacing ? { ...vol, spacing: [1, 1, 1] } : vol);
      const lines = [`voxel → world A (mm), ${vol.nx} × ${vol.ny} × ${vol.nz} voxels:`, formatAffine(A, 1)];
      if (st.mode === "dvr" || st.mode === "iso") {
        const N = 48, r = castImage(vol, st.tf, R.pv(), N, N, { mode: st.mode, step: st.step, correction: st.correction, eps: st.eps, iso: st.iso, ignoreSpacing: st.ignoreSpacing, nearest: st.nearest });
        let u = 0, f = 0, n = 0, ue = 0, fe = 0, ne = 0;
        for (let i = 0; i < N * N; i++) if (r.full[i] > 0) {
          u += r.used[i]; f += r.full[i]; n++;
          if (r.used[i] < r.full[i]) { ue += r.used[i]; fe += r.full[i]; ne++; }
        }
        const pc = (a: number, b: number) => (b ? (100 * (1 - a / b)).toFixed(0) : "0");
        lines.push(`CPU check, ${N} × ${N} rays: ${n} cross the box`, `samples per ray: ${(u / Math.max(1, n)).toFixed(1)} of ${(f / Math.max(1, n)).toFixed(1)} (saved ${pc(u, f)} %)`);
        if (st.mode === "dvr") lines.push(`${ne} rays stop early, saving ${pc(ue, fe)} % of their samples`);
      } else lines.push("projections use every sample on the ray");
      ro.textContent = lines.join("\n");
    }, 120);
  }
  R.onCamera = () => { st.view = [R.camera.azimuth, R.camera.elevation]; stats(); };
  R.setVolume(vol);
  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  sync();
}
