// Tests for the maskfill demo (unit pretext/): the masks, the interpolation and PCA fills
// (visible pixels untouched, k = 0 = the mean window, orthonormal components that are
// eigenvectors of the covariance), and the preset numbers quoted on the pages.
// Run alone (from the repository root):
//   node_modules/.bin/esbuild demos/test/maskfill-test.ts --bundle --platform=node --log-level=warning | node
import { fitPca, fillInterp, fillPca, hiddenPsnr, makeMask, scene, DW, N, P, WIN } from "../src/maskfill/model";
import { PRESETS } from "../src/maskfill/presets";
import { loadCellNode } from "./selfsup-cell";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 25) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

const cell = loadCellNode().data;
const nuc = scene("nuclei"), model = fitPca(nuc.windows);

// ---- masks ----
{
  const m = makeMask("patches", 0.75, 7);
  let ok = true;
  for (let gy = 0; gy < N / P; gy++) for (let gx = 0; gx < N / P; gx++) {
    const v = m[gy * P * N + gx * P];
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) if (m[(gy * P + y) * N + gx * P + x] !== v) ok = false;
    if ((gy === 0 || gx === 0 || gy === N / P - 1 || gx === N / P - 1) && v) ok = false;
  }
  check(ok, "patch mask: whole patches, outer ring never hidden");
  let h = 0; for (const v of m) h += v;
  check(h === 8460, `75 % of the 324 inner patches: ${h / 36} patches hidden (235 expected)`);
}

// ---- fills leave the visible pixels alone ----
{
  const m = makeMask("patches", 0.5, 3);
  const a = fillInterp(nuc.test, m), b = fillPca(nuc.test, m, model, 16);
  let ok = true; for (let i = 0; i < m.length; i++) if (!m[i] && (a[i] !== nuc.test[i] || b[i] !== nuc.test[i])) ok = false;
  check(ok, "visible pixels unchanged by both fills");
  const b0 = fillPca(nuc.test, m, model, 0);
  let ok0 = true;
  for (let gy = 1; gy < 19; gy++) for (let gx = 1; gx < 19; gx++) for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
    const i = (gy * P + y) * N + gx * P + x;
    if (m[i] && !near(b0[i], model.mean[(P + y) * WIN + P + x], 1e-9)) ok0 = false;
  }
  check(ok0, "k = 0 fills with the mean window");
}

// ---- the components: orthonormal eigenvectors of the covariance ----
{
  const n = nuc.windows.length, mean = model.mean;
  let worstOrth = 0;
  for (let a = 0; a < 8; a++) for (let b = 0; b <= a; b++) {
    let s = 0; for (let i = 0; i < DW; i++) s += model.comps[a][i] * model.comps[b][i];
    worstOrth = Math.max(worstOrth, Math.abs(s - (a === b ? 1 : 0)));
  }
  check(worstOrth < 1e-9, `components orthonormal (${worstOrth.toExponential(1)})`);
  // C u = λ u for the first component; components 2 and 3 have almost equal eigenvalues, so
  // only the plane they span is well defined: C u must stay in span(u1, u2, u3).
  for (let a = 0; a < 3; a++) {
    const u = model.comps[a], Cu = new Float64Array(DW);
    for (const w of nuc.windows) { let t = 0; for (let i = 0; i < DW; i++) t += (w[i] - mean[i]) * u[i]; for (let i = 0; i < DW; i++) Cu[i] += t * (w[i] - mean[i]) / n; }
    const r = Float64Array.from(Cu);
    for (let b = 0; b < (a === 0 ? 1 : 3); b++) { let t = 0; for (let i = 0; i < DW; i++) t += r[i] * model.comps[b][i]; for (let i = 0; i < DW; i++) r[i] -= t * model.comps[b][i]; }
    let err = 0, nrm = 0; for (let i = 0; i < DW; i++) { err += r[i] ** 2; nrm += Cu[i] ** 2; }
    check(Math.sqrt(err / nrm) < 1e-3, `component ${a + 1}: C u in the span of the leading components (relative residual ${Math.sqrt(err / nrm).toExponential(1)})`);
  }
}

// ---- the presets (numbers on the pages) ----
{
  const cm = scene("cell", cell), cellModel = fitPca(cm.windows);
  const want: Record<string, [number, number]> = { p25: [19.80, 24.46], p75: [18.48, 20.53], p90: [15.66, 17.05], pixels: [23.66, 29.02], k0: [18.48, 13.32], cell: [17.01, 14.61] };
  for (const p of PRESETS) {
    const [m, t] = p.s.scene === "nuclei" ? [model, nuc.test] : [cellModel, cm.test];
    const mask = makeMask(p.s.kind, p.s.ratio, p.s.seed);
    const a = hiddenPsnr(fillInterp(t, mask), t, mask), b = hiddenPsnr(fillPca(t, mask, m, p.s.k), t, mask);
    check(near(a, want[p.id][0], 0.005) && near(b, want[p.id][1], 0.005), `${p.id}: interpolation ${a.toFixed(2)}, learnt ${b.toFixed(2)} (expected ${want[p.id].join(", ")})`);
  }
}

console.log(`maskfill: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
