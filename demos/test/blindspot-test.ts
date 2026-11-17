// Tests for the blindspot demo (unit self-denoise/): the window, the least-squares filter fit
// against a direct least-squares solve, the identity without a blind spot, Noise2Void = the
// best blind-spot filter and Noise2Noise = supervised for independent noise, the failure on
// correlated noise, mixing the pixel back (exercise B6), and the numbers quoted on the pages.
// Run alone (from the repository root):
//   node_modules/.bin/esbuild demos/test/blindspot-test.ts --bundle --platform=node --log-level=warning | node
import { window, fitFilter, applyFilter, mixBack, noiseField, evaluate, W } from "../src/blindspot/model";
import { PRESETS } from "../src/blindspot/presets";
import { lstsq } from "../src/lib/linalg";
import { rng } from "../src/lib/ui";
import { cellCrop } from "./selfsup-cell";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 25) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ---- the window (mirror border, the edge pixel repeated, as in the course library) ----
{
  const g = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  check(window(g, 3, 3, 1, 1, 1, false).join() === "1,2,3,4,5,6,7,8,9", "window at the centre");
  check(window(g, 3, 3, 1, 1, 1, true).join() === "1,2,3,4,6,7,8,9", "blind window leaves out the centre");
  check(window(g, 3, 3, 0, 0, 1, false).join() === "1,1,2,1,1,2,4,4,5", "mirror border at the corner");
}

// ---- the fit equals a direct least-squares solve ----
{
  const w = 12, h = 10, r = rng(5);
  const a = Float64Array.from({ length: w * h }, () => r.unif()), t = Float64Array.from({ length: w * h }, () => r.unif());
  for (const blind of [false, true]) {
    const f = fitFilter(a, t, w, h, 1, blind);
    const d = (blind ? 8 : 9) + 1, A = new Float64Array(w * h * d);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = window(a, w, h, x, y, 1, blind);
      for (let i = 0; i < d - 1; i++) A[(y * w + x) * d + i] = v[i];
      A[(y * w + x) * d + d - 1] = 1;
    }
    const sol = lstsq(A, t, w * h, d);
    const mine = [...Array.from(f.weights).filter((_, i) => !(blind && i === 4)), f.bias];
    check(mine.every((v, i) => near(v, sol[i], 1e-8)), `normal equations = least squares (blind ${blind})`);
    check(!blind || f.weights[4] === 0, "blind spot: centre weight 0");
  }
}

// ---- the micrograph: identity, Noise2Void = best blind-spot filter, Noise2Noise = supervised ----
const clean = cellCrop(64, 64, W, W);
{
  const n = noiseField(W, W, 25, "white", 101), noisy = clean.map((v, i) => v + n[i]);
  const id = fitFilter(noisy, noisy, W, W, 2, false);
  check(near(id.weights[12], 1, 1e-6) && Array.from(id.weights).every((v, i) => i === 12 || Math.abs(v) < 1e-6), "no blind spot: the identity");
  const out = applyFilter(noisy, W, W, id);
  check(out.every((v, i) => near(v, noisy[i], 1e-6)), "the identity filter returns the noisy image");
  for (const kind of ["white", "corr", "stripes"] as const) {
    const f = noiseField(W, W, 25, kind, 7);
    let s = 0, m = 0; for (const v of f) { s += v * v; m += v; }
    check(near(Math.sqrt(s / f.length), 25, kind === "white" ? 0.5 : kind === "corr" ? 1e-9 : 3), `noise ${kind}: sd ${Math.sqrt(s / f.length).toFixed(2)} ≈ 25`);
    check(Math.abs(m / f.length) < 3, `noise ${kind}: mean near 0`);
  }
}

// ---- mixing the pixel back (B6): residuals 0.2 and 0, sigma 0.1: c = 0.5 ----
{
  const m = mixBack([0.6, 0.3], [0.4, 0.3], 0.1);
  check(near(m.c, 0.5, 1e-12) && near(m.out[0], 0.5, 1e-12) && near(m.out[1], 0.3, 1e-12), "B6: c = 0.5, halfway");
  check(mixBack([0.6, 0.3], [0.4, 0.3], 0.2).c === 0, "c clipped at 0");
}

// ---- the presets: the numbers quoted on the pages (PSNR in dB, peak 255) ----
{
  const res = Object.fromEntries(PRESETS.filter(p => p.img === "cell").map(p => [p.id, evaluate(clean, p.s)]));
  const a = res.n2v.psnr;
  check(near(a.noisy, 20.24, 0.005) && near(a.out, 21.10, 0.005), `n2v: noisy ${a.noisy.toFixed(2)}, out ${a.out.toFixed(2)}`);
  check(near(a.all.n2v, a.all.supblind, 0.05), "white noise: Noise2Void ≈ the best blind-spot filter");
  check(near(a.all.n2n, a.all.sup, 0.05), "white noise: Noise2Noise ≈ supervised");
  check(near(a.all.n2n, 23.75, 0.005) && near(a.all.sup, 23.76, 0.005) && near(a.cmp, 23.00, 0.005), `n2n ${a.all.n2n.toFixed(2)}, sup ${a.all.sup.toFixed(2)}, Gaussian ${a.cmp.toFixed(2)}`);
  check(near(res.mix.psnr.out, 23.75, 0.005) && near(res.mix.c, 0.45, 0.005), `pixel back: ${res.mix.psnr.out.toFixed(2)} dB, c = ${res.mix.c.toFixed(2)}`);
  const c = res.corr.psnr;
  check(c.out < c.noisy && near(c.out, 18.56, 0.005) && near(c.noisy, 20.17, 0.005), `correlated: Noise2Void ${c.out.toFixed(2)} below the noisy ${c.noisy.toFixed(2)}`);
  check(near(c.all.n2n, 22.06, 0.005), `correlated: Noise2Noise ${c.all.n2n.toFixed(2)}`);
  const st = res.stripes.psnr;
  check(st.out < st.noisy && near(st.out, 19.81, 0.005), `stripes: Noise2Void ${st.out.toFixed(2)} below the noisy ${st.noisy.toFixed(2)}`);
}

console.log(`blindspot: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
