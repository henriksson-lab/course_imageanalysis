// Tests for lib/edges.ts and the edges-day demos (gaussderiv, canny, scale1d, blobscale,
// hessian, structtensor): reference / brute-force checks. Run alone:
//   node_modules/.bin/esbuild demos/test/edges-test.ts --bundle --platform=node --log-level=warning | node
import {
  Grey, createGrey, fromFunction, gaussianKernel, gaussianBlur, gaussianDerivative, correlate2d, logFilter, dogFilter,
  laplacianPyramid, collapsePyramid, combine,
} from "../src/lib/imageops";
import {
  correlate1d, gaussian1d, boxKernel1d, countExtrema, nonMaxSuppression, hysteresis, canny, quantiseDirection, nmsNeighbours,
  zeroCrossings, hessian, hessEig, classifyHessianPixel, HessClass, frangiValue, frangi, structureTensor, harris, minEig,
  normLoGStack, geomScales, extrema3D, localMaxima2D, edgeList, screenAngle,
} from "../src/lib/edges";
import { label2d } from "../src/lib/label";
import { rng } from "../src/lib/ui";
import { SIGNALS as S1D, makeSignal } from "../src/scale1d/signals";
import { makeProfile, PROFILES } from "../src/gaussderiv/profile";
import { makeScene } from "../src/canny/scene";
import { makeBlobImage } from "../src/blobscale/scene";
import { responseStack } from "../src/blobscale/stack";
import { makeHessScene } from "../src/hessian/scene";
import { makeTexture } from "../src/structtensor/scene";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ---- kernels ----
for (const s of [0.7, 1, 2, 4, 8]) {
  const k0 = gaussianKernel(s, 0), k1 = gaussianKernel(s, 1), k2 = gaussianKernel(s, 2), r = k0.length >> 1;
  check(near(k0.reduce((a, b) => a + b, 0), 1, 1e-12), `order 0 sums to 1 (σ=${s})`);
  let ramp = 0, par = 0;
  for (let i = -r; i <= r; i++) { ramp += k1[i + r] * i; par += k2[i + r] * i * i / 2; }
  check(near(ramp, 1, 1e-12), `order 1 on a ramp gives 1 (σ=${s})`);
  check(near(par, 1, 1e-12), `order 2 on x²/2 gives 1 (σ=${s})`);
}

// ---- separable = direct 2D ----
{
  const r = rng(3), img = fromFunction(40, 30, () => r.unif() * 100);
  for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]] as [0 | 1 | 2, 0 | 1 | 2][]) {
    const s = 1.5, kx = gaussianKernel(s, ox), ky = gaussianKernel(s, oy), n = kx.length;
    const k2 = new Float64Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) k2[j * n + i] = ky[j] * kx[i];
    const a = gaussianDerivative(img, s, ox, oy), b = correlate2d(img, k2, n, n);
    let err = 0;
    for (let i = 0; i < a.data.length; i++) err = Math.max(err, Math.abs(a.data[i] - b.data[i]));
    check(err < 1e-3, `separable = 2D for order (${ox},${oy}), max err ${err}`);
  }
}

// ---- 1D helpers ----
{
  const f = Array.from({ length: 50 }, (_, i) => i);
  const d = gaussian1d(f, 2, 1);
  check(near(d[25], 1, 1e-9), "1D order-1 on ramp = 1");
  const d0 = gaussian1d(f, 0, 1);
  check(near(d0[25], 1, 1e-12), "1D central difference on ramp = 1");
  const c = correlate1d([0, 0, 3, 6, 6], [1, -2, 1]);
  check(c[1] === 3 && c[2] === 0 && c[3] === -3, "curvature of 0 0 3 6 6 (kernels slide)");
  const bk = boxKernel1d(2);
  check(near(bk.reduce((a, b) => a + b, 0), 1, 1e-12) && bk.length % 2 === 1, "box kernel odd, sums to 1");
  check(countExtrema([0, 1, 0, 1, 0]) === 3 && countExtrema([0, 1, 1, 2]) === 0, "countExtrema");
}

// ---- gaussderiv profile: the finite difference is noisy, σ = 2 cleans it ----
{
  for (const p of PROFILES) { const f = makeProfile(p.id, 10, 1); check(f.length === 256, `profile ${p.id} length`); }
  const f = makeProfile("mixed", 10, 1);
  const fd = gaussian1d(f, 0, 1), g2 = gaussian1d(f, 2, 1);
  const sd = (a: Float64Array, lo: number, hi: number) => { let m = 0; for (let i = lo; i < hi; i++) m += a[i]; m /= hi - lo; let v = 0; for (let i = lo; i < hi; i++) v += (a[i] - m) ** 2; return Math.sqrt(v / (hi - lo)); };
  // flat stretch 0..25 of the mixed profile
  check(sd(fd, 2, 24) > 2.5 * sd(g2, 8, 20), `σ = 2 reduces derivative noise (${sd(fd, 2, 24).toFixed(2)} vs ${sd(g2, 8, 20).toFixed(2)})`);
}

// ---- NMS on a smoothed disc: a 1-pixel ring ----
{
  const N = 64, R = 18;
  const disc = fromFunction(N, N, (x, y) => Math.hypot(x - 31.5, y - 31.5) < R ? 200 : 50);
  for (const sub of [false, true]) {
    const gx = gaussianDerivative(disc, 2, 1, 0), gy = gaussianDerivative(disc, 2, 0, 1), mag = combine(gx, gy, Math.hypot);
    const nms = nonMaxSuppression(mag, gx, gy, sub);
    let kept = 0, off = 0, blocks = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (nms.data[y * N + x] > 20) {
      kept++;
      if (Math.abs(Math.hypot(x - 31.5, y - 31.5) - R) > 1.3) off++;
      if (x + 1 < N && y + 1 < N && nms.data[y * N + x + 1] > 20 && nms.data[(y + 1) * N + x] > 20 && nms.data[(y + 1) * N + x + 1] > 20) blocks++;
    }
    check(off === 0, `NMS ring on the edge (sub=${sub}), ${off} off`);
    check(blocks === 0, `NMS ring has no 2×2 block (sub=${sub})`);
    check(kept > 2 * Math.PI * R * 0.9 && kept < 2 * Math.PI * R * 1.5, `NMS ring size (sub=${sub}) ${kept}`);
  }
}

// ---- direction quantisation and neighbours ----
check(quantiseDirection(1, 0) === 0 && quantiseDirection(1, -1) === 1 && quantiseDirection(0, -1) === 2 && quantiseDirection(-1, -1) === 3, "quantise 0/45/90/135");
check(quantiseDirection(-1, 0) === 0 && quantiseDirection(-1, 1) === 1 && quantiseDirection(1, 1) === 3, "quantise opposite directions");
{
  const nb = nmsNeighbours(1, -1, 5, 5);
  check(nb[0][0] === 6 && nb[0][1] === 4 && nb[1][0] === 4 && nb[1][1] === 6, "45° neighbours are up-right and down-left on screen");
  check(near(screenAngle(3, -4) * 180 / Math.PI, 53.13, 0.01), "screen angle of (3, −4) is 53°");
}

// ---- hysteresis = connected components containing a seed ----
{
  const r = rng(7);
  for (let t = 0; t < 30; t++) {
    const w = 12, h = 9, img = fromFunction(w, h, () => r.unif());
    const lo = 0.4, hi = 0.85;
    const res = hysteresis(img, lo, hi, 8);
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) mask[i] = img.data[i] >= lo ? 1 : 0;
    const L = label2d(mask, w, h, 8), seeded = new Set<number>();
    for (let i = 0; i < w * h; i++) if (img.data[i] >= hi) seeded.add(L.data[i]);
    let ok = true;
    for (let i = 0; i < w * h; i++) if ((res.edges[i] === 1) !== (mask[i] === 1 && seeded.has(L.data[i]))) ok = false;
    check(ok, `hysteresis = seeded components (trial ${t})`);
    // ring = BFS distance (chessboard steps inside the mask) from the nearest seed
    let ringOk = true;
    for (let i = 0; i < w * h; i++) if (res.ring[i] > 0) {
      const x = i % w, y = (i - x) / w;
      let hasPrev = false;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h && res.ring[yy * w + xx] === res.ring[i] - 1) hasPrev = true;
      }
      if (!hasPrev) ringOk = false;
    }
    check(ringOk, `hysteresis rings grow one step at a time (trial ${t})`);
  }
}

// ---- canny scene: the faint edge needs hysteresis ----
{
  const sc = makeScene("synthetic", 128, 8, 1);
  const c = canny(sc.image, 1.5, 4, 10);
  check(c.hyst.strong > 100 && c.hyst.weakKept > 0, `canny scene has seeds and kept weak pixels (${c.hyst.strong}, ${c.hyst.weakKept})`);
  const el = edgeList(c.hyst.edges, c.gx, c.gy, c.mag);
  check(el.length === c.hyst.strong + c.hyst.weakKept, "edge list length");
}

// ---- zero crossings of the LoG of a step lie at the step ----
{
  const img = fromFunction(40, 20, x => (x < 20 ? 0 : 100));
  const zc = zeroCrossings(logFilter(img, 2), 1);
  let ok = true, any = false;
  for (let y = 4; y < 16; y++) for (let x = 0; x < 40; x++) if (zc[y * 40 + x]) { any = true; if (x < 19 || x > 20) ok = false; }
  check(any && ok, "LoG zero crossing at the step");
}

// ---- Hessian of sampled quadratics = closed form ----
{
  const r = rng(11);
  for (let t = 0; t < 20; t++) {
    const a = 2 * r.unif() - 1, b = 2 * r.unif() - 1, c = 2 * r.unif() - 1;
    // f = a x² + b x y + c y²  → H = [[2a, b], [b, 2c]]
    const img = fromFunction(41, 41, (x, y) => a * (x - 20) ** 2 + b * (x - 20) * (y - 20) + c * (y - 20) ** 2);
    const H = hessian(img, 2);
    const i = 20 * 41 + 20;
    check(near(H.xx.data[i], 2 * a, 1e-3) && near(H.xy.data[i], b, 1e-3) && near(H.yy.data[i], 2 * c, 1e-3), `Hessian of quadratic (trial ${t})`);
    const m = (2 * a + 2 * c) / 2, d = Math.sqrt(((2 * a - 2 * c) / 2) ** 2 + b * b);
    const ev = [m - d, m + d].sort((p, q) => Math.abs(p) - Math.abs(q));
    check(near(H.l1.data[i], ev[0], 1e-3) && near(H.l2.data[i], ev[1], 1e-3), `Hessian eigenvalues closed form (trial ${t})`);
  }
  // by-hand examples of the unit
  const e1 = hessEig(-2, 0, 0); check(e1.l1 === 0 && e1.l2 === -2, "f = −x²: eigenvalues 0, −2");
  check(classifyHessianPixel(e1.l1, e1.l2, 0.1) === HessClass.BrightRidge, "f = −x²: bright ridge");
  check(classifyHessianPixel(-2, -2, 0.1) === HessClass.BrightBlob, "f = −(x²+y²): bright blob");
  check(classifyHessianPixel(2, -2, 0.1) === HessClass.Saddle, "x² − y²: saddle");
  const e4 = hessEig(-2, -2, -2); check(near(e4.l1, 0, 1e-12) && near(e4.l2, -4, 1e-12), "−(x+y)²: eigenvalues 0, −4");
  check(near(Math.abs(Math.cos(e4.angle1 - (-Math.PI / 4))), 1, 1e-9), "−(x+y)²: l1 eigenvector along (1, −1)");
}

// ---- Frangi: line kept, blob suppressed; σ²λ2 of a Gaussian line of width s peaks at σ = √2·s ----
{
  check(frangiValue(0, -10, 0.5, 5) > 0.8 && frangiValue(-10, -10, 0.5, 5) < 0.2 && frangiValue(0, 10, 0.5, 5) === 0, "frangi line vs blob vs dark");
  const s = 3, img = fromFunction(81, 21, x => 100 * Math.exp(-((x - 40) ** 2) / (2 * s * s)));
  const sig = geomScales(1, 8, 36);
  let best = 0, bv = 0;
  sig.forEach(sg => { const H = hessian(img, sg, true); const v = -H.l2.data[10 * 81 + 40]; if (v > bv) { bv = v; best = sg; } });
  check(near(best, Math.SQRT2 * s, 0.3), `σ²λ2 of a Gaussian line of width ${s} peaks at σ = √2·s: ${best.toFixed(2)}`);
  const sc = makeHessScene("scene", 128, 0, 1);
  const F = frangi(sc.image, [1.5, 2, 3, 4], 0.5, 15, true);
  check(F.response.data.length === 128 * 128, "frangi runs on the hessian scene");
}

// ---- structure tensor: rotated stripes within 1°, corners of a square ----
{
  for (const deg of [0, 20, 45, 70, 110, 160]) {
    const th = deg * Math.PI / 180, f = 0.15;
    // stripes whose gradient points at screen angle th: intensity varies along (cos th, −sin th) in image coords
    const img = fromFunction(64, 64, (x, y) => 100 + 80 * Math.sin(2 * Math.PI * f * (Math.cos(th) * x - Math.sin(th) * y)));
    const J = structureTensor(img, 1, 4);
    const i = 32 * 64 + 32;
    let a = -J.angle.data[i] * 180 / Math.PI;     // to screen, CCW
    let d = ((a - deg) % 180 + 180) % 180; if (d > 90) d = 180 - d;
    check(d < 1, `stripes at ${deg}°: orientation error ${d.toFixed(3)}°`);
    check(J.coherence.data[i] > 0.95, `stripes at ${deg}°: coherence ${J.coherence.data[i].toFixed(3)}`);
  }
  const sq = fromFunction(64, 64, (x, y) => (x >= 20 && x < 44 && y >= 20 && y < 44 ? 200 : 50));
  const J = structureTensor(sq, 1, 2), H = harris(J, 0.05);
  const mx = localMaxima2D(H, 0.1 * Math.max(...H.data), 3, 2).slice(0, 4);
  const corners = [[19.5, 19.5], [43.5, 19.5], [19.5, 43.5], [43.5, 43.5]];
  check(mx.length === 4 && corners.every(([cx, cy]) => mx.some(m => Math.hypot(m.x - cx, m.y - cy) < 2.5)), `Harris maxima at the corners: ${JSON.stringify(mx.map(m => [m.x, m.y]))}`);
  const me = minEig(J);
  check(me.data[32 * 64 + 20] < 0.05 * me.data[20 * 64 + 20], "min eigenvalue small on an edge, large at a corner");
  // ρ = 0: rank 1 everywhere
  const J0 = structureTensor(sq, 1, 0);
  let maxMu2 = 0; for (const v of J0.mu2.data) maxMu2 = Math.max(maxMu2, v);
  check(maxMu2 < 1e-6 * Math.max(...J0.mu1.data), "ρ = 0: μ2 = 0 everywhere");
  for (const id of ["stripes", "rotating", "fingerprint", "boundary", "checker", "noise"]) {
    const t = makeTexture(id, 128, 0, 1, 30, 0.12);
    check(t.width === 128 && t.data.every(Number.isFinite), `texture ${id}`);
  }
}

// ---- scale space: normalised LoG of a disc peaks near r/√2; DoG ≈ (k−1)σ²LoG ----
{
  for (const R of [4, 6, 10]) {
    const N = 96, disc = fromFunction(N, N, (x, y) => (Math.hypot(x - 47.5, y - 47.5) < R ? 100 : 0));
    const sig = geomScales(1, 16, 41), step = Math.pow(16, 1 / 40);
    const st = normLoGStack(disc, sig, true);
    let best = 0, bv = -Infinity;
    st.forEach((g, k) => { const v = Math.max(g.data[47 * N + 47], g.data[48 * N + 48]); if (v > bv) { bv = v; best = k; } });
    const pred = R / Math.SQRT2;
    check(sig[best] / pred < step * 1.06 && pred / sig[best] < step * 1.06, `disc r=${R}: normalised LoG peaks at σ=${sig[best].toFixed(2)} (r/√2=${pred.toFixed(2)})`);
  }
  const N = 96, blob = fromFunction(N, N, (x, y) => 100 * Math.exp(-((x - 48) ** 2 + (y - 48) ** 2) / (2 * 25)));
  for (const k of [1.1, 1.2, 1.6]) {
    const s = 3, D = dogFilter(blob, s, k), L = logFilter(blob, s, false);
    const ratio = D.data[48 * N + 48] / ((k - 1) * s * s * L.data[48 * N + 48]);
    check(Math.abs(ratio - 1) < (k < 1.5 ? 0.15 : 0.4), `DoG ≈ (k−1)σ²LoG at k=${k}: ratio ${ratio.toFixed(3)}`);
  }
}

// ---- extrema3D vs brute force ----
{
  const r = rng(5);
  for (let t = 0; t < 10; t++) {
    const w = 9, h = 8, S = 4, stack: Grey[] = [];
    for (let s = 0; s < S; s++) stack.push(fromFunction(w, h, () => Math.round(r.unif() * 6)));
    const ex = extrema3D(stack, 2, 0);
    // brute force: strictly greater than all neighbours (ties excluded) ⊆ result; result ⊆ ≥ all neighbours
    let ok = true;
    for (let s = 0; s < S; s++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = stack[s].data[y * w + x];
      if (!(v > 2)) continue;
      let strict = true, weak = true;
      for (let ds = -1; ds <= 1; ds++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!ds && !dx && !dy) continue;
        const ss = s + ds, xx = x + dx, yy = y + dy;
        if (ss < 0 || ss >= S || xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const u = stack[ss].data[yy * w + xx];
        if (u >= v) strict = false;
        if (u > v) weak = false;
      }
      const found = ex.some(e => e.x === x && e.y === y && e.s === s);
      if (strict && !found) ok = false;
      if (found && !weak) ok = false;
    }
    check(ok, `extrema3D vs brute force (trial ${t})`);
  }
}

// ---- blobscale scene: detections near the true discs ----
{
  const sc = makeBlobImage("synthetic", 256, 0, 1);
  const sig = geomScales(1.5, 16, 16);
  const st = normLoGStack(sc.image, sig, true);
  const ex = extrema3D(st, 15, 2);
  let hits = 0;
  for (const b of sc.blobs) if (ex.some(e => Math.hypot(e.x - b.x, e.y - b.y) < Math.max(2, 0.5 * b.r))) hits++;
  check(hits >= 0.8 * sc.blobs.length, `blobscale: ${hits}/${sc.blobs.length} blobs found (no noise)`);
}

// ---- blobscale's fast stack (coarse scales on a shrunk image) ≈ the direct normalised LoG ----
{
  const sc = makeBlobImage("synthetic", 256, 6, 1), sig = geomScales(1, 16, 16);
  const fast = responseStack(sc.image, sig, -1), direct = normLoGStack(sc.image, sig, true);
  sig.forEach((s, k) => {
    let e = 0, m = 0;
    for (let y = 40; y < 216; y++) for (let x = 40; x < 216; x++) { const i = y * 256 + x; e = Math.max(e, Math.abs(fast[k].data[i] - direct[k].data[i])); m = Math.max(m, Math.abs(direct[k].data[i])); }
    check(e < 0.06 * m, `fast stack at σ=${s.toFixed(2)}: max error ${e.toFixed(2)} of ${m.toFixed(1)}`);
  });
  const ex = extrema3D(fast, 26, 3, true);
  let hits = 0;
  for (const b of sc.blobs) if (ex.some(e => Math.hypot(e.x - b.x, e.y - b.y) <= Math.max(3, 0.5 * b.r))) hits++;
  check(hits >= 13, `blobscale default (threshold 26, noise 6): ${hits} of ${sc.blobs.length} blobs found`);
}

// ---- Laplacian pyramid reconstructs exactly ----
{
  const r = rng(9), img = fromFunction(37, 29, () => r.unif() * 255);
  const L = laplacianPyramid(img, 4), back = collapsePyramid(L);
  let err = 0; for (let i = 0; i < img.data.length; i++) err = Math.max(err, Math.abs(back.data[i] - img.data[i]));
  check(err < 1e-3, `Laplacian pyramid reconstruction error ${err}`);
}

// ---- scale1d signals: the box kernel creates an extremum, the Gaussian never does ----
{
  for (const s of S1D) check(makeSignal(s.id, 0).length === 256, `scale1d signal ${s.id}`);
  const f = makeSignal("boxdemo", 0);
  const sig = geomScales(0.5, 32, 64);
  let prevG = countExtrema(f, 1e-6), gaussOk = true, boxCreates = false, prevB = prevG;
  for (const s of sig) {
    const g = countExtrema(correlate1d(f, gaussianKernel(s, 0)), 1e-6);
    if (g > prevG) gaussOk = false;
    prevG = g;
    const b = countExtrema(correlate1d(f, boxKernel1d(s)), 1e-6);
    if (b > prevB) boxCreates = true;
    prevB = b;
  }
  check(gaussOk, "Gaussian blur never adds an extremum to the box-demo signal");
  check(boxCreates, "box blur adds an extremum to the box-demo signal");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
