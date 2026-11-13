// Slide figures for pathtrace/ (second pass: pictures on the slides). fig-mcnoise.svg comes from fig_mcnoise.py.
// The renders use the CPU path tracer of lib/raytrace.ts (the pathtracer demo's fallback) and the demo's denoiser.
// Run from the repo root (the 1024-spp reference takes a few minutes):
//   node_modules/.bin/esbuild tools/figs/pathtrace/figs.ts --bundle --platform=node --log-level=warning | node
import * as RT from "../../../demos/src/lib/raytrace";
import { linearToSrgb8 } from "../../../demos/src/lib/colour";
import { toDisplayPlanes, denoise } from "../../../demos/src/pathtracer/denoise";
import { rmse, psnr } from "../../../demos/src/lib/metrics";
import { pngRGB, imgEl, save, svg, txt, line, circ, arrowDefs, C } from "../raycast/util";

const only = (globalThis as any).process?.argv?.[2] as string | undefined;
const want = (k: string) => !only || only === k;

// ---------------------------------------------------------------- BRDF lobes
if (want("brdf")) {
  const PW = 230, PH = 140, R = 100, top = 46;
  let b = arrowDefs;
  const panel = (k: number, title: string, note: string, lobe: (th: number) => number, spike: boolean) => {
    const cx = PW / 2, gy = top + k * PH + 112;
    let g = line(12, gy, PW - 12, gy, `stroke="#444" stroke-width="3"`);
    const ti = 45 * Math.PI / 180; // incoming from the upper left at 45°
    if (spike) g += line(cx, gy, cx + 95 * Math.sin(ti), gy - 95 * Math.cos(ti), `stroke="${C.blue}" stroke-width="5" marker-end="url(#ah)"`);
    else {
      const pts: string[] = [];
      for (let a = -90; a <= 90; a += 1) { const th = a * Math.PI / 180, r = R * lobe(th); pts.push(`${(cx + r * Math.sin(th)).toFixed(1)},${(gy - r * Math.cos(th)).toFixed(1)}`); }
      g += `<polygon points="${cx},${gy} ${pts.join(" ")}" fill="${C.sky}" fill-opacity="0.35" stroke="${C.blue}" stroke-width="2.5"/>`;
    }
    g += line(cx - 95 * Math.sin(ti), gy - 95 * Math.cos(ti), cx - 4, gy - 4, `stroke="${C.orange}" stroke-width="3.5" marker-end="url(#ah)"`);
    g += circ(cx, gy, 5, `fill="#000"`);
    g += txt(PW + 10, gy - 50, title, `font-weight="600"`) + txt(PW + 10, gy - 22, note, `font-size="21"`);
    return g;
  };
  b += panel(0, "matte", "same in all directions", () => 0.8, false);
  b += panel(1, "glossy", "a lobe near the mirror", th => 0.95 * Math.max(0, Math.cos(th - Math.PI / 4)) ** 12, false);
  b += panel(2, "mirror", "one direction only", () => 0, true);
  b += txt(10, 24, "incoming ray (orange);", `font-size="21"`) + txt(10, 50, "where the BRDF sends it (blue)", `font-size="21"`);
  save("pathtrace/fig-brdf.svg", svg(PW + 270, top + 3 * PH + 10, b, 22));
}

// ---------------------------------------------------------------- the rendering equation: all incoming directions, recursively
if (want("rendereq")) {
  const W = 560, H = 380, x = [190, 300], R = 110;
  let b = arrowDefs;
  b += `<rect x="0" y="${x[1]}" width="${W}" height="${H - x[1]}" fill="#e8e8e8"/>` + line(0, x[1], W, x[1], `stroke="#444" stroke-width="3"`);
  // hemisphere of incoming directions
  b += `<path d="M${x[0] - R},${x[1]} A${R},${R} 0 0 1 ${x[0] + R},${x[1]}" fill="${C.sky}" fill-opacity="0.15" stroke="${C.blue}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  for (const a of [-70, -45, -20, 5, 30, 55, 78]) {
    const th = a * Math.PI / 180, ex = x[0] + R * Math.sin(th), ey = x[1] - R * Math.cos(th);
    b += line(ex, ey, x[0] + 8 * Math.sin(th), x[1] - 8 * Math.cos(th), `stroke="${C.blue}" stroke-width="2" marker-end="url(#ah)"`);
  }
  // to the eye
  b += line(x[0], x[1], x[0] - 170, x[1] - 200, `stroke="${C.orange}" stroke-width="4" marker-end="url(#ah)"`) + txt(x[0] - 185, x[1] - 210, `to the eye: L<tspan baseline-shift="sub" font-size="70%">o</tspan>`, `fill="#8a5a00"`);
  // a lamp
  // one incoming direction traced back to a wall point y, with its own hemisphere
  const y = [500, 140];
  b += `<rect x="${y[0]}" y="40" width="14" height="${x[1] - 40}" fill="#f2c4b6" stroke="${C.red}" stroke-width="2"/>`;
  const dx = y[0] - x[0], dy = y[1] - x[1], L = Math.hypot(dx, dy);
  b += line(y[0], y[1], x[0] + 10 * dx / L, x[1] + 10 * dy / L, `stroke="${C.red}" stroke-width="3.5" marker-end="url(#ah)"`);
  b += `<path d="M${y[0]},${y[1] - 70} A70,70 0 0 0 ${y[0]},${y[1] + 70}" fill="none" stroke="${C.red}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  for (const a of [-50, 0, 50]) { const th = a * Math.PI / 180; b += line(y[0] - 70 * Math.cos(th), y[1] + 70 * Math.sin(th), y[0] - 8 * Math.cos(th), y[1] + 8 * Math.sin(th), `stroke="${C.red}" stroke-width="1.6" marker-end="url(#ah)"`); }
  b += circ(y[0], y[1], 6, `fill="#000"`) + txt(y[0] - 26, y[1] - 80, "y", `font-weight="700"`);
  b += txt(300, 255, `L<tspan baseline-shift="sub" font-size="70%">i</tspan>(x) = L<tspan baseline-shift="sub" font-size="70%">o</tspan>(y)`, `fill="${C.red}" font-size="21"`);
  b += circ(x[0], x[1], 7, `fill="#000"`) + txt(x[0] - 6, x[1] + 34, "x", `font-weight="700"`);
  b += txt(10, H - 12, "x gathers from every direction; y does the same", `font-size="21"`);
  save("pathtrace/fig-rendereq.svg", svg(W, H, b, 22));
}

// ---------------------------------------------------------------- path tracing images (CPU)
const W = 200, H = 150;
const render = (o: RT.BoxSceneOptions, opt: RT.PathOptions, spp: number) => {
  const sc = RT.boxScene(o), P = new RT.Progressive(sc, sc.camera, W, H, opt, 1);
  while (P.spp < spp) P.addSample();
  return { mean: P.mean(), sc };
};
const toRGB8 = (m: Float32Array) => Uint8Array.from(m, v => linearToSrgb8(v));
const planesToRGB8 = (p: Float64Array[]) => { const out = new Uint8Array(3 * W * H); for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = Math.round(255 * Math.max(0, Math.min(1, p[c][i]))); return out; };
const panelRow = (ims: Uint8Array[], labs: [string, string][], k = 1.4) => {
  const w = Math.round(W * k), h = Math.round(H * k);
  let b = "";
  ims.forEach((im, i) => { const x = i * (w + 16); b += imgEl(x, 0, w, h, pngRGB(im, W, H), false) + txt(x, h + 26, labs[i][0], `font-weight="600"`) + txt(x, h + 52, labs[i][1]); });
  return svg(ims.length * (w + 16) - 16, h + 62, b, 21);
};
/** The same, stacked vertically with the labels to the right of each image. */
const panelCol = (ims: Uint8Array[], labs: [string, string][], k = 1.1) => {
  const w = Math.round(W * k), h = Math.round(H * k);
  let b = "";
  ims.forEach((im, i) => { const y = i * (h + 12); b += imgEl(0, y, w, h, pngRGB(im, W, H), false) + txt(w + 14, y + h / 2 - 4, labs[i][0], `font-weight="600"`) + txt(w + 14, y + h / 2 + 24, labs[i][1]); });
  return svg(w + 230, ims.length * (h + 12) - 12, b, 21);
};

if (want("fireflies")) {
  const t0 = Date.now();
  const a = render({ lightSize: 0.1 }, { maxBounces: 5, russianRoulette: true, sampling: "cosine" }, 64);
  const b = render({ lightSize: 0.1 }, { maxBounces: 5, russianRoulette: true, sampling: "light" }, 64);
  console.log("fireflies renders", ((Date.now() - t0) / 1000).toFixed(1), "s");
  save("pathtrace/fig-fireflies.svg", panelRow([toRGB8(a.mean), toRGB8(b.mean)], [["cosine sampling", "fireflies"], ["sample the lamp", "same 64 samples"]]));
}

if (want("denoise")) {
  const t0 = Date.now(), opt: RT.PathOptions = { maxBounces: 5, russianRoulette: true, sampling: "light" };
  const noisy = render({}, opt, 16), ref = render({}, opt, 1024);
  console.log("denoise renders", ((Date.now() - t0) / 1000).toFixed(1), "s");
  const n = W * H, aux = RT.auxBuffers(noisy.sc, noisy.sc.camera, W, H);
  const pN = toDisplayPlanes(noisy.mean, n), pR = toDisplayPlanes(ref.mean, n), pD = denoise("joint", pN, W, H, 3, aux);
  const flat = (a: Float64Array[]) => { const f = new Float64Array(3 * n); for (let c = 0; c < 3; c++) f.set(a[c], c * n); return f; };
  const R = flat(pR), r1 = rmse(flat(pN), R), r2 = rmse(flat(pD), R), p1 = psnr(flat(pN), R, 1), p2 = psnr(flat(pD), R, 1);
  console.log(`noisy RMSE ${r1.toFixed(4)} PSNR ${p1.toFixed(1)} | denoised RMSE ${r2.toFixed(4)} PSNR ${p2.toFixed(1)}`);
  save("pathtrace/fig-denoise.svg", panelCol([planesToRGB8(pN), planesToRGB8(pD), planesToRGB8(pR)],
    [["16 samples", `PSNR ${p1.toFixed(1)} dB`], ["16, denoised", `PSNR ${p2.toFixed(1)} dB`], ["1024 samples", "the reference"]]));
}
