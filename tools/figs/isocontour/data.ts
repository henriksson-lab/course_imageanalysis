// Dumps images and marching-squares contours for tools/figs/isocontour/figs.py. Run from the repo root:
//   node_modules/.bin/esbuild --bundle --platform=node tools/figs/isocontour/data.ts --outfile=<tmp>/iso-data.js && node <tmp>/iso-data.js > <tmp>/iso-data.json
// The "noise" and "blur" images are the presets of demos/src/isocontour/main.ts (same seed, same
// padding, asymptotic decider), so the counts match the demo.
import { synthImage, N } from "../../../demos/src/isocontour/image";
import { addGaussianNoise } from "../../../demos/src/lib/synth";
import { gaussianBlur } from "../../../demos/src/lib/imageops";
import { marchingSquares, padGrid, contourLength, lineArea, Contour, EdgeInterp } from "../../../demos/src/lib/marching";

function contours(values: ArrayLike<number>, w: number, h: number, c: number, pad: number, interp: EdgeInterp = "linear") {
  const P = padGrid(values, w, h, pad);
  const C: Contour = marchingSquares(P.values, P.w, P.h, c, { saddle: "decider", interp });
  const lines = C.lines.map(L => ({ closed: L.closed, area: lineArea(C, L),
    xy: L.ids.map(id => [C.points[2 * id] - 1, C.points[2 * id + 1] - 1]) }));
  return { lines, count: C.lines.length, length: contourLength(C) };
}

const synth = synthImage();
const noisy = addGaussianNoise(synth, 25, 11);
const blurred = gaussianBlur(noisy, 1.5);
const out: Record<string, unknown> = {};
out.N = N;
out.noise = { img: Array.from(noisy.data), ...contours(noisy.data, N, N, 120, -1) };
out.blur = { img: Array.from(blurred.data), ...contours(blurred.data, N, N, 120, -1) };

// a small bright blob on a background of 40, for "a boundary is where the image crosses a value"
const n = 9, small: number[] = [];
for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
  small.push(Math.round(40 + 170 * Math.exp(-(((x - 4.2) / 2.3) ** 2 + ((y - 3.8) / 1.9) ** 2) / 2)));
out.small = { n, img: small, linear: contours(small, n, n, 120, 0), midpoint: contours(small, n, n, 120, 0, "midpoint") };

// a bright ring (outer boundary and a hole), for orientation
const m = 10, ring: number[] = [];
for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
  const r = Math.hypot(x - 4.5, y - 4.5);
  ring.push(Math.round(20 + 180 * Math.exp(-((r - 2.6) ** 2) / (2 * 1.0 ** 2))));
}
out.ring = { n: m, img: ring, ...contours(ring, m, m, 100, 0) };
console.log(JSON.stringify(out));
