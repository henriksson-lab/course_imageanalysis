// Synthetic "cell" image for the image mode, and the inverse-mapping renderer.
import { rng } from "../lib/ui";
import { M2, apply, inverse } from "./geom";

/** Half-width of the source image in world units: it covers [-H, H] x [-H, H]. */
export const H = 1.5;

/** A cell-like picture: frame, cell body with a protrusion, nucleus, mitochondria, and an "F" mark. */
export function makeCellImage(N = 200): ImageData {
  const cv = document.createElement("canvas");
  cv.width = cv.height = N;
  const g = cv.getContext("2d")!;
  g.fillStyle = "#e8eef6";
  g.fillRect(0, 0, N, N);
  g.strokeStyle = "#7d8da3";
  g.lineWidth = N * 0.015;
  g.strokeRect(g.lineWidth / 2, g.lineWidth / 2, N - g.lineWidth, N - g.lineWidth);

  g.save();
  g.translate(N * 0.47, N * 0.53);
  g.rotate(-0.3);
  // cell body + a protrusion to the right (one closed outline)
  g.beginPath();
  g.ellipse(0, 0, N * 0.29, N * 0.21, 0, 0.35, 2 * Math.PI - 0.35);
  g.bezierCurveTo(N * 0.36, -N * 0.05, N * 0.44, -N * 0.04, N * 0.46, 0);
  g.bezierCurveTo(N * 0.44, N * 0.04, N * 0.36, N * 0.05, N * 0.27, N * 0.07);
  g.closePath();
  g.fillStyle = "#a6dcae";
  g.fill();
  g.lineWidth = N * 0.016;
  g.strokeStyle = "#357a46";
  g.stroke();
  // mitochondria (fixed seed)
  const r = rng(7);
  g.fillStyle = "#e07a3c";
  let placed = 0;
  while (placed < 9) {
    const x = (r.unif() - 0.5) * N * 0.5, y = (r.unif() - 0.5) * N * 0.34;
    if ((x / (N * 0.25)) ** 2 + (y / (N * 0.17)) ** 2 > 1) continue;      // inside the body
    if (((x + N * 0.09) / (N * 0.14)) ** 2 + ((y + N * 0.03) / (N * 0.11)) ** 2 < 1) continue; // not on nucleus
    g.beginPath();
    g.ellipse(x, y, N * 0.028, N * 0.013, r.unif() * Math.PI, 0, 2 * Math.PI);
    g.fill();
    placed++;
  }
  // nucleus, off-centre, with a nucleolus
  g.beginPath();
  g.ellipse(-N * 0.09, -N * 0.03, N * 0.11, N * 0.085, 0.2, 0, 2 * Math.PI);
  g.fillStyle = "#8576d0";
  g.fill();
  g.lineWidth = N * 0.012;
  g.strokeStyle = "#45379a";
  g.stroke();
  g.beginPath();
  g.arc(-N * 0.06, -N * 0.05, N * 0.03, 0, 2 * Math.PI);
  g.fillStyle = "#3c2f86";
  g.fill();
  g.restore();

  // an "F" in the top-left corner: shows mirror images at a glance
  g.fillStyle = "#26303d";
  const u = N / 100;
  g.fillRect(8 * u, 8 * u, 4 * u, 20 * u);
  g.fillRect(8 * u, 8 * u, 12 * u, 4 * u);
  g.fillRect(8 * u, 16 * u, 9 * u, 4 * u);
  return g.getImageData(0, 0, N, N);
}

/**
 * Draw A applied to `src` into `out` (square, S x S pixels showing world [-R, R]^2),
 * around the image centre. Uses inverse mapping: for every output pixel, ask where it
 * came from (A^-1) and read the source there (bilinear). If A collapses the plane, the
 * source pixels are pushed forward instead (they all land on a line).
 */
export function renderImage(A: M2, src: ImageData, out: ImageData, R: number): void {
  const S = out.width, N = src.width, s = src.data, o = out.data;
  o.fill(0);
  const k = S / (2 * R);
  const inv = inverse(A);
  if (!inv) {
    for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) {
      const wx = -H + (u + 0.5) * 2 * H / N, wy = H - (v + 0.5) * 2 * H / N;
      const [X, Y] = apply(A, wx, wy);
      const i = Math.floor(S / 2 + X * k), j = Math.floor(S / 2 - Y * k);
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= S || jj >= S) continue;
        const p = (jj * S + ii) * 4, q = (v * N + u) * 4;
        o[p] = s[q]; o[p + 1] = s[q + 1]; o[p + 2] = s[q + 2]; o[p + 3] = 255;
      }
    }
    return;
  }
  const [ia, ib, ic, id] = inv;
  const f = N / (2 * H);
  for (let j = 0; j < S; j++) {
    const Y = (S / 2 - (j + 0.5)) / k;
    for (let i = 0; i < S; i++) {
      const X = ((i + 0.5) - S / 2) / k;
      const sx = ia * X + ib * Y, sy = ic * X + id * Y;
      if (sx < -H || sx > H || sy < -H || sy > H) continue;
      const u = Math.min(Math.max((sx + H) * f - 0.5, 0), N - 1.001);
      const v = Math.min(Math.max((H - sy) * f - 0.5, 0), N - 1.001);
      const u0 = Math.floor(u), v0 = Math.floor(v), fu = u - u0, fv = v - v0;
      const q00 = (v0 * N + u0) * 4, q10 = q00 + 4, q01 = q00 + N * 4, q11 = q01 + 4;
      const p = (j * S + i) * 4;
      for (let ch = 0; ch < 3; ch++) {
        o[p + ch] = (s[q00 + ch] * (1 - fu) + s[q10 + ch] * fu) * (1 - fv)
          + (s[q01 + ch] * (1 - fu) + s[q11 + ch] * fu) * fv;
      }
      o[p + 3] = 255;
    }
  }
}
