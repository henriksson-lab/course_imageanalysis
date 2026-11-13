// Slide figures for autoencoder/ (added for design principle 3b), made with the demo code.
// Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/autoencoder/figs.ts --bundle --platform=node --outfile=/tmp/ae-figs.cjs
//   node /tmp/ae-figs.cjs .
declare const process: { argv: string[] };
import { fromB64, digitVae } from "../../../demos/src/vaeimages/imagevae";
import { DIGIT_EXAMPLES } from "../../../demos/src/vaeimages/digit_examples";
import { makeData, pca, makeAE, aeTrain, aeCode, aeDecode, aeError } from "../../../demos/src/linae/data";
import { rng } from "../../../demos/src/lib/ui";
import { INK, BLUE, RED, GREEN, ORANGE, GREY, writer, svgOpen, sc, path, f, pngURI, pngAny, img } from "../vae/lib";

const write = writer(process.argv[2] ?? ".");

// ------------------------------------------------ fig-image-point.svg: an image is a vector
{
  const seven = fromB64(DIGIT_EXAMPLES[14].img);
  const r = rng(5), noise = Float32Array.from({ length: 784 }, () => r.unif());
  const W = 900, H = 305;
  let s = svgOpen(W, H);
  const a = pngURI([seven], 1, 0, 6), b = pngURI([noise], 1, 0, 6);
  s += img(40, 20, 168, 168, a.uri) + img(470, 20, 168, 168, b.uri);
  s += `<text x="224" y="70" font-weight="bold">a handwritten 7</text><text x="224" y="97">28 × 28 pixels:</text><text x="224" y="124">one point in ℝ⁷⁸⁴,</text><text x="224" y="151">inside the thin</text><text x="224" y="178">region of digits</text>\n`;
  s += `<text x="654" y="70" font-weight="bold">a random point</text><text x="654" y="97">of ℝ⁷⁸⁴: every</text><text x="654" y="124">pixel uniform</text><text x="654" y="151">in [0, 1]: noise,</text><text x="654" y="178">like almost all points</text>\n`;
  // the 7 as one long vector, row after row (784 values, 1 px each, 26 px high)
  const strip = pngAny(seven, 784, 1, 1);
  s += `<image x="58" y="232" width="784" height="30" href="${strip}" preserveAspectRatio="none" style="image-rendering:pixelated"/><rect x="58" y="232" width="784" height="30" fill="none" stroke="${INK}" stroke-width="1.5"/>\n`;
  s += `<text x="450" y="292" text-anchor="middle">the same 7 as one vector: 784 grey values, row after row</text>`;
  s += `<text x="58" y="222" font-size="16" fill="${GREY}">pixel 1</text><text x="842" y="222" font-size="16" fill="${GREY}" text-anchor="end">pixel 784</text>\n</svg>\n`;
  write("autoencoder/fig-image-point.svg", s);
}

// ------------------------------------------------ fig-pca-byhand.svg: encode / decode / error
{
  const W = 620, H = 400, sx = sc(-0.6, 4.6, 50, 610), sy = sc(-0.6, 2.6, 360, 18);
  const u = [0.8, 0.6], x = [3, 1], sc_ = 0.8 * 3 + 0.6 * 1, xh = [sc_ * u[0], sc_ * u[1]];
  let s = svgOpen(W, H, 23);
  for (let v = 0; v <= 4; v++) s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${sy(-0.6)}" y2="${sy(2.6)}" stroke="#e3e3e3"/><text x="${sx(v)}" y="${H - 12}" text-anchor="middle" font-size="21">${v}</text>`;
  for (let v = 0; v <= 2; v++) s += `<line y1="${sy(v)}" y2="${sy(v)}" x1="${sx(-0.6)}" x2="${sx(4.6)}" stroke="#e3e3e3"/><text x="38" y="${sy(v) + 6}" text-anchor="end" font-size="21">${v}</text>`;
  s += `\n<text x="${W - 10}" y="${sy(0) - 10}" text-anchor="end">pixel 1</text><text x="${sx(0) + 8}" y="34">pixel 2</text>\n`;
  // PC1 line through the (zero) mean
  s += `<line x1="${sx(-0.5 * 0.8)}" y1="${sy(-0.5 * 0.6)}" x2="${sx(4.2 * 0.8)}" y2="${sy(4.2 * 0.6)}" stroke="${BLUE}" stroke-width="3"/>\n`;
  s += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(u[0])}" y2="${sy(u[1])}" stroke="${BLUE}" stroke-width="5" marker-end="url(#arB)"/>`;
  s += `<text x="${sx(0.62)}" y="${sy(0.62) - 18}" fill="${BLUE}" font-weight="bold" text-anchor="end">u = (0.8, 0.6)</text>\n`;
  // score bracket along the line, offset below it
  const off = [0.6 * 0.18, -0.8 * 0.18];
  s += `<line x1="${sx(off[0])}" y1="${sy(off[1])}" x2="${sx(xh[0] + off[0])}" y2="${sy(xh[1] + off[1])}" stroke="${GREEN}" stroke-width="2.5" marker-start="url(#ar)" marker-end="url(#ar)"/>`;
  s += `<text x="${sx(1.2 * 0.8 + 0.42)}" y="${sy(1.2 * 0.6 - 0.32)}" fill="${GREEN}" font-weight="bold">score s = 3.0</text>\n`;
  // error segment with right angle mark
  const e = [x[0] - xh[0], x[1] - xh[1]], el = Math.hypot(e[0], e[1]), en = [e[0] / el, e[1] / el], k = 0.12;
  s += `<path d="M${f(sx(xh[0] + en[0] * k))},${f(sy(xh[1] + en[1] * k))}L${f(sx(xh[0] + en[0] * k + u[0] * k))},${f(sy(xh[1] + en[1] * k + u[1] * k))}L${f(sx(xh[0] + u[0] * k))},${f(sy(xh[1] + u[1] * k))}" fill="none" stroke="${GREY}" stroke-width="1.5"/>`;
  s += `<line x1="${sx(x[0])}" y1="${sy(x[1])}" x2="${sx(xh[0])}" y2="${sy(xh[1])}" stroke="${RED}" stroke-width="3" stroke-dasharray="7 4"/>`;
  s += `<text x="${sx(2.75) + 12}" y="${sy(1.4)}" fill="${RED}" font-weight="bold">error (0.6, −0.8)</text>\n`;
  s += `<circle cx="${sx(x[0])}" cy="${sy(x[1])}" r="7" fill="${INK}"/><text x="${sx(x[0]) + 12}" y="${sy(x[1]) + 24}" font-weight="bold">x = (3, 1)</text>`;
  s += `<circle cx="${sx(xh[0])}" cy="${sy(xh[1])}" r="7" fill="#fff" stroke="${INK}" stroke-width="3"/><text x="${sx(xh[0]) - 12}" y="${sy(xh[1]) - 12}" text-anchor="end" font-weight="bold">x̂ = (2.4, 1.8)</text>\n`;
  s += `<circle cx="${sx(0)}" cy="${sy(0)}" r="5" fill="${GREY}"/><text x="${sx(0) - 10}" y="${sy(0) + 22}" text-anchor="end" fill="${GREY}" font-size="21">mean</text>\n</svg>\n`;
  write("autoencoder/fig-pca-byhand.svg", s);
}

// ------------------------------------------------ fig-ae-gap.svg: what the decoder does outside the training codes
{
  const P = makeData("spot"), ae = makeAE();
  aeTrain(ae.net, ae.opt, P, 1500);
  const codes = P.map(p => aeCode(ae.net, p)), lo = Math.min(...codes), hi = Math.max(...codes), d = hi - lo;
  console.log(`spot AE: error ${aeError(ae.net, P).toFixed(4)}, codes of the training images ${lo.toFixed(2)} .. ${hi.toFixed(2)}`);
  const S = 470, sx = sc(-0.45, 1.3, 50, S - 10), sy = sc(-0.45, 1.3, S - 45, 10);
  let s = svgOpen(S, S);
  for (const v of [0, 0.5, 1]) s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${sy(-0.45)}" y2="${sy(1.3)}" stroke="#e3e3e3"/><line y1="${sy(v)}" y2="${sy(v)}" x1="${sx(-0.45)}" x2="${sx(1.3)}" stroke="#e3e3e3"/><text x="${sx(v)}" y="${S - 20}" text-anchor="middle">${v}</text><text x="40" y="${sy(v) + 6}" text-anchor="end">${v}</text>\n`;
  s += `<text x="${S - 12}" y="${S - 52}" text-anchor="end">pixel 1</text><text x="56" y="30">pixel 2</text>\n`;
  const seg = (a: number, b: number) => path(Array.from({ length: 121 }, (_, i) => { const q = aeDecode(ae.net, a + (b - a) * i / 120); return [sx(q[0]), sy(q[1])] as [number, number]; }));
  P.forEach(p => { s += `<circle cx="${f(sx(p[0]))}" cy="${f(sy(p[1]))}" r="4" fill="#555" fill-opacity="0.6"/>`; });
  s += `\n<path d="${seg(lo, hi)}" fill="none" stroke="${GREEN}" stroke-width="4"/>\n`;
  s += `<path d="${seg(lo - 0.5 * d, lo)}" fill="none" stroke="${ORANGE}" stroke-width="4" stroke-dasharray="9 6"/><path d="${seg(hi, hi + 0.5 * d)}" fill="none" stroke="${ORANGE}" stroke-width="4" stroke-dasharray="9 6"/>\n`;
  for (const z of [lo - 0.5 * d, hi + 0.5 * d]) { const q = aeDecode(ae.net, z); s += `<circle cx="${f(sx(q[0]))}" cy="${f(sy(q[1]))}" r="7" fill="${ORANGE}"/>`; console.log(`code ${z.toFixed(2)} decodes to (${q[0].toFixed(2)}, ${q[1].toFixed(2)})`); }
  s += `\n<text x="${S - 14}" y="34" text-anchor="end" fill="${GREEN}" font-weight="bold">codes of training images</text>`;
  s += `<text x="${S - 14}" y="60" text-anchor="end" fill="${GREEN}">z from ${lo.toFixed(2)} to ${hi.toFixed(2)}</text>`;
  s += `<text x="${S - 14}" y="92" text-anchor="end" fill="${ORANGE}" font-weight="bold">other z: never trained</text>\n</svg>\n`;
  write("autoencoder/fig-ae-gap.svg", s);
}
