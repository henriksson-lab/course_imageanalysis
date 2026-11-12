// Static figures for the VAE units (neuralnet/, autoencoder/, vae/, vae-gen/), made with exactly
// the code the demos run. They are the print/PDF fallback for the demos and the pictures on the
// slides. Re-run from the repository root:
//   node_modules/.bin/esbuild demos/test/vae-figs.ts --bundle --platform=node --outfile=/tmp/vae-figs.cjs
//   node /tmp/vae-figs.cjs .
declare const require: (m: string) => any;
declare const process: { argv: string[] };
const fs = require("fs");
import { digitVae, shapeVae, fromB64, mse, type V2 } from "../src/vaeimages/imagevae";
import { MNIST_POINTS } from "../src/vaeimages/mnist_weights";
import { SHAPE_POINTS, SHAPE_EXAMPLES, SHAPE_KINDS } from "../src/vaeimages/shape_weights";
import { DIGIT_EXAMPLES } from "../src/vaeimages/digit_examples";
import { profileData, initHinges } from "../src/nnfit/data";
import { Adam, mlp, predict, trainStep, forward } from "../src/lib/nn";
import { makeData, pca, linEncode, linDecode, makeAE, aeTrain, aeCode, aeDecode } from "../src/linae/data";
import { pngGrey, tile } from "./png";

const ROOT = process.argv[2] ?? ".";
const INK = "#1b2a3a", BLUE = "#2a6f97", RED = "#bc4749", GREEN = "#6a994e", ORANGE = "#e07b1a", GREY = "#8d99ae";
const DIGIT_COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
const SHAPE_COL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"];
const write = (path: string, s: string | Uint8Array) => { fs.writeFileSync(`${ROOT}/${path}`, s); console.log("wrote", path); };
const svgOpen = (w: number, h: number) => `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="${INK}">\n<rect width="${w}" height="${h}" fill="#fff"/>\n`;
const sc = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + (v - d0) * (r1 - r0) / (d1 - d0);
const path = (pts: [number, number][]) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
const pngURI = (imgs: ArrayLike<number>[], cols: number, gap = 2, up = 4) => {
  const t = tile(imgs, cols, gap, up);
  let bin = ""; for (const b of pngGrey(t.grey, t.w, t.h)) bin += String.fromCharCode(b);
  return { uri: "data:image/png;base64," + btoa(bin), w: t.w, h: t.h };
};
const writePng = (file: string, imgs: ArrayLike<number>[], cols: number, gap = 2, up = 4) => {
  const t = tile(imgs, cols, gap, up); write(file, pngGrey(t.grey, t.w, t.h));
};

// ------------------------------------------------------------- neuralnet/fig-hinges.svg
{
  const { X, Y } = profileData();
  const net = mlp([1, 3, 1], ["relu", "id"], 11); initHinges(net, Y, 11);
  const opt = new Adam(net, 0.02);
  for (let s = 0; s < 3000; s++) trainStep(net, X, Y, opt);
  const W = 900, H = 380, sx = sc(0, 1, 70, 880), sy = sc(-0.1, 1.1, 330, 20);
  let s = svgOpen(W, H);
  for (const v of [0, 0.5, 1]) s += `<line x1="70" x2="880" y1="${sy(v)}" y2="${sy(v)}" stroke="#e3e3e3"/><text x="60" y="${sy(v) + 6}" text-anchor="end">${v}</text>\n`;
  s += `<text x="475" y="372" text-anchor="middle">position along the line scan</text><text x="18" y="175" transform="rotate(-90 18 175)" text-anchor="middle">intensity</text>\n`;
  X.forEach((x, i) => { s += `<circle cx="${sx(x[0]).toFixed(1)}" cy="${sy(Y[i][0]).toFixed(1)}" r="3.2" fill="${GREY}"/>`; });
  const xs = Array.from({ length: 301 }, (_, i) => i / 300);
  for (let k = 0; k < 3; k++) {
    const c = net.W[1][k], c0 = net.b[1][0];
    s += `<path d="${path(xs.map(x => [sx(x), sy(Math.max(-0.1, Math.min(1.1, c * forward(net, [x]).a[1][k] + c0)))]))}" fill="none" stroke="${ORANGE}" stroke-width="2.5" stroke-opacity="0.6"/>\n`;
    const xh = -net.b[0][k] / net.W[0][k];
    if (xh > 0 && xh < 1) s += `<path d="M${sx(xh).toFixed(1)},330l-7,12h14z" fill="${ORANGE}"/>`;
  }
  s += `<path d="${path(xs.map(x => [sx(x), sy(predict(net, [x])[0])]))}" fill="none" stroke="${RED}" stroke-width="4"/>\n`;
  s += `<text x="870" y="40" text-anchor="end" fill="${RED}" font-weight="bold">network, 3 hidden units</text><text x="870" y="66" text-anchor="end" fill="${ORANGE}">each unit: one hinge</text>\n</svg>\n`;
  write("neuralnet/fig-hinges.svg", s);
}

// ------------------------------------------------------------- autoencoder figures
{
  // hourglass with a real MNIST test digit and its reconstruction
  const img = fromB64(DIGIT_EXAMPLES[14].img);            // a test 7
  const q = digitVae.encode(img), rec = digitVae.decode(q.mu);
  const a = pngURI([img], 1, 0, 5), b = pngURI([rec], 1, 0, 5);
  let s = svgOpen(960, 330);
  s += `<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${INK}"/></marker></defs>\n`;
  s += `<image x="25" y="45" width="140" height="140" href="${a.uri}" style="image-rendering:pixelated"/><rect x="25" y="45" width="140" height="140" fill="none" stroke="${INK}" stroke-width="2"/>\n`;
  s += `<text x="95" y="215" text-anchor="middle" font-weight="bold">image x</text><text x="95" y="240" text-anchor="middle">784 pixels</text>\n`;
  s += `<path d="M 210 40 L 380 95 L 380 135 L 210 190 Z" fill="#eaf1f7" stroke="${BLUE}" stroke-width="2"/><text x="290" y="122" text-anchor="middle" font-weight="bold">encoder</text>\n`;
  s += `<rect x="400" y="80" width="160" height="70" rx="10" fill="#fff4e6" stroke="#c4621a" stroke-width="2.5"/><text x="480" y="112" text-anchor="middle" font-weight="bold">z = (${q.mu[0].toFixed(2)}, ${q.mu[1].toFixed(2)})</text><text x="480" y="137" text-anchor="middle" font-size="17">2 numbers</text>\n`;
  s += `<text x="480" y="210" text-anchor="middle" font-weight="bold" fill="#c4621a">bottleneck</text><text x="480" y="235" text-anchor="middle" fill="#c4621a">latent space</text>\n`;
  s += `<path d="M 580 95 L 750 40 L 750 190 L 580 135 Z" fill="#eef4ea" stroke="${GREEN}" stroke-width="2"/><text x="670" y="122" text-anchor="middle" font-weight="bold">decoder</text>\n`;
  s += `<image x="795" y="45" width="140" height="140" href="${b.uri}" style="image-rendering:pixelated"/><rect x="795" y="45" width="140" height="140" fill="none" stroke="${INK}" stroke-width="2"/>\n`;
  s += `<text x="865" y="215" text-anchor="middle" font-weight="bold">reconstruction x̂</text><text x="865" y="240" text-anchor="middle">784 pixels</text>\n`;
  for (const [x1, x2] of [[170, 206], [381, 397], [561, 577], [752, 790]]) s += `<line x1="${x1}" y1="115" x2="${x2}" y2="115" stroke="${INK}" stroke-width="2.2" marker-end="url(#ar)"/>`;
  s += `\n<path d="M 865 255 L 865 290 L 95 290 L 95 259" fill="none" stroke="${RED}" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#ar)"/>\n`;
  s += `<text x="480" y="318" text-anchor="middle" fill="${RED}">training: adjust both networks until x̂ is close to x, for many images</text>\n</svg>\n`;
  write("autoencoder/fig-hourglass.svg", s);
  console.log(`hourglass: example 15 (a test 7) z = (${q.mu[0].toFixed(2)}, ${q.mu[1].toFixed(2)}), error ${mse(img, rec).toFixed(4)}`);
}
function cloudFig(kind: "pixels" | "spot", file: string, withAE: boolean) {
  const P = makeData(kind), pc = pca(P), m = pc.m, th = pc.theta;
  const S = 470, sx = sc(-0.1, 1.15, 50, S - 10), sy = sc(-0.1, 1.15, S - 45, 10);
  let s = svgOpen(S, S);
  for (const v of [0, 0.5, 1]) s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${sy(-0.1)}" y2="${sy(1.15)}" stroke="#e3e3e3"/><line y1="${sy(v)}" y2="${sy(v)}" x1="${sx(-0.1)}" x2="${sx(1.15)}" stroke="#e3e3e3"/><text x="${sx(v)}" y="${S - 20}" text-anchor="middle">${v}</text><text x="40" y="${sy(v) + 6}" text-anchor="end">${v}</text>\n`;
  s += `<text x="${S - 12}" y="${S - 52}" text-anchor="end">pixel 1</text><text x="56" y="30">pixel 2</text>\n`;
  const u: V2 = [Math.cos(th), Math.sin(th)];
  let ae: ReturnType<typeof makeAE> | null = null;
  if (withAE) { ae = makeAE(); aeTrain(ae.net, ae.opt, P, 1500); }
  for (const p of P) {
    const qq = ae ? aeDecode(ae.net, aeCode(ae.net, p)) : linDecode(linEncode(p, m, th), m, th);
    s += `<line x1="${sx(p[0]).toFixed(1)}" y1="${sy(p[1]).toFixed(1)}" x2="${sx(qq[0]).toFixed(1)}" y2="${sy(qq[1]).toFixed(1)}" stroke="${RED}" stroke-width="1.5"/>`;
  }
  s += `\n<line x1="${sx(m[0] - 2 * u[0])}" y1="${sy(m[1] - 2 * u[1])}" x2="${sx(m[0] + 2 * u[0])}" y2="${sy(m[1] + 2 * u[1])}" stroke="${BLUE}" stroke-width="3" stroke-opacity="${ae ? 0.4 : 1}"/>\n`;
  if (ae) {
    const codes = P.map(p => aeCode(ae!.net, p)), lo = Math.min(...codes), hi = Math.max(...codes);
    s += `<path d="${path(Array.from({ length: 121 }, (_, i) => { const qq = aeDecode(ae!.net, lo + (hi - lo) * i / 120); return [sx(qq[0]), sy(qq[1])] as [number, number]; }))}" fill="none" stroke="${GREEN}" stroke-width="4"/>\n`;
  }
  P.forEach(p => { s += `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="4" fill="#555" fill-opacity="0.75"/>`; });
  s += `\n<text x="${S - 14}" y="${ae ? 60 : 40}" text-anchor="end" fill="${ae ? GREEN : BLUE}" font-weight="bold">${ae ? "decoder curve (autoencoder)" : "PC1 line"}</text>`;
  if (ae) s += `<text x="${S - 14}" y="34" text-anchor="end" fill="${BLUE}" fill-opacity="0.7">PC1 line</text>`;
  s += `\n</svg>\n`;
  write(file, s);
}
cloudFig("pixels", "autoencoder/fig-pca-pixels.svg", false);
cloudFig("spot", "autoencoder/fig-spot-pca.svg", false);
cloudFig("spot", "autoencoder/fig-spot-ae.svg", true);

// ------------------------------------------------------------- vae figures
{
  // latent space of the digit VAE: 1000 test means, with ±2 sd clouds of six example digits
  const S = 520, L = 4.2, sx = sc(-L, L, 10, S - 10), sy = sc(-L, L, S - 10, 10);
  let s = svgOpen(S, S);
  s += `<rect x="1" y="1" width="${S - 2}" height="${S - 2}" rx="6" fill="none" stroke="#e3e3e3" stroke-width="1.5"/><line x1="${sx(0)}" x2="${sx(0)}" y1="10" y2="${S - 10}" stroke="#e3e3e3"/><line y1="${sy(0)}" y2="${sy(0)}" x1="10" x2="${S - 10}" stroke="#e3e3e3"/>\n`;
  s += `<circle cx="${sx(0)}" cy="${sy(0)}" r="${sx(2) - sx(0)}" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="5 5"/>\n`;
  for (const [a, b, d] of MNIST_POINTS) s += `<circle cx="${sx(a).toFixed(1)}" cy="${sy(b).toFixed(1)}" r="2.3" fill="${DIGIT_COL[d]}" fill-opacity="0.45"/>`;
  s += "\n";
  for (const k of [0, 2, 4, 8, 14, 18]) {
    const q = digitVae.encode(fromB64(DIGIT_EXAMPLES[k].img));
    s += `<ellipse cx="${sx(q.mu[0]).toFixed(1)}" cy="${sy(q.mu[1]).toFixed(1)}" rx="${Math.max(3, sx(2 * q.sd[0]) - sx(0)).toFixed(1)}" ry="${Math.max(3, sx(2 * q.sd[1]) - sx(0)).toFixed(1)}" fill="#f4a261" fill-opacity="0.35" stroke="#c4621a" stroke-width="2"/>`;
  }
  for (let d = 0; d < 10; d++) {
    const p = MNIST_POINTS.filter(q => q[2] === d), med = (k: 0 | 1) => p.map(q => q[k]).sort((a, b) => a - b)[Math.floor(p.length / 2)];
    s += `<text x="${sx(med(0))}" y="${sy(med(1)) + 8}" text-anchor="middle" font-size="24" font-weight="bold" fill="${DIGIT_COL[d]}" stroke="#fff" stroke-width="4" paint-order="stroke">${d}</text>`;
  }
  s += `\n<text x="20" y="30">z₂</text><text x="${S - 20}" y="${S - 20}" text-anchor="end">z₁</text><text x="${sx(0) + 6}" y="${sy(2) - 8}" fill="#666" font-size="16">prior N(0, I): radius 2</text>\n</svg>\n`;
  write("vae/fig-latent-digits.svg", s);
}
{
  // KL: prior and three encoder clouds
  const cases: [number, number, string][] = [[0, 1, "μ = 0, s = 1: KL = 0"], [1.2, Math.sqrt(0.2), "μ = 1.2, s = 0.45: KL = 1.12"], [-1.5, 0.15, "μ = −1.5, s = 0.15: KL = 2.52"]];
  const kl = (m: number, s: number) => 0.5 * (m * m + s * s - 1 - Math.log(s * s));
  const W = 900, H = 330, sx = sc(-3.2, 3.2, 40, 880), sy = sc(0, 1.4, 290, 20);
  const npdf = (z: number, m: number, s: number) => Math.exp(-0.5 * ((z - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  let s = svgOpen(W, H);
  s += `<line x1="40" x2="880" y1="290" y2="290" stroke="#444"/>`;
  for (const t of [-3, -2, -1, 0, 1, 2, 3]) s += `<text x="${sx(t)}" y="314" text-anchor="middle" font-size="16">${t}</text>`;
  s += `<text x="880" y="322" text-anchor="end">z</text>\n`;
  const zs = Array.from({ length: 641 }, (_, i) => -3.2 + 6.4 * i / 640);
  const cols = [GREY, ORANGE, RED];
  cases.forEach(([m, sd], k) => {
    if (k === 0) s += `<path d="${path(zs.map(z => [sx(z), sy(npdf(z, 0, 1))]))}" fill="none" stroke="#555" stroke-width="3" stroke-dasharray="8 5"/>`;
    else s += `<path d="${path(zs.map(z => [sx(z), Math.max(20, sy(npdf(z, m, sd)))]))}" fill="${cols[k]}" fill-opacity="0.2" stroke="${cols[k]}" stroke-width="3"/>`;
    s += `<text x="870" y="${50 + 28 * k}" text-anchor="end" fill="${k ? cols[k] : "#555"}" font-weight="bold">${k ? "q: " : "prior = q: "}${m === 0 && sd === 1 ? "μ = 0, s = 1, KL = 0" : `μ = ${m}, s = ${sd.toFixed(2)}, KL = ${kl(m, sd).toFixed(2)}`}</text>\n`;
  });
  s += `</svg>\n`;
  write("vae/fig-kl.svg", s);
  console.log("KL cases", cases.map(c => kl(c[0], c[1]).toFixed(4)).join(" "));
}

// ------------------------------------------------------------- vae-gen figures
{
  const n = 15, grid = (v: typeof digitVae) => { const imgs: Float32Array[] = []; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) imgs.push(v.decode([-3 + 6 * (j + 0.5) / n, 3 - 6 * (i + 0.5) / n])); return imgs; };
  writePng("vae-gen/fig-grid-digits.png", grid(digitVae), n, 2, 2);
  writePng("vae-gen/fig-grid-shapes.png", grid(shapeVae), n, 2, 2);
  // walk vs crossfade, digits 1 → 7 between class centres (as the demo preset)
  const centre = (pts: [number, number, number][], d: number): V2 => { const p = pts.filter(q => q[2] === d); const med = (k: 0 | 1) => p.map(q => q[k]).sort((a, b) => a - b)[Math.floor(p.length / 2)]; return [med(0), med(1)]; };
  const walk = (v: typeof digitVae, a: V2, b: V2) => {
    const zs = Array.from({ length: 7 }, (_, s) => [a[0] + s / 6 * (b[0] - a[0]), a[1] + s / 6 * (b[1] - a[1])]);
    const A = v.decode(a), B = v.decode(b);
    return [...zs.map(z => v.decode(z)), ...zs.map((_, s) => A.map((x, i) => (1 - s / 6) * x + s / 6 * B[i]))];
  };
  writePng("vae-gen/fig-walk-digits.png", walk(digitVae, centre(MNIST_POINTS, 1), centre(MNIST_POINTS, 7)), 7, 4, 3);
  writePng("vae-gen/fig-walk-shapes.png", walk(shapeVae, centre(SHAPE_POINTS, 0), centre(SHAPE_POINTS, 2)), 7, 4, 3);
  // anomalies: inputs (top) and reconstructions (bottom), shapes
  const pick = [0, 3, 6, 9, 12, 13, 14];
  const ins = pick.map(k => fromB64(SHAPE_EXAMPLES[k].img)), outs = ins.map(x => shapeVae.decode(shapeVae.encode(x).mu));
  writePng("vae-gen/fig-anomaly.png", [...ins, ...outs], 7, 4, 3);
  console.log("anomaly figure:", pick.map((k, i) => `${SHAPE_EXAMPLES[k].label} ${mse(ins[i], outs[i]).toFixed(4)}`).join("; "));
  // denoising: shapes example 2 and 10, with noise 0.3 (rng 1001 as in the demo)
  const { rng } = require("../src/lib/ui") as typeof import("../src/lib/ui");
  const dn: Float32Array[] = [];
  for (const k of [1, 6, 10]) {
    const x = fromB64(SHAPE_EXAMPLES[k].img), r = rng(1001), xn = x.map(v => Math.max(0, Math.min(1, v + 0.3 * r.norm())));
    dn.push(x, xn, shapeVae.decode(shapeVae.encode(xn).mu));
    console.log(`denoise ex ${k + 1}: noisy vs clean ${mse(x, xn).toFixed(4)}, output vs clean ${mse(x, shapeVae.decode(shapeVae.encode(xn).mu)).toFixed(4)}`);
  }
  writePng("vae-gen/fig-denoise.png", dn, 3, 4, 3);
  console.log("SHAPE_KINDS", SHAPE_KINDS.join(","));
}
