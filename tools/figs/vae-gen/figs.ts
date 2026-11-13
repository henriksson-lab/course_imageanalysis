// Slide figures for vae-gen/ (added for design principle 3b), made with the demo code.
// Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/vae-gen/figs.ts --bundle --platform=node --outfile=/tmp/vaegen-figs.cjs
//   node /tmp/vaegen-figs.cjs .
declare const process: { argv: string[] };
import { digitVae } from "../../../demos/src/vaeimages/imagevae";
import { DEFAULT, recon, kl, best, type Params } from "../../../demos/src/elbo1d/model";
import { INK, BLUE, RED, GREEN, ORANGE, GREY, writer, svgOpen, sc, path, f, pngURI, img } from "../vae/lib";

const write = writer(process.argv[2] ?? ".");

// ------------------------------------------------ fig-bar-midpoint.svg: the 3-pixel bar decoder
{
  const dec = (z: number) => [0, 1, 2].map(i => Math.max(0, 1 - Math.abs(z - i)));
  const A = dec(0), B = dec(2), mid = dec(1), avg = A.map((a, i) => (a + B[i]) / 2);
  const rows: [string, number[], string][] = [["A: z = 0", A, INK], ["B: z = 2", B, INK], ["decode z = 1", mid, GREEN], ["pixel average", avg, RED]];
  const W = 450, H = 340, q = 60;
  let s = svgOpen(W, H, 24);
  rows.forEach(([lab, v, col], r) => {
    const y = 14 + r * 80 + (r >= 2 ? 14 : 0);
    v.forEach((a, i) => { const g = Math.round(255 * a); s += `<rect x="${20 + i * (q + 4)}" y="${y}" width="${q}" height="${q}" fill="rgb(${g},${g},${g})" stroke="${INK}" stroke-width="1.5"/>`; s += `<text x="${20 + i * (q + 4) + q / 2}" y="${y + q / 2 + 8}" text-anchor="middle" font-size="22" fill="${a > 0.6 ? INK : "#fff"}">${a}</text>`; });
    s += `<text x="228" y="${y + q / 2 + 8}" fill="${col}" font-weight="${r >= 2 ? "bold" : "normal"}">${lab}</text>\n`;
  });
  s += `<line x1="16" x2="${W - 16}" y1="182" y2="182" stroke="#ccc"/>\n</svg>\n`;
  write("vae-gen/fig-bar-midpoint.svg", s);
}

// ------------------------------------------------ fig-beta-sweep.svg: elbo1d, best cloud's two terms against beta
{
  const p: Params = { ...DEFAULT };
  const W = 600, H = 410, lx = sc(Math.log10(0.1), Math.log10(10), 80, 570), sy = sc(0, 4, 340, 20);
  const bs = Array.from({ length: 201 }, (_, i) => 10 ** (-1 + 2 * i / 200));
  const R = (b: number) => { const o = best({ ...p, beta: b }); return recon(p, o.mu, o.s); };
  const K = (b: number) => { const o = best({ ...p, beta: b }); return kl(o.mu, o.s); };
  let s = svgOpen(W, H, 23);
  for (const v of [0, 1, 2, 3, 4]) s += `<line x1="80" x2="570" y1="${sy(v)}" y2="${sy(v)}" stroke="#e3e3e3"/><text x="70" y="${sy(v) + 6}" text-anchor="end" font-size="21">${v}</text>`;
  for (const b of [0.1, 0.3, 1, 3, 10]) s += `<text x="${lx(Math.log10(b))}" y="368" text-anchor="middle" font-size="21">${b}</text>`;
  s += `<text x="325" y="402" text-anchor="middle">β (log scale)</text><text x="22" y="180" transform="rotate(-90 22 180)" text-anchor="middle">term at the best cloud</text>\n`;
  s += `<path d="${path(bs.map(b => [lx(Math.log10(b)), sy(Math.min(4, R(b)))]))}" fill="none" stroke="${RED}" stroke-width="3.5"/>`;
  s += `<path d="${path(bs.map(b => [lx(Math.log10(b)), sy(Math.min(4, K(b)))]))}" fill="none" stroke="${BLUE}" stroke-width="3.5" stroke-dasharray="9 5"/>\n`;
  for (const b of [1, 5]) {
    const r = R(b), k = K(b), o = best({ ...p, beta: b });
    console.log(`beta ${b}: mu ${o.mu.toFixed(3)}, s ${o.s.toFixed(3)}, recon ${r.toFixed(3)}, KL ${k.toFixed(3)}`);
    s += `<line x1="${lx(Math.log10(b))}" x2="${lx(Math.log10(b))}" y1="${sy(0)}" y2="${sy(4)}" stroke="${GREY}" stroke-dasharray="3 4"/>`;
    s += `<circle cx="${lx(Math.log10(b))}" cy="${sy(r)}" r="6" fill="${RED}"/><circle cx="${lx(Math.log10(b))}" cy="${sy(k)}" r="6" fill="${BLUE}"/>`;
    s += `<text x="${lx(Math.log10(b)) + 8}" y="${sy(r) + (b === 1 ? 24 : -10)}" fill="${RED}" font-size="21" font-weight="bold">${r.toFixed(2)}</text><text x="${lx(Math.log10(b)) + 8}" y="${sy(k) - 10}" fill="${BLUE}" font-size="21" font-weight="bold">${k.toFixed(2)}</text>`;
  }
  s += `\n<text x="96" y="48" fill="${RED}" font-weight="bold">reconstruction (image quality)</text><text x="96" y="78" fill="${BLUE}" font-weight="bold">KL (distance from the prior)</text>\n</svg>\n`;
  write("vae-gen/fig-beta-sweep.svg", s);
}

// ------------------------------------------------ fig-far.svg: decoding ever further from the data
{
  const ts = [0, 1.1, 2.2, 3.3, 5, 8];
  const W = 900, H = 270, cw = 145;
  let s = svgOpen(W, H);
  ts.forEach((t, k) => {
    const im = digitVae.decode([t, -t]), u = pngURI([im], 1, 0, 4), x = 14 + k * cw;
    s += img(x + 7, 14, 112, 112, u.uri);
    s += `<text x="${x + 63}" y="156" text-anchor="middle" font-size="17">z = (${t}, ${t ? "−" + t : 0})</text>`;
    const dens = Math.exp(-t * t);
    const ds = t === 0 ? "1" : dens > 0.001 ? dens.toFixed(dens > 0.1 ? 2 : 4) : (() => { const e = Math.floor(Math.log10(dens)); return `${(dens / 10 ** e).toFixed(1)}·10<tspan baseline-shift="super" font-size="13">${e}</tspan>`; })();
    s += `<text x="${x + 63}" y="184" text-anchor="middle" font-size="17" fill="${t > 2.5 ? RED : GREY}">${ds}</text>\n`;
    console.log(`far: t ${t}, relative prior density ${dens.toExponential(2)}`);
  });
  s += `<text x="450" y="222" text-anchor="middle" fill="${GREY}">prior density relative to the centre</text>`;
  s += `<path d="M 500 250 L 880 250" stroke="${RED}" stroke-width="2.5" marker-end="url(#arR)"/><text x="20" y="256" fill="${RED}">prior almost never draws here, the decoder still answers</text>\n</svg>\n`;
  write("vae-gen/fig-far.svg", s);
}

// ------------------------------------------------ fig-blurry-mean.svg: the mean of sharp strokes is a blurry stroke
{
  const stroke = (ang: number, dx: number) => {
    const a = ang * Math.PI / 180, c = Math.cos(a), sn = Math.sin(a), im = new Float32Array(784);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const px = x + 0.5 - 14 - dx, py = y + 0.5 - 14;
      const along = px * sn + py * c, across = px * c - py * sn;   // along the stroke (vertical at ang = 0)
      im[y * 28 + x] = Math.abs(across) <= 1.5 && Math.abs(along) <= 9.5 ? 1 : 0;
    }
    return im;
  };
  const set = [[-20, -2], [-10, 1], [0, -1], [8, 2], [16, 0], [25, -2]].map(([a, d]) => stroke(a, d));
  const mean = set[0].map((_, i) => set.reduce((t, im) => t + im[i], 0) / set.length);
  const W = 900, H = 230;
  let s = svgOpen(W, H);
  set.forEach((im, k) => { s += img(14 + k * 102, 40, 90, 90, pngURI([im], 1, 0, 4).uri); });
  s += `<text x="320" y="26" text-anchor="middle">six sharp images x that share one z</text>`;
  s += `<line x1="636" y1="85" x2="700" y2="85" stroke="${INK}" stroke-width="2.5" marker-end="url(#ar)"/><text x="668" y="70" text-anchor="middle" font-size="17">mean</text>`;
  s += img(716, 22, 126, 126, pngURI([mean], 1, 0, 5).uri);
  s += `<text x="779" y="174" text-anchor="middle" font-weight="bold">best single output c*</text><text x="779" y="200" text-anchor="middle">= E[x | z]: blurry</text>\n</svg>\n`;
  write("vae-gen/fig-blurry-mean.svg", s);
}
