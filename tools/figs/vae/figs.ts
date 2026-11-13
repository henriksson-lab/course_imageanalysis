// Slide figures for vae/ (added for design principle 3b), made with the demo code.
// Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/vae/figs.ts --bundle --platform=node --outfile=/tmp/vae-figs.cjs
//   node /tmp/vae-figs.cjs .
declare const process: { argv: string[] };
import { fromB64, digitVae, mse } from "../../../demos/src/vaeimages/imagevae";
import { DIGIT_EXAMPLES } from "../../../demos/src/vaeimages/digit_examples";
import { DEFAULT, EPS, recon, kl, loss, best, type Params } from "../../../demos/src/elbo1d/model";
import { rng } from "../../../demos/src/lib/ui";
import { INK, BLUE, RED, GREEN, ORANGE, GREY, DARKOR, DIGIT_COL, writer, svgOpen, sc, path, f, pngURI, img, npdf } from "./lib";

const write = writer(process.argv[2] ?? ".");

// ------------------------------------------------ fig-ae-gaps.svg and fig-clouds.svg (sketches, synthetic points)
{
  const r = rng(7);
  const cls: { d: number; c: [number, number] }[] = [{ d: 0, c: [-5.5, 3.5] }, { d: 1, c: [5, 2.5] }, { d: 7, c: [0.5, -5] }];
  const panel = (vae: boolean) => {
    const S = 440, L = vae ? 3 : 8, sx = sc(-L, L, 40, S - 10), sy = sc(-L, L, S - 40, 10);
    let s = svgOpen(S, S);
    s += `<line x1="${sx(0)}" x2="${sx(0)}" y1="10" y2="${S - 40}" stroke="#e3e3e3"/><line y1="${sy(0)}" y2="${sy(0)}" x1="40" x2="${S - 10}" stroke="#e3e3e3"/>`;
    for (const t of [-L, 0, L]) s += `<text x="${sx(t)}" y="${S - 14}" text-anchor="middle" font-size="17">${t}</text><text x="32" y="${sy(t) + 6}" text-anchor="end" font-size="17">${t}</text>`;
    s += `<text x="${S - 14}" y="${sy(0) - 10}" text-anchor="end">z₁</text><text x="${sx(0) + 8}" y="32">z₂</text>\n`;
    if (vae) s += `<circle cx="${sx(0)}" cy="${sy(0)}" r="${sx(2) - sx(0)}" fill="none" stroke="#666" stroke-width="2" stroke-dasharray="7 5"/><text x="${sx(1.45)}" y="${sy(-2.25)}" fill="#555" font-size="17">prior</text>\n`;
    const vc: [number, number][] = [[-1.2, 0.4], [1.2, 0.5], [0.1, -1.1]];
    cls.forEach(({ d, c }, k) => {
      const cc = vae ? vc[k] : c, sp = vae ? 0.32 : 0.8;
      for (let i = 0; i < 14; i++) {
        const p = [cc[0] + sp * r.norm(), cc[1] + sp * r.norm()];
        if (vae && i < 4) s += `<ellipse cx="${f(sx(p[0]))}" cy="${f(sy(p[1]))}" rx="${f(sx(0.45) - sx(0))}" ry="${f(sx(0.38) - sx(0))}" fill="${DIGIT_COL[d]}" fill-opacity="0.15" stroke="${DIGIT_COL[d]}" stroke-width="1.5"/>`;
        s += `<circle cx="${f(sx(p[0]))}" cy="${f(sy(p[1]))}" r="4.5" fill="${DIGIT_COL[d]}" fill-opacity="0.85"/>`;
      }
      s += `<text x="${f(sx(cc[0]) + (vae ? 0 : 0))}" y="${f(sy(cc[1] + (vae ? 0.95 : 2.1)))}" text-anchor="middle" font-size="26" font-weight="bold" fill="${DIGIT_COL[d]}" stroke="#fff" stroke-width="4" paint-order="stroke">${d}s</text>`;
    });
    if (!vae) {
      s += `\n<circle cx="${sx(-0.5)}" cy="${sy(1.2)}" r="16" fill="#fff" stroke="${RED}" stroke-width="2.5"/><text x="${sx(-0.5)}" y="${sy(1.2) + 8}" text-anchor="middle" font-size="24" font-weight="bold" fill="${RED}">?</text>`;
      s += `<text x="${sx(-0.5)}" y="${sy(1.2) + 44}" text-anchor="middle" fill="${RED}">never trained here</text>`;
    }
    return s + `\n</svg>\n`;
  };
  write("vae/fig-ae-gaps.svg", panel(false));
  write("vae/fig-clouds.svg", panel(true));
}

// ------------------------------------------------ fig-reparam.svg: z = mu + sigma * eps
{
  const W = 690, H = 380, mu = 1, sg = 0.5;
  const sx = sc(-3, 3, 30, 660), yE = 120, yZ = 300, hgt = 80;
  let s = svgOpen(W, H, 23);
  const zs = Array.from({ length: 301 }, (_, i) => -3 + 6 * i / 300);
  for (const [y0, m, sd, lab, col] of [[yE, 0, 1, "ε ~ N(0, 1): drawn first", GREY], [yZ, mu, sg, "z = μ + σ ε  (μ = 1, σ = 0.5)", ORANGE]] as [number, number, number, string, string][]) {
    s += `<line x1="30" x2="660" y1="${y0}" y2="${y0}" stroke="#444"/>`;
    for (const t of [-3, -2, -1, 0, 1, 2, 3]) s += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${y0}" y2="${y0 + 6}" stroke="#444"/><text x="${sx(t)}" y="${y0 + 26}" text-anchor="middle" font-size="20">${t}</text>`;
    s += `<path d="${path(zs.map(z => [sx(z), Math.max(y0 - 130, y0 - hgt * npdf(z, m, sd) / npdf(0, 0, 1) * (sd === 1 ? 1 : 0.55))]))}" fill="${col}" fill-opacity="0.15" stroke="${col}" stroke-width="2.5"/>`;
    s += `<text x="${y0 === yE ? 670 : 20}" y="${y0 - (y0 === yE ? 60 : 90)}" text-anchor="${y0 === yE ? "end" : "start"}" font-weight="bold" fill="${y0 === yE ? "#555" : DARKOR}">${lab}</text>\n`;
  }
  EPS.forEach(e => {
    s += `<line x1="${f(sx(e))}" y1="${yE}" x2="${f(sx(mu + sg * e))}" y2="${yZ}" stroke="${GREY}" stroke-width="1" stroke-opacity="0.6"/>`;
    s += `<circle cx="${f(sx(e))}" cy="${yE}" r="4.5" fill="#555"/><circle cx="${f(sx(mu + sg * e))}" cy="${yZ}" r="4.5" fill="${ORANGE}"/>`;
  });
  for (const [e, col] of [[-1, BLUE], [0.5, RED]] as [number, string][]) {
    s += `<line x1="${sx(e)}" y1="${yE}" x2="${sx(mu + sg * e)}" y2="${yZ}" stroke="${col}" stroke-width="3"/>`;
    s += `<circle cx="${sx(e)}" cy="${yE}" r="7" fill="${col}"/><circle cx="${sx(mu + sg * e)}" cy="${yZ}" r="7" fill="${col}"/>`;
    s += `<text x="${sx(mu + sg * e) + (e < 0 ? -10 : 10)}" y="${yZ + 60}" text-anchor="${e < 0 ? "end" : "start"}" fill="${col}" font-weight="bold">ε = ${e === -1 ? "−1" : e} → z = ${mu + sg * e}</text>`;
  }
  s += `\n</svg>\n`;
  write("vae/fig-reparam.svg", s);
}

// ------------------------------------------------ fig-vae-loss.svg: the VAE pipeline with a real digit and both loss terms
{
  const x = fromB64(DIGIT_EXAMPLES[1].img), q = digitVae.encode(x);
  const r = rng(3), e: [number, number] = [r.norm(), r.norm()], z = [q.mu[0] + q.sd[0] * e[0], q.mu[1] + q.sd[1] * e[1]];
  const xh = digitVae.decode(z);
  const klv = [0, 1].map(i => 0.5 * (q.mu[i] ** 2 + q.sd[i] ** 2 - 1 - Math.log(q.sd[i] ** 2))).reduce((a, b) => a + b);
  console.log(`vae-loss: mu (${q.mu.map(v => v.toFixed(2))}), sd (${q.sd.map(v => v.toFixed(3))}), eps (${e.map(v => v.toFixed(2))}), z (${z.map(v => v.toFixed(2))}), mse ${mse(x, xh).toFixed(4)}, KL ${klv.toFixed(2)}`);
  const W = 1000, H = 310;
  let s = svgOpen(W, H);
  const a = pngURI([x], 1, 0, 4), b = pngURI([xh], 1, 0, 4);
  s += img(20, 50, 112, 112, a.uri) + `<text x="76" y="190" text-anchor="middle" font-weight="bold">image x</text>\n`;
  s += `<path d="M 150 45 L 280 85 L 280 125 L 150 165 Z" fill="#eaf1f7" stroke="${BLUE}" stroke-width="2"/><text x="212" y="112" text-anchor="middle" font-weight="bold">encoder</text>\n`;
  s += `<rect x="298" y="52" width="170" height="104" rx="10" fill="#fff4e6" stroke="${DARKOR}" stroke-width="2.5"/>`;
  s += `<text x="383" y="84" text-anchor="middle" font-size="18">μ = (${q.mu.map(v => v.toFixed(2)).join(", ")})</text><text x="383" y="112" text-anchor="middle" font-size="18">σ = (${q.sd.map(v => v.toFixed(2)).join(", ")})</text><text x="383" y="140" text-anchor="middle" font-size="17" fill="${DARKOR}">a cloud</text>\n`;
  s += `<rect x="490" y="62" width="186" height="84" rx="10" fill="#fff" stroke="${INK}" stroke-width="2"/><text x="583" y="94" text-anchor="middle" font-weight="bold">z = μ + σ ε</text><text x="583" y="124" text-anchor="middle" font-size="17">ε random, N(0, I)</text>\n`;
  s += `<path d="M 696 85 L 826 45 L 826 165 L 696 125 Z" fill="#eef4ea" stroke="${GREEN}" stroke-width="2"/><text x="764" y="112" text-anchor="middle" font-weight="bold">decoder</text>\n`;
  s += img(860, 50, 112, 112, b.uri) + `<text x="916" y="190" text-anchor="middle" font-weight="bold">x̂</text>\n`;
  for (const [x1, x2] of [[134, 148], [281, 296], [469, 488], [677, 694], [827, 856]]) s += `<line x1="${x1}" y1="105" x2="${x2}" y2="105" stroke="${INK}" stroke-width="2.2" marker-end="url(#ar)"/>`;
  s += `\n<path d="M 916 205 L 916 262 L 76 262 L 76 205" fill="none" stroke="${RED}" stroke-width="2.5" stroke-dasharray="7 5"/>`;
  s += `<text x="500" y="292" text-anchor="middle" fill="${RED}" font-weight="bold">reconstruction: how well x̂ matches x  (rebuild this image)</text>\n`;
  s += `<path d="M 383 160 L 383 210" stroke="${DARKOR}" stroke-width="2.5" stroke-dasharray="7 5"/>`;
  s += `<text x="430" y="236" text-anchor="middle" fill="${DARKOR}" font-weight="bold">+ β · KL: how far the cloud is from the prior N(0, I)  (stay near the prior)</text>\n</svg>\n`;
  write("vae/fig-vae-loss.svg", s);
}

// ------------------------------------------------ fig-landscape.svg: total loss over (mu, s) for the by-hand example
{
  const p: Params = { ...DEFAULT }, b = best(p);
  const W = 520, H = 450, m0 = -0.5, m1 = 2.2, s0 = 0.02, s1 = 1.3, sx = sc(m0, m1, 76, 506), sy = sc(s0, s1, 380, 16);
  const lv = [1.705, 2, 2.5, 3, 4, 5, 6.5, 8, 10];
  const cols = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"];
  let s = svgOpen(W, H, 23);
  const n = 90;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const mu = m0 + (m1 - m0) * (i + 0.5) / n, sd = s0 + (s1 - s0) * (j + 0.5) / n, L = loss(p, mu, sd);
    let k = lv.findIndex(t => L < t); if (k < 0) k = cols.length; k = Math.max(0, k - 1);
    const x0 = sx(m0 + (m1 - m0) * i / n), x1 = sx(m0 + (m1 - m0) * (i + 1) / n), y0 = sy(s0 + (s1 - s0) * (j + 1) / n), y1 = sy(s0 + (s1 - s0) * j / n);
    s += `<rect x="${f(x0)}" y="${f(y0)}" width="${f(x1 - x0 + 0.6)}" height="${f(y1 - y0 + 0.6)}" fill="${L >= 10 ? "#ffffff" : cols[Math.min(k, cols.length - 1)]}"/>`;
  }
  s += `\n<rect x="76" y="16" width="430" height="364" fill="none" stroke="#444"/>`;
  for (const t of [0, 1, 2]) s += `<text x="${sx(t)}" y="406" text-anchor="middle" font-size="21">${t}</text>`;
  for (const t of [0.05, 0.5, 1]) s += `<text x="68" y="${sy(t) + 7}" text-anchor="end" font-size="21">${t}</text>`;
  s += `<text x="290" y="440" text-anchor="middle">centre μ</text><text x="18" y="200" transform="rotate(-90 18 200)" text-anchor="middle">spread s</text>\n`;
  const pts: [number, number, string, string, string][] = [[0, 1, "prior: 6.50", RED, "start"], [1.5, 0.05, "too confident: 3.63", ORANGE, "end"], [b.mu, b.s, "best: 1.705", "#fff", "start"]];
  for (const [mu, sd, lab, col, anc] of pts) {
    s += `<circle cx="${f(sx(mu))}" cy="${f(sy(sd))}" r="8" fill="${col}" stroke="${INK}" stroke-width="2"/>`;
    s += `<text x="${f(sx(mu) + (anc === "start" ? 14 : -14))}" y="${f(sy(sd) + (sd < 0.1 ? -14 : 6))}" text-anchor="${anc}" font-weight="bold" fill="${INK}" stroke="#fff" stroke-width="4" paint-order="stroke">${lab}</text>`;
    console.log(`landscape: ${lab} recomputed ${loss(p, mu, sd).toFixed(3)} (recon ${recon(p, mu, sd).toFixed(3)}, KL ${kl(mu, sd).toFixed(3)})`);
  }
  s += `\n</svg>\n`;
  write("vae/fig-landscape.svg", s);
}

// ------------------------------------------------ fig-posterior.svg: prior, likelihood, best cloud for beta 1 and 5
{
  const p: Params = { ...DEFAULT }, b1 = best(p), b5 = best({ ...p, beta: 5 });
  console.log(`best beta 1: ${b1.mu.toFixed(3)}, s² ${(b1.s ** 2).toFixed(3)}; beta 5: ${b5.mu.toFixed(3)}, s² ${(b5.s ** 2).toFixed(3)}`);
  const W = 480, H = 380, sx = sc(-2.6, 3.2, 20, 470), sy = sc(0, 1.0, 320, 30);
  const zs = Array.from({ length: 401 }, (_, i) => -2.6 + 5.8 * i / 400);
  let s = svgOpen(W, H, 23);
  s += `<line x1="20" x2="470" y1="320" y2="320" stroke="#444"/>`;
  for (const t of [-2, -1, 0, 1, 2, 3]) s += `<text x="${sx(t)}" y="346" text-anchor="middle" font-size="21">${t}</text>`;
  s += `<text x="470" y="374" text-anchor="end">z</text>`;
  const ln = p.sx / p.w;
  s += `<path d="${path(zs.map(z => [sx(z), sy(npdf(z, 0, 1))]))}" fill="none" stroke="#555" stroke-width="3" stroke-dasharray="8 5"/>`;
  s += `<path d="${path(zs.map(z => [sx(z), sy(npdf(z, p.x / p.w, ln))]))}" fill="none" stroke="${BLUE}" stroke-width="3" stroke-dasharray="3 4"/>`;
  s += `<path d="${path(zs.map(z => [sx(z), sy(npdf(z, b1.mu, b1.s))]))}" fill="${ORANGE}" fill-opacity="0.2" stroke="${ORANGE}" stroke-width="3"/>`;
  s += `<path d="${path(zs.map(z => [sx(z), sy(npdf(z, b5.mu, b5.s))]))}" fill="none" stroke="${RED}" stroke-width="3"/>\n`;
  const leg: [string, string][] = [["prior", "#555"], ["likelihood of x", BLUE], ["best q, β = 1", ORANGE], ["best q, β = 5", RED]];
  leg.forEach(([t, c], k) => { s += `<text x="14" y="${36 + 30 * k}" fill="${c}" font-weight="bold" font-size="23">${t}</text>`; });
  s += `\n</svg>\n`;
  write("vae/fig-posterior.svg", s);
}
